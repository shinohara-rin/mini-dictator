//! One dictation from hotkey press to inserted text.
//!
//! A shell calls [`run_dictation`] on a worker thread when the user presses the dictation hotkey. It
//! sends microphone blocks on the channel and drops the sender when the user releases the hotkey
//! (or presses it again). The context snapshot is taken first, before any speech is decoded, so a
//! slow accessibility read delays nothing: audio waits in the channel.
//!
//! With a streaming recognizer ([`Asr::stream`]) the encoder reads the audio as it arrives, and a
//! segment waits only until the encoder has passed its end. Otherwise each segment is encoded once
//! the voice activity detector closes it.
//!
//! A recognizer that continues decodes ([`Asr::decode_after`], Qwen3-ASR) decodes the utterance
//! in progress again every [`EngineConfig::partial_ms`], and the decode at its end starts from the
//! words two of those decodes agreed on, so it generates only the last words. Once a pause has
//! lasted [`EngineConfig::early_commit_ms`], the utterance is transcribed as the pause would end
//! it, and the text goes out the moment the segment closes, unless speech came back first.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::asr::{Asr, Bias, EncoderOutput, EncoderStream, Hypothesis};
use crate::audio::{AudioChunk, Resampler, SAMPLE_RATE};
use crate::context::{merge_terms, rank_terms, ContextProvider, ScreenContext, TermMemory};
use crate::join::{JoinConfig, Joiner, Piece};
use crate::rewrite::{user_prompt, RewriteRequest, Rewriter};
use crate::uncertainty::{self, format_for_llm, MonteCarloConfig, Uncertainty};
use crate::vad::{Segmenter, SegmenterConfig, SpeechSegment, VoiceActivityDetector};
use crate::{Error, Result};

/// Receives the final text. Shells type it into the focused app or paste it.
pub trait TextSink {
    fn insert(&mut self, text: &str) -> Result<()>;
}

/// Progress for the tray icon and the log.
#[derive(Debug, Clone)]
pub enum Event {
    /// The context snapshot is ready, with this many hotwords.
    Listening {
        hotwords: usize,
    },
    /// The detector closed a segment of this many seconds, and the recognizer starts on it.
    Heard {
        seconds: f32,
    },
    /// A segment was decoded. `marked` carries the uncertainty markers.
    Transcribed {
        plain: String,
        marked: String,
        unclear_spans: usize,
        /// Set when the segment reached the recognizer this many seconds after its audio ended,
        /// later than [`EngineConfig::max_lag_seconds`], and skipped the uncertainty check.
        behind_seconds: Option<f32>,
    },
    Inserted {
        text: String,
    },
    /// What the utterance in progress says so far. Later words may still change.
    Partial {
        text: String,
    },
    /// A step failed. The dictation goes on when it can: a failed rewrite inserts the plain transcript.
    Warning(String),
    /// What one step read, produced, and how long it took, for a debug log: the screen text and
    /// hotwords, each decode, the uncertainty check, and the rewrite prompt and reply.
    /// Emitted only while [`EngineConfig::debug`] is set.
    Debug(String),
    Finished,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub segmenter: SegmenterConfig,
    pub monte_carlo: MonteCarloConfig,
    /// A segment that reaches the recognizer more than this many seconds after its audio ended
    /// skips the uncertainty check. On a device slower than realtime the noisy decodes would
    /// otherwise make each segment wait longer than the last; skipping them lets it catch up, and
    /// the check runs again once it has. Audio is never dropped.
    pub max_lag_seconds: f32,
    /// Screen terms used as hotwords and shown to the rewrite model.
    pub max_hotwords: usize,
    /// Send focused-field text to the rewrite model. Hotwords are sent either way.
    pub share_screen_text: bool,
    /// Characters of focused-field text, and of text dictated so far, the rewrite model reads.
    pub preceding_chars: usize,
    /// How segments split by a pause join into sentences.
    pub join: JoinConfig,
    /// While an utterance goes on, decode it again each time it has grown by this much, show the
    /// text so far ([`Event::Partial`]), and let the decode at its end continue from the words two
    /// such decodes agreed on. 0 turns it off. Recognizers with a streaming encoder skip it.
    pub partial_ms: u32,
    /// Once a pause has lasted this long, transcribe the utterance as the pause would end it, so the
    /// text is ready when the pause reaches [`SegmenterConfig::min_silence_ms`]. Speech that comes
    /// back first throws that transcript away. 0 turns it off.
    pub early_commit_ms: u32,
    /// Emit [`Event::Debug`]. A shell flips it while dictations run; off, nothing is formatted.
    pub debug: Arc<AtomicBool>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            segmenter: SegmenterConfig::default(),
            monte_carlo: MonteCarloConfig::default(),
            // A segment that keeps up arrives about `min_silence_ms` after its audio ends.
            max_lag_seconds: 2.0,
            max_hotwords: 100,
            share_screen_text: true,
            preceding_chars: 400,
            join: JoinConfig::default(),
            partial_ms: 1000,
            early_commit_ms: 250,
            debug: Arc::default(),
        }
    }
}

/// What one dictation knows about the screen, prepared at its start.
pub struct SessionContext {
    pub screen: ScreenContext,
    pub hotwords: Vec<String>,
    pub bias: Bias,
    /// Text inserted so far in this dictation.
    pub inserted: String,
    /// The last segment's final period, held until the next one shows whether the sentence ended.
    pub joiner: Joiner,
}

/// One decoded segment.
#[derive(Debug, Clone)]
pub struct Transcript {
    pub reference: Hypothesis,
    pub uncertainty: Uncertainty,
    pub marked: String,
    pub timings: Timings,
}

/// Time spent on each step of [`Engine::transcribe`].
#[derive(Debug, Clone, Default)]
pub struct Timings {
    /// Zero when a streaming encoder had encoded the segment while it was spoken.
    pub encode: Duration,
    pub decode: Duration,
    /// The noisy decodes of the uncertainty check.
    pub check: Duration,
}

/// What the pipeline already has for a segment when it closes.
#[derive(Debug, Default)]
pub struct Prepared {
    /// Encoder output a stream computed while the segment was spoken.
    pub encoded: Option<EncoderOutput>,
    /// Ids the decode starts with, from decodes of the utterance in progress.
    pub prefix: Vec<u32>,
    /// The whole transcript, made early in the pause that closed the segment.
    pub transcript: Option<Transcript>,
}

pub struct Engine {
    pub asr: Box<dyn Asr>,
    pub context: Box<dyn ContextProvider>,
    /// Terms of recent screens. A shell that samples the screen between dictations fills it.
    pub memory: Arc<TermMemory>,
    pub rewriter: Box<dyn Rewriter>,
    pub config: EngineConfig,
}

impl Engine {
    fn debug(&self) -> bool {
        self.config.debug.load(Ordering::Relaxed)
    }

    /// Reads the screen and prepares its terms, with the remembered ones, as hotwords. A failed read
    /// gives an empty context.
    pub fn prepare(&self, events: &mut dyn FnMut(Event)) -> SessionContext {
        let started = Instant::now();
        let screen = self.context.snapshot().unwrap_or_else(|error| {
            events(Event::Warning(error.to_string()));
            ScreenContext::empty()
        });
        let read = started.elapsed();
        let now = Instant::now();
        // Every current term, so a remembered weight can lift one above the cut.
        let current = rank_terms(&screen, usize::MAX);
        // Recalled before this snapshot goes in, so the current screen does not count twice.
        let recalled = self.memory.recall(now);
        self.memory.observe(&current, now);
        let hotwords: Vec<String> = merge_terms(current, recalled, self.config.max_hotwords)
            .into_iter()
            .map(|t| t.phrase)
            .collect();
        let bias = self.asr.bias(&hotwords);
        if self.debug() {
            events(Event::Debug(describe_context(
                &screen,
                &hotwords,
                &bias,
                read,
                started.elapsed() - read,
            )));
        }
        SessionContext {
            screen,
            hotwords,
            bias,
            inserted: String::new(),
            joiner: Joiner::default(),
        }
    }

    /// Decodes one segment, with the uncertainty check when `check` is true. `encoded` is the
    /// segment's encoder output when a stream has computed it already, and the decode starts with
    /// the ids in `prefix`.
    pub fn transcribe(
        &self,
        segment: &SpeechSegment,
        encoded: Option<EncoderOutput>,
        prefix: &[u32],
        session: &SessionContext,
        check: bool,
    ) -> Result<Transcript> {
        let started = Instant::now();
        let encoder = match encoded {
            Some(encoder) => encoder,
            None => self.asr.encode(&segment.samples)?,
        };
        let encoded = started.elapsed();
        let reference = self.asr.decode_after(&encoder, &session.bias, prefix)?;
        let decoded = started.elapsed();
        let unchecked = MonteCarloConfig {
            samples: 0,
            ..self.config.monte_carlo.clone()
        };
        let uncertainty = uncertainty::estimate(
            self.asr.as_ref(),
            &encoder,
            &reference,
            &session.bias,
            if check {
                &self.config.monte_carlo
            } else {
                &unchecked
            },
        )?;
        let timings = Timings {
            encode: encoded,
            decode: decoded - encoded,
            check: started.elapsed() - decoded,
        };
        let marked = format_for_llm(&reference, &uncertainty.analysis.spans);
        Ok(Transcript {
            reference,
            uncertainty,
            marked,
            timings,
        })
    }

    /// `pause_seconds` is the silence since the last inserted segment.
    pub fn rewrite_request(
        &self,
        transcript: &Transcript,
        session: &SessionContext,
        pause_seconds: Option<f32>,
    ) -> RewriteRequest {
        RewriteRequest {
            marked_transcript: transcript.marked.clone(),
            plain_transcript: transcript.reference.text(),
            app: session.screen.app.clone(),
            screen_terms: session.hotwords.iter().take(30).cloned().collect(),
            preceding_text: if self.config.share_screen_text {
                session
                    .screen
                    .focused_tail(self.config.preceding_chars)
                    .unwrap_or_default()
            } else {
                String::new()
            },
            session_text: tail(&session.inserted, self.config.preceding_chars),
            pause_seconds,
        }
    }

    /// Decodes, rewrites, and inserts one segment, starting from what `prepared` holds.
    /// `lag_seconds` is how long ago its audio ended. Returns the inserted text, if any.
    pub fn process(
        &self,
        segment: &SpeechSegment,
        prepared: Prepared,
        lag_seconds: f32,
        session: &mut SessionContext,
        sink: &mut dyn TextSink,
        events: &mut dyn FnMut(Event),
    ) -> Result<Option<String>> {
        let behind_seconds = (lag_seconds > self.config.max_lag_seconds).then_some(lag_seconds);
        events(Event::Heard {
            seconds: segment.duration_seconds(),
        });
        let streamed = prepared.encoded.is_some();
        let early = prepared.transcript.is_some();
        let transcript = match prepared.transcript {
            Some(transcript) => transcript,
            None => self.transcribe(
                segment,
                prepared.encoded,
                &prepared.prefix,
                session,
                behind_seconds.is_none() && self.rewriter.reads_uncertainty(),
            )?,
        };
        let plain = transcript.reference.text();
        if self.debug() {
            events(Event::Debug(describe_transcript(
                segment,
                lag_seconds,
                behind_seconds.is_some(),
                streamed,
                early,
                &prepared.prefix,
                &transcript,
            )));
        }
        events(Event::Transcribed {
            plain: plain.clone(),
            marked: transcript.marked.clone(),
            unclear_spans: transcript.uncertainty.analysis.spans.len(),
            behind_seconds,
        });
        if is_nothing(&plain) && transcript.uncertainty.analysis.spans.is_empty() {
            return Ok(None);
        }
        let pause_seconds = session.joiner.pause_before(segment);
        let request = self.rewrite_request(&transcript, session, pause_seconds);
        let started = Instant::now();
        let rewritten = self.rewriter.rewrite(&request);
        if self.debug() {
            events(Event::Debug(format!(
                "rewrite in {}\n  prompt:\n{}\n  reply: {}",
                ms(started.elapsed()),
                indent(&user_prompt(&request), "    "),
                match &rewritten {
                    Ok(Some(text)) => format!("{text:?}"),
                    Ok(None) => "off, the transcript stands".into(),
                    Err(error) => format!("failed: {error}"),
                }
            )));
        }
        // Without a rewrite, the capitals are the recognizer's, chosen without the text before.
        let (text, rewritten) = match rewritten {
            Ok(Some(text)) => (text, true),
            Ok(None) => (plain, false),
            Err(error) => {
                events(Event::Warning(error.to_string()));
                (plain, false)
            }
        };
        if text.trim().is_empty() {
            return Ok(None);
        }
        let piece = Piece {
            rewritten,
            text: &text,
            pause_seconds,
            names: &session.hotwords,
        };
        let before = session.joiner.clone();
        let text = session
            .joiner
            .join(&session.inserted, &piece, &self.config.join);
        session.joiner.typed(segment);
        let inserted = self.insert(text, session, sink, events);
        if inserted.is_err() {
            // Nothing was typed, so the period held before is still the one to type at the end.
            session.joiner = before;
        }
        inserted
    }

    /// Types the period held back from the last segment. Call it when the dictation ends.
    pub fn finish(
        &self,
        session: &mut SessionContext,
        sink: &mut dyn TextSink,
        events: &mut dyn FnMut(Event),
    ) -> Result<Option<String>> {
        let text = session.joiner.finish();
        self.insert(text, session, sink, events)
    }

    fn insert(
        &self,
        text: String,
        session: &mut SessionContext,
        sink: &mut dyn TextSink,
        events: &mut dyn FnMut(Event),
    ) -> Result<Option<String>> {
        if text.is_empty() {
            return Ok(None);
        }
        sink.insert(&text).map_err(|e| Error::Sink(e.to_string()))?;
        session.inserted.push_str(&text);
        events(Event::Inserted { text: text.clone() });
        Ok(Some(text))
    }
}

/// How long ago a segment's last sample was recorded: the age of the latest block plus the audio
/// after the segment. Timing each block keeps this right when the worker starts after the
/// microphone, or capture starts late.
/// A transcript with nothing but full stops, which the recognizer gives for a cough or a breath.
pub fn is_nothing(plain: &str) -> bool {
    plain
        .chars()
        .all(|c| c.is_whitespace() || matches!(c, '.' | '。'))
}

fn lag(segment: &SpeechSegment, (heard, captured): (u64, Instant)) -> f32 {
    let end = segment.start_sample + segment.samples.len() as u64;
    captured.elapsed().as_secs_f32() + heard.saturating_sub(end) as f32 / SAMPLE_RATE as f32
}

/// The last `max_chars` characters of `text`.
fn tail(text: &str, max_chars: usize) -> String {
    let skip = text.chars().count().saturating_sub(max_chars);
    text.chars().skip(skip).collect()
}

fn ms(duration: Duration) -> String {
    format!("{} ms", duration.as_millis())
}

fn indent(text: &str, prefix: &str) -> String {
    text.lines()
        .map(|line| format!("{prefix}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The screen snapshot and the hotwords taken from it, for the debug log.
fn describe_context(
    screen: &ScreenContext,
    hotwords: &[String],
    bias: &Bias,
    read: Duration,
    ranked: Duration,
) -> String {
    let mut text = format!(
        "context from {:?} in {} (hotwords in {}), app {:?}, {} blocks, {} characters",
        screen.source,
        ms(read),
        ms(ranked),
        screen.app.as_deref().unwrap_or("unknown"),
        screen.blocks.len(),
        screen.text_len(),
    );
    for block in &screen.blocks {
        let _ = write!(text, "\n  {:?}: {:?}", block.region, block.text);
    }
    let _ = write!(
        text,
        "\n  hotwords ({}): {}",
        hotwords.len(),
        hotwords.join(", ")
    );
    // A backend can keep fewer: Qwen3-ASR keeps what fits its prompt budget, and a transducer
    // keeps what its tokenizer can spell.
    let _ = write!(
        text,
        "\n  sent to the recognizer ({}): {}",
        bias.phrases.len(),
        bias.phrases.join(", ")
    );
    text
}

/// One decoded segment with its uncertainty check, for the debug log.
fn describe_transcript(
    segment: &SpeechSegment,
    lag_seconds: f32,
    behind: bool,
    streamed: bool,
    early: bool,
    prefix: &[u32],
    transcript: &Transcript,
) -> String {
    let Timings {
        encode,
        decode,
        check,
    } = &transcript.timings;
    let uncertainty = &transcript.uncertainty;
    let mut text = format!(
        "segment at {:.2} s, {:.2} s long{}, {lag_seconds:.2} s behind: encode {}, decode {}{}{}\n  plain:  {:?}",
        segment.start_sample as f32 / SAMPLE_RATE as f32,
        segment.duration_seconds(),
        if segment.truncated { " (truncated)" } else { "" },
        if streamed {
            "streamed".to_owned()
        } else {
            ms(*encode)
        },
        ms(*decode),
        if prefix.is_empty() {
            String::new()
        } else {
            format!(" after {} agreed tokens", prefix.len())
        },
        if early { ", done early in the pause" } else { "" },
        transcript.reference.text(),
    );
    if behind {
        text.push_str("\n  uncertainty check skipped to catch up");
    } else if uncertainty.samples.is_empty() {
        text.push_str("\n  uncertainty check off");
    } else {
        let _ = write!(
            text,
            "\n  {} noisy decodes in {}{}",
            uncertainty.samples.len(),
            ms(*check),
            if uncertainty.stopped_early {
                " (stopped early)"
            } else {
                ""
            }
        );
        for (i, sample) in uncertainty.samples.iter().enumerate() {
            let _ = write!(text, "\n    {}: {:?}", i + 1, sample.text());
        }
    }
    let _ = write!(
        text,
        "\n  {} unclear spans, marked: {:?}",
        uncertainty.analysis.spans.len(),
        transcript.marked
    );
    text
}

/// Runs `step` on the stream. A failed stream is dropped with a warning, and the segments it has
/// not covered are encoded on their own.
fn feed(
    stream: &mut Option<Box<dyn EncoderStream + '_>>,
    step: impl FnOnce(&mut dyn EncoderStream) -> Result<()>,
    events: &mut dyn FnMut(Event),
) {
    if let Some(s) = stream.as_deref_mut() {
        if let Err(error) = step(s) {
            events(Event::Warning(format!(
                "streaming encoder stopped: {error}"
            )));
            *stream = None;
        }
    }
}

/// Processes the waiting segments in order, as far as the stream has encoded them. Without a
/// stream, every segment goes at once, with what the draft of its utterance holds.
#[allow(clippy::too_many_arguments)]
fn drain(
    engine: &Engine,
    stream: &mut Option<Box<dyn EncoderStream + '_>>,
    draft: &mut Option<Draft>,
    waiting: &mut VecDeque<SpeechSegment>,
    clock: (u64, Instant),
    session: &mut SessionContext,
    sink: &mut dyn TextSink,
    events: &mut dyn FnMut(Event),
) -> Result<()> {
    while let Some(segment) = waiting.front() {
        let end = segment.start_sample + segment.samples.len() as u64;
        let encoded = match stream.as_deref_mut() {
            Some(s) => match s.take(segment.start_sample, end) {
                Some(frames) => Some(frames),
                None => break,
            },
            None => None,
        };
        let segment = waiting.pop_front().expect("front exists");
        let mut prepared = Prepared {
            encoded,
            ..Prepared::default()
        };
        let mut lag_seconds = lag(&segment, clock);
        if let Some(d) = draft.take().filter(|d| d.start == segment.start_sample) {
            prepared.prefix = d.prefix;
            if let Some((_, transcript, early_lag)) = d
                .early
                .filter(|(samples, _, _)| *samples == segment.samples.len())
            {
                // The transcript is as late as it was when it was made.
                prepared.transcript = Some(transcript);
                lag_seconds = early_lag;
            }
        }
        engine.process(&segment, prepared, lag_seconds, session, sink, events)?;
    }
    Ok(())
}

/// Decodes of the utterance in progress, for a recognizer without a streaming encoder.
struct Draft {
    /// The utterance's first sample.
    start: u64,
    /// Samples of the utterance at the last decode.
    decoded: usize,
    /// Samples the utterance grows by before the next decode: at least
    /// [`EngineConfig::partial_ms`], and twice what the last decode took, so drafting never takes
    /// more than half the time.
    step: usize,
    last: Option<Hypothesis>,
    /// What the last two decodes agreed on, which later decodes start with.
    prefix: Vec<u32>,
    /// The transcript made early in a pause, with the segment length it is for and how long
    /// after its audio it was made.
    early: Option<(usize, Transcript, f32)>,
}

/// Audio waiting in the channel beyond this means the dictation is behind, and drafting waits.
const DRAFT_BACKLOG: Duration = Duration::from_millis(300);

/// Decodes the utterance in progress again once it has grown by [`EngineConfig::partial_ms`], or
/// transcribes it as it would end once its pause has lasted [`EngineConfig::early_commit_ms`].
fn advance_draft<V: VoiceActivityDetector>(
    engine: &Engine,
    segmenter: &Segmenter<V>,
    draft: &mut Option<Draft>,
    clock: (u64, Instant),
    session: &SessionContext,
    events: &mut dyn FnMut(Event),
) -> Result<()> {
    let config = &engine.config;
    let Some((start, length, silence_ms)) = segmenter.open() else {
        *draft = None;
        return Ok(());
    };
    if draft.as_ref().is_none_or(|d| d.start != start) {
        *draft = Some(Draft {
            start,
            decoded: 0,
            step: 0,
            last: None,
            prefix: Vec::new(),
            early: None,
        });
    }
    let d = draft.as_mut().expect("set above");
    if clock.1.elapsed() > DRAFT_BACKLOG {
        return Ok(());
    }
    let pausing = config.early_commit_ms > 0 && silence_ms >= config.early_commit_ms;
    if pausing {
        if d.early.is_none() {
            if let Some(segment) = segmenter.preview() {
                let lag_seconds = lag(&segment, clock);
                let check =
                    lag_seconds <= config.max_lag_seconds && engine.rewriter.reads_uncertainty();
                let transcript = engine.transcribe(&segment, None, &d.prefix, session, check)?;
                d.early = Some((segment.samples.len(), transcript, lag_seconds));
            }
        }
        return Ok(());
    }
    if silence_ms == 0 {
        // Speech came back, so the pause will not end the utterance where the early transcript
        // thought.
        d.early = None;
    }
    let step = (config.partial_ms as usize * SAMPLE_RATE as usize / 1000).max(d.step);
    if config.partial_ms == 0 || length < d.decoded + step {
        return Ok(());
    }
    let Some(segment) = segmenter.preview() else {
        return Ok(());
    };
    let started = Instant::now();
    let encoder = engine.asr.encode(&segment.samples)?;
    let decoded = engine
        .asr
        .decode_after(&encoder, &session.bias, &d.prefix)?;
    let continued = d.prefix.len();
    if let Some(last) = &d.last {
        let agreed = engine.asr.agreed_prefix(last, &decoded);
        if agreed.len() > d.prefix.len() {
            d.prefix = agreed;
        }
    }
    let took = started.elapsed();
    d.decoded = length;
    d.step = (2.0 * took.as_secs_f32() * SAMPLE_RATE as f32) as usize;
    if engine.debug() {
        events(Event::Debug(format!(
            "draft of {:.2} s in {} after {continued} agreed tokens, {} agreed now: {:?}",
            segment.duration_seconds(),
            ms(took),
            d.prefix.len(),
            decoded.text(),
        )));
    }
    events(Event::Partial {
        text: decoded.text(),
    });
    d.last = Some(decoded);
    Ok(())
}

/// Runs one dictation until `audio` closes. Returns all inserted text.
pub fn run_dictation<V: VoiceActivityDetector>(
    engine: &Engine,
    vad: V,
    audio: Receiver<AudioChunk>,
    sink: &mut dyn TextSink,
    events: &mut dyn FnMut(Event),
) -> Result<String> {
    let mut session = engine.prepare(events);
    events(Event::Listening {
        hotwords: session.hotwords.len(),
    });
    let mut segmenter = Segmenter::new(vad, engine.config.segmenter.clone());
    let mut resampler = Resampler::default();
    // Samples heard so far, and when the last of them was recorded.
    let mut clock = (0, Instant::now());
    let mut stream = engine.asr.stream();
    // Segments that wait for the stream to encode their end.
    let mut waiting = VecDeque::new();
    let mut draft = None;
    let mut drafting = stream.is_none();

    let dictate = || -> Result<()> {
        for chunk in audio {
            let samples = chunk.to_model_input(&mut resampler);
            clock = (clock.0 + samples.len() as u64, chunk.captured);
            segmenter.set_sound_playing(chunk.sound_playing);
            feed(&mut stream, |s| s.accept(&samples), events);
            waiting.extend(segmenter.push(&samples));
            drain(
                engine,
                &mut stream,
                &mut draft,
                &mut waiting,
                clock,
                &mut session,
                sink,
                events,
            )?;
            if let (Some(s), true) = (stream.as_deref_mut(), waiting.is_empty()) {
                s.discard(segmenter.earliest_start());
            }
            if drafting {
                if let Err(error) =
                    advance_draft(engine, &segmenter, &mut draft, clock, &session, events)
                {
                    // The segment is still decoded when it closes, from scratch.
                    events(Event::Warning(format!(
                        "decoding speech in progress: {error}"
                    )));
                    draft = None;
                    drafting = false;
                }
            }
        }
        waiting.extend(segmenter.finish());
        // Finishing pads and encodes another chunk, which only a waiting segment needs.
        if !waiting.is_empty() {
            feed(&mut stream, |s| s.finish(), events);
        }
        drain(
            engine,
            &mut stream,
            &mut draft,
            &mut waiting,
            clock,
            &mut session,
            sink,
            events,
        )
    };
    let dictated = dictate();
    // Text already typed keeps its final period when a later segment fails.
    let finished = engine.finish(&mut session, sink, events);
    dictated?;
    finished?;
    events(Event::Finished);
    Ok(session.inserted)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::asr::{EncoderOutput, Recognizer, SearchConfig, SymbolTable, TransducerModel};
    use crate::context::{ContextSource, Region, TextBlock};
    use crate::features::Features;
    use crate::rewrite::PassThrough;

    /// Hears token 1 for every encoder frame that has speech. The fbank energy decides.
    struct LoudnessModel;

    impl TransducerModel for LoudnessModel {
        fn context_size(&self) -> usize {
            2
        }
        fn vocab_size(&self) -> usize {
            3
        }
        fn encode(&self, features: &Features) -> Result<EncoderOutput> {
            // One frame per 4 feature frames; emit "HELLO" once in the loud middle.
            let frames = features.frames / 4;
            let mut data = vec![0.0; frames * 3];
            for t in 0..frames {
                data[t * 3] = 5.0;
            }
            if frames > 0 {
                data[(frames / 2) * 3 + 1] = 10.0;
            }
            Ok(EncoderOutput {
                frames,
                dim: 3,
                data,
            })
        }
        fn decode(&self, contexts: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
            Ok(contexts.iter().map(|_| vec![0.0; 3]).collect())
        }
        fn join(&self, encoder: &[&[f32]], _: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
            Ok(encoder.iter().map(|f| f.to_vec()).collect())
        }
    }

    struct Collect(Vec<String>);

    impl TextSink for Collect {
        fn insert(&mut self, text: &str) -> Result<()> {
            self.0.push(text.to_owned());
            Ok(())
        }
    }

    struct Screen;

    impl ContextProvider for Screen {
        fn snapshot(&self) -> Result<ScreenContext> {
            Ok(ScreenContext {
                app: Some("Notes".into()),
                blocks: vec![TextBlock {
                    region: Region::Focused,
                    text: "Meeting with Hello Kitty".into(),
                }],
                source: ContextSource::Accessibility,
            })
        }
    }

    fn engine() -> Engine {
        engine_hearing("▁HELLO")
    }

    /// An engine whose recognizer hears `token` in each utterance.
    fn engine_hearing(token: &str) -> Engine {
        let symbols =
            SymbolTable::from_lines(["<blk> 0".into(), format!("{token} 1"), "▁KITTY 2".into()]);
        Engine {
            asr: Box::new(Recognizer::new(
                LoudnessModel,
                symbols,
                SearchConfig {
                    beam: 2,
                    ..SearchConfig::default()
                },
            )),
            context: Box::new(Screen),
            memory: Arc::default(),
            rewriter: Box::new(PassThrough),
            config: EngineConfig::default(),
        }
    }

    #[test]
    fn hotwords_include_names_from_recent_screens() {
        let engine = engine();
        let start = Instant::now();
        engine
            .memory
            .observe(&[], start - std::time::Duration::from_secs(5));
        engine.memory.observe(
            &[crate::context::Term {
                phrase: "Airicraft".into(),
                score: 1.0,
                form: crate::context::TermForm::Name,
            }],
            start,
        );
        let session = engine.prepare(&mut |_| {});
        assert_eq!(session.hotwords, vec!["Kitty", "Airicraft"]);
        // The test vocabulary cannot spell Airicraft, so the hotword graph leaves it out.
        assert_eq!(session.bias.phrases, vec!["Kitty"]);
    }

    fn tone(seconds: f32) -> Vec<f32> {
        (0..(48_000.0 * seconds) as usize)
            .map(|i| 0.3 * (i as f32 * 0.05).sin())
            .collect()
    }

    /// Two utterances, all recorded at `captured`.
    fn speech(captured: Instant) -> Receiver<AudioChunk> {
        let (sender, receiver) = std::sync::mpsc::channel();
        for samples in [
            vec![0.0; 24_000],
            tone(1.0),
            vec![0.0; 48_000],
            tone(1.0),
            vec![0.0; 48_000],
        ] {
            sender
                .send(AudioChunk {
                    samples,
                    sample_rate: 48_000,
                    channels: 1,
                    captured,
                    sound_playing: false,
                })
                .unwrap();
        }
        receiver
    }

    #[test]
    fn a_lone_full_stop_is_not_typed() {
        for token in [".", "。"] {
            let engine = engine_hearing(token);
            let mut sink = Collect(Vec::new());
            let mut events = Vec::new();
            run_dictation(
                &engine,
                crate::vad::EnergyVad::default(),
                speech(Instant::now()),
                &mut sink,
                &mut |e| events.push(e),
            )
            .unwrap();
            assert!(sink.0.is_empty(), "{token}: {:?}", sink.0);
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, Event::Transcribed { plain, .. } if plain == token)),
                "{token}: {events:?}"
            );
        }
    }

    #[test]
    fn full_stops_alone_are_nothing() {
        for plain in ["", " ", ".", "。", " . ", "。。"] {
            assert!(is_nothing(plain), "{plain:?}");
        }
        for plain in ["Hi.", "はい。", "?", "..a"] {
            assert!(!is_nothing(plain), "{plain:?}");
        }
    }

    #[test]
    fn a_sound_the_shell_played_is_not_dictated() {
        let engine = engine();
        let (sender, receiver) = std::sync::mpsc::channel();
        for (samples, sound_playing) in [
            (vec![0.0; 24_000], false),
            (tone(1.0), true),
            (vec![0.0; 48_000], false),
        ] {
            sender
                .send(AudioChunk {
                    samples,
                    sample_rate: 48_000,
                    channels: 1,
                    captured: Instant::now(),
                    sound_playing,
                })
                .unwrap();
        }
        drop(sender);
        let mut sink = Collect(Vec::new());
        let mut events = Vec::new();
        run_dictation(
            &engine,
            crate::vad::EnergyVad::default(),
            receiver,
            &mut sink,
            &mut |e| events.push(e),
        )
        .unwrap();
        assert!(sink.0.is_empty());
        assert!(
            events.iter().all(|e| !matches!(e, Event::Heard { .. })),
            "{events:?}"
        );
    }

    #[test]
    fn dictates_each_utterance_into_the_sink() {
        let engine = engine();
        let mut sink = Collect(Vec::new());
        let mut events = Vec::new();
        let text = run_dictation(
            &engine,
            crate::vad::EnergyVad::default(),
            speech(Instant::now()),
            &mut sink,
            &mut |e| events.push(e),
        )
        .unwrap();
        assert_eq!(sink.0, vec!["HELLO", " HELLO"]);
        assert_eq!(text, "HELLO HELLO");
        assert!(
            matches!(events[0], Event::Listening { hotwords: 1 }),
            "{events:?}"
        );
        // Each segment is announced before its transcript.
        let order: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                Event::Heard { seconds } if *seconds > 0.0 => Some("heard"),
                Event::Transcribed { .. } => Some("transcribed"),
                _ => None,
            })
            .collect();
        assert_eq!(
            order,
            ["heard", "transcribed", "heard", "transcribed"],
            "{events:?}"
        );
        // Debug logging is off.
        assert!(events.iter().all(|e| !matches!(e, Event::Debug(_))));
        // The audio was all recorded just now, so no segment came in late.
        assert!(events.iter().all(|e| !matches!(
            e,
            Event::Transcribed {
                behind_seconds: Some(_),
                ..
            }
        )));
    }

    /// Wraps a recognizer with a stream that encodes a segment only once it has heard `lag`
    /// more samples, and records each segment it hands out.
    struct Streaming {
        inner: Box<dyn Asr>,
        lag: u64,
        /// `(start, end, samples heard)` of each segment taken.
        taken: Arc<std::sync::Mutex<Vec<(u64, u64, u64)>>>,
        /// Whether a stream was finished.
        finished: Arc<AtomicBool>,
    }

    struct LaggingStream<'a> {
        asr: &'a Streaming,
        samples: Vec<f32>,
        finished: bool,
    }

    impl EncoderStream for LaggingStream<'_> {
        fn accept(&mut self, samples: &[f32]) -> Result<()> {
            self.samples.extend_from_slice(samples);
            Ok(())
        }
        fn finish(&mut self) -> Result<()> {
            self.finished = true;
            self.asr.finished.store(true, Ordering::Relaxed);
            Ok(())
        }
        fn take(&mut self, start: u64, end: u64) -> Option<EncoderOutput> {
            let heard = self.samples.len() as u64;
            if !self.finished && heard < end + self.asr.lag {
                return None;
            }
            self.asr.taken.lock().unwrap().push((start, end, heard));
            self.asr
                .inner
                .encode(&self.samples[start as usize..end as usize])
                .ok()
        }
    }

    impl Asr for Streaming {
        fn bias(&self, phrases: &[String]) -> Bias {
            self.inner.bias(phrases)
        }
        fn encode(&self, _: &[f32]) -> Result<EncoderOutput> {
            panic!("a streaming recognizer encodes through its stream")
        }
        fn decode(&self, encoders: &[EncoderOutput], bias: &Bias) -> Result<Vec<Hypothesis>> {
            self.inner.decode(encoders, bias)
        }
        fn frame_seconds(&self) -> f32 {
            self.inner.frame_seconds()
        }
        fn stream(&self) -> Option<Box<dyn EncoderStream + '_>> {
            Some(Box::new(LaggingStream {
                asr: self,
                samples: Vec::new(),
                finished: false,
            }))
        }
    }

    /// An engine whose recognizer streams with `lag`, with what its stream records.
    #[allow(clippy::type_complexity)]
    fn streaming_engine(
        lag: u64,
    ) -> (
        Engine,
        Arc<std::sync::Mutex<Vec<(u64, u64, u64)>>>,
        Arc<AtomicBool>,
    ) {
        let Engine {
            asr,
            context,
            rewriter,
            config,
            memory,
        } = engine();
        let (taken, finished) = (Arc::default(), Arc::default());
        let engine = Engine {
            asr: Box::new(Streaming {
                inner: asr,
                lag,
                taken: Arc::clone(&taken),
                finished: Arc::clone(&finished),
            }),
            context,
            rewriter,
            config,
            memory,
        };
        (engine, taken, finished)
    }

    #[test]
    fn a_streaming_recognizer_decodes_each_segment_once_its_frames_are_ready() {
        let lag = 16_000;
        let (engine, taken, finished) = streaming_engine(lag);
        let mut sink = Collect(Vec::new());
        run_dictation(
            &engine,
            crate::vad::EnergyVad::default(),
            speech(Instant::now()),
            &mut sink,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(sink.0, vec!["HELLO", " HELLO"]);
        let taken = taken.lock().unwrap();
        assert_eq!(taken.len(), 2, "{taken:?}");
        // The first segment waited for the stream; the second ended too close to the end of the
        // audio, so it came out when the stream finished.
        assert!(taken[0].2 >= taken[0].1 + lag, "{taken:?}");
        assert!(taken[1].2 < taken[1].1 + lag, "{taken:?}");
        assert!(taken[0].1 <= taken[1].0, "{taken:?}");
        assert!(finished.load(Ordering::Relaxed));
    }

    #[test]
    fn a_stream_with_no_segment_waiting_is_not_finished() {
        let (engine, taken, finished) = streaming_engine(16_000);
        let (sender, receiver) = std::sync::mpsc::channel();
        sender
            .send(AudioChunk {
                samples: vec![0.0; 96_000],
                sample_rate: 48_000,
                channels: 1,
                captured: Instant::now(),
                sound_playing: false,
            })
            .unwrap();
        drop(sender);
        let text = run_dictation(
            &engine,
            crate::vad::EnergyVad::default(),
            receiver,
            &mut Collect(Vec::new()),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(text, "");
        assert!(taken.lock().unwrap().is_empty());
        assert!(!finished.load(Ordering::Relaxed));
    }

    /// Writes every segment as the sentence "Hello.".
    struct Sentences;

    impl Rewriter for Sentences {
        fn rewrite(&self, _: &RewriteRequest) -> Result<Option<String>> {
            Ok(Some("Hello.".into()))
        }
    }

    #[test]
    fn a_final_period_waits_for_the_next_segment() {
        let mut engine = engine();
        engine.rewriter = Box::new(Sentences);
        let mut sink = Collect(Vec::new());
        let text = run_dictation(
            &engine,
            crate::vad::EnergyVad::default(),
            speech(Instant::now()),
            &mut sink,
            &mut |_| {},
        )
        .unwrap();
        // The second capital starts a new sentence, and the hotkey release ends it.
        assert_eq!(sink.0, vec!["Hello", ". Hello", "."]);
        assert_eq!(text, "Hello. Hello.");
    }

    /// Wraps a recognizer as one that continues decodes, and records the prefix of each decode.
    struct Continuing {
        inner: Box<dyn Asr>,
        prefixes: Arc<std::sync::Mutex<Vec<Vec<u32>>>>,
    }

    impl Asr for Continuing {
        fn bias(&self, phrases: &[String]) -> Bias {
            self.inner.bias(phrases)
        }
        fn encode(&self, samples: &[f32]) -> Result<EncoderOutput> {
            self.inner.encode(samples)
        }
        fn decode(&self, encoders: &[EncoderOutput], bias: &Bias) -> Result<Vec<Hypothesis>> {
            self.inner.decode(encoders, bias)
        }
        fn decode_after(
            &self,
            encoder: &EncoderOutput,
            bias: &Bias,
            prefix: &[u32],
        ) -> Result<Hypothesis> {
            self.prefixes.lock().unwrap().push(prefix.to_vec());
            self.inner.decode_one(encoder, bias)
        }
        fn agreed_prefix(&self, _: &Hypothesis, _: &Hypothesis) -> Vec<u32> {
            vec![1]
        }
        fn frame_seconds(&self) -> f32 {
            self.inner.frame_seconds()
        }
    }

    /// `pattern` of tone and silence seconds, sent in blocks of 100 ms. They count as recorded in
    /// the future, so however slow the test runs, the dictation never falls behind.
    fn blocks(pattern: &[(bool, f32)]) -> Receiver<AudioChunk> {
        let (sender, receiver) = std::sync::mpsc::channel();
        for (loud, seconds) in pattern {
            let samples = if *loud {
                tone(*seconds)
            } else {
                vec![0.0; (48_000.0 * seconds) as usize]
            };
            for block in samples.chunks(4_800) {
                sender
                    .send(AudioChunk {
                        samples: block.to_vec(),
                        sample_rate: 48_000,
                        channels: 1,
                        captured: Instant::now() + Duration::from_secs(600),
                        sound_playing: false,
                    })
                    .unwrap();
            }
        }
        receiver
    }

    #[test]
    fn an_utterance_decoded_early_in_its_pause_goes_out_when_the_pause_ends() {
        let Engine {
            asr,
            context,
            rewriter,
            config,
            memory,
        } = engine();
        let prefixes = Arc::default();
        let engine = Engine {
            asr: Box::new(Continuing {
                inner: asr,
                prefixes: Arc::clone(&prefixes),
            }),
            context,
            rewriter,
            config,
            memory,
        };
        let mut sink = Collect(Vec::new());
        let mut partials = Vec::new();
        run_dictation(
            &engine,
            crate::vad::EnergyVad::default(),
            blocks(&[(false, 0.3), (true, 2.5), (false, 1.5)]),
            &mut sink,
            &mut |e| {
                if let Event::Partial { text } = e {
                    partials.push(text);
                }
            },
        )
        .unwrap();
        assert_eq!(sink.0, vec!["HELLO"]);
        // Two drafts as the utterance grew, then the decode early in the pause, which continues
        // from what the drafts agreed on. Nothing decodes it again when the pause ends.
        assert_eq!(partials, vec!["HELLO", "HELLO"]);
        assert_eq!(*prefixes.lock().unwrap(), vec![vec![], vec![], vec![1]]);
    }

    #[test]
    fn speech_after_an_early_decode_makes_the_segment_decode_again() {
        let Engine {
            asr,
            context,
            rewriter,
            config,
            memory,
        } = engine();
        let prefixes = Arc::default();
        let engine = Engine {
            asr: Box::new(Continuing {
                inner: asr,
                prefixes: Arc::clone(&prefixes),
            }),
            context,
            rewriter,
            config: EngineConfig {
                partial_ms: 0,
                ..config
            },
            memory,
        };
        let mut sink = Collect(Vec::new());
        run_dictation(
            &engine,
            crate::vad::EnergyVad::default(),
            blocks(&[
                (false, 0.3),
                (true, 1.0),
                (false, 0.4),
                (true, 1.0),
                (false, 1.5),
            ]),
            &mut sink,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(sink.0, vec!["HELLO"]);
        // Early in the first pause, early in the second, and nothing when the second one ends.
        assert_eq!(prefixes.lock().unwrap().len(), 2);
    }

    /// A rewrite model that keeps every transcript as it is, reading the uncertainty markers.
    struct KeepsTranscript;

    impl Rewriter for KeepsTranscript {
        fn rewrite(&self, _: &RewriteRequest) -> Result<Option<String>> {
            Ok(None)
        }
    }

    #[test]
    fn without_a_rewrite_model_nothing_checks_uncertainty() {
        let engine = engine();
        engine.config.debug.store(true, Ordering::Relaxed);
        let mut transcripts = Vec::new();
        run_dictation(
            &engine,
            crate::vad::EnergyVad::default(),
            speech(Instant::now()),
            &mut Collect(Vec::new()),
            &mut |e| match e {
                Event::Debug(text) if text.starts_with("segment at") => transcripts.push(text),
                _ => {}
            },
        )
        .unwrap();
        assert_eq!(transcripts.len(), 2);
        assert!(transcripts
            .iter()
            .all(|t| t.contains("uncertainty check off")));
    }

    #[test]
    fn debug_events_show_the_context_the_decodes_and_the_rewrite() {
        let mut engine = engine();
        engine.rewriter = Box::new(KeepsTranscript);
        engine.config.debug.store(true, Ordering::Relaxed);
        let mut debug = Vec::new();
        run_dictation(
            &engine,
            crate::vad::EnergyVad::default(),
            speech(Instant::now()),
            &mut Collect(Vec::new()),
            &mut |e| match e {
                Event::Debug(text) if !text.starts_with("draft of") => debug.push(text),
                _ => {}
            },
        )
        .unwrap();
        // The context, then a decode and a rewrite for each utterance.
        assert_eq!(debug.len(), 5, "{debug:#?}");
        assert!(debug[0].contains("app \"Notes\""), "{}", debug[0]);
        assert!(debug[0].contains("Focused: \"Meeting with Hello Kitty\""));
        assert!(debug[0].contains("hotwords (1): Kitty"), "{}", debug[0]);
        assert!(
            debug[0].contains("sent to the recognizer (1): Kitty"),
            "{}",
            debug[0]
        );
        assert!(debug[1].contains("plain:  \"HELLO\""), "{}", debug[1]);
        assert!(debug[1].contains("noisy decodes"), "{}", debug[1]);
        assert!(debug[2].contains("<transcript>"), "{}", debug[2]);
        assert!(debug[2].contains("reply: off"), "{}", debug[2]);
    }

    #[test]
    fn lag_counts_from_when_the_audio_was_recorded() {
        // The microphone records while the worker still loads the VAD, so audio can wait before
        // the dictation starts.
        let mut behind = Vec::new();
        run_dictation(
            &engine(),
            crate::vad::EnergyVad::default(),
            speech(Instant::now() - Duration::from_secs(5)),
            &mut Collect(Vec::new()),
            &mut |e| {
                if let Event::Transcribed { behind_seconds, .. } = e {
                    behind.push(behind_seconds);
                }
            },
        )
        .unwrap();
        assert_eq!(behind.len(), 2);
        assert!(
            behind.iter().all(|b| b.is_some_and(|s| s > 5.0)),
            "{behind:?}"
        );
    }

    #[test]
    fn a_late_segment_skips_the_uncertainty_check() {
        let engine = engine();
        let mut session = engine.prepare(&mut |_| {});
        let segment = SpeechSegment {
            samples: AudioChunk {
                samples: tone(1.0),
                sample_rate: 48_000,
                channels: 1,
                captured: Instant::now(),
                sound_playing: false,
            }
            .to_model_input(&mut Resampler::default()),
            start_sample: 0,
            truncated: false,
            pre_roll: 0,
            post_roll: 0,
        };
        assert!(!engine
            .transcribe(&segment, None, &[], &session, true)
            .unwrap()
            .uncertainty
            .samples
            .is_empty());
        assert!(engine
            .transcribe(&segment, None, &[], &session, false)
            .unwrap()
            .uncertainty
            .samples
            .is_empty());

        let mut behind = Vec::new();
        for lag in [0.5, 5.0] {
            engine
                .process(
                    &segment,
                    Prepared::default(),
                    lag,
                    &mut session,
                    &mut Collect(Vec::new()),
                    &mut |e| {
                        if let Event::Transcribed { behind_seconds, .. } = e {
                            behind.push(behind_seconds);
                        }
                    },
                )
                .unwrap();
        }
        assert_eq!(behind, [None, Some(5.0)]);
    }
}
