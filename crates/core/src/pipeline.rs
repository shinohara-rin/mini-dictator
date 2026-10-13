//! One dictation from hotkey press to inserted text.
//!
//! A shell calls [`run_dictation`] on a worker thread when the user presses the dictation hotkey. It
//! sends microphone blocks on the channel and drops the sender when the user releases the hotkey
//! (or presses it again). The context snapshot is taken first, before any speech is decoded, so a
//! slow accessibility read delays nothing: audio waits in the channel.

use std::sync::mpsc::Receiver;

use crate::asr::{ContextGraph, Hypothesis, Recognizer, TransducerModel};
use crate::audio::{AudioChunk, Resampler};
use crate::context::{rank_terms, ContextProvider, ScreenContext};
use crate::rewrite::{RewriteRequest, Rewriter};
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
    /// A segment was decoded. `marked` carries the uncertainty markers.
    Transcribed {
        plain: String,
        marked: String,
        unclear_spans: usize,
    },
    Inserted {
        text: String,
    },
    /// A step failed. The dictation goes on when it can: a failed rewrite inserts the plain transcript.
    Warning(String),
    Finished,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub segmenter: SegmenterConfig,
    pub monte_carlo: MonteCarloConfig,
    /// Screen terms used as hotwords and shown to the rewrite model.
    pub max_hotwords: usize,
    /// Send focused-field text to the rewrite model. Hotwords are sent either way.
    pub share_screen_text: bool,
    /// Characters of focused-field text the rewrite model reads.
    pub preceding_chars: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            segmenter: SegmenterConfig::default(),
            monte_carlo: MonteCarloConfig::default(),
            max_hotwords: 100,
            share_screen_text: true,
            preceding_chars: 400,
        }
    }
}

/// What one dictation knows about the screen, prepared at its start.
pub struct SessionContext {
    pub screen: ScreenContext,
    pub hotwords: Vec<String>,
    pub graph: Option<ContextGraph>,
    /// Text inserted so far in this dictation.
    pub inserted: String,
}

/// One decoded segment.
#[derive(Debug, Clone)]
pub struct Transcript {
    pub reference: Hypothesis,
    pub uncertainty: Uncertainty,
    pub marked: String,
}

pub struct Engine<M> {
    pub recognizer: Recognizer<M>,
    pub context: Box<dyn ContextProvider>,
    pub rewriter: Box<dyn Rewriter>,
    pub config: EngineConfig,
}

impl<M: TransducerModel> Engine<M> {
    /// Reads the screen and builds the hotword graph. A failed read gives an empty context.
    pub fn prepare(&self, warn: &mut dyn FnMut(Event)) -> SessionContext {
        let screen = self.context.snapshot().unwrap_or_else(|error| {
            warn(Event::Warning(error.to_string()));
            ScreenContext::empty()
        });
        let hotwords: Vec<String> = rank_terms(&screen, self.config.max_hotwords)
            .into_iter()
            .map(|t| t.phrase)
            .collect();
        let graph = self
            .recognizer
            .context_graph(&hotwords, self.recognizer.search.hotword_boost);
        SessionContext {
            screen,
            hotwords,
            graph: (!graph.is_empty()).then_some(graph),
            inserted: String::new(),
        }
    }

    pub fn transcribe(
        &self,
        segment: &SpeechSegment,
        session: &SessionContext,
    ) -> Result<Transcript> {
        let encoder = self.recognizer.encode(&segment.samples)?;
        let reference = self.recognizer.search(&encoder, session.graph.as_ref())?;
        let uncertainty = uncertainty::estimate(
            &self.recognizer,
            &encoder,
            &reference,
            session.graph.as_ref(),
            &self.config.monte_carlo,
        )?;
        let marked = format_for_llm(&reference, &uncertainty.analysis.spans);
        Ok(Transcript {
            reference,
            uncertainty,
            marked,
        })
    }

    pub fn rewrite_request(
        &self,
        transcript: &Transcript,
        session: &SessionContext,
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
            session_text: session.inserted.clone(),
        }
    }

    /// Decodes, rewrites, and inserts one segment. Returns the inserted text, if any.
    pub fn process(
        &self,
        segment: &SpeechSegment,
        session: &mut SessionContext,
        sink: &mut dyn TextSink,
        events: &mut dyn FnMut(Event),
    ) -> Result<Option<String>> {
        let transcript = self.transcribe(segment, session)?;
        let plain = transcript.reference.text();
        events(Event::Transcribed {
            plain: plain.clone(),
            marked: transcript.marked.clone(),
            unclear_spans: transcript.uncertainty.analysis.spans.len(),
        });
        if plain.is_empty() && transcript.uncertainty.analysis.spans.is_empty() {
            return Ok(None);
        }
        let request = self.rewrite_request(&transcript, session);
        let text = match self.rewriter.rewrite(&request) {
            Ok(text) => text,
            Err(error) => {
                events(Event::Warning(error.to_string()));
                plain
            }
        };
        if text.trim().is_empty() {
            return Ok(None);
        }
        // Segments join with a space unless the text is CJK, which has none between sentences.
        let text = match (session.inserted.chars().last(), text.chars().next()) {
            (Some(last), Some(first))
                if !last.is_whitespace() && !is_cjk(last) && !is_cjk(first) =>
            {
                format!(" {text}")
            }
            _ => text,
        };
        sink.insert(&text).map_err(|e| Error::Sink(e.to_string()))?;
        session.inserted.push_str(&text);
        events(Event::Inserted { text: text.clone() });
        Ok(Some(text))
    }
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x3000..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xFF00..=0xFFEF)
}

/// Runs one dictation until `audio` closes. Returns all inserted text.
pub fn run_dictation<M: TransducerModel, V: VoiceActivityDetector>(
    engine: &Engine<M>,
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

    for chunk in audio {
        for segment in segmenter.push(&chunk.to_model_input(&mut resampler)) {
            engine.process(&segment, &mut session, sink, events)?;
        }
    }
    if let Some(segment) = segmenter.finish() {
        engine.process(&segment, &mut session, sink, events)?;
    }
    events(Event::Finished);
    Ok(session.inserted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::{EncoderOutput, SearchConfig, SymbolTable};
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

    #[test]
    fn dictates_each_utterance_into_the_sink() {
        let symbols =
            SymbolTable::from_lines(["<blk> 0", "▁HELLO 1", "▁KITTY 2"].map(String::from));
        let engine = Engine {
            recognizer: Recognizer::new(
                LoudnessModel,
                symbols,
                SearchConfig {
                    beam: 2,
                    ..SearchConfig::default()
                },
            ),
            context: Box::new(Screen),
            rewriter: Box::new(PassThrough),
            config: EngineConfig::default(),
        };
        let (sender, receiver) = std::sync::mpsc::channel();
        let tone = |seconds: f32| {
            (0..(48_000.0 * seconds) as usize)
                .map(|i| 0.3 * (i as f32 * 0.05).sin())
                .collect::<Vec<f32>>()
        };
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
                })
                .unwrap();
        }
        drop(sender);

        let mut sink = Collect(Vec::new());
        let mut events = Vec::new();
        let text = run_dictation(
            &engine,
            crate::vad::EnergyVad::default(),
            receiver,
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
    }
}
