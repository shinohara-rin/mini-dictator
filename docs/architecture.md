# Architecture

Mini Dictator is a dictation app whose only UI is a tray icon. Press a hotkey, speak, and the text
appears in the focused app. This document describes the core, which is the same on every platform,
and the contract that each platform shell fills.

## The path of one dictation

```
hotkey down
  │
  ├─ context ── ContextProvider::snapshot()      accessibility tree, OCR as fallback
  │               └─ rank_terms() ─► hotwords ─► Asr::bias()   prompt text or token trie
  │
  ├─ audio ──── AudioChunk ─► Resampler ─► Segmenter<VoiceActivityDetector>
  │                                            │ SpeechSegment per utterance
  │                                            ▼
  ├─ asr ────── Asr::encode() ─► EncoderOutput   Qwen3-ASR audio features, or transducer encoder output
  │                                            │
  │             Asr::decode() with the bias ─► reference Hypothesis
  │                                            │
  ├─ uncertainty  EncoderOutput + N(0, σ·std_d) × K ─► batched decode ─► K Hypotheses
  │               analyze() ─► unstable / dropped spans ─► format_for_llm()
  │                                            │  `the [unclear: "B", also heard "D"] ...`
  ├─ rewrite ── Rewriter::rewrite(marked transcript, screen terms, text before cursor)
  │                                            │
  └─ sink ───── TextSink::insert(text)         ▼
hotkey up ─► Segmenter::finish() flushes the last utterance
```

`pipeline::run_dictation` runs this on a worker thread. The shell sends microphone blocks on a
channel and drops the sender on hotkey up. The context snapshot happens before anything is decoded,
so a slow accessibility read costs no audio: blocks wait in the channel.

## Modules

| Module | Role | Main types |
|---|---|---|
| `audio` | Mono 16 kHz conversion | `AudioChunk`, `Resampler` |
| `vad` | Speech gating with hysteresis, pre-roll, hangover, max length | `VoiceActivityDetector`, `Segmenter`, `SpeechSegment`, `SileroVad` (v4 and v5), `EnergyVad` |
| `features` | Kaldi-compatible 80-bin fbank for transducers, Whisper's 128-bin log mel for Qwen3-ASR | `Fbank`, `WhisperMel`, `Features` |
| `asr` | The recognizer interface and its two backends | `Asr`, `Bias`, `load`; `QwenAsr`, `BpeTokenizer`; `TransducerModel`, `Recognizer`, `ContextGraph`, `OnnxTransducer` |
| `uncertainty` | Monte Carlo over encoder output, span analysis | `MonteCarloConfig`, `estimate`, `analyze`, `format_for_llm` |
| `context` | Screen text model and hotword ranking | `ContextProvider`, `ScreenContext`, `TextBlock`, `Region`, `FallbackProvider`, `rank_terms` |
| `rewrite` | LLM post-processing | `Rewriter`, `RewriteRequest`, `OpenAiCompatible`, `PassThrough` |
| `pipeline` | One dictation, end to end | `Engine`, `run_dictation`, `TextSink`, `Event` |

Default features need no native code, so `cargo test` runs anywhere. `onnx` adds ONNX Runtime
(loaded at run time from a bundled shared library). `http` adds the rewrite client.

### Two recognizers

The pipeline drives an `Asr`: `encode` once per segment, then `decode` a batch of encoder outputs
with a `Bias` built from the screen terms. `asr::load` picks the backend from the model directory.

**Qwen3-ASR 0.6B** is the default. It is an audio encoder in front of a Qwen3 LLM and covers about 30
languages and Chinese dialects, with punctuation and casing. The core runs the sherpa-onnx export
(`conv_frontend.onnx`, `encoder.int8.onnx`, `decoder.int8.onnx`) itself:

- Whisper features (`WhisperMel`, within 3e-5 of Hugging Face's extractor), the 8x conv frontend in
  100-frame chunks, and the encoder with a mask of the valid audio tokens (12.5 per second).
- A chat prompt with hotwords in the system turn and one `<|audio_pad|>` per audio token. The decoder
  replaces those with the audio features and returns the key/value entries of the tokens it read; the
  core keeps the cache and decodes greedily, stopping at `<|im_end|>`, `<|endoftext|>`, or a
  repetition loop. It strips the `language X<asr_text>` header and every added token, since the
  model sometimes opens its answer with another `<|im_start|>`.
- `BpeTokenizer` is Qwen2's byte-level BPE (NFC, the Qwen2 split pattern, merges by rank), matching
  Hugging Face on the test strings. Chat tokens go into the prompt by id and are never parsed from
  text, so screen text cannot open a chat turn.
- Hotwords are joined in rank order up to 96 tokens: every prompt token costs a decoder pass in every
  noisy decode.
- The decoder gives no token times, so a hypothesis is a list of words (CJK characters and
  punctuation on their own), which keeps uncertainty from comparing BPE pieces.

The int8 decoder quantizes activations per call, so a row's result depends slightly on the other rows
of its batch. The clean decode therefore runs alone, and noisy decodes batch among themselves.

**Transducers** (sherpa-onnx zipformers) remain as a small, fast fallback for one language.

### Why the core runs its own transducer search

sherpa-onnx ships the recognizer as a black box. Two features need its parts:

- **Hotwords.** `ContextGraph` is an Aho-Corasick trie over token ids, as in icefall. Each beam
  hypothesis carries a state. A token that extends a hotword earns `boost`; a broken prefix gives
  its bonus back; a finished hotword keeps it. On the sherpa-onnx English zipformer,
  `--hotword dishonoured --hotword "for ever"` turns `DISHONORED ... FOREVER` into the reference
  spelling `DISHONOURED ... FOR EVER`.
- **Uncertainty.** Noise goes on the encoder output, so the core needs it on its own.

### Uncertainty: Monte Carlo over the encoder output

The encoder runs once. `estimate` adds Gaussian noise with standard deviation `σ · std_d` to each
encoder dimension `d`, where `std_d` is that dimension's standard deviation over the utterance (the
dimensions differ in scale). It decodes `K` noisy copies with the same search and the same hotwords
as the clean decode, all in one batched search, so each frame costs one decoder call and one joiner
call for every hypothesis of every copy.

AIRI's probe (`stage-ui/libs/voice/asr-uncertainty`) perturbs the audio and re-runs the whole
model. Encoder noise is cheaper, and frames never move, so decodes compare by frame time without
mapping clocks. The span analysis is a port of AIRI's: per-token agreement by time and by text,
`unstable` runs of reference tokens, and `dropped` runs where many samples emit tokens that the
reference does not. Spans then widen to whole words, because a BPE model is often unsure of one piece
(`THEL` in `BROTHELS`), and the rewrite model needs the word.

Each backend sets its defaults (`Asr::monte_carlo`), and `config.toml` can override them.

**Transducer:** `K = 12`, `σ = 1.0`, early stop after 4 samples when all agree. On the model's clean
test clips σ = 1.0 marks at most a spelling variant (`DISHONORED`, also heard `DISHONOURED`); on a clip
at 3 dB SNR it marks the misheard words. For example, with Silero VAD the reference
`LAMPS OF LIGHT UP` comes out as `LAMPS [unclear: "OF", also heard "WOULD", "DID", "SHOULD"] LIGHT UP`,
and the truth is `WOULD`. On a 4-core container, 12 noisy decodes of a 6 s utterance take about 0.35 s
after a 0.2 s encode.

**Qwen3-ASR:** noise goes on the audio features that the LLM reads, `K = 4`, `σ = 0.5`, early stop
after 2. Without token times, tokens match by text alignment alone, and dropped speech is where enough
samples insert words between the same two reference words. On the export's test clips with Silero VAD:

- French, correct: nothing marked.
- Russian, four words wrong (truth: `Барсук, живущий в киевском зоопарке, совершил побег из своего
  вольера`): `[unclear: "борцы, живущие", also heard "Борсук живущий", ...] в Киевском
  [unclear: "запарке, совершили", ...] побег [unclear: "от", also heard "из"] своего
  [unclear: "валиера", also heard "вольера"].` Every wrong word is marked, most with the right reading.
- Code-switched English, French, Italian, Spanish: the English part is clean and the misheard rest
  is one span.

This costs much more than for a transducer, because each noisy decode reruns the whole LLM. On the
4-core container, a 6 s utterance takes about 0.5 s to encode and 3 to 4 s to decode (a 1.4 s prompt
pass, then about 80 ms per token), and 4 noisy decodes take 10 to 14 s. Apple silicon should be
several times faster; measure there before tuning, and set `samples = 0` to turn the check off.

### Active context

A shell returns `TextBlock`s tagged with a `Region` (`Focused` text before the cursor,
`WindowTitle`, other `Window` elements) instead of one string, because a short standalone element
(a contact, a channel, a file) is often a name. `rank_terms` adapts AIRI's conversation term rules:
quoted spans, identifiers (`useMemo`, `ASR`, `Node.js`, `gpt-4o`), runs of capitalized words, and
short CJK elements. Occurrences weigh 3 in the focused field, 2 in the title, and 1 elsewhere.

`FallbackProvider` tries accessibility first and runs OCR only when accessibility returns too
little text (games, canvas apps, some Electron apps).

Hotwords go two places: the `ContextGraph` biases the search, and the top 30 go to the rewrite model
as `Screen terms`, which fixes spellings the acoustic model cannot know. The text before the cursor
goes to the rewrite model only when `share_screen_text` is on.

### Rewrite

`rewrite::SYSTEM_PROMPT` tells the model to fix recognition errors, punctuation, and fillers without
adding content, to resolve `[unclear: ...]` spans from context and screen terms, and to reply with
the final text only. `OpenAiCompatible` talks to any chat completions endpoint (hosted, OpenRouter,
Ollama, llama.cpp). The reply loses a leading `<think>` block and any chat template tokens a server
lets through. A failed rewrite, or one with no text left, inserts the plain transcript and reports a
warning.

## Platform shell contract

A shell provides:

| Need | Core interface | macOS | Windows | Linux |
|---|---|---|---|---|
| Tray menu | none (`Event` drives the icon) | `tray-icon` | `tray-icon` | `tray-icon` (AppIndicator) |
| Global hotkey | calls `run_dictation` | `global-hotkey` | `global-hotkey` | `global-hotkey` (X11), portal on Wayland |
| Microphone | sends `AudioChunk` | `cpal` | `cpal` | `cpal` |
| Screen text | `ContextProvider` | AX API | UI Automation | AT-SPI |
| OCR fallback | `ContextProvider` | Vision | `Windows.Media.Ocr` | Tesseract or a small ONNX OCR |
| Text insertion | `TextSink` | clipboard paste, restored after | `SendInput` Unicode | clipboard paste via portal or `wtype`/`xdotool` |

## Language and runtime: Rust

- A tray app runs all day, so idle memory and start time matter. A Rust binary with
  `tray-icon`, `global-hotkey`, and `cpal` idles at a few MB. An Electron or Tauri webview would hold a
  renderer for a UI that does not exist.
- The core needs the encoder output and the search loop, not a packaged recognizer. `ort` exposes ONNX
  sessions directly and loads ONNX Runtime at run time (`load-dynamic`), so one bundled shared library
  per platform covers every model, with CoreML or DirectML execution providers later.
- The accessibility APIs are C, COM, and D-Bus APIs with mature Rust bindings on all three platforms.
- The core has no async runtime. It runs on one worker thread per dictation, and the shell owns the
  event loop.

## Next steps

1. **Shells.** A macOS shell first (AX + Vision + paste), then Windows, then Linux.
2. **Cheaper uncertainty for the LLM.** Feed the clean transcript to each noisy decode in one pass
   (teacher forcing) and only decode freely where a sample disagrees, which removes most decoder steps.
   The CoreML execution provider may also speed up the prompt pass.
3. **Streaming.** Decode while the user speaks with a streaming zipformer, and run the noisy decodes
   as parallel streams, as AIRI's live probe does. Offline decoding per VAD segment is the first step
   because the rewrite needs the whole utterance anyway.
4. **Phonetic suggestions.** Port AIRI's phonetic matcher (pinyin and ARPAbet adapters) to suggest
   screen terms for spans the hotwords missed.
5. **Untranscribed speech.** Port AIRI's syllable nucleus detection to mark syllables that no token
   covers.
6. **Calibration.** Tune σ and the agreement threshold on recorded dictation with the `md` CLI.
