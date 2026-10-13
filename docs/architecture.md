# Architecture

Mini Dictator is a dictation app whose only UI is a tray icon. Press a hotkey, speak, and the text
appears in the focused app. This document describes the core, which is the same on every platform,
and the contract that each platform shell fills.

## The path of one dictation

```
hotkey down
  │
  ├─ context ── ContextProvider::snapshot()      accessibility tree, OCR as fallback
  │               └─ rank_terms() ─► hotwords ─► ContextGraph (token trie)
  │
  ├─ audio ──── AudioChunk ─► Resampler ─► Segmenter<VoiceActivityDetector>
  │                                            │ SpeechSegment per utterance
  │                                            ▼
  ├─ asr ────── Fbank ─► TransducerModel::encode ─► EncoderOutput
  │                                            │
  │                       modified beam search + ContextGraph ─► reference Hypothesis
  │                                            │
  ├─ uncertainty  EncoderOutput + N(0, σ·std_d) × K ─► batched search ─► K Hypotheses
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
| `features` | Kaldi-compatible 80-bin log mel fbank (no dither, no snipped edges) | `Fbank`, `Features` |
| `asr` | Transducer search that the core owns | `TransducerModel`, `Recognizer`, `ContextGraph`, `SymbolTable`, `Tokenizer`, `OnnxTransducer` |
| `uncertainty` | Monte Carlo over encoder output, span analysis | `MonteCarloConfig`, `estimate`, `analyze`, `format_for_llm` |
| `context` | Screen text model and hotword ranking | `ContextProvider`, `ScreenContext`, `TextBlock`, `Region`, `FallbackProvider`, `rank_terms` |
| `rewrite` | LLM post-processing | `Rewriter`, `RewriteRequest`, `OpenAiCompatible`, `PassThrough` |
| `pipeline` | One dictation, end to end | `Engine`, `run_dictation`, `TextSink`, `Event` |

Default features need no native code, so `cargo test` runs anywhere. `onnx` adds ONNX Runtime
(loaded at run time from a bundled shared library). `http` adds the rewrite client.

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

Defaults: `K = 12`, `σ = 1.0`, early stop after 4 samples when all agree. On the model's clean test
clips σ = 1.0 marks nothing; on the same clip at 3 dB SNR it marks the misheard words. For example,
with Silero VAD the reference `LAMPS OF LIGHT UP` comes out as
`LAMPS [unclear: "OF", also heard "WOULD", "DID", "SHOULD"] LIGHT UP`, and the truth is `WOULD`.
On a 4-core container, 12 noisy decodes of a 6 s utterance take about 0.35 s after a 0.2 s encode.

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
Ollama, llama.cpp). A failed rewrite inserts the plain transcript and reports a warning.

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
2. **Model choice.** A bilingual zh/en transducer (for example a sherpa-onnx zipformer zh-en), and
   int8 by default. The search assumes a stateless decoder, which every sherpa-onnx transducer has.
3. **Streaming.** Decode while the user speaks with a streaming zipformer, and run the noisy decodes
   as parallel streams, as AIRI's live probe does. Offline decoding per VAD segment is the first step
   because the rewrite needs the whole utterance anyway.
4. **Phonetic suggestions.** Port AIRI's phonetic matcher (pinyin and ARPAbet adapters) to suggest
   screen terms for spans the hotwords missed.
5. **Untranscribed speech.** Port AIRI's syllable nucleus detection to mark syllables that no token
   covers.
6. **Calibration.** Tune σ and the agreement threshold on recorded dictation with the `md` CLI.
