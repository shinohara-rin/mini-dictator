# Mini Dictator

A minimal AI dictation app: a tray icon, a hotkey, and your words in the focused app.

The core does VAD gating, speech recognition biased toward the terms on your screen (Qwen3-ASR 0.6B
by default, about 30 languages; sherpa-onnx transducers also work), uncertainty estimation by Monte
Carlo over the encoder output, and an LLM rewrite. See [docs/architecture.md](docs/architecture.md).

## Status

- `crates/core`: the platform-independent core.
- `crates/cli`: runs the core on a WAV file.
- `crates/macos`: the macOS menu bar app.

## macOS

```sh
scripts/fetch-models.sh      # Qwen3-ASR 0.6B (about 1 GB) and Silero VAD into ~/Library/Application Support/Mini Dictator/models
scripts/bundle-macos.sh      # builds target/release/Mini Dictator.app with ONNX Runtime inside
open "target/release/Mini Dictator.app"
```

Press ⌥Space to start and again to stop, or hold it while you speak. A soft tick plays when the
microphone opens and a pop when it closes. With Sounds set to Every Step (`sounds = "verbose"`, the
default for a new config), each step gets a little cue too, which also shows what the pipeline is
doing without a window: a gulp as a segment goes to the recognizer, a puzzled "hm-hm?" when it finds
unclear words, a deflating "pfff" when it heard nothing (or only a full stop), a zip when it is
catching up, a sparkle when text is typed (left out after either of those), and an "uh-oh" or a bonk
when something fails. A cue repeated within half a second plays once. Cancel Echo in the menu
(`echo_cancellation`, on by default) records through macOS voice processing, which takes the cues
out of the microphone so they are never dictated; it also turns down other apps' audio while you
dictate. The first run asks for microphone and accessibility permission; accessibility is needed to
see the hotkey, to read names from the focused window, and to paste. To pick another hotkey, choose
Change Hotkey in the menu, press the new combination, and let go; Esc cancels. A hotkey can be a key
with modifiers (`Ctrl+Shift+D`), a function key or the microphone key alone (`F13`, `Dictation`), or
modifiers alone (`RightCmd`), and Option and Command can name a side (`RightAlt`). Settings live in
`~/Library/Application Support/Mini Dictator/config.toml` (the menu opens its folder): the model,
the hotkey, the sounds (`sounds`: `off`, `minimal` or `verbose`), the uncertainty samples, and the
rewrite endpoint. Edits to the file apply on the next launch. Names seen on screen in the last few
minutes count as well as the current window's; `screen_memory_seconds` sets how fast they fade (0
turns the memory off). `sentence_pause_seconds` (1.5): without a rewrite, a shorter pause continues
the sentence instead of ending it with a period.

To see what a dictation did, check Write Debug Log in the menu, or set `debug = true` to have it on
from launch. It writes `debug.log` next to `config.toml`: the screen text and hotwords, each decode
with its noisy decodes and unclear spans, the rewrite prompt and reply, and timings. It holds what
was on your screen, so it is off by default; past 5 MB it moves to `debug.log.1`.

## Try the core on a WAV file

```sh
# Qwen3-ASR (or any sherpa-onnx offline transducer), Silero VAD, and ONNX Runtime 1.31
scripts/fetch-models.sh models
# ONNX Runtime: https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0

cargo run --release -p mini-dictator-cli -- \
  --ort path/to/libonnxruntime.so \
  --model models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 \
  --vad models/silero_vad.onnx \
  --wav models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/ru1.wav
```

The `--model` directory decides the backend. With a transducer such as
`sherpa-onnx-zipformer-en-2023-06-26`, `--beam` and `--boost` tune the hotword search.

Add `--rewrite-url https://api.openai.com/v1 --rewrite-model <model>` with `MD_API_KEY` set to run
the rewrite. Any OpenAI-compatible endpoint works, including a local Ollama at
`http://localhost:11434/v1`.

## Develop

```sh
cargo test --workspace
cargo clippy --workspace --all-targets --all-features
scripts/e2e.sh   # LibriSpeech clips through a mock microphone into the real models (downloads about 1 GB)
```
