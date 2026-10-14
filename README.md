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
scripts/fetch-models.sh      # Qwen3-ASR 0.6B (about 1 GB), Silero VAD, and Gemma 4 E2B (about 3.7 GB) into ~/Library/Application Support/Mini Dictator/models
scripts/bundle-macos.sh      # builds target/release/Mini Dictator.app with ONNX Runtime and llama.cpp's server inside
open "target/release/Mini Dictator.app"
```

Press ⌥Space to start and again to stop, or hold it while you speak. A soft tick plays when the
microphone opens and a pop when it closes. The first run asks for microphone and accessibility
permission; accessibility is needed both to read names from the focused window and to paste.
Settings live in `~/Library/Application Support/Mini Dictator/config.toml` (the menu opens it): the
model, the hotkey, the sounds (`beep`), the uncertainty samples, and the rewrite endpoint. Names seen
on screen in the last few minutes count as well as the current window's; `screen_memory_seconds`
sets how fast they fade (0 turns the memory off).

The rewrite runs on the Mac by default: the app starts the bundled llama.cpp server with Gemma 4 E2B,
which reads the transcript and also hears the recording, so it can fix names and words the
recognizer misheard. It loads at launch, uses about 4 GB of memory while the app runs, and stops
when the app quits. Set `builtin = false` under `[rewrite]` to send the rewrite to `base_url`
instead (`send_audio = true` attaches the recording there too), or turn Rewrite off in the menu.
`fetch-models.sh` with `REWRITER=0` skips the Gemma download.

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
`http://localhost:11434/v1`. To run the built-in rewriter instead, pass a `llama-server` from a
[llama.cpp release](https://github.com/ggml-org/llama.cpp/releases) and the Gemma files:
`--llama-server path/to/llama-server --rewrite-gguf models/gemma-4-E2B-it-Q4_K_M.gguf
--rewrite-mmproj models/mmproj-gemma-4-E2B-it-Q8_0.gguf`.

## Develop

```sh
cargo test --workspace
cargo clippy --workspace --all-targets --all-features
```
