# Mini Dictator

A minimal AI dictation app: a tray icon, a hotkey, and your words in the focused app.

The core does VAD gating, transducer speech recognition biased toward the terms on your screen,
uncertainty estimation by Monte Carlo over the encoder output, and an LLM rewrite. See
[docs/architecture.md](docs/architecture.md).

## Status

The platform-independent core (`crates/core`) and a test CLI (`crates/cli`). The tray shells are next.

## Try the core on a WAV file

```sh
# A sherpa-onnx offline transducer, Silero VAD, and ONNX Runtime 1.31
curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2 | tar xj -C models
curl -L -o models/silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
# ONNX Runtime: https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0

cargo run --release -p mini-dictator-cli -- \
  --ort path/to/libonnxruntime.so \
  --model models/sherpa-onnx-zipformer-en-2023-06-26 \
  --vad models/silero_vad.onnx \
  --wav models/sherpa-onnx-zipformer-en-2023-06-26/test_wavs/1.wav \
  --hotword dishonoured --hotword "for ever"
```

Add `--rewrite-url https://api.openai.com/v1 --rewrite-model <model>` with `MD_API_KEY` set to run
the rewrite. Any OpenAI-compatible endpoint works, including a local Ollama at
`http://localhost:11434/v1`.

## Develop

```sh
cargo test --workspace
cargo clippy --workspace --all-targets --all-features
```
