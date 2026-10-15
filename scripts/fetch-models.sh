#!/bin/sh
# Downloads the default models into the app's support folder (or the folder given as $1).
set -eu

DEST=${1:-"$HOME/Library/Application Support/Mini Dictator/models"}
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
# Qwen3-ASR 0.6B (about 1 GB, about 30 languages). Set MODEL=sherpa-onnx-zipformer-en-2023-06-26 for
# the small English transducer, or MODEL=sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11
# for streaming (about 680 MB, 40 locales, weaker on Chinese), and point model_dir in config.toml at it.
MODEL=${MODEL:-sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25}

mkdir -p "$DEST"
[ -d "$DEST/$MODEL" ] || curl -fSL "$BASE/$MODEL.tar.bz2" | tar xj -C "$DEST"
[ -f "$DEST/silero_vad.onnx" ] || curl -fsSL -o "$DEST/silero_vad.onnx" "$BASE/silero_vad.onnx"
echo "Models are in $DEST"
