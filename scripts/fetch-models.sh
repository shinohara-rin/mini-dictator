#!/bin/sh
# Downloads the default models into the app's support folder (or the folder given as $1).
set -eu

DEST=${1:-"$HOME/Library/Application Support/Mini Dictator/models"}
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
MODEL=sherpa-onnx-zipformer-en-2023-06-26

mkdir -p "$DEST"
[ -d "$DEST/$MODEL" ] || curl -fsSL "$BASE/$MODEL.tar.bz2" | tar xj -C "$DEST"
[ -f "$DEST/silero_vad.onnx" ] || curl -fsSL -o "$DEST/silero_vad.onnx" "$BASE/silero_vad.onnx"
echo "Models are in $DEST"
