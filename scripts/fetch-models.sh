#!/bin/sh
# Downloads the default models into the app's support folder (or the folder given as $1).
set -eu

DEST=${1:-"$HOME/Library/Application Support/Mini Dictator/models"}
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
# Qwen3-ASR 0.6B (about 1 GB, about 30 languages). Set MODEL=sherpa-onnx-zipformer-en-2023-06-26 for
# the small English transducer instead, and point model_dir in config.toml at it.
MODEL=${MODEL:-sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25}

mkdir -p "$DEST"
[ -d "$DEST/$MODEL" ] || curl -fSL "$BASE/$MODEL.tar.bz2" | tar xj -C "$DEST"
[ -f "$DEST/silero_vad.onnx" ] || curl -fsSL -o "$DEST/silero_vad.onnx" "$BASE/silero_vad.onnx"
# The built-in rewrite model: Gemma 4 E2B (about 3.1 GB) and its audio projector (about 0.6 GB).
# Set REWRITER=0 to skip them when the rewrite goes to an endpoint instead.
if [ "${REWRITER:-1}" != 0 ]; then
  HF=https://huggingface.co
  for file in unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf \
    ggml-org/gemma-4-E2B-it-GGUF/mmproj-gemma-4-E2B-it-Q8_0.gguf; do
    name=${file##*/}
    if [ ! -f "$DEST/$name" ]; then
      curl -fSL -o "$DEST/$name.part" "$HF/${file%/*}/resolve/main/$name"
      mv "$DEST/$name.part" "$DEST/$name"
    fi
  done
fi
echo "Models are in $DEST"
