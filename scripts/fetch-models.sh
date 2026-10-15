#!/bin/sh
# Downloads the default models into the app's support folder (or the folder given as $1).
set -eu

DEST=${1:-"$HOME/Library/Application Support/Mini Dictator/models"}
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
# Qwen3-ASR 0.6B, about 30 languages. On a Mac it runs on the GPU through MLX, from the 4-bit MLX
# checkpoint (about 700 MB); elsewhere through ONNX Runtime, from the int8 sherpa-onnx export (about
# 1 GB). Set MODEL to a sherpa-onnx release name instead, such as
# sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 or the small English transducer
# sherpa-onnx-zipformer-en-2023-06-26, and point model_dir in config.toml at it.
if [ "$(uname -s)" = Darwin ]; then
  MODEL=${MODEL:-mlx-community/Qwen3-ASR-0.6B-4bit}
else
  MODEL=${MODEL:-sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25}
fi

mkdir -p "$DEST"
case $MODEL in
  */*)
    # A Hugging Face repository with an MLX checkpoint, into a folder named after it.
    dir="$DEST/${MODEL#*/}"
    mkdir -p "$dir"
    for file in config.json model.safetensors vocab.json merges.txt tokenizer_config.json \
      preprocessor_config.json chat_template.json generation_config.json; do
      if [ ! -f "$dir/$file" ]; then
        curl -fSL -o "$dir/$file.part" "https://huggingface.co/$MODEL/resolve/main/$file"
        mv "$dir/$file.part" "$dir/$file"
      fi
    done
    ;;
  *) [ -d "$DEST/$MODEL" ] || curl -fSL "$BASE/$MODEL.tar.bz2" | tar xj -C "$DEST" ;;
esac
[ -f "$DEST/silero_vad.onnx" ] || curl -fsSL -o "$DEST/silero_vad.onnx" "$BASE/silero_vad.onnx"
echo "Models are in $DEST"
