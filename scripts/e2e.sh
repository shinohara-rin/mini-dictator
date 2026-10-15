#!/bin/sh
# Runs the end-to-end test (crates/core/tests/e2e.rs): LibriSpeech clips played through a mock
# microphone into the real pipeline. Downloads Qwen3-ASR 0.6B (about 1 GB), Silero VAD and ONNX
# Runtime into the folder given as $1 (target/e2e by default) the first time.
#
#   scripts/e2e.sh [folder]
#
# MODEL picks another sherpa-onnx model, as in fetch-models.sh. With E2E_FETCH_ONLY=1 it only
# downloads, and writes the paths to <folder>/env.sh for scripts/bench.sh.
set -eu

ORT_VERSION=1.31.0
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=$(mkdir -p "${1:-$ROOT/target/e2e}" && cd "${1:-$ROOT/target/e2e}" && pwd)
MODEL=${MODEL:-sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25}

case $(uname -s)-$(uname -m) in
  Linux-x86_64) PLATFORM=linux-x64 LIB=libonnxruntime.so ;;
  Linux-aarch64) PLATFORM=linux-aarch64 LIB=libonnxruntime.so ;;
  Darwin-arm64) PLATFORM=osx-arm64 LIB=libonnxruntime.dylib ;;
  *) echo "no ONNX Runtime build for $(uname -s) $(uname -m)" >&2; exit 1 ;;
esac
ORT="$DEST/onnxruntime-$PLATFORM-$ORT_VERSION"
if [ ! -f "$ORT/lib/$LIB" ]; then
  rm -rf "$ORT" "$ORT.tmp"
  mkdir -p "$ORT.tmp"
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-$PLATFORM-$ORT_VERSION.tgz" |
    tar xz -C "$ORT.tmp"
  # The macOS archive's paths start with ./, so take the one top-level folder instead of stripping.
  mv "$ORT.tmp"/*/ "$ORT"
  rm -rf "$ORT.tmp"
fi

MODEL=$MODEL sh "$ROOT/scripts/fetch-models.sh" "$DEST"

cat > "$DEST/env.sh" <<ENV
export MD_E2E_ORT="$ORT/lib/$LIB" MD_E2E_MODEL="$DEST/$MODEL" MD_E2E_VAD="$DEST/silero_vad.onnx"
ENV
[ "${E2E_FETCH_ONLY:-}" = 1 ] && exit 0

cd "$ROOT"
. "$DEST/env.sh"
cargo test --release -p mini-dictator-core --features onnx --test e2e -- --ignored --nocapture
