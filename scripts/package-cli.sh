#!/bin/sh
# Packages the `md` CLI with ONNX Runtime beside it, so it runs without --ort.
#
#   scripts/package-cli.sh <linux-x64 | osx-arm64 | win-x64> [version]
#
# Writes target/dist/md-<version>-<platform>.tar.gz (a .zip on Windows). Run it after
# `cargo build --release -p mini-dictator-cli`.
set -eu

ORT_VERSION=1.31.0
PLATFORM=$1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VERSION=${2:-$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/crates/cli/Cargo.toml")}
NAME="md-$VERSION-$PLATFORM"
STAGE="$ROOT/target/dist/$NAME"
CACHE="$ROOT/target/onnxruntime-$ORT_VERSION-$PLATFORM"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-$PLATFORM-$ORT_VERSION"

# The archives differ in layout (the macOS one has a leading ./), so unpack into a scratch folder
# and take the one top-level folder from it.
if [ ! -f "$CACHE/LICENSE" ]; then
  rm -rf "$CACHE" "$CACHE.tmp"
  mkdir -p "$CACHE.tmp"
  case $PLATFORM in
    win-*)
      curl -fsSL -o "$CACHE.zip" "$ORT_URL.zip"
      7z x -bso0 -o"$CACHE.tmp" "$CACHE.zip"
      rm "$CACHE.zip"
      ;;
    *) curl -fsSL "$ORT_URL.tgz" | tar xz -C "$CACHE.tmp" ;;
  esac
  mv "$CACHE.tmp"/onnxruntime-* "$CACHE"
  rm -rf "$CACHE.tmp"
fi

rm -rf "$STAGE"
mkdir -p "$STAGE"
# ort looks for the platform's default library name next to the executable.
case $PLATFORM in
  win-*)
    cp "$ROOT/target/release/md.exe" "$STAGE/"
    cp "$CACHE/lib/onnxruntime.dll" "$STAGE/"
    ;;
  osx-*)
    cp "$ROOT/target/release/md" "$STAGE/"
    cp "$CACHE/lib/libonnxruntime.$ORT_VERSION.dylib" "$STAGE/libonnxruntime.dylib"
    ;;
  *)
    cp "$ROOT/target/release/md" "$STAGE/"
    cp "$CACHE/lib/libonnxruntime.so.$ORT_VERSION" "$STAGE/libonnxruntime.so"
    ;;
esac
cp "$ROOT/LICENSE" "$STAGE/LICENSE"
cp "$CACHE/LICENSE" "$STAGE/LICENSE-onnxruntime"
cp "$CACHE/ThirdPartyNotices.txt" "$STAGE/ThirdPartyNotices-onnxruntime.txt"

cd "$ROOT/target/dist"
case $PLATFORM in
  win-*) 7z a -bso0 "$NAME.zip" "$NAME" ;;
  *) tar czf "$NAME.tar.gz" "$NAME" ;;
esac
rm -rf "$STAGE"
echo "Packaged target/dist/$(ls "$NAME".*)"
