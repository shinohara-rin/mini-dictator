#!/bin/sh
# Builds "Mini Dictator.app" (Apple Silicon) in target/release, with ONNX Runtime in Contents/Frameworks.
#
#   scripts/bundle-macos.sh [version]
#
# The app is signed ad hoc. macOS asks for microphone and accessibility permission on first use;
# a rebuilt app counts as a new app for accessibility, so remove and re-add it in
# System Settings > Privacy & Security > Accessibility after a rebuild.
set -eu

ORT_VERSION=1.31.0
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VERSION=${1:-$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/crates/macos/Cargo.toml")}
APP="$ROOT/target/release/Mini Dictator.app"
CACHE="$ROOT/target/onnxruntime-$ORT_VERSION"

cargo build --release -p mini-dictator-macos --manifest-path "$ROOT/Cargo.toml"

# The archive's paths start with ./, so unpack into a scratch folder and take the folder inside.
if [ ! -f "$CACHE/lib/libonnxruntime.$ORT_VERSION.dylib" ]; then
  rm -rf "$CACHE" "$CACHE.tmp"
  mkdir -p "$CACHE.tmp"
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-osx-arm64-$ORT_VERSION.tgz" \
    | tar xz -C "$CACHE.tmp"
  mv "$CACHE.tmp"/onnxruntime-* "$CACHE"
  rm -rf "$CACHE.tmp"
fi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Frameworks" "$APP/Contents/Resources"
cp "$ROOT/target/release/mini-dictator" "$APP/Contents/MacOS/"

# The app icon, every size iconutil wants, scaled down from the 1024 px master.
ICONSET="$ROOT/target/AppIcon.iconset"
rm -rf "$ICONSET"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
  sips -z $size $size "$ROOT/assets/icon/AppIcon.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  sips -z $((size * 2)) $((size * 2)) "$ROOT/assets/icon/AppIcon.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
cp "$CACHE/lib/libonnxruntime.$ORT_VERSION.dylib" "$APP/Contents/Frameworks/libonnxruntime.dylib"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Mini Dictator</string>
  <key>CFBundleDisplayName</key><string>Mini Dictator</string>
  <key>CFBundleIdentifier</key><string>dev.shinohara-rin.mini-dictator</string>
  <key>CFBundleExecutable</key><string>mini-dictator</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION%%-*}</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>Mini Dictator turns your speech into text.</string>
</dict>
</plist>
PLIST

codesign --force --deep --sign - "$APP"
echo "Built $APP"
