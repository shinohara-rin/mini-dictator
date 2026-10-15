#!/bin/sh
# Opens crates/macos/tests/data/page.html in Google Chrome and reads it through the accessibility
# API with the app's own reader (the ignored test in crates/macos/src/context.rs). macOS only; the
# process running it needs accessibility permission, and Chrome should be closed.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PAGE="$ROOT/crates/macos/tests/data/page.html"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TEST=reads_names_from_a_chrome_page
cd "$ROOT"

binary=$(cargo test -p mini-dictator-macos --no-run --message-format=json 2>/dev/null |
  jq -r 'select(.profile.test == true and .executable != null) | .executable' | head -n 1)
echo "test binary: $binary"

if [ -n "${CI:-}" ]; then
  # Which processes macOS may hold responsible, and which of them the runner image trusts.
  echo "process chain:"
  pid=$$
  while [ "$pid" -gt 1 ]; do
    echo "  $pid $(ps -o comm= -p "$pid")"
    pid=$(ps -o ppid= -p "$pid" | tr -d ' ')
  done
  echo "accessibility grants in the runner image:"
  sqlite3 "/Library/Application Support/com.apple.TCC/TCC.db" \
    "SELECT client, auth_value FROM access WHERE service = 'kTCCServiceAccessibility';" 2>&1 || true
fi

profile=$(mktemp -d)
"$CHROME" --user-data-dir="$profile" --no-first-run --no-default-browser-check \
  --disable-search-engine-choice-screen --new-window "file://$PAGE" > /dev/null 2>&1 &
chrome=$!
trap 'kill $chrome 2>/dev/null || true; rm -rf "$profile"' EXIT
sleep 8
osascript -e 'tell application "Google Chrome" to activate' || true
sleep 2

status=0
"$binary" --ignored --nocapture --exact "context::tests::$TEST" || status=$?
if [ "$status" != 0 ] && [ -n "${CI:-}" ]; then
  mkdir -p target/accessibility
  screencapture -x target/accessibility/screen.png || true
fi
exit $status
