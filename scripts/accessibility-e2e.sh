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

# Starts Chrome on the page with the given flags and brings it to the front.
chrome=
profile=
start_chrome() {
  profile=$(mktemp -d)
  "$CHROME" --user-data-dir="$profile" --no-first-run --no-default-browser-check \
    --disable-search-engine-choice-screen "$@" --new-window "file://$PAGE" > /dev/null 2>&1 &
  chrome=$!
  sleep 8
  osascript -e 'tell application "Google Chrome" to activate' || true
  sleep 2
}
stop_chrome() {
  kill "$chrome" 2>/dev/null || true
  wait "$chrome" 2>/dev/null || true
  rm -rf "$profile" 2>/dev/null || true
}
trap stop_chrome EXIT
screenshot() {
  [ -n "${CI:-}" ] || return 0
  mkdir -p target/accessibility
  screencapture -x "target/accessibility/$1.png" || true
}

# A plain Chrome, as on a user's Mac: does the reader get it to build the page's tree? Reported,
# not failed, since GitHub's runner may differ from a Mac with a person at it.
start_chrome
if ! "$binary" --ignored --nocapture --exact context::tests::turns_on_chromes_tree; then
  screenshot plain
  echo "::warning::The reader did not get a plain Chrome to expose the page; see the test output above"
fi
stop_chrome

# Chrome with its tree forced on: the reader must find the page, its names and the focused field.
start_chrome --force-renderer-accessibility
status=0
"$binary" --ignored --nocapture --exact "context::tests::$TEST" || status=$?
[ "$status" = 0 ] || screenshot forced
exit $status
