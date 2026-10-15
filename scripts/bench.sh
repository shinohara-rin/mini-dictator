#!/bin/sh
# Times the recognizer on the end-to-end test's clips (crates/core/examples/bench.rs). With a git
# ref, it times that commit first with this checkout's benchmark and clips, on the same machine,
# and prints a table comparing the two. Models come from scripts/e2e.sh, into $1 or target/e2e.
#
#   scripts/bench.sh [folder] [base ref]
#
# Writes target/bench/{base,head}.json and target/bench/summary.md.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:-$ROOT/target/e2e}
BASE=${2:-}
OUT="$ROOT/target/bench"
RUNS=${RUNS:-3}
mkdir -p "$OUT"
rm -f "$OUT/base.json" "$OUT/head.json"
E2E_FETCH_ONLY=1 sh "$ROOT/scripts/e2e.sh" "$DEST"
. "$DEST/env.sh"

bench() {
  manifest=$1
  shift
  cargo run --quiet --release --manifest-path "$manifest" --target-dir "$ROOT/target" \
    -p mini-dictator-core --features onnx --example bench -- \
    --ort "$MD_E2E_ORT" --model "$MD_E2E_MODEL" --vad "$MD_E2E_VAD" --runs "$RUNS" "$@"
}

note=
if [ -n "$BASE" ]; then
  # The base runs this checkout's benchmark and clips, so both sides measure the same work.
  tree="$OUT/base-tree"
  git -C "$ROOT" worktree remove --force "$tree" 2>/dev/null || rm -rf "$tree"
  git -C "$ROOT" worktree add --detach "$tree" "$BASE" > /dev/null
  mkdir -p "$tree/crates/core/examples" "$tree/crates/core/tests/data"
  cp "$ROOT/crates/core/examples/bench.rs" "$tree/crates/core/examples/"
  rm -rf "$tree/crates/core/tests/data/librispeech"
  cp -R "$ROOT/crates/core/tests/data/librispeech" "$tree/crates/core/tests/data/"
  echo "base: $BASE ($(git -C "$tree" rev-parse --short HEAD))"
  if ! bench "$tree/Cargo.toml" --json "$OUT/base.json"; then
    note="The base did not build or run this benchmark, so there is nothing to compare with."
  fi
  git -C "$ROOT" worktree remove --force "$tree"
fi

echo "head: $(git -C "$ROOT" rev-parse --short HEAD)"
if [ -f "$OUT/base.json" ]; then
  bench "$ROOT/Cargo.toml" --json "$OUT/head.json" --baseline "$OUT/base.json" > "$OUT/table.md"
else
  bench "$ROOT/Cargo.toml" --json "$OUT/head.json" > "$OUT/table.md"
fi
{
  echo "## Benchmark"
  echo
  head -n 1 "$OUT/table.md"
  echo
  [ -z "$note" ] || { echo "$note"; echo; }
  tail -n +2 "$OUT/table.md"
  echo
  echo "Medians of $RUNS runs on one shared runner. Changes under 15% are noise; 🔺 marks slower, 🟢 faster."
} > "$OUT/summary.md"
cat "$OUT/summary.md"
