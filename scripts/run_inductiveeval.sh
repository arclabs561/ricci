#!/usr/bin/env bash
# Run the pinned inductiveeval evaluator on an AnyBURL-style prediction export.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_DIR="${INDUCTIVEEVAL_DIR:-$ROOT/target/inductiveeval}"

usage() {
  printf 'usage: %s PREDICTIONS.txt\n' "${0##*/}" >&2
}

if [ "$#" -ne 1 ]; then
  usage
  exit 1
fi

PREDICTIONS="$1"
if [ ! -s "$PREDICTIONS" ]; then
  printf 'missing prediction export: %s\n' "$PREDICTIONS" >&2
  exit 1
fi

if [ ! -x "$EVAL_DIR/.venv/bin/python" ]; then
  "$ROOT/scripts/setup_inductiveeval.sh"
fi

export LC_ALL=C
cd "$EVAL_DIR"
.venv/bin/python eval.py -d fb237 -v v1 -p "$PREDICTIONS" |
  sed -E 's/([0-9]),([0-9])/\1.\2/g'
