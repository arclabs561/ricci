#!/usr/bin/env bash
# Set up the external inductive-link-prediction evaluator used for
# type-matched-negative and all-entity metrics.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_DIR="${INDUCTIVEEVAL_DIR:-$ROOT/target/inductiveeval}"
REPO="https://github.com/nomisto/inductiveeval.git"
REV="af9858391cd3a48cba7843ffafbd44306cafb213"

mkdir -p "$(dirname "$EVAL_DIR")"
if [ ! -d "$EVAL_DIR/.git" ]; then
  git clone "$REPO" "$EVAL_DIR"
fi

git -C "$EVAL_DIR" fetch --quiet origin "$REV"
git -C "$EVAL_DIR" checkout --quiet --detach "$REV"

uv venv --allow-existing --python 3.10 "$EVAL_DIR/.venv"
uv pip install --python "$EVAL_DIR/.venv/bin/python" --quiet \
  torch==2.12.1 \
  pykeen==1.10.2 \
  class-resolver==0.4.3

printf 'inductiveeval ready -> %s (%s)\n' "$EVAL_DIR" "$REV"
