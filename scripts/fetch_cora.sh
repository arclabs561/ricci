#!/usr/bin/env bash
# Fetch the Cora citation-network dataset (node classification benchmark).
#
# Source: LINQS (UC Santa Cruz). cora.content is `paper_id <1433 binary
# features> label`; cora.cites is `cited_id citing_id`. 2708 nodes, 5429 edges,
# 7 classes. Data lands in data/cora/ which is gitignored.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data"
URL="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"

has_lines() {
  local path="$1"
  local expected="$2"
  [ -s "$path" ] || return 1
  [ "$(wc -l < "$path")" -eq "$expected" ]
}

verify_lines() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(wc -l < "$path")"
  if [ "$actual" -ne "$expected" ]; then
    printf 'unexpected line count for %s: got %s, want %s\n' "$path" "$actual" "$expected" >&2
    exit 1
  fi
}

mkdir -p "$DEST"
if has_lines "$DEST/cora/cora.content" 2708 && has_lines "$DEST/cora/cora.cites" 5429; then
  printf 'have Cora -> %s/cora\n' "$DEST"
  exit 0
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/ricci-cora.XXXXXX")"
cleanup() {
  rm -rf "$tmp"
}
trap cleanup EXIT

printf 'fetching Cora from %s\n' "$URL"
curl --fail --location --silent --show-error --output "$tmp/cora.tgz" "$URL"
tar xzf "$tmp/cora.tgz" -C "$tmp"
verify_lines "$tmp/cora/cora.content" 2708
verify_lines "$tmp/cora/cora.cites" 5429
mkdir -p "$DEST/cora"
mv "$tmp/cora/cora.content" "$DEST/cora/cora.content"
mv "$tmp/cora/cora.cites" "$DEST/cora/cora.cites"
printf 'done -> %s/cora (%s nodes, %s citation rows)\n' \
  "$DEST" \
  "$(wc -l < "$DEST/cora/cora.content")" \
  "$(wc -l < "$DEST/cora/cora.cites")"
