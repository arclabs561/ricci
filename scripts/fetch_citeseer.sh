#!/usr/bin/env bash
# Fetch the Citeseer citation-network dataset (node classification benchmark).
#
# Source: LINQS (UC Santa Cruz), same LBC format as Cora. citeseer.content is
# `paper_id <3703 binary features> label`; citeseer.cites is `cited_id citing_id`.
# 3312 nodes, 4732 edges, 6 classes. Data lands in data/citeseer/ (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data"
URL="https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz"

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
if has_lines "$DEST/citeseer/citeseer.content" 3312 && has_lines "$DEST/citeseer/citeseer.cites" 4732; then
  printf 'have Citeseer -> %s/citeseer\n' "$DEST"
  exit 0
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/ricci-citeseer.XXXXXX")"
cleanup() {
  rm -rf "$tmp"
}
trap cleanup EXIT

printf 'fetching Citeseer from %s\n' "$URL"
curl --fail --location --silent --show-error --output "$tmp/citeseer.tgz" "$URL"
tar xzf "$tmp/citeseer.tgz" -C "$tmp"
verify_lines "$tmp/citeseer/citeseer.content" 3312
verify_lines "$tmp/citeseer/citeseer.cites" 4732
mkdir -p "$DEST/citeseer"
mv "$tmp/citeseer/citeseer.content" "$DEST/citeseer/citeseer.content"
mv "$tmp/citeseer/citeseer.cites" "$DEST/citeseer/citeseer.cites"
printf 'done -> %s/citeseer (%s nodes, %s citation rows)\n' \
  "$DEST" \
  "$(wc -l < "$DEST/citeseer/citeseer.content")" \
  "$(wc -l < "$DEST/citeseer/citeseer.cites")"
