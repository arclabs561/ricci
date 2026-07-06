#!/usr/bin/env bash
# Fetch the GraIL FB15k-237 v1 inductive split (Teru et al., ICML 2020;
# kkteru/grail): fb237_v1 is the training graph, fb237_v1_ind the disjoint
# inference graph with new entities (relations shared). Lines are
# entity \t relation \t entity.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data"
BASE="https://raw.githubusercontent.com/kkteru/grail/master/data"

expected_lines() {
  case "$1/$2" in
    fb237_v1/train) printf '4245' ;;
    fb237_v1/valid) printf '489' ;;
    fb237_v1/test) printf '492' ;;
    fb237_v1_ind/train) printf '1993' ;;
    fb237_v1_ind/valid) printf '206' ;;
    fb237_v1_ind/test) printf '205' ;;
    *) return 1 ;;
  esac
}

has_split() {
  local d="$1"
  local split="$2"
  local path="$DEST/$d/$split.txt"
  local expected
  expected="$(expected_lines "$d" "$split")"
  [ -s "$path" ] || return 1
  [ "$(wc -l < "$path")" -eq "$expected" ]
}

verify_split() {
  local d="$1"
  local split="$2"
  local path="$tmp/$d/$split.txt"
  local expected actual
  expected="$(expected_lines "$d" "$split")"
  actual="$(wc -l < "$path")"
  if [ "$actual" -ne "$expected" ]; then
    printf 'unexpected line count for %s/%s.txt: got %s, want %s\n' "$d" "$split" "$actual" "$expected" >&2
    exit 1
  fi
}

complete() {
  has_split fb237_v1 train &&
    has_split fb237_v1 valid &&
    has_split fb237_v1 test &&
    has_split fb237_v1_ind train &&
    has_split fb237_v1_ind valid &&
    has_split fb237_v1_ind test
}

if complete; then
  printf 'have GraIL FB15k-237 v1 -> %s/fb237_v1 and %s/fb237_v1_ind\n' "$DEST" "$DEST"
  exit 0
fi

mkdir -p "$DEST"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/ricci-grail.XXXXXX")"
cleanup() {
  rm -rf "$tmp"
}
trap cleanup EXIT

for d in fb237_v1 fb237_v1_ind; do
  mkdir -p "$tmp/$d"
  for split in train valid test; do
    curl --fail --location --silent --show-error \
      --output "$tmp/$d/$split.txt" \
      "$BASE/$d/$split.txt"
    verify_split "$d" "$split"
  done
done

mkdir -p "$DEST/fb237_v1" "$DEST/fb237_v1_ind"
for d in fb237_v1 fb237_v1_ind; do
  for split in train valid test; do
    mv "$tmp/$d/$split.txt" "$DEST/$d/$split.txt"
  done
done

printf 'fetched GraIL FB15k-237 v1: %s train / %s inductive-test triples\n' \
  "$(wc -l < "$DEST/fb237_v1/train.txt")" \
  "$(wc -l < "$DEST/fb237_v1_ind/test.txt")"
printf 'next: cargo run --release --example inductive_link_prediction\n'
