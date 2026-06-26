#!/usr/bin/env bash
# Fetch the Citeseer citation-network dataset (node classification benchmark).
#
# Source: LINQS (UC Santa Cruz), same LBC format as Cora. citeseer.content is
# `paper_id <3703 binary features> label`; citeseer.cites is `cited_id citing_id`.
# 3312 nodes, 4732 edges, 6 classes. Data lands in propago/data/citeseer/ (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data"
URL="https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz"

mkdir -p "$DEST"
if [ -f "$DEST/citeseer/citeseer.content" ]; then
  echo "have citeseer/ -> $DEST/citeseer"
  exit 0
fi

echo "fetching citeseer.tgz"
curl -sSL --fail -o "$DEST/citeseer.tgz" "$URL"
tar xzf "$DEST/citeseer.tgz" -C "$DEST"
rm -f "$DEST/citeseer.tgz"
echo "done -> $DEST/citeseer"
