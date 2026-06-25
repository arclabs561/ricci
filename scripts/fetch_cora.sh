#!/usr/bin/env bash
# Fetch the Cora citation-network dataset (node classification benchmark).
#
# Source: LINQS (UC Santa Cruz). cora.content is `paper_id <1433 binary
# features> label`; cora.cites is `cited_id citing_id`. 2708 nodes, 5429 edges,
# 7 classes. Data lands in propago/data/cora/ which is gitignored.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data"
URL="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"

mkdir -p "$DEST"
if [ -f "$DEST/cora/cora.content" ] && [ -f "$DEST/cora/cora.cites" ]; then
  echo "have cora/ -> $DEST/cora"
  exit 0
fi

echo "fetching cora.tgz"
curl -sSL --fail -o "$DEST/cora.tgz" "$URL"
tar xzf "$DEST/cora.tgz" -C "$DEST"
rm -f "$DEST/cora.tgz"
echo "done -> $DEST/cora"
