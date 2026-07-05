#!/usr/bin/env bash
# Fetch the GraIL FB15k-237 v1 inductive split (Teru et al., ICML 2020;
# kkteru/grail): fb237_v1 is the training graph, fb237_v1_ind the disjoint
# inference graph with new entities (relations shared). Lines are
# entity \t relation \t entity.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data && cd data
if [ -f fb237_v1/train.txt ] && [ -f fb237_v1_ind/train.txt ]; then
  echo "data/fb237_v1(_ind) already present"
  exit 0
fi
BASE="https://raw.githubusercontent.com/kkteru/grail/master/data"
for d in fb237_v1 fb237_v1_ind; do
  mkdir -p "$d"
  for split in train valid test; do
    curl -sL -o "$d/$split.txt" "$BASE/$d/$split.txt"
  done
done
echo "fetched: $(wc -l < fb237_v1/train.txt) train / $(wc -l < fb237_v1_ind/test.txt) inductive-test triples"
echo "next: cargo run --release --example inductive_link_prediction"
