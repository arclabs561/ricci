# ricci examples

Each example is runnable from the repo root. Output excerpts below are real,
captured from release runs.

## Which example should I run?

| I want to... | Example |
|---|---|
| Check Poincare exp/log round-trip behavior | `burn_poincare_smoke` |
| Check HGCN layer output shapes | `hgcn_smoke` |
| Train a GCN on a citation graph | `cora_node_classification` |
| Train link prediction that transfers to unseen entities | `inductive_link_prediction` |
| See a graph-derived score bias change attention choices | `attention_bias_tropical` |
| Fit a tiny task-local adapter over frozen features | `test_time_adapter` |

## Geometry

### `burn_poincare_smoke`: does `exp0(log0(x))` recover the point?

Projects two points into the Poincare ball, maps through `log0`, then maps back
through `exp0`.

```bash
cargo run --release --example burn_poincare_smoke
```

```text
x2 (first row): [0.09997763, -0.049988814, 0.019995525]
```

The recovered point is close to the original first row
`[0.10, -0.05, 0.02]`.

### `hgcn_smoke`: what shape does an HGCN layer return?

Runs `HGCNConv` on six synthetic node features with identity adjacency, then
runs the same layer with an explicit basepoint.

```bash
cargo run --release --example hgcn_smoke
```

```text
y shape: [6, 4]
y(basepoint) shape: [6, 4]
```

## Citation Graphs

### `cora_node_classification`: can a two-layer GCN train on Cora?

Trains a full-batch two-layer `GCNConv` model on the Planetoid/LBC Cora dataset
using a 20-per-class train split and a 1000-node test split.

```bash
./scripts/fetch_cora.sh
cargo run --release --example cora_node_classification
```

```text
dataset: cora  nodes: 2708  features: 1433  classes: 7  train: 140  test: 1000
epoch   1  loss 1.9591  train acc 0.1429  test acc 0.1500
epoch  20  loss 0.3667  train acc 0.9857  test acc 0.7850
epoch  80  loss 0.0328  train acc 0.9929  test acc 0.8010
epoch 200  loss 0.0226  train acc 0.9929  test acc 0.7920

final test accuracy: 0.7920
```

The same example accepts `citeseer` after fetching that dataset:

```bash
./scripts/fetch_citeseer.sh
cargo run --release --example cora_node_classification citeseer
```

If the dataset is absent, the example exits 0 and prints the fetch command.

## Knowledge Graphs

### `inductive_link_prediction`: does a trained model transfer to new entities?

Trains an NBFNet-shaped model (Zhu et al., NeurIPS 2021) from `NBFConv`'s
edge-list forward on the GraIL FB15k-237 v1 split, then evaluates on a
disjoint graph whose entities were never seen in training. Nothing
entity-specific is learned, so the trained model runs unchanged on the new
vocabulary. Data-gated: fetch first, then run. Default sum aggregation takes
about 35 minutes on CPU; exact PNA mode is slower.

```bash
scripts/fetch_grail_fb237v1.sh
cargo run --release --example inductive_link_prediction
```

Use WGPU where available:

```bash
cargo run --release --features wgpu --example inductive_link_prediction
```

Reference-shaped PNA mode:

```bash
AGG=pna EPOCHS=8 cargo run --release --features wgpu --example inductive_link_prediction
```

Reference batch size plus full-rank diagnostics:

```bash
BATCH=64 FAILURE_DUMP=/tmp/ricci-failures.tsv EXPORT_PREDICTIONS=/tmp/ricci-predictions.txt \
  AGG=pna EPOCHS=8 cargo run --release --features wgpu --example inductive_link_prediction
```

```text
selected epoch 7 (valid MRR 0.3296)
full rank: mean 126.1  median 29  p90 339  p95 727  p99 1055  max 1088 (of 1093 entities)
full recall@k / Hits@k: @1 0.112  @3 0.232  @10 0.368  @50 0.583
sampled-50 recall@k / Hits@k: @1 0.407  @3 0.622  @10 0.817
sampled-50 rank: mean 6.7  median 2  p90 17  max 51
score margin gold-best-corrupt: mean -3.599  p10 -7.197  median -3.844; eval coverage 1.000  |h| 1.909
fb237_v1 -> fb237_v1_ind (both directions, n = 410):
Hits@10 (50 filtered negatives, GraIL protocol): 0.817
full-ranking filtered Hits@10: 0.368   MRR: 0.201
references on this split: GraIL 0.642, NBFNet 0.834 (50-neg protocol)
```

The sampled protocol is close to NBFNet's reported number. The full-ranking
numbers are lower and are the stricter diagnostic.

## Proof Sketches

### `attention_bias_tropical`: can a graph bias change attention choices?

Computes ordinary query/key scores, adds a positional bias and a max-plus
two-hop graph bias, then compares the row argmax before and after biasing.

```bash
cargo run --release --example attention_bias_tropical
```

```text
base row argmax:   [0, 1, 2, 1]
biased row argmax: [2, 0, 2, 0]
biased scores row 0: [0.9, 0.35, 1.05, 0.6999999]
```

This is an `attbias` / tropical-bias proof sketch, not public API.

### `test_time_adapter`: can a small adapter fit a local task?

Freezes synthetic two-dimensional features, fits a tiny linear adapter on four
support pairs, then evaluates two held-out pairs from the same task.

```bash
cargo run --release --example test_time_adapter
```

```text
support mse: baseline 1.4766 -> adapted 0.0000
heldout mse: baseline 1.8281 -> adapted 0.0000
```

This is an `adaptfit` proof sketch, not public API.
