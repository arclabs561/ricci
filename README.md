# ricci

[![crates.io](https://img.shields.io/crates/v/ricci.svg)](https://crates.io/crates/ricci)
[![Documentation](https://docs.rs/ricci/badge.svg)](https://docs.rs/ricci)
[![CI](https://github.com/arclabs561/ricci/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/ricci/actions/workflows/ci.yml)

Graph neural network layers.

## Quickstart

```toml
[dependencies]
ricci = "0.9"
burn = { version = "0.20", default-features = false, features = ["std"] }
burn-ndarray = "0.20"
```

Hyperbolic distance on the Poincare ball:

```rust
use burn::tensor::{backend::Backend, TensorData};
use burn_ndarray::NdArray;
use ricci::PoincareBall;

type B = NdArray<f32>;
let dev = <B as Backend>::Device::default();
let ball = PoincareBall::new(1.0);

let x = burn::tensor::Tensor::<B, 2>::from_data(
    TensorData::new(vec![0.10f32, 0.00, 0.00], [1, 3]), &dev,
);
let y = burn::tensor::Tensor::<B, 2>::from_data(
    TensorData::new(vec![0.00f32, 0.10, 0.00], [1, 3]), &dev,
);
let d = ball.distance(x, y).to_data().to_vec::<f32>().unwrap()[0];
assert!(d >= 0.0);
```

## Feature flags

- `wgpu`: enables Burn's WGPU backend. On macOS this runs through Metal.

## Geometry

The Poincare ball $\mathbb{B}^d_c = \{x \in \mathbb{R}^d : c\lVert x \rVert^2 < 1\}$ with curvature $-c$:

| Operation | Formula |
|-----------|---------|
| Distance | $d_c(x, y) = \frac{2}{\sqrt{c}} \text{arctanh}\bigl(\sqrt{c}\lVert -x \oplus_c y \rVert\bigr)$ |
| Mobius addition | $x \oplus_c y = \frac{(1 + 2c\langle x,y\rangle + c\lVert y\rVert^2)x + (1 - c\lVert x\rVert^2)y}{1 + 2c\langle x,y\rangle + c^2\lVert x\rVert^2\lVert y\rVert^2}$ |
| Exp map | $\exp_x^c(v) = x \oplus_c \bigl(\tanh\bigl(\frac{\sqrt{c}\lambda_x^c\lVert v\rVert}{2}\bigr)\frac{v}{\sqrt{c}\lVert v\rVert}\bigr)$ |
| Log map | $\log_x^c(y) = \frac{2}{\sqrt{c}\lambda_x^c}\text{arctanh}(\sqrt{c}\lVert -x \oplus_c y\rVert)\frac{-x \oplus_c y}{\lVert -x \oplus_c y\rVert}$ |

where $\lambda_x^c = \frac{2}{1 - c\lVert x\rVert^2}$ is the conformal factor.

## API surface

- `ricci::PoincareBall`: Poincare ball geometry (project, mobius_add, exp/log maps, distance, parallel transport).
- `ricci::GCNConv`: graph convolution (linear projection + adjacency matmul).
- `ricci::HGCNConv`: hyperbolic graph convolution on the Poincare ball.
- `ricci::RGCNConv`: relational graph convolution (per-relation transforms
  over an adjacency stack, optional basis decomposition) for typed graphs.
- `ricci::NBFConv`: conditional message passing (edge-type representations
  as forward inputs; indicator-initialized pair representations).
- `ricci::relgraph`: the graph of relations (four interaction-type
  adjacencies over relation nodes, inverses included).
  All conv layers derive Burn's `Module`, so they embed in trainable models.
- `ricci::scatter`: exact segment max/min helpers for edge-list aggregation.
- `ricci::curvature`: Ollivier-Ricci edge curvature over an adjacency matrix
  (lazy-walk `alpha`, entropic `W1`).
- `ricci::features`: homomorphism-count node features (walk and closed-walk
  profiles); these separate some graphs that 1-WL message passing cannot.

Inputs are shaped `[batch, d]` (row-major feature vectors).

## Examples

See [examples/README.md](examples/README.md) for runnable examples with
captured output.

## References

Each entry links to a mechanism-level summary in [docs/papers.md](docs/papers.md).

- Ollivier. Ricci curvature of Markov chains on metric spaces. Journal of
  Functional Analysis 256(3), 2009. The edge curvature computed here. [notes](docs/papers.md#ricci-curvature-of-markov-chains-on-metric-spaces-ollivier-2009)
- Topping, Di Giovanni, Chamberlain, Dong, Bronstein. Understanding
  over-squashing and bottlenecks on graphs via curvature. ICLR 2022.
  [arXiv:2111.14522](https://arxiv.org/abs/2111.14522). Negative curvature
  marks the bottleneck edges. [notes](docs/papers.md#understanding-over-squashing-and-bottlenecks-on-graphs-via-curvature-topping-di-giovanni-chamberlain-dong-bronstein-iclr-2022)
- Kipf, Welling. Semi-supervised classification with graph convolutional
  networks. ICLR 2017.
  [arXiv:1609.02907](https://arxiv.org/abs/1609.02907). `GCNConv`. [notes](docs/papers.md#semi-supervised-classification-with-graph-convolutional-networks-kipf-welling-iclr-2017)
- Ganea, Bécigneul, Hofmann. Hyperbolic neural networks. NeurIPS 2018.
  [arXiv:1805.09112](https://arxiv.org/abs/1805.09112). The Möbius
  operations behind `PoincareBall`. [notes](docs/papers.md#hyperbolic-neural-networks-ganea-becigneul-hofmann-neurips-2018)
- Chami, Ying, Ré, Leskovec. Hyperbolic graph convolutional neural
  networks. NeurIPS 2019.
  [arXiv:1910.12933](https://arxiv.org/abs/1910.12933). `HGCNConv`. [notes](docs/papers.md#hyperbolic-graph-convolutional-neural-networks-chami-ying-re-leskovec-neurips-2019)
- Cuturi. Sinkhorn distances: lightspeed computation of optimal
  transportation distances. NeurIPS 2013.
  [arXiv:1306.0895](https://arxiv.org/abs/1306.0895). The entropic `W1`
  solved per edge. [notes](docs/papers.md#sinkhorn-distances-cuturi-neurips-2013)
- Dell, Grohe, Rattan. Lovász meets Weisfeiler and Leman. ICALP 2018.
  [arXiv:1802.08876](https://arxiv.org/abs/1802.08876). Homomorphism counts
  as an expressiveness measure. [notes](docs/papers.md#lovasz-meets-weisfeiler-and-leman-dell-grohe-rattan-icalp-2018)
- Barceló, Geerts, Reutter, Ryschkov. Graph neural networks with local
  graph parameters. NeurIPS 2021.
  [arXiv:2106.06707](https://arxiv.org/abs/2106.06707). Hom-count features
  in practice. [notes](docs/papers.md#graph-neural-networks-with-local-graph-parameters-barcelo-geerts-reutter-ryschkov-neurips-2021)
- Schlichtkrull, Kipf, Bloem, van den Berg, Titov, Welling. Modeling
  relational data with graph convolutional networks. ESWC 2018.
  [arXiv:1703.06103](https://arxiv.org/abs/1703.06103). `RGCNConv`. [notes](docs/papers.md#modeling-relational-data-with-graph-convolutional-networks-schlichtkrull-et-al-eswc-2018)
- Zhu, Zhang, Xhonneux, Tang. Neural Bellman-Ford networks: a general
  graph neural network framework for link prediction. NeurIPS 2021.
  [arXiv:2106.06935](https://arxiv.org/abs/2106.06935). `NBFConv`. [notes](docs/papers.md#neural-bellman-ford-networks-zhu-zhang-xhonneux-tang-neurips-2021)
- Galkin, Yuan, Mostafa, Tang, Zhu. Towards foundation models for
  knowledge graph reasoning. ICLR 2024.
  [arXiv:2310.04562](https://arxiv.org/abs/2310.04562). `relgraph`. [notes](docs/papers.md#towards-foundation-models-for-knowledge-graph-reasoning-galkin-yuan-mostafa-tang-zhu-iclr-2024)

## License

MIT OR Apache-2.0
