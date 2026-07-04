# ricci

[![crates.io](https://img.shields.io/crates/v/ricci.svg)](https://crates.io/crates/ricci)
[![Documentation](https://docs.rs/ricci/badge.svg)](https://docs.rs/ricci)
[![CI](https://github.com/arclabs561/ricci/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/ricci/actions/workflows/ci.yml)

Graph neural network layers, plus Ollivier-Ricci edge curvature
(the `curvature` module) for finding oversquashing bottlenecks.

## Quickstart

```toml
[dependencies]
ricci = "0.6"
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
  Both conv layers derive Burn's `Module`, so they embed in trainable models.
- `ricci::curvature`: Ollivier-Ricci edge curvature over an adjacency matrix
  (lazy-walk `alpha`, entropic `W1`).
- `ricci::features`: homomorphism-count node features (walk and closed-walk
  profiles), the interpretable expressiveness lift past 1-WL.

Inputs are shaped `[batch, d]` (row-major feature vectors).

## Examples

See [examples/README.md](examples/README.md) for runnable examples with
captured output.

## License

MIT OR Apache-2.0
