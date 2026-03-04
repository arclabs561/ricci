# propago

[![crates.io](https://img.shields.io/crates/v/propago.svg)](https://crates.io/crates/propago)
[![Documentation](https://docs.rs/propago/badge.svg)](https://docs.rs/propago)
[![CI](https://github.com/arclabs561/propago/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/propago/actions/workflows/ci.yml)

Graph learning primitives built on `candle` tensors.

This repo focuses on a small set of reusable building blocks (layers + small loops), not a full
training framework.

## Quickstart

```toml
[dependencies]
propago = "0.1"
candle-core = "0.9"
candle-nn = "0.9"
```

Hyperbolic distance on the Poincaré ball (Tensor-native):

```rust
use candle_core::{Device, Tensor};
use propago::hyperbolic::CandlePoincareBall;

let dev = &Device::Cpu;
let x = Tensor::from_vec(vec![0.10f32, 0.00, 0.00], (1, 3), dev)?;
let y = Tensor::from_vec(vec![0.00f32, 0.10, 0.00], (1, 3), dev)?;

let ball = CandlePoincareBall::new(1.0);
let d = ball.distance(&x, &y)?.to_vec2::<f32>()?[0][0];
assert!(d >= 0.0);

# Ok::<(), candle_core::Error>(())
```

## Status / scope

- Keep interfaces small so higher-level graph stacks can integrate without tight coupling.
- `HGCNConv` uses a Tensor-native Poincaré ball implementation (`CandlePoincareBall`) so it can run
  on Candle backends (CPU/GPU).

## API surface

- `propago::GCNConv`: simple graph convolution (linear projection + adjacency matmul).
- `propago::HGCNConv`: hyperbolic graph convolution on the Poincaré ball.

Notes:
- Many ops assume inputs are shaped `[batch, d]` (row-major feature vectors).

## Backends

- **Candle (default)**: `--features backend-candle` (enabled by default).
- **Burn (opt-in)**: `--features backend-burn` exposes Burn-tensor Poincaré ops.
- **MLX (opt-in)**: `--features backend-mlx` builds `mlx-rs` and requires `cmake` + Xcode MetalToolchain; tests force CPU for determinism.

## License

MIT OR Apache-2.0
