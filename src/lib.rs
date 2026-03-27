//! propago: graph learning primitives on Burn tensors.
//!
//! Provides Poincare ball geometry and graph convolution layers that run on any
//! Burn backend (ndarray, wgpu, tch, etc.).
//!
//! - [`PoincareBall`]: hyperbolic geometry ops (project, mobius_add, exp/log maps)
//! - [`GCNConv`]: graph convolutional layer (linear + adjacency matmul)
//! - [`HGCNConv`]: hyperbolic graph convolution on the Poincare ball

#![forbid(unsafe_code)]

pub mod hyperbolic;
pub mod nn;

pub use hyperbolic::PoincareBall;
pub use nn::{GCNConv, HGCNConv};
