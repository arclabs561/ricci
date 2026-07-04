//! ricci: graph learning primitives on Burn tensors.
//!
//! Provides Poincare ball geometry and graph convolution layers that run on any
//! Burn backend (ndarray, wgpu, tch, etc.).
//!
//! - [`PoincareBall`]: hyperbolic geometry ops (project, mobius_add, exp/log maps,
//!   hyperbolic activation via [`PoincareBall::hyp_act`])
//! - [`GCNConv`]: graph convolutional layer (linear + adjacency matmul)
//! - [`HGCNConv`]: hyperbolic graph convolution on the Poincare ball, with
//!   optional activation via [`HGCNConv::forward_act`]
//! - [`curvature`]: Ollivier-Ricci edge curvature (the crate's namesake), the
//!   primitive behind curvature-based rewiring of oversquashed graphs
//! - [`features`]: homomorphism-count node features (walks + closed walks),
//!   the interpretable lift past 1-WL expressiveness

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod curvature;
pub mod features;
pub mod hyperbolic;
pub mod nn;

pub use curvature::{ollivier_ricci_curvatures, CurvatureConfig, CurvatureError, EdgeCurvature};
pub use features::hom_profile;
pub use hyperbolic::PoincareBall;
pub use nn::{GCNConv, HGCNConv};
