# Changelog

## [Unreleased]

### Added

- `scatter::{scatter_max, scatter_min, scatter_max_min}`: exact segment
  max/min helpers over edge lists. These compute winner indices from a host
  snapshot, then use differentiable gathers so gradients route to the
  winning edge values.
- `examples/inductive_link_prediction` accepts `AGG=pna`, using the new
  scatter helpers for exact PNA aggregation. On GraIL FB15k-237 v1 with
  `EPOCHS=8`, the 50-negative Hits@10 reached 0.637, essentially matching
  GraIL's 0.642 on the same protocol while still below NBFNet's 0.834.

## [0.9.0] - 2026-07-05

### Added

- `NBFConv::forward_edges`: edge-list form of conditional message passing
  (gather heads, scale by per-edge type representations, scatter-add into
  tails), batched over queries with shared or per-query relation
  representations. The dense per-type-adjacency forward costs
  `O(types * N^2)` memory; the edge-list form makes real relation counts
  (hundreds) feasible. Parity test pins it to the dense path.
- `examples/inductive_link_prediction`: NBFNet-shaped inductive link
  prediction on the GraIL FB15k-237 v1 split, trained on one graph and
  evaluated on a disjoint entity vocabulary, with honest numbers against
  the published references and provenance for every protocol choice.

## [0.8.0] - 2026-07-05

### Added

- `NBFConv`: conditional message passing (Zhu et al., NeurIPS 2021) —
  one generalized Bellman-Ford iteration with edge-type representations as
  forward-time inputs, boundary condition re-added per layer, and
  indicator-initialized pair representations.
- `relgraph::relation_graph`: the graph of relations (Galkin et al., ICLR
  2024) — four interaction-type adjacencies over `2 * num_relations`
  relation nodes, inverses included. With `NBFConv` this is the two-stage
  conditional propagation substrate; oracle tests pin hand-enumerable
  interactions, relation-renaming equivariance, source-conditioning, and
  permutation equivariance.

## [0.7.0] - 2026-07-04

### Added

- `RGCNConv`: relational graph convolution (Schlichtkrull et al., ESWC
  2018) — per-relation transforms over an adjacency stack plus a
  self-loop, with optional basis decomposition for large relation counts.
  The heterogeneous message-passing primitive both the relational-deep-
  learning direction and the geometric-KGFM substrate need.

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-07-04

### Added

- `HGCNConv` derives Burn's `Module` (ball geometry carried as a constant),
  matching `GCNConv`: the hyperbolic layer can now be embedded in trainable
  models.
- `CurvatureError` is re-exported at the crate root alongside the function
  that returns it.
- `#![warn(missing_docs)]`; `PoincareBall::new` documents its f64-to-f32
  narrowing.

### Fixed

- README: version pin and API-surface list caught up with 0.5 (curvature,
  features).

## [0.5.0] - 2026-07-03

### Added

- `features` module: `hom_profile` — homomorphism-count node features
  (walk and closed-walk counts via `graphops`), the interpretable
  expressiveness lift past 1-WL for GCN inputs.

## [0.4.0] - 2026-07-03

### Added

- `curvature` module: Ollivier-Ricci edge curvature
  (`ollivier_ricci_curvatures`) with lazy random-walk `alpha` and entropic
  `W1` via Sinkhorn. Composes `lapl` (transition measures), `graphops`
  (hop distances), and `wass` (transport); these are new dependencies,
  and `ndarray` moved from dev-dependency to dependency.

## [0.3.0] - 2026-07-03

### Changed

- Renamed the crate from `propago` to `ricci`. No API changes; the old
  name remains published at 0.2.0.

## [0.2.0] - 2026-06-10

### Added

- Add Poincare ball geometry formulas with math markup
- Add doctests for PoincareBall and HGCNConv
- Add hyperbolic activation support (Chami et al. 2019 pattern)
- Add accessor methods, fix crate description

### Changed

- Ball-containment invariant tests and make max_norm public
- Expand test coverage to 37 tests
- Harden numerical stability and expand test coverage
- Consolidate to Burn, remove Candle and MLX backends
- Use hyperball as reference impl for Poincare ball tests
- Initial

### Fixed

- Fix operatorname macro for GitHub MathJax
