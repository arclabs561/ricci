# Changelog

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
