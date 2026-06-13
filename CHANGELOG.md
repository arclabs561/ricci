# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
