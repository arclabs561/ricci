//! Ollivier-Ricci edge curvature for graphs.
//!
//! The crate's namesake: the discrete Ricci curvature of an edge `(u, v)`
//! compares how close the neighborhoods of `u` and `v` are, relative to how
//! far apart the nodes themselves are:
//!
//! ```text
//! kappa(u, v) = 1 - W1(mu_u, mu_v) / d(u, v)
//! ```
//!
//! where `mu_x` is the lazy random-walk measure of `x` (`alpha` mass staying
//! at `x`, the rest spread over its neighbors), `W1` is the Wasserstein-1
//! distance under the hop metric, and `d` is the hop distance (`1` for an
//! edge). Positive curvature means neighborhoods overlap (clique-like), zero
//! is grid/cycle-like, negative means neighborhoods pull apart (tree- or
//! bridge-like). Negatively curved edges are the bottlenecks implicated in
//! GNN oversquashing, which makes per-edge curvature the primitive behind
//! curvature-based rewiring.
//!
//! The implementation composes the ecosystem's own primitives: `lapl` builds
//! the random-walk measures, `graphops` supplies hop distances, and `wass`
//! computes an entropically regularized `W1` (log-domain Sinkhorn), so the
//! returned values carry a small smoothing bias controlled by
//! [`CurvatureConfig::reg`].
//!
//! # Example
//!
//! ```
//! use ndarray::array;
//! use ricci::curvature::{ollivier_ricci_curvatures, CurvatureConfig};
//!
//! // Triangle: every edge has curvature 1/2 (alpha = 0).
//! let adj = array![[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];
//! let kappas = ollivier_ricci_curvatures(&adj, &CurvatureConfig::default()).unwrap();
//! assert_eq!(kappas.len(), 3);
//! for e in &kappas {
//!     assert!((e.kappa - 0.5).abs() < 0.05);
//! }
//! ```

use graphops::{bfs_distances, GraphRef};
use ndarray::{Array1, Array2};

/// Configuration for [`ollivier_ricci_curvatures`].
#[derive(Debug, Clone, Copy)]
pub struct CurvatureConfig {
    /// Laziness of the random walk: `mu_x` keeps `alpha` mass at `x` and
    /// spreads `1 - alpha` over its neighbors. `0.0` is Ollivier's original
    /// definition; `0.5` is the other common choice. Must be in `[0, 1]`.
    pub alpha: f64,
    /// Entropic regularization for the Sinkhorn `W1`. Smaller is closer to
    /// the exact transport distance at the cost of more iterations.
    pub reg: f32,
    /// Sinkhorn iteration cap.
    pub max_iter: usize,
}

impl Default for CurvatureConfig {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            reg: 0.01,
            max_iter: 500,
        }
    }
}

/// Curvature of one undirected edge (`u < v`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeCurvature {
    /// Smaller endpoint.
    pub u: usize,
    /// Larger endpoint.
    pub v: usize,
    /// Ollivier-Ricci curvature `1 - W1(mu_u, mu_v)`.
    pub kappa: f32,
}

/// Input problems [`ollivier_ricci_curvatures`] rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurvatureError {
    /// The adjacency matrix is not square.
    NotSquare,
    /// The adjacency matrix is not symmetric; curvature is defined here for
    /// undirected graphs.
    NotSymmetric,
    /// An edge weight is negative or non-finite.
    InvalidWeight,
    /// `alpha` is outside `[0, 1]`.
    InvalidAlpha,
}

impl std::fmt::Display for CurvatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "adjacency matrix must be square"),
            Self::NotSymmetric => write!(f, "adjacency matrix must be symmetric (undirected)"),
            Self::InvalidWeight => write!(f, "edge weights must be finite and non-negative"),
            Self::InvalidAlpha => write!(f, "alpha must be in [0, 1]"),
        }
    }
}

impl std::error::Error for CurvatureError {}

/// Neighbor lists in the shape `graphops` BFS wants.
struct NeighborLists(Vec<Vec<usize>>);

impl GraphRef for NeighborLists {
    fn node_count(&self) -> usize {
        self.0.len()
    }
    fn neighbors_ref(&self, node: usize) -> &[usize] {
        &self.0[node]
    }
}

/// Ollivier-Ricci curvature of every undirected edge of `adj`.
///
/// `adj` is a symmetric non-negative adjacency matrix; `adj[[i, j]] > 0.0`
/// is an edge, and weights shape the random-walk measures (via `lapl`'s
/// degree-normalized transition matrix) but not the ground metric, which is
/// unweighted hop distance. Returns one entry per edge `u < v`.
///
/// Cost is roughly one BFS per node plus one Sinkhorn solve per edge over
/// the union of the two endpoints' neighborhoods, so dense hubs cost the
/// most. Values carry the entropic smoothing bias of
/// [`CurvatureConfig::reg`]; for rankings (which edges are most negative)
/// the default is plenty.
pub fn ollivier_ricci_curvatures(
    adj: &Array2<f64>,
    config: &CurvatureConfig,
) -> Result<Vec<EdgeCurvature>, CurvatureError> {
    let n = adj.nrows();
    if adj.ncols() != n {
        return Err(CurvatureError::NotSquare);
    }
    if !(0.0..=1.0).contains(&config.alpha) {
        return Err(CurvatureError::InvalidAlpha);
    }
    for i in 0..n {
        for j in 0..n {
            let w = adj[[i, j]];
            if !w.is_finite() || w < 0.0 {
                return Err(CurvatureError::InvalidWeight);
            }
            if (w - adj[[j, i]]).abs() > 1e-9 * w.abs().max(1.0) {
                return Err(CurvatureError::NotSymmetric);
            }
        }
    }

    // Random-walk measures (lapl), lazified per config.
    let transition = lapl::transition_matrix(adj);

    // Unweighted hop metric (graphops BFS), one row per node.
    let neighbors = NeighborLists(
        (0..n)
            .map(|i| (0..n).filter(|&j| adj[[i, j]] > 0.0).collect())
            .collect(),
    );
    let hops: Vec<Vec<Option<usize>>> = (0..n).map(|i| bfs_distances(&neighbors, i)).collect();

    let mut out = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            if adj[[u, v]] <= 0.0 {
                continue;
            }

            let mu_u = lazy_measure(&transition, u, config.alpha);
            let mu_v = lazy_measure(&transition, v, config.alpha);

            // Restrict the transport to the union support: every node there
            // is within two hops of the edge, so distances are finite.
            let support: Vec<usize> = (0..n).filter(|&i| mu_u[i] > 0.0 || mu_v[i] > 0.0).collect();
            let m = support.len();
            let a = Array1::from_iter(support.iter().map(|&i| mu_u[i] as f32));
            let b = Array1::from_iter(support.iter().map(|&i| mu_v[i] as f32));
            let mut cost = Array2::zeros((m, m));
            for (si, &i) in support.iter().enumerate() {
                for (sj, &j) in support.iter().enumerate() {
                    // The union support is connected through the edge itself,
                    // so a missing BFS distance cannot happen; 0 mass would
                    // make it irrelevant anyway.
                    cost[[si, sj]] = hops[i][j].unwrap_or(usize::MAX) as f32;
                }
            }

            let (_, w1) = wass::sinkhorn_log(&a, &b, &cost, config.reg, config.max_iter);
            // Hop distance between edge endpoints is exactly 1.
            out.push(EdgeCurvature {
                u,
                v,
                kappa: 1.0 - w1,
            });
        }
    }
    Ok(out)
}

/// `alpha * delta_x + (1 - alpha) * transition_row(x)`.
fn lazy_measure(transition: &Array2<f64>, x: usize, alpha: f64) -> Array1<f64> {
    let mut mu = transition.row(x).to_owned() * (1.0 - alpha);
    mu[x] += alpha;
    mu
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn kappa_of(kappas: &[EdgeCurvature], u: usize, v: usize) -> f32 {
        kappas
            .iter()
            .find(|e| (e.u, e.v) == (u.min(v), u.max(v)))
            .expect("edge present")
            .kappa
    }

    /// Hand-derived: triangle edge, alpha = 0. mu_0 = (delta_1 + delta_2)/2,
    /// mu_1 = (delta_0 + delta_2)/2; half the mass overlaps at 2 (cost 0),
    /// half moves 1 -> 0 (cost 1/2). W1 = 1/2, kappa = 1/2.
    #[test]
    fn triangle_edges_curve_positively() {
        let adj = array![[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];
        let kappas = ollivier_ricci_curvatures(&adj, &CurvatureConfig::default()).unwrap();
        assert_eq!(kappas.len(), 3);
        for e in &kappas {
            assert!((e.kappa - 0.5).abs() < 0.05, "kappa {}", e.kappa);
        }
    }

    /// Hand-derived: 4-cycle edge (0,1). mu_0 = (delta_1 + delta_3)/2,
    /// mu_1 = (delta_0 + delta_2)/2; each half moves distance 1. W1 = 1,
    /// kappa = 0.
    #[test]
    fn four_cycle_edges_are_flat() {
        let adj = array![
            [0., 1., 0., 1.],
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
            [1., 0., 1., 0.]
        ];
        let kappas = ollivier_ricci_curvatures(&adj, &CurvatureConfig::default()).unwrap();
        assert_eq!(kappas.len(), 4);
        for e in &kappas {
            assert!(e.kappa.abs() < 0.05, "kappa {}", e.kappa);
        }
    }

    /// Hand-derived: double star, hubs 1-2, leaves {0, 4} on 1 and {3, 5}
    /// on 2. For the bridge: matching thirds 2->3 (cost 1), 0->1 (cost 1),
    /// 4->5 (cost 3) gives W1 = 5/3, kappa = -2/3. The bridge is the most
    /// negative edge: the oversquashing bottleneck signature.
    #[test]
    fn double_star_bridge_curves_negatively() {
        let mut adj = Array2::zeros((6, 6));
        for &(i, j) in &[(1usize, 2usize), (1, 0), (1, 4), (2, 3), (2, 5)] {
            adj[[i, j]] = 1.0;
            adj[[j, i]] = 1.0;
        }
        let kappas = ollivier_ricci_curvatures(&adj, &CurvatureConfig::default()).unwrap();
        let bridge = kappa_of(&kappas, 1, 2);
        assert!((bridge - (-2.0 / 3.0)).abs() < 0.05, "bridge {bridge}");
        for e in &kappas {
            if (e.u, e.v) != (1, 2) {
                assert!(e.kappa > bridge, "bridge should be the most negative");
            }
        }
    }

    /// alpha = 1 degenerates every measure to its own node: W1 = d = 1, so
    /// every edge has kappa = 0 regardless of structure.
    #[test]
    fn full_laziness_zeroes_curvature() {
        let adj = array![[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];
        let cfg = CurvatureConfig {
            alpha: 1.0,
            ..CurvatureConfig::default()
        };
        for e in &ollivier_ricci_curvatures(&adj, &cfg).unwrap() {
            assert!(e.kappa.abs() < 0.05, "kappa {}", e.kappa);
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        let rect = Array2::<f64>::zeros((2, 3));
        assert_eq!(
            ollivier_ricci_curvatures(&rect, &CurvatureConfig::default()),
            Err(CurvatureError::NotSquare)
        );

        let asym = array![[0., 1.], [0., 0.]];
        assert_eq!(
            ollivier_ricci_curvatures(&asym, &CurvatureConfig::default()),
            Err(CurvatureError::NotSymmetric)
        );

        let neg = array![[0., -1.], [-1., 0.]];
        assert_eq!(
            ollivier_ricci_curvatures(&neg, &CurvatureConfig::default()),
            Err(CurvatureError::InvalidWeight)
        );

        let adj = array![[0., 1.], [1., 0.]];
        let bad_alpha = CurvatureConfig {
            alpha: 1.5,
            ..CurvatureConfig::default()
        };
        assert_eq!(
            ollivier_ricci_curvatures(&adj, &bad_alpha),
            Err(CurvatureError::InvalidAlpha)
        );
    }
}
