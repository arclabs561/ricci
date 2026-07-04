//! Homomorphism-count node features: provable expressiveness beyond 1-WL.
//!
//! Message passing distinguishes nodes only up to color refinement, which by
//! Dell, Grohe & Rattan (ICALP 2018) equals homomorphism counts from *trees*
//! — so a GCN provably cannot count cycles. Injecting rooted closed-walk
//! counts (cycle homomorphism counts, treewidth 2) as input features lifts
//! that ceiling interpretably (Barceló et al., NeurIPS 2021; Jin et al.,
//! ICML 2024): the feature IS a named graph quantity, not a learned blob.
//!
//! [`hom_profile`] assembles the two tractable rooted families from
//! `graphops` — walk counts (path homomorphisms) and closed-walk counts
//! (cycle homomorphisms at lengths 3 and 4) — as a feature matrix ready to
//! feed (or concatenate into) [`crate::GCNConv`] inputs. Counts grow with
//! graph size; consider `ln(1 + x)` normalization before training.

use graphops::{closed_walk_counts, walk_counts, Graph};
use ndarray::Array2;

/// Adjacency view for `graphops` over a dense matrix (edge = entry > 0).
struct Adj<'a>(&'a Array2<f64>);

impl Graph for Adj<'_> {
    fn node_count(&self) -> usize {
        self.0.nrows()
    }
    fn neighbors(&self, node: usize) -> Vec<usize> {
        (0..self.0.ncols())
            .filter(|&j| self.0[[node, j]] > 0.0)
            .collect()
    }
}

/// Per-node hom-count features: columns are walk counts of lengths
/// `1..=max_walk` followed by closed-walk counts of lengths 3 and 4.
///
/// Shape: `[n, max_walk + 2]`. The adjacency must be square; entries `> 0`
/// are edges (weights do not enter the counts).
pub fn hom_profile(adj: &Array2<f64>, max_walk: usize) -> Array2<f32> {
    let g = Adj(adj);
    let n = g.node_count();
    let mut out = Array2::zeros((n, max_walk + 2));
    for len in 1..=max_walk {
        let w = walk_counts(&g, len);
        for v in 0..n {
            out[[v, len - 1]] = w[v] as f32;
        }
    }
    for (col, len) in [(max_walk, 3), (max_walk + 1, 4)] {
        let c = closed_walk_counts(&g, len);
        for v in 0..n {
            out[[v, col]] = c[v] as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Hand-computed on the triangle: degree 2, two-walks 4, closed
    /// 3-walks 2, closed 4-walks 6 (trace A^4 = 2^4 + (−1)^4 + (−1)^4 = 18,
    /// so 6 per node).
    #[test]
    fn triangle_profile_matches_hand_computation() {
        let adj = array![[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]];
        let f = hom_profile(&adj, 2);
        for v in 0..3 {
            assert_eq!(f[[v, 0]], 2.0, "degree");
            assert_eq!(f[[v, 1]], 4.0, "walks-2");
            assert_eq!(f[[v, 2]], 2.0, "closed-3");
            assert_eq!(f[[v, 3]], 6.0, "closed-4");
        }
    }

    /// The classic 1-WL blind spot: a 6-cycle and two triangles have equal
    /// degree sequences everywhere (2-regular), but closed-3-walk counts
    /// separate them — the feature carries what message passing cannot.
    #[test]
    fn cycle_counts_separate_wl_indistinguishable_graphs() {
        let c6 = {
            let mut a = Array2::zeros((6, 6));
            for v in 0..6 {
                a[[v, (v + 1) % 6]] = 1.0;
                a[[(v + 1) % 6, v]] = 1.0;
            }
            a
        };
        let two_triangles = {
            let mut a = Array2::zeros((6, 6));
            for &(i, j) in &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] {
                a[[i, j]] = 1.0;
                a[[j, i]] = 1.0;
            }
            a
        };
        let f6 = hom_profile(&c6, 2);
        let ft = hom_profile(&two_triangles, 2);
        for v in 0..6 {
            // Identical walk profiles (both 2-regular)...
            assert_eq!(f6[[v, 0]], ft[[v, 0]]);
            assert_eq!(f6[[v, 1]], ft[[v, 1]]);
            // ...separated by triangle counts.
            assert_eq!(f6[[v, 2]], 0.0);
            assert_eq!(ft[[v, 2]], 2.0);
        }
    }
}
