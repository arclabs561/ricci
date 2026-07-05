//! The graph of relations: four interaction-type adjacencies over relation
//! nodes (Galkin et al., ICLR 2024).
//!
//! Two relations interact when they share an entity in a given role: the
//! tail of one being the head of another is a tail-to-head interaction,
//! and likewise head-to-head, head-to-tail, tail-to-tail. These four
//! structure-only interaction types are invariant to what the relations
//! are called, which is what makes representations computed over this
//! graph transfer across relation vocabularies. Inverse relations are
//! nodes too (inverse of `r` is `r + num_relations`), so the graph has
//! `2 * num_relations` nodes.
//!
//! The adjacencies come back unnormalized (an edge is present or not);
//! normalization is the caller's choice, as with every conv in this crate.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// Index of the tail-to-head interaction adjacency in the returned stack.
pub const T2H: usize = 0;
/// Index of the head-to-head interaction adjacency.
pub const H2H: usize = 1;
/// Index of the head-to-tail interaction adjacency.
pub const H2T: usize = 2;
/// Index of the tail-to-tail interaction adjacency.
pub const T2T: usize = 3;

/// Build the four interaction adjacencies over `2 * num_relations`
/// relation nodes from `(head, relation, tail)` triples.
///
/// Relation node `r < num_relations` is the original direction; node
/// `r + num_relations` is its inverse (head and tail roles swapped).
/// Entry `[a, b]` of the `T2H` matrix is `1.0` iff some entity is a tail
/// of relation `a` and a head of relation `b`; the other three matrices
/// follow the same pattern for their role pairs. Self-interactions are
/// kept (a relation whose tail set meets its own head set gets a T2H
/// self-loop): they carry real signal (e.g. composable relations).
///
/// Out-of-range relation ids are ignored; entity ids only need to be
/// consistent, not dense.
pub fn relation_graph<B: Backend>(
    triples: &[(usize, usize, usize)],
    num_relations: usize,
    device: &B::Device,
) -> [Tensor<B, 2>; 4] {
    use std::collections::HashMap;
    let n = 2 * num_relations;
    // Role sets per relation node: which entities appear as head / tail.
    let mut heads: HashMap<usize, Vec<usize>> = HashMap::new(); // entity -> rel nodes with entity as head
    let mut tails: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(h, r, t) in triples {
        if r >= num_relations {
            continue;
        }
        let inv = r + num_relations;
        // Original direction: h is head of r, t is tail of r.
        heads.entry(h).or_default().push(r);
        tails.entry(t).or_default().push(r);
        // Inverse direction: t is head of r_inv, h is tail of r_inv.
        heads.entry(t).or_default().push(inv);
        tails.entry(h).or_default().push(inv);
    }
    let mut mats = [
        vec![0.0f32; n * n], // t2h
        vec![0.0f32; n * n], // h2h
        vec![0.0f32; n * n], // h2t
        vec![0.0f32; n * n], // t2t
    ];
    let mark = |m: &mut Vec<f32>, from: &Vec<usize>, to: &Vec<usize>| {
        for &a in from {
            for &b in to {
                m[a * n + b] = 1.0;
            }
        }
    };
    // For every entity, connect the relation nodes it links by role pair.
    let entities: std::collections::BTreeSet<usize> =
        heads.keys().chain(tails.keys()).copied().collect();
    let empty: Vec<usize> = Vec::new();
    for e in entities {
        let h = heads.get(&e).unwrap_or(&empty);
        let t = tails.get(&e).unwrap_or(&empty);
        mark(&mut mats[T2H], t, h);
        mark(&mut mats[H2H], h, h);
        mark(&mut mats[H2T], h, t);
        mark(&mut mats[T2T], t, t);
    }
    let [a, b, c, d] = mats;
    [
        Tensor::from_data(TensorData::new(a, [n, n]), device),
        Tensor::from_data(TensorData::new(b, [n, n]), device),
        Tensor::from_data(TensorData::new(c, [n, n]), device),
        Tensor::from_data(TensorData::new(d, [n, n]), device),
    ]
}

/// Diagnostic summary of a relation graph: the cheap numbers to look at
/// before trusting anything trained over it.
///
/// The load-bearing one is `isolated_nodes`: a relation node with no
/// interaction edges receives no conditioning signal during message
/// passing, so its representation stays at the boundary value and every
/// downstream score involving it is degenerate. Sparse interaction
/// structure is the known failure mode of relation-graph transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationGraphStats {
    /// Edge count per interaction type (`T2H`, `H2H`, `H2T`, `T2T`).
    pub edges_per_type: [usize; 4],
    /// Relation nodes with no interaction edge of any type (in or out).
    pub isolated_nodes: usize,
    /// Self-interaction count per type (e.g. a T2H self-loop marks a
    /// self-composable relation).
    pub self_loops: [usize; 4],
    /// Total relation nodes (`2 * num_relations`).
    pub nodes: usize,
}

/// Compute [`RelationGraphStats`] for a stack built by [`relation_graph`].
pub fn interaction_stats<B: Backend>(adjs: &[Tensor<B, 2>; 4]) -> RelationGraphStats {
    let n = adjs[0].dims()[0];
    let mut edges = [0usize; 4];
    let mut loops = [0usize; 4];
    let mut touched = vec![false; n];
    for (k, adj) in adjs.iter().enumerate() {
        let v: Vec<f32> = adj.clone().into_data().to_vec().unwrap();
        for i in 0..n {
            for j in 0..n {
                if v[i * n + j] != 0.0 {
                    edges[k] += 1;
                    touched[i] = true;
                    touched[j] = true;
                    if i == j {
                        loops[k] += 1;
                    }
                }
            }
        }
    }
    RelationGraphStats {
        edges_per_type: edges,
        isolated_nodes: touched.iter().filter(|&&t| !t).count(),
        self_loops: loops,
        nodes: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn dev() -> <B as Backend>::Device {
        <B as Backend>::Device::default()
    }

    fn at(m: &Tensor<B, 2>, i: usize, j: usize) -> f32 {
        let n = m.dims()[1];
        let v: Vec<f32> = m.clone().into_data().to_vec().unwrap();
        v[i * n + j]
    }

    /// Two chained triples: (0, r0, 1), (1, r1, 2). Relation nodes:
    /// r0=0, r1=1, r0_inv=2, r1_inv=3. Entity 1 is the tail of r0 and the
    /// head of r1, so T2H[r0, r1] = 1; every other cell is checkable the
    /// same way by enumerating entity 1's roles (tail of r0, head of r1,
    /// head of r0_inv, tail of r1_inv).
    #[test]
    fn chained_triples_interactions_by_hand() {
        let triples = [(0usize, 0usize, 1usize), (1, 1, 2)];
        let [t2h, h2h, h2t, t2t] = relation_graph::<B>(&triples, 2, &dev());
        // Entity 1: tails = {r0, r1_inv=3}, heads = {r1, r0_inv=2}.
        assert_eq!(at(&t2h, 0, 1), 1.0, "tail of r0 is head of r1");
        assert_eq!(at(&t2h, 0, 2), 1.0, "tail of r0 is head of r0_inv");
        assert_eq!(at(&h2h, 1, 2), 1.0, "r1 and r0_inv share head 1");
        assert_eq!(at(&t2t, 0, 3), 1.0, "r0 and r1_inv share tail 1");
        assert_eq!(at(&h2t, 1, 0), 1.0, "head of r1 is tail of r0");
        // No interaction between r0 and r1 via heads: entity 0 is head of
        // r0 (and tail of r0_inv) only.
        assert_eq!(at(&h2h, 0, 1), 0.0);
        // Roles are direction-aware: r1's tail set {2} never meets r0.
        assert_eq!(at(&t2h, 1, 0), 0.0);
    }

    /// Diagnostics: the chained-triples graph has no isolated node; adding
    /// a relation that never occurs makes its two nodes isolated.
    #[test]
    fn stats_flag_isolated_relations() {
        let triples = [(0usize, 0usize, 1usize), (1, 1, 2)];
        let g = relation_graph::<B>(&triples, 2, &dev());
        let s = interaction_stats(&g);
        assert_eq!(s.nodes, 4);
        assert_eq!(s.isolated_nodes, 0);
        assert!(s.edges_per_type.iter().sum::<usize>() > 0);

        // Declare 3 relations but only use 2: r2 and r2_inv are isolated.
        let g = relation_graph::<B>(&triples, 3, &dev());
        let s = interaction_stats(&g);
        assert_eq!(s.nodes, 6);
        assert_eq!(s.isolated_nodes, 2, "unused relation + its inverse");
    }

    /// The construction is vocabulary-free: renaming relation ids permutes
    /// rows/columns consistently and changes nothing else.
    #[test]
    fn relation_renaming_permutes_consistently() {
        let orig = [(0usize, 0usize, 1usize), (1, 1, 2), (2, 0, 3)];
        let renamed: Vec<_> = orig
            .iter()
            .map(|&(h, r, t)| (h, 1 - r, t)) // swap relation ids 0 <-> 1
            .collect();
        let a = relation_graph::<B>(&orig, 2, &dev());
        let b = relation_graph::<B>(&renamed, 2, &dev());
        // Permutation on 4 nodes: 0<->1 and inverses 2<->3.
        let p = [1usize, 0, 3, 2];
        for k in 0..4 {
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(
                        at(&a[k], i, j),
                        at(&b[k], p[i], p[j]),
                        "matrix {k} cell ({i},{j}) must permute"
                    );
                }
            }
        }
    }
}
