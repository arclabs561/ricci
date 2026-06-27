//! Integration test for the public `GCNConv` forward pass.
//!
//! Exercises the real layer (`GCNConv::init` + `forward`) end to end on a small
//! fixed graph, asserting both the output shape and concrete numerical
//! invariants of GCN message passing. No reimplementation of the matmul: every
//! assertion is on the output of the crate's own `forward`.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use propago::GCNConv;

type B = NdArray<f32>;

fn dev() -> <B as Backend>::Device {
    <B as Backend>::Device::default()
}

/// L2 norm of a flat row slice.
fn norm(row: &[f32]) -> f32 {
    row.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Invariant: a node with no incoming edges aggregates nothing.
///
/// `GCNConv::forward` computes `adj @ linear(x)`, so output row `i` is
/// `sum_j adj[i,j] * linear(x)[j]`. If node `i` has an all-zero adjacency row
/// (no self-loop, no neighbours) its output must be exactly the zero vector,
/// independent of the (bias-bearing) linear transform. A node with edges
/// aggregates a generically non-zero combination of transformed features.
///
/// This is the message-passing semantics of a GCN, not an algebraic identity of
/// `forward`: it is what makes the adjacency matrix mean "who talks to whom".
#[test]
fn gcn_isolated_node_aggregates_nothing() {
    let n = 4;
    let d_in = 5;
    let d_out = 3;
    let layer = GCNConv::<B>::init(d_in, d_out, &dev());

    // Distinct per-node features so a connected node's aggregate is non-trivial.
    #[rustfmt::skip]
    let x_v = vec![
        0.10f32, -0.20, 0.30, -0.40, 0.50, // node 0 (isolated)
        0.05,     0.15, -0.25, 0.35, -0.45, // node 1
        -0.12,    0.22, 0.32, -0.42, 0.11,  // node 2
        0.33,    -0.13, 0.23, -0.03, 0.18,  // node 3
    ];
    let x = Tensor::<B, 2>::from_data(TensorData::new(x_v, [n, d_in]), &dev());

    // Node 0 is isolated (row of zeros). Nodes 1,2,3 form a connected component
    // with self-loops and mutual edges.
    #[rustfmt::skip]
    let adj_v = vec![
        0.0f32, 0.0, 0.0, 0.0, // node 0: no incoming edges
        0.0,    1.0, 1.0, 0.0, // node 1: self + node 2
        0.0,    1.0, 1.0, 1.0, // node 2: self + nodes 1,3
        0.0,    0.0, 1.0, 1.0, // node 3: self + node 2
    ];
    let adj = Tensor::<B, 2>::from_data(TensorData::new(adj_v, [n, n]), &dev());

    let y = layer.forward(x, adj);

    // Shape: [n, d_out].
    assert_eq!(y.dims(), [n, d_out], "output shape");

    let y_v = y.to_data().to_vec::<f32>().unwrap();
    assert!(
        y_v.iter().all(|v| v.is_finite()),
        "forward produced non-finite values"
    );

    let row0 = &y_v[0..d_out];
    let row2 = &y_v[2 * d_out..3 * d_out];

    // Isolated node: output is exactly zero (no aggregation).
    assert!(
        norm(row0) < 1e-6,
        "isolated node 0 should aggregate to zero, got norm={}",
        norm(row0)
    );
    // Connected node: output is non-zero (it aggregated real features).
    assert!(
        norm(row2) > 1e-6,
        "connected node 2 should aggregate a non-zero output, got norm={}",
        norm(row2)
    );
}

/// Invariant: `forward` is linear in the adjacency matrix.
///
/// Because `linear(x)` does not depend on `adj`, `forward(x, A + B)` must equal
/// `forward(x, A) + forward(x, B)` up to float rounding. This is an exact
/// algebraic property that holds for any (randomly initialised) weights, so it
/// pins the aggregation semantics without depending on the init distribution.
/// It calls the real `forward` three times and compares the tensors.
#[test]
fn gcn_forward_linear_in_adjacency() {
    let n = 3;
    let d = 4;
    let layer = GCNConv::<B>::init(d, d, &dev());

    #[rustfmt::skip]
    let x_v = vec![
        0.1f32, -0.2, 0.3, -0.4,
        0.5,     0.15, -0.25, 0.35,
        -0.12,   0.22, 0.32, -0.42,
    ];
    let x = Tensor::<B, 2>::from_data(TensorData::new(x_v, [n, d]), &dev());

    #[rustfmt::skip]
    let a_v = vec![
        1.0f32, 0.5, 0.0,
        0.5,    1.0, 0.5,
        0.0,    0.5, 1.0,
    ];
    #[rustfmt::skip]
    let b_v = vec![
        0.2f32, 0.0, 0.7,
        0.0,    0.3, 0.0,
        0.7,    0.0, 0.4,
    ];
    let a = Tensor::<B, 2>::from_data(TensorData::new(a_v, [n, n]), &dev());
    let b = Tensor::<B, 2>::from_data(TensorData::new(b_v, [n, n]), &dev());

    let y_sum = layer.forward(x.clone(), a.clone() + b.clone());
    let y_a = layer.forward(x.clone(), a);
    let y_b = layer.forward(x, b);

    assert_eq!(y_sum.dims(), [n, d], "output shape");

    let lhs = y_sum.to_data().to_vec::<f32>().unwrap();
    let ya = y_a.to_data().to_vec::<f32>().unwrap();
    let yb = y_b.to_data().to_vec::<f32>().unwrap();

    let l1: f32 = lhs
        .iter()
        .zip(ya.iter().zip(&yb))
        .map(|(s, (a, b))| (s - (a + b)).abs())
        .sum();
    assert!(
        l1 < 1e-5,
        "forward should be linear in adjacency: l1(forward(A+B), forward(A)+forward(B))={l1}"
    );
}
