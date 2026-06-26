//! Burn attention-score bias toy with a max-plus graph bias.
//!
//! Run:
//!   cargo run --example attention_bias_tropical
//!
//! This is a proof sketch, not public API: compose ordinary score biases with a
//! tropical/max-plus graph bias before extracting a reusable abstraction.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

const NEG_INF: f32 = -1.0e9;

fn tensor2<const N: usize, const M: usize>(
    rows: [[f32; M]; N],
    device: &<B as Backend>::Device,
) -> Tensor<B, 2> {
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    Tensor::from_data(TensorData::new(flat, [N, M]), device)
}

fn max_plus_square<const N: usize>(a: [[f32; N]; N]) -> [[f32; N]; N] {
    let mut out = [[NEG_INF; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut best = NEG_INF;
            for (k, &left) in a[i].iter().enumerate() {
                best = best.max(left + a[k][j]);
            }
            out[i][j] = best;
        }
    }
    out
}

fn row_argmax<const N: usize>(m: &[f32]) -> [usize; N] {
    let mut out = [0; N];
    for i in 0..N {
        let row = &m[i * N..(i + 1) * N];
        out[i] = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
    }
    out
}

fn main() {
    let device = <B as Backend>::Device::default();

    let q = tensor2(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.4, 0.5, 0.1],
        ],
        &device,
    );
    let k = tensor2(
        [
            [0.9, 0.1, 0.0],
            [0.0, 0.9, 0.1],
            [0.1, 0.0, 0.9],
            [0.2, 0.7, 0.1],
        ],
        &device,
    );

    let scores = q.matmul(k.transpose());
    let base_argmax = row_argmax::<4>(&scores.clone().to_data().to_vec::<f32>().unwrap());

    let positional_bias = tensor2(
        [
            [0.0, -0.1, -0.2, -0.3],
            [-0.1, 0.0, -0.1, -0.2],
            [-0.2, -0.1, 0.0, -0.1],
            [-0.3, -0.2, -0.1, 0.0],
        ],
        &device,
    );

    // Weighted graph in max-plus form. Missing edges are -infinity, path scores
    // add along hops, and the max over intermediate nodes chooses the best
    // two-hop explanation.
    let graph = [
        [0.0, 0.9, NEG_INF, NEG_INF],
        [NEG_INF, 0.0, 1.4, 0.7],
        [0.8, NEG_INF, 0.0, NEG_INF],
        [NEG_INF, NEG_INF, 0.4, 0.0],
    ];
    let two_hop = max_plus_square(graph);
    let tropical_bias = tensor2(two_hop, &device) * 0.5;

    let biased = scores + positional_bias + tropical_bias;
    let biased_vec = biased.to_data().to_vec::<f32>().unwrap();
    let biased_argmax = row_argmax::<4>(&biased_vec);

    println!("base row argmax:   {base_argmax:?}");
    println!("biased row argmax: {biased_argmax:?}");
    println!("biased scores row 0: {:?}", &biased_vec[0..4]);

    assert_ne!(base_argmax, biased_argmax);
    assert_eq!(biased_argmax[0], 2);
}
