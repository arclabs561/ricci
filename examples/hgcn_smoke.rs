//! Minimal HGCN smoke example.
//!
//! Run:
//!   cargo run -p ricci --example hgcn_smoke

use burn::tensor::backend::Backend;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;
use ricci::HGCNConv;

type B = NdArray<f32>;

fn main() {
    let device = <B as Backend>::Device::default();

    let n = 6usize;
    let d = 4usize;

    // Small random input (well inside the ball).
    let x_data: Vec<f32> = (0..n * d)
        .map(|i| ((i as f32) * 0.017).sin() * 0.1)
        .collect();
    let x = burn::tensor::Tensor::<B, 2>::from_data(TensorData::new(x_data, [n, d]), &device);

    // Identity adjacency.
    let mut adj_v = vec![0.0f32; n * n];
    for i in 0..n {
        adj_v[i * n + i] = 1.0;
    }
    let adj = burn::tensor::Tensor::<B, 2>::from_data(TensorData::new(adj_v, [n, n]), &device);

    let layer = HGCNConv::<B>::init(d, 1.0, &device);
    let y = layer.forward(x.clone(), adj.clone());
    let [yn, yd] = y.dims();
    println!("y shape: [{yn}, {yd}]");

    let p =
        burn::tensor::Tensor::<B, 2>::from_data(TensorData::new(vec![0.0f32; d], [1, d]), &device);
    let y2 = layer.forward_with_basepoint(x, adj, p);
    let [yn2, yd2] = y2.dims();
    println!("y(basepoint) shape: [{yn2}, {yd2}]");
}
