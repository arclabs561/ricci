//! Minimal Poincare ball smoke example.
//!
//! Run:
//!   cargo run -p ricci --example burn_poincare_smoke

use burn::tensor::backend::Backend;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;
use ricci::PoincareBall;

type B = NdArray<f32>;

fn main() {
    let device = <B as Backend>::Device::default();
    let ball = PoincareBall::new(1.0);

    let x = burn::tensor::Tensor::<B, 2>::from_data(
        TensorData::new(vec![0.10f32, -0.05, 0.02, 0.03, 0.04, -0.01], [2, 3]),
        &device,
    );
    let x = ball.project(x);
    let v = ball.log0(x.clone());
    let x2 = ball.exp0(v);

    let x2v = x2.to_data().to_vec::<f32>().unwrap();
    println!("x2 (first row): {:?}", &x2v[0..3]);
}
