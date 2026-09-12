//! Minimal Poincare ball smoke example.
//!
//! Run:
//!   cargo run -p ricci --example burn_poincare_smoke

use burn::tensor::Device;
use burn::tensor::TensorData;
use ricci::PoincareBall;

fn main() {
    let device = Device::flex();
    let ball = PoincareBall::new(1.0);

    let x = burn::tensor::Tensor::<2>::from_data(
        TensorData::new(vec![0.10f32, -0.05, 0.02, 0.03, 0.04, -0.01], [2, 3]),
        &device,
    );
    let x = ball.project(x);
    let v = ball.log0(x.clone());
    let x2 = ball.exp0(v);

    let x2v = x2.to_data().try_to_vec::<f32>().unwrap();
    println!("x2 (first row): {:?}", &x2v[0..3]);
}
