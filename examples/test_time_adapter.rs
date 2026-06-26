//! Fit a tiny task-local adapter on frozen Burn features.
//!
//! Run:
//!   cargo run --example test_time_adapter
//!
//! This is an `adaptfit` proof sketch, not public API: keep the representation
//! fixed, fit a small adapter on a few support pairs, then evaluate a held-out
//! pair from the same local task.

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;

type Base = NdArray<f32>;

#[derive(Module, Debug)]
struct Adapter<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> Adapter<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(2, 2).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}

fn tensor2<B: Backend, const N: usize>(rows: [[f32; 2]; N], device: &B::Device) -> Tensor<B, 2> {
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    Tensor::from_data(TensorData::new(flat, [N, 2]), device)
}

fn mse<B: Backend>(predicted: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    (predicted - target).powf_scalar(2.0).mean()
}

fn scalar_loss<B: Backend>(loss: Tensor<B, 1>) -> f32 {
    loss.into_data().to_vec::<f32>().unwrap()[0]
}

fn fit<B: AutodiffBackend>(device: B::Device) {
    let support_x = tensor2::<B, 4>([[2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [-2.0, 1.0]], &device);
    let support_y = tensor2::<B, 4>([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [-0.5, 0.25]], &device);
    let heldout_x = tensor2::<B, 2>([[1.0, -2.0], [-2.0, -2.0]], &device);
    let heldout_y = tensor2::<B, 2>([[0.25, -0.5], [-0.5, -0.5]], &device);

    let baseline_support = scalar_loss(mse(support_x.clone(), support_y.clone()));
    let baseline_heldout = scalar_loss(mse(heldout_x.clone(), heldout_y.clone()));

    let mut adapter = Adapter::<B>::init(&device);
    let mut optim = AdamConfig::new().init();

    for _ in 0..300 {
        let predicted = adapter.forward(support_x.clone());
        let loss = mse(predicted, support_y.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &adapter);
        adapter = optim.step(0.03, adapter, grads);
    }

    let adapted_support = scalar_loss(mse(adapter.forward(support_x), support_y));
    let adapted_heldout = scalar_loss(mse(adapter.forward(heldout_x), heldout_y));

    println!("support mse: baseline {baseline_support:.4} -> adapted {adapted_support:.4}");
    println!("heldout mse: baseline {baseline_heldout:.4} -> adapted {adapted_heldout:.4}");

    assert!(adapted_support < baseline_support * 0.05);
    assert!(adapted_heldout < baseline_heldout * 0.05);
}

fn main() {
    fit::<Autodiff<Base>>(Default::default());
}
