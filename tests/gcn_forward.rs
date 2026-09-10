//! Numerical integration tests for the public `GCNConv` forward paths.

#[cfg(feature = "metal")]
use burn::backend::Autodiff;
#[cfg(feature = "metal")]
use burn::backend::Wgpu;
use burn::module::{Module, Param};
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::ops::Device;
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
#[cfg(feature = "metal")]
use cubecl::wgpu::WgpuDevice;
use ricci::GCNConv;

type B = NdArray<f32>;

fn dev() -> Device<B> {
    Device::<B>::default()
}

fn fixed_layer<B: Backend>(device: &Device<B>) -> GCNConv<B> {
    let weight = Tensor::from_data(
        TensorData::new(vec![2.0f32, -1.0, 0.5, 3.0], [2, 2]),
        device,
    );
    let bias = Tensor::from_data(TensorData::new(vec![1.0f32, -2.0], [2]), device);
    GCNConv::new(Linear {
        weight: Param::from_tensor(weight),
        bias: Some(Param::from_tensor(bias)),
    })
}

fn inputs<B: Backend>(device: &Device<B>) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let x = Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, -1.0, 4.0, 3.0, 0.5], [3, 2]),
        device,
    );
    // Deliberately not row-stochastic: row sums are 3, 1.5, and 0.
    let adj = Tensor::from_data(
        TensorData::new(vec![2.0f32, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], [3, 3]),
        device,
    );
    (x, adj)
}

fn assert_close(got: Tensor<B, 2>, expected: &[f32]) {
    let got = got.into_data().to_vec::<f32>().unwrap();
    assert_eq!(got.len(), expected.len());
    for (i, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() < 1e-6,
            "element {i}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn gcn_forward_matches_affine_gcn_equation() {
    let layer = fixed_layer(&dev());
    let (x, adj) = inputs(&dev());

    // XW = [[3,5], [0,13], [6.25,-1.5]], then adj @ XW + b.
    assert_close(layer.forward(x, adj), &[7.0, 21.0, 7.25, 3.0, 1.0, -2.0]);
}

#[test]
fn gcn_legacy_differs_by_adjacency_weighted_bias() {
    let layer = fixed_layer(&dev());
    let (x, adj) = inputs(&dev());

    assert_close(
        layer.forward_legacy(x.clone(), adj.clone()),
        &[9.0, 17.0, 7.75, 2.0, 0.0, 0.0],
    );

    let corrected = layer.forward(x, adj);
    // legacy - corrected = (row_sum(adj) - 1) * b
    assert_close(
        layer.forward_legacy(inputs(&dev()).0, inputs(&dev()).1) - corrected,
        &[2.0, -4.0, 0.5, -1.0, -1.0, 2.0],
    );
}

#[test]
fn gcn_record_round_trip_preserves_both_forward_paths() {
    let layer = fixed_layer(&dev());
    let record = layer.clone().into_record();
    let restored = fixed_layer(&dev()).load_record(record);
    let (x, adj) = inputs(&dev());

    assert_close(
        restored.forward(x.clone(), adj.clone()),
        &[7.0, 21.0, 7.25, 3.0, 1.0, -2.0],
    );
    assert_close(
        restored.forward_legacy(x, adj),
        &[9.0, 17.0, 7.75, 2.0, 0.0, 0.0],
    );
}

/// Run with `cargo test --features metal --test gcn_forward -- --ignored gcn_metal`.
///
/// This selects the first integrated GPU explicitly. On Apple Silicon that is
/// the Metal device; it is ignored because it requires local GPU access.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a local Metal device; run with --features metal -- --ignored gcn_metal"]
fn gcn_metal_matches_ndarray_forward_and_input_weight_gradients() {
    type Cpu = Autodiff<NdArray<f32>>;
    type Metal = Autodiff<Wgpu<f32, i32>>;

    let cpu_device = Device::<Cpu>::default();
    let metal_device = WgpuDevice::IntegratedGpu(0);
    eprintln!("GCN Metal parity device: {metal_device:?}");
    let cpu_layer = fixed_layer::<Cpu>(&cpu_device);
    let metal_layer = fixed_layer::<Metal>(&metal_device);
    let (cpu_x, cpu_adj) = inputs::<Cpu>(&cpu_device);
    let (metal_x, metal_adj) = inputs::<Metal>(&metal_device);
    let cpu_x = cpu_x.require_grad();
    let metal_x = metal_x.require_grad();
    let cpu_weight = cpu_layer.linear().weight.val();
    let metal_weight = metal_layer.linear().weight.val();

    let cpu_output = cpu_layer.forward(cpu_x.clone(), cpu_adj);
    let metal_output = metal_layer.forward(metal_x.clone(), metal_adj);
    let cpu_grads = cpu_output.clone().sum().backward();
    let metal_grads = metal_output.clone().sum().backward();

    assert_slice_close(
        metal_output.into_data().to_vec::<f32>().unwrap().as_slice(),
        &cpu_output.into_data().to_vec::<f32>().unwrap(),
        1e-5,
    );
    assert_slice_close(
        metal_x
            .grad(&metal_grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .as_slice(),
        &cpu_x
            .grad(&cpu_grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        1e-5,
    );
    assert_slice_close(
        metal_weight
            .grad(&metal_grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .as_slice(),
        &cpu_weight
            .grad(&cpu_grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        1e-5,
    );
}

#[cfg(feature = "metal")]
fn assert_slice_close(got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= tolerance,
            "element {index}: got {got}, expected {expected}, tolerance {tolerance}"
        );
    }
}
