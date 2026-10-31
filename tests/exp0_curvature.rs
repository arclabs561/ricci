//! exp0 regressions: the origin exp map must invert log0 for every curvature.
use burn::tensor::{Device, Tensor, TensorData};
use ricci::PoincareBall;

fn t(v: &[f32], shape: [usize; 2]) -> Tensor<2> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), &Device::flex())
}

fn vec_of(x: Tensor<2>) -> Vec<f32> {
    x.to_data().try_to_vec::<f32>().unwrap()
}

/// exp_0(v) = tanh(sqrt(c)|v|) v / (sqrt(c)|v|) (Ganea et al. 2018, Lemma 2),
/// so exp0(log0(x)) = x for every curvature. `exp0` divides by
/// `(sqrt(c)|v| + eps) * sqrt(c)`, an extra 1/sqrt(c): with c = 4 the round
/// trip halves x.
#[test]
fn exp0_log0_round_trip_c4() {
    let ball = PoincareBall::new(4.0);
    let x = t(&[0.1, 0.0, -0.05, 0.08], [2, 2]);
    let x2 = ball.exp0(ball.log0(x.clone()));
    let (a, b) = (vec_of(x), vec_of(x2));
    for (p, q) in a.iter().zip(&b) {
        assert!((p - q).abs() < 1e-4, "exp0(log0(x)) = {b:?}, x = {a:?}");
    }
}

/// exp0(v) must equal exp_map(0, v) (lambda_0 = 2) for any c.
#[test]
fn exp0_matches_exp_map_at_origin_c4() {
    let ball = PoincareBall::new(4.0);
    let v = t(&[0.3, -0.2], [1, 2]);
    let o = t(&[0.0, 0.0], [1, 2]);
    let a = vec_of(ball.exp0(v.clone()));
    let b = vec_of(ball.exp_map(o, v));
    for (p, q) in a.iter().zip(&b) {
        assert!((p - q).abs() < 1e-4, "exp0 = {a:?}, exp_map(0, .) = {b:?}");
    }
}
