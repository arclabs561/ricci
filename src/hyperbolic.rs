//! Poincare ball geometry on Burn tensors.
//!
//! All operations are backend-agnostic: they work on any `B: Backend` (ndarray, wgpu, tch, etc.).
//! The reference implementation for numerical correctness is `hyperball::PoincareBallCore`.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Poincare ball operations on Burn tensors (curvature parameter `c > 0`).
///
/// Implements the core geometric primitives needed for hyperbolic neural networks:
/// projection, Mobius addition, exp/log maps, distance, and parallel transport.
#[derive(Debug, Clone, Copy)]
pub struct PoincareBall {
    /// Curvature parameter `c > 0` where sectional curvature is `K = -c`.
    pub c: f32,
    /// Numerical epsilon for clamps/div-by-zero avoidance.
    pub eps: f32,
}

impl PoincareBall {
    #[must_use]
    pub fn new(c: f64) -> Self {
        Self {
            c: c as f32,
            eps: 1e-5,
        }
    }

    /// Construct with a custom epsilon for numerical stability.
    #[must_use]
    pub fn new_with_eps(c: f64, eps: f32) -> Self {
        Self { c: c as f32, eps }
    }

    fn sqrt_c(&self) -> f32 {
        self.c.sqrt()
    }

    fn max_norm(&self) -> f32 {
        (1.0 / self.sqrt_c()) - 1e-5
    }

    fn atanh<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // atanh(x) = 0.5 * ln((1+x)/(1-x))
        let ones = Tensor::<B, 2>::ones(x.dims(), &x.device());
        let num = ones.clone() + x.clone();
        let den = ones - x;
        (num / (den + self.eps)).log() * 0.5
    }

    fn norm_keepdim<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let b = x.dims()[0];
        x.powf_scalar(2.0).sum_dim(1).sqrt().reshape([b, 1])
    }

    fn norm_sq_keepdim<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let b = x.dims()[0];
        x.powf_scalar(2.0).sum_dim(1).reshape([b, 1])
    }

    fn dot_keepdim<B: Backend>(&self, x: Tensor<B, 2>, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let b = x.dims()[0];
        (x * y).sum_dim(1).reshape([b, 1])
    }

    /// Conformal factor: `lambda_x = 2 / (1 - c ||x||^2)`.
    pub fn lambda_x<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let b = x.dims()[0];
        let dev = x.device();
        let x2 = x.powf_scalar(2.0).sum_dim(1).reshape([b, 1]);
        let denom = (Tensor::<B, 2>::ones([b, 1], &dev) - x2 * self.c).clamp_min(self.eps);
        Tensor::<B, 2>::ones([b, 1], &dev) * 2.0 / denom
    }

    /// Project points to stay inside the ball `||x|| < (1 - eps) / sqrt(c)`.
    pub fn project<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let b = x.dims()[0];
        let norm = self.norm_keepdim(x.clone());
        let max = self.max_norm();
        let denom = norm + self.eps;
        let scale = (Tensor::<B, 2>::ones([b, 1], &x.device()) * max / denom).clamp_max(1.0);
        x * scale
    }

    /// Mobius addition on the ball.
    pub fn mobius_add<B: Backend>(&self, x: Tensor<B, 2>, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let b = x.dims()[0];
        let x2 = self.norm_sq_keepdim(x.clone());
        let y2 = self.norm_sq_keepdim(y.clone());
        let xy = self.dot_keepdim(x.clone(), y.clone());

        let ones = Tensor::<B, 2>::ones([b, 1], &x.device());

        let a = ones.clone() + xy.clone() * (2.0 * self.c) + y2.clone() * self.c;
        let b1 = ones.clone() - x2.clone() * self.c;
        let num = x * a + y * b1;

        let denom =
            (ones + xy * (2.0 * self.c) + (x2 * y2) * (self.c * self.c)).clamp_min(self.eps);
        self.project(num / denom)
    }

    /// Hyperbolic distance `d(x,y)` on the ball.
    pub fn distance<B: Backend>(&self, x: Tensor<B, 2>, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.project(x);
        let y = self.project(y);
        let neg_x = x * -1.0;
        let u = self.mobius_add(neg_x, y);
        let norm_u = self.norm_keepdim(u) * self.sqrt_c();
        let z = norm_u.clamp_max(1.0 - self.eps);
        self.atanh(z) * (2.0 / self.sqrt_c())
    }

    /// Log map at origin: `log_0(x)`.
    pub fn log0<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.project(x);
        let norm = self.norm_keepdim(x.clone());
        let z = (norm.clone() * self.sqrt_c()).clamp_max(1.0 - self.eps);
        let atanh_z = self.atanh(z);
        let scale = atanh_z / (norm + self.eps) / self.sqrt_c();
        x * scale
    }

    /// Exp map at origin: `exp_0(v)`.
    pub fn exp0<B: Backend>(&self, v: Tensor<B, 2>) -> Tensor<B, 2> {
        let norm = self.norm_keepdim(v.clone());
        let z = norm.clone() * self.sqrt_c();
        let scale = z.clone().tanh() / (z + self.eps) / self.sqrt_c();
        self.project(v * scale)
    }

    /// Log map at basepoint p: `log_p(x)`.
    pub fn log_map<B: Backend>(&self, p: Tensor<B, 2>, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let p = self.project(p);
        let x = self.project(x);
        let neg_p = p.clone() * -1.0;
        let delta = self.mobius_add(neg_p, x);
        let norm = self.norm_keepdim(delta.clone());

        let lambda = self.lambda_x(p);
        let z = (norm.clone() * self.sqrt_c()).clamp_max(1.0 - self.eps);
        let atanh_z = self.atanh(z);
        let factor = (atanh_z / (norm + self.eps)) * (2.0 / lambda / self.sqrt_c());
        delta * factor
    }

    /// Exp map at basepoint p: `exp_p(v)`.
    pub fn exp_map<B: Backend>(&self, p: Tensor<B, 2>, v: Tensor<B, 2>) -> Tensor<B, 2> {
        let p = self.project(p);
        let vnorm = self.norm_keepdim(v.clone());
        let lambda = self.lambda_x(p.clone());
        let z = vnorm.clone() * lambda * (self.sqrt_c() * 0.5);
        let scale = z.tanh() / (vnorm + self.eps) / self.sqrt_c();
        let second = v * scale;
        self.mobius_add(p, second)
    }

    /// Parallel transport from 0 to x along the radial geodesic.
    ///
    /// `P^c_{0 -> x}(v) = (lambda_0 / lambda_x) v = (2 / lambda_x) v`.
    pub fn parallel_transport_0_to_x<B: Backend>(
        &self,
        x: Tensor<B, 2>,
        v0: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let x = self.project(x);
        let lambda_x = self.lambda_x(x);
        let scale = 2.0 / (lambda_x + self.eps);
        v0 * scale
    }

    /// Project a tangent vector at the origin to have bounded norm.
    ///
    /// Ensures the tangent vector maps to a point inside the ball under exp0.
    /// Used between curvature changes (HGCN pattern: log0 with c_in, proj_tan0 + exp0 with c_out).
    pub fn proj_tan0<B: Backend>(&self, v: Tensor<B, 2>) -> Tensor<B, 2> {
        // For the Poincare ball, tangent space at origin is Euclidean (no rescaling needed
        // beyond ensuring the exp map stays in the ball). The conformal factor at origin is
        // lambda_0 = 2, so tangent vectors are already in standard coordinates.
        // We just clamp the norm to avoid exp0 overshooting the boundary.
        let max = self.max_norm();
        let norm = self.norm_keepdim(v.clone());
        let b = v.dims()[0];
        let scale =
            (Tensor::<B, 2>::ones([b, 1], &v.device()) * max / (norm + self.eps)).clamp_max(1.0);
        v * scale
    }

    /// Apply a Euclidean activation function in hyperbolic space.
    ///
    /// Pattern from Chami et al. (2019): `exp0_out(act(log0_in(x)))`.
    /// Supports per-layer curvature change when `ball_out` differs from `self`.
    pub fn hyp_act<B: Backend, F>(
        &self,
        x: Tensor<B, 2>,
        act: F,
        ball_out: &PoincareBall,
    ) -> Tensor<B, 2>
    where
        F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    {
        let xt = act(self.log0(x));
        let xt = ball_out.proj_tan0(xt);
        ball_out.project(ball_out.exp0(xt))
    }

    /// Bias translation: `x oplus_c b = exp_x( P_{0->x}(log_0(b)) )`.
    ///
    /// Translates `x` by a hyperbolic bias `b` (Ganea et al. 2018, Eq. 28).
    pub fn bias_translate<B: Backend>(&self, x: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.project(x);
        let b = self.project(b);
        let v0 = self.log0(b);
        let vx = self.parallel_transport_0_to_x(x.clone(), v0);
        self.exp_map(x, vx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;
    use proptest::prelude::*;

    type B = NdArray<f32>;

    fn dev() -> <B as Backend>::Device {
        <B as Backend>::Device::default()
    }

    fn to_burn(v: &[f32], shape: [usize; 2]) -> Tensor<B, 2> {
        Tensor::from_data(TensorData::new(v.to_vec(), shape), &dev())
    }

    fn l1(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }

    fn all_finite(v: &[f32]) -> bool {
        v.iter().all(|x| x.is_finite())
    }

    fn clamp_norm(mut v: Vec<f64>, max: f64) -> Vec<f64> {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > max && norm.is_finite() && norm > 0.0 {
            let s = max / norm;
            for x in &mut v {
                *x *= s;
            }
        }
        v
    }

    // Reference implementations using hyperball (canonical source of truth).
    use hyperball::core::PoincareBallCore;

    fn ref_ball(c: f64) -> PoincareBallCore<f64> {
        PoincareBallCore::new(c)
    }

    fn ref_mobius_add(c: f64, x: &[f64], y: &[f64]) -> Vec<f64> {
        ref_ball(c).mobius_add(x, y)
    }

    fn ref_distance(c: f64, x: &[f64], y: &[f64]) -> f64 {
        ref_ball(c).distance(x, y)
    }

    fn ref_exp_map(c: f64, x: &[f64], v: &[f64]) -> Vec<f64> {
        let v_norm: f64 = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        let c_sqrt = c.sqrt();
        let dot_xx: f64 = x.iter().map(|a| a * a).sum();
        let lambda_x = 2.0 / (1.0 - c * dot_xx);

        if v_norm < 1e-6 {
            return x.to_vec();
        }

        let direction = (c_sqrt * lambda_x * v_norm / 2.0).tanh();
        let scaled: Vec<f64> = v
            .iter()
            .map(|&vi| vi * direction / (c_sqrt * v_norm))
            .collect();
        ref_ball(c).mobius_add(x, &scaled)
    }

    fn ref_log_map(c: f64, x: &[f64], y: &[f64]) -> Vec<f64> {
        let dot_xx: f64 = x.iter().map(|a| a * a).sum();
        let c_sqrt = c.sqrt();
        let lambda_x = 2.0 / (1.0 - c * dot_xx);

        let neg_x: Vec<f64> = x.iter().map(|&a| -a).collect();
        let diff = ref_ball(c).mobius_add(&neg_x, y);
        let diff_norm: f64 = diff.iter().map(|a| a * a).sum::<f64>().sqrt();

        if diff_norm < 1e-6 {
            return vec![0.0; x.len()];
        }

        let scale = (2.0 / (c_sqrt * lambda_x)) * (c_sqrt * diff_norm).atanh();
        diff.iter().map(|&d| d * scale / diff_norm).collect()
    }

    // --- Smoke tests ---

    #[test]
    fn logexp_roundtrip_at_origin() {
        let ball = PoincareBall::new(1.0);
        let x = to_burn(&[0.10, -0.05, 0.02, 0.03, 0.04, -0.01], [2, 3]);
        let x = ball.project(x);

        let v = ball.log0(x.clone());
        let x2 = ball.exp0(v);

        let x_v = x.to_data().to_vec::<f32>().unwrap();
        let x2_v = x2.to_data().to_vec::<f32>().unwrap();
        assert!(l1(&x_v, &x2_v) < 1e-3, "exp0(log0(x)) roundtrip failed");
    }

    #[test]
    fn mobius_add_identity_and_distance_zero() {
        let ball = PoincareBall::new(1.0);
        let x = to_burn(&[0.10, -0.05, 0.02, 0.03, 0.04, -0.01], [2, 3]);
        let z = to_burn(&[0.0; 6], [2, 3]);
        let x = ball.project(x);

        let lhs = ball.mobius_add(x.clone(), z.clone());
        let rhs = ball.mobius_add(z, x.clone());

        let x_v = x.clone().to_data().to_vec::<f32>().unwrap();
        let lhs_v = lhs.to_data().to_vec::<f32>().unwrap();
        let rhs_v = rhs.to_data().to_vec::<f32>().unwrap();
        assert!(l1(&x_v, &lhs_v) < 1e-3, "x + 0 != x");
        assert!(l1(&x_v, &rhs_v) < 1e-3, "0 + x != x");

        let d = ball.distance(x.clone(), x);
        let d_v = d.to_data().to_vec::<f32>().unwrap();
        let d_sum: f32 = d_v.iter().map(|x| x.abs()).sum();
        assert!(d_sum < 1e-3, "d(x,x) should be ~0, got {d_sum}");
    }

    #[test]
    fn log0_matches_log_map_at_origin() {
        let ball = PoincareBall::new(1.0);
        let x = to_burn(&[0.10, -0.05, 0.02, 0.03, 0.04, -0.01], [2, 3]);
        let x = ball.project(x);
        let o = to_burn(&[0.0; 6], [2, 3]);

        let a = ball.log0(x.clone());
        let b = ball.log_map(o.clone(), x.clone());
        let a_v = a.to_data().to_vec::<f32>().unwrap();
        let b_v = b.to_data().to_vec::<f32>().unwrap();
        assert!(l1(&a_v, &b_v) < 1e-3, "log0 != log_map(0, x)");

        let v = to_burn(&[0.01, -0.02, 0.03, -0.01, 0.02, -0.03], [2, 3]);
        let a = ball.exp0(v.clone());
        let b = ball.exp_map(o, v);
        let a_v = a.to_data().to_vec::<f32>().unwrap();
        let b_v = b.to_data().to_vec::<f32>().unwrap();
        assert!(l1(&a_v, &b_v) < 1e-3, "exp0 != exp_map(0, v)");
    }

    #[test]
    fn exp_log_roundtrip_at_basepoint() {
        let ball = PoincareBall::new(1.0);
        let x = ball.project(to_burn(&[0.10, -0.05, 0.02, 0.03, 0.04, -0.01], [2, 3]));
        let y = ball.project(to_burn(&[0.02, 0.01, -0.03, -0.04, 0.01, 0.02], [2, 3]));

        let v = ball.log_map(x.clone(), y.clone());
        let y2 = ball.exp_map(x, v);

        let y_v = y.to_data().to_vec::<f32>().unwrap();
        let y2_v = y2.to_data().to_vec::<f32>().unwrap();
        assert!(l1(&y_v, &y2_v) < 5e-2, "exp_p(log_p(y)) roundtrip failed");
    }

    #[test]
    fn bias_translate_matches_mobius_add() {
        let ball = PoincareBall::new(1.0);
        let x = ball.project(to_burn(&[0.10, -0.05, 0.02, 0.03, 0.04, -0.01], [2, 3]));
        let b = ball.project(to_burn(&[0.01, 0.02, -0.01, -0.02, 0.01, 0.01], [2, 3]));

        let a = ball.mobius_add(x.clone(), b.clone());
        let c = ball.bias_translate(x, b);
        let a_v = a.to_data().to_vec::<f32>().unwrap();
        let c_v = c.to_data().to_vec::<f32>().unwrap();
        assert!(
            l1(&a_v, &c_v) < 5e-3,
            "bias_translate should match mobius_add"
        );
    }

    // --- Edge case tests ---

    #[test]
    fn zero_vector_ops_produce_finite_results() {
        let ball = PoincareBall::new(1.0);
        let z = to_burn(&[0.0; 6], [2, 3]);

        // exp0(0) should be near origin
        let e = ball.exp0(z.clone());
        let e_v = e.to_data().to_vec::<f32>().unwrap();
        assert!(all_finite(&e_v), "exp0(0) produced non-finite");
        assert!(l1(&e_v, &[0.0; 6]) < 1e-3, "exp0(0) should be ~0");

        // log0(0) should be near zero (or at least finite)
        let l = ball.log0(z.clone());
        let l_v = l.to_data().to_vec::<f32>().unwrap();
        assert!(all_finite(&l_v), "log0(0) produced non-finite");

        // distance(0, 0) should be 0
        let d = ball.distance(z.clone(), z.clone());
        let d_v = d.to_data().to_vec::<f32>().unwrap();
        assert!(all_finite(&d_v), "distance(0, 0) produced non-finite");

        // mobius_add(0, 0) should be 0
        let m = ball.mobius_add(z.clone(), z);
        let m_v = m.to_data().to_vec::<f32>().unwrap();
        assert!(all_finite(&m_v), "mobius_add(0, 0) produced non-finite");
    }

    #[test]
    fn near_boundary_ops_produce_finite_results() {
        let ball = PoincareBall::new(1.0);
        let max = ball.max_norm();
        // Points at 95% of ball radius.
        let r = max * 0.95;
        let x = to_burn(&[r, 0.0, 0.0, 0.0, r, 0.0], [2, 3]);
        let y = to_burn(&[0.0, 0.0, r, -r, 0.0, 0.0], [2, 3]);

        let x = ball.project(x);
        let y = ball.project(y);

        let d = ball.distance(x.clone(), y.clone());
        let d_v = d.to_data().to_vec::<f32>().unwrap();
        assert!(all_finite(&d_v), "near-boundary distance non-finite");
        assert!(d_v.iter().all(|&v| v >= 0.0), "negative distance");

        let v = ball.log_map(x.clone(), y.clone());
        let v_v = v.to_data().to_vec::<f32>().unwrap();
        assert!(all_finite(&v_v), "near-boundary log_map non-finite");

        let y2 = ball.exp_map(x.clone(), v);
        let y2_v = y2.to_data().to_vec::<f32>().unwrap();
        assert!(all_finite(&y2_v), "near-boundary exp_map non-finite");

        let m = ball.mobius_add(x, y);
        let m_v = m.to_data().to_vec::<f32>().unwrap();
        assert!(all_finite(&m_v), "near-boundary mobius_add non-finite");
    }

    #[test]
    fn parallel_transport_scaling_is_correct() {
        // P_{0->x}(v) = (lambda_0 / lambda_x) v = (2 / lambda_x) v
        let ball = PoincareBall::new(1.0);
        let x = ball.project(to_burn(&[0.3, -0.2, 0.1], [1, 3]));
        let v0 = to_burn(&[0.05, -0.03, 0.02], [1, 3]);

        let vx = ball.parallel_transport_0_to_x(x.clone(), v0.clone());
        let vx_v = vx.to_data().to_vec::<f32>().unwrap();
        assert!(all_finite(&vx_v), "parallel transport non-finite");

        // Manual check: scale = 2 / lambda_x = (1 - c||x||^2)
        let x_v = x.to_data().to_vec::<f32>().unwrap();
        let x_norm_sq: f32 = x_v.iter().map(|a| a * a).sum();
        let expected_scale = 1.0 - ball.c * x_norm_sq;
        let v0_v = v0.to_data().to_vec::<f32>().unwrap();
        for i in 0..3 {
            let expected = v0_v[i] * expected_scale;
            assert!(
                (vx_v[i] - expected).abs() < 1e-4,
                "parallel transport dim {i}: got {} expected {}",
                vx_v[i],
                expected
            );
        }
    }

    #[test]
    fn multi_curvature_smoke() {
        // Operations should produce finite results across curvature range.
        for c in [0.1, 0.5, 1.0, 2.0, 5.0] {
            let ball = PoincareBall::new(c);
            let max = ball.max_norm();
            let r = max * 0.5;
            let x = ball.project(to_burn(&[r, 0.0, 0.0], [1, 3]));
            let y = ball.project(to_burn(&[0.0, r, 0.0], [1, 3]));

            let d = ball.distance(x.clone(), y.clone());
            let d_v = d.to_data().to_vec::<f32>().unwrap();
            assert!(all_finite(&d_v), "c={c}: distance non-finite");
            assert!(d_v[0] > 0.0, "c={c}: distance should be positive");

            let v = ball.log_map(x.clone(), y.clone());
            let y2 = ball.exp_map(x, v);
            let y_v = y.to_data().to_vec::<f32>().unwrap();
            let y2_v = y2.to_data().to_vec::<f32>().unwrap();
            assert!(all_finite(&y2_v), "c={c}: roundtrip non-finite");
            assert!(
                l1(&y_v, &y2_v) < 0.1,
                "c={c}: roundtrip error {}",
                l1(&y_v, &y2_v)
            );
        }
    }

    #[test]
    fn proj_tan0_clamps_large_tangent_vectors() {
        let ball = PoincareBall::new(1.0);
        // Large tangent vector that would overshoot the ball under exp0.
        let v = to_burn(&[5.0, 5.0, 5.0], [1, 3]);
        let v_proj = ball.proj_tan0(v);
        let v_v = v_proj.to_data().to_vec::<f32>().unwrap();
        let norm: f32 = v_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            norm <= ball.max_norm() + 1e-4,
            "proj_tan0 should clamp norm to max_norm, got {norm}"
        );
        // Small tangent vector should pass through unchanged.
        let v_small = to_burn(&[0.01, -0.01, 0.005], [1, 3]);
        let v_small_proj = ball.proj_tan0(v_small.clone());
        let a = v_small.to_data().to_vec::<f32>().unwrap();
        let b = v_small_proj.to_data().to_vec::<f32>().unwrap();
        assert!(
            l1(&a, &b) < 1e-5,
            "proj_tan0 should not change small vectors"
        );
    }

    #[test]
    fn hyp_act_identity_is_close_to_input() {
        let ball = PoincareBall::new(1.0);
        let x = ball.project(to_burn(&[0.1, -0.05, 0.02, 0.03, 0.04, -0.01], [2, 3]));
        let x_v = x.clone().to_data().to_vec::<f32>().unwrap();

        // Identity activation: exp0(log0(x)) should round-trip.
        let y = ball.hyp_act(x, |t| t, &ball);
        let y_v = y.to_data().to_vec::<f32>().unwrap();
        assert!(l1(&x_v, &y_v) < 1e-3, "identity hyp_act should round-trip");
    }

    #[test]
    fn hyp_act_relu_produces_finite() {
        let ball = PoincareBall::new(1.0);
        let x = ball.project(to_burn(&[0.1, -0.05, 0.02, -0.03, 0.04, -0.01], [2, 3]));
        let y = ball.hyp_act(x, |t| t.clamp_min(0.0), &ball);
        let y_v = y.to_data().to_vec::<f32>().unwrap();
        assert!(all_finite(&y_v), "hyp_act(relu) non-finite");
        // Result should be inside the ball.
        for row in 0..2 {
            let norm: f32 = y_v[row * 3..row * 3 + 3]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            assert!(
                norm < ball.max_norm() + 1e-4,
                "hyp_act output outside ball: norm={norm}"
            );
        }
    }

    #[test]
    fn hyp_act_curvature_change() {
        let ball_in = PoincareBall::new(1.0);
        let ball_out = PoincareBall::new(2.0);
        let x = ball_in.project(to_burn(&[0.1, -0.05, 0.02], [1, 3]));

        let y = ball_in.hyp_act(x, |t| t.clamp_min(0.0), &ball_out);
        let y_v = y.to_data().to_vec::<f32>().unwrap();
        assert!(all_finite(&y_v), "curvature-change hyp_act non-finite");
        // Output should be inside ball_out (radius 1/sqrt(2) ~ 0.707).
        let norm: f32 = y_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            norm < ball_out.max_norm() + 1e-4,
            "output outside ball_out: norm={norm} max={}",
            ball_out.max_norm()
        );
    }

    // --- Cross-validation against hyperball reference ---

    #[test]
    fn matches_hyperball_on_core_ops() {
        let c = 1.0f64;
        let ball = PoincareBall::new(c);

        let x_v = vec![0.10f32, -0.05, 0.02, 0.03, 0.04, -0.01, -0.08, 0.02, 0.05];
        let y_v = vec![0.02f32, 0.01, -0.03, 0.05, -0.02, 0.02, 0.01, 0.02, 0.03];

        let x = ball.project(to_burn(&x_v, [3, 3]));
        let y = ball.project(to_burn(&y_v, [3, 3]));
        let x_f = x.clone().to_data().to_vec::<f32>().unwrap();
        let y_f = y.clone().to_data().to_vec::<f32>().unwrap();

        // mobius_add
        let z = ball.mobius_add(x.clone(), y.clone());
        let z_f = z.to_data().to_vec::<f32>().unwrap();
        for i in 0..3 {
            let xi: Vec<f64> = x_f[i * 3..i * 3 + 3].iter().map(|v| *v as f64).collect();
            let yi: Vec<f64> = y_f[i * 3..i * 3 + 3].iter().map(|v| *v as f64).collect();
            let z_ref: Vec<f32> = ref_mobius_add(c, &xi, &yi)
                .iter()
                .map(|v| *v as f32)
                .collect();
            assert!(
                l1(&z_f[i * 3..i * 3 + 3], &z_ref) < 5e-2,
                "mobius_add mismatch row {i}"
            );
        }

        // distance
        let d = ball.distance(x.clone(), y.clone());
        let d_f = d.to_data().to_vec::<f32>().unwrap();
        for i in 0..3 {
            let xi: Vec<f64> = x_f[i * 3..i * 3 + 3].iter().map(|v| *v as f64).collect();
            let yi: Vec<f64> = y_f[i * 3..i * 3 + 3].iter().map(|v| *v as f64).collect();
            let d_ref = ref_distance(c, &xi, &yi) as f32;
            assert!(
                (d_f[i] - d_ref).abs() < 1e-2,
                "distance mismatch row {i}: burn={} ref={}",
                d_f[i],
                d_ref
            );
        }

        // log0/exp0 roundtrip
        let v = ball.log0(x.clone());
        let x2 = ball.exp0(v);
        let x2_f = x2.to_data().to_vec::<f32>().unwrap();
        assert!(l1(&x_f, &x2_f) < 1e-3, "exp0(log0(x)) roundtrip mismatch");
    }

    // --- Property tests ---

    fn arb_point(dim: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(-0.3f64..0.3f64, dim)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_distance_matches_hyperball(x0 in arb_point(3), y0 in arb_point(3)) {
            let c = 1.0f64;
            let ball = PoincareBall::new(c);
            let x0 = clamp_norm(x0, 0.70);
            let y0 = clamp_norm(y0, 0.70);

            let x_f: Vec<f32> = x0.iter().map(|v| *v as f32).collect();
            let y_f: Vec<f32> = y0.iter().map(|v| *v as f32).collect();
            let x = ball.project(to_burn(&x_f, [1, 3]));
            let y = ball.project(to_burn(&y_f, [1, 3]));
            let x_v = x.clone().to_data().to_vec::<f32>().unwrap();
            let y_v = y.clone().to_data().to_vec::<f32>().unwrap();

            let d_burn = ball.distance(x, y).to_data().to_vec::<f32>().unwrap()[0];
            let xi: Vec<f64> = x_v.iter().map(|v| *v as f64).collect();
            let yi: Vec<f64> = y_v.iter().map(|v| *v as f64).collect();
            let d_ref = ref_distance(c, &xi, &yi) as f32;

            prop_assert!((d_burn - d_ref).abs() < 5e-2,
                "distance mismatch: burn={d_burn} ref={d_ref}");
        }

        #[test]
        fn prop_mobius_add_matches_hyperball(x0 in arb_point(3), y0 in arb_point(3)) {
            let c = 1.0f64;
            let ball = PoincareBall::new(c);
            let x0 = clamp_norm(x0, 0.35);
            let y0 = clamp_norm(y0, 0.35);

            let x_f: Vec<f32> = x0.iter().map(|v| *v as f32).collect();
            let y_f: Vec<f32> = y0.iter().map(|v| *v as f32).collect();
            let x = ball.project(to_burn(&x_f, [1, 3]));
            let y = ball.project(to_burn(&y_f, [1, 3]));
            let x_v = x.clone().to_data().to_vec::<f32>().unwrap();
            let y_v = y.clone().to_data().to_vec::<f32>().unwrap();

            let z = ball.mobius_add(x, y);
            let z_f = z.to_data().to_vec::<f32>().unwrap();
            let xi: Vec<f64> = x_v.iter().map(|v| *v as f64).collect();
            let yi: Vec<f64> = y_v.iter().map(|v| *v as f64).collect();
            let z_ref: Vec<f32> = ref_mobius_add(c, &xi, &yi).iter().map(|v| *v as f32).collect();

            prop_assert!(l1(&z_f, &z_ref) < 5e-2,
                "mobius_add mismatch l1={}", l1(&z_f, &z_ref));
        }

        #[test]
        fn prop_log_exp_roundtrip(x0 in arb_point(3), y0 in arb_point(3)) {
            let c = 1.0f64;
            let ball = PoincareBall::new(c);
            let x0 = clamp_norm(x0, 0.55);
            let y0 = clamp_norm(y0, 0.55);

            let x_f: Vec<f32> = x0.iter().map(|v| *v as f32).collect();
            let y_f: Vec<f32> = y0.iter().map(|v| *v as f32).collect();
            let x = ball.project(to_burn(&x_f, [1, 3]));
            let y = ball.project(to_burn(&y_f, [1, 3]));
            let y_v = y.clone().to_data().to_vec::<f32>().unwrap();

            let v = ball.log_map(x.clone(), y);
            let y2 = ball.exp_map(x, v);
            let y2_v = y2.to_data().to_vec::<f32>().unwrap();

            prop_assert!(l1(&y_v, &y2_v) < 8e-2,
                "exp_p(log_p(y)) roundtrip l1={}", l1(&y_v, &y2_v));
        }

        #[test]
        fn prop_log_map_matches_hyperball(
            p0 in arb_point(3),
            y0 in arb_point(3),
        ) {
            let c = 1.0f64;
            let ball = PoincareBall::new(c);
            let p0 = clamp_norm(p0, 0.55);
            let y0 = clamp_norm(y0, 0.55);

            let p_f: Vec<f32> = p0.iter().map(|v| *v as f32).collect();
            let y_f: Vec<f32> = y0.iter().map(|v| *v as f32).collect();
            let p = ball.project(to_burn(&p_f, [1, 3]));
            let y = ball.project(to_burn(&y_f, [1, 3]));
            let p_v = p.clone().to_data().to_vec::<f32>().unwrap();
            let y_v = y.clone().to_data().to_vec::<f32>().unwrap();

            let v_burn = ball.log_map(p, y).to_data().to_vec::<f32>().unwrap();
            let pi: Vec<f64> = p_v.iter().map(|v| *v as f64).collect();
            let yi: Vec<f64> = y_v.iter().map(|v| *v as f64).collect();
            let v_ref: Vec<f32> = ref_log_map(c, &pi, &yi).iter().map(|v| *v as f32).collect();

            prop_assert!(l1(&v_burn, &v_ref) < 8e-2,
                "log_map mismatch l1={}", l1(&v_burn, &v_ref));
        }

        #[test]
        fn prop_exp_map_matches_hyperball(
            x0 in arb_point(3),
            v0 in prop::collection::vec(-0.2f64..0.2f64, 3usize),
        ) {
            let c = 1.0f64;
            let ball = PoincareBall::new(c);
            let x0 = clamp_norm(x0, 0.55);

            let x_f: Vec<f32> = x0.iter().map(|v| *v as f32).collect();
            let v_f: Vec<f32> = v0.iter().map(|v| *v as f32).collect();
            let x = ball.project(to_burn(&x_f, [1, 3]));
            let v = to_burn(&v_f, [1, 3]);
            let x_v = x.clone().to_data().to_vec::<f32>().unwrap();

            let y_burn = ball.exp_map(x, v).to_data().to_vec::<f32>().unwrap();
            let xi: Vec<f64> = x_v.iter().map(|v| *v as f64).collect();
            let vi: Vec<f64> = v0.clone();
            let y_ref: Vec<f32> = ref_exp_map(c, &xi, &vi).iter().map(|v| *v as f32).collect();

            prop_assert!(l1(&y_burn, &y_ref) < 5e-2,
                "exp_map mismatch l1={}", l1(&y_burn, &y_ref));
        }

        // Near-boundary: test at 95% of max norm for all core ops.
        #[test]
        fn prop_near_boundary_finite(x0 in arb_point(3), y0 in arb_point(3)) {
            let c = 1.0f64;
            let ball = PoincareBall::new(c);
            let max = ball.max_norm() as f64;
            let x0 = clamp_norm(x0, max * 0.95);
            let y0 = clamp_norm(y0, max * 0.95);

            let x_f: Vec<f32> = x0.iter().map(|v| *v as f32).collect();
            let y_f: Vec<f32> = y0.iter().map(|v| *v as f32).collect();
            let x = ball.project(to_burn(&x_f, [1, 3]));
            let y = ball.project(to_burn(&y_f, [1, 3]));

            let d = ball.distance(x.clone(), y.clone()).to_data().to_vec::<f32>().unwrap();
            prop_assert!(all_finite(&d), "near-boundary distance non-finite");

            let v = ball.log_map(x.clone(), y.clone());
            let v_v = v.clone().to_data().to_vec::<f32>().unwrap();
            prop_assert!(all_finite(&v_v), "near-boundary log_map non-finite");

            let y2 = ball.exp_map(x, v).to_data().to_vec::<f32>().unwrap();
            prop_assert!(all_finite(&y2), "near-boundary exp_map non-finite");
        }

        // Multi-curvature: test roundtrips at various curvatures.
        #[test]
        fn prop_multi_curvature_roundtrip(
            x0 in arb_point(3),
            y0 in arb_point(3),
            c_idx in 0..5usize,
        ) {
            let curvatures = [0.1, 0.5, 1.0, 2.0, 5.0];
            let c = curvatures[c_idx];
            let ball = PoincareBall::new(c);
            let max = ball.max_norm() as f64;
            let x0 = clamp_norm(x0, max * 0.5);
            let y0 = clamp_norm(y0, max * 0.5);

            let x_f: Vec<f32> = x0.iter().map(|v| *v as f32).collect();
            let y_f: Vec<f32> = y0.iter().map(|v| *v as f32).collect();
            let x = ball.project(to_burn(&x_f, [1, 3]));
            let y = ball.project(to_burn(&y_f, [1, 3]));
            let y_v = y.clone().to_data().to_vec::<f32>().unwrap();

            let v = ball.log_map(x.clone(), y);
            let y2 = ball.exp_map(x, v);
            let y2_v = y2.to_data().to_vec::<f32>().unwrap();

            prop_assert!(all_finite(&y2_v), "c={c}: roundtrip non-finite");
            prop_assert!(l1(&y_v, &y2_v) < 0.15,
                "c={c}: roundtrip error {}", l1(&y_v, &y2_v));
        }
    }
}
