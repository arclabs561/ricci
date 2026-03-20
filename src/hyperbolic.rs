use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

/// Poincaré ball operations implemented directly on `candle_core::Tensor`.
///
/// This exists to resolve the backend mismatch:
/// - a CPU/ndarray reference implementation is great for geometry + correctness,
/// - `propago` uses `candle` tensors (CPU/GPU) for learning.
///
/// We start with the origin tangent-space maps (`log0`, `exp0`) which are sufficient
/// for the common “log0 → Euclidean ops → exp0” HGCN pattern.
#[derive(Debug, Clone, Copy)]
pub struct CandlePoincareBall {
    /// Curvature parameter `c > 0` where sectional curvature is `K = -c`.
    pub c: f64,
    /// Numerical epsilon for clamps/div-by-zero avoidance.
    pub eps: f64,
}

impl CandlePoincareBall {
    pub fn new(c: f64) -> Self {
        Self { c, eps: 1e-6 }
    }

    #[inline]
    fn sqrt_c(&self) -> f64 {
        self.c.sqrt()
    }

    fn atanh(&self, x: &Tensor) -> Result<Tensor> {
        // atanh(x) = 0.5 * (ln(1+x) - ln(1-x))
        let one = Tensor::full(1f32, x.shape(), x.device())?;
        let half = Tensor::full(0.5f32, x.shape(), x.device())?;
        let one_plus = x.add(&one)?;
        let one_minus = one.sub(x)?;
        one_plus.log()?.sub(&one_minus.log()?)?.mul(&half)
    }

    fn norm_keepdim(&self, x: &Tensor) -> Result<Tensor> {
        // Assumes `x` has shape [batch, d].
        x.sqr()?.sum_keepdim(1)?.sqrt()
    }

    fn norm_sq_keepdim(&self, x: &Tensor) -> Result<Tensor> {
        // Assumes `x` has shape [batch, d].
        x.sqr()?.sum_keepdim(1)
    }

    fn dot_keepdim(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        x.mul(y)?.sum_keepdim(1)
    }

    fn scalar_keepdim_like(&self, x: &Tensor, v: f32) -> Result<Tensor> {
        // Shape [batch, 1] to broadcast across feature dim.
        let (b, _d) = x.dims2()?;
        Tensor::full(v, (b, 1), x.device())
    }

    /// Project points to the open ball `||x|| < (1 - eps) / sqrt(c)`.
    pub fn project(&self, x: &Tensor) -> Result<Tensor> {
        let norm = self.norm_keepdim(x)?;
        let max_norm = (1.0 - self.eps) / self.sqrt_c();
        // scale = clamp(max_norm / (norm + eps), 0, 1)
        let eps = Tensor::full(self.eps as f32, norm.shape(), norm.device())?;
        let denom = norm.add(&eps)?;
        let scale = denom
            .recip()?
            .mul(&Tensor::full(
                max_norm as f32,
                denom.shape(),
                denom.device(),
            )?)?
            .clamp(0.0, 1.0)?;

        // Broadcast scale [batch,1] across feature dim.
        // If scale == 1, returns x unchanged.
        x.broadcast_mul(&scale)
    }

    /// Conformal factor λ_x = 2 / (1 - c||x||²).
    pub fn lambda_x(&self, x: &Tensor) -> Result<Tensor> {
        let c = self.c as f32;
        let norm_sq = self.norm_sq_keepdim(x)?;
        let denom = self
            .scalar_keepdim_like(x, 1.0)?
            .sub(&norm_sq.mul(&self.scalar_keepdim_like(x, c)?)?)?;
        // Clamp away from 0 for numerical safety.
        let denom = denom.clamp(self.eps as f32, f32::INFINITY)?;
        self.scalar_keepdim_like(x, 2.0)?.broadcast_div(&denom)
    }

    /// Parallel transport of a tangent vector from the origin to `x`.
    ///
    /// For the Poincaré ball \((D^n_c, g_c)\), Ganea et al. (2018) show:
    /// \(P^c_{0 \\to x}(v) = (\\lambda^c_0 / \\lambda^c_x) v\).
    ///
    /// This is the “shared-parameter” workhorse: you can keep a learnable bias in \(T_0\) and
    /// transport it to \(T_x\) without introducing a full gyration-based transport.
    pub fn parallel_transport_0_to_x(&self, x: &Tensor, v0: &Tensor) -> Result<Tensor> {
        let x = self.project(x)?;
        let lambda_x = self.lambda_x(&x)?; // [b,1]
                                           // lambda_0 = 2 (for any c, since ||0||=0).
        let scale = self
            .scalar_keepdim_like(&x, 2.0)?
            .broadcast_div(&lambda_x)?;
        v0.broadcast_mul(&scale)
    }

    /// Bias translation using parallel transport + exp map (Ganea et al. 2018, Eq. (28)).
    ///
    /// This translates `x` by a hyperbolic bias `b` (a point in the ball), using:
    /// `x ⊕_c b = exp_x( P_{0→x}(log_0(b)) )`.
    pub fn bias_translate_ball(&self, x: &Tensor, b: &Tensor) -> Result<Tensor> {
        let x = self.project(x)?;
        let b = self.project(b)?;
        let v0 = self.log0(&b)?;
        let vx = self.parallel_transport_0_to_x(&x, &v0)?;
        self.exp_map(&x, &vx)
    }

    /// Möbius addition x ⊕ y in the Poincaré ball (curvature -c).
    pub fn mobius_add(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let x = self.project(x)?;
        let y = self.project(y)?;
        let c = self.c as f32;

        let x2 = self.norm_sq_keepdim(&x)?;
        let y2 = self.norm_sq_keepdim(&y)?;
        let xy = self.dot_keepdim(&x, &y)?;

        // a = 1 + 2c<x,y> + c||y||^2
        let a = self
            .scalar_keepdim_like(&x, 1.0)?
            .add(&xy.mul(&self.scalar_keepdim_like(&x, 2.0 * c)?)?)?
            .add(&y2.mul(&self.scalar_keepdim_like(&x, c)?)?)?;

        // b = 1 - c||x||^2
        let b = self
            .scalar_keepdim_like(&x, 1.0)?
            .sub(&x2.mul(&self.scalar_keepdim_like(&x, c)?)?)?;

        // denom = 1 + 2c<x,y> + c^2||x||^2||y||^2
        let denom = self
            .scalar_keepdim_like(&x, 1.0)?
            .add(&xy.mul(&self.scalar_keepdim_like(&x, 2.0 * c)?)?)?
            .add(&x2.mul(&y2)?.mul(&self.scalar_keepdim_like(&x, c * c)?)?)?;
        let denom = denom.clamp(self.eps as f32, f32::INFINITY)?;

        // out = (a*x + b*y) / denom
        let num = x.broadcast_mul(&a)?.add(&y.broadcast_mul(&b)?)?;
        let out = num.broadcast_div(&denom)?;
        self.project(&out)
    }

    /// Hyperbolic distance d(x,y) using the Möbius-add + atanh form.
    pub fn distance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        // d(x,y) = (2/sqrt(c)) atanh( sqrt(c) * || (-x) ⊕ y || )
        let c_sqrt = self.sqrt_c() as f32;
        let neg_x = x.neg()?;
        let delta = self.mobius_add(&neg_x, y)?;
        let norm = self.norm_keepdim(&delta)?;
        let z = norm
            .mul(&self.scalar_keepdim_like(&delta, c_sqrt)?)?
            .clamp(0.0, 1.0 - self.eps as f32)?;
        let atanh_z = self.atanh(&z)?;
        atanh_z.mul(&self.scalar_keepdim_like(&z, 2.0 / c_sqrt)?)
    }

    /// Exponential map at point x: exp_x(v).
    pub fn exp_map(&self, x: &Tensor, v: &Tensor) -> Result<Tensor> {
        let x = self.project(x)?;
        let lambda = self.lambda_x(&x)?; // [b,1]
        let vnorm = self.norm_keepdim(v)?; // [b,1]
        let c_sqrt = self.sqrt_c() as f32;

        // factor = tanh( sqrt(c) * λ_x * ||v|| / 2 ) / (sqrt(c) * ||v||)
        let half = self.scalar_keepdim_like(v, 0.5)?;
        let arg = lambda
            .mul(&vnorm)?
            .mul(&self.scalar_keepdim_like(v, c_sqrt)?)?
            .broadcast_mul(&half)?;
        let num = arg.tanh()?;
        let denom = vnorm
            .mul(&self.scalar_keepdim_like(v, c_sqrt)?)?
            .add(&self.scalar_keepdim_like(v, self.eps as f32)?)?;
        let factor = num.broadcast_div(&denom)?;

        let u = v.broadcast_mul(&factor)?;
        self.mobius_add(&x, &u)
    }

    /// Logarithmic map at point x: log_x(y).
    pub fn log_map(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let x = self.project(x)?;
        let y = self.project(y)?;
        let c_sqrt = self.sqrt_c() as f32;

        let neg_x = x.neg()?;
        let delta = self.mobius_add(&neg_x, &y)?; // [b,d]
        let norm = self.norm_keepdim(&delta)?; // [b,1]
        let z = norm
            .mul(&self.scalar_keepdim_like(&delta, c_sqrt)?)?
            .clamp(0.0, 1.0 - self.eps as f32)?;
        let atanh_z = self.atanh(&z)?;

        let lambda = self.lambda_x(&x)?; // [b,1]
                                         // scale = (2/(sqrt(c)*λ_x)) * atanh(z) / ||delta||
        let denom = norm.add(&self.scalar_keepdim_like(&delta, self.eps as f32)?)?;
        let s1 = atanh_z.broadcast_div(&denom)?;
        let s2 = self
            .scalar_keepdim_like(&delta, 2.0)?
            .broadcast_div(&lambda.mul(&self.scalar_keepdim_like(&delta, c_sqrt)?)?)?;
        let scale = s1.mul(&s2)?;
        delta.broadcast_mul(&scale)
    }

    /// Logarithmic map at the origin: `log_0(x)`.
    pub fn log0(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.project(x)?;
        let norm = self.norm_keepdim(&x)?;
        let sqrt_c = self.sqrt_c();

        // z = sqrt(c) * ||x||, clamped to < 1.
        let z = norm
            .mul(&Tensor::full(sqrt_c as f32, norm.shape(), norm.device())?)?
            .clamp(0.0, 1.0 - self.eps)?;
        let atanh_z = self.atanh(&z)?;
        let scale = atanh_z
            .div(&z.add(&Tensor::full(self.eps as f32, z.shape(), z.device())?)?)?
            .div(&Tensor::full(sqrt_c as f32, z.shape(), z.device())?)?;

        x.broadcast_mul(&scale)
    }

    /// Exponential map at the origin: `exp_0(v)`.
    pub fn exp0(&self, v: &Tensor) -> Result<Tensor> {
        let norm = self.norm_keepdim(v)?;
        let sqrt_c = self.sqrt_c();

        let z = norm.mul(&Tensor::full(sqrt_c as f32, norm.shape(), norm.device())?)?;
        let scale = z
            .tanh()?
            .div(&z.add(&Tensor::full(self.eps as f32, z.shape(), z.device())?)?)?
            .div(&Tensor::full(sqrt_c as f32, z.shape(), z.device())?)?;

        let x = v.broadcast_mul(&scale)?;
        self.project(&x)
    }
}

/// Hyperbolic Graph Convolutional Network Layer (H2H-GCN).
///
/// Operates entirely in the Poincaré ball to minimize distortion.
pub struct HGCNConv {
    lin: Linear,
    ball: CandlePoincareBall,
}

impl HGCNConv {
    pub fn new(lin: Linear, c: f64) -> Self {
        Self {
            lin,
            ball: CandlePoincareBall::new(c),
        }
    }

    pub fn forward(&self, x: &Tensor, adj: &Tensor) -> Result<Tensor> {
        // 1) Map to tangent space at origin.
        let x_tangent = self.ball.log0(x)?;

        // 2) Euclidean message passing in the tangent space.
        let x_tangent = self.lin.forward(&x_tangent)?;
        let aggregated = adj.matmul(&x_tangent)?;

        // 3) Map back to the ball at the origin.
        self.ball.exp0(&aggregated)
    }

    /// Like [`Self::forward`], but uses an explicit basepoint `p` for the log/exp maps.
    ///
    /// Shapes:
    /// - `x`: `[n, d]`
    /// - `adj`: `[n, n]`
    /// - `p`: `[1, d]` (global basepoint) or `[n, d]` (per-row basepoint)
    pub fn forward_with_basepoint(&self, x: &Tensor, adj: &Tensor, p: &Tensor) -> Result<Tensor> {
        let (n, d) = x.dims2()?;
        let (pn, pd) = p.dims2()?;
        if pd != d {
            candle_core::bail!("basepoint dimension mismatch: x has d={d}, p has d={pd}");
        }
        if pn != 1 && pn != n {
            candle_core::bail!(
                "basepoint batch mismatch: p must have 1 or n rows (got {pn}, n={n})"
            );
        }
        let p = if pn == 1 {
            p.broadcast_as((n, d))?
        } else {
            p.clone()
        };

        // 1) Map to tangent space at basepoint.
        let x_tangent = self.ball.log_map(&p, x)?;

        // 2) Euclidean message passing in the tangent space.
        let x_tangent = self.lin.forward(&x_tangent)?;
        let aggregated = adj.matmul(&x_tangent)?;

        // 3) Map back to the manifold at basepoint.
        self.ball.exp_map(&p, &aggregated)
    }

    /// Like [`Self::forward_with_basepoint`], but adds a shared bias `b0` living in `T_0`.
    ///
    /// We transport `b0` to `T_p` using `P_{0→p}` and add it in the tangent space before `exp_p`.
    pub fn forward_with_basepoint_and_bias0(
        &self,
        x: &Tensor,
        adj: &Tensor,
        p: &Tensor,
        b0: &Tensor,
    ) -> Result<Tensor> {
        let (n, d) = x.dims2()?;
        let (pn, pd) = p.dims2()?;
        if pd != d {
            candle_core::bail!("basepoint dimension mismatch: x has d={d}, p has d={pd}");
        }
        if pn != 1 && pn != n {
            candle_core::bail!(
                "basepoint batch mismatch: p must have 1 or n rows (got {pn}, n={n})"
            );
        }
        let p = if pn == 1 {
            p.broadcast_as((n, d))?
        } else {
            p.clone()
        };

        let (bn, bd) = b0.dims2()?;
        if bd != d {
            candle_core::bail!("bias0 dimension mismatch: expected d={d}, got d={bd}");
        }
        if bn != 1 && bn != n {
            candle_core::bail!("bias0 batch mismatch: b0 must have 1 or n rows (got {bn}, n={n})");
        }
        let b0 = if bn == 1 {
            b0.broadcast_as((n, d))?
        } else {
            b0.clone()
        };

        let x_tangent = self.ball.log_map(&p, x)?;
        let x_tangent = self.lin.forward(&x_tangent)?;
        let aggregated = adj.matmul(&x_tangent)?;

        let bias_p = self.ball.parallel_transport_0_to_x(&p, &b0)?;
        let aggregated = aggregated.add(&bias_p)?;

        self.ball.exp_map(&p, &aggregated)
    }

    /// Dense “local tangent” aggregation (research-reference implementation).
    ///
    /// This matches the HGCN design choice: aggregate in the tangent space **at each center point**
    /// (not a single global basepoint), which reduces distortion for relative distances.
    ///
    /// Cost: \(O(n^2 d)\) compute and \(O(n^2 d)\) intermediate memory; intended for small graphs
    /// and correctness/reference, not large-scale training.
    pub fn forward_local_dense(&self, x: &Tensor, adj: &Tensor) -> Result<Tensor> {
        let (n, d) = x.dims2()?;
        let x = self.ball.project(x)?;

        // Build all pairs (i,j) with broadcasting, then flatten to a batch.
        // p[i,j,:] = x[i,:], y[i,j,:] = x[j,:]
        let p = x
            .reshape((n, 1, d))?
            .broadcast_as((n, n, d))?
            .reshape((n * n, d))?;
        let y = x
            .reshape((1, n, d))?
            .broadcast_as((n, n, d))?
            .reshape((n * n, d))?;

        // v_ij = log_{x_i}(x_j)
        let v = self.ball.log_map(&p, &y)?; // [n*n, d]
        let v = self.lin.forward(&v)?; // apply shared linear to edge features
        let v = v.reshape((n, n, d))?;

        // Weighted sum over neighbors: sum_j adj[i,j] * v[i,j,:]
        let w = adj.unsqueeze(2)?.broadcast_as((n, n, d))?;
        let agg = v.mul(&w)?.sum(1)?; // [n, d]

        // exp_{x_i}(agg_i)
        self.ball.exp_map(&x, &agg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use ndarray::Array1;
    use proptest::prelude::*;

    // Reference Poincare ball operations from hyperball (canonical implementation).
    // These are used to validate CandlePoincareBall's tensor-based ops.
    use hyperball::core::PoincareBallCore;

    fn ref_ball(c: f64) -> PoincareBallCore<f64> {
        PoincareBallCore::new(c)
    }

    fn mobius_add(
        c: f64,
        x: &ndarray::ArrayView1<'_, f64>,
        y: &ndarray::ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        Array1::from_vec(ref_ball(c).mobius_add(x.as_slice().unwrap(), y.as_slice().unwrap()))
    }

    fn exp_map(
        c: f64,
        x: &ndarray::ArrayView1<'_, f64>,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        // hyperball only exposes exp_map_zero; for general basepoint, compose:
        // exp_x(v) = x ⊕ exp_0(PT_{x->0}(v))
        // But the Candle tests use a direct formula. Keep the inline version for
        // general-basepoint exp/log since hyperball::PoincareBallCore doesn't expose them.
        let xs = x.as_slice().unwrap();
        let vs = v.as_slice().unwrap();
        let dot_xx: f64 = xs.iter().map(|a| a * a).sum();
        let v_norm: f64 = vs.iter().map(|a| a * a).sum::<f64>().sqrt();
        let c_sqrt = c.sqrt();
        let lambda_x = 2.0 / (1.0 - c * dot_xx);

        if v_norm < 1e-6 {
            return x.to_owned();
        }

        let direction = (c_sqrt * lambda_x * v_norm / 2.0).tanh();
        let scaled: Vec<f64> = vs.iter().map(|&vi| vi * direction / (c_sqrt * v_norm)).collect();
        Array1::from_vec(ref_ball(c).mobius_add(xs, &scaled))
    }

    fn log_map(
        c: f64,
        x: &ndarray::ArrayView1<'_, f64>,
        y: &ndarray::ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let xs = x.as_slice().unwrap();
        let ys = y.as_slice().unwrap();
        let dot_xx: f64 = xs.iter().map(|a| a * a).sum();
        let c_sqrt = c.sqrt();
        let lambda_x = 2.0 / (1.0 - c * dot_xx);

        let neg_x: Vec<f64> = xs.iter().map(|&a| -a).collect();
        let diff = ref_ball(c).mobius_add(&neg_x, ys);
        let diff_norm: f64 = diff.iter().map(|a| a * a).sum::<f64>().sqrt();

        if diff_norm < 1e-6 {
            return Array1::zeros(x.len());
        }

        let scale = (2.0 / (c_sqrt * lambda_x)) * (c_sqrt * diff_norm).atanh();
        Array1::from_vec(diff.iter().map(|&d| d * scale / diff_norm).collect())
    }

    fn distance(c: f64, x: &ndarray::ArrayView1<'_, f64>, y: &ndarray::ArrayView1<'_, f64>) -> f64 {
        ref_ball(c).distance(x.as_slice().unwrap(), y.as_slice().unwrap())
    }

    #[test]
    fn hgcn_forward_shapes_smoke() -> Result<()> {
        let dev = &Device::Cpu;
        let dtype = DType::F32;

        let n = 5usize;
        let d = 3usize;
        let x = Tensor::randn(0f32, 0.1f32, (n, d), dev)?.to_dtype(dtype)?;

        // Simple identity adjacency.
        let adj = Tensor::eye(n, dtype, dev)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, dev);
        let lin = candle_nn::linear(d, d, vb)?;
        let layer = HGCNConv::new(lin, 1.0);
        let y = layer.forward(&x, &adj)?;

        let (yn, yd) = y.dims2()?;
        assert_eq!(yn, n);
        assert_eq!(yd, d);
        Ok(())
    }

    #[test]
    fn hgcn_forward_with_basepoint_shapes_smoke() -> Result<()> {
        let dev = &Device::Cpu;
        let dtype = DType::F32;

        let n = 6usize;
        let d = 4usize;
        let x = Tensor::randn(0f32, 0.1f32, (n, d), dev)?.to_dtype(dtype)?;
        let p = Tensor::zeros((1, d), dtype, dev)?;
        let adj = Tensor::eye(n, dtype, dev)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, dev);
        let lin = candle_nn::linear(d, d, vb)?;
        let layer = HGCNConv::new(lin, 1.0);

        let y = layer.forward_with_basepoint(&x, &adj, &p)?;
        let (yn, yd) = y.dims2()?;
        assert_eq!(yn, n);
        assert_eq!(yd, d);
        Ok(())
    }

    #[test]
    fn hgcn_forward_local_dense_identity_is_close_to_input() -> Result<()> {
        let dev = &Device::Cpu;
        let dtype = DType::F32;

        let n = 6usize;
        let d = 3usize;
        let ball = CandlePoincareBall::new(1.0);
        let x = ball.project(&Tensor::randn(0f32, 0.1f32, (n, d), dev)?.to_dtype(dtype)?)?;
        let adj = Tensor::eye(n, dtype, dev)?;

        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, dev);
        let lin = candle_nn::linear(d, d, vb)?;

        // For identity adjacency, the geometric part should behave like identity.
        // Set linear weights and bias to zero so we don't introduce drift from an arbitrary init.
        varmap.set_one("weight", Tensor::zeros((d, d), dtype, dev)?)?;
        varmap.set_one("bias", Tensor::zeros(d, dtype, dev)?)?;
        let layer = HGCNConv::new(lin, 1.0);

        let y = layer.forward_local_dense(&x, &adj)?;

        // Identity adjacency means each node only sees itself; log_x(x)=0 so output ~ x.
        let err = y.sub(&x)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            err < 1e-3,
            "expected near-identity under self-only adjacency, err={err}"
        );
        Ok(())
    }

    #[test]
    fn mobius_add_identity_and_distance_zero() -> Result<()> {
        let dev = &Device::Cpu;
        let dtype = DType::F32;
        let ball = CandlePoincareBall::new(1.0);

        let x = Tensor::randn(0f32, 0.1f32, (4, 3), dev)?.to_dtype(dtype)?;
        let z = Tensor::zeros((4, 3), dtype, dev)?;

        let x_proj = ball.project(&x)?;
        let lhs = ball.mobius_add(&x, &z)?;
        let rhs = ball.mobius_add(&z, &x)?;

        // x ⊕ 0 = x and 0 ⊕ x = x (up to projection).
        let diff1 = lhs.sub(&x_proj)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        let diff2 = rhs.sub(&x_proj)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff1 < 1e-3, "mobius_add identity mismatch diff={diff1}");
        assert!(diff2 < 1e-3, "mobius_add identity mismatch diff={diff2}");

        let d0 = ball.distance(&x, &x)?;
        let d0s = d0.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(d0s < 1e-3, "distance(x,x) should be ~0, got {d0s}");
        Ok(())
    }

    #[test]
    fn exp_log_roundtrip_at_point_smoke() -> Result<()> {
        let dev = &Device::Cpu;
        let dtype = DType::F32;
        let ball = CandlePoincareBall::new(1.0);

        let x = ball.project(&Tensor::randn(0f32, 0.1f32, (4, 3), dev)?.to_dtype(dtype)?)?;
        let y = ball.project(&Tensor::randn(0f32, 0.1f32, (4, 3), dev)?.to_dtype(dtype)?)?;
        let v = ball.log_map(&x, &y)?;
        let y2 = ball.exp_map(&x, &v)?;

        let err = y2.sub(&y)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(err < 5e-2, "exp_x(log_x(y)) roundtrip too large: {err}");
        Ok(())
    }

    #[test]
    fn log0_matches_log_map_at_origin_and_exp0_matches_exp_map_at_origin() -> Result<()> {
        let dev = &Device::Cpu;
        let dtype = DType::F32;
        let ball = CandlePoincareBall::new(1.0);

        let x = ball.project(&Tensor::randn(0f32, 0.1f32, (4, 3), dev)?.to_dtype(dtype)?)?;
        let v = Tensor::randn(0f32, 0.1f32, (4, 3), dev)?.to_dtype(dtype)?;
        let o = Tensor::zeros((1, 3), dtype, dev)?;
        let o = o.broadcast_as((4, 3))?;

        let a = ball.log0(&x)?;
        let b = ball.log_map(&o, &x)?;
        let err_log = a.sub(&b)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(err_log < 1e-3, "log0 != log_map(o,x): {err_log}");

        let a = ball.exp0(&v)?;
        let b = ball.exp_map(&o, &v)?;
        let err_exp = a.sub(&b)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(err_exp < 1e-3, "exp0 != exp_map(o,v): {err_exp}");

        Ok(())
    }

    #[test]
    fn parallel_transport_0_to_x_matches_lambda_ratio() -> Result<()> {
        let dev = &Device::Cpu;
        let dtype = DType::F32;
        let ball = CandlePoincareBall::new(1.0);

        let x = ball.project(&Tensor::randn(0f32, 0.1f32, (4, 3), dev)?.to_dtype(dtype)?)?;
        let v0 = Tensor::randn(0f32, 0.1f32, (4, 3), dev)?.to_dtype(dtype)?;

        let v = ball.parallel_transport_0_to_x(&x, &v0)?;

        let lambda_x = ball.lambda_x(&x)?; // [b,1]
        let scale = Tensor::full(2f32, (4, 1), dev)?.broadcast_div(&lambda_x)?;
        let v_ref = v0.broadcast_mul(&scale)?;

        let err = v.sub(&v_ref)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(err < 1e-4, "parallel transport scaling mismatch: {err}");
        Ok(())
    }

    #[test]
    fn bias_translate_ball_matches_mobius_add() -> Result<()> {
        let dev = &Device::Cpu;
        let dtype = DType::F32;
        let ball = CandlePoincareBall::new(1.0);

        let x = ball.project(&Tensor::randn(0f32, 0.1f32, (4, 3), dev)?.to_dtype(dtype)?)?;
        let b = ball.project(&Tensor::randn(0f32, 0.05f32, (4, 3), dev)?.to_dtype(dtype)?)?;

        let a = ball.mobius_add(&x, &b)?;
        let c = ball.bias_translate_ball(&x, &b)?;
        let err = a.sub(&c)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            err < 5e-3,
            "bias_translate_ball should match mobius_add (numerical tol), err={err}"
        );
        Ok(())
    }

    fn vec_l1(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }

    fn flatten2(v: Vec<Vec<f32>>) -> Vec<f32> {
        v.into_iter().flatten().collect()
    }

    #[test]
    fn candle_matches_hyp_on_core_ops_smoke() -> Result<()> {
        // Spec test: CandlePoincareBall must numerically match a reference Poincaré implementation
        // (project/mobius_add/distance/log0/exp0) in a safe regime.
        let dev = &Device::Cpu;
        let dtype = DType::F32;

        let c = 1.0f64;
        let ball = CandlePoincareBall::new(c);

        // Small, deterministic batch well inside ball.
        let x_v: Vec<Vec<f32>> = vec![
            vec![0.10, -0.05, 0.02],
            vec![0.03, 0.04, -0.01],
            vec![-0.08, 0.02, 0.05],
        ];
        let y_v: Vec<Vec<f32>> = vec![
            vec![0.02, 0.01, -0.03],
            vec![0.05, -0.02, 0.02],
            vec![0.01, 0.02, 0.03],
        ];

        let x = Tensor::from_vec(flatten2(x_v.clone()), (x_v.len(), x_v[0].len()), dev)?
            .to_dtype(dtype)?;
        let y = Tensor::from_vec(flatten2(y_v.clone()), (y_v.len(), y_v[0].len()), dev)?
            .to_dtype(dtype)?;

        let x_p = ball.project(&x)?;
        let y_p = ball.project(&y)?;

        // mobius_add
        let z_c = ball.mobius_add(&x_p, &y_p)?;
        let z_c = z_c.to_vec2::<f32>()?;

        let mut z_h = Vec::new();
        for (xr, yr) in x_v.iter().zip(y_v.iter()) {
            let xa = Array1::from_vec(xr.iter().map(|v| *v as f64).collect());
            let ya = Array1::from_vec(yr.iter().map(|v| *v as f64).collect());
            let out = mobius_add(c, &xa.view(), &ya.view());
            z_h.push(out.iter().map(|v| *v as f32).collect::<Vec<_>>());
        }

        let err_add = vec_l1(&flatten2(z_c), &flatten2(z_h));
        assert!(err_add < 5e-2, "mobius_add mismatch l1={err_add}");

        // distance
        let d_c = ball.distance(&x_p, &y_p)?.to_vec2::<f32>()?; // [b,1]
        let mut d_h = Vec::new();
        for (xr, yr) in x_v.iter().zip(y_v.iter()) {
            let xa = Array1::from_vec(xr.iter().map(|v| *v as f64).collect());
            let ya = Array1::from_vec(yr.iter().map(|v| *v as f64).collect());
            let d = distance(c, &xa.view(), &ya.view());
            d_h.push(vec![d as f32]);
        }
        let err_d = vec_l1(&flatten2(d_c), &flatten2(d_h));
        assert!(err_d < 1e-2, "distance mismatch l1={err_d}");

        // log0/exp0 roundtrip
        let v0 = ball.log0(&x_p)?;
        let x2 = ball.exp0(&v0)?;
        let err_rt = x2.sub(&x_p)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(err_rt < 1e-3, "exp0(log0(x)) mismatch err={err_rt}");

        Ok(())
    }

    fn clamp_norm_f64(mut v: Vec<f64>, max_norm: f64) -> Vec<f64> {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > max_norm && norm.is_finite() && norm > 0.0 {
            let s = max_norm / norm;
            for x in &mut v {
                *x *= s;
            }
        }
        v
    }

    fn to_tensor_row(dev: &Device, dtype: DType, v: &[f64]) -> Result<Tensor> {
        let vf: Vec<f32> = v.iter().map(|x| *x as f32).collect();
        Tensor::from_vec(vf, (1, v.len()), dev)?.to_dtype(dtype)
    }

    fn to_f64_vec(v: &[f32]) -> Array1<f64> {
        Array1::from_vec(v.iter().map(|x| *x as f64).collect())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_candle_matches_hyp_on_log_exp_and_bias(
            x0 in prop::collection::vec(-0.3f64..0.3f64, 3),
            y0 in prop::collection::vec(-0.3f64..0.3f64, 3),
            b0 in prop::collection::vec(-0.2f64..0.2f64, 3),
        ) {
            let dev = &Device::Cpu;
            let dtype = DType::F32;
            let c = 1.0f64;
            let ball = CandlePoincareBall::new(c);

            // Keep points safely inside ball.
            let x0 = clamp_norm_f64(x0, 0.35);
            let y0 = clamp_norm_f64(y0, 0.35);
            let b0 = clamp_norm_f64(b0, 0.25);

            let x_h = Array1::from_vec(x0.clone());
            let y_h = Array1::from_vec(y0.clone());
            let b_h = Array1::from_vec(b0.clone());

            // Reference: v = log_x(y), y' = exp_x(v)
            let v_h = log_map(c, &x_h.view(), &y_h.view());
            let y_h2 = exp_map(c, &x_h.view(), &v_h.view());

            // Candle: y' = exp_x(v_h) (using hyp's tangent vector)
            let x_t = to_tensor_row(dev, dtype, &x0).unwrap();
            let v_t = to_tensor_row(dev, dtype, &v_h.iter().copied().collect::<Vec<_>>()).unwrap();
            let y_c2 = ball.exp_map(&x_t, &v_t).unwrap().to_vec2::<f32>().unwrap();
            let y_h2f: Vec<f32> = y_h2.iter().map(|v| *v as f32).collect();

            let err_exp = vec_l1(&flatten2(y_c2), &y_h2f);
            prop_assert!(err_exp < 5e-2, "exp_map mismatch l1={err_exp}");

            // Bias translate: compare candle bias_translate_ball to hyp mobius_add (they should match).
            let b_t = to_tensor_row(dev, dtype, &b0).unwrap();
            let bt_c = ball.bias_translate_ball(&x_t, &b_t).unwrap().to_vec2::<f32>().unwrap();
            let bt_h = mobius_add(c, &x_h.view(), &b_h.view());
            let bt_hf: Vec<f32> = bt_h.iter().map(|v| *v as f32).collect();

            let err_bias = vec_l1(&flatten2(bt_c), &bt_hf);
            prop_assert!(err_bias < 5e-2, "bias_translate_ball mismatch l1={err_bias}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_candle_log_map_matches_hyp_and_roundtrips(
            x0 in prop::collection::vec(-0.4f64..0.4f64, 3),
            y0 in prop::collection::vec(-0.4f64..0.4f64, 3),
        ) {
            let dev = &Device::Cpu;
            let dtype = DType::F32;
            let c = 1.0f64;
            let ball = CandlePoincareBall::new(c);

            // A slightly “harder” regime than the prior test, but still safely inside.
            let x0 = clamp_norm_f64(x0, 0.55);
            let y0 = clamp_norm_f64(y0, 0.55);

            let x_h = Array1::from_vec(x0.clone());
            let y_h = Array1::from_vec(y0.clone());
            let x_t = to_tensor_row(dev, dtype, &x0).unwrap();
            let y_t = to_tensor_row(dev, dtype, &y0).unwrap();

            // Compare log_map directly.
            let v_h = log_map(c, &x_h.view(), &y_h.view());
            let v_c = ball.log_map(&x_t, &y_t).unwrap().to_vec2::<f32>().unwrap();
            let v_hf: Vec<f32> = v_h.iter().map(|v| *v as f32).collect();

            let err_log = vec_l1(&flatten2(v_c), &v_hf);
            prop_assert!(err_log < 8e-2, "log_map mismatch l1={err_log}");

            // Roundtrip using Candle's own tangent output: exp_x(log_x(y)) ≈ y.
            let v_c_t = ball.log_map(&x_t, &y_t).unwrap();
            let y2_c = ball.exp_map(&x_t, &v_c_t).unwrap();
            let y_err = y2_c.sub(&ball.project(&y_t).unwrap()).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            prop_assert!(y_err < 8e-2, "exp_x(log_x(y)) roundtrip too large: {y_err}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_candle_distance_matches_hyp(
            x0 in prop::collection::vec(-0.5f64..0.5f64, 3),
            y0 in prop::collection::vec(-0.5f64..0.5f64, 3),
        ) {
            let dev = &Device::Cpu;
            let dtype = DType::F32;
            let c = 1.0f64;
            let ball = CandlePoincareBall::new(c);

            // “Harder” regime: closer to boundary but still safe.
            let x0 = clamp_norm_f64(x0, 0.70);
            let y0 = clamp_norm_f64(y0, 0.70);

            let x_h = Array1::from_vec(x0.clone());
            let y_h = Array1::from_vec(y0.clone());
            let x_t = to_tensor_row(dev, dtype, &x0).unwrap();
            let y_t = to_tensor_row(dev, dtype, &y0).unwrap();

            let d_h = distance(c, &x_h.view(), &y_h.view()) as f32;
            let d_c = ball.distance(&x_t, &y_t).unwrap().to_vec2::<f32>().unwrap()[0][0];

            // Distance is a scalar; allow a bit of slack because Candle uses f32 ops.
            let err = (d_c - d_h).abs();
            prop_assert!(err < 5e-2, "distance mismatch |dc-dh|={err} (dc={d_c}, dh={d_h})");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        #[test]
        fn prop_candle_basepoint_log_exp_matches_hyp(
            p0 in prop::collection::vec(-0.4f64..0.4f64, 3),
            y0 in prop::collection::vec(-0.4f64..0.4f64, 3),
        ) {
            let dev = &Device::Cpu;
            let dtype = DType::F32;
            let c = 1.0f64;
            let ball = CandlePoincareBall::new(c);

            // Safe basepoint + target.
            let p0 = clamp_norm_f64(p0, 0.55);
            let y0 = clamp_norm_f64(y0, 0.55);

            let p_h = Array1::from_vec(p0.clone());
            let y_h = Array1::from_vec(y0.clone());
            let p_t = to_tensor_row(dev, dtype, &p0).unwrap();
            let y_t = to_tensor_row(dev, dtype, &y0).unwrap();

            // log_p(y)
            let v_h = log_map(c, &p_h.view(), &y_h.view());
            let v_c = ball.log_map(&p_t, &y_t).unwrap().to_vec2::<f32>().unwrap();
            let v_hf: Vec<f32> = v_h.iter().map(|v| *v as f32).collect();
            let err_log = vec_l1(&flatten2(v_c), &v_hf);
            prop_assert!(err_log < 8e-2, "log_map(basepoint) mismatch l1={err_log}");

            // exp_p(v_h) should land close to y (compare Candle vs reference).
            let v_t = to_tensor_row(dev, dtype, &v_h.iter().copied().collect::<Vec<_>>()).unwrap();
            let y_c2 = ball.exp_map(&p_t, &v_t).unwrap().to_vec2::<f32>().unwrap();
            let y_h2 = exp_map(c, &p_h.view(), &v_h.view());
            let y_h2f: Vec<f32> = y_h2.iter().map(|v| *v as f32).collect();
            let err_exp = vec_l1(&flatten2(y_c2), &y_h2f);
            prop_assert!(err_exp < 8e-2, "exp_map(basepoint) mismatch l1={err_exp}");
        }
    }

    #[test]
    fn hgcn_forward_with_basepoint_matches_hyp_reference_when_linear_is_identity() -> Result<()> {
        // Spec-ish test: when the linear layer is identity and bias is zero, HGCN reduces to:
        //   out = exp_p( adj @ log_p(x) )
        // We compare Candle implementation to a hyp-based reference computation.
        let dev = &Device::Cpu;
        let dtype = DType::F32;
        let c = 1.0f64;

        let n = 4usize;
        let d = 3usize;

        let ball = CandlePoincareBall::new(c);

        // Deterministic, safe points.
        let x = Tensor::from_vec(
            vec![
                0.10f32, -0.05, 0.02, 0.03, 0.04, -0.01, -0.08, 0.02, 0.05, 0.02, -0.01, 0.06,
            ],
            (n, d),
            dev,
        )?
        .to_dtype(dtype)?;
        let x = ball.project(&x)?;

        let p = Tensor::from_vec(vec![0.02f32, 0.01, -0.03], (1, d), dev)?.to_dtype(dtype)?;
        let p = ball.project(&p)?;

        // Small dense adjacency (not just identity) to exercise aggregation.
        let adj = Tensor::from_vec(
            vec![
                1.0f32, 0.2, 0.0, 0.0, 0.2, 1.0, 0.3, 0.0, 0.0, 0.3, 1.0, 0.4, 0.0, 0.0, 0.4, 1.0,
            ],
            (n, n),
            dev,
        )?
        .to_dtype(dtype)?;

        // Build an identity linear layer via VarMap, then run the Candle layer.
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, dev);
        let lin = candle_nn::linear(d, d, vb)?;
        varmap.set_one("weight", Tensor::eye(d, dtype, dev)?)?;
        varmap.set_one("bias", Tensor::zeros(d, dtype, dev)?)?;
        let layer = HGCNConv::new(lin, c);

        let y_c = layer.forward_with_basepoint(&x, &adj, &p)?; // [n,d]
        let y_c = y_c.to_vec2::<f32>()?;

        // Reference: compute in f64, then compare.
        let x_rows = x.to_vec2::<f32>()?;
        let p_row = p.broadcast_as((n, d))?.to_vec2::<f32>()?;

        // log_p(x) per row
        let mut log_px: Vec<Array1<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let pi = to_f64_vec(&p_row[i]);
            let xi = to_f64_vec(&x_rows[i]);
            let v = log_map(c, &pi.view(), &xi.view());
            log_px.push(v);
        }

        // agg_i = sum_j adj[i,j] * log_p(x_j)
        let adj_v = adj.to_vec2::<f32>()?;
        let mut agg: Vec<Array1<f64>> = Vec::with_capacity(n);
        for adj_i in adj_v.iter().take(n) {
            let mut a = Array1::<f64>::zeros(d);
            for (j, &w_ij) in adj_i.iter().enumerate().take(n) {
                let w = w_ij as f64;
                if w != 0.0 {
                    a = a + log_px[j].mapv(|t| t * w);
                }
            }
            agg.push(a);
        }

        // exp_p(agg_i)
        let mut y_h: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let pi = to_f64_vec(&p_row[i]);
            let yi = exp_map(c, &pi.view(), &agg[i].view());
            y_h.push(yi.iter().map(|v| *v as f32).collect());
        }

        let err = vec_l1(&flatten2(y_c), &flatten2(y_h));
        assert!(err < 1e-1, "hgcn(basepoint) mismatch l1={err}");
        Ok(())
    }
}
