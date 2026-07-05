//! Graph neural network layers on Burn tensors.

use burn::module::{Ignored, Module, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

use crate::hyperbolic::PoincareBall;

/// Graph Convolutional Network layer: `A_hat * X * W`.
///
/// Derives [`Module`] so it can be embedded in a trainable model and optimized
/// by a Burn optimizer (see `examples/cora_node_classification.rs`).
#[derive(Module, Debug)]
pub struct GCNConv<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> GCNConv<B> {
    /// Construct from a pre-built `Linear`.
    pub fn new(linear: Linear<B>) -> Self {
        Self { linear }
    }

    /// Construct with a fresh `Linear(d_in, d_out)`.
    pub fn init(d_in: usize, d_out: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(d_in, d_out).init(device),
        }
    }

    /// Access the underlying linear layer.
    pub fn linear(&self) -> &Linear<B> {
        &self.linear
    }

    /// Forward: `adj @ linear(x)`.
    pub fn forward(&self, x: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear.forward(x);
        adj.matmul(x)
    }
}

/// Hyperbolic Graph Convolutional Network layer (H2H-GCN).
///
/// Operates entirely in the Poincare ball to minimize distortion:
/// 1. Map to tangent space (log map)
/// 2. Euclidean message passing (linear + adjacency matmul)
/// 3. Map back to ball (exp map)
///
/// # Example
///
/// ```
/// use burn::tensor::{backend::Backend, TensorData};
/// use burn_ndarray::NdArray;
/// use ricci::HGCNConv;
///
/// type B = NdArray<f32>;
/// let dev = <B as Backend>::Device::default();
///
/// let layer = HGCNConv::<B>::init(4, 1.0, &dev);
/// let x = burn::tensor::Tensor::<B, 2>::from_data(
///     TensorData::new(vec![0.01f32; 3 * 4], [3, 4]), &dev,
/// );
/// let adj = burn::tensor::Tensor::<B, 2>::from_data(
///     TensorData::new(vec![
///         1.0, 0.5, 0.0,
///         0.5, 1.0, 0.5,
///         0.0, 0.5, 1.0f32,
///     ], [3, 3]), &dev,
/// );
/// let y = layer.forward(x, adj);
/// assert_eq!(y.dims(), [3, 4]);
/// ```
///
/// Derives [`Module`] like [`GCNConv`], so it can be embedded in a trainable
/// model; the ball geometry is a constant carried via [`Ignored`] (it holds
/// no learnable parameters).
#[derive(Module, Debug)]
pub struct HGCNConv<B: Backend> {
    linear: Linear<B>,
    ball: Ignored<PoincareBall>,
}

impl<B: Backend> HGCNConv<B> {
    /// Construct from a pre-built `Linear` and curvature parameter.
    pub fn new(linear: Linear<B>, c: f64) -> Self {
        Self {
            linear,
            ball: Ignored(PoincareBall::new(c)),
        }
    }

    /// Construct with a fresh `Linear(d, d)` and curvature parameter.
    pub fn init(d: usize, c: f64, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(d, d).init(device),
            ball: Ignored(PoincareBall::new(c)),
        }
    }

    /// Access the underlying linear layer.
    pub fn linear(&self) -> &Linear<B> {
        &self.linear
    }

    /// Access the Poincare ball geometry.
    pub fn ball(&self) -> &PoincareBall {
        &self.ball.0
    }

    /// Forward pass using log/exp at the origin (no activation).
    pub fn forward(&self, x: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let x_tangent = self.ball.log0(x);
        let x_tangent = self.linear.forward(x_tangent);
        let aggregated = adj.matmul(x_tangent);
        self.ball.exp0(aggregated)
    }

    /// Forward pass with activation applied in tangent space (Chami et al. 2019).
    ///
    /// Full HGCN pattern: linear -> aggregate -> activate.
    /// Activation is applied via log0 -> act -> exp0 after aggregation.
    /// `ball_out` allows per-layer curvature change; pass `self.ball()` for same curvature.
    pub fn forward_act<F>(
        &self,
        x: Tensor<B, 2>,
        adj: Tensor<B, 2>,
        act: F,
        ball_out: &PoincareBall,
    ) -> Tensor<B, 2>
    where
        F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    {
        let h = self.forward(x, adj);
        self.ball.hyp_act(h, act, ball_out)
    }

    /// Forward pass using log/exp at an explicit basepoint `p`.
    ///
    /// `p` shape: `[1, d]` (global) or `[n, d]` (per-node).
    pub fn forward_with_basepoint(
        &self,
        x: Tensor<B, 2>,
        adj: Tensor<B, 2>,
        p: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [n, d] = x.dims();
        let [pn, _pd] = p.dims();
        let p = if pn == 1 { p.expand([n, d]) } else { p };

        let x_tangent = self.ball.log_map(p.clone(), x);
        let x_tangent = self.linear.forward(x_tangent);
        let aggregated = adj.matmul(x_tangent);
        self.ball.exp_map(p, aggregated)
    }

    /// Forward pass with basepoint and a shared bias `b0` in `T_0`.
    ///
    /// Transports `b0` to `T_p` before adding to the aggregated tangent vectors.
    pub fn forward_with_basepoint_and_bias(
        &self,
        x: Tensor<B, 2>,
        adj: Tensor<B, 2>,
        p: Tensor<B, 2>,
        b0: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [n, d] = x.dims();
        let [pn, _] = p.dims();
        let [bn, _] = b0.dims();
        let p = if pn == 1 { p.expand([n, d]) } else { p };
        let b0 = if bn == 1 { b0.expand([n, d]) } else { b0 };

        let x_tangent = self.ball.log_map(p.clone(), x);
        let x_tangent = self.linear.forward(x_tangent);
        let aggregated = adj.matmul(x_tangent);
        let bias_p = self.ball.parallel_transport_0_to_x(p.clone(), b0);
        let aggregated = aggregated + bias_p;
        self.ball.exp_map(p, aggregated)
    }

    /// Dense local-tangent aggregation (reference implementation).
    ///
    /// Aggregates in the tangent space at each center node (not a global basepoint),
    /// reducing distortion for relative distances.
    ///
    /// Cost: O(n^2 d) compute and memory. For small graphs and correctness reference.
    pub fn forward_local_dense(&self, x: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let [n, d] = x.dims();
        let x = self.ball.project(x);

        // All pairs: p[i,j,:] = x[i,:], y[i,j,:] = x[j,:]
        let p = x
            .clone()
            .reshape([n, 1, d])
            .expand([n, n, d])
            .reshape([n * n, d]);
        let y = x
            .clone()
            .reshape([1, n, d])
            .expand([n, n, d])
            .reshape([n * n, d]);

        // v_ij = log_{x_i}(x_j), then linear transform
        let v = self.ball.log_map(p, y);
        let v = self.linear.forward(v);
        let v = v.reshape([n, n, d]);

        // Weighted sum: sum_j adj[i,j] * v[i,j,:]
        let w = adj.reshape([n, n, 1]).expand([n, n, d]);
        let agg = (v * w).sum_dim(1).reshape([n, d]);

        self.ball.exp_map(x, agg)
    }
}

/// Relational Graph Convolutional Network layer (R-GCN):
/// `Σ_r A_hat_r X W_r + X W_self` (Schlichtkrull et al., ESWC 2018, Eq. 2).
///
/// One node set, many relations: each relation type gets its own learned
/// transform applied through its own (pre-normalized, as with [`GCNConv`])
/// adjacency, plus a self-loop transform. Directions count as distinct
/// relations: to message both ways along a relation, pass `A_r` and its
/// transpose as separate stack entries (the paper's canonical + inverse
/// convention). The paper normalizes by `1/|N_i^r|` per relation, or a
/// shared across-relation constant for link prediction, and notes fixed
/// normalization can degrade on high-degree hub nodes; the choice is the
/// caller's, encoded in the adjacencies.
///
/// [`with_bases`](Self::with_bases) shares the relation transforms through
/// a basis (their Eq. 3: `W_r = Σ_b a_rb V_b`), keeping parameters
/// sublinear in the relation count; load-bearing for KGs with hundreds of
/// relations. The paper's block-diagonal variant (Eq. 4) is not
/// implemented.
///
/// # Example
///
/// ```
/// use burn::tensor::{backend::Backend, TensorData};
/// use burn_ndarray::NdArray;
/// use ricci::RGCNConv;
///
/// type B = NdArray<f32>;
/// let dev = <B as Backend>::Device::default();
///
/// let layer = RGCNConv::<B>::init(4, 4, 2, &dev);
/// let x = burn::tensor::Tensor::<B, 2>::from_data(
///     TensorData::new(vec![0.1f32; 3 * 4], [3, 4]), &dev,
/// );
/// let a = burn::tensor::Tensor::<B, 2>::from_data(
///     TensorData::new(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0f32], [3, 3]), &dev,
/// );
/// let y = layer.forward(x, &[a.clone(), a.transpose()]);
/// assert_eq!(y.dims(), [3, 4]);
/// ```
#[derive(Module, Debug)]
pub struct RGCNConv<B: Backend> {
    /// Per-relation transforms (empty when basis-decomposed).
    rel: Vec<Linear<B>>,
    /// Shared bases `V_b`, `[num_bases, d_in, d_out]` (basis mode only).
    basis: Option<Param<Tensor<B, 3>>>,
    /// Per-relation basis coefficients `a_rb`, `[num_relations, num_bases]`.
    coef: Option<Param<Tensor<B, 2>>>,
    self_loop: Linear<B>,
}

impl<B: Backend> RGCNConv<B> {
    /// One full `Linear(d_in, d_out)` per relation, plus the self-loop.
    pub fn init(d_in: usize, d_out: usize, num_relations: usize, device: &B::Device) -> Self {
        Self {
            rel: (0..num_relations)
                .map(|_| LinearConfig::new(d_in, d_out).init(device))
                .collect(),
            basis: None,
            coef: None,
            self_loop: LinearConfig::new(d_in, d_out).init(device),
        }
    }

    /// Basis-decomposed relation transforms: `num_bases` shared `V_b`
    /// matrices with per-relation coefficients (Eq. 3).
    pub fn with_bases(
        d_in: usize,
        d_out: usize,
        num_relations: usize,
        num_bases: usize,
        device: &B::Device,
    ) -> Self {
        let std = (1.0 / d_in as f64).sqrt();
        let mk3 = Tensor::random(
            [num_bases, d_in, d_out],
            Distribution::Normal(0.0, std),
            device,
        )
        .require_grad();
        let mk2 = Tensor::random(
            [num_relations, num_bases],
            Distribution::Normal(0.0, (1.0 / num_bases as f64).sqrt()),
            device,
        )
        .require_grad();
        Self {
            rel: Vec::new(),
            basis: Some(Param::initialized(ParamId::new(), mk3)),
            coef: Some(Param::initialized(ParamId::new(), mk2)),
            self_loop: LinearConfig::new(d_in, d_out).init(device),
        }
    }

    /// Number of relation types this layer expects.
    pub fn num_relations(&self) -> usize {
        match &self.coef {
            Some(c) => c.val().dims()[0],
            None => self.rel.len(),
        }
    }

    /// Forward: `Σ_r adjs[r] @ (x @ W_r) + self_loop(x)`.
    ///
    /// # Panics
    /// Panics if `adjs.len()` differs from [`num_relations`](Self::num_relations).
    pub fn forward(&self, x: Tensor<B, 2>, adjs: &[Tensor<B, 2>]) -> Tensor<B, 2> {
        assert_eq!(
            adjs.len(),
            self.num_relations(),
            "one adjacency per relation"
        );
        let mut out = self.self_loop.forward(x.clone());
        if let (Some(basis), Some(coef)) = (&self.basis, &self.coef) {
            let [nb, d_in, d_out] = basis.val().dims();
            let flat = basis.val().reshape([nb, d_in * d_out]);
            let ws = coef.val().matmul(flat); // [R, d_in * d_out]
            for (r, adj) in adjs.iter().enumerate() {
                let w = ws
                    .clone()
                    .slice([r..r + 1, 0..d_in * d_out])
                    .reshape([d_in, d_out]);
                out = out + adj.clone().matmul(x.clone().matmul(w));
            }
        } else {
            for (lin, adj) in self.rel.iter().zip(adjs) {
                out = out + adj.clone().matmul(lin.forward(x.clone()));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn dev() -> <B as Backend>::Device {
        <B as Backend>::Device::default()
    }

    /// Swapping which adjacency carries which relation changes RGCN output
    /// (per-relation weights are real), while any relation-collapsing model
    /// is invariant to the swap by construction. The layer-level analogue
    /// of the direction-blindness failure of reciprocal-free DistMult.
    #[test]
    fn rgcn_distinguishes_relations() {
        let (n, d) = (4, 3);
        let layer = RGCNConv::<B>::init(d, d, 2, &dev());
        let x = Tensor::from_data(
            TensorData::new((0..n * d).map(|i| i as f32 / 5.0).collect(), [n, d]),
            &dev(),
        );
        let mut a_v = vec![0.0f32; n * n];
        a_v[1] = 1.0; // 0 -> 1 under relation A
        let mut b_v = vec![0.0f32; n * n];
        b_v[2] = 1.0; // 0 -> 2 under relation B
        let a = Tensor::from_data(TensorData::new(a_v, [n, n]), &dev());
        let b = Tensor::from_data(TensorData::new(b_v, [n, n]), &dev());

        let fwd: Vec<f32> = layer
            .forward(x.clone(), &[a.clone(), b.clone()])
            .into_data()
            .to_vec()
            .unwrap();
        let swp: Vec<f32> = layer.forward(x, &[b, a]).into_data().to_vec().unwrap();
        let diff: f32 = fwd.iter().zip(&swp).map(|(p, q)| (p - q).abs()).sum();
        assert!(diff > 1e-4, "relation swap must change the output: {diff}");
    }

    /// Empty relation stack degenerates to the self-loop transform alone.
    #[test]
    fn rgcn_zero_relations_is_self_loop() {
        let (n, d) = (3, 2);
        let layer = RGCNConv::<B>::init(d, d, 0, &dev());
        let x = Tensor::from_data(TensorData::new(vec![0.3f32; n * d], [n, d]), &dev());
        let y: Vec<f32> = layer.forward(x.clone(), &[]).into_data().to_vec().unwrap();
        let s: Vec<f32> = layer.self_loop.forward(x).into_data().to_vec().unwrap();
        assert_eq!(y, s);
    }

    /// Basis decomposition: forward shape holds, relation count reads from
    /// the coefficient table, and parameters stay sublinear in relations
    /// (6 relations share 2 bases).
    #[test]
    fn rgcn_basis_decomposition() {
        let (n, d, r, nb) = (4, 3, 6, 2);
        let layer = RGCNConv::<B>::with_bases(d, d, r, nb, &dev());
        assert_eq!(layer.num_relations(), r);
        let x = Tensor::from_data(TensorData::new(vec![0.2f32; n * d], [n, d]), &dev());
        let eye = {
            let mut v = vec![0.0f32; n * n];
            for i in 0..n {
                v[i * n + i] = 1.0;
            }
            Tensor::from_data(TensorData::new(v, [n, n]), &dev())
        };
        let adjs: Vec<_> = (0..r).map(|_| eye.clone()).collect();
        let y = layer.forward(x, &adjs);
        assert_eq!(y.dims(), [n, d]);
        // Shared bases: 2 * d * d + coefficients 6 * 2 < full 6 * d * d.
        assert!(nb * d * d + r * nb < r * d * d);
    }

    #[test]
    fn gcn_forward_shapes() {
        let n = 5;
        let d = 3;
        let layer = GCNConv::<B>::init(d, d, &dev());
        let x = Tensor::from_data(TensorData::new(vec![0.1f32; n * d], [n, d]), &dev());
        let adj = Tensor::from_data(TensorData::new(vec![1.0f32; n * n], [n, n]), &dev());
        let y = layer.forward(x, adj);
        assert_eq!(y.dims(), [n, d]);
    }

    #[test]
    fn hgcn_forward_shapes() {
        let n = 5;
        let d = 3;
        let layer = HGCNConv::<B>::init(d, 1.0, &dev());
        let x = Tensor::from_data(TensorData::new(vec![0.01f32; n * d], [n, d]), &dev());
        // Identity adjacency
        let mut adj_v = vec![0.0f32; n * n];
        for i in 0..n {
            adj_v[i * n + i] = 1.0;
        }
        let adj = Tensor::from_data(TensorData::new(adj_v, [n, n]), &dev());
        let y = layer.forward(x, adj);
        assert_eq!(y.dims(), [n, d]);
    }

    #[test]
    fn hgcn_with_basepoint_shapes() {
        let n = 6;
        let d = 4;
        let layer = HGCNConv::<B>::init(d, 1.0, &dev());
        let x = Tensor::from_data(TensorData::new(vec![0.01f32; n * d], [n, d]), &dev());
        let p = Tensor::from_data(TensorData::new(vec![0.0f32; d], [1, d]), &dev());
        let mut adj_v = vec![0.0f32; n * n];
        for i in 0..n {
            adj_v[i * n + i] = 1.0;
        }
        let adj = Tensor::from_data(TensorData::new(adj_v, [n, n]), &dev());
        let y = layer.forward_with_basepoint(x, adj, p);
        assert_eq!(y.dims(), [n, d]);
    }

    #[test]
    fn hgcn_local_dense_identity_adj_shapes() {
        let n = 4;
        let d = 3;
        let layer = HGCNConv::<B>::init(d, 1.0, &dev());
        let x = Tensor::<B, 2>::from_data(TensorData::new(vec![0.01f32; n * d], [n, d]), &dev());
        let mut adj_v = vec![0.0f32; n * n];
        for i in 0..n {
            adj_v[i * n + i] = 1.0;
        }
        let adj = Tensor::from_data(TensorData::new(adj_v, [n, n]), &dev());
        let y = layer.forward_local_dense(x, adj);
        assert_eq!(y.dims(), [n, d]);
    }

    #[test]
    fn hgcn_forward_act_produces_finite() {
        let n = 4;
        let d = 3;
        let layer = HGCNConv::<B>::init(d, 1.0, &dev());
        let x = Tensor::from_data(TensorData::new(vec![0.05f32; n * d], [n, d]), &dev());
        let mut adj_v = vec![0.0f32; n * n];
        for i in 0..n {
            adj_v[i * n + i] = 1.0;
        }
        let adj = Tensor::from_data(TensorData::new(adj_v, [n, n]), &dev());

        // ReLU activation via burn's clamp_min
        let ball = *layer.ball();
        let y = layer.forward_act(x, adj, |t| t.clamp_min(0.0), &ball);
        assert_eq!(y.dims(), [n, d]);
        let y_v = y.to_data().to_vec::<f32>().unwrap();
        assert!(y_v.iter().all(|v| v.is_finite()), "forward_act non-finite");
    }

    #[test]
    fn hgcn_forward_act_with_curvature_change() {
        let n = 3;
        let d = 3;
        let layer = HGCNConv::<B>::init(d, 1.0, &dev());
        let x = Tensor::from_data(
            TensorData::new(
                vec![0.05f32, -0.03, 0.02, 0.01, 0.04, -0.01, -0.02, 0.01, 0.03],
                [n, d],
            ),
            &dev(),
        );
        let mut adj_v = vec![0.0f32; n * n];
        for i in 0..n {
            adj_v[i * n + i] = 1.0;
        }
        let adj = Tensor::from_data(TensorData::new(adj_v, [n, n]), &dev());

        // Different output curvature (c_in=1.0, c_out=2.0)
        let ball_out = crate::PoincareBall::new(2.0);
        let y = layer.forward_act(x, adj, |t| t.clamp_min(0.0), &ball_out);
        assert_eq!(y.dims(), [n, d]);
        let y_v = y.to_data().to_vec::<f32>().unwrap();
        assert!(
            y_v.iter().all(|v| v.is_finite()),
            "curvature-change non-finite"
        );
    }

    #[test]
    fn hgcn_forward_with_bias_shapes_and_finite() {
        let n = 4;
        let d = 3;
        let layer = HGCNConv::<B>::init(d, 1.0, &dev());
        let x = Tensor::from_data(TensorData::new(vec![0.05f32; n * d], [n, d]), &dev());
        // Shared bias (broadcasted from [1, d])
        let b0 = Tensor::from_data(TensorData::new(vec![0.01f32, -0.01, 0.005], [1, d]), &dev());
        let p = Tensor::from_data(TensorData::new(vec![0.0f32; d], [1, d]), &dev());
        let mut adj_v = vec![0.0f32; n * n];
        for i in 0..n {
            adj_v[i * n + i] = 1.0;
        }
        let adj = Tensor::from_data(TensorData::new(adj_v, [n, n]), &dev());
        let y = layer.forward_with_basepoint_and_bias(x, adj, p, b0);
        assert_eq!(y.dims(), [n, d]);
        let y_v = y.to_data().to_vec::<f32>().unwrap();
        assert!(
            y_v.iter().all(|v| v.is_finite()),
            "forward_with_bias non-finite"
        );
    }

    #[test]
    fn hgcn_forward_with_bias_differs_from_without() {
        let n = 3;
        let d = 3;
        let layer = HGCNConv::<B>::init(d, 1.0, &dev());
        let x = Tensor::from_data(
            TensorData::new(
                vec![0.05f32, -0.03, 0.02, 0.01, 0.04, -0.01, -0.02, 0.01, 0.03],
                [n, d],
            ),
            &dev(),
        );
        let p = Tensor::from_data(TensorData::new(vec![0.0f32; d], [1, d]), &dev());
        let b0 = Tensor::from_data(TensorData::new(vec![0.1f32, -0.05, 0.02], [1, d]), &dev());
        let mut adj_v = vec![0.0f32; n * n];
        for i in 0..n {
            adj_v[i * n + i] = 1.0;
        }
        let adj = Tensor::from_data(TensorData::new(adj_v, [n, n]), &dev());

        let y_no_bias = layer.forward_with_basepoint(x.clone(), adj.clone(), p.clone());
        let y_with_bias = layer.forward_with_basepoint_and_bias(x, adj, p, b0);

        let a = y_no_bias.to_data().to_vec::<f32>().unwrap();
        let b = y_with_bias.to_data().to_vec::<f32>().unwrap();
        // Bias should cause a meaningful difference.
        let diff: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1e-3, "bias should change output, diff={diff}");
    }

    #[test]
    fn gcn_non_square_dimensions() {
        let n = 4;
        let d_in = 5;
        let d_out = 3;
        let layer = GCNConv::<B>::init(d_in, d_out, &dev());
        let x = Tensor::from_data(TensorData::new(vec![0.1f32; n * d_in], [n, d_in]), &dev());
        let adj = Tensor::from_data(TensorData::new(vec![1.0f32; n * n], [n, n]), &dev());
        let y = layer.forward(x, adj);
        assert_eq!(y.dims(), [n, d_out]);
    }

    #[test]
    fn hgcn_with_real_adjacency() {
        // Non-identity adjacency: a simple 4-node chain graph.
        let n = 4;
        let d = 3;
        let layer = HGCNConv::<B>::init(d, 1.0, &dev());
        let x = Tensor::from_data(
            TensorData::new(
                vec![
                    0.05f32, -0.03, 0.02, // node 0
                    0.01, 0.04, -0.01, // node 1
                    -0.02, 0.01, 0.03, // node 2
                    0.03, -0.02, 0.01, // node 3
                ],
                [n, d],
            ),
            &dev(),
        );
        // Normalized adjacency for chain: 0-1-2-3
        // A_hat = D^{-1/2} (A + I) D^{-1/2}, approximated as row-normalized
        #[rustfmt::skip]
        let adj_v = vec![
            0.5, 0.5, 0.0, 0.0,
            0.33, 0.33, 0.33, 0.0,
            0.0, 0.33, 0.33, 0.33,
            0.0, 0.0, 0.5, 0.5,
        ];
        let adj = Tensor::from_data(TensorData::new(adj_v, [n, n]), &dev());

        let y = layer.forward(x, adj);
        assert_eq!(y.dims(), [n, d]);
        let y_v = y.to_data().to_vec::<f32>().unwrap();
        assert!(
            y_v.iter().all(|v| v.is_finite()),
            "chain graph forward non-finite"
        );

        // Interior nodes (1,2) should differ from boundary nodes (0,3)
        // because they aggregate from more neighbors.
        let row0: Vec<f32> = y_v[0..d].to_vec();
        let row1: Vec<f32> = y_v[d..2 * d].to_vec();
        let diff: f32 = row0.iter().zip(&row1).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-4,
            "boundary and interior nodes should differ, diff={diff}"
        );
    }

    #[test]
    fn two_layer_hgcn_pipeline() {
        // Chain two HGCN layers: layer1 -> act -> layer2, simulating a real model.
        let n = 4;
        let d = 3;
        let layer1 = HGCNConv::<B>::init(d, 1.0, &dev());
        let layer2 = HGCNConv::<B>::init(d, 1.0, &dev());

        let x = Tensor::from_data(TensorData::new(vec![0.05f32; n * d], [n, d]), &dev());
        let mut adj_v = vec![0.0f32; n * n];
        for i in 0..n {
            adj_v[i * n + i] = 1.0;
            if i + 1 < n {
                adj_v[i * n + i + 1] = 0.5;
                adj_v[(i + 1) * n + i] = 0.5;
            }
        }
        let adj = Tensor::from_data(TensorData::new(adj_v, [n, n]), &dev());

        let ball = *layer1.ball();
        // Layer 1 with ReLU activation
        let h = layer1.forward_act(x, adj.clone(), |t| t.clamp_min(0.0), &ball);
        // Layer 2 (no activation on final layer, per HGCN convention)
        let y = layer2.forward(h, adj);

        assert_eq!(y.dims(), [n, d]);
        let y_v = y.to_data().to_vec::<f32>().unwrap();
        assert!(
            y_v.iter().all(|v| v.is_finite()),
            "two-layer pipeline non-finite"
        );
    }

    #[test]
    fn forward_local_dense_and_forward_agree_on_identity_adj() {
        // With identity adjacency, both forward paths should produce similar results
        // because each node only aggregates from itself.
        let n = 3;
        let d = 3;
        let layer = HGCNConv::<B>::init(d, 1.0, &dev());
        let x = Tensor::<B, 2>::from_data(
            TensorData::new(
                vec![0.05f32, -0.03, 0.02, 0.01, 0.04, -0.01, -0.02, 0.01, 0.03],
                [n, d],
            ),
            &dev(),
        );
        let mut adj_v = vec![0.0f32; n * n];
        for i in 0..n {
            adj_v[i * n + i] = 1.0;
        }
        let adj = Tensor::from_data(TensorData::new(adj_v, [n, n]), &dev());

        // Both should apply: log -> linear -> adj(=I) @ x -> exp, so results match.
        let y_origin = layer.forward(x.clone(), adj.clone());
        let y_local = layer.forward_local_dense(x, adj);

        let y_o = y_origin.to_data().to_vec::<f32>().unwrap();
        let y_l = y_local.to_data().to_vec::<f32>().unwrap();

        fn l1(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
        }

        assert!(
            y_o.iter().all(|v| v.is_finite()),
            "forward produced non-finite"
        );
        assert!(
            y_l.iter().all(|v| v.is_finite()),
            "forward_local_dense produced non-finite"
        );
        // With identity adj, forward (origin basepoint) and forward_local_dense
        // (per-node basepoint) use different tangent spaces but should be close
        // for small inputs.
        assert!(
            l1(&y_o, &y_l) < 0.5,
            "forward vs forward_local_dense diverged: l1={}",
            l1(&y_o, &y_l)
        );
    }

    /// Helper: check all rows of a [n, d] tensor are inside the ball.
    fn assert_inside_ball(t: &Tensor<B, 2>, ball: &crate::PoincareBall, label: &str) {
        let [n, d] = t.dims();
        let v = t.to_data().to_vec::<f32>().unwrap();
        let max = ball.max_norm();
        for i in 0..n {
            let row = &v[i * d..(i + 1) * d];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                norm <= max + 1e-4,
                "{label} row {i} outside ball: norm={norm} max={max}"
            );
            assert!(
                row.iter().all(|x| x.is_finite()),
                "{label} row {i} non-finite"
            );
        }
    }

    #[test]
    fn all_forward_variants_stay_inside_ball() {
        let n = 4;
        let d = 3;
        let ball = crate::PoincareBall::new(1.0);
        let layer = HGCNConv::<B>::init(d, 1.0, &dev());
        let x = Tensor::from_data(
            TensorData::new(
                vec![
                    0.3f32, -0.2, 0.1, 0.1, 0.2, -0.3, -0.1, 0.3, 0.2, 0.2, -0.1, 0.1,
                ],
                [n, d],
            ),
            &dev(),
        );
        #[rustfmt::skip]
        let adj_v = vec![
            0.5, 0.5, 0.0, 0.0,
            0.33, 0.34, 0.33, 0.0,
            0.0, 0.33, 0.34, 0.33,
            0.0, 0.0, 0.5, 0.5f32,
        ];
        let adj = Tensor::from_data(TensorData::new(adj_v, [n, n]), &dev());
        let p = Tensor::from_data(TensorData::new(vec![0.05f32, -0.02, 0.01], [1, d]), &dev());
        let b0 = Tensor::from_data(TensorData::new(vec![0.02f32, -0.01, 0.005], [1, d]), &dev());

        let y1 = layer.forward(x.clone(), adj.clone());
        assert_inside_ball(&y1, &ball, "forward");

        let y2 = layer.forward_with_basepoint(x.clone(), adj.clone(), p.clone());
        assert_inside_ball(&y2, &ball, "forward_with_basepoint");

        let y3 =
            layer.forward_with_basepoint_and_bias(x.clone(), adj.clone(), p.clone(), b0.clone());
        assert_inside_ball(&y3, &ball, "forward_with_basepoint_and_bias");

        let y4 = layer.forward_local_dense(x.clone(), adj.clone());
        assert_inside_ball(&y4, &ball, "forward_local_dense");

        let y5 = layer.forward_act(x, adj, |t| t.clamp_min(0.0), &ball);
        assert_inside_ball(&y5, &ball, "forward_act");
    }

    #[test]
    fn per_node_basepoint() {
        // Per-node basepoints [n, d] (not broadcasted from [1, d]).
        let n = 3;
        let d = 3;
        let ball = crate::PoincareBall::new(1.0);
        let layer = HGCNConv::<B>::init(d, 1.0, &dev());
        let x = Tensor::from_data(
            TensorData::new(
                vec![0.05f32, -0.03, 0.02, 0.01, 0.04, -0.01, -0.02, 0.01, 0.03],
                [n, d],
            ),
            &dev(),
        );
        // Each node has its own basepoint.
        let p = Tensor::from_data(
            TensorData::new(
                vec![0.02f32, 0.01, -0.01, -0.01, 0.02, 0.01, 0.01, -0.02, 0.02],
                [n, d],
            ),
            &dev(),
        );
        let mut adj_v = vec![0.0f32; n * n];
        for i in 0..n {
            adj_v[i * n + i] = 1.0;
        }
        let adj = Tensor::from_data(TensorData::new(adj_v, [n, n]), &dev());

        let y = layer.forward_with_basepoint(x, adj, p);
        assert_eq!(y.dims(), [n, d]);
        assert_inside_ball(&y, &ball, "per_node_basepoint");
    }
}
