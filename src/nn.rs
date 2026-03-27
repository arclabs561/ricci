//! Graph neural network layers on Burn tensors.

use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::hyperbolic::PoincareBall;

/// Graph Convolutional Network layer: `A_hat * X * W`.
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
pub struct HGCNConv<B: Backend> {
    linear: Linear<B>,
    ball: PoincareBall,
}

impl<B: Backend> HGCNConv<B> {
    /// Construct from a pre-built `Linear` and curvature parameter.
    pub fn new(linear: Linear<B>, c: f64) -> Self {
        Self {
            linear,
            ball: PoincareBall::new(c),
        }
    }

    /// Construct with a fresh `Linear(d, d)` and curvature parameter.
    pub fn init(d: usize, c: f64, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(d, d).init(device),
            ball: PoincareBall::new(c),
        }
    }

    /// Access the underlying linear layer.
    pub fn linear(&self) -> &Linear<B> {
        &self.linear
    }

    /// Access the Poincare ball geometry.
    pub fn ball(&self) -> &PoincareBall {
        &self.ball
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn dev() -> <B as Backend>::Device {
        <B as Backend>::Device::default()
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
}
