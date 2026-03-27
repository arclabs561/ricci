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

    /// Forward pass using log/exp at the origin.
    pub fn forward(&self, x: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let x_tangent = self.ball.log0(x);
        let x_tangent = self.linear.forward(x_tangent);
        let aggregated = adj.matmul(x_tangent);
        self.ball.exp0(aggregated)
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
