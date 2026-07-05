//! Exact segment max/min over edge lists, built from ops every backend
//! already has.
//!
//! Burn's `select_assign` scatters with `IndexingUpdateOp::Add` only, so
//! max/min segment statistics (the PNA aggregation family, Corso et al.,
//! NeurIPS 2020) have no direct op. The route here: the argmax choice is
//! piecewise constant in the values, so it carries no gradient — compute
//! winning edge indices on the host from a value snapshot, then read the
//! winners back out with a differentiable `gather`. The forward result is
//! exact, and the backward routes gradient only to each segment's winning
//! element, which is the almost-everywhere-correct gradient of max.

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

/// Segment maximum over an edge list, batched over queries.
///
/// `values`: `[Q, E, d]` per-edge values. `segments[e]` is the target
/// segment (node) of edge `e`, each `< num_segments`. Returns
/// `[Q, num_segments, d]` where entry `[q, n, k]` is the max of
/// `values[q, e, k]` over edges with `segments[e] == n`, or `fill` for
/// segments with no incoming edge.
///
/// Differentiable in `values`: gradient flows to each segment's argmax
/// element only.
pub fn scatter_max<B: Backend>(
    values: Tensor<B, 3>,
    segments: &[usize],
    num_segments: usize,
    fill: f32,
) -> Tensor<B, 3> {
    let [q, e, d] = values.dims();
    assert_eq!(e, segments.len(), "one segment id per edge");
    let host: Vec<f32> = values.clone().into_data().to_vec().unwrap();
    // Argmax edge per (query, segment, dim); -1 marks an empty segment.
    let mut arg = vec![-1i64; q * num_segments * d];
    for b in 0..q {
        for (edge, &n) in segments.iter().enumerate() {
            debug_assert!(n < num_segments);
            for k in 0..d {
                let v = host[(b * e + edge) * d + k];
                let slot = (b * num_segments + n) * d + k;
                if arg[slot] < 0 || v > host[(b * e + arg[slot] as usize) * d + k] {
                    arg[slot] = edge as i64;
                }
            }
        }
    }
    let device = values.device();
    let mask: Vec<f32> = arg.iter().map(|&a| if a < 0 { 0.0 } else { 1.0 }).collect();
    let idx: Vec<i64> = arg.into_iter().map(|a| a.max(0)).collect();
    let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(idx, [q * num_segments * d]), &device)
        .reshape([q, num_segments, d]);
    let mask = Tensor::<B, 3>::from_data(TensorData::new(mask, [q, num_segments, d]), &device);
    let gathered = values.gather(1, idx);
    gathered * mask.clone() + (mask * (-1.0) + 1.0) * fill
}

/// Segment maximum AND minimum in one pass: one value snapshot, one host
/// argmax/argmin sweep, two differentiable gathers. Empty segments come
/// back as `0.0` in both outputs (the boundary term callers fold in next
/// makes that the conventional choice). Prefer this over separate
/// [`scatter_max`] + [`scatter_min`] calls in hot loops: the snapshot of
/// `values` is the dominant cost and here it is paid once.
pub fn scatter_max_min<B: Backend>(
    values: Tensor<B, 3>,
    segments: &[usize],
    num_segments: usize,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let [q, e, d] = values.dims();
    assert_eq!(e, segments.len(), "one segment id per edge");
    let host: Vec<f32> = values.clone().into_data().to_vec().unwrap();
    let mut arg_max = vec![-1i64; q * num_segments * d];
    let mut arg_min = vec![-1i64; q * num_segments * d];
    for b in 0..q {
        for (edge, &n) in segments.iter().enumerate() {
            debug_assert!(n < num_segments);
            let row = (b * e + edge) * d;
            let slot0 = (b * num_segments + n) * d;
            for k in 0..d {
                let v = host[row + k];
                let slot = slot0 + k;
                if arg_max[slot] < 0 || v > host[(b * e + arg_max[slot] as usize) * d + k] {
                    arg_max[slot] = edge as i64;
                }
                if arg_min[slot] < 0 || v < host[(b * e + arg_min[slot] as usize) * d + k] {
                    arg_min[slot] = edge as i64;
                }
            }
        }
    }
    let device = values.device();
    let pick = |arg: Vec<i64>| {
        let mask: Vec<f32> = arg.iter().map(|&a| if a < 0 { 0.0 } else { 1.0 }).collect();
        let idx: Vec<i64> = arg.into_iter().map(|a| a.max(0)).collect();
        let idx =
            Tensor::<B, 1, Int>::from_data(TensorData::new(idx, [q * num_segments * d]), &device)
                .reshape([q, num_segments, d]);
        let mask = Tensor::<B, 3>::from_data(TensorData::new(mask, [q, num_segments, d]), &device);
        values.clone().gather(1, idx) * mask
    };
    (pick(arg_max), pick(arg_min))
}

/// Segment minimum over an edge list: `-scatter_max(-values)`, with the
/// same fill and gradient-routing semantics.
pub fn scatter_min<B: Backend>(
    values: Tensor<B, 3>,
    segments: &[usize],
    num_segments: usize,
    fill: f32,
) -> Tensor<B, 3> {
    scatter_max(values.neg(), segments, num_segments, -fill).neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn dev() -> <B as Backend>::Device {
        <B as Backend>::Device::default()
    }

    fn t3(data: Vec<f32>, shape: [usize; 3]) -> Tensor<B, 3> {
        Tensor::from_data(TensorData::new(data, shape), &dev())
    }

    /// Hand-checkable oracle: two queries, four edges into three segments,
    /// one segment left empty.
    #[test]
    fn max_matches_naive_loop() {
        // edges -> segments [0, 1, 0, 1]; segment 2 empty.
        let segs = [0usize, 1, 0, 1];
        let vals = t3(
            vec![
                1.0, -2.0, /* e0 */ 5.0, 0.5, /* e1 */ 3.0, -7.0, /* e2 */ 4.0,
                0.25, /* e3 */
                // second query: negated
                -1.0, 2.0, -5.0, -0.5, -3.0, 7.0, -4.0, -0.25,
            ],
            [2, 4, 2],
        );
        let out = scatter_max(vals, &segs, 3, 0.0);
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        // q0: seg0 = max(e0, e2) = [3, -2]; seg1 = max(e1, e3) = [5, 0.5];
        // seg2 empty = fill 0.
        assert_eq!(&v[0..6], &[3.0, -2.0, 5.0, 0.5, 0.0, 0.0]);
        // q1: seg0 = [-1, 7]; seg1 = [-4, -0.25].
        assert_eq!(&v[6..12], &[-1.0, 7.0, -4.0, -0.25, 0.0, 0.0]);
    }

    #[test]
    fn min_is_negated_max() {
        let segs = [0usize, 0, 1];
        let vals = t3(vec![1.0, 5.0, -2.0, 4.0, 3.0, 0.0], [1, 3, 2]);
        let out = scatter_min(vals, &segs, 3, 9.0);
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        assert_eq!(v, vec![-2.0, 4.0, 3.0, 0.0, 9.0, 9.0]);
    }

    /// The fused pass agrees with the two separate calls (fill 0).
    #[test]
    fn fused_matches_separate() {
        let segs = [0usize, 1, 0, 1];
        let vals = t3(
            vec![
                1.0, -2.0, 5.0, 0.5, 3.0, -7.0, 4.0, 0.25, -1.0, 2.0, -5.0, -0.5, -3.0, 7.0, -4.0,
                -0.25,
            ],
            [2, 4, 2],
        );
        let (mx, mn) = scatter_max_min(vals.clone(), &segs, 3);
        let mx_ref = scatter_max(vals.clone(), &segs, 3, 0.0);
        let mn_ref = scatter_min(vals, &segs, 3, 0.0);
        assert_eq!(
            mx.into_data().to_vec::<f32>().unwrap(),
            mx_ref.into_data().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            mn.into_data().to_vec::<f32>().unwrap(),
            mn_ref.into_data().to_vec::<f32>().unwrap()
        );
    }

    /// The backward routes gradient ONLY to each segment's winning
    /// element: d(sum of maxes)/d(values) is one-hot per segment per dim.
    #[test]
    fn gradient_reaches_argmax_only() {
        type A = Autodiff<NdArray<f32>>;
        let device = <A as Backend>::Device::default();
        let segs = [0usize, 0, 0];
        let vals = Tensor::<A, 3>::from_data(
            TensorData::new(vec![1.0f32, 9.0, 5.0, 2.0, 3.0, 4.0], [1, 3, 2]),
            &device,
        )
        .require_grad();
        let out = scatter_max(vals.clone(), &segs, 1, 0.0);
        let grads = out.sum().backward();
        let g: Vec<f32> = vals.grad(&grads).unwrap().into_data().to_vec().unwrap();
        // dim 0 winner: edge 1 (5.0); dim 1 winner: edge 0 (9.0).
        assert_eq!(g, vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }
}
