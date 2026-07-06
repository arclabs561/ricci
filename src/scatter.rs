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
use burn::tensor::{Int, Tensor as BurnTensor, TensorData};
#[cfg(any(feature = "wgpu", feature = "metal"))]
use burn::tensor::{Shape, TensorPrimitive};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[cfg(any(feature = "wgpu", feature = "metal"))]
use burn::backend::wgpu::{
    BoolElement, CubeBackend, CubeDim, CubeTensor, FloatElement, IntElement, WgpuRuntime,
};
#[cfg(any(feature = "wgpu", feature = "metal"))]
use burn::cubecl::{calculate_cube_count_elemwise, prelude::*};

static SCATTER_CALLS: AtomicU64 = AtomicU64::new(0);
static SCATTER_ELEMENTS: AtomicU64 = AtomicU64::new(0);
static SCATTER_SNAPSHOT_NANOS: AtomicU64 = AtomicU64::new(0);
static SCATTER_SCAN_NANOS: AtomicU64 = AtomicU64::new(0);
static SCATTER_GATHER_NANOS: AtomicU64 = AtomicU64::new(0);

/// Runtime counters for the exact segment max/min host fallback.
///
/// `scatter_max_min` is exact and differentiable through `gather`, but today
/// it snapshots edge messages to the host to choose max/min winners. These
/// counters make that cost visible in examples and training harnesses.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScatterMetrics {
    /// Number of segment max/min helper calls.
    pub calls: u64,
    /// Number of `[query, edge, dim]` values inspected on the host.
    pub elements: u64,
    /// Nanoseconds spent copying tensor values from device to host.
    pub snapshot_nanos: u64,
    /// Nanoseconds spent choosing argmax/argmin edge indices on the host.
    pub scan_nanos: u64,
    /// Nanoseconds spent building index/mask tensors and gathering winners.
    pub gather_nanos: u64,
}

impl ScatterMetrics {
    /// Returns `self - earlier`, saturating at zero per field.
    pub fn saturating_sub(self, earlier: Self) -> Self {
        Self {
            calls: self.calls.saturating_sub(earlier.calls),
            elements: self.elements.saturating_sub(earlier.elements),
            snapshot_nanos: self.snapshot_nanos.saturating_sub(earlier.snapshot_nanos),
            scan_nanos: self.scan_nanos.saturating_sub(earlier.scan_nanos),
            gather_nanos: self.gather_nanos.saturating_sub(earlier.gather_nanos),
        }
    }

    /// Returns total measured seconds across snapshot, scan, and gather.
    pub fn total_seconds(self) -> f64 {
        nanos_to_seconds(self.snapshot_nanos + self.scan_nanos + self.gather_nanos)
    }
}

/// Reads the process-wide scatter fallback counters.
pub fn scatter_metrics() -> ScatterMetrics {
    ScatterMetrics {
        calls: SCATTER_CALLS.load(Ordering::Relaxed),
        elements: SCATTER_ELEMENTS.load(Ordering::Relaxed),
        snapshot_nanos: SCATTER_SNAPSHOT_NANOS.load(Ordering::Relaxed),
        scan_nanos: SCATTER_SCAN_NANOS.load(Ordering::Relaxed),
        gather_nanos: SCATTER_GATHER_NANOS.load(Ordering::Relaxed),
    }
}

/// Resets the process-wide scatter fallback counters.
pub fn reset_scatter_metrics() {
    SCATTER_CALLS.store(0, Ordering::Relaxed);
    SCATTER_ELEMENTS.store(0, Ordering::Relaxed);
    SCATTER_SNAPSHOT_NANOS.store(0, Ordering::Relaxed);
    SCATTER_SCAN_NANOS.store(0, Ordering::Relaxed);
    SCATTER_GATHER_NANOS.store(0, Ordering::Relaxed);
}

fn record_scatter_metrics(elements: usize, snapshot: Duration, scan: Duration, gather: Duration) {
    SCATTER_CALLS.fetch_add(1, Ordering::Relaxed);
    SCATTER_ELEMENTS.fetch_add(elements as u64, Ordering::Relaxed);
    SCATTER_SNAPSHOT_NANOS.fetch_add(duration_nanos(snapshot), Ordering::Relaxed);
    SCATTER_SCAN_NANOS.fetch_add(duration_nanos(scan), Ordering::Relaxed);
    SCATTER_GATHER_NANOS.fetch_add(duration_nanos(gather), Ordering::Relaxed);
}

fn duration_nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn nanos_to_seconds(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000_000.0
}

#[cfg(any(feature = "wgpu", feature = "metal"))]
#[cube(launch)]
fn segment_argmax_min_kernel<F: Float, I: Numeric>(
    values: &Tensor<F>,
    edge_order: &Tensor<I>,
    offsets: &Tensor<I>,
    idx_max: &mut Tensor<I>,
    idx_min: &mut Tensor<I>,
    mask: &mut Tensor<F>,
    #[define(F, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= idx_max.len() {
        terminate!();
    }

    let d = values.shape(2);
    let num_segments = offsets.len() - 1;
    let k = ABSOLUTE_POS % d;
    let segment = (ABSOLUTE_POS / d) % num_segments;
    let query = ABSOLUTE_POS / (d * num_segments);

    let start = usize::cast_from(offsets[segment]);
    let end = usize::cast_from(offsets[segment + 1]);
    if start == end {
        idx_max[ABSOLUTE_POS] = I::from_int(0);
        idx_min[ABSOLUTE_POS] = I::from_int(0);
        mask[ABSOLUTE_POS] = F::from_int(0);
        terminate!();
    }

    let first_edge = usize::cast_from(edge_order[start]);
    let mut best_max_edge = first_edge;
    let mut best_min_edge = first_edge;
    let first_offset =
        query * values.stride(0) + first_edge * values.stride(1) + k * values.stride(2);
    let mut best_max = values[first_offset];
    let mut best_min = values[first_offset];

    for pos in start + 1..end {
        let edge = usize::cast_from(edge_order[pos]);
        let offset = query * values.stride(0) + edge * values.stride(1) + k * values.stride(2);
        let candidate = values[offset];
        if candidate > best_max || (candidate == best_max && edge < best_max_edge) {
            best_max = candidate;
            best_max_edge = edge;
        }
        if candidate < best_min || (candidate == best_min && edge < best_min_edge) {
            best_min = candidate;
            best_min_edge = edge;
        }
    }

    idx_max[ABSOLUTE_POS] = I::cast_from(best_max_edge);
    idx_min[ABSOLUTE_POS] = I::cast_from(best_min_edge);
    mask[ABSOLUTE_POS] = F::from_int(1);
}

#[cfg(any(feature = "wgpu", feature = "metal"))]
type WgpuBackend<F, I, BT> = CubeBackend<WgpuRuntime, F, I, BT>;
#[cfg(any(feature = "wgpu", feature = "metal"))]
type WgpuFloatTensor<F, I, BT, const D: usize> = BurnTensor<WgpuBackend<F, I, BT>, D>;
#[cfg(any(feature = "wgpu", feature = "metal"))]
type WgpuIntTensor<F, I, BT, const D: usize> = BurnTensor<WgpuBackend<F, I, BT>, D, Int>;
#[cfg(any(feature = "wgpu", feature = "metal"))]
type WgpuScatterIndices<F, I, BT> = (
    WgpuIntTensor<F, I, BT, 3>,
    WgpuIntTensor<F, I, BT, 3>,
    WgpuFloatTensor<F, I, BT, 3>,
);
#[cfg(any(feature = "wgpu", feature = "metal"))]
type WgpuScatterValues<F, I, BT> = (WgpuFloatTensor<F, I, BT, 3>, WgpuFloatTensor<F, I, BT, 3>);

/// Computes exact segment max/min winner indices on WGPU/Metal.
///
/// This returns argmax indices, argmin indices, and an occupancy mask, all
/// shaped `[Q, num_segments, d]`. It does not define a custom backward pass:
/// callers should gather from the original floating tensor with the returned
/// indices so Burn routes gradients to the winning edge values.
#[cfg(any(feature = "wgpu", feature = "metal"))]
pub fn scatter_max_min_indices_wgpu<F, I, BT>(
    values: WgpuFloatTensor<F, I, BT, 3>,
    segments: &[usize],
    num_segments: usize,
) -> WgpuScatterIndices<F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let [q, e, d] = values.dims();
    assert_eq!(e, segments.len(), "one segment id per edge");
    let (edge_order, offsets) = segment_csr(segments, num_segments);
    let device = values.device();
    let edge_order = BurnTensor::<WgpuBackend<F, I, BT>, 1, Int>::from_data(
        TensorData::new(edge_order, [e]),
        &device,
    );
    let offsets = BurnTensor::<WgpuBackend<F, I, BT>, 1, Int>::from_data(
        TensorData::new(offsets, [num_segments + 1]),
        &device,
    );

    let values = values.into_primitive().tensor();
    let edge_order = edge_order.into_primitive();
    let offsets = offsets.into_primitive();
    let shape = Shape::new([q, num_segments, d]);
    let idx_max = empty_like(&values, shape.clone(), I::dtype());
    let idx_min = empty_like(&values, shape.clone(), I::dtype());
    let mask = empty_like(&values, shape, F::dtype());
    let cube_dim = CubeDim::new(&values.client, q * num_segments * d);
    let cube_count = calculate_cube_count_elemwise(&values.client, q * num_segments * d, cube_dim);

    segment_argmax_min_kernel::launch::<WgpuRuntime>(
        &values.client,
        cube_count,
        cube_dim,
        values.as_tensor_arg(1),
        edge_order.as_tensor_arg(1),
        offsets.as_tensor_arg(1),
        idx_max.as_tensor_arg(1),
        idx_min.as_tensor_arg(1),
        mask.as_tensor_arg(1),
        [values.dtype.into(), edge_order.dtype.into()],
    )
    .expect("segment max/min kernel should launch");

    (
        BurnTensor::from_primitive(idx_max),
        BurnTensor::from_primitive(idx_min),
        BurnTensor::from_primitive(TensorPrimitive::Float(mask)),
    )
}

/// Exact segment max/min on WGPU/Metal without a host value snapshot.
///
/// Segment ids are still grouped on the host, but edge values stay on device.
/// The returned values are produced with Burn `gather`, so gradients follow
/// the same winner-only path as [`scatter_max_min`].
#[cfg(any(feature = "wgpu", feature = "metal"))]
pub fn scatter_max_min_wgpu<F, I, BT>(
    values: WgpuFloatTensor<F, I, BT, 3>,
    segments: &[usize],
    num_segments: usize,
) -> WgpuScatterValues<F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let (idx_max, idx_min, mask) =
        scatter_max_min_indices_wgpu(values.clone(), segments, num_segments);
    (
        values.clone().gather(1, idx_max) * mask.clone(),
        values.gather(1, idx_min) * mask,
    )
}

#[cfg(any(feature = "wgpu", feature = "metal"))]
fn empty_like(
    values: &CubeTensor<WgpuRuntime>,
    shape: Shape,
    dtype: burn::tensor::DType,
) -> CubeTensor<WgpuRuntime> {
    let handle = values.client.empty(shape.num_elements() * dtype.size());
    CubeTensor::new_contiguous(
        values.client.clone(),
        values.device.clone(),
        shape,
        handle,
        dtype,
    )
}

#[cfg(any(feature = "wgpu", feature = "metal"))]
fn segment_csr(segments: &[usize], num_segments: usize) -> (Vec<i64>, Vec<i64>) {
    assert!(
        segments.len() <= i32::MAX as usize,
        "too many edges for WGPU i32 indexing"
    );
    let mut counts = vec![0usize; num_segments];
    for &segment in segments {
        assert!(segment < num_segments, "segment id out of bounds");
        counts[segment] += 1;
    }
    let mut offsets = vec![0usize; num_segments + 1];
    for segment in 0..num_segments {
        offsets[segment + 1] = offsets[segment] + counts[segment];
    }
    let mut cursor = offsets[..num_segments].to_vec();
    let mut edge_order = vec![0i64; segments.len()];
    for (edge, &segment) in segments.iter().enumerate() {
        let pos = cursor[segment];
        edge_order[pos] = edge as i64;
        cursor[segment] += 1;
    }
    let offsets = offsets.into_iter().map(|offset| offset as i64).collect();
    (edge_order, offsets)
}

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
    values: BurnTensor<B, 3>,
    segments: &[usize],
    num_segments: usize,
    fill: f32,
) -> BurnTensor<B, 3> {
    let [q, e, d] = values.dims();
    assert_eq!(e, segments.len(), "one segment id per edge");
    let snapshot_start = Instant::now();
    let host: Vec<f32> = values.clone().into_data().to_vec().unwrap();
    let snapshot = snapshot_start.elapsed();
    let scan_start = Instant::now();
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
    let scan = scan_start.elapsed();
    let gather_start = Instant::now();
    let device = values.device();
    let mask: Vec<f32> = arg.iter().map(|&a| if a < 0 { 0.0 } else { 1.0 }).collect();
    let idx: Vec<i64> = arg.into_iter().map(|a| a.max(0)).collect();
    let idx =
        BurnTensor::<B, 1, Int>::from_data(TensorData::new(idx, [q * num_segments * d]), &device)
            .reshape([q, num_segments, d]);
    let mask = BurnTensor::<B, 3>::from_data(TensorData::new(mask, [q, num_segments, d]), &device);
    let gathered = values.gather(1, idx);
    let out = gathered * mask.clone() + (mask * (-1.0) + 1.0) * fill;
    record_scatter_metrics(q * e * d, snapshot, scan, gather_start.elapsed());
    out
}

/// Segment maximum AND minimum in one pass: one value snapshot, one host
/// argmax/argmin sweep, two differentiable gathers. Empty segments come
/// back as `0.0` in both outputs (the boundary term callers fold in next
/// makes that the conventional choice). Prefer this over separate
/// [`scatter_max`] + [`scatter_min`] calls in hot loops: the snapshot of
/// `values` is the dominant cost and here it is paid once.
pub fn scatter_max_min<B: Backend>(
    values: BurnTensor<B, 3>,
    segments: &[usize],
    num_segments: usize,
) -> (BurnTensor<B, 3>, BurnTensor<B, 3>) {
    let [q, e, d] = values.dims();
    assert_eq!(e, segments.len(), "one segment id per edge");
    let snapshot_start = Instant::now();
    let host: Vec<f32> = values.clone().into_data().to_vec().unwrap();
    let snapshot = snapshot_start.elapsed();
    let scan_start = Instant::now();
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
    let scan = scan_start.elapsed();
    let gather_start = Instant::now();
    let device = values.device();
    let pick = |arg: Vec<i64>| {
        let mask: Vec<f32> = arg.iter().map(|&a| if a < 0 { 0.0 } else { 1.0 }).collect();
        let idx: Vec<i64> = arg.into_iter().map(|a| a.max(0)).collect();
        let idx = BurnTensor::<B, 1, Int>::from_data(
            TensorData::new(idx, [q * num_segments * d]),
            &device,
        )
        .reshape([q, num_segments, d]);
        let mask =
            BurnTensor::<B, 3>::from_data(TensorData::new(mask, [q, num_segments, d]), &device);
        values.clone().gather(1, idx) * mask
    };
    let out = (pick(arg_max), pick(arg_min));
    record_scatter_metrics(q * e * d, snapshot, scan, gather_start.elapsed());
    out
}

/// Segment minimum over an edge list: `-scatter_max(-values)`, with the
/// same fill and gradient-routing semantics.
pub fn scatter_min<B: Backend>(
    values: BurnTensor<B, 3>,
    segments: &[usize],
    num_segments: usize,
    fill: f32,
) -> BurnTensor<B, 3> {
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

    fn t3(data: Vec<f32>, shape: [usize; 3]) -> BurnTensor<B, 3> {
        BurnTensor::from_data(TensorData::new(data, shape), &dev())
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
        let vals = BurnTensor::<A, 3>::from_data(
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

    #[cfg(any(feature = "wgpu", feature = "metal"))]
    #[test]
    fn wgpu_matches_host_fused_max_min() {
        type G = burn::backend::Wgpu<f32, i32>;
        let device = <G as Backend>::Device::default();
        let segs = [0usize, 1, 0, 1];
        let vals = BurnTensor::<G, 3>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, 5.0, 0.5, 5.0, -7.0, 4.0, 0.5, -1.0, 2.0, -5.0, -0.5, -3.0, 7.0,
                    -4.0, -0.5,
                ],
                [2, 4, 2],
            ),
            &device,
        );
        let (mx, mn) = scatter_max_min_wgpu(vals.clone(), &segs, 3);
        let (mx_ref, mn_ref) = scatter_max_min(vals, &segs, 3);
        assert_eq!(
            mx.into_data().to_vec::<f32>().unwrap(),
            mx_ref.into_data().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            mn.into_data().to_vec::<f32>().unwrap(),
            mn_ref.into_data().to_vec::<f32>().unwrap()
        );
    }
}
