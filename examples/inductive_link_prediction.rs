//! Inductive link prediction on the GraIL FB15k-237 v1 split: train on one
//! graph, predict on a disjoint graph with entirely new entities.
//!
//! An NBFNet-shaped model (Zhu et al., NeurIPS 2021) built from
//! [`NBFConv`]'s edge-list forward: learned per-relation query embeddings
//! seed the indicator (labeling trick), six conditional message-passing
//! layers with per-layer relation representations propagate pair states,
//! and a two-layer MLP scores tails. Nothing entity-specific is learned,
//! which is why the trained model runs unchanged on `fb237_v1_ind`'s new
//! entity vocabulary (relations are shared across the split pair).
//!
//! Protocol: GraIL's 50-negative filtered Hits@10 over both query
//! directions (references on this split: GraIL 0.642, NBFNet 0.834), plus
//! the stricter full-entity ranking metrics, which the 50-negative number
//! overestimates (Galkin et al., ICLR 2024, Fig. 4). Hyperparameters and
//! training protocol follow NBFNet's config/inductive/fb15k237.yaml and
//! script/run.py. The default is fast sum aggregation; `AGG=pna` switches
//! to exact PNA aggregation via ricci's segment max/min helpers.
//!
//! Observed on this harness with strict candidate-level `remove_one_hop`:
//! `AGG=pna EPOCHS=8 --features wgpu` reaches 0.817 50-negative Hits@10,
//! close to NBFNet's 0.834 and above GraIL's 0.642; full-rank Hits@10 is
//! 0.368, MRR 0.201. PNA is slower than sum here because the exact
//! max/min workaround snapshots edge messages to the host. One negative
//! finding worth keeping: selecting the checkpoint by validation MRR on
//! the TRAINING graph (the reference protocol) tracks in-distribution
//! quality, not cross-graph transfer, and can pick a worse-transferring
//! epoch.
//!
//! Data-gated: run `scripts/fetch_grail_fb237v1.sh` first; without data
//! this prints instructions and exits 0.
//!
//! Run: cargo run --release --example inductive_link_prediction

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;

use burn::backend::Autodiff;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Int, Tensor, TensorData};
#[cfg(not(feature = "wgpu"))]
use burn_ndarray::NdArray;
use ricci::scatter::scatter_max_min;
use ricci::NBFConv;

#[cfg(feature = "wgpu")]
type TB = Autodiff<Wgpu<f32, i32>>;
#[cfg(not(feature = "wgpu"))]
type TB = Autodiff<NdArray<f32>>;

// Hyperparameters follow NBFNet's config/inductive/fb15k237.yaml (dim 32,
// 6 layers, lr 5e-3, 32 negatives, adversarial temperature 0.5, 20
// epochs); note our query list enumerates both directions explicitly and
// uses batch 32, so one epoch here is about four of theirs in gradient
// steps — validation-based selection (below) is what bounds the budget.
const DIM: usize = 32;
const LAYERS: usize = 6;
const EPOCHS: usize = 20; // override with EPOCHS=n (PNA mode is ~15x costlier per epoch on CPU)
const BATCH: usize = 32;
const NEGATIVES: usize = 32;
const LR: f64 = 5e-3;
const ADV_TEMPERATURE: f32 = 0.5;

#[derive(Module, Debug)]
struct NbfNet<B: Backend> {
    query: burn::module::Param<Tensor<B, 2>>, // [R', d] indicator seeds
    // Per-layer relation representations are PROJECTED from the query
    // embedding (the reference's `dependent: yes`), so message modulation
    // is query-conditional, not a shared table.
    rel_proj: Vec<Linear<B>>, // d -> R' * d
    layers: Vec<NBFConv<B>>,
    // The reference combine is Linear(cat[state, messages]); a concat
    // linear is the sum of two linears on the parts, so adding this
    // self-state path to forward_edges' output is exactly equivalent.
    self_lin: Vec<Linear<B>>,
    // PNA mode (AGG=pna): 4 statistics x 3 degree scalers. The reference
    // applies one Linear to the 12d scaled features; algebraically that is
    // three 4d->d linears whose outputs are scaled per node, which avoids
    // materializing the [Q, N, 4d, 3] product (the profiled hot spot).
    stats_lin: Vec<[Linear<B>; 3]>,
    norms: Vec<LayerNorm<B>>, // sum aggregation over hub nodes explodes without per-layer normalization
    head1: Linear<B>,
    head2: Linear<B>,
    n_rel2: burn::module::Ignored<usize>,
    pna: burn::module::Ignored<bool>,
}

fn init_model<B: Backend>(n_rel2: usize, pna: bool, device: &B::Device) -> NbfNet<B> {
    let emb = |rows: usize| {
        burn::module::Param::initialized(
            burn::module::ParamId::new(),
            Tensor::random(
                [rows, DIM],
                burn::tensor::Distribution::Normal(0.0, 0.3),
                device,
            )
            .require_grad(),
        )
    };
    NbfNet {
        query: emb(n_rel2),
        rel_proj: (0..LAYERS)
            .map(|_| LinearConfig::new(DIM, n_rel2 * DIM).init(device))
            .collect(),
        layers: (0..LAYERS).map(|_| NBFConv::init(DIM, device)).collect(),
        self_lin: (0..LAYERS)
            .map(|_| LinearConfig::new(DIM, DIM).with_bias(false).init(device))
            .collect(),
        stats_lin: (0..LAYERS)
            .map(|_| {
                [
                    LinearConfig::new(4 * DIM, DIM).init(device),
                    LinearConfig::new(4 * DIM, DIM)
                        .with_bias(false)
                        .init(device),
                    LinearConfig::new(4 * DIM, DIM)
                        .with_bias(false)
                        .init(device),
                ]
            })
            .collect(),
        norms: (0..LAYERS)
            .map(|_| LayerNormConfig::new(DIM).init(device))
            .collect(),
        head1: LinearConfig::new(2 * DIM, 2 * DIM).init(device),
        head2: LinearConfig::new(2 * DIM, 1).init(device),
        n_rel2: burn::module::Ignored(n_rel2),
        pna: burn::module::Ignored(pna),
    }
}

impl<B: Backend> NbfNet<B> {
    /// Pair states for a batch of queries `(source, relation)` over the
    /// given edge list: `[Q, N, d]`.
    #[allow(clippy::too_many_arguments)]
    fn propagate(
        &self,
        n: usize,
        sources: &[usize],
        rels_q: &[usize],
        heads: Tensor<B, 1, Int>,
        tails: Tensor<B, 1, Int>,
        etypes: Tensor<B, 1, Int>,
        tails_host: &[usize],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let q = sources.len();
        // Boundary: h0[b, sources[b], :] = query[rels_q[b], :], built as a
        // host one-hot mask times the query rows so it stays differentiable.
        let mask = {
            let mut m = vec![0.0f32; q * n];
            for (b, &s) in sources.iter().enumerate() {
                m[b * n + s] = 1.0;
            }
            Tensor::<B, 3>::from_data(TensorData::new(m, [q, n, 1]), device)
        };
        let rq_flat = {
            let idx: Vec<i64> = rels_q.iter().map(|&r| r as i64).collect();
            let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(idx, [q]), device);
            self.query.val().select(0, idx) // [Q, d]
        };
        // h0 is [Q, N, d]. States start AT the boundary (not zero), so
        // layer 1 already propagates from the source: six layers, six hops.
        let h0 = mask * rq_flat.clone().reshape([q, 1, DIM]);
        // Degree scalers for PNA (recomputed per call: edge drops change
        // degrees). degree_out + 1 counts the boundary as one message.
        let (degp1, scale_t, inv_scale_t) = if self.pna.0 {
            let mut deg = vec![1.0f32; n];
            for &t in tails_host {
                deg[t] += 1.0;
            }
            let logd: Vec<f32> = deg.iter().map(|d| d.ln()).collect();
            let smean = (logd.iter().sum::<f32>() / n as f32).max(1e-6);
            let sc: Vec<f32> = logd.iter().map(|&l| l / smean).collect();
            let inv: Vec<f32> = sc.iter().map(|&s| 1.0 / s.max(0.01)).collect();
            (
                Tensor::<B, 3>::from_data(TensorData::new(deg, [1, n, 1]), device),
                Tensor::<B, 3>::from_data(TensorData::new(sc, [1, n, 1]), device),
                Tensor::<B, 3>::from_data(TensorData::new(inv, [1, n, 1]), device),
            )
        } else {
            let z = || Tensor::<B, 3>::zeros([1, 1, 1], device);
            (z(), z(), z())
        };
        let mut h = h0.clone();
        for ((((layer, norm), proj), selfl), statsl) in self
            .layers
            .iter()
            .zip(self.norms.iter())
            .zip(self.rel_proj.iter())
            .zip(self.self_lin.iter())
            .zip(self.stats_lin.iter())
        {
            // Query-conditional relation representations for this layer.
            let rel = proj
                .forward(rq_flat.clone())
                .reshape([q, self.n_rel2.0, DIM]);
            let msgs_out = if self.pna.0 {
                // PNA aggregation (Corso et al., NeurIPS 2020) as in the
                // reference layer: mean/max/min/std over incoming messages
                // (boundary included as one message), times three degree
                // scalers (identity, log-degree, inverse log-degree).
                let msgs = h.clone().select(1, heads.clone()) * rel.select(1, etypes.clone());
                let zeros = || Tensor::<B, 3>::zeros([q, n, DIM], device);
                let sums = zeros().select_assign(
                    1,
                    tails.clone(),
                    msgs.clone(),
                    burn::tensor::IndexingUpdateOp::Add,
                );
                let sq_sums = zeros().select_assign(
                    1,
                    tails.clone(),
                    msgs.clone().powf_scalar(2.0),
                    burn::tensor::IndexingUpdateOp::Add,
                );
                let mean = (sums + h0.clone()) / degp1.clone();
                let sq_mean = (sq_sums + h0.clone().powf_scalar(2.0)) / degp1.clone();
                let (mx, mn) = scatter_max_min(msgs, tails_host, n);
                let mx = mx.max_pair(h0.clone());
                let mn = mn.min_pair(h0.clone());
                let std = (sq_mean - mean.clone().powf_scalar(2.0))
                    .clamp_min(1e-6)
                    .sqrt();
                let feats = Tensor::cat(vec![mean, mx, mn, std], 2);
                statsl[0].forward(feats.clone())
                    + statsl[1].forward(feats.clone()) * scale_t.clone()
                    + statsl[2].forward(feats) * inv_scale_t.clone()
            } else {
                layer.forward_edges(
                    h.clone(),
                    h0.clone(),
                    heads.clone(),
                    tails.clone(),
                    etypes.clone(),
                    rel,
                )
            };
            let out = msgs_out + selfl.forward(h.clone());
            // Residual short-cut after norm + activation, as in the reference.
            h = activation::relu(norm.forward(out)) + h;
        }
        h
    }

    /// Score candidate tails: `states` `[Q, N, d]`, `cands` `[Q, C]` ->
    /// logits `[Q, C]`. The head sees the pair state concatenated with the
    /// query embedding (per-relation calibration, as in the reference MLP).
    fn score(&self, states: Tensor<B, 3>, cands: &[Vec<usize>], rels_q: &[usize]) -> Tensor<B, 2> {
        let q = cands.len();
        let c = cands[0].len();
        let flat: Vec<i64> = cands.iter().flatten().map(|&x| x as i64).collect();
        let device = states.device();
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(flat, [q * c]), &device);
        // Per-query gather: offset candidate ids into the flattened [Q*N] axis.
        let n = states.dims()[1];
        let offsets: Vec<i64> = (0..q)
            .flat_map(|b| std::iter::repeat_n((b * n) as i64, c))
            .collect();
        let idx = idx + Tensor::from_data(TensorData::new(offsets, [q * c]), &device);
        let picked = states
            .reshape([q * n, DIM])
            .select(0, idx)
            .reshape([q * c, DIM]);
        let rq = {
            let ridx: Vec<i64> = rels_q
                .iter()
                .flat_map(|&r| std::iter::repeat_n(r as i64, c))
                .collect();
            let ridx = Tensor::<B, 1, Int>::from_data(TensorData::new(ridx, [q * c]), &device);
            self.query.val().select(0, ridx)
        };
        let feat = Tensor::cat(vec![picked, rq], 1);
        let hidden = activation::relu(self.head1.forward(feat));
        self.head2.forward(hidden).reshape([q, c])
    }
}

fn main() {
    let Some((train_g, n_train_ent, rel_names)) = load_graph(Path::new("data/fb237_v1"), None)
    else {
        eprintln!("GraIL fb237_v1 data not found: run scripts/fetch_grail_fb237v1.sh");
        return; // data-gated no-op.
    };
    let Some((ind_g, n_ind_ent, _)) = load_graph(Path::new("data/fb237_v1_ind"), Some(&rel_names))
    else {
        eprintln!("fb237_v1_ind missing or has unseen relations");
        return;
    };
    let n_rel2 = rel_names.len() * 2;
    eprintln!(
        "fb237_v1: {} entities, {} relations ({} with inverses), {} train triples; \
         inductive graph: {} entities, {} observed / {} test triples",
        n_train_ent,
        rel_names.len(),
        n_rel2,
        train_g.train.len(),
        n_ind_ent,
        ind_g.train.len(),
        ind_g.test.len(),
    );

    let device = <TB as Backend>::Device::default();
    let pna = std::env::var("AGG").is_ok_and(|v| v == "pna");
    let epochs = std::env::var("EPOCHS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(EPOCHS);
    eprintln!(
        "aggregation: {}  epochs: {epochs}  backend: {}",
        if pna { "pna" } else { "sum" },
        backend_name()
    );
    let mut model = init_model::<TB>(n_rel2, pna, &device);
    // Burn's Adam epsilon defaults to 1e-5; match the 1e-8 the reference
    // implementations assume.
    let mut optim = AdamConfig::new()
        .with_epsilon(1e-8)
        .init::<TB, NbfNet<TB>>();

    // Queries: every train triple in both directions.
    let mut queries: Vec<(usize, usize, usize)> = Vec::new(); // (src, rel, tgt)
    for &(h, r, t) in &train_g.train {
        queries.push((h, r, t));
        queries.push((t, r + rel_names.len(), h));
    }
    let known = train_g.known_tails();

    // Validation queries on the training graph (both directions), capped at
    // a fixed deterministic subsample to keep the per-check cost bounded.
    let mut valid_queries: Vec<(usize, usize, usize)> = Vec::new();
    for &(h, r, t) in &train_g.valid {
        valid_queries.push((h, r, t));
        valid_queries.push((t, r + rel_names.len(), h));
    }
    let mut vstate = 0x51ce5_u64;
    for i in (1..valid_queries.len()).rev() {
        vstate = vstate
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        valid_queries.swap(i, (vstate % (i as u64 + 1)) as usize);
    }
    valid_queries.truncate(256);
    let edges_valid = train_g.edge_tensors::<TB>(&device, None);
    let known_full = train_g.known_tails_full();
    let mut best_mrr = f64::MIN;
    let mut best_epoch = 0usize;
    let mut best_model = model.clone();
    let full_edge_count = train_g.directed_edge_count();

    for epoch in 0..epochs {
        let mut order: Vec<usize> = (0..queries.len()).collect();
        // Deterministic shuffle (LCG) keeps runs reproducible.
        let mut state = 0x2545f491_u64.wrapping_add(epoch as u64);
        for i in (1..order.len()).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            order.swap(i, (state % (i as u64 + 1)) as usize);
        }
        let mut total = 0.0_f32;
        let mut batches = 0u32;
        let mut diag_h = 0.0_f32; // mean |state| on the first batch: explosion/collapse probe
        let mut drop_pairs_total = 0usize;
        let mut dropped_edges_total = 0usize;
        let mut first_score_diag: Option<(f64, f64, f64)> = None;
        for chunk in order.chunks(BATCH) {
            let batch: Vec<_> = chunk.iter().map(|&i| queries[i]).collect();
            // Candidates: positive tail + uniform negatives. Sample these
            // before building the message graph: reference `remove_one_hop`
            // removes edges between the source and every candidate, not only
            // the positive tail.
            let mut state2 = state ^ 0x9e3779b97f4a7c15;
            let cands: Vec<Vec<usize>> = batch
                .iter()
                .map(|&(s, r, t)| {
                    let mut row = vec![t];
                    let kt = known.get(&(s, r));
                    while row.len() < 1 + NEGATIVES {
                        state2 = state2
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let cand = (state2 >> 16) as usize % n_train_ent;
                        if cand != t && kt.is_none_or(|set| !set.contains(&cand)) {
                            row.push(cand);
                        }
                    }
                    row
                })
                .collect();
            // Drop ALL edges between each query pair from the message graph
            // (any relation, both orientations): FB15k-237 is dense in
            // pair-parallel relations, and any surviving 1-hop edge is a
            // copy path the model exploits instead of learning multi-hop
            // structure (the reference's remove_one_hop).
            let drop: HashSet<(usize, usize)> = batch
                .iter()
                .zip(cands.iter())
                .flat_map(|(&(s, _, _), row)| row.iter().flat_map(move |&t| [(s, t), (t, s)]))
                .collect();
            drop_pairs_total += drop.len();
            dropped_edges_total += train_g.dropped_directed_edges(&drop);
            let (heads, tails, etypes, tails_host) =
                train_g.edge_tensors::<TB>(&device, Some(&drop));
            let sources: Vec<usize> = batch.iter().map(|q| q.0).collect();
            let rels_q: Vec<usize> = batch.iter().map(|q| q.1).collect();
            let states = model.propagate(
                n_train_ent,
                &sources,
                &rels_q,
                heads,
                tails,
                etypes,
                &tails_host,
                &device,
            );
            if batches == 0 {
                diag_h = states
                    .clone()
                    .abs()
                    .mean()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()[0];
            }
            let logits = model.score(states, &cands, &rels_q);
            let q = cands.len();
            let c = 1 + NEGATIVES;
            if batches == 0 {
                let lv: Vec<f32> = logits.clone().into_data().to_vec().unwrap();
                let mut pos_sum = 0.0f64;
                let mut neg_sum = 0.0f64;
                let mut hard_margin_sum = 0.0f64;
                for row in lv.chunks(c) {
                    let pos = row[0] as f64;
                    let max_neg = row[1..].iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    pos_sum += pos;
                    neg_sum += row[1..].iter().map(|&x| x as f64).sum::<f64>() / NEGATIVES as f64;
                    hard_margin_sum += max_neg - pos;
                }
                first_score_diag = Some((
                    pos_sum / q as f64,
                    neg_sum / q as f64,
                    hard_margin_sum / q as f64,
                ));
            }
            // Self-adversarial BCE (RotatE-style, T = 0.5): negatives are
            // weighted by a detached softmax over their own logits, so the
            // hardest negatives dominate the gradient.
            let pos = logits.clone().slice([0..q, 0..1]);
            let neg = logits.slice([0..q, 1..c]);
            let w = activation::softmax(neg.clone() / ADV_TEMPERATURE, 1).detach();
            let loss_pos = activation::softplus(-pos, 1.0).mean();
            let loss_neg = (activation::softplus(neg, 1.0) * w).sum_dim(1).mean();
            let loss = (loss_pos + loss_neg) / 2.0;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(LR, model, grads);
            total += loss.into_data().to_vec::<f32>().unwrap()[0];
            batches += 1;
        }
        eprint!(
            "epoch {epoch}: loss {:.4}  |h| {:.3}",
            total / batches as f32,
            diag_h
        );
        let drop_pairs = drop_pairs_total as f64 / batches as f64;
        let dropped_edges = dropped_edges_total as f64 / batches as f64;
        eprint!(
            "  drop pairs/batch {:.1}  edges dropped {:.1}/{full_edge_count}",
            drop_pairs, dropped_edges
        );
        if let Some((pos, neg, hard)) = first_score_diag {
            eprint!(
                "  first scores pos {:.3} neg {:.3} hard-pos {:.3}",
                pos, neg, hard
            );
        }
        // Model selection by validation MRR every 2 epochs, as in the
        // reference harness (train_and_validate: eval every num_epoch/10
        // epochs, load best checkpoint before test). Validation queries
        // live on the TRAINING graph; the inductive graph stays untouched
        // until test.
        if (epoch + 1) % 2 == 0 {
            let v = evaluate(
                &model,
                n_train_ent,
                &edges_valid,
                &valid_queries,
                &known_full,
            );
            eprint!("  valid MRR {:.4}", v.mrr);
            if v.mrr > best_mrr {
                best_mrr = v.mrr;
                best_epoch = epoch;
                best_model = model.clone();
            }
        }
        eprintln!();
    }
    if best_mrr == f64::MIN {
        let v = evaluate(
            &model,
            n_train_ent,
            &edges_valid,
            &valid_queries,
            &known_full,
        );
        best_mrr = v.mrr;
        best_epoch = epochs.saturating_sub(1);
        best_model = model.clone();
    }
    eprintln!("selected epoch {best_epoch} (valid MRR {best_mrr:.4})");

    // Inductive evaluation on the disjoint graph: same relations, new
    // entities; the model transfers because nothing entity-wise was learned.
    let (heads_e, tails_e, etypes_e, tails_host_e) = ind_g.edge_tensors::<TB>(&device, None);
    let known_ind = ind_g.known_tails_full();
    let mut test_queries: Vec<(usize, usize, usize)> = Vec::new();
    for &(h, r, t) in &ind_g.test {
        test_queries.push((h, r, t));
        test_queries.push((t, r + rel_names.len(), h));
    }
    let out = evaluate(
        &best_model,
        n_ind_ent,
        &(heads_e, tails_e, etypes_e, tails_host_e),
        &test_queries,
        &known_ind,
    );
    eprintln!(
        "full rank: mean {:.1}  median {}  p90 {}  p95 {}  p99 {}  max {} (of {} entities)",
        out.mean_rank, out.median, out.p90, out.p95, out.p99, out.max, n_ind_ent
    );
    eprintln!(
        "full recall@k / Hits@k: @1 {:.3}  @3 {:.3}  @10 {:.3}  @50 {:.3}",
        out.hits_full_1, out.hits_full_3, out.hits_full_10, out.hits_full_50
    );
    eprintln!(
        "sampled-50 recall@k / Hits@k: @1 {:.3}  @3 {:.3}  @10 {:.3}",
        out.hits50_1, out.hits50_3, out.hits50_10
    );
    eprintln!(
        "sampled-50 rank: mean {:.1}  median {}  p90 {}  max 51",
        out.mean_rank50, out.median50, out.p90_50
    );
    eprintln!(
        "score margin gold-best-corrupt: mean {:.3}  p10 {:.3}  median {:.3}; eval coverage {:.3}  |h| {:.3}",
        out.margin_mean, out.margin_p10, out.margin_median, out.coverage, out.state_abs
    );
    let _ = std::io::stderr().flush();
    println!(
        "fb237_v1 -> fb237_v1_ind (both directions, n = {}):\n\
         Hits@10 (50 filtered negatives, GraIL protocol): {:.3}\n\
         full-ranking filtered Hits@10: {:.3}   MRR: {:.3}\n\
         references on this split: GraIL 0.642, NBFNet 0.834 (50-neg protocol)",
        test_queries.len(),
        out.hits50_10,
        out.hits_full_10,
        out.mrr,
    );
}

fn backend_name() -> &'static str {
    #[cfg(feature = "wgpu")]
    {
        "wgpu"
    }
    #[cfg(not(feature = "wgpu"))]
    {
        "ndarray"
    }
}

type EdgeTensorsOf<B> = (
    Tensor<B, 1, Int>,
    Tensor<B, 1, Int>,
    Tensor<B, 1, Int>,
    Vec<usize>, // host-side tails, for segment argmax and degrees
);
type EdgeTensors = EdgeTensorsOf<TB>;

struct EvalOut {
    mrr: f64,
    hits_full_1: f64,
    hits_full_3: f64,
    hits_full_10: f64,
    hits_full_50: f64,
    hits50_1: f64,
    hits50_3: f64,
    hits50_10: f64,
    mean_rank50: f64,
    median50: usize,
    p90_50: usize,
    mean_rank: f64,
    median: usize,
    p90: usize,
    p95: usize,
    p99: usize,
    max: usize,
    margin_mean: f64,
    margin_p10: f64,
    margin_median: f64,
    coverage: f32,
    state_abs: f32,
}

/// Filtered ranking over all entities plus the 50-sampled-negative
/// protocol, for a query set against a fixed message graph.
fn evaluate(
    model: &NbfNet<TB>,
    n_ent: usize,
    edges: &EdgeTensors,
    queries: &[(usize, usize, usize)],
    known: &HashMap<(usize, usize), HashSet<usize>>,
) -> EvalOut {
    let (heads, tails, etypes, tails_host) = edges;
    let device = heads.device();
    let mut mrr = 0.0f64;
    let mut hits_full = [0.0f64; 4]; // @1, @3, @10, @50
    let mut hits50 = [0.0f64; 3]; // @1, @3, @10
    let mut ranks: Vec<usize> = Vec::new();
    let mut sample_ranks: Vec<usize> = Vec::new();
    let mut margins: Vec<f64> = Vec::new();
    let mut coverage_sum = 0.0f32;
    let mut state_abs_sum = 0.0f32;
    let mut chunks = 0u32;
    let mut rng = 0xabcdef12345_u64;
    for chunk in queries.chunks(BATCH) {
        let sources: Vec<usize> = chunk.iter().map(|q| q.0).collect();
        let rels_q: Vec<usize> = chunk.iter().map(|q| q.1).collect();
        let states = model.propagate(
            n_ent,
            &sources,
            &rels_q,
            heads.clone(),
            tails.clone(),
            etypes.clone(),
            tails_host,
            &device,
        );
        coverage_sum += batched_coverage(&states);
        state_abs_sum += states
            .clone()
            .abs()
            .mean()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        chunks += 1;
        // Full ranking: score everything, filter known tails.
        let all: Vec<Vec<usize>> = (0..chunk.len()).map(|_| (0..n_ent).collect()).collect();
        let logits = model.score(states, &all, &rels_q);
        let flat: Vec<f32> = logits.into_data().to_vec().unwrap();
        for (b, &(s, r, t)) in chunk.iter().enumerate() {
            let row = &flat[b * n_ent..(b + 1) * n_ent];
            let gold = row[t];
            let filt = known.get(&(s, r));
            let mut rank_full = 1usize;
            let mut best_corrupt = f32::NEG_INFINITY;
            for (e, &sc) in row.iter().enumerate() {
                if e != t && filt.is_none_or(|set| !set.contains(&e)) {
                    if sc > gold {
                        rank_full += 1;
                    }
                    if sc > best_corrupt {
                        best_corrupt = sc;
                    }
                }
            }
            for (slot, &k) in [1usize, 3, 10, 50].iter().enumerate() {
                if rank_full <= k {
                    hits_full[slot] += 1.0;
                }
            }
            mrr += 1.0 / rank_full as f64;
            ranks.push(rank_full);
            margins.push((gold - best_corrupt) as f64);
            // 50 sampled filtered negatives (GraIL protocol).
            let mut better = 0usize;
            let mut drawn = 0usize;
            while drawn < 50 {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let cand = (rng >> 16) as usize % n_ent;
                if cand == t || filt.is_some_and(|set| set.contains(&cand)) {
                    continue;
                }
                if row[cand] > gold {
                    better += 1;
                }
                drawn += 1;
            }
            let rank50 = 1 + better;
            for (slot, &k) in [1usize, 3, 10].iter().enumerate() {
                if rank50 <= k {
                    hits50[slot] += 1.0;
                }
            }
            sample_ranks.push(rank50);
        }
    }
    ranks.sort_unstable();
    sample_ranks.sort_unstable();
    margins.sort_by(|a, b| a.total_cmp(b));
    let n = queries.len() as f64;
    EvalOut {
        mrr: mrr / n,
        hits_full_1: hits_full[0] / n,
        hits_full_3: hits_full[1] / n,
        hits_full_10: hits_full[2] / n,
        hits_full_50: hits_full[3] / n,
        hits50_1: hits50[0] / n,
        hits50_3: hits50[1] / n,
        hits50_10: hits50[2] / n,
        mean_rank50: sample_ranks.iter().sum::<usize>() as f64 / n,
        median50: sample_ranks[sample_ranks.len() / 2],
        p90_50: sample_ranks[sample_ranks.len() * 9 / 10],
        mean_rank: ranks.iter().sum::<usize>() as f64 / n,
        median: ranks[ranks.len() / 2],
        p90: ranks[ranks.len() * 9 / 10],
        p95: ranks[ranks.len() * 95 / 100],
        p99: ranks[ranks.len() * 99 / 100],
        max: *ranks.last().unwrap(),
        margin_mean: margins.iter().sum::<f64>() / n,
        margin_p10: margins[margins.len() / 10],
        margin_median: margins[margins.len() / 2],
        coverage: coverage_sum / chunks as f32,
        state_abs: state_abs_sum / chunks as f32,
    }
}

fn batched_coverage<B: Backend>(states: &Tensor<B, 3>) -> f32 {
    let [q, n, d] = states.dims();
    let vals: Vec<f32> = states.clone().into_data().to_vec().unwrap();
    let reached = vals
        .chunks(d)
        .filter(|row| row.iter().any(|v| v.abs() > 1e-6))
        .count();
    reached as f32 / (q * n) as f32
}

struct Graph {
    train: Vec<(usize, usize, usize)>,
    valid: Vec<(usize, usize, usize)>,
    test: Vec<(usize, usize, usize)>,
    all: Vec<(usize, usize, usize)>,
    n_rel: usize,
}

impl Graph {
    fn directed_edge_count(&self) -> usize {
        self.train.len() * 2
    }

    fn dropped_directed_edges(&self, drop: &HashSet<(usize, usize)>) -> usize {
        self.train
            .iter()
            .map(|&(a, _, b)| {
                usize::from(drop.contains(&(a, b))) + usize::from(drop.contains(&(b, a)))
            })
            .sum()
    }

    /// Edge tensors over the observed (train) triples, both directions,
    /// optionally dropping every edge between given (src, tgt) pairs.
    fn edge_tensors<B: Backend>(
        &self,
        device: &B::Device,
        drop: Option<&HashSet<(usize, usize)>>,
    ) -> EdgeTensorsOf<B> {
        let mut h = Vec::new();
        let mut t = Vec::new();
        let mut ty = Vec::new();
        let mut push = |a: usize, r: usize, b: usize| {
            if drop.is_none_or(|d| !d.contains(&(a, b))) {
                h.push(a as i64);
                t.push(b as i64);
                ty.push(r as i64);
            }
        };
        for &(a, r, b) in &self.train {
            push(a, r, b);
            push(b, r + self.n_rel, a);
        }
        let e = h.len();
        let tails_host: Vec<usize> = t.iter().map(|&x| x as usize).collect();
        (
            Tensor::from_data(TensorData::new(h, [e]), device),
            Tensor::from_data(TensorData::new(t, [e]), device),
            Tensor::from_data(TensorData::new(ty, [e]), device),
            tails_host,
        )
    }

    /// Known tails per (source, rel') over train triples (training filter).
    fn known_tails(&self) -> HashMap<(usize, usize), HashSet<usize>> {
        let mut m: HashMap<(usize, usize), HashSet<usize>> = HashMap::new();
        for &(a, r, b) in &self.train {
            m.entry((a, r)).or_default().insert(b);
            m.entry((b, r + self.n_rel)).or_default().insert(a);
        }
        m
    }

    /// Known tails over ALL splits (evaluation filter).
    fn known_tails_full(&self) -> HashMap<(usize, usize), HashSet<usize>> {
        let mut m: HashMap<(usize, usize), HashSet<usize>> = HashMap::new();
        for &(a, r, b) in &self.all {
            m.entry((a, r)).or_default().insert(b);
            m.entry((b, r + self.n_rel)).or_default().insert(a);
        }
        m
    }
}

/// Load a GraIL-format directory. With `fixed_rels`, relation names must
/// resolve in the given vocabulary (the inductive split shares relations);
/// entities are interned fresh per graph.
fn load_graph(dir: &Path, fixed_rels: Option<&Vec<String>>) -> Option<(Graph, usize, Vec<String>)> {
    let mut rels: Vec<String> = fixed_rels.cloned().unwrap_or_default();
    let mut rel_id: HashMap<String, usize> = rels
        .iter()
        .enumerate()
        .map(|(i, r)| (r.clone(), i))
        .collect();
    let mut ents: HashMap<String, usize> = HashMap::new();
    let mut read = |name: &str| -> Option<Vec<(usize, usize, usize)>> {
        let text = std::fs::read_to_string(dir.join(name)).ok()?;
        let mut out = Vec::new();
        for line in text.lines() {
            let mut it = line.trim().split('\t');
            let (h, r, t) = (it.next()?, it.next()?, it.next()?);
            let next = ents.len();
            let hid = *ents.entry(h.to_string()).or_insert(next);
            let rid = match rel_id.get(r) {
                Some(&i) => i,
                None if fixed_rels.is_none() => {
                    rels.push(r.to_string());
                    rel_id.insert(r.to_string(), rels.len() - 1);
                    rels.len() - 1
                }
                None => return None, // unseen relation in fixed mode
            };
            let next = ents.len();
            let tid = *ents.entry(t.to_string()).or_insert(next);
            out.push((hid, rid, tid));
        }
        Some(out)
    };
    let train = read("train.txt")?;
    let valid = read("valid.txt")?;
    let test = read("test.txt")?;
    let mut all = train.clone();
    all.extend_from_slice(&valid);
    all.extend_from_slice(&test);
    let n_rel = rels.len();
    Some((
        Graph {
            train,
            valid,
            test,
            all,
            n_rel,
        },
        ents.len(),
        rels,
    ))
}
