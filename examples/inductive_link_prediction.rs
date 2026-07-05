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
//! script/run.py, with one deliberate divergence: sum aggregation instead
//! of PNA, because burn 0.20 scatters support only Add (no scatter-max /
//! min for PNA's statistics). The paper's own ablation puts DistMult+sum
//! at MRR 0.388 vs PNA 0.415 on transductive FB15k-237.
//!
//! Observed on this harness (ndarray CPU, ~35 min): 50-negative Hits@10
//! in the 0.52-0.59 range across protocol-faithful variants (n = 410
//! queries, so the 95% CI is about +/- 0.05); far above the ~0.196
//! random floor of the 50-negative protocol, below the PNA-equipped
//! references. Transfer to unseen entities genuinely happens; parity
//! with the published aggregator does not. One negative finding worth
//! keeping: selecting the checkpoint by validation MRR on the TRAINING
//! graph (the reference protocol) tracks in-distribution quality, not
//! cross-graph transfer, and can pick a worse-transferring epoch.
//!
//! Data-gated: run `scripts/fetch_grail_fb237v1.sh` first; without data
//! this prints instructions and exits 0.
//!
//! Run: cargo run --release --example inductive_link_prediction

use std::collections::{HashMap, HashSet};
use std::path::Path;

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Int, Tensor, TensorData};
use burn_ndarray::NdArray;
use ricci::scatter::{scatter_max, scatter_min};
use ricci::NBFConv;

type TB = Autodiff<NdArray<f32>>;

// Hyperparameters follow NBFNet's config/inductive/fb15k237.yaml (dim 32,
// 6 layers, lr 5e-3, 32 negatives, adversarial temperature 0.5, 20
// epochs); note our query list enumerates both directions explicitly and
// uses batch 32, so one epoch here is about four of theirs in gradient
// steps — validation-based selection (below) is what bounds the budget.
const DIM: usize = 32;
const LAYERS: usize = 6;
const EPOCHS: usize = 20;
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
    // PNA mode (AGG=pna): 4 statistics x 3 degree scalers -> 12d features
    // per node, combined by this linear instead of NBFConv's update.
    stats_lin: Vec<Linear<B>>,
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
            .map(|_| LinearConfig::new(12 * DIM, DIM).init(device))
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
        let (degp1, scales) = if self.pna.0 {
            let mut deg = vec![1.0f32; n];
            for &t in tails_host {
                deg[t] += 1.0;
            }
            let logd: Vec<f32> = deg.iter().map(|d| d.ln()).collect();
            let smean = (logd.iter().sum::<f32>() / n as f32).max(1e-6);
            let mut sc = Vec::with_capacity(n * 3);
            for &l in &logd {
                let s = l / smean;
                sc.extend_from_slice(&[1.0, s, 1.0 / s.max(0.01)]);
            }
            (
                Tensor::<B, 3>::from_data(TensorData::new(deg, [1, n, 1]), device),
                Tensor::<B, 4>::from_data(TensorData::new(sc, [1, n, 1, 3]), device),
            )
        } else {
            (
                Tensor::zeros([1, 1, 1], device),
                Tensor::zeros([1, 1, 1, 1], device),
            )
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
                let mx = scatter_max(msgs.clone(), tails_host, n, 0.0).max_pair(h0.clone());
                let mn = scatter_min(msgs, tails_host, n, 0.0).min_pair(h0.clone());
                let std = (sq_mean - mean.clone().powf_scalar(2.0))
                    .clamp_min(1e-6)
                    .sqrt();
                let feats = Tensor::cat(vec![mean, mx, mn, std], 2).reshape([q, n, 4 * DIM, 1]);
                let pna = (feats * scales.clone()).reshape([q, n, 12 * DIM]);
                statsl.forward(pna)
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
    eprintln!("aggregation: {}", if pna { "pna" } else { "sum" });
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

    for epoch in 0..EPOCHS {
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
        for chunk in order.chunks(BATCH) {
            let batch: Vec<_> = chunk.iter().map(|&i| queries[i]).collect();
            // Drop ALL edges between each query pair from the message graph
            // (any relation, both orientations): FB15k-237 is dense in
            // pair-parallel relations, and any surviving 1-hop edge is a
            // copy path the model exploits instead of learning multi-hop
            // structure (the reference's remove_one_hop).
            let drop: HashSet<(usize, usize)> = batch
                .iter()
                .flat_map(|&(s, _, t)| [(s, t), (t, s)])
                .collect();
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
            // Candidates: positive tail + uniform negatives.
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
        "rank distribution: median {}  p90 {}  max {} (of {} entities)",
        out.median, out.p90, out.max, n_ind_ent
    );
    println!(
        "fb237_v1 -> fb237_v1_ind (both directions, n = {}):\n\
         Hits@10 (50 filtered negatives, GraIL protocol): {:.3}\n\
         full-ranking filtered Hits@10: {:.3}   MRR: {:.3}\n\
         references on this split: GraIL 0.642, NBFNet 0.834 (50-neg protocol)",
        test_queries.len(),
        out.hits50,
        out.hits_full,
        out.mrr,
    );
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
    hits_full: f64,
    hits50: f64,
    median: usize,
    p90: usize,
    max: usize,
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
    let (mut hits50, mut hits_full, mut mrr) = (0.0f64, 0.0f64, 0.0f64);
    let mut ranks: Vec<usize> = Vec::new();
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
        // Full ranking: score everything, filter known tails.
        let all: Vec<Vec<usize>> = (0..chunk.len()).map(|_| (0..n_ent).collect()).collect();
        let logits = model.score(states, &all, &rels_q);
        let flat: Vec<f32> = logits.into_data().to_vec().unwrap();
        for (b, &(s, r, t)) in chunk.iter().enumerate() {
            let row = &flat[b * n_ent..(b + 1) * n_ent];
            let gold = row[t];
            let filt = known.get(&(s, r));
            let mut rank_full = 1usize;
            for (e, &sc) in row.iter().enumerate() {
                if e != t && sc > gold && filt.is_none_or(|set| !set.contains(&e)) {
                    rank_full += 1;
                }
            }
            if rank_full <= 10 {
                hits_full += 1.0;
            }
            mrr += 1.0 / rank_full as f64;
            ranks.push(rank_full);
            // 50 sampled filtered negatives (GraIL protocol).
            let mut worse = 0usize;
            let mut drawn = 0usize;
            while drawn < 50 {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let cand = (rng >> 16) as usize % n_ent;
                if cand == t || filt.is_some_and(|set| set.contains(&cand)) {
                    continue;
                }
                if row[cand] <= gold {
                    worse += 1;
                }
                drawn += 1;
            }
            if 50 - worse < 10 {
                hits50 += 1.0;
            }
        }
    }
    ranks.sort_unstable();
    let n = queries.len() as f64;
    EvalOut {
        mrr: mrr / n,
        hits_full: hits_full / n,
        hits50: hits50 / n,
        median: ranks[ranks.len() / 2],
        p90: ranks[ranks.len() * 9 / 10],
        max: *ranks.last().unwrap(),
    }
}

struct Graph {
    train: Vec<(usize, usize, usize)>,
    valid: Vec<(usize, usize, usize)>,
    test: Vec<(usize, usize, usize)>,
    all: Vec<(usize, usize, usize)>,
    n_rel: usize,
}

impl Graph {
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
