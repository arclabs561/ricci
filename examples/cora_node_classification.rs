//! 2-layer GCN node classification on Planetoid/LBC citation graphs (Cora, Citeseer).
//!
//! Classifies each paper into a topic from bag-of-words features plus the
//! citation graph (Kipf & Welling 2017). Dataset dims (feature count, class
//! count) are detected from the data, so the same model runs on any
//! linqs/LBC-format dataset (`<name>.content` + `<name>.cites`).
//!
//! ```sh
//! ./scripts/fetch_cora.sh        # or ./scripts/fetch_citeseer.sh
//! cargo run --release --example cora_node_classification           # cora (default)
//! cargo run --release --example cora_node_classification citeseer  # citeseer
//! ```
//!
//! Two [`GCNConv`] layers (`A_hat @ (X W)`) with a ReLU between, full-batch Adam
//! with weight decay, cross-entropy on a 20-per-class split, accuracy on a
//! 1000-node test split. `A_hat = D~^{-1/2} (A + I) D~^{-1/2}`. Expect ~0.80
//! test accuracy on Cora, ~0.70 on Citeseer.

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLoss;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{activation, Int, Tensor, TensorData};
use burn_ndarray::NdArray;

use ricci::GCNConv;

const HIDDEN: usize = 16;

/// Parsed citation graph: dense features, labels, normalized adjacency, dims.
struct Graph {
    n: usize,
    n_features: usize,
    n_classes: usize,
    features: Vec<f32>, // [n * n_features] row-major
    labels: Vec<i32>,   // [n]
    adj_norm: Vec<f32>, // [n * n] row-major, symmetric-normalized with self-loops
}

/// Load an LBC-format graph (`<name>.content` = `id <features...> label`,
/// `<name>.cites` = edges). Feature and class counts are detected from the data.
fn load_planetoid(dir: &Path, name: &str) -> std::io::Result<Graph> {
    let content = std::fs::read_to_string(dir.join(format!("{name}.content")))?;
    let cites = std::fs::read_to_string(dir.join(format!("{name}.cites")))?;

    // cols = id + features + label, so n_features = cols - 2.
    let n_features = content
        .lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').count().saturating_sub(2))
        .unwrap_or(0);

    let mut label_names: Vec<&str> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.rsplit('\t').next().unwrap())
        .collect();
    label_names.sort_unstable();
    label_names.dedup();
    let n_classes = label_names.len();
    let class_id: HashMap<&str, i32> = label_names
        .iter()
        .enumerate()
        .map(|(i, &name)| (name, i as i32))
        .collect();

    let mut id_to_idx: HashMap<String, usize> = HashMap::new();
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for line in content.lines().filter(|l| !l.trim().is_empty()) {
        let cols: Vec<&str> = line.split('\t').collect();
        let idx = id_to_idx.len();
        id_to_idx.insert(cols[0].to_string(), idx);
        for f in &cols[1..=n_features] {
            features.push(f.parse::<f32>().unwrap_or(0.0));
        }
        labels.push(class_id[cols[n_features + 1]]);
    }
    let n = labels.len();

    let mut adj = vec![0.0f32; n * n];
    for i in 0..n {
        adj[i * n + i] = 1.0;
    }
    for line in cites.lines().filter(|l| !l.trim().is_empty()) {
        let mut it = line.split_whitespace();
        let (a, b) = (it.next().unwrap(), it.next().unwrap());
        if let (Some(&i), Some(&j)) = (id_to_idx.get(a), id_to_idx.get(b)) {
            adj[i * n + j] = 1.0;
            adj[j * n + i] = 1.0;
        }
    }

    // D~^{-1/2} (A + I) D~^{-1/2}
    let mut deg = vec![0.0f32; n];
    for i in 0..n {
        deg[i] = (0..n).map(|j| adj[i * n + j]).sum();
    }
    let inv_sqrt: Vec<f32> = deg
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();
    let mut adj_norm = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let a = adj[i * n + j];
            if a != 0.0 {
                adj_norm[i * n + j] = inv_sqrt[i] * a * inv_sqrt[j];
            }
        }
    }

    Ok(Graph {
        n,
        n_features,
        n_classes,
        features,
        labels,
        adj_norm,
    })
}

/// 2-layer GCN. Both fields are `Module`s, so the whole net is trainable.
#[derive(Module, Debug)]
struct Gcn<B: Backend> {
    gc1: GCNConv<B>,
    gc2: GCNConv<B>,
}

impl<B: Backend> Gcn<B> {
    fn init(n_features: usize, n_classes: usize, device: &B::Device) -> Self {
        Self {
            gc1: GCNConv::init(n_features, HIDDEN, device),
            gc2: GCNConv::init(HIDDEN, n_classes, device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.gc1.forward(x, adj.clone());
        let h = activation::relu(h);
        self.gc2.forward(h, adj)
    }
}

/// Fraction of `idx` nodes whose argmax logit matches the label.
fn accuracy(logits: &[f32], labels: &[i32], idx: &[usize], n_classes: usize) -> f64 {
    let mut correct = 0;
    for &i in idx {
        let row = &logits[i * n_classes..(i + 1) * n_classes];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as i32;
        if pred == labels[i] {
            correct += 1;
        }
    }
    correct as f64 / idx.len() as f64
}

/// Deterministic xorshift shuffle (no rng dependency).
fn shuffle(v: &mut [usize], state: &mut u64) {
    for i in (1..v.len()).rev() {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        v.swap(i, (*state % (i as u64 + 1)) as usize);
    }
}

/// 20-per-class train split and a 1000-node test split from the remainder.
fn split(labels: &[i32], n_classes: usize) -> (Vec<usize>, Vec<usize>) {
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, &c) in labels.iter().enumerate() {
        by_class[c as usize].push(i);
    }
    let mut train = Vec::new();
    for bucket in &mut by_class {
        shuffle(bucket, &mut rng);
        train.extend(bucket.iter().take(20).copied());
    }
    let train_set: std::collections::HashSet<usize> = train.iter().copied().collect();
    let mut rest: Vec<usize> = (0..labels.len())
        .filter(|i| !train_set.contains(i))
        .collect();
    shuffle(&mut rest, &mut rng);
    let test = rest.into_iter().take(1000).collect();
    (train, test)
}

fn train<B: AutodiffBackend>(device: B::Device, dir: &Path, name: &str) {
    let g = load_planetoid(dir, name).unwrap();
    let (train_idx, test_idx) = split(&g.labels, g.n_classes);
    println!(
        "dataset: {name}  nodes: {}  features: {}  classes: {}  train: {}  test: {}",
        g.n,
        g.n_features,
        g.n_classes,
        train_idx.len(),
        test_idx.len()
    );

    let x = Tensor::<B, 2>::from_data(
        TensorData::new(g.features.clone(), [g.n, g.n_features]),
        &device,
    );
    let adj = Tensor::<B, 2>::from_data(TensorData::new(g.adj_norm.clone(), [g.n, g.n]), &device);
    let targets = Tensor::<B, 1, Int>::from_data(TensorData::new(g.labels.clone(), [g.n]), &device);
    let train_sel = Tensor::<B, 1, Int>::from_data(
        TensorData::new(
            train_idx.iter().map(|&i| i as i32).collect::<Vec<_>>(),
            [train_idx.len()],
        ),
        &device,
    );

    let mut model = Gcn::<B>::init(g.n_features, g.n_classes, &device);
    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(5e-4)))
        .init();
    let lr = 0.01;
    let epochs = 200;

    for epoch in 1..=epochs {
        let logits = model.forward(x.clone(), adj.clone());
        let train_logits = logits.clone().select(0, train_sel.clone());
        let train_targets = targets.clone().select(0, train_sel.clone());
        let loss = CrossEntropyLoss::new(None, &device).forward(train_logits, train_targets);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);

        if epoch % 20 == 0 || epoch == 1 {
            let logits_v = logits.into_data().to_vec::<f32>().unwrap();
            let tr = accuracy(&logits_v, &g.labels, &train_idx, g.n_classes);
            let te = accuracy(&logits_v, &g.labels, &test_idx, g.n_classes);
            let loss_v = loss.into_data().to_vec::<f32>().unwrap()[0];
            println!("epoch {epoch:>3}  loss {loss_v:.4}  train acc {tr:.4}  test acc {te:.4}");
        }
    }

    let logits_v = model.forward(x, adj).into_data().to_vec::<f32>().unwrap();
    println!(
        "\nfinal test accuracy: {:.4}",
        accuracy(&logits_v, &g.labels, &test_idx, g.n_classes)
    );
}

fn main() -> ExitCode {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cora".to_string());
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(&name);
    if !dir.join(format!("{name}.content")).exists() {
        eprintln!(
            "dataset not found at {}\nrun: ./scripts/fetch_{name}.sh",
            dir.display()
        );
        return ExitCode::SUCCESS;
    }
    train::<Autodiff<NdArray<f32>>>(Default::default(), &dir, &name);
    ExitCode::SUCCESS
}
