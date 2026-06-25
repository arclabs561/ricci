//! 2-layer GCN node classification on Cora (Kipf & Welling 2017): classify each
//! paper into 7 topics from bag-of-words features plus the citation graph.
//!
//! ```sh
//! ./scripts/fetch_cora.sh
//! cargo run --release --example cora_node_classification
//! ```
//!
//! Two [`GCNConv`] layers (`A_hat @ (X W)`) with a ReLU between, full-batch Adam
//! with weight decay, cross-entropy on a 20-per-class split, accuracy on a
//! 1000-node test split. `A_hat = D~^{-1/2} (A + I) D~^{-1/2}`. Expect test
//! accuracy around 0.80.

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

use propago::GCNConv;

const N_FEATURES: usize = 1433;
const N_CLASSES: usize = 7;
const HIDDEN: usize = 16;

/// Parsed Cora graph: dense feature matrix, labels, and normalized adjacency.
struct Cora {
    n: usize,
    features: Vec<f32>, // [n * N_FEATURES] row-major
    labels: Vec<i32>,   // [n]
    adj_norm: Vec<f32>, // [n * n] row-major, symmetric-normalized with self-loops
}

fn load_cora(dir: &Path) -> std::io::Result<Cora> {
    let content = std::fs::read_to_string(dir.join("cora.content"))?;
    let cites = std::fs::read_to_string(dir.join("cora.cites"))?;

    // Stable class ids from sorted label names.
    let mut label_names: Vec<&str> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.rsplit('\t').next().unwrap())
        .collect();
    label_names.sort_unstable();
    label_names.dedup();
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
        let paper_id = cols[0].to_string();
        let idx = id_to_idx.len();
        id_to_idx.insert(paper_id, idx);
        for f in &cols[1..=N_FEATURES] {
            features.push(f.parse::<f32>().unwrap());
        }
        labels.push(class_id[cols[N_FEATURES + 1]]);
    }
    let n = labels.len();

    // Symmetric adjacency with self-loops.
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

    Ok(Cora {
        n,
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
    fn init(device: &B::Device) -> Self {
        Self {
            gc1: GCNConv::init(N_FEATURES, HIDDEN, device),
            gc2: GCNConv::init(HIDDEN, N_CLASSES, device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.gc1.forward(x, adj.clone());
        let h = activation::relu(h);
        self.gc2.forward(h, adj)
    }
}

/// Fraction of `idx` nodes whose argmax logit matches the label.
fn accuracy(logits: &[f32], labels: &[i32], idx: &[usize]) -> f64 {
    let mut correct = 0;
    for &i in idx {
        let row = &logits[i * N_CLASSES..(i + 1) * N_CLASSES];
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
fn split(labels: &[i32]) -> (Vec<usize>, Vec<usize>) {
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); N_CLASSES];
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

fn train<B: AutodiffBackend>(device: B::Device) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cora");
    let cora = load_cora(&dir).unwrap();
    let (train_idx, test_idx) = split(&cora.labels);
    println!(
        "nodes: {}  features: {N_FEATURES}  classes: {N_CLASSES}  train: {}  test: {}",
        cora.n,
        train_idx.len(),
        test_idx.len()
    );

    let x = Tensor::<B, 2>::from_data(
        TensorData::new(cora.features.clone(), [cora.n, N_FEATURES]),
        &device,
    );
    let adj = Tensor::<B, 2>::from_data(
        TensorData::new(cora.adj_norm.clone(), [cora.n, cora.n]),
        &device,
    );
    let targets =
        Tensor::<B, 1, Int>::from_data(TensorData::new(cora.labels.clone(), [cora.n]), &device);
    let train_sel = Tensor::<B, 1, Int>::from_data(
        TensorData::new(
            train_idx.iter().map(|&i| i as i32).collect::<Vec<_>>(),
            [train_idx.len()],
        ),
        &device,
    );

    let mut model = Gcn::<B>::init(&device);
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
            let tr = accuracy(&logits_v, &cora.labels, &train_idx);
            let te = accuracy(&logits_v, &cora.labels, &test_idx);
            let loss_v = loss.into_data().to_vec::<f32>().unwrap()[0];
            println!("epoch {epoch:>3}  loss {loss_v:.4}  train acc {tr:.4}  test acc {te:.4}");
        }
    }

    let logits_v = model.forward(x, adj).into_data().to_vec::<f32>().unwrap();
    println!(
        "\nfinal test accuracy: {:.4}",
        accuracy(&logits_v, &cora.labels, &test_idx)
    );
}

fn main() -> ExitCode {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cora");
    if !dir.join("cora.content").exists() {
        eprintln!(
            "dataset not found at {}\nrun: ./scripts/fetch_cora.sh",
            dir.display()
        );
        return ExitCode::SUCCESS;
    }
    train::<Autodiff<NdArray<f32>>>(Default::default());
    ExitCode::SUCCESS
}
