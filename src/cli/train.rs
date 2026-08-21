use std::{
    fs::{self, File},
    io::{BufReader, Write as IoWrite},
    path::{Path, PathBuf},
};

use clap::Args;
use ndarray::{Array1, Array2};
use rand::{RngExt, SeedableRng, rngs::Xoshiro128PlusPlus};
use rand_distr::Uniform;
use serde::Deserialize;

use mininn_verifier::{
    interpreters::{EvalError, GradInterpreter},
    mininn::{ComputeGraph, Layer, MininnError, encode_f64, init_mlp, load_input_as_arr2},
};

#[derive(Args)]
pub struct TrainArgs {
    /// Directory to write checkpoint `.mininn` files into.
    #[arg(long)]
    output_dir: PathBuf,
    /// Dataset identifier (selects hyperparameters from ./hyperparams/<id>.json).
    dataset: String,
    /// Float64 input array (N, in_size), row-major binary.
    train_inputs: PathBuf,
    /// Float64 one-hot label array (N, num_classes), row-major binary.
    train_labels: PathBuf,
}

#[derive(Deserialize, Debug)]
struct Hyperparams {
    /// input size
    in_size: usize,
    /// size of each layer
    layer_sizes: Vec<usize>,
    /// number of epochs for training
    num_epochs: usize,
    /// batch size for training
    batch_size: usize,
    /// learning rate for gradient descent
    learning_rate: f64,
    /// batch size for validation / evaluation
    eval_batch_size: usize,
    /// Seed for random number generator (RNG)
    rng_key: u64,
}

fn load_hyperparams(id: &str) -> Result<Hyperparams, MininnError> {
    let path = PathBuf::from(format!("./hyperparams/{id}.json"));
    let file = File::open(path)?;
    Ok(serde_json::from_reader(BufReader::new(file))?)
}

// ---------------------------------------------------------------------------
// Adam optimizer
// ---------------------------------------------------------------------------

struct AdamState {
    /// first moment (mean) of weights
    mw: Vec<Array2<f64>>,
    /// second moment (variance) of weights
    vw: Vec<Array2<f64>>,
    /// first moment (mean) of biases
    mb: Vec<Array1<f64>>,
    /// second moment (variance) of biases
    vb: Vec<Array1<f64>>,
    /// decay rate for first moment (mean)
    beta1powt: f64,
    /// decay rate for second moment (variance)
    beta2powt: f64,
}

impl AdamState {
    fn new(layers: &[Layer]) -> Self {
        Self {
            mw: layers
                .iter()
                .map(|l| Array2::zeros(l.w.raw_dim()))
                .collect(),
            vw: layers
                .iter()
                .map(|l| Array2::zeros(l.w.raw_dim()))
                .collect(),
            mb: layers.iter().map(|l| Array1::zeros(l.b.len())).collect(),
            vb: layers.iter().map(|l| Array1::zeros(l.b.len())).collect(),
            beta1powt: 1.0,
            beta2powt: 1.0,
        }
    }

    /// as in the original python implementation
    fn step(
        &mut self,
        layers: &mut [Layer],
        grads_w: &[Array2<f64>],
        grads_b: &[Array1<f64>],
        lr: f64,
    ) {
        // default values
        const B1: f64 = 0.9;
        const B2: f64 = 0.999;
        const EPS: f64 = 1e-8;

        // update beta1powt / beta2powt
        self.beta1powt *= B1;
        self.beta2powt *= B2;
        let bc1 = 1.0 - self.beta1powt;
        let bc2 = 1.0 - self.beta2powt;

        // for each layer...
        for i in 0..layers.len() {
            // ... update weight
            self.mw[i] = &self.mw[i] * B1 + &grads_w[i] * (1.0 - B1);
            self.vw[i] = &self.vw[i] * B2 + grads_w[i].mapv(|x| x * x) * (1.0 - B2);

            let mw_hat = &self.mw[i] / bc1;
            let vw_hat = &self.vw[i] / bc2;

            let dw = mw_hat / (vw_hat.mapv(f64::sqrt) + EPS) * lr;
            layers[i].w -= &dw;

            // ... and update bias
            self.mb[i] = &self.mb[i] * B1 + &grads_b[i] * (1.0 - B1);
            self.vb[i] = &self.vb[i] * B2 + grads_b[i].mapv(|x| x * x) * (1.0 - B2);

            let mb_hat = &self.mb[i] / bc1;
            let vb_hat = &self.vb[i] / bc2;

            let db = mb_hat / (vb_hat.mapv(f64::sqrt) + EPS) * lr;
            layers[i].b -= &db;
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint saving
// ---------------------------------------------------------------------------

fn save_checkpoint(
    path: &Path,
    layers: &[Layer],
    eval_batch_size: usize,
    in_size: usize,
) -> Result<(), EvalError> {
    // Build graph.txt — parameters embedded as named constants (uppercase).
    let mut graph = format!("input: x[{eval_batch_size},{in_size}]");
    let mut prev = format!("x[{eval_batch_size},{in_size}]");
    let mut prev_size = in_size;

    for (i, layer) in layers.iter().enumerate() {
        let li = i + 1;
        let out_size = layer.b.len();
        let h = format!("h{li}[{eval_batch_size},{out_size}]");
        let a = format!("a{li}[{eval_batch_size},{out_size}]");
        let w = format!("W{li}[{prev_size},{out_size}]");
        let b = format!("B{li}[{out_size}]");
        graph += &format!("\n{h} = dot {{}} {prev} {w}");
        graph += &format!("\n{a} = add {{}} {h} {b}");
        if i < layers.len() - 1 {
            let r = format!("r{li}[{eval_batch_size},{out_size}]");
            graph += &format!("\n{r} = relu {{}} {a}");
            prev = r;
        } else {
            prev = a;
        }
        prev_size = out_size;
    }
    graph += &format!("\noutput: {prev}");

    let file = File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default();

    zip.start_file("graph.txt", opts)
        .map_err(|e| EvalError::Eval(e.to_string()))?;
    zip.write_all(graph.as_bytes())?;

    for (i, layer) in layers.iter().enumerate() {
        let li = i + 1;
        zip.start_file(format!("W{li}.bin"), opts)
            .map_err(|e| EvalError::Eval(e.to_string()))?;
        zip.write_all(&encode_f64(&layer.w.iter().copied().collect::<Vec<_>>()))?;
        zip.start_file(format!("B{li}.bin"), opts)
            .map_err(|e| EvalError::Eval(e.to_string()))?;
        zip.write_all(&encode_f64(&layer.b.iter().copied().collect::<Vec<_>>()))?;
    }

    zip.finish().map_err(|e| EvalError::Eval(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run_train(args: TrainArgs) -> Result<(), EvalError> {
    // Load configuration / hyperparameters for MLP
    let hyperparams = load_hyperparams(&args.dataset)?;
    let d = hyperparams.in_size;
    let c = *hyperparams
        .layer_sizes
        .last()
        .expect("layer_sizes must be non-empty");

    // Load input data and labels
    let input_data = load_input_as_arr2(&args.train_inputs, d)?;
    let input_labels = load_input_as_arr2(&args.train_labels, c)?;

    if input_data.shape()[0] != input_labels.shape()[0] {
        return Err(MininnError::SizeMismatch {
            expected: input_data.shape()[0],
            shape: input_data.shape().to_vec(),
            got: input_labels.shape()[0],
        })?;
    }

    let num_samples = input_data.shape()[0];

    // Initialize layers of MLP
    let mut layers = init_mlp(d, hyperparams.layer_sizes, hyperparams.rng_key)?;

    // Initialize optimizer (Adam)
    let mut opt = AdamState::new(&layers);

    // Create output_dir
    fs::create_dir_all(&args.output_dir)?;

    // print eval_batch_size as specified in milestone doc
    println!("eval_batch_size: {}", hyperparams.eval_batch_size);

    // prepare sample for per epoch shuffle
    let mut rng = Xoshiro128PlusPlus::seed_from_u64(hyperparams.rng_key);
    let distr = Uniform::new(0, num_samples - 1).map_err(|e| MininnError::RandUniform(e))?;

    let mut idx: Vec<usize> = (0..num_samples).collect();

    // run epochs
    let batch_size = hyperparams.batch_size;
    for epoch in 1..=hyperparams.num_epochs {
        // Shuffle input samples
        for i in (0..num_samples).rev() {
            // swap each i with a j before it (or with itself)
            let j = rng.sample(distr) % (i + 1);
            idx.swap(i, j);
        }

        // generate compute graph from layers
        let cg = ComputeGraph::from_layers(&layers, batch_size, d);

        // batchwise
        let mut pos = 0;
        while pos + batch_size <= num_samples {
            let batch = &idx[pos..pos + batch_size];
            let x = Array2::from_shape_fn((batch_size, d), |(i, j)| input_data[[batch[i], j]]);
            let y = Array2::from_shape_fn((batch_size, c), |(i, j)| input_labels[[batch[i], j]]);

            // inputs: [x, W0, b0, W1, b1, ...]
            let mut inputs = vec![x.clone().into_dyn()];
            for layer in &layers {
                inputs.push(layer.w.clone().into_dyn());
                inputs.push(layer.b.clone().into_dyn());
            }

            let grads = GradInterpreter::run_loss(&cg, &inputs, Some(&y.clone().into_dyn()))?;

            // grads[0] = dx (unused), grads[1 + 2i] = dWi, grads[2 + 2i] = dbi
            let grads_w: Vec<Array2<f64>> = (0..layers.len())
                .map(|i| {
                    grads[1 + 2 * i]
                        .clone()
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap()
                })
                .collect();
            let grads_b: Vec<Array1<f64>> = (0..layers.len())
                .map(|i| {
                    grads[2 + 2 * i]
                        .clone()
                        .into_dimensionality::<ndarray::Ix1>()
                        .unwrap()
                })
                .collect();

            // optimizer: walks grad
            opt.step(&mut layers, &grads_w, &grads_b, hyperparams.learning_rate);

            pos += batch_size;
        }

        // save epoch
        let cp_path = args
            .output_dir
            .join(format!("checkpoint_epoch_{epoch}.mininn"));
        save_checkpoint(&cp_path, &layers, hyperparams.eval_batch_size, d)?;
        println!("{}", cp_path.canonicalize()?.display());
    }

    Ok(())
}
