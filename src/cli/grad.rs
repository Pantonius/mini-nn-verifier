use std::path::PathBuf;

use clap::Args;
use mininn_verifier::{
    interpreters::{EvalError, Interpreter, concrete::grad::GradInterpreter},
    mininn::{load_mininn, write_output_bin},
};

#[derive(Args)]
pub struct GradArgs {
    /// Directory to write output `.bin` files into.
    #[arg(long)]
    output_dir: PathBuf,

    /// The `.mininn` network file to load.
    mininn_file: PathBuf,

    /// Input `.bin` files, one per network input variable (in graph order).
    input_files: Vec<PathBuf>,
}

pub fn run_grad(args: GradArgs) -> Result<(), EvalError> {
    let graph = load_mininn(args.mininn_file.as_path())?;

    if args.input_files.len() != graph.invars.len() {
        return Err(EvalError::Eval(format!(
            "network expects {} input(s), but {} input file(s) were provided",
            graph.invars.len(),
            args.input_files.len()
        )));
    }

    let inputs = graph
        .invars
        .iter()
        .zip(&args.input_files)
        .map(|(var, path)| super::load_input_as_tensor(path.as_path(), &var.shape))
        .collect::<Result<Vec<_>, _>>()?;

    std::fs::create_dir_all(&args.output_dir)?;

    let outputs = GradInterpreter::run(&graph, &inputs)?;

    for (i, tensor) in outputs.iter().enumerate() {
        let path = args.output_dir.join(format!("output_{i}.bin"));
        let values: Vec<f64> = tensor.iter().copied().collect();
        write_output_bin(&path, &values)?;
        println!("{}", path.display());
    }

    Ok(())
}
