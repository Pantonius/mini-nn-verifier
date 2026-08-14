use std::path::PathBuf;

use clap::Args;
use mininn_verifier::{
    interpreters::{EvalError, EvalInterpreter},
    mininn::{load_input_bin, load_mininn, write_output_bin},
};

#[derive(Args)]
pub struct EvalArgs {
    /// Directory to write output `.bin` files into.
    #[arg(long)]
    output_dir: PathBuf,

    /// The `.mininn` network file to load.
    mininn_file: PathBuf,

    /// Input `.bin` files, one per network input variable (in graph order).
    input_files: Vec<PathBuf>,
}

pub fn run_eval(args: EvalArgs) -> Result<(), EvalError> {
    let graph = load_mininn(args.mininn_file.as_path())?;

    if args.input_files.len() != graph.invars.len() {
        return Err(EvalError::Eval(format!(
            "network expects {} input(s), but {} input file(s) were provided",
            graph.invars.len(),
            args.input_files.len()
        )));
    }

    let inputs: Vec<Vec<f64>> = graph
        .invars
        .iter()
        .zip(&args.input_files)
        .map(|(var, path)| load_input_bin(path.as_path(), &var.shape))
        .collect::<Result<_, _>>()?;

    std::fs::create_dir_all(&args.output_dir)?;

    let outputs = EvalInterpreter::new().run(&graph, inputs)?;

    // Write each output as output_<i>.bin and print its path for the testrunner
    // (matched against expected_outputs by position).
    for (i, values) in outputs.iter().enumerate() {
        let path = args.output_dir.join(format!("output_{i}.bin"));
        write_output_bin(&path, values)?;
        println!("{}", path.display());
    }

    Ok(())
}
