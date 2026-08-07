mod core;
mod mininn;

use std::path::{Path, PathBuf};

use crate::mininn::{MinninError, load_input_bin, load_mininn};
use clap::Parser;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Eval {
    #[arg(long)]
    output_dir: PathBuf,

    mininn_file: PathBuf,
    input_files: Vec<PathBuf>,
}

fn main() -> Result<(), MinninError> {
    let args = Eval::parse();

    let graph = load_mininn(args.mininn_file.as_path())?;

    let input_paths: Vec<PathBuf> = args.input_files;

    let inputs: Vec<Vec<f64>> = graph
        .invars
        .iter()
        .zip(input_paths) // however many the network needs
        .map(|(var, buf)| load_input_bin(Path::new(buf.as_path()), &var.shape))
        .collect::<Result<_, _>>()?;

    println!("{:#?}", graph.invars);
    println!("loaded {} inputs", inputs.len());
    Ok(())
}
