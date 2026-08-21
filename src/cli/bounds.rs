use std::path::{Path, PathBuf};

use clap::Args;
use mininn_verifier::{
    interpreters::{Bound, EvalError, IBPInterpreter, Interpreter},
    mininn::{load_input_as_f64, load_mininn},
};
use ndarray::{Array1, ArrayD};

#[derive(Args)]
pub struct BoundsArgs {
    /// Directory to write output `.bin` files into.
    #[arg(long)]
    output_dir: PathBuf,

    /// The `.mininn` network file to load.
    mininn_file: PathBuf,

    /// Input specifications: `box <lb.bin> <ub.bin>` or `point <x.bin>`, repeatable.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    input_spec: Vec<String>,
}

impl BoundsArgs {
    pub fn parse_inputs(&self) -> Result<Vec<ArrayD<Bound>>, EvalError> {
        let mut specs = Vec::new();
        let mut iter = self.input_spec.iter();
        while let Some(marker) = iter.next() {
            match marker.as_str() {
                "box" => {
                    let lb_str = iter
                        .next()
                        .ok_or_else(|| EvalError::Eval("expected lb path after `box`".into()))?;
                    let lb_vec = load_input_as_f64(Path::new(lb_str))?;

                    let ub_str = iter
                        .next()
                        .ok_or_else(|| EvalError::Eval("expected ub path after `box`".into()))?;
                    let ub_vec = load_input_as_f64(Path::new(ub_str))?;

                    let b = Array1::from_vec(
                        lb_vec
                            .iter()
                            .zip(ub_vec)
                            .map(|(&lb, ub)| Bound {
                                lb,
                                ub,
                                is_point: false,
                            })
                            .collect(),
                    );

                    specs.push(b.into_dyn());
                }
                "point" => {
                    let x_str = iter
                        .next()
                        .ok_or_else(|| EvalError::Eval("expected path after `point`".into()))?;
                    let x_vec = load_input_as_f64(Path::new(x_str))?;
                    let b = Array1::from_vec(
                        x_vec
                            .iter()
                            .map(|&x| Bound {
                                lb: x,
                                ub: x,
                                is_point: true,
                            })
                            .collect(),
                    );

                    specs.push(b.into_dyn());
                }
                other => {
                    return Err(EvalError::Eval(format!("unknown input marker `{other}`")));
                }
            }
        }
        Ok(specs)
    }
}

pub fn run_bounds(args: BoundsArgs) -> Result<(), EvalError> {
    let graph = load_mininn(args.mininn_file.as_path())?;
    let inputs = args.parse_inputs()?;

    std::fs::create_dir_all(&args.output_dir)?;

    let outputs = IBPInterpreter::run(&graph, &inputs);

    // TODO write out

    Ok(())
}
