use std::path::{Path, PathBuf};

use clap::Args;
use mininn_verifier::{
    interpreters::{
        EvalError, Interpreter,
        bounds::{ibp::IBPInterpreter, ibp_util::IBPTensor},
    },
    mininn::{ComputeGraph, load_mininn, write_output_bin},
};

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
    pub fn parse_inputs(&self, graph: &ComputeGraph) -> Result<Vec<IBPTensor>, EvalError> {
        let tokens = &self.input_spec;
        let mut specs = Vec::new();
        let mut i = 0;

        for var in &graph.invars {
            if i >= tokens.len() {
                return Err(EvalError::Eval(format!(
                    "network has {} input(s), but only {} input spec(s) were provided",
                    graph.invars.len(),
                    specs.len()
                )));
            }
            match tokens[i].as_str() {
                "box" => {
                    if i + 2 >= tokens.len() {
                        return Err(EvalError::Eval(
                            "'box' marker requires two file paths (lb, ub)".into(),
                        ));
                    }
                    let lb = super::load_input_as_tensor(Path::new(&tokens[i + 1]), &var.shape)?;
                    let ub = super::load_input_as_tensor(Path::new(&tokens[i + 2]), &var.shape)?;
                    specs.push(IBPTensor::new(lb, ub));
                    i += 3;
                }
                "point" => {
                    if i + 1 >= tokens.len() {
                        return Err(EvalError::Eval(
                            "'point' marker requires one file path".into(),
                        ));
                    }
                    let arr = super::load_input_as_tensor(Path::new(&tokens[i + 1]), &var.shape)?;
                    specs.push(IBPTensor::new(arr.clone(), arr));
                    i += 2;
                }
                other => {
                    return Err(EvalError::Eval(format!(
                        "unknown input marker `{other}` (expected one of 'box', 'point')"
                    )));
                }
            }
        }

        if i != tokens.len() {
            return Err(EvalError::Eval(format!(
                "trailing arguments after parsing {} input(s): {:?}",
                graph.invars.len(),
                &tokens[i..]
            )));
        }

        Ok(specs)
    }
}

pub fn run_bounds(args: BoundsArgs) -> Result<(), EvalError> {
    let graph = load_mininn(args.mininn_file.as_path())?;
    let inputs = args.parse_inputs(&graph)?;

    std::fs::create_dir_all(&args.output_dir)?;

    let outputs = IBPInterpreter::run(&graph, &inputs)?;

    for (i, tensor) in outputs.iter().enumerate() {
        let lb_path = args.output_dir.join(format!("output_{i}_lb.bin"));
        let ub_path = args.output_dir.join(format!("output_{i}_ub.bin"));

        write_output_bin(&lb_path, &tensor.lb.iter().copied().collect::<Vec<_>>())?;
        write_output_bin(&ub_path, &tensor.ub.iter().copied().collect::<Vec<_>>())?;

        println!("{}", lb_path.display());
        println!("{}", ub_path.display());
    }

    Ok(())
}
