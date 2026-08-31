use mininn_verifier::{
    interpreters::{
        EvalError,
        bounds::bab::{BaBConfig, BaBResult, node_splitting_bab, split_smart},
    },
    mininn::{load_mininn, write_output_bin},
};

use crate::verify::VerifyArgs;

pub fn run_verify2(args: VerifyArgs) -> Result<(), EvalError> {
    let graph = load_mininn(args.mininn_file.as_path())?;
    let inputs = args.parse_inputs(&graph)?;

    std::fs::create_dir_all(&args.output_dir)?;

    match node_splitting_bab(&graph, &inputs, split_smart, BaBConfig::default())? {
        // match input_splitting_bab(&graph, &inputs, uniform_split, BaBConfig::default())? {
        BaBResult::Safe => println!("sat"),
        BaBResult::Unsafe(cex) => {
            for (i, arr) in cex.iter().enumerate() {
                let path = args.output_dir.join(format!("counterexample_{i}.bin"));
                write_output_bin(&path, &arr.iter().copied().collect::<Vec<_>>())?;
                println!("{}", path.display());
            }
            println!("viol");
        }
        BaBResult::Undecided => {
            println!("unknown")
        }
    }

    Ok(())
}
