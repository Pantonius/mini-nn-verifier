use clap::{Parser, Subcommand};

mod eval;
use eval::{EvalArgs, run_eval};
use mininn_verifier::interpreters::EvalError;

/// mininnverifier-compatible CLI. The testrunner invokes the binary as
/// `<prog> <command> ...`, so each command is a subcommand.
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Evaluate a network on the given inputs and write the output tensors.
    Eval(EvalArgs),
}

fn main() -> Result<(), EvalError> {
    match Cli::parse().command {
        Command::Eval(args) => run_eval(args),
    }
}
