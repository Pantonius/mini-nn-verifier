use clap::{Parser, Subcommand};

mod eval;
use eval::{EvalArgs, run_eval};

mod grad;
use grad::{GradArgs, run_grad};

mod train;
use train::{TrainArgs, run_train};

mod bounds;
use bounds::{BoundsArgs, run_bounds};

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
    /// Compute the gradient of the given inputs
    Grad(GradArgs),
    /// Trains a classifier on the given dataset
    Train(TrainArgs),
    /// Propagates interval bounds through the entire network (forward)
    Bounds(BoundsArgs),
}

fn main() -> Result<(), EvalError> {
    match Cli::parse().command {
        Command::Eval(args) => run_eval(args),
        Command::Grad(args) => run_grad(args),
        Command::Train(args) => run_train(args),
        Command::Bounds(args) => run_bounds(args),
    }
}
