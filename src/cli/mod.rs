use clap::{Parser, Subcommand};

mod eval;
use eval::{EvalArgs, run_eval};

mod grad;
use grad::{GradArgs, run_grad};

mod train;
use train::{TrainArgs, run_train};

mod bounds;
use bounds::{BoundsArgs, run_bounds};

mod verify;
use verify::{VerifyArgs, run_verify};

mod affine_bounds;
use affine_bounds::{AffineBoundsArgs, run_affine_bounds};

mod verify2;
use verify2::{Verify2Args, run_verify2};

use mininn_verifier::{
    interpreters::{EvalError, concrete::eval_util::Tensor},
    mininn::load_input_as_arr,
};

fn load_input_as_tensor(path: &std::path::Path, shape: &[usize]) -> Result<Tensor, EvalError> {
    Ok(load_input_as_arr(path, shape)?.into())
}

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
    /// Input-Splitting BaB verification of non-zero output
    Verify(VerifyArgs),
    /// Propagates affine bounds through the entire network (backward)
    #[command(name = "affine_bounds")]
    AffineBounds(AffineBoundsArgs),
    ///
    Verify2(Verify2Args),
}

fn main() -> Result<(), EvalError> {
    match Cli::parse().command {
        Command::Eval(args) => run_eval(args),
        Command::Grad(args) => run_grad(args),
        Command::Train(args) => run_train(args),
        Command::Bounds(args) => run_bounds(args),
        Command::Verify(args) => run_verify(args),
        Command::AffineBounds(args) => run_affine_bounds(args),
        Command::Verify2(args) => run_verify2(args),
    }
}
