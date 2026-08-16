mod eval;
pub use eval::*;
use ndarray::ArrayD;

mod grad;
// pub use grad::*;

use crate::mininn::{ComputeGraph, MinninError, Value};

pub trait Interpreter<T: Value> {
    fn run(
        &mut self,
        graph: &ComputeGraph,
        inputs: &Vec<ArrayD<T>>,
    ) -> Result<Vec<ArrayD<T>>, EvalError>;
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error(transparent)]
    Mininn(#[from] MinninError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("eval error: {0}")]
    Eval(String),
}
