mod eval_util;

mod eval;
pub use eval::*;

mod grad;
pub use grad::*;

mod ibp_util;
pub use ibp_util::*;

mod ibp;
pub use ibp::*;

use crate::mininn::{ComputeGraph, MininnError, Value};

pub trait Interpreter<V: Value> {
    fn run(graph: &ComputeGraph, inputs: &Vec<V>) -> Result<Vec<V>, EvalError>;
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error(transparent)]
    Mininn(#[from] MininnError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("eval error: {0}")]
    Eval(String),
}
