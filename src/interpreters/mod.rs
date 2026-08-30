pub mod bounds;
pub mod compute_graph;
pub mod concrete;

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
