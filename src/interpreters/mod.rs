mod eval_util;
pub use eval_util::Tensor;

mod eval;
pub use eval::EvalInterpreter;

mod grad;
pub use grad::GradInterpreter;

mod ibp_util;
pub use ibp_util::{IBPBatchedTensor, IBPTensor};

mod ibp;
pub use ibp::IBPInterpreter;

mod ibp_batched;
pub use ibp_batched::IBPBatchedInterpreter;

// mod ibp_batched_sensitivity;
// pub use ibp_batched_sensitivity::IBPBatchedSensitivityInterpreter;

mod bab;
pub use bab::{BaBConfig, BaBResult, input_splitting_bab, uniform_split};

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
