mod eval;
pub use eval::*;

mod grad;
pub use grad::*;

use crate::mininn::{Env, Primitive, Value};

/// An interpreter maps a [`Primitive`] (whose operands are already this
/// interpreter's own value type `T`) to a single output value.
pub trait Interpreter<T: Value> {
    fn process_primitive(&mut self, primitive: Primitive, env: &Env<T>) -> Result<T, EvalError>;
}
