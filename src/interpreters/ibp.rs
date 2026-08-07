use crate::core::{Primitive, Value};
use crate::interpreters::Interpreter;

struct IBPInterpreter {}

impl Interpreter for IBPInterpreter {
    fn process_primitive(&mut self, primitive: Primitive) -> Box<dyn Value> {
        todo!()
    }
}

struct IBPBox {
    lb: Box<dyn Value>,
    ub: Box<dyn Value>,
}
impl Value for IBPBox {}
