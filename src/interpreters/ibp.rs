use crate::{
    interpreters::{
        EvalError, Interpreter,
        ibp_util::{Bound, IBPTensor},
    },
    mininn::{ComputeGraph, Env, Primitive},
};

pub struct IBPInterpreter {}

impl IBPInterpreter {
    fn process_primitive(primitive: &Primitive, env: &Env<Bound>) -> Result<IBPTensor, EvalError> {
        todo!()
    }
}

impl Interpreter<Bound> for IBPInterpreter {
    fn run(
        graph: &ComputeGraph,
        inputs: &Vec<IBPTensor>,
    ) -> Result<Vec<IBPTensor>, super::EvalError> {
        todo!()
    }
}
