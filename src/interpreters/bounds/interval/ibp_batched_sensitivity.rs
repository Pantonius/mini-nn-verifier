use crate::{
    interpreters::{EvalError, IBPBatchedInterpreter, IBPTensor, Interpreter, ibp_util::IBPBatchedTensor},
    mininn::{Atom, ComputeGraph, Env, Primitive},
};

pub struct IBPBatchedSensitivityInterpreter {}

impl IBPBatchedSensitivityInterpreter {
    fn process_primitive(
        primitive: &Primitive,
        inter_bounds: &Env<IBPBatchedTensor>,
        env: &Env<IBPBatchedTensor>,
        outvar: &Atom,
    ) -> Result<Vec<IBPBatchedTensor>, EvalError> {
        let ib = |a: &Atom| inter_bounds.resolve(a);

        let coeff = env.resolve(outvar)?;

        use Primitive::*;
        Ok(match primitive {
            Add(a, b) => vec![coeff.clone(), coeff],
            Dot(a, b) => ,
            Relu(x) => {
                let b = ib(x)?;                 // intermediate bounds on the pre-activation
                let gate_lb = b.lb.mapv(|l| if l >= 0.0 { 1.0 } else { 0.0 });
                let gate_ub = b.ub.mapv(|u| if u > 0.0  { 1.0 } else { 0.0 });
                let gate = IBPBatchedTensor { lb: gate_lb, ub: gate_ub };

                vec![coeff * gate]
            },
            _ => unimplemented!()
        })
    }
}

impl Interpreter<IBPBatchedTensor> for IBPBatchedSensitivityInterpreter {
    fn run(
        graph: &ComputeGraph,
        inputs: &Vec<IBPBatchedTensor>,
    ) -> Result<Vec<IBPBatchedTensor>, EvalError> {
        // Forward
        let mut inter_bounds = Env::new();

        for (var, tensor) in graph.invars.iter().zip(inputs) {
            inter_bounds.insert(var.name.clone(), tensor.clone());
        }

        for eqn in &graph.equations {
            let out = IBPBatchedInterpreter::process_primitive(&eqn.primitive, &inter_bounds)?;
            inter_bounds.insert(eqn.outvar.name.clone(), out);
        }

        // Backward

        graph
            .outvars
            .iter()
            .map(|var| {
                let tensor = env.get(&var.name).ok_or_else(|| {
                    EvalError::Eval(format!("output '{}' was never computed", var.name))
                })?;
                Ok(tensor.clone())
            })
            .collect()
    }
}
