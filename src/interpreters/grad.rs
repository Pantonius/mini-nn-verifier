use ndarray::array;

use crate::{
    interpreters::{EvalError, EvalInterpreter, Interpreter, Tensor},
    mininn::{Atom, AtomKind, ComputeGraph, Env, Primitive},
};

pub struct GradInterpreter;

impl GradInterpreter {
    pub fn new() -> Self {
        GradInterpreter
    }

    fn process_primitive(
        &self,
        primitive: &Primitive,
        outvar: &Atom,
        primals: &Env<f64>,
        env: &Env<f64>,
    ) -> Result<Vec<Tensor>, EvalError> {
        let p = |a: &Atom| primals.resolve(a);

        let tangent = env.resolve(outvar)?;

        use Primitive::*;
        Ok(match primitive {
            // elementwise unary
            Neg(a) => vec![-tangent],
            Reciprocal(a) => todo!(),
            Square(a) => todo!(),
            Sqrt(a) => todo!(),
            Exp(a) => todo!(),
            Log(a) => todo!(),
            // elementwise binary (numpy broadcasting)
            Add(a, b) => vec![tangent.clone(), tangent],
            Mul(a, b) => todo!(),
            Where(c, x, y) => todo!(),
            // activations
            Relu(a) => todo!(),
            LeakyRelu { operand, slope } => todo!(),
            NormalCdf(a) => todo!(),
            // linear algebra
            Dot(a, b) => todo!(),
            // reduction
            ReduceSum { operand, axes } => todo!(),
            // shape manipulation
            ExpandDims { operand, axes } => todo!(),
            MoveAxis {
                operand,
                source,
                destination,
            } => todo!(),
            Reshape { operand, new_shape } => todo!(),
            // padding
            Pad { operand, options } => todo!(),
            // 2d convolution
            Conv {
                input,
                kernel,
                options,
            } => todo!(),
            // pooling
            AvgPool { operand, options } => todo!(),
            SumPool { operand, options } => todo!(),
        })
    }
}

impl Interpreter<f64> for GradInterpreter {
    /// Evaluate `graph` on `inputs` (one flat buffer per input var, in graph
    /// order) and return the output tensors flattened in row-major order.
    fn run(
        &mut self,
        graph: &ComputeGraph,
        inputs: &Vec<Tensor>,
    ) -> Result<Vec<Tensor>, EvalError> {
        // ---- FORWARD (primals) ----
        let eval_interp = EvalInterpreter::new();
        let mut primals = Env::<f64>::new();

        for (var, tensor) in graph.invars.iter().zip(inputs) {
            primals.insert(var.name.clone(), tensor.clone());
        }

        for eqn in &graph.equations {
            let out = eval_interp.process_primitive(&eqn.primitive, &primals)?;
            primals.insert(eqn.outvar.name.clone(), out);
        }

        // ---- BACKWARD ----
        let mut env = Env::<f64>::new();

        fn combine(var_name: String, tangent: Tensor, env: &mut Env<f64>) {
            if let Some(t) = env.get(&var_name) {
                env.update(var_name, t + tangent);
            } else {
                env.insert(var_name, tangent);
            }
        }

        for var in &graph.outvars {
            match var.kind {
                AtomKind::Var => {
                    env.insert(var.name.clone(), array![1.0].into_dyn());
                }
                AtomKind::Const(_) => continue,
            }
        }

        for eqn in graph.equations.iter().rev() {
            let out = self.process_primitive(&eqn.primitive, &eqn.outvar, &primals, &env)?;

            for (atom, tangent) in eqn.primitive.operands().into_iter().zip(out) {
                combine(atom.name.clone(), tangent, &mut env);
            }
        }

        graph
            .invars
            .iter()
            .map(|var| {
                let tangent = env.get(&var.name).ok_or_else(|| {
                    EvalError::Eval(format!("output '{}' was never computed", var.name))
                })?;
                Ok(tangent.clone())
            })
            .collect()
    }
}
