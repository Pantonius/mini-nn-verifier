use ndarray::{ArrayD, IxDyn, array};

use crate::{
    interpreters::{EvalError, EvalInterpreter, Interpreter, Tensor},
    mininn::{Atom, AtomKind, ComputeGraph, Env, Primitive},
};

/// A tangent value in the `grad` interpreter.
pub type Tangent = ArrayD<f64>;

pub struct GradInterpreter;

impl GradInterpreter {
    pub fn new() -> Self {
        GradInterpreter
    }

    fn resolve(atom: &Atom, env: &Env<Tangent>) -> Result<Tangent, EvalError> {
        match &atom.kind {
            AtomKind::Const(data) => ArrayD::from_shape_vec(IxDyn(&atom.shape), data.clone())
                .map_err(|e| EvalError::Eval(format!("const {}: {e}", atom.name))),
            AtomKind::Var => env
                .get(&atom.name)
                .cloned()
                .ok_or_else(|| EvalError::Eval(format!("undefined variable '{}'", atom.name))),
        }
    }

    fn process_primitive(
        &self,
        primitive: &Primitive,
        outvar: &Atom,
        env: &Env<Tangent>,
    ) -> Result<Vec<Tangent>, EvalError> {
        let tangent = Self::resolve(outvar, env)?;

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

impl Interpreter<Tangent> for GradInterpreter {
    /// Evaluate `graph` on `inputs` (one flat buffer per input var, in graph
    /// order) and return the output tensors flattened in row-major order.
    fn run(
        &mut self,
        graph: &ComputeGraph,
        inputs: &Vec<Tangent>,
    ) -> Result<Vec<Tangent>, EvalError> {
        // ---- FORWARD (primals) ----
        let eval_interp = EvalInterpreter::new();
        let mut primals = Env::<Tensor>::new();

        for (var, tensor) in graph.invars.iter().zip(inputs) {
            primals.insert(var.name.clone(), tensor.clone());
        }

        for eqn in &graph.equations {
            let mut out = eval_interp.process_primitive(&eqn.primitive, &eqn.outvar, &primals)?;
            assert_eq!(out.len(), 1);

            primals.insert(eqn.outvar.name.clone(), out.remove(0));
        }

        // ---- BACKWARD ----
        let mut env = Env::<Tangent>::new();

        fn combine(var_name: String, tangent: Tangent, env: &mut Env<Tangent>) {
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
            let out = self.process_primitive(&eqn.primitive, &eqn.outvar, &env)?;

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
