use crate::mininn::Value;
use crate::{
    interpreters::{EvalError, Interpreter},
    mininn::{Atom, ComputeGraph, Env, Primitive},
};

pub struct EvalGenInterpreter;

impl EvalGenInterpreter {
    pub fn new() -> Self {
        EvalGenInterpreter
    }

    pub fn process_primitive<T: Value>(
        primitive: &Primitive,
        env: &Env<T>,
    ) -> Result<T, EvalError> {
        let r = |a: &Atom| env.resolve(a);

        use Primitive::*;
        Ok(match primitive {
            // elementwise unary
            Neg(a) => -r(a)?,
            Reciprocal(a) => r(a)?.reciprocal(),
            Square(a) => r(a)?.square(),
            Sqrt(a) => r(a)?.sqrt(),
            Exp(a) => r(a)?.exp(),
            Log(a) => r(a)?.log(),
            // elementwise binary (numpy broadcasting)
            Add(a, b) => r(a)? + r(b)?,
            Mul(a, b) => r(a)? * r(b)?,
            Where(c, x, y) => T::r#where(&r(c)?, &r(x)?, &r(y)?)?,
            // activations
            Relu(a) => r(a)?.relu(),
            LeakyRelu { operand, slope } => r(operand)?.leaky_relu(*slope),

            Elu { operand, slope } => r(operand)?.elu(*slope),
            Gelu(a) => r(a)?.gelu(),
            NormalCdf(a) => r(a)?.normcdf(),
            // linear algebra
            Dot(a, b) => r(a)?.dot(&r(b)?)?,
            // reduction
            ReduceSum { operand, axes } => r(operand)?.reduce_sum(axes),
            // shape manipulation
            ExpandDims { operand, axes } => r(operand)?.expand_dims(axes),
            MoveAxis {
                operand,
                source,
                destination,
            } => r(operand)?.moveaxis(*source, *destination),
            Reshape { operand, new_shape } => r(operand)?.reshape(&new_shape)?,
            // slicing
            Slice { operand, axis, start, end, step } => r(operand)?.slice(*axis, *start, *end, *step),
            // padding
            Pad { operand, options } => r(operand)?.pad(options),
            // 2d convolution
            Conv {
                input,
                kernel,
                options,
            } => r(&input)?.conv(&r(&kernel)?, options.stride)?,
            ConvKernelGrad { grad_out, input, options, kernel_shape } =>
                r(grad_out)?.conv_kernel_grad(&r(input)?, options.stride, kernel_shape)?,
            // pooling
            AvgPool { operand, options } => r(operand)?.pool(options, true)?,
            SumPool { operand, options } => r(operand)?.pool(options, false)?,
        })
    }
}

impl<T: Value> Interpreter<T> for EvalGenInterpreter {
    fn run(graph: &ComputeGraph, inputs: &Vec<T>) -> Result<Vec<T>, EvalError> {
        let mut env = Env::new();

        for (var, tensor) in graph.invars.iter().zip(inputs) {
            env.insert(var.name.clone(), tensor.clone());
        }

        for eqn in &graph.equations {
            let out = Self::process_primitive(&eqn.primitive, &env)?;
            env.insert(eqn.outvar.name.clone(), out);
        }

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
