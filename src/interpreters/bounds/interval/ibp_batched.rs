use crate::{
    interpreters::{
        EvalError, Interpreter,
        bounds::interval::ibp_util::IBPBatchedTensor,
    },
    mininn::{Atom, ComputeGraph, Env, Primitive, Value},
};

pub struct IBPBatchedInterpreter {}

impl IBPBatchedInterpreter {
    pub fn process_primitive(
        primitive: &Primitive,
        env: &Env<IBPBatchedTensor>,
    ) -> Result<IBPBatchedTensor, EvalError> {
        let r = |a: &Atom| env.resolve(a);

        use Primitive::*;
        Ok(match primitive {
            Neg(a) => -r(a)?,
            Reciprocal(a) => r(a)?.reciprocal(),
            Square(a) => r(a)?.square(),
            Sqrt(a) => r(a)?.sqrt(),
            Exp(a) => r(a)?.exp(),
            Log(a) => r(a)?.log(),
            Add(a, b) => r(a)? + r(b)?,
            Mul(a, b) => r(a)? * r(b)?,
            Where(c, x, y) => IBPBatchedTensor::r#where(&r(c)?, &r(x)?, &r(y)?)?,
            Relu(a) => r(a)?.relu(),
            LeakyRelu { operand, slope } => r(operand)?.leaky_relu(*slope),
            Elu { operand, slope } => r(operand)?.elu(*slope),
            Gelu(a) => r(a)?.gelu(),
            NormalCdf(a) => r(a)?.normcdf(),
            Dot(a, b) => r(a)?.dot(&r(b)?)?,
            ReduceSum { operand, axes } => r(operand)?.reduce_sum(axes),
            ExpandDims { operand, axes } => r(operand)?.expand_dims(axes),
            MoveAxis { operand, source, destination } => r(operand)?.moveaxis(*source, *destination),
            Reshape { operand, new_shape } => r(operand)?.reshape(new_shape)?,
            Slice { operand, axis, start, end, step } => r(operand)?.slice(*axis, *start, *end, *step),
            Pad { operand, options } => r(operand)?.pad(options),
            Conv { input, kernel, options } => r(input)?.conv(&r(kernel)?, options.stride)?,
            ConvKernelGrad { grad_out, input, options, kernel_shape } =>
                r(grad_out)?.conv_kernel_grad(&r(input)?, options.stride, kernel_shape)?,
            AvgPool { operand, options } => r(operand)?.pool(options, true)?,
            SumPool { operand, options } => r(operand)?.pool(options, false)?,
        })
    }
}

impl Interpreter<IBPBatchedTensor> for IBPBatchedInterpreter {
    fn run(
        graph: &ComputeGraph,
        inputs: &Vec<IBPBatchedTensor>,
    ) -> Result<Vec<IBPBatchedTensor>, EvalError> {
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
