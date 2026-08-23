use crate::{
    interpreters::{
        EvalError, Interpreter,
        eval_util::{
            add, conv, dot, elu, expand_dims, leaky_relu, log, moveaxis, mul, normcdf, pad, pool,
            reduce_sum, relu, reshape,
        },
        ibp_gelu, ibp_linear, ibp_monotonic_non_dec, ibp_monotonic_non_dec2, ibp_monotonic_non_inc,
        ibp_reciprocal, ibp_square,
        ibp_util::IBPTensor,
        ibp_where,
    },
    mininn::{Atom, ComputeGraph, Env, Primitive},
};

pub struct IBPInterpreter {}

impl IBPInterpreter {
    fn process_primitive(
        primitive: &Primitive,
        env: &Env<IBPTensor>,
    ) -> Result<IBPTensor, EvalError> {
        let r = |a: &Atom| env.resolve(a);

        use Primitive::*;
        match primitive {
            // Non-Dec
            ExpandDims { operand, axes } => {
                ibp_monotonic_non_dec(|a| Ok(expand_dims(&a, axes)), &r(operand)?)
            }
            MoveAxis {
                operand,
                source,
                destination,
            } => ibp_monotonic_non_dec(|a| Ok(moveaxis(&a, *source, *destination)), &r(operand)?),
            Reshape { operand, new_shape } => {
                ibp_monotonic_non_dec(|a| reshape(&a, new_shape), &r(operand)?)
            }
            Pad { operand, options } => {
                ibp_monotonic_non_dec(|a| Ok(pad(&a, options)), &r(operand)?)
            }
            Add(aa, ba) => ibp_monotonic_non_dec2(|a, b| add(&a, &b), &r(aa)?, &r(ba)?),
            ReduceSum { operand, axes } => {
                ibp_monotonic_non_dec(|a| Ok(reduce_sum(&a, axes)), &r(operand)?)
            }
            Relu(atom) => ibp_monotonic_non_dec(|a| Ok(relu(&a)), &r(atom)?),
            LeakyRelu { operand, slope } => {
                ibp_monotonic_non_dec(|a| Ok(leaky_relu(&a, *slope)), &r(operand)?)
            }
            Elu { operand, slope } => ibp_monotonic_non_dec(|a| Ok(elu(&a, *slope)), &r(operand)?),
            Exp(atom) => ibp_monotonic_non_dec(|a| Ok(a.exp()), &r(atom)?),
            Log(atom) => ibp_monotonic_non_dec(|a| Ok(log(&a)), &r(atom)?),
            Sqrt(atom) => ibp_monotonic_non_dec(|a| Ok(a.sqrt()), &r(atom)?),
            SumPool { operand, options } => {
                ibp_monotonic_non_dec(|a| pool(&a, options, false), &r(operand)?)
            }
            AvgPool { operand, options } => {
                ibp_monotonic_non_dec(|a| pool(&a, options, true), &r(operand)?)
            }
            NormalCdf(atom) => ibp_monotonic_non_dec(|a| Ok(a.mapv(|x| normcdf(x))), &r(atom)?),
            // Non-Inc
            Neg(atom) => ibp_monotonic_non_inc(|a| Ok(a.mapv(|x| -x)), &r(atom)?),
            // Linear
            Dot(aa, ba) => ibp_linear(|a, b| dot(&a, &b), &r(aa)?, &r(ba)?),
            Mul(aa, ba) => ibp_linear(|a, b| mul(&a, &b), &r(aa)?, &r(ba)?),
            Conv {
                input,
                kernel,
                options,
            } => ibp_linear(|a, b| conv(&a, &b, options.stride), &r(input)?, &r(kernel)?),
            // Special cases
            Square(atom) => ibp_square(&r(atom)?),
            Reciprocal(atom) => ibp_reciprocal(&r(atom)?),
            Where(ca, aa, ba) => ibp_where(&r(ca)?, &r(aa)?, &r(ba)?),
            Gelu(atom) => ibp_gelu(&r(atom)?),
        }
    }
}

impl Interpreter<IBPTensor> for IBPInterpreter {
    fn run(graph: &ComputeGraph, inputs: &Vec<IBPTensor>) -> Result<Vec<IBPTensor>, EvalError> {
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
