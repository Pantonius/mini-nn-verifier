use ndarray::{Ix1, Ix2};

use crate::{
    interpreters::{
        EvalError, Interpreter, Tensor,
        eval_util::{
            add, conv, dot, elu, expand_dims, leaky_relu, log, moveaxis, mul, normcdf, pad, pool,
            reduce_sum, relu, reshape, reshape_c,
        },
        ibp_util::{
            IBPBatchedTensor, IBPTensor, ibp_gelu, ibp_linear, ibp_monotonic_non_dec,
            ibp_monotonic_non_dec2, ibp_monotonic_non_inc, ibp_reciprocal, ibp_square, ibp_where,
        },
    },
    mininn::{Atom, ComputeGraph, Env, PaddingOptions, Primitive},
};

pub struct IBPBatchedInterpreter {}

impl IBPBatchedInterpreter {
    pub fn process_primitive(
        primitive: &Primitive,
        env: &Env<IBPBatchedTensor>,
    ) -> Result<IBPBatchedTensor, EvalError> {
        let r = |a: &Atom| -> Result<IBPTensor, EvalError> {
            env.resolve(a).map(|t| IBPTensor::new(t.lb, t.ub))
        };

        use Primitive::*;
        let out = match primitive {
            // Non-Dec
            ExpandDims { operand, axes } => {
                ibp_monotonic_non_dec(|a| Ok(expand_dims_batched(&a, axes)), &r(operand)?)
            }
            MoveAxis {
                operand,
                source,
                destination,
            } => ibp_monotonic_non_dec(
                |a| Ok(moveaxis_batched(&a, *source, *destination)),
                &r(operand)?,
            ),
            Reshape { operand, new_shape } => {
                ibp_monotonic_non_dec(|a| reshape_batched(&a, new_shape), &r(operand)?)
            }
            Pad { operand, options } => {
                ibp_monotonic_non_dec(|a| Ok(pad_batched(&a, options)), &r(operand)?)
            }
            Add(aa, ba) => ibp_monotonic_non_dec2(|a, b| add(&a, &b), &r(aa)?, &r(ba)?),
            ReduceSum { operand, axes } => {
                ibp_monotonic_non_dec(|a| Ok(reduce_sum_batched(&a, axes)), &r(operand)?)
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
                ibp_monotonic_non_dec(|a| pool_batched(&a, options, false), &r(operand)?)
            }
            AvgPool { operand, options } => {
                ibp_monotonic_non_dec(|a| pool_batched(&a, options, true), &r(operand)?)
            }
            NormalCdf(atom) => ibp_monotonic_non_dec(|a| Ok(a.mapv(|x| normcdf(x))), &r(atom)?),
            // Non-Inc
            Neg(atom) => ibp_monotonic_non_inc(|a| Ok(a.mapv(|x| -x)), &r(atom)?),
            // Linear
            Dot(aa, ba) => {
                let a = r(aa)?;
                let b = r(ba)?;

                if a.is_point() {
                    ibp_linear(
                        |act, w| {
                            if act.ndim() == 2 && w.ndim() == 2 {
                                // act[k,d_in] @ W[d_out,d_in].T → [k,d_out]
                                let a2 = act.view().into_dimensionality::<Ix2>().unwrap();
                                let w2 = w.view().into_dimensionality::<Ix2>().unwrap();
                                Ok(a2.dot(&w2.t()).into_dyn())
                            } else if act.ndim() == 2 && w.ndim() == 1 {
                                // act[k,d] @ w[d] → [k]  (e.g. final output-vector dot)
                                let a2 = act.view().into_dimensionality::<Ix2>().unwrap();
                                let w1 = w.view().into_dimensionality::<Ix1>().unwrap();
                                Ok(a2.dot(&w1).into_dyn())
                            } else {
                                unreachable!()
                                // dot(w, act) // fallback for 1-D act (non-batched)
                            }
                        },
                        &b, // activation → "a" slot (varying)
                        &a, // weight    → "b" slot (point)
                    )
                } else {
                    // b is weight W [d_in, d_out] (or both intervals).
                    // The closure receives (activation_mid_or_ran, W) and computes act @ W.
                    ibp_linear(|act, w| dot(act, w), &a, &b)
                }
            }
            Mul(aa, ba) => ibp_linear(|a, b| mul(&a, &b), &r(aa)?, &r(ba)?),
            Conv {
                input,
                kernel,
                options,
            } => ibp_linear(
                |a, b| conv_batched(&a, &b, options.stride),
                &r(input)?,
                &r(kernel)?,
            ),
            // Special cases
            Square(atom) => ibp_square(&r(atom)?),
            Reciprocal(atom) => ibp_reciprocal(&r(atom)?),
            Where(ca, aa, ba) => ibp_where(&r(ca)?, &r(aa)?, &r(ba)?),
            Gelu(atom) => ibp_gelu(&r(atom)?),
        }?;

        Ok(IBPBatchedTensor {
            lb: out.lb,
            ub: out.ub,
        })
    }
}

fn conv_batched(input: &Tensor, kernel: &Tensor, stride: isize) -> Result<Tensor, EvalError> {
    // NOTE assumption that input has 5 dims [k, 1, C, H, W]
    // Want to make that [k, C, H, W]

    let sh = input.shape().to_vec();
    let (k, n) = (sh[0], sh[1]);
    let flat = reshape_c(input, &[k * n, sh[2], sh[3], sh[4]]);

    let out = conv(&flat, kernel, stride)?;
    let osh = out.shape().to_vec();

    Ok(reshape_c(&out, &[k, n, osh[1], osh[2], osh[3]]))
}

fn reshape_batched(a: &Tensor, new_shape: &[isize]) -> Result<Tensor, EvalError> {
    let k = a.shape()[0] as isize;
    let mut batch_shape = vec![k];
    batch_shape.extend_from_slice(new_shape);
    reshape(a, &batch_shape)
}

fn expand_dims_batched(a: &Tensor, axes: &[isize]) -> Tensor {
    let shifted: Vec<isize> = axes
        .iter()
        .map(|&ax| if ax >= 0 { ax + 1 } else { ax })
        .collect();
    expand_dims(a, &shifted)
}

fn moveaxis_batched(a: &Tensor, src: isize, dst: isize) -> Tensor {
    let src_b = if src >= 0 { src + 1 } else { src };
    let dst_b = if dst >= 0 { dst + 1 } else { dst };
    moveaxis(a, src_b, dst_b)
}

fn reduce_sum_batched(a: &Tensor, axes: &[isize]) -> Tensor {
    let shifted: Vec<isize> = axes
        .iter()
        .map(|&ax| if ax >= 0 { ax + 1 } else { ax })
        .collect();
    reduce_sum(a, &shifted)
}

fn pad_batched(a: &Tensor, opt: &PaddingOptions) -> Tensor {
    let shifted_axes: Vec<isize> = opt
        .axes
        .iter()
        .map(|&ax| if ax >= 0 { ax + 1 } else { ax })
        .collect();
    pad(
        a,
        &PaddingOptions {
            axes: shifted_axes,
            ..opt.clone()
        },
    )
}

fn pool_batched(
    a: &Tensor,
    opt: &crate::mininn::PoolOptions,
    average: bool,
) -> Result<Tensor, EvalError> {
    if a.ndim() == 5 {
        let sh = a.shape().to_vec();
        let (k, n) = (sh[0], sh[1]);
        let flat = reshape_c(a, &[k * n, sh[2], sh[3], sh[4]]);
        let out = pool(&flat, opt, average)?;
        let osh = out.shape().to_vec();
        Ok(reshape_c(&out, &[k, n, osh[1], osh[2], osh[3]]))
    } else {
        pool(a, opt, average)
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
