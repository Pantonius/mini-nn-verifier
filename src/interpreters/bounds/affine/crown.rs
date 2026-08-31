use ndarray::{ArrayD, Zip};

use crate::{
    interpreters::{
        EvalError,
        bounds::{
            affine::lbp_util::{
                AffineBounds, bound_convex, exp_lines, lbp_inner, normcdf_lines, reciprocal_lines,
                sqrt_lines, square_lines,
            },
            ibp_util::IBPTensor,
        },
        concrete::{
            eval_util::{Tensor, norm_axis_index},
            grad::{
                unbroadcast, vjp_conv, vjp_expanddims, vjp_moveaxis, vjp_pad, vjp_pool, vjp_reshape,
            },
        },
    },
    mininn::{
        Activation, Atom, AtomKind, ComputeGraph, Env, PaddingOptionConfig, PaddingOptions,
        Primitive, Value,
    },
};

pub(crate) fn linear_lower_bound<T, R>(
    graph: &ComputeGraph,
    var_bounds: &Env<IBPTensor>,
    mut bound_activation: R,
    start: Option<String>,
) -> Result<AffineBounds<T>, EvalError>
where
    T: Value,
    R: FnMut(&Atom, &T, &IBPTensor, &Activation) -> Result<AffineBounds<T>, EvalError>,
{
    // checks
    if graph.outvars.len() != 1 {
        return Err(EvalError::Eval(format!(
            "Found {} outvars, but Affine Bound Propogation only supports nets with a single outvar",
            graph.outvars.len()
        )));
    }

    if graph.invars.len() != 1 {
        return Err(EvalError::Eval(format!(
            "Found {} invars, but Affine Bound Propogation only supports nets with a single invar.",
            graph.invars.len()
        )));
    }

    // optionally start from other outvar
    let (idx, outvar_eqn) = if let Some(outvar_name) = start {
        graph
            .equations
            .iter()
            .enumerate()
            .find_map(|(i, eqn)| {
                if eqn.outvar.name == outvar_name {
                    Some((i, eqn))
                } else {
                    None
                }
            })
            .unwrap()
    } else {
        (graph.equations.len() - 1, graph.equations.last().unwrap())
    };

    let mut weights = Env::new();
    weights.insert(
        outvar_eqn.outvar.name.clone(),
        T::from(ArrayD::from_elem(outvar_eqn.outvar.shape.clone(), 1.0)),
    );
    let mut bias = T::from(0.0_f64);

    let b = |var: &Atom| var_bounds.resolve(var);

    let eqns = &graph.equations[..=idx];
    for eqn in eqns.iter().rev() {
        if weights.get(&eqn.outvar.name).is_none() {
            continue;
        }

        let out_w = weights.resolve(&eqn.outvar)?;
        // process primitive
        let affs = match &eqn.primitive {
            Primitive::Neg(_) => {
                vec![AffineBounds {
                    weights: -out_w,
                    biases: T::from(0.0_f64),
                }]
            }
            Primitive::Reciprocal(xa) => {
                let x = b(&xa)?;

                vec![bound_convex(&x, &out_w, reciprocal_lines, |l, u| {
                    if l <= 0.0 && u >= 0.0 {
                        Err(EvalError::Eval(format!(
                            "Reciprocal relaxation requires an input interval away from 0, got [{l}, {u}]"
                        )))
                    } else {
                        Ok(())
                    }
                })?]
            }
            Primitive::Square(xa) => {
                let x = b(&xa)?;

                vec![bound_convex(&x, &out_w, square_lines, |_, _| Ok(()))?]
            }
            Primitive::Sqrt(xa) => {
                let x = b(&xa)?;

                vec![bound_convex(&x, &out_w, sqrt_lines, |l, u| {
                    if l < 0.0 {
                        Err(EvalError::Eval(format!(
                            "Sqrt relaxation requires a non-negative input interval, got [{l}, {u}]"
                        )))
                    } else {
                        Ok(())
                    }
                })?]
            }
            Primitive::Exp(xa) => {
                let x = b(&xa)?;

                vec![bound_convex(&x, &out_w, exp_lines, |_, _| Ok(()))?]
            }
            Primitive::Log(_) => todo!(),
            Primitive::Add(..) => {
                let zero = T::from(0.0_f64);
                vec![
                    AffineBounds {
                        weights: out_w.clone(),
                        biases: zero.clone(),
                    },
                    AffineBounds {
                        weights: out_w,
                        biases: zero,
                    },
                ]
            }
            Primitive::Mul(xa, ya) => {
                let x = b(xa)?;
                let y = b(ya)?;

                crown_mul(&out_w, &x, &y)
            }
            Primitive::Where(cond, xa, ya) => crown_where(&out_w, &b(cond)?, &b(xa)?, &b(ya)?),
            Primitive::Relu(atom) => {
                vec![bound_activation(
                    &eqn.outvar,
                    &out_w,
                    &b(&atom)?,
                    &Activation::try_from(&eqn.primitive)?,
                )?]
            }
            Primitive::LeakyRelu { operand, slope: _ } => {
                vec![bound_activation(
                    &eqn.outvar,
                    &out_w,
                    &b(&operand)?,
                    &Activation::try_from(&eqn.primitive)?,
                )?]
            }
            Primitive::Elu { operand, slope: _ } => {
                vec![bound_activation(
                    &eqn.outvar,
                    &out_w,
                    &b(&operand)?,
                    &Activation::try_from(&eqn.primitive)?,
                )?]
            }
            Primitive::Gelu(operand) => {
                vec![bound_activation(
                    &eqn.outvar,
                    &out_w,
                    &b(&operand)?,
                    &Activation::try_from(&eqn.primitive)?,
                )?]
            }
            Primitive::NormalCdf(xa) => {
                let x = b(&xa)?;

                vec![bound_convex(&x, &out_w, normcdf_lines, |_, _| Ok(()))?]
            }
            Primitive::Dot(xa, ya) => {
                let x = b(&xa)?;
                let y = b(&ya)?;

                crown_dot(&out_w, &x, &y)?
            }
            Primitive::ReduceSum { operand, axes } => {
                let zero = T::from(0.0_f64);
                let broadcast_target = T::from(ArrayD::zeros(ndarray::IxDyn(&operand.shape)));
                vec![AffineBounds {
                    weights: out_w.expand_dims(axes) + broadcast_target,
                    biases: zero,
                }]
            }
            Primitive::ExpandDims { operand: _, axes } => {
                let in_w = vjp_expanddims(&out_w, axes)[0].clone();
                vec![AffineBounds {
                    weights: in_w,
                    biases: T::from(0.0_f64),
                }]
            }
            Primitive::MoveAxis {
                operand: _,
                source,
                destination,
            } => {
                let in_w = vjp_moveaxis(&out_w, *source, *destination)[0].clone();
                vec![AffineBounds {
                    weights: in_w,
                    biases: T::from(0.0_f64),
                }]
            }
            Primitive::Reshape {
                operand,
                new_shape: _,
            } => {
                let in_w = vjp_reshape(&T::from(b(operand)?.lb), &out_w)?[0].clone();
                vec![AffineBounds {
                    weights: in_w,
                    biases: T::from(0.0_f64),
                }]
            }
            Primitive::Slice {
                operand,
                axis,
                start,
                end: _,
                step,
            } => {
                let ax = norm_axis_index(*axis, operand.shape.len());
                let n = out_w.shape()[ax];
                let s = *start as usize;
                let st = *step as usize;
                let right = operand.shape[ax] - s - if n > 0 { (n - 1) * st + 1 } else { 0 };
                vec![AffineBounds {
                    weights: out_w.pad(&PaddingOptions {
                        axes: vec![*axis],
                        config: PaddingOptionConfig {
                            left: s,
                            interior: st - 1,
                            right,
                        },
                        value: 0.0,
                    }),
                    biases: T::from(0.0_f64),
                }]
            }
            Primitive::Pad { operand, options } => {
                vec![crown_pad(&out_w, &operand.shape, options)]
            }
            Primitive::Conv {
                input,
                kernel,
                options,
            } => {
                let AtomKind::Const(kernel_val) = &kernel.kind else {
                    return Err(EvalError::Eval(
                        "Conv affine bound requires a constant kernel".to_string(),
                    ));
                };
                let input_zeros = T::from(ArrayD::zeros(ndarray::IxDyn(&input.shape)));
                let in_w = vjp_conv(
                    &out_w,
                    &input_zeros,
                    &T::from(kernel_val.clone()),
                    options.stride,
                )?[0]
                    .clone();
                vec![AffineBounds {
                    weights: in_w,
                    biases: T::from(0.0),
                }]
            }
            Primitive::ConvKernelGrad { .. } => todo!(),
            Primitive::AvgPool { operand, options } => {
                let operand_zeros = T::from(ArrayD::zeros(ndarray::IxDyn(&operand.shape)));
                let in_w = vjp_pool(&out_w, &operand_zeros, options, true)[0].clone();
                vec![AffineBounds {
                    weights: in_w,
                    biases: T::from(0.0),
                }]
            }
            Primitive::SumPool { operand, options } => {
                let operand_zeros = T::from(ArrayD::zeros(ndarray::IxDyn(&operand.shape)));
                let in_w = vjp_pool(&out_w, &operand_zeros, options, false)[0].clone();
                vec![AffineBounds {
                    weights: in_w,
                    biases: T::from(0.0),
                }]
            }
        };

        // accumulate / early concretize
        for (invar, aff) in eqn.primitive.operands().iter().zip(affs) {
            let in_w = unbroadcast(&aff.weights, &invar.shape);
            bias = bias + aff.biases;

            if let AtomKind::Const(val) = &invar.kind {
                let iw = T::from(val.clone()) * in_w;
                let axes: Vec<isize> = (0..iw.ndim() as isize).collect();
                bias = bias + iw.reduce_sum(&axes);
            } else if let Some(existing) = weights.get(&invar.name) {
                weights.update(&invar.name, existing.clone() + in_w);
            } else {
                weights.insert(invar.name.clone(), in_w);
            }
        }
    }

    let invar = &graph.invars[0];
    let invar_w = weights
        .get(&invar.name)
        .cloned()
        .unwrap_or_else(|| T::from(ArrayD::zeros(ndarray::IxDyn(&invar.shape))));

    Ok(AffineBounds {
        weights: invar_w,
        biases: bias,
    })
}
pub(super) fn crown_relu<T: Value>(
    alpha: &T,
    out_w: &T,
    x: &IBPTensor,
    slope: f64,
) -> Result<AffineBounds<T>, EvalError> {
    // Compute concrete slope/offset tensors from IBP bounds.
    let upper_slope: Tensor = Zip::from(&x.lb)
        .and(&x.ub)
        .map_collect(|&l, &u| {
            if l >= 0.0 {
                1.0
            } else if u <= 0.0 {
                slope
            } else {
                u / (u - l)
            }
        })
        .into();
    let upper_offset: Tensor = Zip::from(&x.lb)
        .and(&x.ub)
        .map_collect(|&l, &u| {
            if l >= 0.0 || u <= 0.0 {
                0.0
            } else {
                -u * l / (u - l)
            }
        })
        .into();
    // Mask selects the (remapped) alpha for ambiguous neurons, fixed slope elsewhere.
    let ambiguous: Tensor = Zip::from(&x.lb)
        .and(&x.ub)
        .map_collect(|&l, &u| if l < 0.0 && u > 0.0 { 1.0 } else { 0.0 })
        .into();
    let fixed_slopes: Tensor = x.lb.mapv(|l| if l >= 0.0 { 1.0 } else { slope });

    // remap alpha into [slope, 1]
    // (slope = 0 : plain relu, where the remap is the identity.)
    let lower_alpha = alpha.clone() * T::from(1.0 - slope) + T::from(slope);
    let lower_slope = T::r#where(&T::from(ambiguous), &lower_alpha, &T::from(fixed_slopes))?;
    let upper_slope = T::from(upper_slope);
    let upper_offset = T::from(upper_offset);

    let pos_w = out_w.relu();
    let neg_w = out_w.clone() - pos_w.clone();

    Ok(AffineBounds {
        weights: lower_slope * pos_w + upper_slope * neg_w.clone(),
        biases: lbp_inner(&upper_offset, &neg_w),
    })
}

pub(super) fn crown_where<T: Value>(
    out_w: &T,
    cond: &IBPTensor,
    x: &IBPTensor,
    y: &IBPTensor,
) -> Vec<AffineBounds<T>> {
    // just true
    let mask_true: Tensor = Zip::from(&cond.lb)
        .and(&cond.ub)
        .map_collect(|&cl, &cu| if cl > 0.0 || cu < 0.0 { 1.0 } else { 0.0 })
        .into();
    // just false
    let mask_false: Tensor = Zip::from(&cond.lb)
        .and(&cond.ub)
        .map_collect(|&cl, &cu| if cl == 0.0 && cu == 0.0 { 1.0 } else { 0.0 })
        .into();
    // anything else
    let mask_amb: Tensor = Zip::from(&mask_true)
        .and(&mask_false)
        .map_collect(|&t, &f| 1.0 - t - f)
        .into();

    let e_lo: Tensor = Zip::from(&x.lb)
        .and(&y.lb)
        .map_collect(|&a, &b| a.min(b))
        .into();
    let e_hi: Tensor = Zip::from(&x.ub)
        .and(&y.ub)
        .map_collect(|&a, &b| a.max(b))
        .into();

    let pos_w = out_w.relu();
    let neg_w = out_w.clone() - pos_w.clone();

    let in_w_x = out_w.clone() * T::from(mask_true);
    let in_w_y = out_w.clone() * T::from(mask_false);
    let amb_bias = lbp_inner(&(pos_w * T::from(mask_amb.clone())), &T::from(e_lo))
        + lbp_inner(&(neg_w * T::from(mask_amb)), &T::from(e_hi));

    vec![
        AffineBounds {
            weights: T::from(ArrayD::zeros(ndarray::IxDyn(cond.lb.shape()))),
            biases: T::from(0.0),
        },
        AffineBounds {
            weights: in_w_x,
            biases: amb_bias,
        },
        AffineBounds {
            weights: in_w_y,
            biases: T::from(0.0),
        },
    ]
}

pub(super) fn crown_mul<T: Value>(w: &T, x: &IBPTensor, y: &IBPTensor) -> Vec<AffineBounds<T>> {
    let pos_w = w.relu();
    let neg_w = w.clone() - pos_w.clone();

    let in_w_x = pos_w.clone() * T::from(y.lb.clone()) + neg_w.clone() * T::from(y.ub.clone());
    let in_w_y = w.clone() * T::from(x.lb.clone());

    let in_bias = lbp_inner(&pos_w, &T::from(-x.lb.clone() * y.lb.clone()))
        + lbp_inner(&neg_w, &T::from(-x.lb.clone() * y.ub.clone()));

    vec![
        AffineBounds {
            weights: in_w_x,
            biases: in_bias.clone(),
        },
        AffineBounds {
            weights: in_w_y,
            biases: T::from(0.0),
        },
    ]
}

pub(super) fn crown_dot<T: Value>(
    w: &T,
    x: &IBPTensor,
    y: &IBPTensor,
) -> Result<Vec<AffineBounds<T>>, EvalError> {
    let pos_w = w.relu();
    let neg_w = w.clone() - pos_w.clone();

    // If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
    // If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
    if x.ndim() == 0 || y.ndim() == 0 || (x.ndim() == 1 && y.ndim() == 1) {
        return Ok(crown_mul(w, x, y));
    }

    let in_w_x: T;
    let in_w_y: T;
    let lower_c: T;
    let upper_c: T;

    if x.ndim() == 1 && y.ndim() == 2 {
        in_w_x = T::from(y.lb.clone()).dot(&pos_w)? + T::from(y.ub.clone()).dot(&neg_w)?;
        in_w_y = T::from(x.lb.expand_dims(&[1])) * w.clone().expand_dims(&[0]);

        lower_c = T::from(y.lb.moveaxis(0, 1).dot(&x.lb)?);
        upper_c = T::from(y.ub.moveaxis(0, 1).dot(&x.lb)?);
    } else if y.ndim() == 1 {
        // Matrix · vector: x = [.., M, K], y = [K] → z = [.., M]. Mirror of the
        // 2-D·2-D case with `x` pinned at its lower bound.
        in_w_x = pos_w.expand_dims(&[-1]) * T::from(y.lb.expand_dims(&[0]))
            + neg_w.expand_dims(&[-1]) * T::from(y.ub.expand_dims(&[0]));
        in_w_y = T::from(x.lb.moveaxis(0, 1)).dot(&w)?;

        lower_c = T::from(x.lb.clone().dot(&y.lb.clone())?);
        upper_c = T::from(x.lb.clone().dot(&y.ub.clone())?);
    } else {
        in_w_x =
            pos_w.dot(&T::from(y.lb.moveaxis(0, 1)))? + neg_w.dot(&T::from(y.ub.moveaxis(0, 1)))?;
        in_w_y = T::from(x.lb.moveaxis(0, 1)).dot(&w)?;

        lower_c = T::from(x.lb.clone().dot(&y.lb.clone())?);
        upper_c = T::from(x.lb.clone().dot(&y.ub.clone())?);
    }
    let in_bias = lbp_inner(&pos_w, &(-lower_c)) + lbp_inner(&neg_w, &(-upper_c));

    Ok(vec![
        AffineBounds {
            weights: in_w_x,
            biases: in_bias.clone(),
        },
        AffineBounds {
            weights: in_w_y,
            biases: T::from(0.0),
        },
    ])
}

pub(super) fn crown_pad<T: Value>(
    out_w: &T,
    operand_shape: &[usize],
    options: &PaddingOptions,
) -> AffineBounds<T> {
    let operand_zeros = T::from(ArrayD::zeros(ndarray::IxDyn(operand_shape)));
    let in_w = vjp_pad(out_w, &operand_zeros, options)[0].clone();
    let pad_fill = operand_zeros.pad(options);
    AffineBounds {
        weights: in_w,
        biases: lbp_inner(out_w, &pad_fill),
    }
}
