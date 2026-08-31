use ndarray::{ArrayD, Zip};

use crate::{
    interpreters::{
        EvalError, Interpreter,
        bounds::{
            abp_util::{
                ABPTensor, bound_convex, exp_lines, lbp_inner, normcdf_lines, reciprocal_lines,
                sqrt_lines, square_lines,
            },
            ibp::IBPInterpreter,
            ibp_util::IBPTensor,
        },
        compute_graph::{tracer::Tracer, try_trace_graph},
        concrete::{
            eval_util::{Tensor, norm_axis_index},
            grad::{
                GradInterpreter, unbroadcast, vjp_conv, vjp_expanddims, vjp_moveaxis, vjp_pad,
                vjp_pool, vjp_reshape,
            },
        },
    },
    mininn::{
        Atom, AtomKind, ComputeGraph, Env, PaddingOptionConfig, PaddingOptions, Primitive, Value,
    },
};

pub struct ABPInterpreter {}

impl ABPInterpreter {
    fn crown_relu<T: Value>(
        alpha: &T,
        out_w: &T,
        x: &IBPTensor,
        slope: f64,
    ) -> Result<ABPTensor<T>, EvalError> {
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

        Ok(ABPTensor {
            weights: lower_slope * pos_w + upper_slope * neg_w.clone(),
            biases: lbp_inner(&upper_offset, &neg_w),
        })
    }

    fn crown_where<T: Value>(
        out_w: &T,
        cond: &IBPTensor,
        x: &IBPTensor,
        y: &IBPTensor,
    ) -> Vec<ABPTensor<T>> {
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
            ABPTensor {
                weights: T::from(ArrayD::zeros(ndarray::IxDyn(cond.lb.shape()))),
                biases: T::from(0.0),
            },
            ABPTensor {
                weights: in_w_x,
                biases: amb_bias,
            },
            ABPTensor {
                weights: in_w_y,
                biases: T::from(0.0),
            },
        ]
    }

    fn crown_mul<T: Value>(w: &T, x: &IBPTensor, y: &IBPTensor) -> Vec<ABPTensor<T>> {
        let pos_w = w.relu();
        let neg_w = w.clone() - pos_w.clone();

        let in_w_x = pos_w.clone() * T::from(y.lb.clone()) + neg_w.clone() * T::from(y.ub.clone());
        let in_w_y = w.clone() * T::from(x.lb.clone());

        let in_bias = lbp_inner(&pos_w, &T::from(-x.lb.clone() * y.lb.clone()))
            + lbp_inner(&neg_w, &T::from(-x.lb.clone() * y.ub.clone()));

        vec![
            ABPTensor {
                weights: in_w_x,
                biases: in_bias.clone(),
            },
            ABPTensor {
                weights: in_w_y,
                biases: T::from(0.0),
            },
        ]
    }

    fn crown_dot<T: Value>(
        w: &T,
        x: &IBPTensor,
        y: &IBPTensor,
    ) -> Result<Vec<ABPTensor<T>>, EvalError> {
        let pos_w = w.relu();
        let neg_w = w.clone() - pos_w.clone();

        // If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
        // If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
        if x.ndim() == 0 || y.ndim() == 0 || (x.ndim() == 1 && y.ndim() == 1) {
            return Ok(Self::crown_mul(&w, &x, &y));
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
            in_w_x = pos_w.dot(&T::from(y.lb.moveaxis(0, 1)))?
                + neg_w.dot(&T::from(y.ub.moveaxis(0, 1)))?;
            in_w_y = T::from(x.lb.moveaxis(0, 1)).dot(&w)?;

            lower_c = T::from(x.lb.clone().dot(&y.lb.clone())?);
            upper_c = T::from(x.lb.clone().dot(&y.ub.clone())?);
        }
        let in_bias = lbp_inner(&pos_w, &(-lower_c)) + lbp_inner(&neg_w, &(-upper_c));

        Ok(vec![
            ABPTensor {
                weights: in_w_x,
                biases: in_bias.clone(),
            },
            ABPTensor {
                weights: in_w_y,
                biases: T::from(0.0),
            },
        ])
    }

    fn crown_pad<T: Value>(
        out_w: &T,
        operand_shape: &[usize],
        options: &PaddingOptions,
    ) -> ABPTensor<T> {
        let operand_zeros = T::from(ArrayD::zeros(ndarray::IxDyn(operand_shape)));
        let in_w = vjp_pad(out_w, &operand_zeros, options)[0].clone();
        let pad_fill = operand_zeros.pad(options);
        ABPTensor {
            weights: in_w,
            biases: lbp_inner(out_w, &pad_fill),
        }
    }

    pub fn linear_lower_bound<T: Value>(
        graph: &ComputeGraph,
        var_bounds: &Env<IBPTensor>,
        params: &Env<T>,
    ) -> Result<ABPTensor<T>, EvalError> {
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

        let outvar = &graph.outvars[0];

        let mut weights = Env::new();
        weights.insert(
            outvar.name.clone(),
            T::from(ArrayD::from_elem(outvar.shape.clone(), 1.0)),
        );
        let mut bias = T::from(0.0_f64);

        let b = |var: &Atom| var_bounds.resolve(var);
        let p = |var: &Atom| params.resolve(var);

        for eqn in graph.equations.iter().rev() {
            if weights.get(&eqn.outvar.name).is_none() {
                continue;
            }

            let out_w = weights.resolve(&eqn.outvar)?;
            // process primitive
            let affs = match &eqn.primitive {
                Primitive::Neg(_) => {
                    vec![ABPTensor {
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
                        ABPTensor {
                            weights: out_w.clone(),
                            biases: zero.clone(),
                        },
                        ABPTensor {
                            weights: out_w,
                            biases: zero,
                        },
                    ]
                }
                Primitive::Mul(xa, ya) => {
                    let x = b(xa)?;
                    let y = b(ya)?;

                    Self::crown_mul(&out_w, &x, &y)
                }
                Primitive::Where(cond, xa, ya) => {
                    Self::crown_where(&out_w, &b(cond)?, &b(xa)?, &b(ya)?)
                }
                Primitive::Relu(atom) => {
                    vec![Self::crown_relu(
                        &p(&eqn.outvar)?,
                        &weights.resolve(&eqn.outvar)?,
                        &b(&atom)?,
                        0.0,
                    )?]
                }
                Primitive::LeakyRelu { operand, slope } => vec![Self::crown_relu(
                    &p(&eqn.outvar)?,
                    &weights.resolve(&eqn.outvar)?,
                    &b(&operand)?,
                    *slope,
                )?],
                Primitive::Elu {
                    operand: _,
                    slope: _,
                } => todo!(),
                Primitive::Gelu(_) => todo!(),
                Primitive::NormalCdf(xa) => {
                    let x = b(&xa)?;

                    vec![bound_convex(&x, &out_w, normcdf_lines, |_, _| Ok(()))?]
                }
                Primitive::Dot(xa, ya) => {
                    let x = b(&xa)?;
                    let y = b(&ya)?;

                    Self::crown_dot(&out_w, &x, &y)?
                }
                Primitive::ReduceSum { operand, axes } => {
                    let zero = T::from(0.0_f64);
                    let broadcast_target = T::from(ArrayD::zeros(ndarray::IxDyn(&operand.shape)));
                    vec![ABPTensor {
                        weights: out_w.expand_dims(axes) + broadcast_target,
                        biases: zero,
                    }]
                }
                Primitive::ExpandDims { operand: _, axes } => {
                    let in_w = vjp_expanddims(&out_w, axes)[0].clone();
                    vec![ABPTensor {
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
                    vec![ABPTensor {
                        weights: in_w,
                        biases: T::from(0.0_f64),
                    }]
                }
                Primitive::Reshape {
                    operand,
                    new_shape: _,
                } => {
                    let in_w = vjp_reshape(&T::from(b(operand)?.lb), &out_w)?[0].clone();
                    vec![ABPTensor {
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
                    vec![ABPTensor {
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
                    vec![Self::crown_pad(&out_w, &operand.shape, options)]
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
                    vec![ABPTensor {
                        weights: in_w,
                        biases: T::from(0.0),
                    }]
                }
                Primitive::ConvKernelGrad { .. } => todo!(),
                Primitive::AvgPool { operand, options } => {
                    let operand_zeros = T::from(ArrayD::zeros(ndarray::IxDyn(&operand.shape)));
                    let in_w = vjp_pool(&out_w, &operand_zeros, options, true)[0].clone();
                    vec![ABPTensor {
                        weights: in_w,
                        biases: T::from(0.0),
                    }]
                }
                Primitive::SumPool { operand, options } => {
                    let operand_zeros = T::from(ArrayD::zeros(ndarray::IxDyn(&operand.shape)));
                    let in_w = vjp_pool(&out_w, &operand_zeros, options, false)[0].clone();
                    vec![ABPTensor {
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

        Ok(ABPTensor {
            weights: invar_w,
            biases: bias,
        })
    }

    fn alpha_crown_optim(
        graph: &ComputeGraph,
        ibp_bounds: &Env<IBPTensor>,
        mut params: Env<Tensor>,
    ) -> Result<ABPTensor<Tensor>, EvalError> {
        if params.len() > 0 {
            const ITERS: usize = 10;
            const LR: f64 = 0.01;

            let invar_bounds = ibp_bounds.resolve(&graph.invars[0])?;
            let grad_graph = try_trace_graph(
                graph,
                Some(
                    graph
                        .equations
                        .iter()
                        .filter_map(|eqn| match &eqn.primitive {
                            Primitive::Relu(_)
                            | Primitive::LeakyRelu { .. }
                            | Primitive::Elu { .. }
                            | Primitive::Gelu(_) => params
                                .get(&eqn.outvar.name)
                                .map(|alpha| (eqn.outvar.name.clone(), alpha.shape().to_vec())),
                            _ => None,
                        })
                        .collect(),
                ),
                |tracer_params| -> Result<Tracer, EvalError> {
                    let alb = Self::linear_lower_bound(graph, ibp_bounds, &tracer_params)?;
                    Ok(alb.concretize(
                        &Tracer::from(invar_bounds.lb.clone()),
                        &Tracer::from(invar_bounds.ub.clone()),
                    ))
                },
            )?;

            for _ in 0..ITERS {
                let alpha_inputs: Vec<Tensor> = grad_graph
                    .invars
                    .iter()
                    .map(|a| params.get(&a.name).cloned().expect("alpha not found"))
                    .collect();

                let grads = GradInterpreter::run(&grad_graph, &alpha_inputs)?;

                for (invar, grad) in grad_graph.invars.iter().zip(grads) {
                    let alpha = params.get(&invar.name).cloned().unwrap();
                    params.update(&invar.name, (alpha + grad * LR).mapv(|v| v.clamp(0.0, 1.0)));
                }
            }
        }

        Self::linear_lower_bound(graph, &ibp_bounds, &params)
    }
}

impl ABPInterpreter {
    pub fn run(
        graph: &ComputeGraph,
        inputs: &Vec<IBPTensor>,
    ) -> Result<(ABPTensor<Tensor>, ABPTensor<Tensor>), EvalError> {
        // === alpha-CROWN ===

        // --- Forward ---
        // - Bound all vars in the network via IBP
        // - Compute initial params (alpha per activation) given IBP bounds

        // --- Backward ---
        // Optimize affine bounds by optimizing over alpha
        //
        // Loss
        // ----
        // 1. compute linear lower bound by backward pass from W_out = ones, bias_out = zeros using
        //    IBP bounds and alpha params from forward pass
        // 2. concretize linear lower bound via "concrete" invar bounds

        // ------------------------------------------------------------
        // 1. init bounds (ibp forward pass)
        // 2. init params (alpha depending on mode of each activation given bounds)
        // 3. optimize alpha (gradient ascent over loss; loss is the concretization of linear lower bound
        //    from backprop starting at W_out = ones, bias_out = zeros, out_var_bounds from IBP
        //    pass; a few iterations of that)

        // 1. Init Bounds (IBP forward pass)
        let mut ibp_bounds = Env::new();
        let mut params: Env<Tensor> = Env::new();

        for (var, tensor) in graph.invars.iter().zip(inputs) {
            ibp_bounds.insert(var.name.clone(), tensor.clone());
        }

        for eqn in &graph.equations {
            let out = IBPInterpreter::process_primitive(&eqn.primitive, &ibp_bounds)?;
            ibp_bounds.insert(eqn.outvar.name.clone(), out);

            // 2. Init Params (Alpha per activation given IBP bounds on respective invar)
            match &eqn.primitive {
                Primitive::Relu(operand)
                | Primitive::LeakyRelu { operand, .. }
                | Primitive::Elu { operand, .. }
                | Primitive::Gelu(operand) => {
                    let bound = ibp_bounds.resolve(operand)?;
                    let alpha = Zip::from(&bound.lb)
                        .and(&bound.ub)
                        .map_collect(|&l, &u| if -l >= u { 0.0 } else { 1.0 });

                    params.insert(eqn.outvar.name.clone(), Tensor::from(alpha));
                }
                _ => continue,
            }
        }

        // 3. Optimize Alpha (Gradient Ascent over alpha)
        let lb = Self::alpha_crown_optim(graph, &ibp_bounds, params.clone())?;

        let neg_graph = try_trace_graph(graph, None, |env| -> Result<Tracer, EvalError> {
            let out = env.resolve(&graph.outvars[0])?;
            Ok(-out)
        })?;

        let lb_neg = Self::alpha_crown_optim(&neg_graph, &ibp_bounds, params.clone())?;
        let ub = ABPTensor {
            weights: -lb_neg.weights,
            biases: -lb_neg.biases,
        };

        Ok((lb, ub))
    }
}

#[cfg(test)]
mod tests {
    use crate::interpreters::concrete::eval_util::normcdf;

    use super::*;
    use ndarray::IxDyn;

    /// The linear relaxation must sandwich Φ over the whole interval:
    /// `lower_slope·x + lower_offset ≤ Φ(x) ≤ upper_slope·x + upper_offset`.
    #[test]
    fn normcdf_lines_are_sound() {
        // A spread of intervals: convex-only, concave-only, crossing (symmetric,
        // skewed, wide, narrow), and degenerate.
        let cases = [
            (-2.0, -1.0),
            (-4.0, -0.5),
            (1.0, 2.0),
            (0.5, 4.0),
            (-2.0, 3.0),
            (-0.5, 0.5),
            (-3.0, 0.3),
            (-0.3, 3.0),
            (-6.0, 6.0),
            (-0.01, 0.01),
            (2.5, 2.5), // degenerate
        ];

        // Small tolerance for the bisection's finite precision on the tangent lines.
        let tol = 1e-6;
        let n = 400;

        for (l, u) in cases {
            let (ls, lo, us, uo) = normcdf_lines(l, u);
            for i in 0..=n {
                let x = l + (u - l) * (i as f64) / (n as f64);
                let f = normcdf(x);
                let lower = ls * x + lo;
                let upper = us * x + uo;
                assert!(
                    lower <= f + tol,
                    "lower line above Φ on [{l}, {u}] at x={x}: {lower} > {f}",
                );
                assert!(
                    upper >= f - tol,
                    "upper line below Φ on [{l}, {u}] at x={x}: {upper} < {f}",
                );
            }
        }
    }

    /// In the pure convex/concave regions one side is the exact chord, so it must
    /// touch Φ at both endpoints.
    #[test]
    fn normcdf_lines_chord_touches_endpoints() {
        // Concave region: lower bound is the chord.
        let (ls, lo, _, _) = normcdf_lines(1.0, 2.0);
        assert!((ls * 1.0 + lo - normcdf(1.0)).abs() < 1e-9);
        assert!((ls * 2.0 + lo - normcdf(2.0)).abs() < 1e-9);

        // Convex region: upper bound is the chord.
        let (_, _, us, uo) = normcdf_lines(-2.0, -1.0);
        assert!((us * -2.0 + uo - normcdf(-2.0)).abs() < 1e-9);
        assert!((us * -1.0 + uo - normcdf(-1.0)).abs() < 1e-9);
    }

    /// The linear relaxation must sandwich `1/x` over the whole interval:
    /// `lower_slope·x + lower_offset ≤ 1/x ≤ upper_slope·x + upper_offset`.
    #[test]
    fn reciprocal_lines_are_sound() {
        // Positive (convex) and negative (concave) intervals: wide, narrow,
        // near-zero, and degenerate. The interval must stay off zero.
        let cases = [
            (1.0, 2.0),
            (0.5, 4.0),
            (0.01, 10.0),
            (2.5, 2.5), // degenerate
            (-2.0, -1.0),
            (-4.0, -0.5),
            (-10.0, -0.01),
            (-2.5, -2.5), // degenerate
        ];

        let tol = 1e-9;
        let n = 400;

        for (l, u) in cases {
            let (ls, lo, us, uo) = reciprocal_lines(l, u);
            for i in 0..=n {
                let x = l + (u - l) * (i as f64) / (n as f64);
                let f = x.recip();
                let lower = ls * x + lo;
                let upper = us * x + uo;
                assert!(
                    lower <= f + tol,
                    "lower line above 1/x on [{l}, {u}] at x={x}: {lower} > {f}",
                );
                assert!(
                    upper >= f - tol,
                    "upper line below 1/x on [{l}, {u}] at x={x}: {upper} < {f}",
                );
            }
        }
    }

    /// In each region one side is the exact chord, so it must touch `1/x` at both
    /// endpoints.
    #[test]
    fn reciprocal_lines_chord_touches_endpoints() {
        // Convex (positive) region: upper bound is the chord.
        let (_, _, us, uo) = reciprocal_lines(1.0, 4.0);
        assert!((us * 1.0 + uo - 1.0_f64.recip()).abs() < 1e-9);
        assert!((us * 4.0 + uo - 4.0_f64.recip()).abs() < 1e-9);

        // Concave (negative) region: lower bound is the chord.
        let (ls, lo, _, _) = reciprocal_lines(-4.0, -1.0);
        assert!((ls * -4.0 + lo - (-4.0_f64).recip()).abs() < 1e-9);
        assert!((ls * -1.0 + lo - (-1.0_f64).recip()).abs() < 1e-9);
    }

    /// The linear relaxation must sandwich `exp(x)` over the whole interval:
    /// `lower_slope·x + lower_offset ≤ exp(x) ≤ upper_slope·x + upper_offset`.
    #[test]
    fn exp_lines_are_sound() {
        // Negative, zero-crossing, positive, wide, narrow, and degenerate.
        let cases = [
            (-3.0, -1.0),
            (-2.0, 2.0),
            (0.0, 1.0),
            (1.0, 4.0),
            (-5.0, 5.0),
            (0.5, 0.6),
            (2.5, 2.5), // degenerate
        ];

        let tol = 1e-9;
        let n = 400;

        for (l, u) in cases {
            let (ls, lo, us, uo) = exp_lines(l, u);
            for i in 0..=n {
                let x = l + (u - l) * (i as f64) / (n as f64);
                let f = x.exp();
                let lower = ls * x + lo;
                let upper = us * x + uo;
                assert!(
                    lower <= f + tol,
                    "lower line above exp on [{l}, {u}] at x={x}: {lower} > {f}",
                );
                assert!(
                    upper >= f - tol,
                    "upper line below exp on [{l}, {u}] at x={x}: {upper} < {f}",
                );
            }
        }
    }

    /// `exp` is convex, so the upper bound is the exact chord and must touch
    /// `exp` at both endpoints.
    #[test]
    fn exp_lines_chord_touches_endpoints() {
        let (_, _, us, uo) = exp_lines(-1.0, 2.0);
        assert!((us * -1.0 + uo - (-1.0_f64).exp()).abs() < 1e-9);
        assert!((us * 2.0 + uo - 2.0_f64.exp()).abs() < 1e-9);
    }

    /// The linear relaxation must sandwich `x²` over the whole interval, and the
    /// lower line must stay non-negative (its output feeds Sqrt).
    #[test]
    fn square_lines_are_sound() {
        // Negative (decreasing), positive (increasing), zero-crossing (symmetric
        // and skewed), wide, narrow, and degenerate.
        let cases = [
            (-3.0, -1.0),
            (1.0, 4.0),
            (-2.0, 2.0),
            (-4.0, 1.0),
            (-1.0, 5.0),
            (-6.0, 6.0),
            (0.5, 0.6),
            (2.5, 2.5),   // degenerate
            (-2.5, -2.5), // degenerate
        ];

        let tol = 1e-9;
        let n = 400;

        for (l, u) in cases {
            let (ls, lo, us, uo) = square_lines(l, u);
            for i in 0..=n {
                let x = l + (u - l) * (i as f64) / (n as f64);
                let f = x * x;
                let lower = ls * x + lo;
                let upper = us * x + uo;
                assert!(
                    lower <= f + tol,
                    "lower line above x² on [{l}, {u}] at x={x}: {lower} > {f}",
                );
                assert!(
                    upper >= f - tol,
                    "upper line below x² on [{l}, {u}] at x={x}: {upper} < {f}",
                );
                assert!(
                    lower >= -tol,
                    "lower line negative on [{l}, {u}] at x={x}: {lower}",
                );
            }
        }
    }

    /// `x²` is convex, so the upper bound is the exact chord and must touch `x²`
    /// at both endpoints.
    #[test]
    fn square_lines_chord_touches_endpoints() {
        let (_, _, us, uo) = square_lines(-1.0, 3.0);
        assert!((us * -1.0 + uo - 1.0).abs() < 1e-9);
        assert!((us * 3.0 + uo - 9.0).abs() < 1e-9);
    }

    /// The linear relaxation must sandwich `√x` over the whole interval.
    #[test]
    fn sqrt_lines_are_sound() {
        // At-zero, general, wide, narrow, and degenerate. Domain is x ≥ 0.
        let cases = [
            (0.0, 4.0),
            (1.0, 9.0),
            (0.25, 100.0),
            (4.0, 4.1),
            (2.5, 2.5), // degenerate
        ];

        let tol = 1e-9;
        let n = 400;

        for (l, u) in cases {
            let (ls, lo, us, uo) = sqrt_lines(l, u);
            for i in 0..=n {
                let x = l + (u - l) * (i as f64) / (n as f64);
                let f = x.sqrt();
                let lower = ls * x + lo;
                let upper = us * x + uo;
                assert!(
                    lower <= f + tol,
                    "lower line above √x on [{l}, {u}] at x={x}: {lower} > {f}",
                );
                assert!(
                    upper >= f - tol,
                    "upper line below √x on [{l}, {u}] at x={x}: {upper} < {f}",
                );
            }
        }
    }

    /// `√x` is concave, so the lower bound is the exact chord and must touch `√x`
    /// at both endpoints.
    #[test]
    fn sqrt_lines_chord_touches_endpoints() {
        let (ls, lo, _, _) = sqrt_lines(1.0, 9.0);
        assert!((ls * 1.0 + lo - 1.0).abs() < 1e-9);
        assert!((ls * 9.0 + lo - 3.0).abs() < 1e-9);
    }

    // ---- dot / bilinear relaxation ----

    fn arr(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from(ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap())
    }

    /// Full flatten dot: sum of elementwise products (same shape assumed).
    fn fold_dot(a: &Tensor, b: &Tensor) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn scalar(t: &Tensor) -> f64 {
        t.iter().sum()
    }

    /// `Pad` is an exact affine map `y = P·x + b`, so the bound `crown_pad`
    /// produces must satisfy `⟨out_w, pad(x)⟩ == ⟨weights, x⟩ + bias` for every
    /// operand `x`. Checked with mixed-sign weights and several fill values
    /// (nonzero values exercise the bias term; `value == 0` must give zero bias).
    #[test]
    fn crown_pad_is_exact() {
        let operand_shape = [2usize, 3];
        let options = PaddingOptions {
            axes: vec![0, 1],
            config: PaddingOptionConfig {
                left: 1,
                right: 2,
                interior: 1,
            },
            value: 0.0,
        };

        let x = arr(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &operand_shape);

        for value in [0.0, 0.7, -1.5] {
            let options = PaddingOptions {
                value,
                ..options.clone()
            };
            let px = x.pad(&options);

            // Mixed-sign weights over the padded shape.
            let w_data: Vec<f64> = px
                .iter()
                .enumerate()
                .map(|(i, _)| ((i as f64) * 0.37 - 1.3).sin())
                .collect();
            let out_w = arr(&w_data, px.shape());

            let aff = ABPInterpreter::crown_pad(&out_w, &operand_shape, &options);

            let lhs = fold_dot(&out_w, &px);
            let rhs = fold_dot(&aff.weights, &x) + scalar(&aff.biases);
            assert!(
                (lhs - rhs).abs() < 1e-12,
                "pad affine identity broken (value={value}): lhs={lhs} rhs={rhs}",
            );

            if value == 0.0 {
                assert!(
                    scalar(&aff.biases).abs() < 1e-12,
                    "zero-value pad should have zero bias, got {}",
                    scalar(&aff.biases),
                );
            }
        }
    }

    /// Assert the affine bound `crown_dot` produces is a sound lower bound of
    /// `⟨out_w, dot(x, y)⟩` everywhere in the input box. `out_w` should carry mixed
    /// signs so both McCormick estimators are exercised. Weights are `unbroadcast`
    /// to the operand shapes exactly as `linear_lower_bound` accumulates them.
    ///
    /// `⟨out_w, dot(x,y)⟩ − rhs` is bilinear in `(x, y)`, so its minimum over the
    /// box is attained at a vertex — we enumerate all `2^(nx+ny)` of them, which
    /// makes this an exact soundness check (not just a sampling heuristic).
    fn assert_dot_sound(xl: &Tensor, xu: &Tensor, yl: &Tensor, yu: &Tensor, out_w: &Tensor) {
        let x = IBPTensor::new(xl.clone(), xu.clone());
        let y = IBPTensor::new(yl.clone(), yu.clone());
        let affs = ABPInterpreter::crown_dot(out_w, &x, &y).unwrap();
        assert_eq!(affs.len(), 2);

        let wx = unbroadcast(&affs[0].weights, xl.shape());
        let wy = unbroadcast(&affs[1].weights, yl.shape());
        let bias = scalar(&affs[0].biases) + scalar(&affs[1].biases);

        // Flattened endpoints, so a vertex is just a bitmask over all coordinates.
        let (xlf, xuf): (Vec<f64>, Vec<f64>) =
            (xl.iter().copied().collect(), xu.iter().copied().collect());
        let (ylf, yuf): (Vec<f64>, Vec<f64>) =
            (yl.iter().copied().collect(), yu.iter().copied().collect());
        let (nx, ny) = (xlf.len(), ylf.len());
        assert!(nx + ny <= 20, "too many vertices to enumerate");

        let pick = |lo: &[f64], hi: &[f64], bits: usize, off: usize, shape: &[usize]| {
            let data: Vec<f64> = (0..lo.len())
                .map(|i| {
                    if (bits >> (off + i)) & 1 == 1 {
                        hi[i]
                    } else {
                        lo[i]
                    }
                })
                .collect();
            Tensor::from(ArrayD::from_shape_vec(IxDyn(shape), data).unwrap())
        };

        for bits in 0..(1usize << (nx + ny)) {
            let xs = pick(&xlf, &xuf, bits, 0, xl.shape());
            let ys = pick(&ylf, &yuf, bits, nx, yl.shape());
            let z = xs.dot(&ys).unwrap();

            let lhs = fold_dot(out_w, &z);
            let rhs = fold_dot(&wx, &xs) + fold_dot(&wy, &ys) + bias;
            assert!(
                lhs + 1e-9 >= rhs,
                "unsound dot bound at vertex {bits:b}: lhs={lhs} rhs={rhs} (violation {})",
                rhs - lhs
            );
        }
    }

    #[test]
    fn dot_1d_1d_sound() {
        // x·y inner product → scalar. out_w is 0-D; check both signs.
        let xl = arr(&[-1.0, 0.5, -2.0], &[3]);
        let xu = arr(&[1.0, 2.0, -0.5], &[3]);
        let yl = arr(&[0.0, -1.0, -1.0], &[3]);
        let yu = arr(&[2.0, 1.0, 3.0], &[3]);
        assert_dot_sound(&xl, &xu, &yl, &yu, &arr(&[1.5], &[]));
        assert_dot_sound(&xl, &xu, &yl, &yu, &arr(&[-1.5], &[]));
    }

    #[test]
    fn dot_scalar_operand_sound() {
        // 0-D · [3] = scalar * vector → [3] (weights get unbroadcast back to 0-D).
        let xl = arr(&[-2.0], &[]);
        let xu = arr(&[1.0], &[]);
        let yl = arr(&[-1.0, 0.0, -2.0], &[3]);
        let yu = arr(&[1.0, 2.0, 0.5], &[3]);
        assert_dot_sound(&xl, &xu, &yl, &yu, &arr(&[1.0, -1.0, 2.0], &[3]));
    }

    #[test]
    fn dot_1d_2d_sound() {
        // [K] · [K, N] → [N], with K=3, N=2.
        let xl = arr(&[-1.0, 0.0, -2.0], &[3]);
        let xu = arr(&[1.0, 2.0, 1.0], &[3]);
        let yl = arr(&[-1.0, 0.0, -2.0, 1.0, 0.5, -1.0], &[3, 2]);
        let yu = arr(&[1.0, 2.0, 0.0, 3.0, 1.5, 2.0], &[3, 2]);
        assert_dot_sound(&xl, &xu, &yl, &yu, &arr(&[1.0, -2.0], &[2]));
    }

    #[test]
    fn dot_2d_1d_sound() {
        // [M, K] · [K] → [M], with M=2, K=3.
        let xl = arr(&[-1.0, 0.0, -2.0, 0.5, -1.0, 1.0], &[2, 3]);
        let xu = arr(&[1.0, 2.0, 0.0, 2.0, 1.0, 2.0], &[2, 3]);
        let yl = arr(&[-1.0, -2.0, 0.0], &[3]);
        let yu = arr(&[2.0, 0.5, 3.0], &[3]);
        assert_dot_sound(&xl, &xu, &yl, &yu, &arr(&[1.5, -1.0], &[2]));
    }

    #[test]
    fn dot_2d_2d_sound() {
        // [M, K] · [K, N] → [M, N], with M=2, K=3, N=2.
        let xl = arr(&[-1.0, 0.0, -2.0, 0.5, -1.0, 1.0], &[2, 3]);
        let xu = arr(&[1.0, 2.0, 0.0, 2.0, 1.0, 2.0], &[2, 3]);
        let yl = arr(&[-1.0, 0.0, -2.0, 1.0, 0.5, -1.0], &[3, 2]);
        let yu = arr(&[1.0, 2.0, 0.0, 3.0, 1.5, 2.0], &[3, 2]);
        assert_dot_sound(&xl, &xu, &yl, &yu, &arr(&[1.0, -1.0, -2.0, 0.5], &[2, 2]));
    }
}
