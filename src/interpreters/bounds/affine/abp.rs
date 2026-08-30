use ndarray::{ArrayD, Zip};

use crate::{
    interpreters::{
        EvalError, Interpreter,
        bounds::{
            abp_util::{ABPTensor, lbp_inner},
            ibp::IBPInterpreter,
            ibp_util::IBPTensor,
        },
        compute_graph::{tracer::Tracer, try_trace_graph},
        concrete::{
            eval_util::{Tensor, norm_axis_index},
            grad::{GradInterpreter, unbroadcast, vjp_expanddims, vjp_moveaxis, vjp_reshape},
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
    ) -> Result<ABPTensor<T>, EvalError> {
        // Compute concrete slope/offset tensors from IBP bounds.
        let upper_slope: Tensor = Zip::from(&x.lb)
            .and(&x.ub)
            .map_collect(|&l, &u| {
                if l >= 0.0 {
                    1.0
                } else if u <= 0.0 {
                    0.0
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
        // Mask selects alpha for ambiguous neurons, fixed slope elsewhere.
        let ambiguous: Tensor = Zip::from(&x.lb)
            .and(&x.ub)
            .map_collect(|&l, &u| if l < 0.0 && u > 0.0 { 1.0 } else { 0.0 })
            .into();
        let fixed_slopes: Tensor = x.lb.mapv(|l| if l >= 0.0 { 1.0 } else { 0.0 });

        let lower_slope = T::r#where(&T::from(ambiguous), alpha, &T::from(fixed_slopes))?;
        let upper_slope = T::from(upper_slope);
        let upper_offset = T::from(upper_offset);

        let pos_w = out_w.relu();
        let neg_w = out_w.clone() - pos_w.clone();

        Ok(ABPTensor {
            weights: lower_slope * pos_w + upper_slope * neg_w.clone(),
            biases: lbp_inner(&upper_offset, &neg_w),
        })
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

        // If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.

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
                Primitive::Reciprocal(atom) => todo!(),
                Primitive::Square(atom) => todo!(),
                Primitive::Sqrt(atom) => todo!(),
                Primitive::Exp(atom) => todo!(),
                Primitive::Log(atom) => todo!(),
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
                Primitive::Where(atom, atom1, atom2) => todo!(),
                Primitive::Relu(atom) => {
                    vec![Self::crown_relu(
                        &p(&eqn.outvar)?,
                        &weights.resolve(&eqn.outvar)?,
                        &b(&atom)?,
                    )?]
                }
                Primitive::LeakyRelu { operand, slope } => todo!(),
                Primitive::Elu { operand, slope } => todo!(),
                Primitive::Gelu(atom) => todo!(),
                Primitive::NormalCdf(atom) => todo!(),
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
                    end,
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
                Primitive::Pad { operand, options } => todo!(),
                Primitive::Conv {
                    input,
                    kernel,
                    options,
                } => todo!(),
                Primitive::ConvKernelGrad { .. } => todo!(),
                Primitive::AvgPool { operand, options } => todo!(),
                Primitive::SumPool { operand, options } => todo!(),
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
