use ndarray::{ArrayD, Zip};

use crate::{
    interpreters::{
        EvalError, Interpreter,
        bounds::{affine::crown::linear_lower_bound, ibp_util::IBPTensor, lbp_util::AffineBounds},
        compute_graph::{tracer::Tracer, try_trace_graph},
        concrete::{eval_util::Tensor, grad::GradInterpreter},
    },
    mininn::{Activation, Atom, AtomKind, ComputeGraph, Env, Primitive, Value},
};

use super::alpha_crown::AlphaCrownInterpreter;

pub struct BetaCrownInterpreter {}

const BETA_SUFFIX: &str = "::beta";

impl BetaCrownInterpreter {
    fn bunel_score(out_w: &Tensor, x: &IBPTensor, bias: &Tensor) -> Tensor {
        Zip::from(out_w)
            .and(&x.lb)
            .and(&x.ub)
            .and(bias)
            .map_collect(|&pre, &l, &u, &b| {
                if !(l < 0.0 && u > 0.0) {
                    return 0.0; // stable neuron: nothing to gain from splitting
                }
                let span = u - l;
                let slope = u / span;
                let term_a = slope * pre * b;
                let term_b = (slope - 1.0) * pre * b;
                let m = term_a.min(term_b);
                (m - (u * l / span) * pre.max(0.0)).abs()
            })
            .into()
    }

    fn feeding_bias(graph: &ComputeGraph, operand: &Atom, shape: &[usize]) -> Tensor {
        let zeros = || Tensor::from(ArrayD::zeros(ndarray::IxDyn(shape)));
        let Some(eqn) = graph
            .equations
            .iter()
            .find(|e| e.outvar.name == operand.name)
        else {
            return zeros();
        };
        if let Primitive::Add(a, b) = &eqn.primitive {
            for atom in [a, b] {
                if let AtomKind::Const(c) = &atom.kind {
                    return &zeros() + c;
                }
            }
        }
        zeros()
    }

    fn linear_lower_bound<T: Value>(
        graph: &ComputeGraph,
        var_bounds: &Env<IBPTensor>,
        alphas: &Env<T>,
        betas: &Env<T>,
        splits: &Env<Tensor>,
        start: Option<String>,
    ) -> Result<(AffineBounds<T>, Env<Tensor>), EvalError> {
        let mut relu_bias: Env<Tensor> = Env::new();
        for eqn in &graph.equations {
            if let Primitive::Relu(operand) = &eqn.primitive {
                let shape = var_bounds.resolve(operand)?.shape().to_vec();
                relu_bias.insert(
                    eqn.outvar.name.clone(),
                    Self::feeding_bias(graph, operand, &shape),
                );
            }
        }

        let mut scores = Env::new();
        let bound = linear_lower_bound(
            graph,
            var_bounds,
            |outvar, out_w, x, activation| {
                // get the alpha crown bound
                let aff = AlphaCrownInterpreter::crown_activation(
                    &alphas.resolve(outvar)?,
                    out_w,
                    x,
                    activation,
                )?;

                // Score this ReLU for branching (concrete pass only).
                if let (Some(pre), Activation::Relu(_)) = (out_w.as_tensor(), activation) {
                    let bias = relu_bias
                        .get(&outvar.name)
                        .cloned()
                        .unwrap_or_else(|| Tensor::from(ArrayD::zeros(ndarray::IxDyn(x.shape()))));
                    scores.insert(outvar.name.clone(), Self::bunel_score(pre, x, &bias));
                }

                // actual beta * s contribution
                let s = splits
                    .get(&outvar.name)
                    .cloned()
                    .unwrap_or_else(|| Tensor::from(ArrayD::zeros(ndarray::IxDyn(x.shape()))));
                let beta = betas.resolve(outvar)?;
                Ok(AffineBounds {
                    weights: aff.weights - beta * T::from(s),
                    biases: aff.biases,
                })
            },
            start,
        )?;
        Ok((bound, scores))
    }

    fn beta_crown_optim(
        graph: &ComputeGraph,
        ibp_bounds: &Env<IBPTensor>,
        splits: &Env<Tensor>,
        mut alphas: Env<Tensor>,
        mut betas: Env<Tensor>,
        start: Option<String>,
    ) -> Result<(AffineBounds<Tensor>, Env<Tensor>, Env<Tensor>, Env<Tensor>), EvalError> {
        if alphas.len() > 0 {
            const ITERS: usize = 10;
            const LR: f64 = 0.01;

            let invar_bounds = ibp_bounds.resolve(&graph.invars[0])?;

            let extra_invars: Vec<(String, Vec<usize>)> = graph
                .equations
                .iter()
                .filter_map(|eqn| match &eqn.primitive {
                    Primitive::Relu(_)
                    | Primitive::LeakyRelu { .. }
                    | Primitive::Elu { .. }
                    | Primitive::Gelu(_) => alphas
                        .get(&eqn.outvar.name)
                        .map(|a| (eqn.outvar.name.clone(), a.shape().to_vec())),
                    _ => None,
                })
                .flat_map(|(name, shape)| {
                    [
                        (name.clone(), shape.clone()),
                        (format!("{name}{BETA_SUFFIX}"), shape),
                    ]
                })
                .collect();

            let grad_graph = try_trace_graph(
                graph,
                Some(extra_invars),
                |tracer_params| -> Result<Tracer, EvalError> {
                    // params
                    let mut a_env = Env::new();
                    let mut b_env = Env::new();
                    for eqn in &graph.equations {
                        if let Some(a) = tracer_params.get(&eqn.outvar.name) {
                            a_env.insert(eqn.outvar.name.clone(), a.clone());
                        }
                        if let Some(bt) =
                            tracer_params.get(&format!("{}{BETA_SUFFIX}", eqn.outvar.name))
                        {
                            b_env.insert(eqn.outvar.name.clone(), bt.clone());
                        }
                    }
                    // affine_lb (tracer pass: scores come back empty, discard)
                    let (alb, _) = Self::linear_lower_bound(
                        graph,
                        ibp_bounds,
                        &a_env,
                        &b_env,
                        splits,
                        start.clone(),
                    )?;
                    // concretize
                    Ok(alb.concretize(
                        &Tracer::from(invar_bounds.lb.clone()),
                        &Tracer::from(invar_bounds.ub.clone()),
                    ))
                },
            )?;

            for _ in 0..ITERS {
                let param_inputs: Vec<Tensor> = grad_graph
                    .invars
                    .iter()
                    .map(|a| match a.name.strip_suffix(BETA_SUFFIX) {
                        Some(name) => betas.get(name).cloned().expect("beta not found"),
                        None => alphas.get(&a.name).cloned().expect("alpha not found"),
                    })
                    .collect();

                let grads = GradInterpreter::run(&grad_graph, &param_inputs)?;

                for (invar, grad) in grad_graph.invars.iter().zip(grads) {
                    match invar.name.strip_suffix(BETA_SUFFIX) {
                        Some(name) => {
                            // project b into >= 0
                            let beta = betas.get(name).cloned().unwrap();
                            betas
                                .update(&name.to_string(), (beta + grad * LR).mapv(|v| v.max(0.0)));
                        }
                        None => {
                            // project alpha into [0, 1]: lower ReLU relaxation slope.
                            let alpha = alphas.get(&invar.name).cloned().unwrap();
                            alphas.update(
                                &invar.name,
                                (alpha + grad * LR).mapv(|v| v.clamp(0.0, 1.0)),
                            );
                        }
                    }
                }
            }
        }

        let (bound, scores) =
            Self::linear_lower_bound(graph, ibp_bounds, &alphas, &betas, splits, start)?;
        Ok((bound, alphas, betas, scores))
    }

    pub fn run(
        graph: &ComputeGraph,
        var_bounds: &Env<IBPTensor>,
        splits: &Env<Tensor>,
        start: Option<String>,
        warmstart: Option<(Env<Tensor>, Env<Tensor>)>,
    ) -> Result<(AffineBounds<Tensor>, Env<Tensor>, Env<Tensor>, Env<Tensor>), EvalError> {
        let (mut alphas, mut betas) = warmstart.unwrap_or_else(|| (Env::new(), Env::new()));

        for eqn in &graph.equations {
            let operand = match &eqn.primitive {
                Primitive::Relu(operand)
                | Primitive::LeakyRelu { operand, .. }
                | Primitive::Elu { operand, .. }
                | Primitive::Gelu(operand) => operand,
                _ => continue,
            };

            // Seed any params not supplied by the warm start.
            if alphas.get(&eqn.outvar.name).is_none() {
                let bound = var_bounds.resolve(operand)?;
                let alpha = Zip::from(&bound.lb)
                    .and(&bound.ub)
                    .map_collect(|&l, &u| if -l >= u { 0.0 } else { 1.0 });
                alphas.insert(eqn.outvar.name.clone(), Tensor::from(alpha));
            }
            if betas.get(&eqn.outvar.name).is_none() {
                let bound = var_bounds.resolve(operand)?;
                betas.insert(
                    eqn.outvar.name.clone(),
                    Tensor::from(ArrayD::zeros(ndarray::IxDyn(bound.lb.shape()))),
                );
            }
        }

        Self::beta_crown_optim(graph, &var_bounds, &splits, alphas, betas, start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreters::{bounds::ibp::IBPInterpreter, compute_graph::trace};
    use ndarray::IxDyn;

    fn arr(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from(ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap())
    }

    fn scalar(t: &Tensor) -> f64 {
        t.iter().copied().sum()
    }

    // A tiny one-hidden-layer ReLU net: out = relu(x·W1 + b1)·W2 + b2, with mixed
    // sign weights so the hidden neurons are genuinely unstable over the box.
    const W1: [f64; 6] = [1.0, -1.0, 0.5, -0.5, 1.0, 1.0]; // [2, 3]
    const B1: [f64; 3] = [0.1, -0.2, 0.0];
    const W2: [f64; 3] = [1.0, -1.0, 0.5];
    const B2: f64 = 0.0;

    fn relu_net() -> ComputeGraph {
        trace([("x".to_string(), vec![2])], |env| {
            let x = env.get("x").unwrap().clone();
            let h = (x.dot(&Tracer::from(arr(&W1, &[2, 3]))).unwrap()
                + Tracer::from(arr(&B1, &[3])))
            .relu();
            h.dot(&Tracer::from(arr(&W2, &[3]))).unwrap() + Tracer::from(arr(&[B2], &[]))
        })
    }

    fn eval_net(x: &Tensor) -> Tensor {
        let z = x.dot(&arr(&W1, &[2, 3])).unwrap() + arr(&B1, &[3]);
        z.relu().dot(&arr(&W2, &[3])).unwrap() + arr(&[B2], &[])
    }

    fn pre_activation(x: &Tensor) -> Tensor {
        x.dot(&arr(&W1, &[2, 3])).unwrap() + arr(&B1, &[3])
    }

    /// IBP forward pass over the whole net, producing bounds for every var — the
    /// `var_bounds` env `run` now expects (it no longer bounds the net itself).
    fn ibp_forward(graph: &ComputeGraph, lb: &Tensor, ub: &Tensor) -> Env<IBPTensor> {
        let mut bounds = Env::new();
        bounds.insert(
            graph.invars[0].name.clone(),
            IBPTensor::new(lb.clone(), ub.clone()),
        );
        for eqn in &graph.equations {
            let out = IBPInterpreter::process_primitive(&eqn.primitive, &bounds).unwrap();
            bounds.insert(eqn.outvar.name.clone(), out);
        }
        bounds
    }

    /// With no splits the beta term vanishes (`s ≡ 0`), so beta-CROWN must reduce
    /// exactly to alpha-CROWN: same init, same alpha ascent, β pinned at 0.
    #[test]
    fn no_splits_matches_alpha_crown() {
        let graph = relu_net();
        let (lb, ub) = (arr(&[-1.0, -1.0], &[2]), arr(&[1.0, 1.0], &[2]));

        let inputs = vec![IBPTensor::new(lb.clone(), ub.clone())];
        let var_bounds = ibp_forward(&graph, &lb, &ub);

        let (alpha_lb, _) = AlphaCrownInterpreter::run(&graph, &inputs).unwrap();
        let (beta_lb, _, _, _) =
            BetaCrownInterpreter::run(&graph, &var_bounds, &Env::new(), None, None).unwrap();

        let a = scalar(&alpha_lb.concretize(&lb, &ub));
        let b = scalar(&beta_lb.concretize(&lb, &ub));
        assert!(
            (a - b).abs() < 1e-9,
            "beta-CROWN with empty splits diverged from alpha-CROWN: {a} vs {b}",
        );
    }

    /// The bound must remain a valid lower bound of the output over the region cut
    /// out by the split constraints. We split every hidden neuron to the phase it
    /// takes at the box center (guaranteeing a non-empty region) and sample: every
    /// feasible point's output must sit at or above the reported bound.
    #[test]
    fn split_bound_is_sound() {
        let graph = relu_net();
        let (lb, ub) = (arr(&[-1.0, -1.0], &[2]), arr(&[1.0, 1.0], &[2]));
        let var_bounds = ibp_forward(&graph, &lb, &ub);

        // Split sign = sign of each pre-activation at the center → center feasible.
        let center = arr(&[0.0, 0.0], &[2]);
        let s: Tensor = pre_activation(&center).mapv(|z| if z >= 0.0 { 1.0 } else { -1.0 });

        let relu_name = graph
            .equations
            .iter()
            .find(|e| matches!(e.primitive, Primitive::Relu(_)))
            .unwrap()
            .outvar
            .name
            .clone();
        let mut splits = Env::new();
        splits.insert(relu_name, s.clone());

        let (bound, _, _, _) =
            BetaCrownInterpreter::run(&graph, &var_bounds, &splits, None, None).unwrap();
        let bound_val = scalar(&bound.concretize(&lb, &ub));

        // Grid sample the box; check only points consistent with the split.
        let n = 40;
        let mut feasible = 0;
        for i in 0..=n {
            for j in 0..=n {
                let x = arr(
                    &[
                        -1.0 + 2.0 * i as f64 / n as f64,
                        -1.0 + 2.0 * j as f64 / n as f64,
                    ],
                    &[2],
                );
                let z = pre_activation(&x);
                let feasible_here = Zip::from(&z).and(&s).all(|&zj, &sj| sj * zj >= -1e-12);
                if !feasible_here {
                    continue;
                }
                feasible += 1;
                let out = scalar(&eval_net(&x));
                assert!(
                    out >= bound_val - 1e-6,
                    "beta-CROWN bound unsound: output {out} < bound {bound_val} at {x:?}",
                );
            }
        }
        assert!(
            feasible > 0,
            "split region had no sampled points (vacuous test)"
        );
    }

    /// The BaBSR bias term must use the real feeding-layer bias: `feeding_bias`
    /// pulls `B1` out of the `Add` that produces the ReLU's pre-activation,
    /// broadcast to the pre-activation shape — not the zero placeholder.
    #[test]
    fn feeding_bias_extracts_layer_bias() {
        let graph = relu_net();
        let relu = graph
            .equations
            .iter()
            .find(|e| matches!(e.primitive, Primitive::Relu(_)))
            .unwrap();
        let Primitive::Relu(operand) = &relu.primitive else {
            unreachable!()
        };

        let bias = BetaCrownInterpreter::feeding_bias(&graph, operand, &[B1.len()]);
        assert_eq!(bias.iter().copied().collect::<Vec<_>>(), B1.to_vec());
    }
}
