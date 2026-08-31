use ndarray::Zip;

use crate::{
    interpreters::{
        EvalError, Interpreter,
        bounds::{
            affine::crown::{crown_relu, linear_lower_bound},
            ibp::IBPInterpreter,
            ibp_util::IBPTensor,
            lbp_util::{AffineBounds, bound_convex, elu_lines, gelu_lines},
        },
        compute_graph::{tracer::Tracer, try_trace_graph},
        concrete::{eval_util::Tensor, grad::GradInterpreter},
    },
    mininn::{Activation, ComputeGraph, Env, Primitive, Value},
};

pub struct AlphaCrownInterpreter {}

impl AlphaCrownInterpreter {
    pub(super) fn crown_activation<T: Value>(
        alpha: &T,
        out_w: &T,
        x: &IBPTensor,
        activation: &Activation,
    ) -> Result<AffineBounds<T>, EvalError> {
        match activation {
            Activation::Relu(_) => crown_relu(alpha, out_w, x, 0.0),
            Activation::LeakyRelu { slope, .. } => crown_relu(alpha, out_w, x, *slope),
            Activation::Elu { slope, .. } => Ok(bound_convex(
                x,
                out_w,
                |l, u| elu_lines(l, u, *slope),
                |_, _| Ok(()),
            )?),
            // Like Elu, GELU ignores `alpha` and uses fixed relaxation lines.
            Activation::Gelu(_) => Ok(bound_convex(x, out_w, gelu_lines, |_, _| Ok(()))?),
        }
    }

    pub fn alpha_crown_optim(
        graph: &ComputeGraph,
        ibp_bounds: &Env<IBPTensor>,
        mut params: Env<Tensor>,
    ) -> Result<AffineBounds<Tensor>, EvalError> {
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
                    let alb = linear_lower_bound(
                        graph,
                        ibp_bounds,
                        |outvar, out_w, x, activation| {
                            Self::crown_activation(
                                &tracer_params.resolve(outvar)?,
                                out_w,
                                x,
                                activation,
                            )
                        },
                        None,
                    )?;
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

        linear_lower_bound(
            graph,
            &ibp_bounds,
            |outvar, out_w, x, activation| {
                Self::crown_activation(&params.resolve(outvar)?, out_w, x, activation)
            },
            None,
        )
    }
}

impl AlphaCrownInterpreter {
    pub fn run(
        graph: &ComputeGraph,
        inputs: &Vec<IBPTensor>,
    ) -> Result<(AffineBounds<Tensor>, AffineBounds<Tensor>), EvalError> {
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
        let ub = AffineBounds {
            weights: -lb_neg.weights,
            biases: -lb_neg.biases,
        };

        Ok((lb, ub))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        interpreters::{
            bounds::{
                affine::crown::{crown_dot, crown_pad},
                lbp_util::{exp_lines, normcdf_lines, reciprocal_lines, sqrt_lines, square_lines},
            },
            concrete::{eval_util::normcdf, grad::unbroadcast},
        },
        mininn::{PaddingOptionConfig, PaddingOptions},
    };

    use super::*;
    use ndarray::{ArrayD, IxDyn};

    /// The GELU relaxation must sandwich `g(x) = x·Φ(x)` over the whole interval.
    /// Enable (drop `#[ignore]`) once `gelu_lines` is implemented.
    #[test]
    // #[ignore = "enable once gelu_lines is implemented"]
    fn gelu_lines_are_sound() {
        use crate::interpreters::bounds::lbp_util::gelu_lines;
        let g = |x: f64| x * normcdf(x);
        // Convex-only, concave-only (both tails), crossing one/both inflections
        // at ±√2, plus narrow/degenerate.
        let cases = [
            (-1.0, 1.0),   // fully convex
            (-1.4, 1.4),   // convex, up to the inflection edges
            (-3.0, -1.5),  // fully concave (left tail)
            (1.5, 3.0),    // fully concave (right tail)
            (-2.0, 0.5),   // crosses left inflection
            (-0.5, 2.0),   // crosses right inflection
            (-3.0, 3.0),   // crosses both inflections
            (-2.0, -1.0),  // straddles left inflection
            (1.0, 3.0),    // crosses right inflection, both endpoints steep
            (1.4, 1.45),   // crosses right inflection, very narrow convex sliver
            (-1.6, -1.41), // straddles left inflection, narrow
            (-6.0, 6.0),   // crosses both, wide
            (-0.01, 0.01), // near-degenerate around the origin
            (0.7, 0.7),    // degenerate
        ];

        let tol = 1e-6;
        let n = 400;
        let check = |l: f64, u: f64| {
            let (ls, lo, us, uo) = gelu_lines(l, u);
            for i in 0..=n {
                let x = l + (u - l) * (i as f64) / (n as f64);
                let f = g(x);
                assert!(
                    ls * x + lo <= f + tol,
                    "lower line above g on [{l}, {u}] at x={x}: {} > {f}",
                    ls * x + lo,
                );
                assert!(
                    us * x + uo >= f - tol,
                    "upper line below g on [{l}, {u}] at x={x}: {} < {f}",
                    us * x + uo,
                );
            }
        };

        for (l, u) in cases {
            check(l, u);
        }

        // Dense grid sweep over endpoints straddling the ±√2 inflections.
        let grid = 40;
        for i in 0..=grid {
            let l = -4.0 + 8.0 * (i as f64) / (grid as f64);
            for j in 0..=grid {
                let u = l + 6.0 * (j as f64) / (grid as f64);
                check(l, u);
            }
        }
    }

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

            let aff = crown_pad(&out_w, &operand_shape, &options);

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
        let affs = crown_dot(out_w, &x, &y).unwrap();
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
