use ndarray::{ArrayD, Axis, IxDyn};

use crate::{
    interpreters::{
        EvalError, Interpreter,
        concrete::{
            eval::EvalInterpreter,
            eval_util::{Tensor, norm_axis_index},
        },
    },
    mininn::{
        Atom, AtomKind, ComputeGraph, Env, PaddingOptionConfig, PaddingOptions, PoolOptions,
        Primitive, Value,
    },
};

pub struct GradInterpreter;

pub fn unbroadcast<T: Value>(g: &T, target: &[usize]) -> T {
    let mut g = g.clone();
    while g.ndim() > target.len() {
        g = g.reduce_sum(&[0]);
    }
    for i in 0..target.len() {
        if target[i] == 1 && g.shape()[i] > 1 {
            g = g.reduce_sum(&[i as isize]).expand_dims(&[i as isize]);
        }
    }
    g
}

fn vjp_where<T: Value>(tangent: &T, condition: &T) -> Result<Vec<T>, EvalError> {
    let zero: T = ArrayD::zeros(IxDyn(condition.shape())).into();
    let first = T::r#where(condition, tangent, &zero)?;
    let second = T::r#where(condition, &zero, tangent)?;
    Ok(vec![zero, first, second])
}

fn vjp_dot<T: Value>(tangent: &T, a: &T, b: &T) -> Result<Vec<T>, EvalError> {
    let dx = if b.ndim() == 0 {
        tangent.clone() * b.clone()
    } else if b.ndim() == 1 {
        tangent.expand_dims(&[-1]).dot(&b.expand_dims(&[0]))?
    } else {
        tangent.dot(&b.moveaxis(-1, -2))?
    };

    let dy = if a.ndim() == 0 {
        a.clone() * tangent.clone()
    } else if a.ndim() == 1 {
        a.expand_dims(&[-1]).dot(&tangent.expand_dims(&[0]))?
    } else {
        a.moveaxis(-1, -2).dot(tangent)?
    };

    Ok(vec![dx, dy])
}

fn vjp_conv<T: Value>(
    tangent: &T,
    input: &T,
    kernel: &T,
    stride: isize,
) -> Result<Vec<T>, EvalError> {
    if input.ndim() != 4 || kernel.ndim() != 4 {
        return Err(EvalError::Eval(
            "vjp_conv expects 4-D input and kernel".to_string(),
        ));
    }

    let s = stride as usize;
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    // d_input: dilate tangent, pad borders, convolve with spatially-flipped transposed kernel
    let dilated = tangent.pad(&PaddingOptions {
        axes: vec![-2, -1],
        config: PaddingOptionConfig {
            left: 0,
            interior: s - 1,
            right: 0,
        },
        value: 0.0,
    });
    let padded = dilated
        .pad(&PaddingOptions {
            axes: vec![-2],
            config: PaddingOptionConfig {
                left: kh - 1,
                interior: 0,
                right: kh - 1,
            },
            value: 0.0,
        })
        .pad(&PaddingOptions {
            axes: vec![-1],
            config: PaddingOptionConfig {
                left: kw - 1,
                interior: 0,
                right: kw - 1,
            },
            value: 0.0,
        });
    // flip kernel spatial dims and swap OC/IC: [OC,IC,KH,KW] → [IC,OC,KH,KW] flipped
    let k_flip = kernel
        .moveaxis(0, 1)
        .slice(-2, -1, None, -1)
        .slice(-1, -1, None, -1);
    let d_input = padded.conv(&k_flip, 1)?;

    // d_kernel via the ConvKernelGrad primitive
    let d_kernel = tangent.conv_kernel_grad(input, stride, kernel.shape())?;

    Ok(vec![d_input, d_kernel])
}

fn vjp_pad<T: Value>(tangent: &T, a: &T, opt: &PaddingOptions) -> Vec<T> {
    let mut t = tangent.clone();
    for &ax in &opt.axes {
        let axis = norm_axis_index(ax, a.ndim());
        let si = a.shape()[axis];
        let left = opt.config.left as isize;
        let step = (opt.config.interior + 1) as isize;
        let end = if si == 0 {
            left
        } else {
            left + step * (si as isize - 1) + 1
        };
        t = t.slice(ax, left, Some(end), step);
    }
    vec![t]
}

fn vjp_pool<T: Value>(tangent: &T, input: &T, opt: &PoolOptions, average: bool) -> Vec<T> {
    let ndim = input.ndim();
    let window_total: usize = opt.window_size.iter().product();
    let scale = T::from(if average {
        1.0 / window_total as f64
    } else {
        1.0_f64
    });

    let mut d_input: Option<T> = None;

    for flat_j in 0..window_total {
        // decode flat index into per-axis window offset
        let mut j = vec![0usize; ndim];
        let mut rem = flat_j;
        for d in (0..ndim).rev() {
            j[d] = rem % opt.window_size[d];
            rem /= opt.window_size[d];
        }

        // scatter tangent contribution for this window offset via pad
        let mut t = tangent.clone() * scale.clone();
        for d in 0..ndim {
            let nd = input.shape()[d];
            let md = tangent.shape()[d];
            let sd = opt.stride[d];
            let right = nd - j[d] - (md - 1) * sd - 1;
            t = t.pad(&PaddingOptions {
                axes: vec![d as isize],
                config: PaddingOptionConfig {
                    left: j[d],
                    interior: sd - 1,
                    right,
                },
                value: 0.0,
            });
        }

        d_input = Some(match d_input {
            None => t,
            Some(acc) => acc + t,
        });
    }

    vec![d_input.unwrap_or_else(|| T::from(ArrayD::zeros(IxDyn(input.shape()))))]
}

pub fn vjp_reducesum<T: Value>(operand: &T, tangent: &T, axes: &[isize]) -> Vec<T> {
    let expanded = tangent.expand_dims(axes);
    vec![expanded + T::from(ArrayD::zeros(IxDyn(operand.shape())))]
}

pub fn vjp_expanddims<T: Value>(tangent: &T, axes: &[isize]) -> Vec<T> {
    vec![tangent.reduce_sum(axes)]
}

pub fn vjp_moveaxis<T: Value>(tangent: &T, source: isize, destination: isize) -> Vec<T> {
    vec![tangent.moveaxis(destination, source)]
}

pub fn vjp_reshape<T: Value>(operand: &T, tangent: &T) -> Result<Vec<T>, EvalError> {
    let orig_shape: Vec<isize> = operand.shape().iter().map(|&x| x as isize).collect();
    Ok(vec![tangent.reshape(&orig_shape)?])
}

fn softmax_xent_vjp(logits: &Tensor, labels: &Tensor) -> Result<Tensor, EvalError> {
    let n = logits.shape()[0] as f64;
    let logits_arr = logits.clone().into_inner();
    let labels_arr = labels.clone().into_inner();

    let max = logits_arr.map_axis(Axis(1), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    let shifted = logits_arr - &max.insert_axis(Axis(1));
    let exp = shifted.mapv(f64::exp);
    let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
    let probs = exp / sum;

    Ok(((probs - labels_arr) / n).into())
}

impl GradInterpreter {
    pub fn process_primitive<T: Value>(
        primitive: &Primitive,
        outvar: &Atom,
        primals: &Env<T>,
        env: &Env<T>,
    ) -> Result<Vec<T>, EvalError> {
        let p = |a: &Atom| primals.resolve(a);

        let tangent = env.resolve(outvar)?;

        use Primitive::*;
        Ok(match primitive {
            // elementwise unary
            Neg(_) => vec![-tangent],
            Reciprocal(a) => vec![-(p(a)?.square().reciprocal()) * tangent],
            Square(a) => vec![tangent * T::from(2.0) * p(a)?],
            Sqrt(_) => vec![tangent * (p(outvar)? * T::from(2.0)).reciprocal()],
            Exp(_) => vec![tangent * p(outvar)?],
            Log(a) => vec![tangent * p(a)?.reciprocal()],
            // elementwise binary (numpy broadcasting)
            Add(_, _) => vec![tangent.clone(), tangent],
            Mul(a, b) => vec![tangent.clone() * p(b)?, p(a)? * tangent],
            Where(c, _, _) => vjp_where(&tangent, &p(c)?)?,
            // activations
            Relu(_) => {
                let zero = T::from(ArrayD::zeros(IxDyn(tangent.shape())));
                vec![T::r#where(&p(outvar)?, &tangent, &zero)?]
            }
            LeakyRelu { operand, slope } => {
                let x = p(operand)?;
                let steep = tangent.clone() * T::from(*slope);
                vec![T::r#where(&x.relu(), &tangent, &steep)?]
            }
            Elu { operand, slope } => {
                let val = p(operand)?;
                let d_neg = tangent.clone() * T::from(*slope) * val.exp();
                vec![T::r#where(&val, &tangent, &d_neg)?]
            }
            Gelu(a) => {
                let val = p(a)?;
                let pdf = (-(val.square() * T::from(0.5))).exp()
                    * T::from(1.0 / (2.0 * std::f64::consts::PI).sqrt());
                vec![tangent * val.normcdf() + val * pdf]
            }
            NormalCdf(a) => {
                let val = p(a)?;
                let pdf = (-(val.square() * T::from(0.5))).exp()
                    * T::from(1.0 / (2.0 * std::f64::consts::PI).sqrt());
                vec![tangent * pdf]
            }
            // linear algebra
            Dot(a, b) => return vjp_dot(&tangent, &p(a)?, &p(b)?),
            // reduction
            ReduceSum { operand, axes } => vjp_reducesum(&p(operand)?, &tangent, axes),
            // shape manipulation
            ExpandDims { operand: _, axes } => vjp_expanddims(&tangent, axes),
            MoveAxis {
                operand: _,
                source,
                destination,
            } => vjp_moveaxis(&tangent, *source, *destination),
            Reshape {
                operand,
                new_shape: _,
            } => vjp_reshape(&p(operand)?, &tangent)?,
            // slicing
            Slice {
                operand,
                axis,
                start,
                end,
                step,
            } => {
                let op = p(operand)?;
                let ax = norm_axis_index(*axis, op.shape().len());
                let n = tangent.shape()[ax];
                let s = *start as usize;
                let st = *step as usize;
                let right = op.shape()[ax] - s - if n > 0 { (n - 1) * st + 1 } else { 0 };
                vec![tangent.pad(&PaddingOptions {
                    axes: vec![*axis],
                    config: PaddingOptionConfig {
                        left: s,
                        interior: st - 1,
                        right,
                    },
                    value: 0.0,
                })]
            }
            // padding
            Pad { operand, options } => vjp_pad(&tangent, &p(operand)?, options),
            // 2d convolution
            Conv {
                input,
                kernel,
                options,
            } => return vjp_conv(&tangent, &p(input)?, &p(kernel)?, options.stride),
            ConvKernelGrad { .. } => todo!("second-order conv gradient not yet implemented"),
            // pooling
            AvgPool { operand, options } => vjp_pool(&tangent, &p(operand)?, options, true),
            SumPool { operand, options } => vjp_pool(&tangent, &p(operand)?, options, false),
        })
    }

    pub fn run_loss(
        graph: &ComputeGraph,
        inputs: &Vec<Tensor>,
        labels: Option<&Tensor>,
    ) -> Result<Vec<Tensor>, EvalError> {
        // ---- FORWARD (primals) ----
        let mut primals: Env<Tensor> = Env::new();

        for (var, tensor) in graph.invars.iter().zip(inputs) {
            primals.insert(var.name.clone(), tensor.clone());
        }

        for eqn in &graph.equations {
            let out = EvalInterpreter::process_primitive(&eqn.primitive, &primals)?;
            primals.insert(eqn.outvar.name.clone(), out);
        }

        // ---- BACKWARD ----
        let mut env: Env<Tensor> = Env::new();

        fn combine(var: &Atom, tangent: Tensor, env: &mut Env<Tensor>) {
            if let AtomKind::Const(_) = var.kind {
                return;
            }

            if let Some(t) = env.get(&var.name) {
                env.update(&var.name, t.clone() + unbroadcast(&tangent, &var.shape));
            } else {
                env.insert(var.name.clone(), unbroadcast(&tangent, &var.shape));
            }
        }

        for var in &graph.outvars {
            match var.kind {
                AtomKind::Var => {
                    let cotangent: Tensor = match labels {
                        None => ArrayD::ones(IxDyn(&var.shape)).into(),
                        Some(y) => {
                            let logits = primals.get(&var.name).ok_or_else(|| {
                                EvalError::Eval(format!(
                                    "output '{}' not found in primals",
                                    var.name
                                ))
                            })?;
                            softmax_xent_vjp(logits, y)?
                        }
                    };
                    env.insert(var.name.clone(), cotangent);
                }
                AtomKind::Const(_) => continue,
            }
        }

        for eqn in graph.equations.iter().rev() {
            if env.get(&eqn.outvar.name).is_none() {
                continue;
            }

            let out = Self::process_primitive(&eqn.primitive, &eqn.outvar, &primals, &env)?;

            for (atom, tangent) in eqn.primitive.operands().into_iter().zip(out) {
                combine(atom, tangent, &mut env);
            }
        }

        graph
            .invars
            .iter()
            .map(|var| {
                Ok(env
                    .get(&var.name)
                    .cloned()
                    .unwrap_or_else(|| ArrayD::zeros(IxDyn(&var.shape)).into()))
            })
            .collect()
    }
}

impl Interpreter<Tensor> for GradInterpreter {
    fn run(graph: &ComputeGraph, inputs: &Vec<Tensor>) -> Result<Vec<Tensor>, EvalError> {
        GradInterpreter::run_loss(graph, inputs, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        interpreters::concrete::eval_util::normcdf,
        mininn::{Atom, AtomKind, Primitive},
    };
    use ndarray::{ArrayD, IxDyn};

    fn carr(data: &[f64], shape: &[usize]) -> Tensor {
        ArrayD::from_shape_vec(IxDyn(shape), data.to_vec())
            .unwrap()
            .into()
    }

    fn const_atom(data: &[f64], shape: &[usize]) -> Atom {
        Atom {
            name: String::new(),
            shape: shape.to_vec(),
            kind: AtomKind::Const(carr(data, shape)),
        }
    }

    /// Run VJP for `primitive`. `outvar_primal` must contain the forward-pass output
    /// for primitives that call `p(outvar)` (Sqrt, Exp, Relu); pass zeros otherwise.
    fn run_vjp(primitive: Primitive, outvar_primal: Tensor, tangent: Tensor) -> Vec<Tensor> {
        let outvar = Atom {
            name: "out".to_string(),
            shape: tangent.shape().to_vec(),
            kind: AtomKind::Var,
        };
        let mut primals: Env<Tensor> = Env::new();
        primals.insert("out".to_string(), outvar_primal);
        let mut env: Env<Tensor> = Env::new();
        env.insert("out".to_string(), tangent);
        GradInterpreter::process_primitive(&primitive, &outvar, &primals, &env).unwrap()
    }

    fn ones(shape: &[usize]) -> Tensor {
        ArrayD::ones(IxDyn(shape)).into()
    }
    fn zeros(shape: &[usize]) -> Tensor {
        ArrayD::zeros(IxDyn(shape)).into()
    }

    fn assert_close(a: &Tensor, b: &Tensor) {
        assert_eq!(a.shape(), b.shape(), "shapes differ");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-9, "values differ: {x} vs {y}");
        }
    }

    fn assert_close_tol(a: &Tensor, b: &Tensor, tol: f64) {
        assert_eq!(a.shape(), b.shape(), "shapes differ");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < tol, "values differ: {x} vs {y} (tol {tol})");
        }
    }

    /// Numerical VJP via central differences.
    fn num_vjp(x: &Tensor, tangent: &Tensor, f: impl Fn(&Tensor) -> Tensor) -> Tensor {
        let eps = 1e-5;
        let x_arr = x.view();
        let x_flat: Vec<f64> = x_arr.iter().copied().collect();
        let grad_flat: Vec<f64> = (0..x_flat.len())
            .map(|i| {
                let mut xp = x_flat.clone();
                let mut xm = x_flat.clone();
                xp[i] += eps;
                xm[i] -= eps;
                let fp = f(&ArrayD::from_shape_vec(x_arr.raw_dim(), xp).unwrap().into());
                let fm = f(&ArrayD::from_shape_vec(x_arr.raw_dim(), xm).unwrap().into());
                (&((fp - fm) / (2.0 * eps)) * tangent).sum()
            })
            .collect();
        ArrayD::from_shape_vec(x_arr.raw_dim(), grad_flat)
            .unwrap()
            .into()
    }

    // ---- unary ----

    #[test]
    fn neg() {
        let t = carr(&[2.0, -1.0, 3.0], &[3]);
        let g = run_vjp(
            Primitive::Neg(const_atom(&[0.0; 3], &[3])),
            zeros(&[3]),
            t.clone(),
        );
        assert_close(&g[0], &t.mapv(|x| -x));
    }

    #[test]
    fn reciprocal() {
        let xd = &[2.0, 4.0, -1.0];
        let x = carr(xd, &[3]);
        let t = carr(&[1.0, 2.0, 3.0], &[3]);
        let g = run_vjp(
            Primitive::Reciprocal(const_atom(xd, &[3])),
            zeros(&[3]),
            t.clone(),
        );
        assert_close_tol(&g[0], &num_vjp(&x, &t, |a| a.mapv(|v| 1.0 / v)), 1e-9);
    }

    #[test]
    fn square() {
        let xd = &[1.0, 2.0, 3.0];
        let x = carr(xd, &[3]);
        let t = ones(&[3]);
        let g = run_vjp(
            Primitive::Square(const_atom(xd, &[3])),
            zeros(&[3]),
            t.clone(),
        );
        assert_close_tol(&g[0], &num_vjp(&x, &t, |a| a.mapv(|v| v * v)), 1e-9);
    }

    #[test]
    fn sqrt() {
        let xd = &[1.0, 4.0, 9.0];
        let x = carr(xd, &[3]);
        let t = carr(&[1.0, 2.0, 0.5], &[3]);
        let outvar_primal = x.mapv(f64::sqrt);
        let g = run_vjp(
            Primitive::Sqrt(const_atom(xd, &[3])),
            outvar_primal,
            t.clone(),
        );
        assert_close_tol(&g[0], &num_vjp(&x, &t, |a| a.mapv(f64::sqrt)), 1e-9);
    }

    #[test]
    fn exp() {
        let xd = &[0.0, 1.0, -1.0];
        let x = carr(xd, &[3]);
        let t = ones(&[3]);
        let outvar_primal = x.mapv(f64::exp);
        let g = run_vjp(
            Primitive::Exp(const_atom(xd, &[3])),
            outvar_primal,
            t.clone(),
        );
        assert_close_tol(&g[0], &num_vjp(&x, &t, |a| a.mapv(f64::exp)), 1e-9);
    }

    #[test]
    fn log() {
        let xd = &[1.0, 2.0, 4.0];
        let x = carr(xd, &[3]);
        let t = carr(&[2.0, 1.0, 3.0], &[3]);
        let g = run_vjp(Primitive::Log(const_atom(xd, &[3])), zeros(&[3]), t.clone());
        assert_close_tol(&g[0], &num_vjp(&x, &t, |a| a.mapv(f64::ln)), 1e-9);
    }

    // ---- binary ----

    #[test]
    fn add() {
        let t = carr(&[1.0, 2.0, 3.0], &[3]);
        let g = run_vjp(
            Primitive::Add(const_atom(&[0.0; 3], &[3]), const_atom(&[0.0; 3], &[3])),
            zeros(&[3]),
            t.clone(),
        );
        assert_close(&g[0], &t);
        assert_close(&g[1], &t);
    }

    #[test]
    fn mul() {
        let ad = &[1.0, 2.0, 3.0];
        let bd = &[4.0, 5.0, 6.0];
        let a = carr(ad, &[3]);
        let b = carr(bd, &[3]);
        let t = ones(&[3]);
        let g = run_vjp(
            Primitive::Mul(const_atom(ad, &[3]), const_atom(bd, &[3])),
            zeros(&[3]),
            t.clone(),
        );
        assert_close(&g[0], &(&t * &b));
        assert_close(&g[1], &(&a * &t));
    }

    #[test]
    fn where_true() {
        let t = carr(&[3.0, 4.0], &[2]);
        let g = run_vjp(
            Primitive::Where(
                const_atom(&[1.0, 1.0], &[2]),
                const_atom(&[0.0; 2], &[2]),
                const_atom(&[0.0; 2], &[2]),
            ),
            zeros(&[2]),
            t.clone(),
        );
        assert_close(&g[0], &zeros(&[2])); // d_cond = 0
        assert_close(&g[1], &t); // d_x = t
        assert_close(&g[2], &zeros(&[2])); // d_y = 0
    }

    #[test]
    fn where_false() {
        let t = carr(&[3.0, 4.0], &[2]);
        let g = run_vjp(
            Primitive::Where(
                const_atom(&[0.0, 0.0], &[2]),
                const_atom(&[0.0; 2], &[2]),
                const_atom(&[0.0; 2], &[2]),
            ),
            zeros(&[2]),
            t.clone(),
        );
        assert_close(&g[0], &zeros(&[2]));
        assert_close(&g[1], &zeros(&[2])); // d_x = 0
        assert_close(&g[2], &t); // d_y = t
    }

    // ---- activations ----

    #[test]
    fn relu() {
        let g = run_vjp(
            Primitive::Relu(const_atom(&[2.0, -1.0, 0.0], &[3])),
            carr(&[2.0, 0.0, 0.0], &[3]), // outvar_primal = relu(x)
            ones(&[3]),
        );
        assert_close(&g[0], &carr(&[1.0, 0.0, 0.0], &[3]));
    }

    #[test]
    fn leaky_relu() {
        let slope = 0.1_f64;
        let g = run_vjp(
            Primitive::LeakyRelu {
                operand: const_atom(&[2.0, -3.0], &[2]),
                slope,
            },
            zeros(&[2]),
            ones(&[2]),
        );
        assert_close_tol(&g[0], &carr(&[1.0, slope], &[2]), 1e-10);
    }

    #[test]
    fn normalcdf() {
        let xd = &[0.0, 1.0, -1.0];
        let x = carr(xd, &[3]);
        let t = ones(&[3]);
        let g = run_vjp(
            Primitive::NormalCdf(const_atom(xd, &[3])),
            zeros(&[3]),
            t.clone(),
        );
        assert_close_tol(&g[0], &num_vjp(&x, &t, |a| a.mapv(|x| normcdf(x))), 1e-8);
    }

    // ---- linear algebra ----

    #[test]
    fn dot_1d_1d() {
        let t: Tensor = ArrayD::from_elem(IxDyn(&[]), 1.0).into();
        let g = run_vjp(
            Primitive::Dot(const_atom(&[1.0, 2.0], &[2]), const_atom(&[3.0, 4.0], &[2])),
            ArrayD::from_elem(IxDyn(&[]), 11.0).into(),
            t,
        );
        assert_close(&g[0], &carr(&[3.0, 4.0], &[2]));
        assert_close(&g[1], &carr(&[1.0, 2.0], &[2]));
    }

    #[test]
    fn dot_2d_2d() {
        let t = ones(&[2, 2]);
        let g = run_vjp(
            Primitive::Dot(
                const_atom(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
                const_atom(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
            ),
            zeros(&[2, 2]),
            t,
        );
        assert_close(&g[0], &carr(&[11.0, 15.0, 11.0, 15.0], &[2, 2]));
        assert_close(&g[1], &carr(&[4.0, 4.0, 6.0, 6.0], &[2, 2]));
    }

    #[test]
    fn dot_2d_1d() {
        let t = ones(&[2]);
        let g = run_vjp(
            Primitive::Dot(
                const_atom(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
                const_atom(&[1.0, 1.0], &[2]),
            ),
            zeros(&[2]),
            t,
        );
        assert_close(&g[0], &carr(&[1.0, 1.0, 1.0, 1.0], &[2, 2]));
        assert_close(&g[1], &carr(&[4.0, 6.0], &[2]));
    }

    #[test]
    fn dot_nd_1d() {
        let t = ones(&[2]);
        let g = run_vjp(
            Primitive::Dot(
                const_atom(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
                const_atom(&[1.0, 0.0, 0.0], &[3]),
            ),
            zeros(&[2]),
            t,
        );
        assert_close(&g[0], &carr(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 3]));
        assert_close(&g[1], &carr(&[5.0, 7.0, 9.0], &[3]));
    }

    // ---- reduction ----

    #[test]
    fn reduce_sum_axis() {
        let t = carr(&[1.0, 2.0], &[2]);
        let g = run_vjp(
            Primitive::ReduceSum {
                operand: const_atom(&[0.0; 6], &[2, 3]),
                axes: vec![1],
            },
            zeros(&[2]),
            t,
        );
        assert_close(&g[0], &carr(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3]));
    }

    #[test]
    fn reduce_sum_negative_axis() {
        let t = carr(&[1.0, 2.0, 3.0], &[3]);
        let g = run_vjp(
            Primitive::ReduceSum {
                operand: const_atom(&[0.0; 12], &[3, 4]),
                axes: vec![-1],
            },
            zeros(&[3]),
            t,
        );
        let expected: Vec<f64> = [1.0, 2.0, 3.0].iter().flat_map(|&v| vec![v; 4]).collect();
        assert_close(&g[0], &carr(&expected, &[3, 4]));
    }

    // ---- shape manipulation ----

    #[test]
    fn expand_dims_front() {
        let t = carr(&[2.0, 3.0, 4.0], &[1, 3]);
        let g = run_vjp(
            Primitive::ExpandDims {
                operand: const_atom(&[0.0; 3], &[3]),
                axes: vec![0],
            },
            zeros(&[1, 3]),
            t,
        );
        assert_close(&g[0], &carr(&[2.0, 3.0, 4.0], &[3]));
    }

    #[test]
    fn expand_dims_back() {
        let t = carr(&[1.0, 2.0, 3.0], &[3, 1]);
        let g = run_vjp(
            Primitive::ExpandDims {
                operand: const_atom(&[0.0; 3], &[3]),
                axes: vec![-1],
            },
            zeros(&[3, 1]),
            t,
        );
        assert_close(&g[0], &carr(&[1.0, 2.0, 3.0], &[3]));
    }

    #[test]
    fn move_axis() {
        let t = carr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let g = run_vjp(
            Primitive::MoveAxis {
                operand: const_atom(&[0.0; 6], &[2, 3]),
                source: 0,
                destination: -1,
            },
            zeros(&[3, 2]),
            t,
        );
        assert_close(&g[0], &carr(&[1.0, 3.0, 5.0, 2.0, 4.0, 6.0], &[2, 3]));
    }

    #[test]
    fn reshape() {
        let t = carr(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 3]);
        let g = run_vjp(
            Primitive::Reshape {
                operand: const_atom(&[0.0; 6], &[6]),
                new_shape: vec![2, 3],
            },
            zeros(&[2, 3]),
            t,
        );
        assert_close(&g[0], &carr(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[6]));
    }
}
