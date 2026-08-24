use ndarray::{ArrayD, Axis, IxDyn, Slice, indices};

use crate::{
    interpreters::{
        EvalError, Interpreter,
        concrete::{
            eval::EvalInterpreter,
            eval_util::{Tensor, norm_axis_index},
        },
    },
    mininn::{Atom, AtomKind, ComputeGraph, Env, PaddingOptions, PoolOptions, Primitive, Value},
};

pub struct GradInterpreter;

fn unbroadcast(g: Tensor, target: &[usize]) -> Tensor {
    let mut arr = g.into_inner();
    while arr.ndim() > target.len() {
        arr = arr.sum_axis(Axis(0));
    }
    for i in 0..target.len() {
        if target[i] == 1 && arr.shape()[i] > 1 {
            arr = arr.sum_axis(Axis(i)).insert_axis(Axis(i));
        }
    }
    arr.into()
}

fn vjp_where(tangent: &Tensor, condition: &Tensor) -> Result<Vec<Tensor>, EvalError> {
    let zero: Tensor = ArrayD::zeros(IxDyn(condition.shape())).into();
    let first = Tensor::r#where(condition, tangent, &zero)?;
    let second = Tensor::r#where(condition, &zero, tangent)?;
    Ok(vec![zero, first, second])
}

fn vjp_dot(tangent: &Tensor, a: &Tensor, b: &Tensor) -> Result<Vec<Tensor>, EvalError> {
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

fn vjp_conv(
    tangent: &Tensor,
    input: &Tensor,
    kernel: &Tensor,
    stride: isize,
) -> Result<Vec<Tensor>, EvalError> {
    if input.ndim() != 4 || kernel.ndim() != 4 {
        return Err(EvalError::Eval(
            "vjp_conv expects 4-D input and kernel".to_string(),
        ));
    }

    let s = stride as usize;
    let (n, c, h, w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let (ko, kh, kw) = (kernel.shape()[0], kernel.shape()[2], kernel.shape()[3]);
    let oh = (h - kh) / s + 1;
    let ow = (w - kw) / s + 1;

    let t_arr = tangent.inner();
    let inp_arr = input.inner();
    let ker_arr = kernel.inner();

    let mut d_input = ArrayD::zeros(IxDyn(&[n, c, h, w]));
    let mut d_kernel = ArrayD::zeros(IxDyn(kernel.shape()));

    for ni in 0..n {
        for ci in 0..ko {
            for hi in 0..oh {
                for wi in 0..ow {
                    let t = t_arr[[ni, ci, hi, wi]];
                    for cpi in 0..c {
                        for i in 0..kh {
                            for j in 0..kw {
                                d_input[[ni, cpi, s * hi + i, s * wi + j]] +=
                                    t * ker_arr[[ci, cpi, i, j]];
                                d_kernel[[ci, cpi, i, j]] +=
                                    t * inp_arr[[ni, cpi, s * hi + i, s * wi + j]];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(vec![d_input.into(), d_kernel.into()])
}

fn vjp_pad(tangent: &Tensor, a: &Tensor, opt: &PaddingOptions) -> Vec<Tensor> {
    let mut arr = tangent.inner().clone();
    for &ax in &opt.axes {
        let axis = norm_axis_index(ax, a.ndim());
        let si = a.shape()[axis];
        let left = opt.config.left;
        let step = opt.config.interior + 1;
        let end = if si == 0 {
            left
        } else {
            left + step * (si - 1) + 1
        };
        arr = arr
            .slice_axis(
                Axis(axis),
                Slice::new(left as isize, Some(end as isize), step as isize),
            )
            .to_owned();
    }
    vec![arr.into()]
}

fn vjp_pool(tangent: &Tensor, input: &Tensor, opt: &PoolOptions, average: bool) -> Vec<Tensor> {
    let mut d_input = ArrayD::zeros(IxDyn(input.shape()));
    let window_size = opt.window_size.iter().product::<usize>() as f64;

    for (out_idx, &t) in tangent.inner().indexed_iter() {
        let contrib = if average { t / window_size } else { t };
        for win_idx in indices(IxDyn(&opt.window_size)) {
            let in_idx: Vec<usize> = (0..input.ndim())
                .map(|ax| out_idx[ax] * opt.stride[ax] + win_idx[ax])
                .collect();
            d_input[IxDyn(&in_idx)] += contrib;
        }
    }

    vec![d_input.into()]
}

fn softmax_xent_vjp(logits: &Tensor, labels: &Tensor) -> Result<Tensor, EvalError> {
    let n = logits.shape()[0] as f64;
    let logits_arr = logits.inner();
    let labels_arr = labels.inner();

    let max = logits_arr.map_axis(Axis(1), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    let shifted = logits_arr - &max.insert_axis(Axis(1));
    let exp = shifted.mapv(f64::exp);
    let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
    let probs = exp / sum;

    Ok(((probs - labels_arr) / n).into())
}

impl GradInterpreter {
    pub fn new() -> Self {
        GradInterpreter
    }

    fn process_primitive(
        primitive: &Primitive,
        outvar: &Atom,
        primals: &Env<Tensor>,
        env: &Env<Tensor>,
    ) -> Result<Vec<Tensor>, EvalError> {
        let p = |a: &Atom| primals.resolve(a);

        let tangent = env.resolve(outvar)?;

        use Primitive::*;
        Ok(match primitive {
            // elementwise unary
            Neg(_) => vec![-tangent],
            Reciprocal(a) => {
                let d = p(a)?.mapv(|x| -1.0 / (x * x));
                vec![d * tangent]
            }
            Square(a) => vec![tangent * 2.0 * p(a)?],
            Sqrt(_) => vec![tangent / (p(outvar)? * 2.0)],
            Exp(_) => vec![tangent * p(outvar)?],
            Log(a) => vec![tangent / p(a)?],
            // elementwise binary (numpy broadcasting)
            Add(_, _) => vec![tangent.clone(), tangent],
            Mul(a, b) => vec![tangent.clone() * p(b)?, p(a)? * tangent],
            Where(c, _, _) => vjp_where(&tangent, &p(c)?)?,
            // activations
            Relu(_) => {
                let zero: Tensor = ArrayD::zeros(IxDyn(tangent.shape())).into();
                vec![Tensor::r#where(&p(outvar)?, &tangent, &zero)?]
            }
            LeakyRelu { operand, slope } => {
                let x = p(operand)?;
                let steep = tangent.clone() * *slope;
                vec![Tensor::r#where(&x.mapv(|v| v.max(0.0)), &tangent, &steep)?]
            }
            Elu { operand, slope } => {
                let val = p(operand)?;
                let d_neg = tangent.clone() * *slope * val.exp();
                vec![Tensor::r#where(&val, &tangent, &d_neg)?]
            }
            Gelu(a) => {
                let val = p(a)?;
                let pdf = val.mapv(|x| (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt());
                vec![tangent * val.normcdf() + val * pdf]
            }
            NormalCdf(a) => {
                let pdf =
                    p(a)?.mapv(|x| (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt());
                vec![tangent * pdf]
            }
            // linear algebra
            Dot(a, b) => return vjp_dot(&tangent, &p(a)?, &p(b)?),
            // reduction
            ReduceSum { operand, axes } => {
                let x = p(operand)?;
                let expanded = tangent.expand_dims(axes);
                let arr = expanded
                    .inner()
                    .broadcast(IxDyn(x.shape()))
                    .ok_or_else(|| EvalError::Eval("reduce_sum vjp: broadcast failed".to_string()))?
                    .to_owned();
                vec![arr.into()]
            }
            // shape manipulation
            ExpandDims { operand: _, axes } => {
                let shape = tangent.shape().to_vec();
                let arr = tangent.into_inner();
                vec![EvalInterpreter::process_primitive(
                    &Primitive::ReduceSum {
                        operand: Atom {
                            name: String::new(),
                            shape,
                            kind: AtomKind::Const(arr),
                        },
                        axes: axes.clone(),
                    },
                    env,
                )?]
            }
            MoveAxis {
                operand: _,
                source,
                destination,
            } => {
                let shape = tangent.shape().to_vec();
                let arr = tangent.into_inner();
                vec![EvalInterpreter::process_primitive(
                    &Primitive::MoveAxis {
                        operand: Atom {
                            name: String::new(),
                            shape,
                            kind: AtomKind::Const(arr),
                        },
                        source: *destination,
                        destination: *source,
                    },
                    env,
                )?]
            }
            Reshape {
                operand,
                new_shape: _,
            } => {
                let orig_shape: Vec<isize> =
                    p(operand)?.shape().iter().map(|&x| x as isize).collect();
                let shape = tangent.shape().to_vec();
                let arr = tangent.into_inner();
                vec![EvalInterpreter::process_primitive(
                    &Primitive::Reshape {
                        operand: Atom {
                            name: String::new(),
                            shape,
                            kind: AtomKind::Const(arr),
                        },
                        new_shape: orig_shape,
                    },
                    env,
                )?]
            }
            // padding
            Pad { operand, options } => vjp_pad(&tangent, &p(operand)?, options),
            // 2d convolution
            Conv {
                input,
                kernel,
                options,
            } => return vjp_conv(&tangent, &p(input)?, &p(kernel)?, options.stride),
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
            if let Some(t) = env.get(&var.name) {
                env.update(&var.name, t.clone() + unbroadcast(tangent, &var.shape));
            } else {
                env.insert(var.name.clone(), unbroadcast(tangent, &var.shape));
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

    fn carr(data: &[f64], shape: &[usize]) -> ArrayD<f64> {
        ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap()
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
    fn run_vjp(
        primitive: Primitive,
        outvar_primal: ArrayD<f64>,
        tangent: ArrayD<f64>,
    ) -> Vec<Tensor> {
        let outvar = Atom {
            name: "out".to_string(),
            shape: tangent.shape().to_vec(),
            kind: AtomKind::Var,
        };
        let mut primals: Env<Tensor> = Env::new();
        primals.insert("out".to_string(), outvar_primal.into());
        let mut env: Env<Tensor> = Env::new();
        env.insert("out".to_string(), tangent.into());
        GradInterpreter::process_primitive(&primitive, &outvar, &primals, &env).unwrap()
    }

    fn ones(shape: &[usize]) -> ArrayD<f64> {
        ArrayD::ones(IxDyn(shape))
    }
    fn zeros(shape: &[usize]) -> ArrayD<f64> {
        ArrayD::zeros(IxDyn(shape))
    }

    fn assert_close(a: &ArrayD<f64>, b: &ArrayD<f64>) {
        assert_eq!(a.shape(), b.shape(), "shapes differ");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-9, "values differ: {x} vs {y}");
        }
    }

    fn assert_close_tol(a: &ArrayD<f64>, b: &ArrayD<f64>, tol: f64) {
        assert_eq!(a.shape(), b.shape(), "shapes differ");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < tol, "values differ: {x} vs {y} (tol {tol})");
        }
    }

    /// Numerical VJP via central differences.
    fn num_vjp(
        x: &ArrayD<f64>,
        tangent: &ArrayD<f64>,
        f: impl Fn(&ArrayD<f64>) -> ArrayD<f64>,
    ) -> ArrayD<f64> {
        let eps = 1e-5;
        let x_flat: Vec<f64> = x.iter().copied().collect();
        let grad_flat: Vec<f64> = (0..x_flat.len())
            .map(|i| {
                let mut xp = x_flat.clone();
                let mut xm = x_flat.clone();
                xp[i] += eps;
                xm[i] -= eps;
                let fp = f(&ArrayD::from_shape_vec(x.raw_dim(), xp).unwrap());
                let fm = f(&ArrayD::from_shape_vec(x.raw_dim(), xm).unwrap());
                ((&fp - &fm) / (2.0 * eps) * tangent).sum()
            })
            .collect();
        ArrayD::from_shape_vec(x.raw_dim(), grad_flat).unwrap()
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
        assert_close(g[0].inner(), &t.mapv(|x| -x));
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
        assert_close_tol(
            g[0].inner(),
            &num_vjp(&x, &t, |a| a.mapv(|v| 1.0 / v)),
            1e-9,
        );
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
        assert_close_tol(g[0].inner(), &num_vjp(&x, &t, |a| a.mapv(|v| v * v)), 1e-9);
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
        assert_close_tol(g[0].inner(), &num_vjp(&x, &t, |a| a.mapv(f64::sqrt)), 1e-9);
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
        assert_close_tol(g[0].inner(), &num_vjp(&x, &t, |a| a.mapv(f64::exp)), 1e-9);
    }

    #[test]
    fn log() {
        let xd = &[1.0, 2.0, 4.0];
        let x = carr(xd, &[3]);
        let t = carr(&[2.0, 1.0, 3.0], &[3]);
        let g = run_vjp(Primitive::Log(const_atom(xd, &[3])), zeros(&[3]), t.clone());
        assert_close_tol(g[0].inner(), &num_vjp(&x, &t, |a| a.mapv(f64::ln)), 1e-9);
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
        assert_close(g[0].inner(), &t);
        assert_close(g[1].inner(), &t);
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
        assert_close(g[0].inner(), &(&t * &b));
        assert_close(g[1].inner(), &(&a * &t));
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
        assert_close(g[0].inner(), &zeros(&[2])); // d_cond = 0
        assert_close(g[1].inner(), &t); // d_x = t
        assert_close(g[2].inner(), &zeros(&[2])); // d_y = 0
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
        assert_close(g[0].inner(), &zeros(&[2]));
        assert_close(g[1].inner(), &zeros(&[2])); // d_x = 0
        assert_close(g[2].inner(), &t); // d_y = t
    }

    // ---- activations ----

    #[test]
    fn relu() {
        let g = run_vjp(
            Primitive::Relu(const_atom(&[2.0, -1.0, 0.0], &[3])),
            carr(&[2.0, 0.0, 0.0], &[3]), // outvar_primal = relu(x)
            ones(&[3]),
        );
        assert_close(g[0].inner(), &carr(&[1.0, 0.0, 0.0], &[3]));
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
        assert_close_tol(g[0].inner(), &carr(&[1.0, slope], &[2]), 1e-10);
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
        assert_close_tol(
            g[0].inner(),
            &num_vjp(&x, &t, |a| a.mapv(|x| normcdf(x))),
            1e-8,
        );
    }

    // ---- linear algebra ----

    #[test]
    fn dot_1d_1d() {
        let t = ArrayD::from_elem(IxDyn(&[]), 1.0);
        let g = run_vjp(
            Primitive::Dot(const_atom(&[1.0, 2.0], &[2]), const_atom(&[3.0, 4.0], &[2])),
            ArrayD::from_elem(IxDyn(&[]), 11.0),
            t,
        );
        assert_close(g[0].inner(), &carr(&[3.0, 4.0], &[2]));
        assert_close(g[1].inner(), &carr(&[1.0, 2.0], &[2]));
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
        assert_close(g[0].inner(), &carr(&[11.0, 15.0, 11.0, 15.0], &[2, 2]));
        assert_close(g[1].inner(), &carr(&[4.0, 4.0, 6.0, 6.0], &[2, 2]));
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
        assert_close(g[0].inner(), &carr(&[1.0, 1.0, 1.0, 1.0], &[2, 2]));
        assert_close(g[1].inner(), &carr(&[4.0, 6.0], &[2]));
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
        assert_close(
            g[0].inner(),
            &carr(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 3]),
        );
        assert_close(g[1].inner(), &carr(&[5.0, 7.0, 9.0], &[3]));
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
        assert_close(
            g[0].inner(),
            &carr(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3]),
        );
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
        assert_close(g[0].inner(), &carr(&expected, &[3, 4]));
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
        assert_close(g[0].inner(), &carr(&[2.0, 3.0, 4.0], &[3]));
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
        assert_close(g[0].inner(), &carr(&[1.0, 2.0, 3.0], &[3]));
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
        assert_close(
            g[0].inner(),
            &carr(&[1.0, 3.0, 5.0, 2.0, 4.0, 6.0], &[2, 3]),
        );
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
        assert_close(g[0].inner(), &carr(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[6]));
    }
}
