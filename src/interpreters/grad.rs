use ndarray::{ArrayD, Axis, IxDyn, Slice, indices};

use crate::{
    interpreters::{
        EvalError, EvalInterpreter, Interpreter,
        eval_util::{Tensor, dot, expand_dims, moveaxis, norm_axis_index, where_},
    },
    mininn::{Atom, AtomKind, ComputeGraph, Env, PaddingOptions, PoolOptions, Primitive},
};

fn unbroadcast(mut g: Tensor, target: &[usize]) -> Tensor {
    while g.ndim() > target.len() {
        g = g.sum_axis(Axis(0));
    }
    for i in 0..target.len() {
        if target[i] == 1 && g.shape()[i] > 1 {
            g = g.sum_axis(Axis(i)).insert_axis(Axis(i));
        }
    }
    g
}

fn vjp_where(tangent: &Tensor, condition: &Tensor) -> Result<Vec<Tensor>, EvalError> {
    let zero = ArrayD::zeros(condition.shape());

    let first = where_(condition, tangent, &zero)?;
    let second = where_(condition, &zero, tangent)?;

    Ok(vec![zero, first, second])
}

fn vjp_dot(tangent: &Tensor, a: &Tensor, b: &Tensor) -> Result<Vec<Tensor>, EvalError> {
    let dx = if b.ndim() == 0 {
        tangent * b
    } else if b.ndim() == 1 {
        dot(&expand_dims(tangent, &[-1]), &expand_dims(b, &[0]))?
    } else {
        dot(tangent, &moveaxis(b, -1, -2))?
    };

    let dy = if a.ndim() == 0 {
        a * tangent
    } else if a.ndim() == 1 {
        dot(&expand_dims(a, &[-1]), &expand_dims(tangent, &[0]))?
    } else {
        dot(&moveaxis(a, -1, -2), tangent)?
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
    // Notes from python version
    // -------------------------
    // The jacobian consists of two parts -- gradient w.r.t. the kernel and gradient w.r.t. the input matrix
    // Because convolution is a sum of products, each gradient is going to look like a sum of gradients

    // wrt kernel: (rough sketch)
    // (x[n, c', stride * h + i, stride * w + j] * K[c, c', i, j])' = x[n, c', stride * h + i, stride * w + j] => t * (sum over these x) => sum over t * these x
    //
    // wrt input: (rough sketch)
    // (x[n, c', stride * h + i, stride * w + j] * K[c, c', i, j])' = K[c, c', i, j] => t * (sum over these K) => sum over t * these K

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

    let mut d_input = ArrayD::zeros(IxDyn(&[n, c, h, w]));
    let mut d_kernel = ArrayD::zeros(kernel.raw_dim());

    for ni in 0..n {
        for ci in 0..ko {
            for hi in 0..oh {
                for wi in 0..ow {
                    let t = tangent[[ni, ci, hi, wi]];
                    for cpi in 0..c {
                        for i in 0..kh {
                            for j in 0..kw {
                                // essentially like mul
                                d_input[[ni, cpi, s * hi + i, s * wi + j]] +=
                                    t * kernel[[ci, cpi, i, j]];
                                d_kernel[[ci, cpi, i, j]] +=
                                    t * input[[ni, cpi, s * hi + i, s * wi + j]];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(vec![d_input, d_kernel])
}

fn vjp_pad(tangent: &Tensor, a: &Tensor, opt: &PaddingOptions) -> Vec<Tensor> {
    let mut grad = tangent.clone();
    for &ax in &opt.axes {
        let axis = norm_axis_index(ax, a.ndim());
        let si = a.shape()[axis];
        let left = opt.config.left;
        let step = opt.config.interior + 1;
        grad = grad
            .slice_axis(
                Axis(axis),
                Slice::new(
                    left as isize,
                    Some((left + step * si) as isize),
                    step as isize,
                ),
            )
            .to_owned();
    }
    vec![grad]
}

fn vjp_pool(tangent: &Tensor, input: &Tensor, opt: &PoolOptions, average: bool) -> Vec<Tensor> {
    // sum of products
    let mut d_input = ArrayD::zeros(input.raw_dim());
    let window_size = opt.window_size.iter().product::<usize>() as f64;

    for (out_idx, &t) in tangent.indexed_iter() {
        let contrib = if average { t / window_size } else { t };
        for win_idx in indices(IxDyn(&opt.window_size)) {
            let in_idx: Vec<usize> = (0..input.ndim())
                .map(|ax| out_idx[ax] * opt.stride[ax] + win_idx[ax])
                .collect();
            d_input[IxDyn(&in_idx)] += contrib;
        }
    }

    vec![d_input]
}

pub struct GradInterpreter;

impl GradInterpreter {
    pub fn new() -> Self {
        GradInterpreter
    }

    fn process_primitive(
        primitive: &Primitive,
        outvar: &Atom,
        primals: &Env<f64>,
        env: &Env<f64>,
    ) -> Result<Vec<Tensor>, EvalError> {
        let p = |a: &Atom| primals.resolve(a);

        let tangent = env.resolve(outvar)?;

        use Primitive::*;
        Ok(match primitive {
            // elementwise unary
            Neg(_) => vec![-tangent],
            Reciprocal(a) => vec![-p(a)?.mapv(|x| x * x).recip() * tangent],
            Square(a) => vec![tangent * 2.0 * p(a)?],
            Sqrt(_) => {
                vec![tangent / (2.0 * p(outvar)?)]
            }
            Exp(_) => vec![tangent * p(outvar)?],
            Log(a) => vec![tangent / p(a)?],
            // elementwise binary (numpy broadcasting)
            Add(_, _) => vec![tangent.clone(), tangent],
            Mul(a, b) => vec![tangent.clone() * p(b)?, p(a)? * tangent],
            Where(c, _, _) => vjp_where(&tangent, &p(c)?)?,
            // activations
            Relu(_) => vec![where_(
                &p(outvar)?,
                &tangent,
                &ArrayD::zeros(tangent.shape()),
            )?],
            LeakyRelu { operand, slope } => vec![where_(
                &p(operand)?.mapv(|x| x.max(0.0)),
                &tangent,
                &(tangent.clone() * *slope),
            )?],
            NormalCdf(a) => vec![
                tangent
                    * p(a)?.mapv(|x| (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()),
            ],
            // linear algebra
            Dot(a, b) => return vjp_dot(&tangent, &p(a)?, &p(b)?),
            // reduction
            ReduceSum { operand, axes } => {
                let x = p(operand)?;
                let expanded = expand_dims(&tangent, axes);

                let grad = expanded
                    .broadcast(x.raw_dim())
                    .ok_or_else(|| EvalError::Eval("reduce_sum vjp: broadcast failed".to_string()))?
                    .to_owned();
                vec![grad]
            }
            // shape manipulation
            ExpandDims { operand: _, axes } => vec![EvalInterpreter::process_primitive(
                &Primitive::ReduceSum {
                    operand: Atom {
                        name: String::new(),
                        shape: tangent.shape().to_vec(),
                        kind: AtomKind::Const(tangent),
                    },
                    axes: axes.clone(),
                },
                env,
            )?],
            MoveAxis {
                operand: _,
                source,
                destination,
            } => vec![EvalInterpreter::process_primitive(
                &Primitive::MoveAxis {
                    operand: Atom {
                        name: String::new(),
                        shape: tangent.shape().to_vec(),
                        kind: AtomKind::Const(tangent),
                    },
                    source: destination.clone(),
                    destination: source.clone(),
                },
                env,
            )?],
            Reshape {
                operand,
                new_shape: _,
            } => vec![EvalInterpreter::process_primitive(
                &Primitive::Reshape {
                    operand: Atom {
                        name: String::new(),
                        shape: tangent.shape().to_vec(),
                        kind: AtomKind::Const(tangent),
                    },
                    new_shape: p(operand)?.shape().iter().map(|x| *x as isize).collect(),
                },
                env,
            )?],
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
}

impl Interpreter<f64> for GradInterpreter {
    /// Evaluate `graph` on `inputs` (one flat buffer per input var, in graph
    /// order) and return the output tensors flattened in row-major order.
    fn run(graph: &ComputeGraph, inputs: &Vec<Tensor>) -> Result<Vec<Tensor>, EvalError> {
        // ---- FORWARD (primals) ----
        let mut primals = Env::<f64>::new();

        for (var, tensor) in graph.invars.iter().zip(inputs) {
            primals.insert(var.name.clone(), tensor.clone());
        }

        for eqn in &graph.equations {
            let out = EvalInterpreter::process_primitive(&eqn.primitive, &primals)?;
            primals.insert(eqn.outvar.name.clone(), out);
        }

        // ---- BACKWARD ----
        let mut env = Env::<f64>::new();

        fn combine(var: &Atom, tangent: Tensor, env: &mut Env<f64>) {
            if let Some(t) = env.get(&var.name) {
                env.update(&var.name, t + unbroadcast(tangent, &var.shape));
            } else {
                env.insert(var.name.clone(), unbroadcast(tangent, &var.shape));
            }
        }

        for var in &graph.outvars {
            match var.kind {
                AtomKind::Var => {
                    env.insert(var.name.clone(), ArrayD::ones(IxDyn(&var.shape)));
                }
                AtomKind::Const(_) => continue,
            }
        }

        for eqn in graph.equations.iter().rev() {
            let out = Self::process_primitive(&eqn.primitive, &eqn.outvar, &primals, &env)?;

            for (atom, tangent) in eqn.primitive.operands().into_iter().zip(out) {
                combine(atom, tangent, &mut env);
            }
        }

        graph
            .invars
            .iter()
            .map(|var| {
                let tangent = env.get(&var.name).ok_or_else(|| {
                    EvalError::Eval(format!("output '{}' was never computed", var.name))
                })?;
                Ok(tangent.clone())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        interpreters::eval_util::normcdf,
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
        let mut primals = Env::new();
        primals.insert("out".to_string(), outvar_primal);
        let mut env = Env::new();
        env.insert("out".to_string(), tangent);
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

    /// Numerical VJP via central differences: perturb each input element, measure
    /// the resulting output change, dot with tangent.
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
        // mix of positive, negative, zero inputs
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
        assert_close_tol(&g[0], &num_vjp(&x, &t, |a| normcdf(a)), 1e-8);
    }

    // ---- linear algebra ----

    #[test]
    fn dot_1d_1d() {
        // a·b scalar, tangent is scalar 1
        let t = ArrayD::from_elem(IxDyn(&[]), 1.0);
        let g = run_vjp(
            Primitive::Dot(const_atom(&[1.0, 2.0], &[2]), const_atom(&[3.0, 4.0], &[2])),
            ArrayD::from_elem(IxDyn(&[]), 11.0), // 1*3+2*4
            t,
        );
        assert_close(&g[0], &carr(&[3.0, 4.0], &[2])); // dx = t*b
        assert_close(&g[1], &carr(&[1.0, 2.0], &[2])); // dy = t*a
    }

    #[test]
    fn dot_2d_2d() {
        // a:(2,2) @ b:(2,2) = c:(2,2), tangent = ones(2,2)
        // dx = t @ b.T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        // dy = a.T @ t = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
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
        // a:(2,2) @ b:(2,) = c:(2,), tangent = ones(2)
        // dx = outer(t, b) = [[1,1],[1,1]]
        // dy = a.T @ t = [[1,3],[2,4]] @ [1,1] = [4,6]
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
        // a:(2,3) @ b:(3,) = c:(2,), tangent = ones(2)
        // dx = outer(t, b) = [[1,0,0],[1,0,0]]
        // dy = a.T @ t = [[1,4],[2,5],[3,6]] @ [1,1] = [5,7,9]
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
        // x:(2,3) summed over axis 1 → (2,), tangent (2,) broadcast back to (2,3)
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
        // x:(3,4) summed over axis -1 → (3,), tangent (3,) broadcast back to (3,4)
        let t = carr(&[1.0, 2.0, 3.0], &[3]);
        let g = run_vjp(
            Primitive::ReduceSum {
                operand: const_atom(&[0.0; 12], &[3, 4]),
                axes: vec![-1],
            },
            zeros(&[3]),
            t,
        );
        // each row of g[0] is the corresponding tangent value broadcast across 4 columns
        let expected: Vec<f64> = [1.0, 2.0, 3.0].iter().flat_map(|&v| vec![v; 4]).collect();
        assert_close(&g[0], &carr(&expected, &[3, 4]));
    }

    // ---- shape manipulation ----

    #[test]
    fn expand_dims_front() {
        // x:(3,) → (1,3), tangent sum over axis 0 → (3,)
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
        // x:(3,) → (3,1), tangent sum over axis -1 → (3,)
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
        // x:(2,3) → MoveAxis(src=0,dst=-1) → (3,2), VJP swaps: MoveAxis(t, src=-1, dst=0)
        // t:(3,2) = [[1,2],[3,4],[5,6]] → (2,3) = [[1,3,5],[2,4,6]]
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
        // x:(6,) → Reshape([2,3]) → (2,3), VJP reshapes tangent back to (6,)
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
