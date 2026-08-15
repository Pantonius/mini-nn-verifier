use ndarray::{ArrayD, Axis, Ix0, Ix1, Ix2, IxDyn, Zip, arr0, indices, linalg::Dot};

use crate::{
    interpreters::Interpreter,
    mininn::{
        Atom, AtomKind, ComputeGraph, Env, MinninError, PaddingOptions, PoolOptions, Primitive,
        Value,
    },
};

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error(transparent)]
    Mininn(#[from] MinninError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("eval error: {0}")]
    Eval(String),
}

/// A concrete tensor value in the `eval` interpreter.
type Tensor = ArrayD<f64>;
impl Value for Tensor {}

/// Handles negative python style axis index and converts it into an absolute axis index
fn norm_axis_index(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        (axis + ndim as isize) as usize
    } else {
        axis as usize
    }
}

/// numpy broadcasting: align shapes from the right, expanding size-1 dims.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    // exand to the larger of the two lengths
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];

    for i in 0..n {
        let ad = if i + a.len() < n {
            1
        } else {
            a[i + a.len() - n]
        };
        let bd = if i + b.len() < n {
            1
        } else {
            b[i + b.len() - n]
        };

        if ad == bd || ad == 1 || bd == 1 {
            // take the larger of the two dimensions
            out[i] = ad.max(bd);
        } else {
            // cannot reconcile
            return None;
        }
    }
    Some(out)
}

/// Elementwise binary op with numpy broadcasting.
/// Given two tensors and a binary function (f64, f64) -> f64 returns a tensor on successful
/// application
fn binary(a: &Tensor, b: &Tensor, f: impl Fn(f64, f64) -> f64) -> Result<Tensor, EvalError> {
    // try to compute the broadcast shape
    let shape = broadcast_shape(a.shape(), b.shape()).ok_or_else(|| {
        EvalError::Eval(format!(
            "incompatible shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        ))
    })?;

    // broadcast each to the computed shape
    let av = a
        .broadcast(IxDyn(&shape))
        .ok_or_else(|| EvalError::Eval("broadcast failed".to_string()))?;

    let bv = b
        .broadcast(IxDyn(&shape))
        .ok_or_else(|| EvalError::Eval("broadcast failed".to_string()))?;

    // zip and map elementwise via given binary function
    // NOTE panic is impossible since av and bv are of same shape
    Ok(Zip::from(&av).and(&bv).map_collect(|&x, &y| f(x, y)))
}

/// numpy.where(cond, x, y) with broadcasting; `cond` is truthy when non-zero.
fn where_(cond: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor, EvalError> {
    // compute broadcasted shape for cond and x, then for that s and y
    let s = broadcast_shape(cond.shape(), x.shape())
        .and_then(|s| broadcast_shape(&s, y.shape()))
        .ok_or_else(|| EvalError::Eval("where: incompatible shapes".to_string()))?;

    // then actually broadcast each to that shape
    let cv = cond
        .broadcast(IxDyn(&s))
        .ok_or_else(|| EvalError::Eval("broadcast failed".to_string()))?;

    let xv = x
        .broadcast(IxDyn(&s))
        .ok_or_else(|| EvalError::Eval("broadcast failed".to_string()))?;

    let yv = y
        .broadcast(IxDyn(&s))
        .ok_or_else(|| EvalError::Eval("broadcast failed".to_string()))?;

    // zip cv, xv, yv into a triple and apply the condition logic elementwise
    // NOTE panic is impossible since cv, xv, yv are of same shape
    Ok(Zip::from(&cv)
        .and(&xv)
        .and(&yv)
        .map_collect(|&c, &x, &y| if c != 0.0 { x } else { y }))
}

/// Reshape into `shape` in C (row-major) order.
/// NOTE on terminology:
/// - C stands for contigous, mathematically speaking row-major order
/// - F stands for fortran-contigous, mathematically speaking column-major order
fn reshape_c(a: &Tensor, shape: &[usize]) -> Tensor {
    let data: Vec<f64> = a.iter().copied().collect();
    ArrayD::from_shape_vec(IxDyn(shape), data).expect("reshape element count mismatch")
}

/// numpy.moveaxis: move `src` axis to `dst`, keeping the relative order of the rest.
fn moveaxis(a: &Tensor, src: isize, dst: isize) -> Tensor {
    let nd = a.ndim(); // number of dimensions

    // proper indices
    let (s, d) = (norm_axis_index(src, nd), norm_axis_index(dst, nd));

    // all indices except the one being moved (s)
    let mut order: Vec<usize> = (0..nd).filter(|&x| x != s).collect();

    // insert s at d
    order.insert(d, s);

    // materialize
    a.view()
        .permuted_axes(order)
        .as_standard_layout()
        .to_owned()
}

/// numpy.dot: contract the last axis of `a` with the second-to-last of `b`
/// (or the only axis of `b` when it is 1-D).
/// numpy.org/doc/stable/reference/generated/numpy.dot.html
fn dot(a: &Tensor, b: &Tensor) -> Result<Tensor, EvalError> {
    // shapes of each input
    let (ash, bsh) = (a.shape().to_vec(), b.shape().to_vec());

    // If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
    if ash.is_empty() {
        let a2 = a
            .clone()
            .into_dimensionality::<Ix0>()
            .expect("dot lhs not 0-D")
            .into_scalar();

        return Ok(a2 * b);
    } else if bsh.is_empty() {
        let b2 = b
            .clone()
            .into_dimensionality::<Ix0>()
            .expect("dot rhs not 0-D")
            .into_scalar();

        return Ok(b2 * a);
    }

    if (ash.len() == 1 && bsh.len() == 1) || (ash.len() == 2 && bsh.len() <= 2) {
        // If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
        // If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.

        // that we can delegate to the implement dot product
        if ash.len() == 1 {
            if ash[0] != bsh[0] {
                return Err(EvalError::Eval(format!(
                    "dot: {ash:?} · {bsh:?} axis mismatch"
                )));
            }

            let a2 = a
                .clone()
                .into_dimensionality::<Ix1>()
                .expect("dot lhs not 1-D");

            let b2 = b
                .clone()
                .into_dimensionality::<Ix1>()
                .expect("dot rhs not 1-D");

            return Ok(arr0(a2.dot(&b2)).into_dyn());
        } else {
            if ash[1] != bsh[0] {
                return Err(EvalError::Eval(format!(
                    "dot: {ash:?} · {bsh:?} axis mismatch"
                )));
            }

            let a2 = a
                .clone()
                .into_dimensionality::<Ix2>()
                .expect("dot lhs not 2-D");

            if bsh.len() == 1 {
                let b2 = b
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .expect("dot rhs not 1-D");

                return Ok(a2.dot(&b2).into_dyn());
            } else {
                let b2 = b
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .expect("dot rhs not 2-D");

                return Ok(a2.dot(&b2).into_dyn());
            };
        }
    }

    let k = ash[ash.len() - 1];
    let a_prelim_shape = [ash[..ash.len() - 1].iter().product(), k];
    let a2 = reshape_c(a, &a_prelim_shape);

    // If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
    if bsh.len() == 1 {
        if bsh[0] != k {
            return Err(EvalError::Eval(format!(
                "dot: {ash:?} · {bsh:?} axis mismatch"
            )));
        }

        let prelim_res = a2.dot(b);
        return Ok(reshape_c(&prelim_res, &ash[..ash.len() - 1]));
    }

    // If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b:
    if bsh[bsh.len() - 2] != k {
        return Err(EvalError::Eval(format!(
            "dot: {ash:?} · {bsh:?} axis mismatch"
        )));
    }

    let b_moved = moveaxis(b, -2, 0); // move
    let b_prelim_shape = [k, b_moved.shape()[1..].iter().product()]; // flatten
    let b2 = reshape_c(&b_moved, &b_prelim_shape); // apply

    let prelim_res = a2.dot(&b2); // flat result

    let mut out_shape = ash[..ash.len() - 1].to_vec();
    out_shape.extend_from_slice(&bsh[..bsh.len() - 2]);
    out_shape.push(bsh[bsh.len() - 1]);

    Ok(reshape_c(&prelim_res, &out_shape)) // unflattened result
}

/// Sum over the given axes (numpy default: axes are removed).
fn reduce_sum(a: &Tensor, axes: &[isize]) -> Tensor {
    // normalizes axes and then sort and iterate from behind such that we don't need to shift higher
    // axes when reducing lower axes
    let mut norm_axes: Vec<usize> = axes
        .iter()
        .map(|&ax| norm_axis_index(ax, a.ndim()))
        .collect();
    norm_axes.sort_unstable();
    norm_axes.dedup();

    let mut out = a.clone();
    for ax in norm_axes.into_iter().rev() {
        out = out.sum_axis(Axis(ax));
    }

    out
}

/// numpy.expand_dims: insert size-1 axes at the given positions (which refer to
/// the result's axes).
fn expand_dims(a: &Tensor, axes: &[isize]) -> Tensor {
    let mut norm_axes: Vec<usize> = axes
        .iter()
        .map(|&ax| norm_axis_index(ax, a.ndim() + axes.len()))
        .collect();
    norm_axes.sort_unstable();

    let mut out = a.clone();
    for pos in norm_axes {
        out = out.insert_axis(Axis(pos));
    }

    out
}

/// Reshape resolving a single inferred dimensions (specified by -1).
fn reshape(a: &Tensor, new_shape: &[isize]) -> Result<Tensor, EvalError> {
    let known_dims: usize = new_shape
        .iter()
        .filter(|&&d| d >= 0)
        .map(|&d| d as usize)
        .product();

    let shape: Vec<usize> = new_shape
        .iter()
        .map(|&d| {
            if d < 0 {
                // infer
                if known_dims == 0 {
                    0
                } else {
                    a.len() / known_dims
                }
            } else {
                d as usize
            }
        })
        .collect();

    if shape.iter().product::<usize>() != a.len() {
        return Err(EvalError::Eval(format!(
            "reshape {:?} -> {new_shape:?} changes element count",
            a.shape()
        )));
    }

    Ok(reshape_c(a, &shape))
}

/// jax.lax.pad: per listed axis, add `left`/`right` padding and `interior`
/// dilation between elements, filling with `value`.
fn pad(a: &Tensor, opt: &PaddingOptions) -> Tensor {
    let is_padded = |i: usize| {
        opt.axes
            .iter()
            .any(|&ax| norm_axis_index(ax, a.ndim()) == i)
    };

    let out_shape: Vec<usize> = (0..a.ndim())
        .map(|i| {
            let si = a.shape()[i];
            if is_padded(i) {
                opt.config.left + si + (si - 1) * opt.config.interior + opt.config.right
            } else {
                si
            }
        })
        .collect();

    let mut out = ArrayD::from_elem(IxDyn(&out_shape), opt.value);
    for (j, &val) in a.indexed_iter() {
        let dest: Vec<usize> = (0..a.ndim())
            .map(|a| {
                if is_padded(a) {
                    opt.config.left + (opt.config.interior + 1) * j[a]
                } else {
                    j[a]
                }
            })
            .collect();
        out[IxDyn(&dest)] = val;
    }

    out
}

/// 2-D cross-correlation (NCHW input, OIHW kernel), single stride for H and W.
fn conv(input: &Tensor, kernel: &Tensor, stride: isize) -> Result<Tensor, EvalError> {
    if input.ndim() != 4 || kernel.ndim() != 4 {
        return Err(EvalError::Eval(
            "conv expects 4-D input and kernel".to_string(),
        ));
    }

    let s = stride as usize;
    let (n, c, h, w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let (ko, kc, kh, kw) = (
        kernel.shape()[0],
        kernel.shape()[1],
        kernel.shape()[2],
        kernel.shape()[3],
    );

    if kc != c {
        return Err(EvalError::Eval(format!(
            "conv channel mismatch: input {c}, kernel {kc}"
        )));
    }

    let oh = (h - kh) / s + 1;
    let ow = (w - kw) / s + 1;

    let mut out = ArrayD::zeros(IxDyn(&[n, ko, oh, ow]));

    for ni in 0..n {
        for ci in 0..ko {
            for hi in 0..oh {
                for wi in 0..ow {
                    let mut agg = 0.0;

                    for cpi in 0..c {
                        for i in 0..kh {
                            for j in 0..kw {
                                agg += input[[ni, cpi, s * hi + i, s * wi + j]]
                                    * kernel[[ci, cpi, i, j]];
                            }
                        }
                    }

                    out[[ni, ci, hi, wi]] = agg;
                }
            }
        }
    }

    Ok(out)
}

/// Windowed sum/average pooling over every axis (per-axis window and stride).
fn pool(a: &Tensor, opt: &PoolOptions, average: bool) -> Result<Tensor, EvalError> {
    if opt.window_size.len() != a.ndim() || opt.stride.len() != a.ndim() {
        return Err(EvalError::Eval(
            "pool: window/stride rank must match input".to_string(),
        ));
    }

    let out_shape: Vec<usize> = (0..a.ndim())
        .map(|i| (a.shape()[i] - opt.window_size[i]) / opt.stride[i] + 1)
        .collect();

    let mut out = ArrayD::zeros(IxDyn(&out_shape));

    let window_total: usize = opt.window_size.iter().product();

    for (oidx, slot) in out.indexed_iter_mut() {
        let mut acc = 0.0;

        for widx in indices(IxDyn(&opt.window_size)) {
            let sidx: Vec<usize> = (0..a.ndim())
                .map(|ax| opt.stride[ax] * oidx[ax] + widx[ax])
                .collect();
            acc += a[IxDyn(&sidx)]
        }

        *slot = if average {
            acc / window_total as f64
        } else {
            acc
        }
    }
    Ok(out)
}

pub struct EvalInterpreter;

impl EvalInterpreter {
    pub fn new() -> Self {
        EvalInterpreter
    }

    fn resolve(atom: &Atom, env: &Env<Tensor>) -> Result<Tensor, EvalError> {
        match &atom.kind {
            AtomKind::Const(data) => ArrayD::from_shape_vec(IxDyn(&atom.shape), data.clone())
                .map_err(|e| EvalError::Eval(format!("const {}: {e}", atom.name))),
            AtomKind::Var => env
                .get(&atom.name)
                .cloned()
                .ok_or_else(|| EvalError::Eval(format!("undefined variable '{}'", atom.name))),
        }
    }

    /// Evaluate `graph` on `inputs` (one flat buffer per input var, in graph
    /// order) and return the output tensors flattened in row-major order.
    pub fn run(
        mut self,
        graph: &ComputeGraph,
        inputs: Vec<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, EvalError> {
        let mut env = Env::new();

        for (var, data) in graph.invars.iter().zip(inputs) {
            let tensor = ArrayD::from_shape_vec(IxDyn(&var.shape), data)
                .map_err(|e| EvalError::Eval(format!("input {}: {e}", var.name)))?;
            env.insert(var.name.clone(), tensor);
        }

        for eqn in &graph.equations {
            let out = self.process_primitive(eqn.primitive.clone(), &env)?;
            env.insert(eqn.outvar.name.clone(), out);
        }

        graph
            .outvars
            .iter()
            .map(|var| {
                let tensor = env.get(&var.name).ok_or_else(|| {
                    EvalError::Eval(format!("output '{}' was never computed", var.name))
                })?;
                Ok(tensor.iter().copied().collect())
            })
            .collect()
    }
}

impl Interpreter<Tensor> for EvalInterpreter {
    fn process_primitive(
        &mut self,
        primitive: Primitive,
        env: &Env<Tensor>,
    ) -> Result<Tensor, EvalError> {
        let r = |a: &Atom| Self::resolve(a, env);

        use Primitive::*;
        Ok(match primitive {
            // elementwise unary
            Neg(a) => r(&a)?.mapv(|x| -x),
            Reciprocal(a) => r(&a)?.recip(),
            Square(a) => r(&a)?.mapv(|x| x * x),
            Sqrt(a) => r(&a)?.sqrt(),
            Exp(a) => r(&a)?.exp(),
            Log(a) => r(&a)?.mapv(f64::ln),
            // elementwise binary (numpy broadcasting)
            Add(a, b) => binary(&r(&a)?, &r(&b)?, |x, y| x + y)?,
            Mul(a, b) => binary(&r(&a)?, &r(&b)?, |x, y| x * y)?,
            Where(c, x, y) => where_(&r(&c)?, &r(&x)?, &r(&y)?)?,
            // activations
            Relu(a) => r(&a)?.mapv(|x| x.max(0.0)),
            LeakyRelu { operand, slope } => {
                r(&operand)?.mapv(|x| if x >= 0.0 { x } else { slope * x })
            }
            NormalCdf(a) => r(&a)?.mapv(|x| 0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))),
            // linear algebra
            Dot(a, b) => dot(&r(&a)?, &r(&b)?)?,
            // reduction
            ReduceSum { operand, axes } => reduce_sum(&r(&operand)?, &axes),
            // shape manipulation
            ExpandDims { operand, axes } => expand_dims(&r(&operand)?, &axes),
            MoveAxis {
                operand,
                source,
                destination,
            } => moveaxis(&r(&operand)?, source, destination),
            Reshape { operand, new_shape } => reshape(&r(&operand)?, &new_shape)?,
            // padding
            Pad { operand, options } => pad(&r(&operand)?, &options),
            // 2d convolution
            Conv {
                input,
                kernel,
                options,
            } => conv(&r(&input)?, &r(&kernel)?, options.stride)?,
            // pooling
            AvgPool { operand, options } => pool(&r(&operand)?, &options, true)?,
            SumPool { operand, options } => pool(&r(&operand)?, &options, false)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::mininn::PaddingOptionConfig;

    use super::*;
    use ndarray::{array, s};

    // --- norm_axis_index tests ---

    #[test]
    fn norm_axis_positive() {
        assert_eq!(norm_axis_index(0, 4), 0);
        assert_eq!(norm_axis_index(3, 4), 3);
    }

    #[test]
    fn norm_axis_negative() {
        assert_eq!(norm_axis_index(-1, 4), 3);
        assert_eq!(norm_axis_index(-4, 4), 0);
    }

    // --- broadcast_shape tests ---

    #[test]
    fn broadcast_same_shape() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn broadcast_left_pad() {
        // shorter shape is left-padded with 1s
        assert_eq!(broadcast_shape(&[3], &[2, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn broadcast_size1_expansion() {
        assert_eq!(broadcast_shape(&[1, 3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[2, 1], &[1, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn broadcast_scalar() {
        assert_eq!(broadcast_shape(&[], &[2, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn broadcast_incompatible() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 4]), None);
    }

    // --- binary tests ---

    #[test]
    fn binary_add_elementwise() {
        let a = array![1.0, 2.0, 3.0].into_dyn();
        let b = array![4.0, 5.0, 6.0].into_dyn();
        let result = binary(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(result, array![5.0, 7.0, 9.0].into_dyn());
    }

    #[test]
    fn binary_broadcast() {
        // [2,1] + [3] → [2,3]
        let a = array![[1.0], [2.0]].into_dyn();
        let b = array![10.0, 20.0, 30.0].into_dyn();
        let result = binary(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 11.0);
        assert_eq!(result[[1, 2]], 32.0);
    }

    #[test]
    fn binary_incompatible_error() {
        let a = ArrayD::zeros(IxDyn(&[2]));
        let b = ArrayD::zeros(IxDyn(&[3]));
        assert!(binary(&a, &b, |x, y| x + y).is_err());
    }

    // --- where_ tests ---

    #[test]
    fn where_basic() {
        let cond = array![1.0, 0.0, 1.0].into_dyn();
        let x = array![10.0, 20.0, 30.0].into_dyn();
        let y = array![40.0, 50.0, 60.0].into_dyn();
        let result = where_(&cond, &x, &y).unwrap();
        assert_eq!(result, array![10.0, 50.0, 30.0].into_dyn());
    }

    #[test]
    fn where_broadcast_scalar_branches() {
        // scalar x and y broadcast to cond's shape
        let cond = array![1.0, 0.0].into_dyn();
        let x = arr0(5.0).into_dyn();
        let y = arr0(0.0).into_dyn();
        let result = where_(&cond, &x, &y).unwrap();
        assert_eq!(result, array![5.0, 0.0].into_dyn());
    }

    // --- reshape_c tests ---

    #[test]
    fn reshape_c_flatten() {
        let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let result = reshape_c(&a, &[4]);
        assert_eq!(result, array![1.0, 2.0, 3.0, 4.0].into_dyn());
    }

    #[test]
    fn reshape_c_unflatten() {
        let a = array![1.0, 2.0, 3.0, 4.0].into_dyn();
        let result = reshape_c(&a, &[2, 2]);
        assert_eq!(result, array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
    }

    // --- moveaxis tests ---

    #[test]
    fn moveaxis_first_to_last() {
        // [2,3,4]: move axis 0 to 2 → permutation [1,2,0] → shape [3,4,2]
        let a = ArrayD::zeros(IxDyn(&[2, 3, 4]));
        let result = moveaxis(&a, 0, 2);
        assert_eq!(result.shape(), &[3, 4, 2]);
    }

    #[test]
    fn moveaxis_last_to_first() {
        // [2,3,4]: move axis -1 to 0 → permutation [2,0,1] → shape [4,2,3]
        let a = ArrayD::zeros(IxDyn(&[2, 3, 4]));
        let result = moveaxis(&a, -1, 0);
        assert_eq!(result.shape(), &[4, 2, 3]);
    }

    #[test]
    fn moveaxis_preserves_values() {
        // verify element [0,1,2] ends up at [1,2,0] after moving axis 0 → 2
        let a = ArrayD::from_shape_fn(IxDyn(&[2, 3, 4]), |idx| {
            (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
        });
        let result = moveaxis(&a, 0, 2);
        // original [0,1,2] = 012.0 should now be at [1,2,0]
        assert_eq!(result[[1, 2, 0]], 12.0);
    }

    #[test]
    fn dot_1d_1d() {
        let a = array![1.0, 2.0, 3.0].into_dyn();
        let b = array![4.0, 5.0, 6.0].into_dyn();
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.ndim(), 0);
        assert_eq!(result[[]], 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn dot_2d_2d() {
        let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let b = array![[5.0, 6.0], [7.0, 8.0]].into_dyn();
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 19.0); // 1*5 + 2*7
        assert_eq!(result[[1, 1]], 50.0); // 3*6 + 4*8
    }

    #[test]
    fn dot_2d_1d() {
        let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let b = array![1.0, 1.0].into_dyn();
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result[[0]], 3.0);
        assert_eq!(result[[1]], 7.0);
    }

    #[test]
    fn dot_nd_1d() {
        // a: [2, 3, 4], b: [4] → result: [2, 3]
        let a = ArrayD::ones(IxDyn(&[2, 3, 4]));
        let b = ArrayD::from_elem(IxDyn(&[4]), 2.0);
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert!(result.iter().all(|&x| x == 8.0)); // sum of 4 * (1*2)
    }

    #[test]
    fn dot_nd_md() {
        // a: [2, 3], b: [3, 4] → result: [2, 4]
        let a = ArrayD::ones(IxDyn(&[2, 3]));
        let b = ArrayD::ones(IxDyn(&[3, 4]));
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 4]);
        assert!(result.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn dot_scalar() {
        let a = arr0(3.0).into_dyn();
        let b = ArrayD::from_elem(IxDyn(&[2, 3]), 2.0);
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert!(result.iter().all(|&x| x == 6.0));
    }

    #[test]
    fn dot_1d_1d_mismatch() {
        let a = ArrayD::ones(IxDyn(&[3]));
        let b = ArrayD::ones(IxDyn(&[4]));
        assert!(dot(&a, &b).is_err());
    }

    #[test]
    fn dot_2d_1d_mismatch() {
        let a = ArrayD::ones(IxDyn(&[2, 3]));
        let b = ArrayD::ones(IxDyn(&[4]));
        assert!(dot(&a, &b).is_err());
    }

    #[test]
    fn dot_2d_2d_mismatch() {
        let a = ArrayD::ones(IxDyn(&[2, 3]));
        let b = ArrayD::ones(IxDyn(&[4, 2]));
        assert!(dot(&a, &b).is_err());
    }

    #[test]
    fn dot_nd_1d_mismatch() {
        let a = ArrayD::ones(IxDyn(&[2, 3, 4]));
        let b = ArrayD::ones(IxDyn(&[5]));
        assert!(dot(&a, &b).is_err());
    }

    #[test]
    fn dot_nd_md_mismatch() {
        let a = ArrayD::ones(IxDyn(&[2, 3]));
        let b = ArrayD::ones(IxDyn(&[4, 5])); // second-to-last is 4, not 3
        assert!(dot(&a, &b).is_err());
    }

    #[test]
    fn reduce_sum_1d() {
        let a = array![1.0, 2.0, 3.0].into_dyn();
        let result = reduce_sum(&a, &[0]);
        assert_eq!(result.shape(), &[]);
        assert_eq!(result[[]], 6.0);
    }

    #[test]
    fn reduce_sum_3d_middle_axis() {
        let a = ArrayD::ones(IxDyn(&[2, 3, 4]));
        let result = reduce_sum(&a, &[1]);
        assert_eq!(result.shape(), &[2, 4]);
        assert!(result.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn reduce_sum_3d_multi_axes() {
        let a = ArrayD::ones(IxDyn(&[2, 3, 4]));
        let result = reduce_sum(&a, &[0, 2]);
        assert_eq!(result.shape(), &[3]);
        assert!(result.iter().all(|&x| x == 8.0)); // 2 * 4 * 1
    }

    #[test]
    fn expand_dims_prepend() {
        let a = ArrayD::ones(IxDyn(&[3, 4]));
        let result = expand_dims(&a, &[0]);
        assert_eq!(result.shape(), &[1, 3, 4]);
    }

    #[test]
    fn expand_dims_insert() {
        let a = ArrayD::ones(IxDyn(&[3, 4]));
        let result = expand_dims(&a, &[1]);
        assert_eq!(result.shape(), &[3, 1, 4]);
    }

    #[test]
    fn expand_dims_multi_axes_neg() {
        let a = ArrayD::ones(IxDyn(&[3, 4]));
        let result = expand_dims(&a, &[0, -1]);
        assert_eq!(result.shape(), &[1, 3, 4, 1]);
    }

    #[test]
    fn reshape_flatten() {
        let a = ArrayD::ones(IxDyn(&[2, 3]));
        assert!(reshape(&a, &[6]).is_ok())
    }

    #[test]
    fn reshape_unflatten() {
        let a = ArrayD::ones(IxDyn(&[6]));
        assert!(reshape(&a, &[2, -1]).is_ok())
    }

    #[test]
    fn reshape_mismatch() {
        let a = ArrayD::ones(IxDyn(&[6]));
        assert!(reshape(&a, &[2, 4]).is_err())
    }

    // --- conv tests ---

    #[test]
    fn conv_identity_kernel() {
        // 1x1 all-ones kernel is the identity for a single channel
        let input =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), (1..=9).map(|x| x as f64).collect())
                .unwrap();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0]).unwrap();
        let result = conv(&input, &kernel, 1).unwrap();
        assert_eq!(result.shape(), &[1, 1, 3, 3]);
        assert_eq!(result, input);
    }

    #[test]
    fn conv_single_channel_sum_kernel() {
        // 2x2 all-ones kernel produces the sliding 2x2 sum
        let input =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), (1..=9).map(|x| x as f64).collect())
                .unwrap();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0; 4]).unwrap();
        let result = conv(&input, &kernel, 1).unwrap();
        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        assert_eq!(result[[0, 0, 0, 0]], 12.0); // 1+2+4+5
        assert_eq!(result[[0, 0, 0, 1]], 16.0); // 2+3+5+6
        assert_eq!(result[[0, 0, 1, 0]], 24.0); // 4+5+7+8
        assert_eq!(result[[0, 0, 1, 1]], 28.0); // 5+6+8+9
    }

    #[test]
    fn conv_multi_output_channels() {
        // kernel shape [2,1,2,2]: two different output filters on one input channel
        let input =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), (1..=9).map(|x| x as f64).collect())
                .unwrap();
        // output channel 0: all ones (sum); output channel 1: top-left only
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[2, 1, 2, 2]),
            vec![
                1.0, 1.0, 1.0, 1.0, // oc=0
                1.0, 0.0, 0.0, 0.0, // oc=1
            ],
        )
        .unwrap();
        let result = conv(&input, &kernel, 1).unwrap();
        assert_eq!(result.shape(), &[1, 2, 2, 2]);
        // channel 0: sliding sum
        assert_eq!(result[[0, 0, 0, 0]], 12.0);
        assert_eq!(result[[0, 0, 0, 1]], 16.0);
        assert_eq!(result[[0, 0, 1, 0]], 24.0);
        assert_eq!(result[[0, 0, 1, 1]], 28.0);
        // channel 1: top-left element of each window
        assert_eq!(result[[0, 1, 0, 0]], 1.0);
        assert_eq!(result[[0, 1, 0, 1]], 2.0);
        assert_eq!(result[[0, 1, 1, 0]], 4.0);
        assert_eq!(result[[0, 1, 1, 1]], 5.0);
    }

    #[test]
    fn conv_multi_input_channels() {
        // kernel shape [1,2,2,2]: single output channel, sums over two input channels
        let ch0: Vec<f64> = (1..=9).map(|x| x as f64).collect(); // 1..9
        let ch1 = vec![1.0f64; 9]; // all ones
        let input_data: Vec<f64> = ch0.into_iter().chain(ch1).collect();
        let input = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3, 3]), input_data).unwrap();

        // ic=0: pick only top-left; ic=1: all ones (2x2 sum over the all-ones channel)
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 2, 2]),
            vec![
                1.0, 0.0, 0.0, 0.0, // ic=0
                1.0, 1.0, 1.0, 1.0, // ic=1
            ],
        )
        .unwrap();
        let result = conv(&input, &kernel, 1).unwrap();
        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        // [0,0]: ch0 contrib=1, ch1 contrib=4 → 5
        assert_eq!(result[[0, 0, 0, 0]], 5.0);
        // [0,1]: ch0=2, ch1=4 → 6
        assert_eq!(result[[0, 0, 0, 1]], 6.0);
        // [1,0]: ch0=4, ch1=4 → 8
        assert_eq!(result[[0, 0, 1, 0]], 8.0);
        // [1,1]: ch0=5, ch1=4 → 9
        assert_eq!(result[[0, 0, 1, 1]], 9.0);
    }

    #[test]
    fn conv_stride_2() {
        // stride=2 skips alternate positions
        let input =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 4, 4]), (1..=16).map(|x| x as f64).collect())
                .unwrap();
        // kernel picks only the top-left element of each window
        let kernel =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = conv(&input, &kernel, 2).unwrap();
        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        assert_eq!(result[[0, 0, 0, 0]], 1.0); // input[0,0,0,0]
        assert_eq!(result[[0, 0, 0, 1]], 3.0); // input[0,0,0,2]
        assert_eq!(result[[0, 0, 1, 0]], 9.0); // input[0,0,2,0]
        assert_eq!(result[[0, 0, 1, 1]], 11.0); // input[0,0,2,2]
    }

    #[test]
    fn conv_batch_size_2() {
        // two batch items are processed independently
        let input = ArrayD::from_shape_vec(
            IxDyn(&[2, 1, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![2.0]).unwrap();
        let result = conv(&input, &kernel, 1).unwrap();
        assert_eq!(result.shape(), &[2, 1, 2, 2]);
        assert_eq!(result[[0, 0, 0, 0]], 2.0);
        assert_eq!(result[[0, 0, 1, 1]], 8.0);
        assert_eq!(result[[1, 0, 0, 0]], 10.0);
        assert_eq!(result[[1, 0, 1, 1]], 16.0);
    }

    #[test]
    fn conv_channel_mismatch_error() {
        let input = ArrayD::zeros(IxDyn(&[1, 3, 4, 4])); // 3 input channels
        let kernel = ArrayD::zeros(IxDyn(&[2, 4, 2, 2])); // expects 4 input channels
        assert!(conv(&input, &kernel, 1).is_err());
    }

    #[test]
    fn conv_non_4d_input_error() {
        let input = ArrayD::zeros(IxDyn(&[1, 3, 4]));
        let kernel = ArrayD::zeros(IxDyn(&[2, 3, 2, 2]));
        assert!(conv(&input, &kernel, 1).is_err());
    }

    #[test]
    fn conv_non_4d_kernel_error() {
        let input = ArrayD::zeros(IxDyn(&[1, 3, 4, 4]));
        let kernel = ArrayD::zeros(IxDyn(&[2, 3, 2]));
        assert!(conv(&input, &kernel, 1).is_err());
    }

    // --- pool tests ---

    #[test]
    fn pool_1d_sum_stride1() {
        // [1,2,3,4] with window=2, stride=1 → [3,5,7]
        let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let opt = PoolOptions {
            window_size: vec![2],
            stride: vec![1],
        };
        let result = pool(&a, &opt, false).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result[[0]], 3.0);
        assert_eq!(result[[1]], 5.0);
        assert_eq!(result[[2]], 7.0);
    }

    #[test]
    fn pool_1d_avg_stride1() {
        // same windows, averaged
        let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let opt = PoolOptions {
            window_size: vec![2],
            stride: vec![1],
        };
        let result = pool(&a, &opt, true).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result[[0]], 1.5);
        assert_eq!(result[[1]], 2.5);
        assert_eq!(result[[2]], 3.5);
    }

    #[test]
    fn pool_1d_sum_stride2() {
        // [1,2,3,4] with window=2, stride=2 → [3,7]
        let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let opt = PoolOptions {
            window_size: vec![2],
            stride: vec![2],
        };
        let result = pool(&a, &opt, false).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result[[0]], 3.0); // 1+2
        assert_eq!(result[[1]], 7.0); // 3+4
    }

    #[test]
    fn pool_2d_sum_stride1() {
        // 3x3 input, 2x2 window, stride 1 → 2x2 output; same values as conv sum-kernel test
        let a =
            ArrayD::from_shape_vec(IxDyn(&[3, 3]), (1..=9).map(|x| x as f64).collect()).unwrap();
        let opt = PoolOptions {
            window_size: vec![2, 2],
            stride: vec![1, 1],
        };
        let result = pool(&a, &opt, false).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 12.0); // 1+2+4+5
        assert_eq!(result[[0, 1]], 16.0); // 2+3+5+6
        assert_eq!(result[[1, 0]], 24.0); // 4+5+7+8
        assert_eq!(result[[1, 1]], 28.0); // 5+6+8+9
    }

    #[test]
    fn pool_2d_avg_stride1() {
        let a =
            ArrayD::from_shape_vec(IxDyn(&[3, 3]), (1..=9).map(|x| x as f64).collect()).unwrap();
        let opt = PoolOptions {
            window_size: vec![2, 2],
            stride: vec![1, 1],
        };
        let result = pool(&a, &opt, true).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 3.0); // 12/4
        assert_eq!(result[[0, 1]], 4.0); // 16/4
        assert_eq!(result[[1, 0]], 6.0); // 24/4
        assert_eq!(result[[1, 1]], 7.0); // 28/4
    }

    #[test]
    fn pool_2d_sum_stride2() {
        // 4x4 input, 2x2 window, stride 2 → 2x2 output
        let a =
            ArrayD::from_shape_vec(IxDyn(&[4, 4]), (1..=16).map(|x| x as f64).collect()).unwrap();
        let opt = PoolOptions {
            window_size: vec![2, 2],
            stride: vec![2, 2],
        };
        let result = pool(&a, &opt, false).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 14.0); // 1+2+5+6
        assert_eq!(result[[0, 1]], 22.0); // 3+4+7+8
        assert_eq!(result[[1, 0]], 46.0); // 9+10+13+14
        assert_eq!(result[[1, 1]], 54.0); // 11+12+15+16
    }

    #[test]
    fn pad_single_axis() {
        let a = ArrayD::ones(IxDyn(&[2, 3, 4]));
        let result = pad(
            &a,
            &PaddingOptions {
                config: PaddingOptionConfig {
                    left: 2,
                    right: 2,
                    interior: 1,
                },
                axes: vec![0],
                value: 0.0,
            },
        );

        let mut expected = ArrayD::from_elem(IxDyn(&[7, 3, 4]), 0.0);
        expected.slice_mut(s![2, .., ..]).fill(1.0);
        expected.slice_mut(s![4, .., ..]).fill(1.0);

        assert_eq!(result.shape(), &[7, 3, 4]);
        assert_eq!(result, expected);
    }
}
