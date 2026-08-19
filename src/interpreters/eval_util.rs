use ndarray::{ArrayD, Axis, Ix0, Ix1, Ix2, IxDyn, Zip, arr0, indices, linalg::Dot};

use crate::{
    interpreters::EvalError,
    mininn::{PaddingOptions, PoolOptions},
};

/// A concrete tensor value in the `eval` interpreter.
pub type Tensor = ArrayD<f64>;

/// Handles negative python style axis index and converts it into an absolute axis index
pub(crate) fn norm_axis_index(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        (axis + ndim as isize) as usize
    } else {
        axis as usize
    }
}

/// numpy broadcasting: align shapes from the right, expanding size-1 dims.
pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
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
pub(crate) fn binary(
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Tensor, EvalError> {
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
pub(crate) fn where_(cond: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor, EvalError> {
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
pub(crate) fn reshape_c(a: &Tensor, shape: &[usize]) -> Tensor {
    let data: Vec<f64> = a.iter().copied().collect();
    ArrayD::from_shape_vec(IxDyn(shape), data).expect("reshape element count mismatch")
}

pub(crate) fn normcdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}

/// numpy.moveaxis: move `src` axis to `dst`, keeping the relative order of the rest.
pub(crate) fn moveaxis(a: &Tensor, src: isize, dst: isize) -> Tensor {
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
pub(crate) fn dot(a: &Tensor, b: &Tensor) -> Result<Tensor, EvalError> {
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
pub(crate) fn reduce_sum(a: &Tensor, axes: &[isize]) -> Tensor {
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
pub(crate) fn expand_dims(a: &Tensor, axes: &[isize]) -> Tensor {
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
pub(crate) fn reshape(a: &Tensor, new_shape: &[isize]) -> Result<Tensor, EvalError> {
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
pub(crate) fn pad(a: &Tensor, opt: &PaddingOptions) -> Tensor {
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
pub(crate) fn conv(input: &Tensor, kernel: &Tensor, stride: isize) -> Result<Tensor, EvalError> {
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
pub(crate) fn pool(a: &Tensor, opt: &PoolOptions, average: bool) -> Result<Tensor, EvalError> {
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
