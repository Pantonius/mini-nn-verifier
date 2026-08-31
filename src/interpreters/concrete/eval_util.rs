use std::ops::{Add, Div, Index, Mul, Neg, Sub};

use ndarray::{
    ArrayD, ArrayView, ArrayView1, ArrayView2, Axis, IntoNdProducer, Ix0, Ix1, Ix2, IxDyn, NdIndex,
    Zip, arr0, indices, linalg::Dot,
};

use crate::mininn::{MininnError, PaddingOptions, PoolOptions, Value};

/// Handles negative python style axis index and converts it into an absolute axis index
pub fn norm_axis_index(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        (axis + ndim as isize) as usize
    } else {
        axis as usize
    }
}

/// numpy broadcasting: align shapes from the right, expanding size-1 dims.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
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

/// Reshape into `shape` in C (row-major) order.
/// NOTE on terminology:
/// - C stands for contigous, mathematically speaking row-major order
/// - F stands for fortran-contigous, mathematically speaking column-major order
pub fn reshape_c(a: &Tensor, shape: &[usize]) -> Tensor {
    let data: Vec<f64> = a.inner.iter().copied().collect();
    ArrayD::from_shape_vec(IxDyn(shape), data)
        .expect("reshape element count mismatch")
        .into()
}

/// Elementwise binary op with numpy broadcasting.
/// Given two tensors and a binary function (f64, f64) -> f64 returns a tensor on successful
/// application
pub fn binary(a: &Tensor, b: &Tensor, f: impl Fn(f64, f64) -> f64) -> Result<Tensor, MininnError> {
    // try to compute the broadcast shape
    let shape = broadcast_shape(a.shape(), b.shape()).ok_or_else(|| {
        MininnError::Parse(format!(
            "incompatible shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        ))
    })?;

    // broadcast each to the computed shape
    let av = a
        .inner
        .broadcast(IxDyn(&shape))
        .ok_or_else(|| MininnError::Parse("broadcast failed".to_string()))?;

    let bv = b
        .inner
        .broadcast(IxDyn(&shape))
        .ok_or_else(|| MininnError::Parse("broadcast failed".to_string()))?;

    // zip and map elementwise via given binary function
    // NOTE panic is impossible since av and bv are of same shape
    Ok(Zip::from(&av).and(&bv).map_collect(|&x, &y| f(x, y)).into())
}

pub fn normcdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}

pub fn normpdf(x: f64) -> f64 {
    (-(0.5 * x * x)).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// A concrete tensor value in the `eval` interpreter.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    inner: ArrayD<f64>,
}

impl<'a> IntoNdProducer for &'a Tensor {
    type Item = &'a f64;
    type Dim = IxDyn;
    type Output = ArrayView<'a, f64, IxDyn>;

    fn into_producer(self) -> Self::Output {
        self.inner.view()
    }
}

impl Neg for Tensor {
    type Output = Self;

    fn neg(self) -> Self::Output {
        (-self.inner).into()
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self.mapv(|x| -x)
    }
}

impl Add for Tensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        binary(&self, &rhs, |a, b| a + b).unwrap()
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        binary(self, rhs, |a, b| a + b).unwrap()
    }
}

impl Sub for Tensor {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        binary(&self, &rhs, |a, b| a - b).unwrap()
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        binary(self, rhs, |a, b| a - b).unwrap()
    }
}

impl Mul for Tensor {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        binary(&self, &rhs, |a, b| a * b).unwrap()
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        binary(self, rhs, |a, b| a * b).unwrap()
    }
}

impl<Idx: NdIndex<IxDyn>> Index<Idx> for Tensor {
    type Output = f64;

    fn index(&self, idx: Idx) -> &f64 {
        &self.inner[idx]
    }
}

impl Tensor {
    pub fn into_inner(self) -> ArrayD<f64> {
        self.inner
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn iter(&self) -> ndarray::iter::Iter<'_, f64, IxDyn> {
        self.inner.iter()
    }

    pub fn view(&self) -> ArrayView<'_, f64, IxDyn> {
        self.inner.view()
    }

    pub fn index_axis(&self, axis: Axis, index: usize) -> Tensor {
        self.inner.index_axis(axis, index).to_owned().into()
    }

    pub fn as_1d(&self) -> ArrayView1<'_, f64> {
        self.inner.view().into_dimensionality::<Ix1>().unwrap()
    }

    pub fn as_2d(&self) -> ArrayView2<'_, f64> {
        self.inner.view().into_dimensionality::<Ix2>().unwrap()
    }

    pub fn mapv(&self, f: impl Fn(f64) -> f64) -> Self {
        self.inner.mapv(f).into()
    }

    pub fn pos_part(&self) -> Self {
        self.mapv(|x| if x >= 0.0 { x } else { 0.0 })
    }

    pub fn neg_part(&self) -> Self {
        self.mapv(|x| if x < 0.0 { x } else { 0.0 })
    }

    pub fn sum(&self) -> f64 {
        self.inner.sum()
    }
}

impl IntoIterator for Tensor {
    type Item = f64;
    type IntoIter = ndarray::iter::IntoIter<f64, IxDyn>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a> IntoIterator for &'a Tensor {
    type Item = &'a f64;
    type IntoIter = ndarray::iter::Iter<'a, f64, IxDyn>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl Mul<f64> for Tensor {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        self.inner.mapv(|x| x * rhs).into()
    }
}

impl Div<f64> for Tensor {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        self.inner.mapv(|x| x / rhs).into()
    }
}

impl Div<Tensor> for Tensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        binary(&self, &rhs, |a, b| a / b).unwrap()
    }
}

impl From<f64> for Tensor {
    fn from(value: f64) -> Self {
        ArrayD::from_elem(IxDyn(&[]), value).into()
    }
}

impl From<ArrayD<f64>> for Tensor {
    fn from(value: ArrayD<f64>) -> Self {
        Self { inner: value }
    }
}

impl Value for Tensor {
    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    /// numpy.where(cond, x, y) with broadcasting; `cond` is truthy when non-zero.
    fn r#where(cond: &Self, x: &Self, y: &Self) -> Result<Self, MininnError> {
        // compute broadcasted shape for cond and x, then for that s and y
        let s = broadcast_shape(cond.shape(), x.shape())
            .and_then(|s| broadcast_shape(&s, y.shape()))
            .ok_or_else(|| MininnError::Parse("where: incompatible shapes".to_string()))?;

        // then actually broadcast each to that shape
        let cv = cond
            .inner
            .broadcast(IxDyn(&s))
            .ok_or_else(|| MininnError::Parse("broadcast failed".to_string()))?;

        let xv = x
            .inner
            .broadcast(IxDyn(&s))
            .ok_or_else(|| MininnError::Parse("broadcast failed".to_string()))?;

        let yv = y
            .inner
            .broadcast(IxDyn(&s))
            .ok_or_else(|| MininnError::Parse("broadcast failed".to_string()))?;

        // zip cv, xv, yv into a triple and apply the condition logic elementwise
        // NOTE panic is impossible since cv, xv, yv are of same shape
        Ok(Tensor {
            inner: Zip::from(&cv)
                .and(&xv)
                .and(&yv)
                .map_collect(|&c, &x, &y| if c != 0.0 { x } else { y }),
        })
    }

    /// numpy.moveaxis: move `src` axis to `dst`, keeping the relative order of the rest.
    fn moveaxis(&self, src: isize, dst: isize) -> Self {
        let nd = self.ndim(); // number of dimensions

        // proper indices
        let (s, d) = (norm_axis_index(src, nd), norm_axis_index(dst, nd));

        // all indices except the one being moved (s)
        let mut order: Vec<usize> = (0..nd).filter(|&x| x != s).collect();

        // insert s at d
        order.insert(d, s);

        // materialize
        Self {
            inner: self
                .inner
                .view()
                .permuted_axes(order)
                .as_standard_layout()
                .to_owned(),
        }
    }

    /// numpy.dot: contract the last axis of `a` with the second-to-last of `b`
    /// (or the only axis of `b` when it is 1-D).
    /// numpy.org/doc/stable/reference/generated/numpy.dot.html
    fn dot(&self, b: &Self) -> Result<Self, MininnError> {
        // shapes of each input
        let (ash, bsh) = (self.shape().to_vec(), b.shape().to_vec());

        // If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
        if ash.is_empty() {
            let a2 = self
                .inner
                .clone()
                .into_dimensionality::<Ix0>()
                .expect("dot lhs not 0-D")
                .into_scalar();

            return Ok((a2 * b.inner.clone()).into());
        } else if bsh.is_empty() {
            let b2 = b
                .inner
                .clone()
                .into_dimensionality::<Ix0>()
                .expect("dot rhs not 0-D")
                .into_scalar();

            return Ok((b2 * self.inner.clone()).into());
        }

        if (ash.len() == 1 && bsh.len() == 1) || (ash.len() == 2 && bsh.len() <= 2) {
            // If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
            // If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.

            // that we can delegate to the implement dot product
            if ash.len() == 1 {
                if ash[0] != bsh[0] {
                    return Err(MininnError::Parse(format!(
                        "dot: {ash:?} · {bsh:?} axis mismatch"
                    )));
                }

                let a2 = self
                    .inner
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .expect("dot lhs not 1-D");

                let b2 = b
                    .inner
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .expect("dot rhs not 1-D");

                return Ok(arr0(a2.dot(&b2)).into_dyn().into());
            } else {
                if ash[1] != bsh[0] {
                    return Err(MininnError::Parse(format!(
                        "dot: {ash:?} · {bsh:?} axis mismatch"
                    )));
                }

                let a2 = self
                    .inner
                    .clone()
                    .into_dimensionality::<Ix2>()
                    .expect("dot lhs not 2-D");

                if bsh.len() == 1 {
                    let b2 = b
                        .inner
                        .clone()
                        .into_dimensionality::<Ix1>()
                        .expect("dot rhs not 1-D");

                    return Ok(a2.dot(&b2).into_dyn().into());
                } else {
                    let b2 = b
                        .inner
                        .clone()
                        .into_dimensionality::<Ix2>()
                        .expect("dot rhs not 2-D");

                    return Ok(a2.dot(&b2).into_dyn().into());
                };
            }
        }

        let k = ash[ash.len() - 1];
        let a_prelim_shape = [ash[..ash.len() - 1].iter().product(), k];
        let a2 = reshape_c(self, &a_prelim_shape).into_inner();

        // If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
        if bsh.len() == 1 {
            if bsh[0] != k {
                return Err(MininnError::Parse(format!(
                    "dot: {ash:?} · {bsh:?} axis mismatch"
                )));
            }

            let prelim_res = a2.dot(&b.inner);
            return Ok(reshape_c(&prelim_res.into(), &ash[..ash.len() - 1]));
        }

        // If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b:
        if bsh[bsh.len() - 2] != k {
            return Err(MininnError::Parse(format!(
                "dot: {ash:?} · {bsh:?} axis mismatch"
            )));
        }

        let b_moved = Self::moveaxis(b, -2, 0); // move
        let b_prelim_shape = [k, b_moved.shape()[1..].iter().product()]; // flatten
        let b2 = reshape_c(&b_moved, &b_prelim_shape).into_inner(); // apply

        let prelim_res = a2.dot(&b2); // flat result

        let mut out_shape = ash[..ash.len() - 1].to_vec();
        out_shape.extend_from_slice(&bsh[..bsh.len() - 2]);
        out_shape.push(bsh[bsh.len() - 1]);

        Ok(reshape_c(&prelim_res.into(), &out_shape)) // unflattened result
    }

    /// Sum over the given axes (numpy default: axes are removed).
    fn reduce_sum(&self, axes: &[isize]) -> Self {
        // normalizes axes and then sort and iterate from behind such that we don't need to shift higher
        // axes when reducing lower axes
        let mut norm_axes: Vec<usize> = axes
            .iter()
            .map(|&ax| norm_axis_index(ax, self.ndim()))
            .collect();
        norm_axes.sort_unstable();
        norm_axes.dedup();

        let mut out = self.inner.clone();
        for ax in norm_axes.into_iter().rev() {
            out = out.sum_axis(Axis(ax));
        }

        out.into()
    }

    /// numpy.expand_dims: insert size-1 axes at the given positions (which refer to
    /// the result's axes).
    fn expand_dims(&self, axes: &[isize]) -> Self {
        let mut norm_axes: Vec<usize> = axes
            .iter()
            .map(|&ax| norm_axis_index(ax, self.ndim() + axes.len()))
            .collect();
        norm_axes.sort_unstable();

        let mut out = self.inner.clone();
        for pos in norm_axes {
            out = out.insert_axis(Axis(pos));
        }

        out.into()
    }

    /// Reshape resolving a single inferred dimensions (specified by -1).
    fn reshape(&self, new_shape: &[isize]) -> Result<Self, MininnError> {
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
                        self.len() / known_dims
                    }
                } else {
                    d as usize
                }
            })
            .collect();

        if shape.iter().product::<usize>() != self.len() {
            return Err(MininnError::Parse(format!(
                "reshape {:?} -> {new_shape:?} changes element count",
                self.shape()
            )));
        }

        Ok(reshape_c(self, &shape))
    }

    fn slice(&self, axis: isize, start: isize, end: Option<isize>, step: isize) -> Self {
        let ax = norm_axis_index(axis, self.ndim());
        self.inner
            .slice_axis(Axis(ax), ndarray::Slice::new(start, end, step))
            .to_owned()
            .into()
    }

    /// jax.lax.pad: per listed axis, add `left`/`right` padding and `interior`
    /// dilation between elements, filling with `value`.
    fn pad(&self, opt: &PaddingOptions) -> Self {
        let is_padded = |i: usize| {
            opt.axes
                .iter()
                .any(|&ax| norm_axis_index(ax, self.ndim()) == i)
        };

        let out_shape: Vec<usize> = (0..self.ndim())
            .map(|i| {
                let si = self.shape()[i];
                if is_padded(i) {
                    opt.config.left + si + (si - 1) * opt.config.interior + opt.config.right
                } else {
                    si
                }
            })
            .collect();

        let mut out = ArrayD::from_elem(IxDyn(&out_shape), opt.value);
        for (j, &val) in self.inner.indexed_iter() {
            let dest: Vec<usize> = (0..self.ndim())
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

        out.into()
    }

    fn conv_kernel_grad(
        &self,
        input: &Self,
        stride: isize,
        kernel_shape: &[usize],
    ) -> Result<Self, MininnError> {
        let s = stride as usize;
        let (oc, ic, kh, kw) = (
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            kernel_shape[3],
        );
        let n = self.shape()[0];
        let oh = self.shape()[2];
        let ow = self.shape()[3];
        let mut d_kernel = ArrayD::zeros(IxDyn(kernel_shape));
        let g = self.inner.view();
        let inp = input.inner.view();
        for ni in 0..n {
            for oci in 0..oc {
                for hi in 0..oh {
                    for wi in 0..ow {
                        let t = g[[ni, oci, hi, wi]];
                        for ici in 0..ic {
                            for khi in 0..kh {
                                for kwi in 0..kw {
                                    d_kernel[[oci, ici, khi, kwi]] +=
                                        t * inp[[ni, ici, hi * s + khi, wi * s + kwi]];
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(d_kernel.into())
    }

    /// 2-D cross-correlation (NCHW input, OIHW kernel), single stride for H and W.
    fn conv(&self, kernel: &Self, stride: isize) -> Result<Self, MininnError> {
        if self.ndim() != 4 || kernel.ndim() != 4 {
            return Err(MininnError::Parse(
                "conv expects 4-D self and kernel".to_string(),
            ));
        }

        let s = stride as usize;
        let (n, c, h, w) = (
            self.shape()[0],
            self.shape()[1],
            self.shape()[2],
            self.shape()[3],
        );
        let (ko, kc, kh, kw) = (
            kernel.shape()[0],
            kernel.shape()[1],
            kernel.shape()[2],
            kernel.shape()[3],
        );

        if kc != c {
            return Err(MininnError::Parse(format!(
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
                                    agg += self.inner[[ni, cpi, s * hi + i, s * wi + j]]
                                        * kernel.inner[[ci, cpi, i, j]];
                                }
                            }
                        }

                        out[[ni, ci, hi, wi]] = agg;
                    }
                }
            }
        }

        Ok(out.into())
    }

    /// Windowed sum/average pooling over every axis (per-axis window and stride).
    fn pool(&self, opt: &PoolOptions, average: bool) -> Result<Self, MininnError> {
        if opt.window_size.len() != self.ndim() || opt.stride.len() != self.ndim() {
            return Err(MininnError::Parse(
                "pool: window/stride rank must match input".to_string(),
            ));
        }

        let out_shape: Vec<usize> = (0..self.ndim())
            .map(|i| (self.shape()[i] - opt.window_size[i]) / opt.stride[i] + 1)
            .collect();

        let mut out = ArrayD::zeros(IxDyn(&out_shape));

        let window_total: usize = opt.window_size.iter().product();

        for (oidx, slot) in out.indexed_iter_mut() {
            let mut acc = 0.0;

            for widx in indices(IxDyn(&opt.window_size)) {
                let sidx: Vec<usize> = (0..self.ndim())
                    .map(|ax| opt.stride[ax] * oidx[ax] + widx[ax])
                    .collect();
                acc += self.inner[IxDyn(&sidx)]
            }

            *slot = if average {
                acc / window_total as f64
            } else {
                acc
            }
        }
        Ok(out.into())
    }

    fn exp(&self) -> Self {
        self.inner.exp().into()
    }

    fn log(&self) -> Self {
        self.inner.mapv(f64::ln).into()
    }

    fn relu(&self) -> Self {
        self.inner.mapv(|x| x.max(0.0)).into()
    }

    fn leaky_relu(&self, slope: f64) -> Self {
        self.inner
            .mapv(|x| if x >= 0.0 { x } else { slope * x })
            .into()
    }

    fn elu(&self, slope: f64) -> Self {
        self.inner
            .mapv(|x| x.max(0.0) + slope * (x.exp() - 1.0).min(0.0))
            .into()
    }

    fn normcdf(&self) -> Self {
        self.inner.mapv(|x| normcdf(x)).into()
    }

    fn gelu(&self) -> Self {
        self.inner.mapv(|x| x * normcdf(x)).into()
    }

    fn square(&self) -> Self {
        self.inner.mapv(|x| x * x).into()
    }

    fn sqrt(&self) -> Self {
        self.inner.sqrt().into()
    }

    fn reciprocal(&self) -> Self {
        self.inner.recip().into()
    }
}
