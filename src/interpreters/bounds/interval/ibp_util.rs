use std::ops::{Add, Mul, Neg, Sub};

use ndarray::{ArrayD, ArrayView, Axis, Zip, arr0};

use crate::{
    interpreters::{
        EvalError,
        concrete::eval_util::{Tensor, binary, reshape_c},
    },
    mininn::{PaddingOptions, PoolOptions, Value},
};

// ================================
// IBPTensor
// ================================
#[derive(Debug, Clone)]
pub struct IBPTensor {
    pub lb: Tensor,
    pub ub: Tensor,
    is_point: bool,
}
impl IBPTensor {
    pub fn new(lb: Tensor, ub: Tensor) -> Self {
        assert_eq!(lb.shape(), ub.shape());
        let is_point = lb == ub;
        Self { lb, ub, is_point }
    }

    pub fn is_point(&self) -> bool {
        self.is_point
    }
}

impl Mul for IBPTensor {
    type Output = IBPTensor;

    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.lb.clone() * rhs.lb.clone();
        let b = self.lb * rhs.ub.clone();
        let c = self.ub.clone() * rhs.lb;
        let d = self.ub * rhs.ub;

        let min = min4(&a, &b, &c, &d);
        let max = max4(&a, &b, &c, &d);

        IBPTensor::new(min, max)
    }
}

impl Add for IBPTensor {
    type Output = IBPTensor;

    fn add(self, rhs: Self) -> Self::Output {
        IBPTensor {
            lb: self.lb + rhs.lb,
            ub: self.ub + rhs.ub,
            is_point: self.is_point && rhs.is_point,
        }
    }
}

impl Add for &IBPTensor {
    type Output = IBPTensor;

    fn add(self, rhs: &IBPTensor) -> IBPTensor {
        IBPTensor {
            lb: &self.lb + &rhs.lb,
            ub: &self.ub + &rhs.ub,
            is_point: self.is_point && rhs.is_point,
        }
    }
}

impl Mul for &IBPTensor {
    type Output = IBPTensor;

    fn mul(self, rhs: &IBPTensor) -> IBPTensor {
        let a = &self.lb * &rhs.lb;
        let b = &self.lb * &rhs.ub;
        let c = &self.ub * &rhs.lb;
        let d = &self.ub * &rhs.ub;
        IBPTensor::new(min4(&a, &b, &c, &d), max4(&a, &b, &c, &d))
    }
}

impl Neg for IBPTensor {
    type Output = IBPTensor;

    fn neg(self) -> Self::Output {
        IBPTensor::new(-self.ub, -self.lb)
    }
}

impl Neg for &IBPTensor {
    type Output = IBPTensor;

    fn neg(self) -> IBPTensor {
        IBPTensor::new(-&self.ub, -&self.lb)
    }
}

impl Sub for IBPTensor {
    type Output = IBPTensor;

    fn sub(self, rhs: Self) -> IBPTensor {
        IBPTensor {
            lb: self.lb - rhs.ub,
            ub: self.ub - rhs.lb,
            is_point: self.is_point && rhs.is_point,
        }
    }
}

impl Sub for &IBPTensor {
    type Output = IBPTensor;

    fn sub(self, rhs: &IBPTensor) -> IBPTensor {
        IBPTensor {
            lb: &self.lb - &rhs.ub,
            ub: &self.ub - &rhs.lb,
            is_point: self.is_point && rhs.is_point,
        }
    }
}

impl From<f64> for IBPTensor {
    fn from(value: f64) -> Self {
        let t: Tensor = arr0(value).into_dyn().into();
        IBPTensor::new(t.clone(), t)
    }
}

impl From<ArrayD<f64>> for IBPTensor {
    fn from(value: ArrayD<f64>) -> Self {
        let t: Tensor = value.into();
        IBPTensor::new(t.clone(), t)
    }
}

impl Value for IBPTensor {
    fn shape(&self) -> &[usize] {
        self.lb.shape()
    }

    fn ndim(&self) -> usize {
        self.lb.ndim()
    }

    fn len(&self) -> usize {
        self.lb.len()
    }

    fn r#where(cond: &Self, x: &Self, y: &Self) -> Result<Self, EvalError> {
        ibp_where(cond, x, y)
    }

    fn moveaxis(&self, src: isize, dst: isize) -> Self {
        IBPTensor::new(self.lb.moveaxis(src, dst), self.ub.moveaxis(src, dst))
    }

    fn dot(&self, b: &Self) -> Result<Self, EvalError> {
        ibp_linear(|a, bv| a.dot(bv), self, b)
    }

    fn square(&self) -> Self {
        ibp_square(self)
    }

    fn sqrt(&self) -> Self {
        IBPTensor::new(self.lb.sqrt(), self.ub.sqrt())
    }

    fn reciprocal(&self) -> Self {
        ibp_reciprocal(self)
    }

    fn reduce_sum(&self, axes: &[isize]) -> Self {
        IBPTensor::new(self.lb.reduce_sum(axes), self.ub.reduce_sum(axes))
    }

    fn expand_dims(&self, axes: &[isize]) -> Self {
        IBPTensor::new(self.lb.expand_dims(axes), self.ub.expand_dims(axes))
    }

    fn reshape(&self, new_shape: &[isize]) -> Result<Self, EvalError> {
        Ok(IBPTensor::new(
            self.lb.reshape(new_shape)?,
            self.ub.reshape(new_shape)?,
        ))
    }

    fn pad(&self, opt: &PaddingOptions) -> Self {
        IBPTensor::new(self.lb.pad(opt), self.ub.pad(opt))
    }

    fn conv(&self, kernel: &Self, stride: isize) -> Result<Self, EvalError> {
        ibp_linear(|a, b| a.conv(b, stride), self, kernel)
    }

    fn pool(&self, opt: &PoolOptions, average: bool) -> Result<Self, EvalError> {
        Ok(IBPTensor::new(
            self.lb.pool(opt, average)?,
            self.ub.pool(opt, average)?,
        ))
    }

    fn exp(&self) -> Self {
        IBPTensor::new(self.lb.exp(), self.ub.exp())
    }

    fn log(&self) -> Self {
        IBPTensor::new(self.lb.log(), self.ub.log())
    }

    fn relu(&self) -> Self {
        IBPTensor::new(self.lb.relu(), self.ub.relu())
    }

    fn leaky_relu(&self, slope: f64) -> Self {
        IBPTensor::new(self.lb.leaky_relu(slope), self.ub.leaky_relu(slope))
    }

    fn elu(&self, slope: f64) -> Self {
        IBPTensor::new(self.lb.elu(slope), self.ub.elu(slope))
    }

    fn normcdf(&self) -> Self {
        IBPTensor::new(self.lb.normcdf(), self.ub.normcdf())
    }

    fn gelu(&self) -> Self {
        ibp_gelu(self)
    }
}

// ================================
// IBPTensor Batched
// ================================
#[derive(Debug, Clone)]
pub struct IBPBatchedTensor {
    pub lb: Tensor, // shape [k, d ...]
    pub ub: Tensor, // shape [k, d ...]
}

impl IBPBatchedTensor {
    pub fn batch_size(&self) -> usize {
        self.lb.shape()[0]
    }

    pub fn get(&self, i: usize) -> IBPTensor {
        let lb = self.lb.index_axis(Axis(0), i);
        let ub = self.ub.index_axis(Axis(0), i);
        IBPTensor::new(lb, ub)
    }

    pub fn stack_input(input_tensors: &[&Vec<IBPTensor>], idx: usize) -> Self {
        assert!(!input_tensors.is_empty());

        let lbs: Vec<ArrayView<f64, _>> = input_tensors
            .iter()
            .map(|v| v[idx].lb.view())
            .collect();
        let ubs: Vec<ArrayView<f64, _>> = input_tensors
            .iter()
            .map(|v| v[idx].ub.view())
            .collect();

        Self {
            lb: ndarray::stack(Axis(0), &lbs)
                .expect("lb stack failed")
                .into(),
            ub: ndarray::stack(Axis(0), &ubs)
                .expect("ub stack failed")
                .into(),
        }
    }

    pub(crate) fn expand_dims_batched(a: &Tensor, axes: &[isize]) -> Tensor {
        let shifted: Vec<isize> = axes
            .iter()
            .map(|&ax| if ax >= 0 { ax + 1 } else { ax })
            .collect();
        a.expand_dims(&shifted)
    }

    pub(crate) fn moveaxis_batched(a: &Tensor, src: isize, dst: isize) -> Tensor {
        let src_b = if src >= 0 { src + 1 } else { src };
        let dst_b = if dst >= 0 { dst + 1 } else { dst };
        a.moveaxis(src_b, dst_b)
    }

    pub(crate) fn reshape_batched(a: &Tensor, new_shape: &[isize]) -> Result<Tensor, EvalError> {
        let k = a.shape()[0] as isize;
        let mut batch_shape = vec![k];
        batch_shape.extend_from_slice(new_shape);
        a.reshape(&batch_shape)
    }

    pub(crate) fn pad_batched(a: &Tensor, opt: &PaddingOptions) -> Tensor {
        let shifted_axes: Vec<isize> = opt
            .axes
            .iter()
            .map(|&ax| if ax >= 0 { ax + 1 } else { ax })
            .collect();
        a.pad(&PaddingOptions {
            axes: shifted_axes,
            ..opt.clone()
        })
    }

    pub(crate) fn reduce_sum_batched(a: &Tensor, axes: &[isize]) -> Tensor {
        let shifted: Vec<isize> = axes
            .iter()
            .map(|&ax| if ax >= 0 { ax + 1 } else { ax })
            .collect();
        a.reduce_sum(&shifted)
    }

    pub(crate) fn conv_batched(
        input: &Tensor,
        kernel: &Tensor,
        stride: isize,
    ) -> Result<Tensor, EvalError> {
        let sh = input.shape().to_vec();
        let (k, n) = (sh[0], sh[1]);
        let flat = reshape_c(input, &[k * n, sh[2], sh[3], sh[4]]);
        let out = flat.conv(kernel, stride)?;
        let osh = out.shape().to_vec();
        Ok(reshape_c(&out, &[k, n, osh[1], osh[2], osh[3]]))
    }

    pub(crate) fn pool_batched(
        a: &Tensor,
        opt: &PoolOptions,
        average: bool,
    ) -> Result<Tensor, EvalError> {
        if a.ndim() == 5 {
            let sh = a.shape().to_vec();
            let (k, n) = (sh[0], sh[1]);
            let flat = reshape_c(a, &[k * n, sh[2], sh[3], sh[4]]);
            let out = flat.pool(opt, average)?;
            let osh = out.shape().to_vec();
            Ok(reshape_c(&out, &[k, n, osh[1], osh[2], osh[3]]))
        } else {
            a.pool(opt, average)
        }
    }
}

impl From<&Vec<IBPTensor>> for IBPBatchedTensor {
    fn from(tensors: &Vec<IBPTensor>) -> Self {
        assert!(!tensors.is_empty(), "cannot batch zero tensors");

        let lbs: Vec<ArrayView<f64, _>> = tensors.iter().map(|t| t.lb.view()).collect();
        let ubs: Vec<ArrayView<f64, _>> = tensors.iter().map(|t| t.ub.view()).collect();

        IBPBatchedTensor {
            lb: ndarray::stack(Axis(0), &lbs)
                .expect("lb stack failed")
                .into(),
            ub: ndarray::stack(Axis(0), &ubs)
                .expect("ub stack failed")
                .into(),
        }
    }
}

impl Add for IBPBatchedTensor {
    type Output = IBPBatchedTensor;

    fn add(self, rhs: Self) -> Self::Output {
        IBPBatchedTensor {
            lb: self.lb + rhs.lb,
            ub: self.ub + rhs.ub,
        }
    }
}

impl Add for &IBPBatchedTensor {
    type Output = IBPBatchedTensor;

    fn add(self, rhs: &IBPBatchedTensor) -> IBPBatchedTensor {
        IBPBatchedTensor {
            lb: &self.lb + &rhs.lb,
            ub: &self.ub + &rhs.ub,
        }
    }
}

impl Mul for IBPBatchedTensor {
    type Output = IBPBatchedTensor;

    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.lb.clone() * rhs.lb.clone();
        let b = self.lb * rhs.ub.clone();
        let c = self.ub.clone() * rhs.lb;
        let d = self.ub * rhs.ub;

        let min = min4(&a, &b, &c, &d);
        let max = max4(&a, &b, &c, &d);

        IBPBatchedTensor { lb: min, ub: max }
    }
}

impl Mul for &IBPBatchedTensor {
    type Output = IBPBatchedTensor;

    fn mul(self, rhs: &IBPBatchedTensor) -> IBPBatchedTensor {
        let a = &self.lb * &rhs.lb;
        let b = &self.lb * &rhs.ub;
        let c = &self.ub * &rhs.lb;
        let d = &self.ub * &rhs.ub;
        IBPBatchedTensor {
            lb: min4(&a, &b, &c, &d),
            ub: max4(&a, &b, &c, &d),
        }
    }
}

impl Neg for IBPBatchedTensor {
    type Output = IBPBatchedTensor;

    fn neg(self) -> Self::Output {
        IBPBatchedTensor {
            lb: -self.ub,
            ub: -self.lb,
        }
    }
}

impl Neg for &IBPBatchedTensor {
    type Output = IBPBatchedTensor;

    fn neg(self) -> IBPBatchedTensor {
        IBPBatchedTensor {
            lb: -&self.ub,
            ub: -&self.lb,
        }
    }
}

impl Sub for IBPBatchedTensor {
    type Output = IBPBatchedTensor;

    fn sub(self, rhs: Self) -> IBPBatchedTensor {
        IBPBatchedTensor {
            lb: self.lb - rhs.ub,
            ub: self.ub - rhs.lb,
        }
    }
}

impl Sub for &IBPBatchedTensor {
    type Output = IBPBatchedTensor;

    fn sub(self, rhs: &IBPBatchedTensor) -> IBPBatchedTensor {
        IBPBatchedTensor {
            lb: &self.lb - &rhs.ub,
            ub: &self.ub - &rhs.lb,
        }
    }
}

impl From<ArrayD<f64>> for IBPBatchedTensor {
    fn from(value: ArrayD<f64>) -> Self {
        let t: Tensor = value.into();
        IBPBatchedTensor {
            lb: t.clone(),
            ub: t,
        }
    }
}

impl From<IBPTensor> for IBPBatchedTensor {
    fn from(t: IBPTensor) -> Self {
        IBPBatchedTensor { lb: t.lb, ub: t.ub }
    }
}

impl From<&IBPBatchedTensor> for IBPTensor {
    fn from(t: &IBPBatchedTensor) -> Self {
        IBPTensor::new(t.lb.clone(), t.ub.clone())
    }
}

impl Value for IBPBatchedTensor {
    fn shape(&self) -> &[usize] {
        self.lb.shape()
    }
    fn ndim(&self) -> usize {
        self.lb.ndim()
    }
    fn len(&self) -> usize {
        self.lb.len()
    }

    fn r#where(cond: &Self, x: &Self, y: &Self) -> Result<Self, EvalError> {
        Ok(ibp_where(
            &IBPTensor::from(cond),
            &IBPTensor::from(x),
            &IBPTensor::from(y),
        )?
        .into())
    }

    fn moveaxis(&self, src: isize, dst: isize) -> Self {
        IBPBatchedTensor {
            lb: Self::moveaxis_batched(&self.lb, src, dst),
            ub: Self::moveaxis_batched(&self.ub, src, dst),
        }
    }

    fn dot(&self, b: &Self) -> Result<Self, EvalError> {
        let a_t = IBPTensor::from(self);
        let b_t = IBPTensor::from(b);
        let result = if a_t.is_point() {
            ibp_linear(
                |act, w| {
                    if act.ndim() == 2 && w.ndim() == 2 {
                        let a2 = act.as_2d();
                        let w2 = w.as_2d();
                        Ok(a2.dot(&w2.t()).into_dyn().into())
                    } else if act.ndim() == 2 && w.ndim() == 1 {
                        act.dot(w)
                    } else {
                        unreachable!()
                    }
                },
                &b_t, // activation → "a" slot (varying)
                &a_t, // weight    → "b" slot (point)
            )
        } else {
            ibp_linear(|act, w| act.dot(w), &a_t, &b_t)
        }?;
        Ok(result.into())
    }

    fn square(&self) -> Self {
        ibp_square(&IBPTensor::from(self)).into()
    }
    fn sqrt(&self) -> Self {
        IBPBatchedTensor {
            lb: self.lb.sqrt(),
            ub: self.ub.sqrt(),
        }
    }
    fn reciprocal(&self) -> Self {
        ibp_reciprocal(&IBPTensor::from(self)).into()
    }

    fn reduce_sum(&self, axes: &[isize]) -> Self {
        IBPBatchedTensor {
            lb: Self::reduce_sum_batched(&self.lb, axes),
            ub: Self::reduce_sum_batched(&self.ub, axes),
        }
    }

    fn expand_dims(&self, axes: &[isize]) -> Self {
        IBPBatchedTensor {
            lb: Self::expand_dims_batched(&self.lb, axes),
            ub: Self::expand_dims_batched(&self.ub, axes),
        }
    }

    fn reshape(&self, new_shape: &[isize]) -> Result<Self, EvalError> {
        Ok(IBPBatchedTensor {
            lb: Self::reshape_batched(&self.lb, new_shape)?,
            ub: Self::reshape_batched(&self.ub, new_shape)?,
        })
    }

    fn pad(&self, opt: &PaddingOptions) -> Self {
        IBPBatchedTensor {
            lb: Self::pad_batched(&self.lb, opt),
            ub: Self::pad_batched(&self.ub, opt),
        }
    }

    fn conv(&self, kernel: &Self, stride: isize) -> Result<Self, EvalError> {
        Ok(ibp_linear(
            |a, b| Self::conv_batched(a, b, stride),
            &IBPTensor::from(self),
            &IBPTensor::from(kernel),
        )?
        .into())
    }

    fn pool(&self, opt: &PoolOptions, average: bool) -> Result<Self, EvalError> {
        Ok(IBPBatchedTensor {
            lb: Self::pool_batched(&self.lb, opt, average)?,
            ub: Self::pool_batched(&self.ub, opt, average)?,
        })
    }

    fn exp(&self) -> Self {
        IBPBatchedTensor {
            lb: self.lb.exp(),
            ub: self.ub.exp(),
        }
    }
    fn log(&self) -> Self {
        IBPBatchedTensor {
            lb: self.lb.log(),
            ub: self.ub.log(),
        }
    }
    fn relu(&self) -> Self {
        IBPBatchedTensor {
            lb: self.lb.relu(),
            ub: self.ub.relu(),
        }
    }
    fn leaky_relu(&self, slope: f64) -> Self {
        IBPBatchedTensor {
            lb: self.lb.leaky_relu(slope),
            ub: self.ub.leaky_relu(slope),
        }
    }
    fn elu(&self, slope: f64) -> Self {
        IBPBatchedTensor {
            lb: self.lb.elu(slope),
            ub: self.ub.elu(slope),
        }
    }
    fn normcdf(&self) -> Self {
        IBPBatchedTensor {
            lb: self.lb.normcdf(),
            ub: self.ub.normcdf(),
        }
    }
    fn gelu(&self) -> Self {
        ibp_gelu(&IBPTensor::from(self)).into()
    }
}

// ================================
// IBP Special-case helpers (exported for use in ibp.rs / ibp_batched.rs)
// ================================

pub(crate) fn ibp_square(tensor: &IBPTensor) -> IBPTensor {
    let lb = binary(&tensor.lb, &tensor.ub, |x_lb, x_ub| {
        if x_lb >= 0.0 {
            x_lb * x_lb
        } else if x_ub < 0.0 {
            x_ub * x_ub
        } else {
            0.0
        }
    })
    .unwrap();

    let ub = binary(&tensor.lb, &tensor.ub, |x_lb, x_ub| {
        let yl = x_lb * x_lb;
        let yr = x_ub * x_ub;
        if x_lb >= 0.0 {
            yr
        } else if x_ub < 0.0 {
            yl
        } else {
            yl.max(yr)
        }
    })
    .unwrap();

    IBPTensor::new(lb, ub)
}

pub(crate) fn ibp_reciprocal(tensor: &IBPTensor) -> IBPTensor {
    let lb = binary(&tensor.lb, &tensor.ub, |x_lb, x_ub| {
        if x_lb <= 0.0 && x_ub >= 0.0 {
            f64::NEG_INFINITY
        } else {
            1.0 / x_ub
        }
    })
    .unwrap();
    let ub = binary(&tensor.lb, &tensor.ub, |x_lb, x_ub| {
        if x_lb <= 0.0 && x_ub >= 0.0 {
            f64::INFINITY
        } else {
            1.0 / x_lb
        }
    })
    .unwrap();

    IBPTensor::new(lb, ub)
}

pub(crate) fn ibp_where(
    condition: &IBPTensor,
    x: &IBPTensor,
    y: &IBPTensor,
) -> Result<IBPTensor, EvalError> {
    if condition.is_point() {
        let lb = Tensor::r#where(&condition.lb, &x.lb, &y.lb)?;
        let ub = Tensor::r#where(&condition.ub, &x.ub, &y.ub)?;
        return Ok(IBPTensor::new(lb, ub));
    }

    let ll = Tensor::r#where(&condition.lb, &x.lb, &y.lb)?;
    let lu = Tensor::r#where(&condition.lb, &x.ub, &y.ub)?;
    let rl = Tensor::r#where(&condition.ub, &x.lb, &y.lb)?;
    let ru = Tensor::r#where(&condition.ub, &x.ub, &y.ub)?;

    let lb = binary(&ll, &rl, |a, b| a.min(b)).unwrap();
    let ub = binary(&lu, &ru, |a, b| a.max(b)).unwrap();

    Ok(IBPTensor::new(lb, ub))
}

pub(crate) fn ibp_gelu(tensor: &IBPTensor) -> IBPTensor {
    const GELU_ARG_MIN: f64 = -0.751791524693564;
    const GELU_MIN: f64 = -0.16997120747990366;

    let yl = tensor.lb.gelu();
    let yr = tensor.ub.gelu();

    let lb = Zip::from(&tensor.lb)
        .and(&tensor.ub)
        .and(&yl)
        .and(&yr)
        .map_collect(|&x_lb, &x_ub, &yl, &yr| {
            if x_lb >= GELU_ARG_MIN {
                yl
            } else if x_ub < GELU_ARG_MIN {
                yr
            } else {
                GELU_MIN
            }
        });

    let ub = Zip::from(&tensor.lb)
        .and(&tensor.ub)
        .and(&yl)
        .and(&yr)
        .map_collect(|&x_lb, &x_ub, &yl, &yr| {
            if x_lb >= GELU_ARG_MIN {
                yr
            } else if x_ub < GELU_ARG_MIN {
                yl
            } else {
                yl.max(yr)
            }
        });

    IBPTensor::new(lb.into(), ub.into())
}

// ================================
// IBP Helper Functions
// ================================

pub(crate) fn min4(a: &Tensor, b: &Tensor, c: &Tensor, d: &Tensor) -> Tensor {
    Zip::from(a)
        .and(b)
        .and(c)
        .and(d)
        .map_collect(|a, b, c, d| a.min(*b).min(*c).min(*d))
        .into()
}

pub(crate) fn max4(a: &Tensor, b: &Tensor, c: &Tensor, d: &Tensor) -> Tensor {
    Zip::from(a)
        .and(b)
        .and(c)
        .and(d)
        .map_collect(|a, b, c, d| a.max(*b).max(*c).max(*d))
        .into()
}

pub(crate) fn ibp_linear<F>(f: F, a: &IBPTensor, b: &IBPTensor) -> Result<IBPTensor, EvalError>
where
    F: Fn(&Tensor, &Tensor) -> Result<Tensor, EvalError>,
{
    if a.is_point() {
        let x = a.lb.clone();
        let y_mid = (b.ub.clone() + b.lb.clone()) * 0.5;
        let y_ran = (b.ub.clone() - b.lb.clone()) * 0.5;
        let out_mid = f(&x, &y_mid)?;
        let out_ran = f(&x.mapv(|v| v.abs()), &y_ran)?;

        Ok(IBPTensor::new(
            out_mid.clone() - out_ran.clone(),
            out_mid + out_ran,
        ))
    } else if b.is_point() {
        let y = b.lb.clone();
        let x_mid = (a.lb.clone() + a.ub.clone()) * 0.5;
        let x_ran = (a.ub.clone() - a.lb.clone()) * 0.5;
        let out_mid = f(&x_mid, &y)?;
        let out_ran = f(&x_ran, &y.mapv(|v| v.abs()))?;

        Ok(IBPTensor::new(
            out_mid.clone() - out_ran.clone(),
            out_mid + out_ran,
        ))
    } else {
        let ll = f(&a.lb, &b.lb)?;
        let lu = f(&a.lb, &b.ub)?;
        let ul = f(&a.ub, &b.lb)?;
        let uu = f(&a.ub, &b.ub)?;

        Ok(IBPTensor::new(
            min4(&ll, &lu, &ul, &uu),
            max4(&ll, &lu, &ul, &uu),
        ))
    }
}
