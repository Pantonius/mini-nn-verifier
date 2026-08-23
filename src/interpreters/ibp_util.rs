use std::ops::{Add, Mul};

use ndarray::{ArrayD, Zip, arr0};

use crate::{
    interpreters::{
        EvalError,
        eval_util::{Tensor, gelu, where_},
    },
    mininn::Value,
};

/// interval bounds
#[derive(Debug, Clone)]
pub struct IBPTensor {
    pub lb: Tensor,
    pub ub: Tensor,
    is_point: bool,
}
impl IBPTensor {
    pub fn new(lb: Tensor, ub: Tensor) -> Self {
        // TODO maybe there is a prettier way here
        assert_eq!(lb.shape(), ub.shape());

        let is_point = lb == ub;
        Self { lb, ub, is_point }
    }

    fn is_point(&self) -> bool {
        self.is_point
    }

    fn shape(&self) -> &[usize] {
        self.lb.shape()
    }
}

impl Mul for IBPTensor {
    type Output = IBPTensor;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut a = self.lb.clone() * rhs.lb.clone();
        let b = self.lb * rhs.ub.clone();
        let c = self.ub.clone() * rhs.lb;
        let d = self.ub * rhs.ub;

        let min = min4(&mut a, &b, &c, &d);
        let max = max4(&mut a, &b, &c, &d);

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

impl From<f64> for IBPTensor {
    fn from(value: f64) -> Self {
        IBPTensor {
            lb: arr0(value).into_dyn(),
            ub: arr0(value).into_dyn(),
            is_point: true,
        }
    }
}

impl Value for IBPTensor {
    fn from_tensor(tensor: &ArrayD<f64>) -> Self {
        IBPTensor::new(tensor.clone(), tensor.clone())
    }
}

// ================================
// IBP Helper Functions
// ================================
// As in the original python implementation
pub(crate) fn min4(a: &Tensor, b: &Tensor, c: &Tensor, d: &Tensor) -> Tensor {
    Zip::from(a)
        .and(b)
        .and(c)
        .and(d)
        .map_collect(|a, b, c, d| a.min(*b).min(*c).min(*d))
}

pub(crate) fn max4(a: &Tensor, b: &Tensor, c: &Tensor, d: &Tensor) -> Tensor {
    Zip::from(a)
        .and(b)
        .and(c)
        .and(d)
        .map_collect(|a, b, c, d| a.max(*b).max(*c).max(*d))
}

pub(crate) fn ibp_monotonic_non_dec<T: Fn(&Tensor) -> Result<Tensor, EvalError>>(
    f: T,
    tensor: &IBPTensor,
) -> Result<IBPTensor, EvalError> {
    let lb = f(&tensor.lb)?;
    let ub = f(&tensor.ub)?;

    Ok(IBPTensor::new(lb, ub))
}

pub(crate) fn ibp_monotonic_non_dec2<T: Fn(&Tensor, &Tensor) -> Result<Tensor, EvalError>>(
    f: T,
    a: &IBPTensor,
    b: &IBPTensor,
) -> Result<IBPTensor, EvalError> {
    let lb = f(&a.lb, &b.lb)?;
    let ub = f(&a.ub, &b.ub)?;

    Ok(IBPTensor::new(lb, ub))
}

pub(crate) fn ibp_monotonic_non_inc<T: Fn(&Tensor) -> Result<Tensor, EvalError>>(
    f: T,
    tensor: &IBPTensor,
) -> Result<IBPTensor, EvalError> {
    let arr = ibp_monotonic_non_dec(f, &tensor)?;
    Ok(IBPTensor::new(arr.ub, arr.lb))
}

pub(crate) fn ibp_linear<T: Fn(&Tensor, &Tensor) -> Result<Tensor, EvalError>>(
    f: T,
    a: &IBPTensor,
    b: &IBPTensor,
) -> Result<IBPTensor, EvalError> {
    if a.is_point() {
        let x = a.lb.clone();
        let y_mid = (b.ub.clone() + b.lb.clone()) * 0.5;
        let y_ran = (b.ub.clone() - b.lb.clone()) * 0.5;
        let out_mid = f(&x, &y_mid)?;
        let out_ran = f(&x.abs(), &y_ran)?;

        Ok(IBPTensor::new(
            out_mid.clone() - out_ran.clone(),
            out_mid + out_ran,
        ))
    } else if b.is_point() {
        let y = b.lb.clone();
        let x_mid = (a.lb.clone() + a.ub.clone()) * 0.5;
        let x_ran = (a.ub.clone() - a.lb.clone()) * 0.5;
        let out_mid = f(&x_mid, &y)?;
        let out_ran = f(&x_ran, &y.abs())?;

        Ok(IBPTensor::new(
            out_mid.clone() - out_ran.clone(),
            out_mid + out_ran,
        ))
    } else {
        let mut ll = f(&a.lb, &b.lb)?;
        let lu = f(&a.lb, &b.ub)?;
        let ul = f(&a.ub, &b.lb)?;
        let uu = f(&a.ub, &b.ub)?;

        Ok(IBPTensor::new(
            min4(&mut ll, &lu, &ul, &uu),
            max4(&mut ll, &lu, &ul, &uu),
        ))
    }
}

pub(crate) fn ibp_square(tensor: &IBPTensor) -> Result<IBPTensor, EvalError> {
    let yl = tensor.lb.mapv(|x| x * x);
    let yr = tensor.ub.mapv(|x| x * x);

    let lb = Zip::from(&tensor.lb)
        .and(&tensor.ub)
        .and(&yl)
        .and(&yr)
        .map_collect(|&x_lb, &x_ub, &yl, &yr| {
            if x_lb >= 0.0 {
                yl
            } else if x_ub < 0.0 {
                yr
            } else {
                0.0
            }
        });
    let ub = Zip::from(&tensor.lb)
        .and(&tensor.ub)
        .and(&yl)
        .and(&yr)
        .map_collect(|&x_lb, &x_ub, &yl, &yr| {
            if x_lb >= 0.0 {
                yr
            } else if x_ub < 0.0 {
                yl
            } else {
                yl.max(yr)
            }
        });

    Ok(IBPTensor::new(lb, ub))
}

pub(crate) fn ibp_reciprocal(tensor: &IBPTensor) -> Result<IBPTensor, EvalError> {
    let lb_safe = tensor.ub.recip();
    let ub_safe = tensor.lb.recip();

    let straddles_zero = |ta, fa| {
        Zip::from(&tensor.lb)
            .and(&tensor.ub)
            .and(&ta)
            .and(&fa)
            .map_collect(|&x_lb, &x_ub, &t, &f| if (x_lb <= 0.0) & (x_ub >= 0.0) { t } else { f })
    };

    let neg_inf = ArrayD::from_elem(tensor.shape(), f64::NEG_INFINITY);
    let inf = ArrayD::from_elem(tensor.shape(), f64::INFINITY);

    let lb = straddles_zero(neg_inf, lb_safe);
    let ub = straddles_zero(inf, ub_safe);

    Ok(IBPTensor::new(lb, ub))
}

pub(crate) fn ibp_where(
    condition: &IBPTensor,
    x: &IBPTensor,
    y: &IBPTensor,
) -> Result<IBPTensor, EvalError> {
    let lb;
    let ub;

    if condition.is_point() {
        lb = where_(&condition.lb, &x.lb, &y.lb)?;
        ub = where_(&condition.ub, &x.ub, &y.ub)?;
        return Ok(IBPTensor::new(lb, ub));
    }

    let mut ll = where_(&condition.lb, &x.lb, &y.lb)?;
    let mut lu = where_(&condition.lb, &x.ub, &y.ub)?;
    let rl = where_(&condition.ub, &x.lb, &y.lb)?;
    let ru = where_(&condition.ub, &x.ub, &y.ub)?;

    lb = Zip::from(&mut ll)
        .and(&rl)
        .map_collect(|&mut a, &b| a.min(b));
    ub = Zip::from(&mut lu)
        .and(&ru)
        .map_collect(|&mut a, &b| a.max(b));

    Ok(IBPTensor::new(lb, ub))
}

pub(crate) fn ibp_gelu(tensor: &IBPTensor) -> Result<IBPTensor, EvalError> {
    let yl = gelu(&tensor.lb);
    let yr = gelu(&tensor.ub);

    const GELU_ARG_MIN: f64 = -0.751791524693564;
    const GELU_MIN: f64 = -0.16997120747990366;

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

    Ok(IBPTensor::new(lb, ub))
}
