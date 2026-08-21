use std::ops::{Add, Mul};

use ndarray::ArrayD;

use crate::mininn::Value;

// ======================
// BOUNDS
// ======================

// interval bounds
#[derive(Debug, Clone)]
pub struct Bound {
    pub lb: f64,
    pub ub: f64,
    pub is_point: bool,
}

impl Mul for Bound {
    type Output = Bound;

    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.lb * rhs.lb;
        let b = self.lb * rhs.ub;
        let c = self.ub * rhs.lb;
        let d = self.ub * rhs.ub;

        let min = a.min(b).min(c).min(d);
        let max = a.max(b).max(c).max(d);

        Bound {
            lb: min,
            ub: max,
            is_point: self.is_point && rhs.is_point,
        }
    }
}

impl Add for Bound {
    type Output = Bound;

    fn add(self, rhs: Self) -> Self::Output {
        Bound {
            lb: self.lb + rhs.lb,
            ub: self.ub + rhs.ub,
            is_point: self.is_point && rhs.is_point,
        }
    }
}

impl From<f64> for Bound {
    fn from(value: f64) -> Self {
        Bound {
            lb: value,
            ub: value,
            is_point: true,
        }
    }
}

impl Value for Bound {}

// ======================
// IBPTensor
// ======================
pub type IBPTensor = ArrayD<Bound>;

// /// An interval tensor value in the `ibp` interpreter.
// #[derive(Debug, Clone)]
// pub struct IBPTensor {
//     inner: ArrayD<Bound>,
// }
//
// impl From<f64> for IBPTensor {
//     fn from(value: f64) -> Self {
//         IBPTensor {
//             inner: arr0(Bound::from(value)).into_dyn(),
//         }
//     }
// }
//
// impl Mul for IBPTensor {
//     type Output = IBPTensor;
//
//     fn mul(self, rhs: Self) -> Self::Output {
//         IBPTensor {
//             inner: self.inner * rhs.inner,
//         }
//     }
// }
//
// impl Add for IBPTensor {
//     type Output = IBPTensor;
//
//     fn add(self, rhs: Self) -> Self::Output {
//         IBPTensor {
//             inner: self.inner + rhs.inner,
//         }
//     }
// }
//
