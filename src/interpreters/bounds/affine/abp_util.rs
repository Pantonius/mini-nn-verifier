use std::ops::{Add, Mul};

use ndarray::ArrayD;

use crate::{interpreters::concrete::eval_util::Tensor, mininn::Value};

// ================================
// ABPTensor (Affine Bounds)
// ================================
#[derive(Debug, Clone)]
pub struct ABPTensor {
    pub weights: Tensor,
    pub biases: Tensor,
}

impl Mul for ABPTensor {
    type Output = ABPTensor;

    fn mul(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Add for ABPTensor {
    type Output = ABPTensor;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Value for ABPTensor {
    fn from_tensor(tensor: &ArrayD<f64>) -> Self {
        todo!()
    }
}

// ================================
// TODO ABPTensor Batched
// ================================

// ================================
// ABP Helper Functions
// ================================
