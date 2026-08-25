use std::ops::{Add, Mul};

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
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            todo!()
        }
    }
}

impl Add for ABPTensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            weights: self.weights + rhs.weights,
            biases: self.biases + rhs.biases,
        }
    }
}

impl Value for ABPTensor {}

// ================================
// TODO ABPTensor Batched
// ================================
