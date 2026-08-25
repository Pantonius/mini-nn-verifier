use std::ops::{Add, Mul};

use crate::{interpreters::{bounds::ibp_util::IBPTensor, concrete::eval_util::Tensor}, mininn::Value};

pub fn lbp_inner(w: &Tensor, x: &Tensor) -> Tensor {
    let product = w * x;
    let axes: Vec<isize> = (0..product.ndim() as isize).collect();
    product.reduce_sum(&axes)
}

// ================================
// ABPTensor (Affine Bounds)
// ================================
#[derive(Debug, Clone)]
pub struct ABPTensor {
    pub weights: Tensor,
    pub biases: Tensor,
}

impl ABPTensor {
    pub fn concretize(&self, in_bounds: &IBPTensor) -> Tensor {
        self.biases.clone() + lbp_inner(&self.weights.pos_part(), &in_bounds.lb) + lbp_inner(&self.weights.neg_part(), &in_bounds.ub)
    }
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
