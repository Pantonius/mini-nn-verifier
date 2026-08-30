use crate::mininn::Value;

pub fn lbp_inner<T: Value>(w: &T, x: &T) -> T {
    let product = w.clone() * x.clone();
    let axes: Vec<isize> = (0..product.ndim() as isize).collect();
    product.reduce_sum(&axes)
}

// ================================
// ABPTensor (Affine Bounds)
// ================================
#[derive(Debug, Clone)]
pub struct ABPTensor<T: Value> {
    pub weights: T,
    pub biases: T,
}

impl<T: Value> ABPTensor<T> {
    /// Concretize the affine bound given concrete lower/upper bounds on the input.
    pub fn concretize(&self, lb: &T, ub: &T) -> T {
        let pos_w = self.weights.relu();
        let neg_w = self.weights.clone() - pos_w.clone();
        self.biases.clone() + lbp_inner(&pos_w, lb) + lbp_inner(&neg_w, ub)
    }
}
