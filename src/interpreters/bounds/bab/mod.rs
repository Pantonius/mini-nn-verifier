mod input_splitting;

pub use input_splitting::{input_splitting_bab, uniform_split};

mod node_splitting;
pub use node_splitting::{node_splitting_bab, split_smart};

use crate::interpreters::concrete::eval_util::Tensor;

pub enum BaBResult {
    Safe,
    Unsafe(Vec<Tensor>),
    Undecided,
}

pub struct BaBConfig {
    /// how many branches to classify per batched IBP forward pass
    batch_size: usize,
    /// minimum width of intervals for further splitting (a formal break-off point)
    min_width: f64,
    /// maximum iterations of branch-and-bound (another formal break-off point)
    max_iters: usize,
}

impl Default for BaBConfig {
    fn default() -> Self {
        Self {
            batch_size: 64,
            min_width: 1e-6,
            max_iters: 1_000_000,
        }
    }
}
