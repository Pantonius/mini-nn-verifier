use core::f64;
use std::{cmp::Ordering, collections::BinaryHeap};

use ndarray::Zip;

use crate::{
    interpreters::{
        EvalError, Interpreter,
        bounds::{
            bab::{BaBConfig, BaBResult},
            interval::{
                ibp_batched::IBPBatchedInterpreter,
                ibp_util::{IBPBatchedTensor, IBPTensor},
            },
        },
        concrete::eval_util::Tensor,
    },
    mininn::{ComputeGraph, Value},
};

struct Branch {
    lb: f64,
    inputs: Vec<IBPTensor>,
}

// Order so that `BinaryHeap` (a max-heap) yields the *smallest* priority first.
// `total_cmp` gives a total order over f64, so NaN can never poison the heap.
impl Ord for Branch {
    fn cmp(&self, other: &Self) -> Ordering {
        other.lb.total_cmp(&self.lb)
    }
}
impl PartialOrd for Branch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Branch {
    fn eq(&self, other: &Self) -> bool {
        self.lb.total_cmp(&other.lb) == Ordering::Equal
    }
}
impl Eq for Branch {}

fn branch_width(branch: &Branch) -> f64 {
    branch.inputs.iter().fold(f64::NEG_INFINITY, |a, t| {
        let max_width = Zip::from(&t.lb)
            .and(&t.ub)
            .fold(f64::NEG_INFINITY, |b, &l, &u| b.max(u - l));
        a.max(max_width)
    })
}
// --------------------
// Split Functions
// --------------------

pub fn uniform_split(inputs: &Vec<IBPTensor>) -> Vec<Vec<IBPTensor>> {
    // NOTE: assumes exactly one box input; all others are points.
    // TODO If multiple box inputs were supported, we should search across all of them for the widest dimension.

    let (split_idx, split_tensor) = inputs
        .iter()
        .enumerate()
        .find(|(_, t)| t.lb != t.ub)
        .expect("no interval input to split");

    // take the difference between the two...
    let range = &split_tensor.ub - &split_tensor.lb;

    // ... find the index with the biggest difference
    let (flat_idx, _) = range
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    // ... then unflatten the index into the actual shape
    let ix = ndarray::IxDyn(&flat_to_multi(flat_idx, split_tensor.lb.shape()));

    // compute the midpoint for split
    let mid = (split_tensor.lb[ix.clone()] + split_tensor.ub[ix.clone()]) / 2.0;

    // ... and construct new ub for left partition, as well as new lb for right partition (the
    // midpoint at the widest index)
    let mut left_ub_arr = split_tensor.ub.clone().into_inner();
    left_ub_arr[ix.clone()] = mid;
    let left_ub: Tensor = left_ub_arr.into();

    let mut right_lb_arr = split_tensor.lb.clone().into_inner();
    right_lb_arr[ix] = mid;
    let right_lb: Tensor = right_lb_arr.into();

    // construct the left and right partition
    let left_child = {
        let mut child = inputs.clone();
        child[split_idx] = IBPTensor::new(split_tensor.lb.clone(), left_ub);
        child
    };
    let right_child = {
        let mut child = inputs.clone();
        child[split_idx] = IBPTensor::new(right_lb, split_tensor.ub.clone());
        child
    };

    vec![left_child, right_child]
}

// TODO smart branching

fn flat_to_multi(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = flat;
    let mut multi = vec![0usize; shape.len()];
    for (i, &s) in shape.iter().enumerate().rev() {
        multi[i] = idx % s;
        idx /= s;
    }
    multi
}

// --------------------
// BaB Algorithms
// --------------------

pub fn input_splitting_bab<S: Fn(&Vec<IBPTensor>) -> Vec<Vec<IBPTensor>>>(
    graph: &ComputeGraph,
    inputs: &Vec<IBPTensor>,
    split: S,
    config: BaBConfig,
) -> Result<BaBResult, EvalError> {
    assert!(config.batch_size >= 1);

    // priority heap (lower lb first)
    let mut heap: BinaryHeap<Branch> = BinaryHeap::new();
    heap.push(Branch {
        lb: f64::NEG_INFINITY,
        inputs: inputs.clone(),
    });

    // max iterations
    for _ in 0..config.max_iters {
        if heap.is_empty() {
            return Ok(BaBResult::Safe);
        }

        let k = config.batch_size.min(heap.len());

        // choose k branches
        let k_branches: Vec<Branch> = (0..k).map(|_| heap.pop().unwrap()).collect();

        // stack per input position -> one IBPBatchedTensor per network input.
        let ibp_inputs: Vec<&Vec<IBPTensor>> = k_branches.iter().map(|b| &b.inputs).collect();
        let batched_inputs: Vec<IBPBatchedTensor> = (0..inputs.len())
            .map(|i| IBPBatchedTensor::stack_input(&ibp_inputs, i))
            .collect();

        // single batched IBP forward pass
        let batched_outputs = IBPBatchedInterpreter::run(graph, &batched_inputs)?;

        // Unstack and classify each output
        for (i, branch) in k_branches.iter().enumerate() {
            let out: Vec<IBPTensor> = batched_outputs.iter().map(|t| t.get(i)).collect();

            // ub < 0: definite violation of property
            if out.iter().any(|t| t.ub.iter().any(|&v| v < 0.0)) {
                let cex = branch
                    .inputs
                    .iter()
                    // any point in [lb, ub] is fine
                    .map(|t| t.ub.clone())
                    .collect();
                return Ok(BaBResult::Unsafe(cex));
            }

            // lb < 0 <= ub: undecided; keep splitting
            if out.iter().any(|t| t.lb.iter().any(|&v| v < 0.0)) {
                // check if maximal dimension (branch width) is below min_width threshold
                if branch_width(&branch) < config.min_width {
                    // ... if so, give up and declare the satisfiability as undecided
                    // (Since one undecided branch suffices for an undecided verdict, just return
                    // right away without considering other branches)
                    return Ok(BaBResult::Undecided);
                }

                // split
                let children = split(&branch.inputs);
                let child_lb = out
                    .iter()
                    .flat_map(|t| t.lb.iter().copied())
                    .fold(f64::INFINITY, f64::min);

                // and push to priority heap
                for child in children {
                    heap.push(Branch {
                        lb: child_lb,
                        inputs: child,
                    });
                }
            }
            // else lb >= 0: this branch is safe, discard.
        }
    }

    Ok(BaBResult::Undecided)
}
