use crate::{
    interpreters::{EvalError, IBPInterpreter, IBPTensor, Interpreter, Tensor},
    mininn::ComputeGraph,
};

pub enum BaBResult {
    Safe,
    Unsafe(Vec<Tensor>),
}

fn pick_worst_lb(branches: &Vec<(f64, Vec<IBPTensor>)>) -> usize {
    branches
        .iter()
        .enumerate()
        .min_by(|(_, x), (_, y)| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap()
}

pub fn uniform_split(inputs: &Vec<IBPTensor>) -> Vec<Vec<IBPTensor>> {
    // skip points
    // NOTE: assumes exactly one box input; all others are points.
    // TODO If multiple box inputs were supported, we'd search across all of them for the widest dimension.
    let (split_idx, split_tensor) = inputs
        .iter()
        .enumerate()
        .find(|(_, t)| t.lb != t.ub)
        .expect("no interval input to split");

    // range of the box
    let range = &split_tensor.ub - &split_tensor.lb;

    // flat index of the widest dimension
    let (flat_idx, _) = range
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    // to proper index
    let ix = ndarray::IxDyn(&flat_to_multi(flat_idx, split_tensor.lb.shape()));

    let mid = (split_tensor.lb[ix.clone()] + split_tensor.ub[ix.clone()]) / 2.0;

    let mut left_ub = split_tensor.ub.clone();
    left_ub[ix.clone()] = mid;

    let mut right_lb = split_tensor.lb.clone();
    right_lb[ix] = mid;

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

fn flat_to_multi(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = flat;
    let mut multi = vec![0usize; shape.len()];
    for (i, &s) in shape.iter().enumerate().rev() {
        multi[i] = idx % s;
        idx /= s;
    }
    multi
}

pub fn input_splitting_bab<S: Fn(&Vec<IBPTensor>) -> Vec<Vec<IBPTensor>>>(
    graph: &ComputeGraph,
    inputs: &Vec<IBPTensor>,
    split: S,
) -> Result<BaBResult, EvalError> {
    // list of pairs (lower_bound, inputs)
    let mut branches = vec![(f64::NEG_INFINITY, inputs.clone())];

    while !branches.is_empty() {
        let branch_i = pick_worst_lb(&branches);
        let (_, branch) = branches.remove(branch_i);
        let children = split(&branch);

        for cb in children {
            let child_bounds = IBPInterpreter::run(graph, &cb)?;

            if child_bounds.iter().any(|t| t.ub.iter().any(|&v| v < 0.0)) {
                // counter-example
                let cex = cb.iter().map(|t| (&t.lb + &t.ub) / 2.0).collect();
                return Ok(BaBResult::Unsafe(cex));
            }

            if child_bounds.iter().any(|t| t.lb.iter().any(|&v| v < 0.0)) {
                // lb < 0 but ub >= 0: bounds too loose to decide → keep splitting
                let child_lb = child_bounds
                    .iter()
                    .flat_map(|t| t.lb.iter().copied())
                    .fold(f64::INFINITY, f64::min);
                branches.push((child_lb, cb));
            }

            // otherwise safe (branch entirely > 0)
        }
    }

    return Ok(BaBResult::Safe);
}
