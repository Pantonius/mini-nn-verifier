use std::{cmp::Ordering, collections::BinaryHeap};

use ndarray::{ArrayD, Zip};

use crate::{
    interpreters::{
        EvalError, Interpreter,
        bounds::{
            affine::beta_crown::BetaCrownInterpreter,
            bab::{BaBConfig, BaBResult, input_splitting_bab, uniform_split},
            ibp::IBPInterpreter,
            ibp_util::IBPTensor,
            lbp_util::AffineBounds,
        },
        concrete::{eval::EvalInterpreter, eval_util::Tensor},
    },
    mininn::{ComputeGraph, Env, Primitive, Value},
};

struct Branch {
    lb: f64,
    // ub: f64,
    /// relu splits
    splits: Env<Tensor>,
    /// alpha params
    a_params: Env<Tensor>,
    /// beta params
    b_params: Env<Tensor>,
    /// Per-ReLU BaBSR branching scores from this branch's bounding pass.
    scores: Env<Tensor>,
    affine_lb: AffineBounds<Tensor>,
}

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

fn flatten_splits(env: &Env<Tensor>) -> Vec<f64> {
    env.iter().flat_map(|(_, t)| t.iter().copied()).collect()
}

fn unflatten_splits(flat: &[f64], template: &Env<Tensor>) -> Env<Tensor> {
    let mut out = Env::new();
    let mut offset = 0;
    for (name, t) in template.iter() {
        let n: usize = t.shape().iter().product();
        let chunk = flat[offset..offset + n].to_vec();
        offset += n;
        out.insert(
            name.clone(),
            Tensor::from(ArrayD::from_shape_vec(ndarray::IxDyn(t.shape()), chunk).unwrap()),
        );
    }
    out
}

pub fn split_smart(splits: &Env<Tensor>, relu_scores: &Env<Tensor>) -> Vec<Env<Tensor>> {
    let split_vec = flatten_splits(splits);
    let score_vec = flatten_splits(relu_scores);
    debug_assert_eq!(split_vec.len(), score_vec.len());

    // Most promising undecided neuron (first-wins on ties, matching argmax).
    let mut best: Option<usize> = None;
    let mut best_score = f64::NEG_INFINITY;
    for (i, (&s, &score)) in split_vec.iter().zip(&score_vec).enumerate() {
        if s == 0.0 && (best.is_none() || score > best_score) {
            best = Some(i);
            best_score = score;
        }
    }

    let Some(best) = best else {
        // Nothing left undecided — no split to make.
        return vec![splits.clone()];
    };

    let mut left = split_vec.clone(); // chosen ReLU forced active
    let mut right = split_vec; // chosen ReLU forced inactive
    left[best] = 1.0;
    right[best] = -1.0;

    vec![
        unflatten_splits(&left, splits),
        unflatten_splits(&right, splits),
    ]
}

/// Returns false if `splits` are structurally infeasible given the original input domain.
///
/// Runs a forward IBP pass that propagates split-tightened ReLU bounds through the graph.
/// At each decided ReLU:
///   split +1 (forced active):   requires pre_ub >= 0; tightens output to [max(pre_lb,0), pre_ub]
///   split -1 (forced inactive): requires pre_lb <= 0; tightens output to [0, 0]
///
/// Compound infeasibility (e.g. split A forces split B's pre-activation entirely negative)
/// is detected because the tightened bounds propagate forward.
fn is_feasible(graph: &ComputeGraph, in_bounds: &IBPTensor, splits: &Env<Tensor>) -> bool {
    let mut bounds: Env<IBPTensor> = Env::new();
    bounds.insert(graph.invars[0].name.clone(), in_bounds.clone());

    for eqn in &graph.equations {
        let out = if let Primitive::Relu(operand) = &eqn.primitive {
            let pre = match bounds.resolve(operand) {
                Ok(b) => b,
                Err(_) => return false,
            };
            if let Some(s) = splits.get(&eqn.outvar.name) {
                // Check element-wise feasibility with the split constraint.
                let feasible = Zip::from(&pre.lb)
                    .and(&pre.ub)
                    .and(s)
                    .all(|&pl, &pu, &split| {
                        if split > 0.0 {
                            pu >= 0.0
                        }
                        // forced active: ub must be reachable
                        else if split < 0.0 {
                            pl <= 0.0
                        }
                        // forced inactive: lb must be reachable
                        else {
                            true
                        }
                    });
                if !feasible {
                    return false;
                }
                // Propagate tightened output bounds so downstream layers see sharper intervals.
                let lb: ArrayD<f64> = Zip::from(&pre.lb).and(s).map_collect(|&pl, &split| {
                    if split > 0.0 {
                        pl.max(0.0)
                    } else if split < 0.0 {
                        0.0
                    } else {
                        pl.max(0.0)
                    }
                });
                let ub: ArrayD<f64> = Zip::from(&pre.ub).and(s).map_collect(|&pu, &split| {
                    if split > 0.0 {
                        pu
                    } else if split < 0.0 {
                        0.0
                    } else {
                        pu.max(0.0)
                    }
                });
                IBPTensor::new(lb.into(), ub.into())
            } else {
                pre.relu()
            }
        } else {
            match IBPInterpreter::process_primitive(&eqn.primitive, &bounds) {
                Ok(b) => b,
                Err(_) => return false,
            }
        };
        bounds.insert(eqn.outvar.name.clone(), out);
    }

    true
}

pub fn node_splitting_bab<S: Fn(&Env<Tensor>, &Env<Tensor>) -> Vec<Env<Tensor>>>(
    graph: &ComputeGraph,
    inputs: &Vec<IBPTensor>,
    split: S,
    config: BaBConfig,
) -> Result<BaBResult, EvalError> {
    let in_bounds = inputs[0].clone();

    // --- Forward Pass ---
    let mut var_bounds = Env::new();
    var_bounds.insert(graph.invars[0].name.clone(), in_bounds.clone());
    for eqn in &graph.equations {
        let out = IBPInterpreter::process_primitive(&eqn.primitive, &var_bounds)?;
        var_bounds.insert(eqn.outvar.name.clone(), out);
    }

    // --- Init Split ---
    let mut splits = Env::new();
    for eqn in &graph.equations {
        if let Primitive::Relu(xa) = &eqn.primitive {
            let x = var_bounds.resolve(xa)?;
            splits.insert(
                eqn.outvar.name.clone(),
                Tensor::from(ArrayD::zeros(ndarray::IxDyn(x.shape()))),
            );
        }
    }

    // Bound one subproblem: run beta-CROWN and concretize
    let bound = |splits: &Env<Tensor>,
                 parent_split_var: Option<String>,
                 warmstart: Option<(Env<Tensor>, Env<Tensor>)>|
     -> Result<Branch, EvalError> {
        let (affine_lb, a_params_lb, b_params_lb, scores_lb) = BetaCrownInterpreter::run(
            graph,
            &var_bounds,
            splits,
            parent_split_var.clone(),
            warmstart.clone(),
        )?;

        let lb = affine_lb
            .concretize(&in_bounds.lb, &in_bounds.ub)
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        // let neg_graph = try_trace_graph(graph, None, |env| -> Result<Tracer, EvalError> {
        //     let out = env.resolve(&graph.outvars[0])?;
        //     Ok(-out)
        // })?;
        // let (affine_lb_neg, a_params_lb_neg, b_params_lb_neg, scores_lb_neg) =
        //     BetaCrownInterpreter::run(
        //         &neg_graph,
        //         &var_bounds,
        //         &splits,
        //         parent_split_var,
        //         warmstart,
        //     )?;
        //
        // let affine_ub = AffineBounds {
        //     weights: -affine_lb_neg.weights,
        //     biases: -affine_lb_neg.biases,
        // };
        //
        // let ub = affine_ub
        //     .concretize(&in_bounds.lb, &in_bounds.ub)
        //     .iter()
        //     .copied()
        //     .fold(f64::INFINITY, f64::max);

        Ok(Branch {
            lb,
            // ub,
            splits: splits.clone(),
            a_params: a_params_lb,
            b_params: b_params_lb,
            scores: scores_lb,
            affine_lb,
        })
    };

    let try_falsify = |branch: &Branch| -> Result<Option<Vec<Tensor>>, EvalError> {
        let x = branch.affine_lb.argmin_corner(&in_bounds.lb, &in_bounds.ub);
        let out = EvalInterpreter::run(graph, &vec![x.clone()])?;
        let min = out
            .iter()
            .flat_map(|t| t.iter().copied())
            .fold(f64::INFINITY, f64::min);
        Ok((min < 0.0).then(|| vec![x]))
    };

    let root = bound(&splits, None, None)?;
    if let Some(cex) = try_falsify(&root)? {
        return Ok(BaBResult::Unsafe(cex));
    }
    let mut heap: BinaryHeap<Branch> = BinaryHeap::new();
    heap.push(root);

    // main BaB
    for _ in 0..config.max_iters {
        let Some(branch) = heap.pop() else {
            return Ok(BaBResult::Safe);
        };

        // Smallest lb first, so if the best open branch already clears 0, all do.
        if branch.lb >= 0.0 {
            return Ok(BaBResult::Safe);
        }

        let children = split(&branch.splits, &branch.scores);

        if children.is_empty() || all_decided(&branch.splits) {
            // try input_splitting
            return input_splitting_bab(graph, inputs, uniform_split, config);
        }

        for child_splits in children {
            if !is_feasible(graph, &in_bounds, &child_splits) {
                continue;
            }
            // The single neuron this child newly forces (0 → ±1) relative to its
            // parent — beta-CROWN starts its backward pass there.
            // let split_var = child_splits.iter().find_map(|(name, t)| {
            //     let parent = branch.splits.get(name)?;
            //     let changed = t
            //         .iter()
            //         .zip(parent.iter())
            //         .any(|(&c, &p)| p == 0.0 && c != 0.0);
            //     changed.then(|| name.clone())
            // });
            let warmstart = Some((branch.a_params.clone(), branch.b_params.clone()));
            let child = bound(&child_splits, None, warmstart)?;

            if let Some(cex) = try_falsify(&child)? {
                return Ok(BaBResult::Unsafe(cex));
            }

            if child.lb < 0.0 {
                heap.push(child);
            }
        }
    }

    Ok(BaBResult::Undecided)
}

/// True when no ReLU in `splits` is left undecided (every entry is ±1), meaning the
/// branch is a leaf and cannot be refined further.
fn all_decided(splits: &Env<Tensor>) -> bool {
    splits.iter().all(|(_, t)| t.iter().all(|&s| s != 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    fn arr(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from(ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap())
    }

    fn env(pairs: &[(&str, Tensor)]) -> Env<Tensor> {
        let mut e = Env::new();
        for (name, t) in pairs {
            e.insert((*name).to_string(), t.clone());
        }
        e
    }

    fn flat(env: &Env<Tensor>, name: &str) -> Vec<f64> {
        env.get(name).unwrap().iter().copied().collect()
    }

    /// Forks the highest-scoring undecided neuron; every other neuron is untouched.
    #[test]
    fn forks_highest_scoring_undecided() {
        let splits = env(&[("a1", arr(&[0.0, 0.0], &[2])), ("a2", arr(&[0.0], &[1]))]);
        // Flatten order (BTreeMap): a1[0], a1[1], a2[0]. Max score at a1[1].
        let scores = env(&[("a1", arr(&[0.5, 2.0], &[2])), ("a2", arr(&[1.0], &[1]))]);

        let children = split_smart(&splits, &scores);
        assert_eq!(children.len(), 2);

        // Child 0 forces the chosen neuron active (+1), child 1 inactive (-1).
        assert_eq!(flat(&children[0], "a1"), vec![0.0, 1.0]);
        assert_eq!(flat(&children[1], "a1"), vec![0.0, -1.0]);
        // Untouched neurons stay at their prior state in both children.
        assert_eq!(flat(&children[0], "a2"), vec![0.0]);
        assert_eq!(flat(&children[1], "a2"), vec![0.0]);
    }

    /// Already-decided neurons are ineligible even with the highest score.
    #[test]
    fn skips_decided_neurons() {
        // a1[1] already forced to +1 and has the top score, so the next-best
        // undecided (a2[0], score 1.0 > a1[0]'s 0.5) must be chosen instead.
        let splits = env(&[("a1", arr(&[0.0, 1.0], &[2])), ("a2", arr(&[0.0], &[1]))]);
        let scores = env(&[("a1", arr(&[0.5, 9.0], &[2])), ("a2", arr(&[1.0], &[1]))]);

        let children = split_smart(&splits, &scores);
        assert_eq!(children.len(), 2);
        assert_eq!(flat(&children[0], "a2"), vec![1.0]);
        assert_eq!(flat(&children[1], "a2"), vec![-1.0]);
        // The pre-decided neuron is preserved in both children.
        assert_eq!(flat(&children[0], "a1"), vec![0.0, 1.0]);
        assert_eq!(flat(&children[1], "a1"), vec![0.0, 1.0]);
    }

    /// With nothing left undecided, a single unchanged child is returned.
    #[test]
    fn all_decided_returns_single_child() {
        let splits = env(&[("a1", arr(&[1.0, -1.0], &[2]))]);
        let scores = env(&[("a1", arr(&[9.0, 9.0], &[2]))]);

        let children = split_smart(&splits, &scores);
        assert_eq!(children.len(), 1);
        assert_eq!(flat(&children[0], "a1"), vec![1.0, -1.0]);
    }

    // ---- is_feasible tests ----

    // Network: out = relu( relu(x) - 2 )
    //   x ∈ [-1, 1]
    //   first ReLU  pre-activation: [-1, 1]     → output [0, 1]
    //   second ReLU pre-activation: [0,1]-2 = [-2,-1] → output [0, 0]
    fn two_relu_net() -> ComputeGraph {
        use crate::interpreters::compute_graph::trace;
        use crate::interpreters::compute_graph::tracer::Tracer;
        trace([("x".to_string(), vec![1])], |env| {
            let x = env.get("x").unwrap().clone();
            let h = x.relu();
            let z = h + Tracer::from(arr(&[-2.0], &[1]));
            z.relu()
        })
    }

    fn relu_names(graph: &ComputeGraph) -> Vec<String> {
        graph
            .equations
            .iter()
            .filter(|e| matches!(e.primitive, Primitive::Relu(_)))
            .map(|e| e.outvar.name.clone())
            .collect()
    }

    /// No splits → always feasible.
    #[test]
    fn no_splits_always_feasible() {
        let graph = two_relu_net();
        let in_bounds = IBPTensor::new(arr(&[-1.0], &[1]), arr(&[1.0], &[1]));
        assert!(is_feasible(&graph, &in_bounds, &Env::new()));
    }

    /// first +1, second -1:
    ///   first pre_ub = 1 ≥ 0 ✓, tightened output [0, 1]
    ///   second pre bounds [-2, -1], pre_lb = -2 ≤ 0 ✓
    #[test]
    fn feasible_splits_accepted() {
        let graph = two_relu_net();
        let names = relu_names(&graph);
        let in_bounds = IBPTensor::new(arr(&[-1.0], &[1]), arr(&[1.0], &[1]));
        let splits = env(&[
            (&names[0], arr(&[1.0], &[1])),
            (&names[1], arr(&[-1.0], &[1])),
        ]);
        assert!(is_feasible(&graph, &in_bounds, &splits));
    }

    /// first +1, second +1:
    ///   first tightened output [0, 1]
    ///   second pre bounds [-2, -1], pre_ub = -1 < 0 → INFEASIBLE
    #[test]
    fn compound_infeasible_splits_rejected() {
        let graph = two_relu_net();
        let names = relu_names(&graph);
        let in_bounds = IBPTensor::new(arr(&[-1.0], &[1]), arr(&[1.0], &[1]));
        let splits = env(&[
            (&names[0], arr(&[1.0], &[1])),
            (&names[1], arr(&[1.0], &[1])),
        ]);
        assert!(!is_feasible(&graph, &in_bounds, &splits));
    }

    /// x ∈ [0.5, 1]: first relu pre_lb = 0.5 > 0, forcing it inactive → INFEASIBLE
    #[test]
    fn directly_infeasible_split_rejected() {
        let graph = two_relu_net();
        let names = relu_names(&graph);
        let in_bounds = IBPTensor::new(arr(&[0.5], &[1]), arr(&[1.0], &[1]));
        let splits = env(&[(&names[0], arr(&[-1.0], &[1]))]);
        assert!(!is_feasible(&graph, &in_bounds, &splits));
    }
}
