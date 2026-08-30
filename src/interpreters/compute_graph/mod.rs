pub mod tracer;

use std::{cell::RefCell, rc::Rc};

use crate::mininn::{ComputeGraph, Env};

use tracer::{ComputeGraphBuilder, Tracer};

pub fn trace(
    inputs: impl IntoIterator<Item = (String, Vec<usize>)>,
    f: impl FnOnce(Env<Tracer>) -> Tracer,
) -> ComputeGraph {
    let (builder, env) = build_env(inputs);
    let out = f(env);
    builder.borrow().clone().build(out.atom().clone())
}

pub fn try_trace<E>(
    inputs: impl IntoIterator<Item = (String, Vec<usize>)>,
    f: impl FnOnce(Env<Tracer>) -> Result<Tracer, E>,
) -> Result<ComputeGraph, E> {
    let (builder, env) = build_env(inputs);
    let out = f(env)?;

    Ok(builder.borrow().clone().build(out.atom().clone()))
}

pub fn try_trace_graph<E>(
    graph: &ComputeGraph,
    extra_invars: Option<Vec<(String, Vec<usize>)>>,
    f: impl FnOnce(Env<Tracer>) -> Result<Tracer, E>,
) -> Result<ComputeGraph, E> {
    let mut env = Env::new();

    let builder = match extra_invars {
        // Re-open the existing graph: keep all of its equations (and their names) and
        // expose every existing variable — invars plus equation outputs — as a Tracer,
        // so `f` can reference any of them and simply append new equations. This keeps
        // the returned graph's variable names aligned with bounds/params computed on
        // the original graph.
        None => {
            let builder = Rc::new(RefCell::new(ComputeGraphBuilder::from(graph.clone())));
            for atom in graph
                .invars
                .iter()
                .chain(graph.equations.iter().map(|eqn| &eqn.outvar))
            {
                env.insert(atom.name.clone(), Tracer::new(atom.clone(), builder.clone()));
            }
            builder
        }
        // Build a fresh graph whose only inputs are the supplied extra invars (e.g. the
        // alpha parameters); the original equations are not carried over.
        Some(extras) => {
            let builder = Rc::new(RefCell::new(ComputeGraphBuilder::new()));
            for (name, shape) in extras {
                let atom = builder.borrow_mut().register_invar(name.clone(), shape);
                env.insert(name, Tracer::new(atom, builder.clone()));
            }
            builder
        }
    };

    let out = f(env)?;
    Ok(builder.borrow().clone().build(out.atom().clone()))
}

fn build_env(
    inputs: impl IntoIterator<Item = (String, Vec<usize>)>,
) -> (Rc<RefCell<ComputeGraphBuilder>>, Env<Tracer>) {
    let builder = Rc::new(RefCell::new(ComputeGraphBuilder::new()));
    let mut env = Env::new();
    for (name, shape) in inputs {
        let atom = builder.borrow_mut().register_invar(name.clone(), shape);
        env.insert(name, Tracer::new(atom, builder.clone()));
    }
    (builder, env)
}
