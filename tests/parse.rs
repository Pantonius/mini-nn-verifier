use std::path::Path;

use mininn_verifier::mininn::{AtomKind, load_mininn};

/// The shallow circles networks all share the same topology:
///   a -> dot(A) -> add(B) -> relu -> dot(C) -> add(D) -> f
/// differing only in the hidden width.
fn check_shallow_circles(file: &str, hidden: usize) {
    let path = Path::new("tests/common").join(file);
    let graph = load_mininn(&path).unwrap_or_else(|e| panic!("load {file}: {e}"));

    // one input `a[2]`, one output `f[1]`
    assert_eq!(graph.invars.len(), 1);
    assert_eq!(graph.invars[0].name, "a");
    assert_eq!(graph.invars[0].shape, vec![2]);
    assert!(matches!(graph.invars[0].kind, AtomKind::Var));

    assert_eq!(graph.outvars.len(), 1);
    assert_eq!(graph.outvars[0].name, "f");
    assert_eq!(graph.outvars[0].shape, vec![1]);

    // five equations in a fixed order
    let prims: Vec<&str> = graph.equations.iter().map(|e| e.primitive.name()).collect();
    assert_eq!(prims, vec!["dot", "add", "relu", "dot", "add"]);

    // first dot: A[hidden, 2] @ a[2] -> b[hidden]
    let dot = &graph.equations[0];
    assert_eq!(dot.outvar.shape, vec![hidden]);
    let operands = dot.primitive.operands();
    let weight = operands[0];
    assert_eq!(weight.name, "A");
    assert_eq!(weight.shape, vec![hidden, 2]);
    match &weight.kind {
        AtomKind::Const(data) => assert_eq!(data.len(), hidden * 2),
        AtomKind::Var => panic!("A should be a constant"),
    }
}

#[test]
fn parses_shallow_5() {
    check_shallow_circles("circles_shallow_5.mininn", 5);
}

#[test]
fn parses_shallow_32() {
    check_shallow_circles("circles_shallow_32.mininn", 32);
}

#[test]
fn parses_shallow_64() {
    check_shallow_circles("circles_shallow_64.mininn", 64);
}
