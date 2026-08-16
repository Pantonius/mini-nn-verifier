use std::collections::BTreeMap;

use ndarray::{ArrayD, IxDyn};

use crate::{
    interpreters::EvalError,
    mininn::{Atom, AtomKind, Value},
};

/// Generic mapping of variable names to *something*.
/// For example: Env<f64> maps variable names to concrete floating-point values
pub struct Env<T: Value> {
    inner: BTreeMap<String, ArrayD<T>>,
}
impl<T: Value> Env<T> {
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&ArrayD<T>> {
        self.inner.get(key)
    }

    pub fn resolve(&self, atom: &Atom) -> Result<ArrayD<T>, EvalError> {
        match &atom.kind {
            AtomKind::Const(data) => {
                let converted: Vec<T> = data.iter().map(|&x| T::from(x)).collect();
                ArrayD::from_shape_vec(IxDyn(&atom.shape), converted)
                    .map_err(|e| EvalError::Eval(format!("const {}: {e}", atom.name)))
            }
            AtomKind::Var => self
                .get(&atom.name)
                .cloned()
                .ok_or_else(|| EvalError::Eval(format!("undefined variable '{}'", atom.name))),
        }
    }

    pub fn insert(&mut self, key: String, value: ArrayD<T>) {
        self.inner.insert(key, value);
    }

    pub fn update(&mut self, key: String, new_value: ArrayD<T>) -> bool {
        match self.inner.get_mut(&key) {
            Some(v) => {
                *v = new_value;
                true
            }
            None => false,
        }
    }
}
