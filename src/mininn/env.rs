use std::collections::BTreeMap;

use crate::{
    interpreters::EvalError,
    mininn::{Atom, AtomKind, Value},
};

/// Generic mapping of variable names to *something*.
/// For example: Env<f64> maps variable names to concrete floating-point values
pub struct Env<T: Value> {
    inner: BTreeMap<String, T>,
}
impl<T: Value> Env<T> {
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&T> {
        self.inner.get(key)
    }

    pub fn insert(&mut self, key: String, value: T) {
        self.inner.insert(key, value);
    }

    pub fn update(&mut self, key: &String, new_value: T) -> bool {
        match self.inner.get_mut(key) {
            Some(v) => {
                *v = new_value;
                true
            }
            None => false,
        }
    }
    pub fn resolve(&self, atom: &Atom) -> Result<T, EvalError> {
        match &atom.kind {
            AtomKind::Const(data) => Ok(T::from_tensor(&data)),
            AtomKind::Var => self
                .get(&atom.name)
                .cloned()
                .ok_or_else(|| EvalError::Eval(format!("undefined variable '{}'", atom.name))),
        }
    }
}
