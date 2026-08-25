use std::{
    collections::BTreeMap,
    ops::{Add, Mul, Neg, Sub},
};

use ndarray::ArrayD;

use crate::{
    interpreters::{EvalError, concrete::eval_util::Tensor},
    mininn::{Atom, AtomKind, PaddingOptions, PoolOptions},
};

pub trait Value:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Sized
    + Clone
    + From<ArrayD<f64>>
    + From<Tensor>
{
    fn shape(&self) -> &[usize];
    fn ndim(&self) -> usize;
    fn len(&self) -> usize;

    fn r#where(cond: &Self, x: &Self, y: &Self) -> Result<Self, EvalError>;
    fn moveaxis(&self, src: isize, dst: isize) -> Self;
    fn dot(&self, b: &Self) -> Result<Self, EvalError>;
    fn square(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn reciprocal(&self) -> Self;
    fn reduce_sum(&self, axes: &[isize]) -> Self;
    fn expand_dims(&self, axes: &[isize]) -> Self;
    fn reshape(&self, new_shape: &[isize]) -> Result<Self, EvalError>;
    fn pad(&self, opt: &PaddingOptions) -> Self;
    fn conv(&self, kernel: &Self, stride: isize) -> Result<Self, EvalError>;
    fn pool(&self, opt: &PoolOptions, average: bool) -> Result<Self, EvalError>;
    fn exp(&self) -> Self;
    fn log(&self) -> Self;
    fn relu(&self) -> Self;
    fn leaky_relu(&self, slope: f64) -> Self;
    fn elu(&self, slope: f64) -> Self;
    fn normcdf(&self) -> Self;
    fn gelu(&self) -> Self;
}

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
            AtomKind::Const(data) => Ok(T::from(data.clone())),
            AtomKind::Var => self
                .get(&atom.name)
                .cloned()
                .ok_or_else(|| EvalError::Eval(format!("undefined variable '{}'", atom.name))),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}
