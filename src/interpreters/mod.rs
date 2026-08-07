use crate::core::{Primitive, Value};

pub trait Interpreter<T: Value> {
    fn process_primitive(&mut self, primitive: Primitive<T>) -> Box<T>;
}

pub struct IStack<T: Value> {
    inner: Vec<Box<dyn Interpreter<T>>>,
}
impl<T: Value> IStack<T> {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }
    pub fn push(&mut self, interpreter: Box<dyn Interpreter<T>>) {
        self.inner
    }
}

mod cg;
mod eval;
mod ibp;

pub use cg::*;
pub use eval::*;
pub use ibp::*;
