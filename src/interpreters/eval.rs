use std::ops::Add;

use ndarray::{ArrayBase, RawData};

use crate::{
    core::{Primitive, Value},
    interpreters::Interpreter,
};

#[derive(Debug, Clone, Copy)]
struct Array {}

impl Add for Array {
    type Output = Array;
    fn add(self, rhs: Self) -> Self::Output {}
}
impl Value for Array {}

struct EvalInterpreter {}

impl<D> Interpreter<ArrayBase<f64, D>> for EvalInterpreter {
    fn process_primitive(&mut self, primitive: Primitive<ArrayBase<f64, D>>) -> Box<dyn Value> {
        use Primitive::*;
        Box::new(match primitive {
            Neg(x) => -x,
            Reciprocal(x) => 1. / x,
            Add(x, y) => x + y,
            Mul(x, y) => x * y,
            Sq(x) => x * x,
            Sqrt(x) => x.sqrt(),
            Ln(x) => x.ln(),
            Exp(x) => x.exp(),
            Dot(A, B) => A.dot(&B),
            ReLU(x) => {
                if x > 0 {
                    x
                } else {
                    0
                }
            }
        })
    }
}
