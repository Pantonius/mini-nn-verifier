use std::{
    cell::RefCell,
    ops::{Add, Mul, Neg, Sub},
    rc::Rc,
};

use ndarray::ArrayD;

use crate::{
    interpreters::concrete::eval_util::Tensor,
    mininn::{Atom, AtomKind, ComputeGraph, ConvOptions, Equation, Primitive, Value},
};

#[derive(Debug, Clone)]
pub struct ComputeGraphBuilder {
    n_atoms: usize,
    equations: Vec<Equation>,
}

impl ComputeGraphBuilder {
    pub fn new() -> Self {
        ComputeGraphBuilder {
            n_atoms: 0,
            equations: Vec::new(),
        }
    }

    pub fn build_equation(&mut self, primitive: Primitive, shape: Vec<usize>) -> Atom {
        let outvar = Atom {
            shape,
            kind: AtomKind::Var,
            name: format!("a{}", self.n_atoms),
        };
        self.n_atoms += 1;
        let res = outvar.clone();
        self.equations.push(Equation { primitive, outvar });
        res
    }

    pub fn build(self) -> ComputeGraph {
        assert_ne!(self.equations.len(), 0);

        let invars = self.equations[0]
            .primitive
            .operands()
            .iter()
            .map(|&a| a.clone())
            .collect();
        let outvars = self
            .equations
            .last()
            .unwrap()
            .primitive
            .operands()
            .iter()
            .map(|&a| a.clone())
            .collect();

        ComputeGraph {
            invars,
            outvars,
            equations: self.equations,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tracer {
    atom: Atom,
    builder: Option<Rc<RefCell<ComputeGraphBuilder>>>,
}

impl Tracer {
    pub fn new(atom: Atom, builder: Rc<RefCell<ComputeGraphBuilder>>) -> Self {
        Self {
            atom,
            builder: Some(builder),
        }
    }

    fn pick_builder(
        self_b: Option<Rc<RefCell<ComputeGraphBuilder>>>,
        rhs_b: Option<Rc<RefCell<ComputeGraphBuilder>>>,
    ) -> Rc<RefCell<ComputeGraphBuilder>> {
        self_b
            .or(rhs_b)
            .expect("at least one operand must be a Var tracer")
    }

    fn emit_unary(&self, primitive: Primitive, out_shape: Vec<usize>) -> Self {
        let builder = self
            .builder
            .clone()
            .expect("cannot trace a unary op on a Const tracer");
        let outvar = builder.borrow_mut().build_equation(primitive, out_shape);
        Self {
            atom: outvar,
            builder: Some(builder),
        }
    }
}

impl Add for Tracer {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let shape = self.atom.shape.clone();
        let builder = Self::pick_builder(self.builder, rhs.builder);
        let prim = Primitive::Add(self.atom, rhs.atom);
        let outvar = builder.borrow_mut().build_equation(prim, shape);
        Self {
            atom: outvar,
            builder: Some(builder),
        }
    }
}

impl Mul for Tracer {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let shape = self.atom.shape.clone();
        let builder = Self::pick_builder(self.builder, rhs.builder);
        let prim = Primitive::Mul(self.atom, rhs.atom);
        let outvar = builder.borrow_mut().build_equation(prim, shape);
        Self {
            atom: outvar,
            builder: Some(builder),
        }
    }
}

impl From<Tensor> for Tracer {
    fn from(value: Tensor) -> Self {
        let shape = value.shape().to_vec();
        let atom = Atom {
            name: String::new(),
            shape,
            kind: AtomKind::Const(value),
        };
        Self {
            atom,
            builder: None,
        }
    }
}

impl From<ArrayD<f64>> for Tracer {
    fn from(value: ArrayD<f64>) -> Self {
        let tensor = Tensor::from(value);
        Self::from(tensor)
    }
}

impl Sub for Tracer {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Neg for Tracer {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let shape = self.atom.shape.clone();
        let builder = self.builder.expect("cannot negate a Const tracer");
        let prim = Primitive::Neg(self.atom);
        let outvar = builder.borrow_mut().build_equation(prim, shape);
        Self {
            atom: outvar,
            builder: Some(builder),
        }
    }
}

impl Value for Tracer {
    fn shape(&self) -> &[usize] {
        &self.atom.shape
    }

    fn ndim(&self) -> usize {
        self.atom.shape.len()
    }

    fn len(&self) -> usize {
        self.atom.shape.iter().product()
    }

    fn r#where(cond: &Self, x: &Self, y: &Self) -> Result<Self, crate::interpreters::EvalError> {
        let builder = cond
            .builder
            .clone()
            .or_else(|| x.builder.clone())
            .or_else(|| y.builder.clone())
            .expect("at least one operand must be a Var tracer");
        let shape = x.atom.shape.clone();
        let prim = Primitive::Where(cond.atom.clone(), x.atom.clone(), y.atom.clone());
        let outvar = builder.borrow_mut().build_equation(prim, shape);
        Ok(Self {
            atom: outvar,
            builder: Some(builder),
        })
    }

    fn moveaxis(&self, src: isize, dst: isize) -> Self {
        let ndim = self.atom.shape.len() as isize;
        let src_n = if src < 0 {
            (src + ndim) as usize
        } else {
            src as usize
        };
        let dst_n = if dst < 0 {
            (dst + ndim) as usize
        } else {
            dst as usize
        };
        let mut out_shape = self.atom.shape.clone();
        let dim = out_shape.remove(src_n);
        out_shape.insert(dst_n, dim);
        self.emit_unary(
            Primitive::MoveAxis {
                operand: self.atom.clone(),
                source: src,
                destination: dst,
            },
            out_shape,
        )
    }

    fn dot(&self, b: &Self) -> Result<Self, crate::interpreters::EvalError> {
        // [..., M, K] · [..., K, N] → [..., M, N]
        let ndim = self.atom.shape.len();
        let mut out_shape = self.atom.shape[..ndim - 1].to_vec();
        out_shape.push(*b.atom.shape.last().expect("dot: rhs must be at least 1-D"));
        let builder = Self::pick_builder(self.builder.clone(), b.builder.clone());
        let prim = Primitive::Dot(self.atom.clone(), b.atom.clone());
        let outvar = builder.borrow_mut().build_equation(prim, out_shape);
        Ok(Self {
            atom: outvar,
            builder: Some(builder),
        })
    }

    fn square(&self) -> Self {
        self.emit_unary(
            Primitive::Square(self.atom.clone()),
            self.atom.shape.clone(),
        )
    }

    fn sqrt(&self) -> Self {
        self.emit_unary(Primitive::Sqrt(self.atom.clone()), self.atom.shape.clone())
    }

    fn reciprocal(&self) -> Self {
        self.emit_unary(
            Primitive::Reciprocal(self.atom.clone()),
            self.atom.shape.clone(),
        )
    }

    fn reduce_sum(&self, axes: &[isize]) -> Self {
        let ndim = self.atom.shape.len() as isize;
        let reduced: std::collections::HashSet<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + ndim) as usize
                } else {
                    a as usize
                }
            })
            .collect();
        let out_shape: Vec<usize> = self
            .atom
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduced.contains(i))
            .map(|(_, &s)| s)
            .collect();
        self.emit_unary(
            Primitive::ReduceSum {
                operand: self.atom.clone(),
                axes: axes.to_vec(),
            },
            out_shape,
        )
    }

    fn expand_dims(&self, axes: &[isize]) -> Self {
        let out_ndim = self.atom.shape.len() + axes.len();
        let normed: std::collections::HashSet<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + out_ndim as isize) as usize
                } else {
                    a as usize
                }
            })
            .collect();
        let mut out_shape = Vec::with_capacity(out_ndim);
        let mut src = 0;
        for i in 0..out_ndim {
            if normed.contains(&i) {
                out_shape.push(1);
            } else {
                out_shape.push(self.atom.shape[src]);
                src += 1;
            }
        }
        self.emit_unary(
            Primitive::ExpandDims {
                operand: self.atom.clone(),
                axes: axes.to_vec(),
            },
            out_shape,
        )
    }

    fn reshape(&self, new_shape: &[isize]) -> Result<Self, crate::interpreters::EvalError> {
        let total: usize = self.atom.shape.iter().product();
        let known: usize = new_shape
            .iter()
            .filter(|&&d| d >= 0)
            .map(|&d| d as usize)
            .product();
        let out_shape: Vec<usize> = new_shape
            .iter()
            .map(|&d| if d < 0 { total / known } else { d as usize })
            .collect();
        Ok(self.emit_unary(
            Primitive::Reshape {
                operand: self.atom.clone(),
                new_shape: new_shape.to_vec(),
            },
            out_shape,
        ))
    }

    fn pad(&self, opt: &super::PaddingOptions) -> Self {
        let ndim = self.atom.shape.len() as isize;
        let padded: std::collections::HashSet<usize> = opt
            .axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + ndim) as usize
                } else {
                    a as usize
                }
            })
            .collect();
        let out_shape: Vec<usize> = self
            .atom
            .shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if padded.contains(&i) {
                    d + opt.config.left
                        + opt.config.right
                        + opt.config.interior * d.saturating_sub(1)
                } else {
                    d
                }
            })
            .collect();
        self.emit_unary(
            Primitive::Pad {
                operand: self.atom.clone(),
                options: opt.clone(),
            },
            out_shape,
        )
    }

    fn conv(&self, kernel: &Self, stride: isize) -> Result<Self, crate::interpreters::EvalError> {
        // input: [N, C_in, H, W], kernel: [C_out, C_in, kH, kW] → [N, C_out, H_out, W_out]
        let s = stride as usize;
        let out_shape = vec![
            self.atom.shape[0],
            kernel.atom.shape[0],
            (self.atom.shape[2] - kernel.atom.shape[2]) / s + 1,
            (self.atom.shape[3] - kernel.atom.shape[3]) / s + 1,
        ];
        let builder = Self::pick_builder(self.builder.clone(), kernel.builder.clone());
        let prim = Primitive::Conv {
            input: self.atom.clone(),
            kernel: kernel.atom.clone(),
            options: ConvOptions { stride },
        };
        let outvar = builder.borrow_mut().build_equation(prim, out_shape);
        Ok(Self {
            atom: outvar,
            builder: Some(builder),
        })
    }

    fn pool(
        &self,
        opt: &super::PoolOptions,
        average: bool,
    ) -> Result<Self, crate::interpreters::EvalError> {
        let spatial = self.atom.shape.len() - opt.window_size.len();
        let out_shape: Vec<usize> = self
            .atom
            .shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if i < spatial {
                    d
                } else {
                    (d - opt.window_size[i - spatial]) / opt.stride[i - spatial] + 1
                }
            })
            .collect();
        let prim = if average {
            Primitive::AvgPool {
                operand: self.atom.clone(),
                options: opt.clone(),
            }
        } else {
            Primitive::SumPool {
                operand: self.atom.clone(),
                options: opt.clone(),
            }
        };
        Ok(self.emit_unary(prim, out_shape))
    }

    fn exp(&self) -> Self {
        self.emit_unary(Primitive::Exp(self.atom.clone()), self.atom.shape.clone())
    }

    fn log(&self) -> Self {
        self.emit_unary(Primitive::Log(self.atom.clone()), self.atom.shape.clone())
    }

    fn relu(&self) -> Self {
        self.emit_unary(Primitive::Relu(self.atom.clone()), self.atom.shape.clone())
    }

    fn leaky_relu(&self, slope: f64) -> Self {
        self.emit_unary(
            Primitive::LeakyRelu {
                operand: self.atom.clone(),
                slope,
            },
            self.atom.shape.clone(),
        )
    }

    fn elu(&self, slope: f64) -> Self {
        self.emit_unary(
            Primitive::Elu {
                operand: self.atom.clone(),
                slope,
            },
            self.atom.shape.clone(),
        )
    }

    fn normcdf(&self) -> Self {
        self.emit_unary(
            Primitive::NormalCdf(self.atom.clone()),
            self.atom.shape.clone(),
        )
    }

    fn gelu(&self) -> Self {
        self.emit_unary(Primitive::Gelu(self.atom.clone()), self.atom.shape.clone())
    }
}
