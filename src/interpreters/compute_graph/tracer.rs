use std::{
    cell::RefCell,
    ops::{Add, Mul, Neg, Sub},
    rc::Rc,
};

use ndarray::{ArrayD, IxDyn};

use crate::{
    interpreters::concrete::eval_util::{Tensor, broadcast_shape},
    mininn::{
        Atom, AtomKind, ComputeGraph, ConvOptions, Equation, MininnError, PaddingOptions,
        PoolOptions, Primitive, Value,
    },
};

#[derive(Debug, Clone)]
pub struct ComputeGraphBuilder {
    n_atoms: usize,
    equations: Vec<Equation>,
    invars: Vec<Atom>,
}

impl ComputeGraphBuilder {
    pub fn new() -> Self {
        ComputeGraphBuilder {
            n_atoms: 0,
            equations: Vec::new(),
            invars: Vec::new(),
        }
    }

    pub fn register_invar(&mut self, name: String, shape: Vec<usize>) -> Atom {
        let atom = Atom {
            name,
            shape,
            kind: AtomKind::Var,
        };
        self.invars.push(atom.clone());
        atom
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

    pub fn build(self, outvar: Atom) -> ComputeGraph {
        ComputeGraph {
            invars: self.invars,
            outvars: vec![outvar],
            equations: self.equations,
        }
    }
}

impl From<ComputeGraph> for ComputeGraphBuilder {
    fn from(graph: ComputeGraph) -> Self {
        // Continue the `a{n}` counter past any existing generated names so freshly
        // built equations can't collide with atoms already in the graph.
        let max_gen = graph
            .invars
            .iter()
            .map(|a| &a.name)
            .chain(graph.equations.iter().map(|e| &e.outvar.name))
            .filter_map(|name| name.strip_prefix('a').and_then(|n| n.parse::<usize>().ok()))
            .max();
        ComputeGraphBuilder {
            n_atoms: max_gen.map_or(0, |n| n + 1),
            equations: graph.equations,
            invars: graph.invars,
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

    pub fn atom(&self) -> &Atom {
        &self.atom
    }

    fn pick_builder(
        self_b: Option<Rc<RefCell<ComputeGraphBuilder>>>,
        rhs_b: Option<Rc<RefCell<ComputeGraphBuilder>>>,
    ) -> Rc<RefCell<ComputeGraphBuilder>> {
        self_b
            .or(rhs_b)
            .expect("at least one operand must be a Var tracer")
    }

    fn emit_unary(
        &self,
        primitive: Primitive,
        out_shape: Vec<usize>,
        fallback: impl FnOnce(&Tensor) -> Tensor,
    ) -> Self {
        if let Some(builder) = &self.builder {
            let builder = builder.clone();
            let outvar = builder.borrow_mut().build_equation(primitive, out_shape);
            Self {
                atom: outvar,
                builder: Some(builder),
            }
        } else {
            let AtomKind::Const(t) = &self.atom.kind else {
                panic!("Var tracer without builder")
            };
            Self::from(fallback(t))
        }
    }
}

impl Add for Tracer {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.builder.is_none() && rhs.builder.is_none() {
            if let (AtomKind::Const(lt), AtomKind::Const(rt)) = (&self.atom.kind, &rhs.atom.kind) {
                return Self::from(lt.clone() + rt.clone());
            }
        }
        let shape = broadcast_shape(&self.atom.shape, &rhs.atom.shape)
            .expect("Tracer::add: shapes not broadcastable");
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
        if self.builder.is_none() && rhs.builder.is_none() {
            if let (AtomKind::Const(lt), AtomKind::Const(rt)) = (&self.atom.kind, &rhs.atom.kind) {
                return Self::from(lt.clone() * rt.clone());
            }
        }
        let shape = broadcast_shape(&self.atom.shape, &rhs.atom.shape)
            .expect("Tracer::mul: shapes not broadcastable");
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

impl From<f64> for Tracer {
    fn from(value: f64) -> Self {
        Self::from(ArrayD::from_elem(IxDyn(&[]), value))
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
        if let (AtomKind::Const(t), None) = (&self.atom.kind, &self.builder) {
            return Self::from(-t.clone());
        }
        let shape = self.atom.shape.clone();
        let builder = self.builder.expect("Var tracer missing builder");
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

    fn r#where(cond: &Self, x: &Self, y: &Self) -> Result<Self, MininnError> {
        if cond.builder.is_none() && x.builder.is_none() && y.builder.is_none() {
            if let (AtomKind::Const(c), AtomKind::Const(xt), AtomKind::Const(yt)) =
                (&cond.atom.kind, &x.atom.kind, &y.atom.kind)
            {
                return Tensor::r#where(c, xt, yt).map(Self::from);
            }
        }
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
            move |t| t.moveaxis(src, dst),
        )
    }

    fn dot(&self, b: &Self) -> Result<Self, MininnError> {
        if self.builder.is_none() && b.builder.is_none() {
            if let (AtomKind::Const(lt), AtomKind::Const(rt)) = (&self.atom.kind, &b.atom.kind) {
                return lt.dot(rt).map(Self::from);
            }
        }

        let (ash, bsh) = (&self.atom.shape, &b.atom.shape);

        let out_shape: Vec<usize> = if ash.is_empty() || bsh.is_empty() {
            broadcast_shape(ash, bsh).expect("dot: scalar operand not broadcastable")
        } else if bsh.len() == 1 {
            ash[..ash.len() - 1].to_vec()
        } else {
            let mut s = ash[..ash.len() - 1].to_vec();
            s.extend_from_slice(&bsh[..bsh.len() - 2]);
            s.push(bsh[bsh.len() - 1]);
            s
        };

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
            |t| t.square(),
        )
    }

    fn sqrt(&self) -> Self {
        self.emit_unary(
            Primitive::Sqrt(self.atom.clone()),
            self.atom.shape.clone(),
            |t| t.sqrt(),
        )
    }

    fn reciprocal(&self) -> Self {
        self.emit_unary(
            Primitive::Reciprocal(self.atom.clone()),
            self.atom.shape.clone(),
            |t| t.reciprocal(),
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
            |t| t.reduce_sum(axes),
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
            |t| t.expand_dims(axes),
        )
    }

    fn reshape(&self, new_shape: &[isize]) -> Result<Self, MininnError> {
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
            |t| t.reshape(new_shape).unwrap(),
        ))
    }

    fn slice(&self, axis: isize, start: isize, end: Option<isize>, step: isize) -> Self {
        let ndim = self.atom.shape.len();
        let ax = if axis < 0 {
            (axis + ndim as isize) as usize
        } else {
            axis as usize
        };
        let size = self.atom.shape[ax] as isize;

        let start_n = if start >= 0 {
            start.min(size)
        } else {
            (start + size).max(0)
        };
        let end_n = match end {
            Some(e) if e >= 0 => e.min(size),
            Some(e) => (e + size).max(0),
            None => {
                if step > 0 {
                    size
                } else {
                    -1
                }
            }
        };
        let out_ax = if step > 0 {
            let diff = (end_n - start_n).max(0) as usize;
            (diff + step as usize - 1) / step as usize
        } else {
            let diff = (start_n - end_n).max(0) as usize;
            (diff + (-step) as usize - 1) / (-step) as usize
        };

        let mut out_shape = self.atom.shape.clone();
        out_shape[ax] = out_ax;
        self.emit_unary(
            Primitive::Slice {
                operand: self.atom.clone(),
                axis,
                start,
                end,
                step,
            },
            out_shape,
            move |t| t.slice(axis, start, end, step),
        )
    }

    fn pad(&self, opt: &PaddingOptions) -> Self {
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
            |t| t.pad(opt),
        )
    }

    fn conv_kernel_grad(
        &self,
        input: &Self,
        stride: isize,
        kernel_shape: &[usize],
    ) -> Result<Self, MininnError> {
        if self.builder.is_none() && input.builder.is_none() {
            if let (AtomKind::Const(gt), AtomKind::Const(it)) =
                (&self.atom.kind, &input.atom.kind)
            {
                return gt
                    .conv_kernel_grad(it, stride, kernel_shape)
                    .map(Self::from);
            }
        }
        let builder = Self::pick_builder(self.builder.clone(), input.builder.clone());
        let prim = Primitive::ConvKernelGrad {
            grad_out: self.atom.clone(),
            input: input.atom.clone(),
            options: ConvOptions { stride },
            kernel_shape: kernel_shape.to_vec(),
        };
        let outvar = builder
            .borrow_mut()
            .build_equation(prim, kernel_shape.to_vec());
        Ok(Self {
            atom: outvar,
            builder: Some(builder),
        })
    }

    fn conv(&self, kernel: &Self, stride: isize) -> Result<Self, MininnError> {
        if self.builder.is_none() && kernel.builder.is_none() {
            if let (AtomKind::Const(it), AtomKind::Const(kt)) =
                (&self.atom.kind, &kernel.atom.kind)
            {
                return it.conv(kt, stride).map(Self::from);
            }
        }
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

    fn pool(&self, opt: &PoolOptions, average: bool) -> Result<Self, MininnError> {
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
        Ok(self.emit_unary(prim, out_shape, |t| t.pool(opt, average).unwrap()))
    }

    fn exp(&self) -> Self {
        self.emit_unary(
            Primitive::Exp(self.atom.clone()),
            self.atom.shape.clone(),
            |t| t.exp(),
        )
    }

    fn log(&self) -> Self {
        self.emit_unary(
            Primitive::Log(self.atom.clone()),
            self.atom.shape.clone(),
            |t| t.log(),
        )
    }

    fn relu(&self) -> Self {
        self.emit_unary(
            Primitive::Relu(self.atom.clone()),
            self.atom.shape.clone(),
            |t| t.relu(),
        )
    }

    fn leaky_relu(&self, slope: f64) -> Self {
        self.emit_unary(
            Primitive::LeakyRelu {
                operand: self.atom.clone(),
                slope,
            },
            self.atom.shape.clone(),
            move |t| t.leaky_relu(slope),
        )
    }

    fn elu(&self, slope: f64) -> Self {
        self.emit_unary(
            Primitive::Elu {
                operand: self.atom.clone(),
                slope,
            },
            self.atom.shape.clone(),
            move |t| t.elu(slope),
        )
    }

    fn normcdf(&self) -> Self {
        self.emit_unary(
            Primitive::NormalCdf(self.atom.clone()),
            self.atom.shape.clone(),
            |t| t.normcdf(),
        )
    }

    fn gelu(&self) -> Self {
        self.emit_unary(
            Primitive::Gelu(self.atom.clone()),
            self.atom.shape.clone(),
            |t| t.gelu(),
        )
    }
}
