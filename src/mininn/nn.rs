use std::fmt::Display;

use crate::{
    interpreters::{EvalError, concrete::eval_util::Tensor},
    mininn::Layer,
};

#[derive(Debug, Clone)]
pub enum AtomKind {
    Var,
    Const(Tensor),
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub name: String,
    pub shape: Vec<usize>,
    pub kind: AtomKind,
}

impl Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.kind {
            AtomKind::Var => "var".to_string(),
            AtomKind::Const(data) => format!("const[{} elems]", data.len()),
        };
        write!(f, "{}{:?} ({kind})", self.name, self.shape)
    }
}

#[derive(Debug, Clone)]
/// The padding configuration, that is:
/// - *left* padding
/// - *right* padding and
/// - *interior* padding between each pair of neighboring elements
/// along each padded axis
pub struct PaddingOptionConfig {
    /// Padding at the beginning of an axis
    pub left: usize,
    /// Padding at the end of an axis
    pub right: usize,
    /// Padding between each pair of neighboring elements
    pub interior: usize,
}

#[derive(Debug, Clone)]
/// Padding options, that is:
/// - *config* of left, right and interior padding
/// - *axes* that are to be padded according to the padding configuration
/// - *value* that is used to fill the padding with
pub struct PaddingOptions {
    /// Padding configuration (left, right and interior padding)
    pub config: PaddingOptionConfig,
    /// Axes to be padded
    pub axes: Vec<isize>,
    /// Scalar fill value (a literal in the `.mininn` options, e.g. `0.0`).
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct ConvOptions {
    pub stride: isize,
}

#[derive(Debug, Clone)]
pub struct PoolOptions {
    pub window_size: Vec<usize>,
    pub stride: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum Primitive {
    // elementwise unary
    Neg(Atom),
    Reciprocal(Atom),
    Square(Atom),
    Sqrt(Atom),
    Exp(Atom),
    Log(Atom),
    // elementwise binary
    Add(Atom, Atom),
    Mul(Atom, Atom),
    // selection: where(cond, x, y)
    Where(Atom, Atom, Atom),
    // activations
    Relu(Atom),
    LeakyRelu {
        operand: Atom,
        slope: f64,
    },
    Elu {
        operand: Atom,
        slope: f64,
    },
    Gelu(Atom),
    /// Standard-normal CDF (the nonlinearity behind GELU).
    NormalCdf(Atom),
    // linear algebra
    Dot(Atom, Atom),
    // reduction
    ReduceSum {
        operand: Atom,
        axes: Vec<isize>,
    },
    // shape manipulation
    ExpandDims {
        operand: Atom,
        axes: Vec<isize>,
    },
    MoveAxis {
        operand: Atom,
        source: isize,
        destination: isize,
    },
    Reshape {
        operand: Atom,
        new_shape: Vec<isize>,
    },
    // strided slice: y = x[start:end:step] along a single axis
    Slice {
        operand: Atom,
        axis: isize,
        start: isize,
        end: Option<isize>,
        step: isize,
    },
    // padding
    Pad {
        operand: Atom,
        options: PaddingOptions,
    },
    // 2d convolution: conv(input, kernel)
    Conv {
        input: Atom,
        kernel: Atom,
        options: ConvOptions,
    },
    // gradient of conv w.r.t. kernel: conv_kernel_grad(grad_out, input)
    ConvKernelGrad {
        grad_out: Atom,
        input: Atom,
        options: ConvOptions,
        kernel_shape: Vec<usize>,
    },
    // average pooling
    AvgPool {
        operand: Atom,
        options: PoolOptions,
    },
    // sum pooling (same options as average pooling)
    SumPool {
        operand: Atom,
        options: PoolOptions,
    },
}

impl Primitive {
    /// The `.mininn` primitive name (as it appears in `graph.txt`).
    pub fn name(&self) -> &'static str {
        use Primitive::*;
        match self {
            Neg(_) => "neg",
            Reciprocal(_) => "reciprocal",
            Square(_) => "square",
            Sqrt(_) => "sqrt",
            Exp(_) => "exp",
            Log(_) => "log",
            Add(..) => "add",
            Mul(..) => "mul",
            Where(..) => "where",
            Relu(_) => "relu",
            LeakyRelu { .. } => "leaky_relu",
            Elu { .. } => "elu",
            Gelu(_) => "gelu",
            NormalCdf(_) => "normalcdf",
            Dot(..) => "dot",
            ReduceSum { .. } => "reduce_sum",
            ExpandDims { .. } => "expand_dims",
            MoveAxis { .. } => "moveaxis",
            Reshape { .. } => "reshape",
            Slice { .. } => "slice",
            Pad { .. } => "pad",
            Conv { .. } => "conv",
            ConvKernelGrad { .. } => "conv_kernel_grad",
            AvgPool { .. } => "avgpool",
            SumPool { .. } => "sumpool",
        }
    }

    /// The operands, in graph order.
    pub fn operands(&self) -> Vec<&Atom> {
        use Primitive::*;
        match self {
            Neg(x) | Reciprocal(x) | Square(x) | Sqrt(x) | Exp(x) | Log(x) | Relu(x) | Gelu(x)
            | NormalCdf(x) => vec![x],
            LeakyRelu { operand, .. }
            | Elu { operand, .. }
            | ReduceSum { operand, .. }
            | ExpandDims { operand, .. }
            | MoveAxis { operand, .. }
            | Reshape { operand, .. }
            | Pad { operand, .. }
            | Slice { operand, .. }
            | AvgPool { operand, .. }
            | SumPool { operand, .. } => vec![operand],
            Add(a, b) | Mul(a, b) | Dot(a, b) => vec![a, b],
            Conv { input, kernel, .. } => vec![input, kernel],
            ConvKernelGrad {
                grad_out, input, ..
            } => vec![grad_out, input],
            Where(c, x, y) => vec![c, x, y],
        }
    }
}

impl Display for Primitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.name())?;
        for (i, operand) in self.operands().into_iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{operand}")?;
        }
        write!(f, ")")
    }
}

pub enum Activation {
    Relu(Atom),
    LeakyRelu { operand: Atom, slope: f64 },
    Elu { operand: Atom, slope: f64 },
    Gelu(Atom),
}

impl TryFrom<&Primitive> for Activation {
    type Error = EvalError;

    fn try_from(primitive: &Primitive) -> Result<Self, EvalError> {
        match primitive {
            Primitive::Relu(a) => Ok(Activation::Relu(a.clone())),
            Primitive::LeakyRelu { operand, slope } => Ok(Activation::LeakyRelu {
                operand: operand.clone(),
                slope: *slope,
            }),
            Primitive::Gelu(a) => Ok(Activation::Gelu(a.clone())),
            Primitive::Elu { operand, slope } => Ok(Activation::Elu {
                operand: operand.clone(),
                slope: *slope,
            }),
            _ => Err(EvalError::Eval(format!(
                "Could not convert {primitive} into Activation"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Equation {
    pub primitive: Primitive,
    pub outvar: Atom,
}

#[derive(Debug, Clone)]
pub struct ComputeGraph {
    pub invars: Vec<Atom>,
    pub outvars: Vec<Atom>,
    pub equations: Vec<Equation>,
}

impl ComputeGraph {
    pub fn from_layers(layers: &[Layer], batch_size: usize, in_size: usize) -> ComputeGraph {
        let x = Atom {
            name: "x".to_string(),
            shape: vec![batch_size, in_size],
            kind: AtomKind::Var,
        };

        let mut invars = vec![x.clone()];
        let mut equations = Vec::new();

        let mut prev_size = in_size;
        let mut prev = x.clone();

        for i in 0..layers.len() {
            let out_size = layers[i].b.len();

            let w = Atom {
                name: format!("W{i}"),
                shape: vec![prev_size, out_size],
                kind: AtomKind::Var,
            };
            let b = Atom {
                name: format!("b{i}"),
                shape: vec![out_size],
                kind: AtomKind::Var,
            };
            invars.push(w.clone());
            invars.push(b.clone());

            let h = Atom {
                name: format!("h{i}"),
                shape: vec![batch_size, out_size],
                kind: AtomKind::Var,
            };

            equations.push(Equation {
                primitive: Primitive::Dot(prev, w),
                outvar: h.clone(),
            });

            let a = Atom {
                name: format!("a{i}"),
                shape: vec![batch_size, out_size],
                kind: AtomKind::Var,
            };

            equations.push(Equation {
                primitive: Primitive::Add(h, b),
                outvar: a.clone(),
            });

            if i < layers.len() - 1 {
                let r = Atom {
                    name: format!("r{i}"),
                    shape: vec![batch_size, out_size],
                    kind: AtomKind::Var,
                };

                equations.push(Equation {
                    primitive: Primitive::Relu(a.clone()),
                    outvar: r.clone(),
                });

                prev_size = out_size;
                prev = r.clone()
            } else {
                prev = a.clone()
            }
        }

        ComputeGraph {
            invars,
            outvars: vec![prev],
            equations,
        }
    }
}
