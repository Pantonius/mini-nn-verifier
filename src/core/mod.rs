use std::ops::{Add, Mul};

pub trait Value: Add<Output = Self> + Mul<Output = Self> + Sized + Copy {}

struct PaddingOptionConfig {
    left: usize,
    right: usize,
    interior: usize,
}

struct PaddingOptions<T: Value> {
    config: PaddingOptionConfig,
    axes: Vec<isize>,
    value: Box<T>,
}

struct ConvOptions {
    stride: isize,
}

struct AvgPoolOptions {
    window_size: Vec<usize>,
    stride: Vec<usize>,
}

pub enum Primitive<T: Value> {
    // elementwise
    Neg(Box<T>),
    Reciprocal(Box<T>),
    Sq(Box<T>),
    Sqrt(Box<T>),
    Exp(Box<T>),
    Ln(Box<T>),
    Add(Box<T>, Box<T>),
    Mul(Box<T>, Box<T>),
    Where(Vec<bool>, Box<T>, Box<T>),
    // activations
    ReLU(Box<T>),
    LReLU(Box<T>),
    ELU(Box<T>),
    GELU(Box<T>),
    // linear algebra
    Dot(Box<T>, Box<T>),
    // reduction
    ReduceSum(),
    // shape manipulation
    ExpandDims(),
    MoveAxis(),
    Reshape(),
    // padding
    Padding(Box<T>, PaddingOptions<T>),
    // 2d convolution
    Conv(Box<T>, Box<T>, ConvOptions),
    // average pooling
    AvgPool(Box<T>, AvgPoolOptions),
}
