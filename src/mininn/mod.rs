mod env;
use std::ops::{Add, Mul};

pub use env::*;

mod nn;
use ndarray::ArrayD;
pub use nn::*;

mod mlp;
pub use mlp::*;

mod parse;
pub use parse::*;

pub trait Value: Add<Output = Self> + Mul<Output = Self> + Sized + Clone {
    fn from_tensor(tensor: &ArrayD<f64>) -> Self;
}

#[derive(Debug, thiserror::Error)]
pub enum MininnError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("serde_json error: {0}")]
    Json(#[from] serde_json::error::Error),
    #[error("expected {expected} bytes for shape {shape:?}, got {got}")]
    SizeMismatch {
        expected: usize,
        shape: Vec<usize>,
        got: usize,
    },
    #[error("parse error: {0}")]
    Parse(String),
    #[error("rand error: {0}")]
    RandUniform(#[from] rand::distr::uniform::Error),
    #[error("rand error: {0}")]
    RandBinomial(#[from] rand_distr::BinomialError),
}
