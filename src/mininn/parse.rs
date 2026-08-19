use ndarray::{Array2, ArrayD, IxDyn};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

use crate::mininn::{
    Atom, AtomKind, ComputeGraph, ConvOptions, Equation, MininnError, PaddingOptionConfig,
    PaddingOptions, PoolOptions, Primitive,
};

/// Decode a flat buffer of little-endian float64 values into an f64 vec,
/// checking it matches the expected element count for `shape`.
pub fn decode_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Encode f64 values as a flat little-endian byte buffer (inverse of [`decode_f64`]).
pub fn encode_f64(values: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Read a standalone input .bin file given the input variable's shape.
pub fn load_input_bin(path: &Path, shape: &[usize]) -> Result<ArrayD<f64>, MininnError> {
    let bytes = fs::read(path)?;
    let data = decode_f64(&bytes);

    ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|e| MininnError::Parse(format!("input shape mismatch: {e}")))
}

/// Read a standalone input .bin file given the input.
pub fn load_input_data_bin(path: &Path, d: usize) -> Result<Array2<f64>, MininnError> {
    let bytes = fs::read(path)?;
    let data = decode_f64(&bytes);

    if data.len() % d != 0 {
        return Err(MininnError::Parse(format!(
            "input length mismatch for input size; {} not divisible by {d}",
            data.len()
        )));
    }

    let n = data.len() / d;

    Array2::from_shape_vec((n, d), data)
        .map_err(|e| MininnError::Parse(format!("input shape mismatch: {e}")))
}

/// Write f64 values to a .bin file as flat little-endian float64.
pub fn write_output_bin(path: &Path, values: &[f64]) -> Result<(), MininnError> {
    fs::write(path, encode_f64(values))?;
    Ok(())
}

fn parse_shape(s: &str) -> Result<Vec<usize>, MininnError> {
    if s.is_empty() {
        return Ok(vec![]);
    }
    s.split(',')
        .map(|d| {
            d.trim()
                .parse::<usize>()
                .map_err(|e| MininnError::Parse(e.to_string()))
        })
        .collect()
}

/// Parse an option value that is either a bare int (`0`) or a tuple
/// (`(-1,)`, `(-2, -1)`) into a list of signed ints.
fn parse_isize_list(s: &str) -> Result<Vec<isize>, MininnError> {
    let inner = s.trim().trim_start_matches('(').trim_end_matches(')');
    inner
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<isize>()
                .map_err(|e| MininnError::Parse(e.to_string()))
        })
        .collect()
}

/// Like [`parse_isize_list`] but for unsigned ints (e.g. window sizes).
fn parse_usize_list(s: &str) -> Result<Vec<usize>, MininnError> {
    let inner = s.trim().trim_start_matches('(').trim_end_matches(')');
    inner
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<usize>()
                .map_err(|e| MininnError::Parse(e.to_string()))
        })
        .collect()
}

fn parse_scalar<T: std::str::FromStr>(s: &str, prim: &str, key: &str) -> Result<T, MininnError> {
    s.trim()
        .parse::<T>()
        .map_err(|_| MininnError::Parse(format!("{prim}: bad value for '{key}': {s:?}")))
}

/// Map a primitive name + its (already-parsed) options + operand [`Atom`]s onto
/// a typed [`Primitive`], validating operand arity and required options.
fn build_primitive(
    name: &str,
    options: &HashMap<String, String>,
    inputs: Vec<Atom>,
) -> Result<Primitive, MininnError> {
    use super::Primitive::*;

    let n = inputs.len();
    let want = |k: usize| -> Result<(), MininnError> {
        if n == k {
            Ok(())
        } else {
            Err(MininnError::Parse(format!(
                "{name}: expected {k} operand(s), got {n}"
            )))
        }
    };
    let opt = |k: &str| -> Result<&str, MininnError> {
        options
            .get(k)
            .map(String::as_str)
            .ok_or_else(|| MininnError::Parse(format!("{name}: missing option '{k}'")))
    };

    // Consume operands in graph order (arity is checked first in each arm).
    let mut it = inputs.into_iter();
    let mut op = || it.next().unwrap();

    Ok(match name {
        // elementwise unary
        "neg" => {
            want(1)?;
            Neg(op())
        }
        "reciprocal" => {
            want(1)?;
            Reciprocal(op())
        }
        "square" => {
            want(1)?;
            Square(op())
        }
        "sqrt" => {
            want(1)?;
            Sqrt(op())
        }
        "exp" => {
            want(1)?;
            Exp(op())
        }
        "log" => {
            want(1)?;
            Log(op())
        }
        // elementwise binary
        "add" => {
            want(2)?;
            Add(op(), op())
        }
        "mul" => {
            want(2)?;
            Mul(op(), op())
        }
        // selection
        "where" => {
            want(3)?;
            Where(op(), op(), op())
        }
        // activations
        "relu" => {
            want(1)?;
            Relu(op())
        }
        "normalcdf" => {
            want(1)?;
            NormalCdf(op())
        }
        "leaky_relu" => {
            want(1)?;
            LeakyRelu {
                operand: op(),
                slope: parse_scalar(opt("slope")?, name, "slope")?,
            }
        }
        "elu" => {
            want(1)?;
            let slope = match options.get("slope") {
                Some(s) => parse_scalar(s, name, "slope")?,
                None => 1.0,
            };
            Elu {
                operand: op(),
                slope,
            }
        }
        "gelu" => {
            want(1)?;
            Gelu(op())
        }
        // linear algebra
        "dot" => {
            want(2)?;
            Dot(op(), op())
        }
        // reduction
        "reduce_sum" => {
            want(1)?;
            ReduceSum {
                operand: op(),
                axes: parse_isize_list(opt("axes")?)?,
            }
        }
        // shape manipulation
        "expand_dims" => {
            want(1)?;
            ExpandDims {
                operand: op(),
                axes: parse_isize_list(opt("axes")?)?,
            }
        }
        "moveaxis" => {
            want(1)?;
            MoveAxis {
                operand: op(),
                source: parse_scalar(opt("source")?, name, "source")?,
                destination: parse_scalar(opt("destination")?, name, "destination")?,
            }
        }
        "reshape" => {
            want(1)?;
            Reshape {
                operand: op(),
                new_shape: parse_isize_list(opt("new_shape")?)?,
            }
        }
        // padding
        "pad" => {
            want(1)?;
            let cfg = parse_usize_list(opt("config")?)?;
            if cfg.len() != 3 {
                return Err(MininnError::Parse(format!(
                    "pad: config needs 3 values (left, right, interior), got {}",
                    cfg.len()
                )));
            }
            Pad {
                operand: op(),
                options: PaddingOptions {
                    config: PaddingOptionConfig {
                        left: cfg[0],
                        right: cfg[1],
                        interior: cfg[2],
                    },
                    axes: parse_isize_list(opt("axes")?)?,
                    value: parse_scalar(opt("value")?, name, "value")?,
                },
            }
        }
        // 2d convolution
        "conv" => {
            want(2)?;
            Conv {
                input: op(),
                kernel: op(),
                options: ConvOptions {
                    stride: parse_scalar(opt("stride")?, name, "stride")?,
                },
            }
        }
        // average / sum pooling (identical option shape)
        "avgpool" | "sumpool" => {
            want(1)?;
            let options = PoolOptions {
                window_size: parse_usize_list(opt("window_size")?)?,
                stride: parse_usize_list(opt("stride")?)?,
            };
            let operand = op();
            if name == "avgpool" {
                AvgPool { operand, options }
            } else {
                SumPool { operand, options }
            }
        }
        other => return Err(MininnError::Parse(format!("unknown primitive '{other}'"))),
    })
}

/// Parse "name[shape]" into (name, shape).
fn parse_atom_header(s: &str) -> Result<(String, Vec<usize>), MininnError> {
    let s = s.trim();
    let open = s
        .find('[')
        .ok_or_else(|| MininnError::Parse(format!("bad atom: {s}")))?;
    let name = s[..open].to_string();
    let dims = s[open + 1..].trim_end_matches(']');
    Ok((name, parse_shape(dims)?))
}

fn parse_atom<'a>(
    s: &str,
    atoms: &'a mut HashMap<String, Atom>,
    consts: &HashMap<String, Vec<u8>>,
) -> Result<Atom, MininnError> {
    let (name, shape) = parse_atom_header(s)?;

    if let Some(existing) = atoms.get(&name) {
        if existing.shape != shape {
            return Err(MininnError::Parse(format!(
                "inconsistent shape for {name}: {:?} vs {:?}",
                existing.shape, shape
            )));
        }
        return Ok(existing.clone());
    }

    let is_const = name
        .chars()
        .next()
        .map(|c| c.is_uppercase())
        .unwrap_or(false);
    let kind = if is_const {
        let bytes = consts
            .get(&name)
            .ok_or_else(|| MininnError::Parse(format!("missing const data for {name}")))?;
        AtomKind::Const(
            ArrayD::from_shape_vec(IxDyn(&shape), decode_f64(bytes))
                .map_err(|e| MininnError::Parse(format!("const shape mismatch: {e}")))?,
        )
    } else {
        AtomKind::Var
    };

    let atom = Atom {
        name: name.clone(),
        shape,
        kind,
    };
    atoms.insert(name, atom.clone());
    Ok(atom)
}

fn parse_equation(
    line: &str,
    atoms: &mut HashMap<String, Atom>,
    consts: &HashMap<String, Vec<u8>>,
) -> Result<Equation, MininnError> {
    let (outvar_str, expr) = line
        .split_once('=')
        .ok_or_else(|| MininnError::Parse(format!("bad equation: {line}")))?;
    let expr = expr.trim();

    let brace_open = expr
        .find('{')
        .ok_or_else(|| MininnError::Parse(format!("no options block: {expr}")))?;
    let brace_close = expr
        .find('}')
        .ok_or_else(|| MininnError::Parse(format!("no options block: {expr}")))?;

    let primitive = expr[..brace_open].trim().to_string();
    let opts_block = expr[brace_open + 1..brace_close].trim();
    let mut options = HashMap::new();
    if !opts_block.is_empty() {
        for pair in opts_block.split(';') {
            let (k, v) = pair
                .split_once(':')
                .ok_or_else(|| MininnError::Parse(format!("bad option: {pair}")))?;
            options.insert(k.trim().to_string(), v.trim().to_string());
        }
    }

    let outvar = parse_atom(outvar_str, atoms, consts)?;
    let inputs_str = expr[brace_close + 1..].trim();
    let inputs = inputs_str
        .split_whitespace()
        .map(|t| parse_atom(t, atoms, consts))
        .collect::<Result<Vec<_>, _>>()?;

    let primitive = build_primitive(&primitive, &options, inputs)?;

    Ok(Equation { primitive, outvar })
}

pub fn load_mininn(path: &Path) -> Result<ComputeGraph, MininnError> {
    let file = File::open(path)?;
    let mut zip = ZipArchive::new(file)?;

    // 1. Read graph.txt
    let graph_str = {
        let mut f = zip.by_name("graph.txt")?;
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        s
    };

    // 2. Read all *.bin entries (constants) up front
    let mut consts: HashMap<String, Vec<u8>> = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let mut buf = Vec::new();

        entry.read_to_end(&mut buf)?;
        consts.insert(
            entry
                .mangled_name()
                .as_path()
                .file_stem()
                .expect("dnwadwao")
                .to_os_string()
                .into_string()
                .unwrap(),
            buf,
        );
    }

    // 3. Parse the graph text
    let lines: Vec<&str> = graph_str.lines().collect();
    let input_line = lines
        .first()
        .ok_or_else(|| MininnError::Parse("empty graph".into()))?
        .trim();
    let output_line = lines.last().unwrap().trim();
    let eqn_lines = &lines[1..lines.len() - 1];

    let mut atoms = HashMap::new();

    let invars = input_line
        .strip_prefix("input:")
        .ok_or_else(|| MininnError::Parse("missing input line".into()))?
        .split(';')
        .map(|s| parse_atom(s, &mut atoms, &consts))
        .collect::<Result<Vec<_>, _>>()?;

    let equations = eqn_lines
        .iter()
        .map(|l| parse_equation(l, &mut atoms, &consts))
        .collect::<Result<Vec<_>, _>>()?;

    let outvars = output_line
        .strip_prefix("output:")
        .ok_or_else(|| MininnError::Parse("missing output line".into()))?
        .split(';')
        .map(|s| parse_atom(s, &mut atoms, &consts))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ComputeGraph {
        invars,
        outvars,
        equations,
    })
}
