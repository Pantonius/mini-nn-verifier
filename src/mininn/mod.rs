use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

#[derive(Debug, thiserror::Error)]
pub enum MinninError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("expected {expected} bytes for shape {shape:?}, got {got}")]
    SizeMismatch {
        expected: usize,
        shape: Vec<usize>,
        got: usize,
    },
    #[error("parse error: {0}")]
    Parse(String),
}

/// Decode a flat buffer of little-endian float64 values into an f64 vec,
/// checking it matches the expected element count for `shape`.
pub fn decode_f64(bytes: &[u8], shape: &[usize]) -> Result<Vec<f64>, MinninError> {
    // scalar (shape == []) has exactly 1 element
    let expected_elems = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };

    if bytes.len() != expected_elems * 8 {
        return Err(MinninError::SizeMismatch {
            expected: expected_elems * 8,
            shape: shape.to_vec(),
            got: bytes.len(),
        });
    }

    Ok(bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Read a standalone input .bin file given the input variable's shape.
pub fn load_input_bin(path: &Path, shape: &[usize]) -> Result<Vec<f64>, MinninError> {
    let bytes = fs::read(path)?;
    decode_f64(&bytes, shape)
}
#[derive(Debug, Clone)]
pub enum AtomKind {
    Var,
    Const(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub name: String,
    pub shape: Vec<usize>,
    pub kind: AtomKind,
}

#[derive(Debug, Clone)]
pub struct Equation {
    pub primitive: String,
    pub options: HashMap<String, String>, // raw literal text; parse further as needed
    pub inputs: Vec<Atom>,
    pub outvar: Atom,
}

#[derive(Debug, Clone)]
pub struct ComputeGraph {
    pub invars: Vec<Atom>,
    pub outvars: Vec<Atom>,
    pub equations: Vec<Equation>,
}

fn parse_shape(s: &str) -> Result<Vec<usize>, MinninError> {
    if s.is_empty() {
        return Ok(vec![]);
    }
    s.split(',')
        .map(|d| {
            d.trim()
                .parse::<usize>()
                .map_err(|e| MinninError::Parse(e.to_string()))
        })
        .collect()
}

/// Parse "name[shape]" into (name, shape).
fn parse_atom_header(s: &str) -> Result<(String, Vec<usize>), MinninError> {
    let s = s.trim();
    let open = s
        .find('[')
        .ok_or_else(|| MinninError::Parse(format!("bad atom: {s}")))?;
    let name = s[..open].to_string();
    let dims = s[open + 1..].trim_end_matches(']');
    Ok((name, parse_shape(dims)?))
}

fn parse_atom<'a>(
    s: &str,
    atoms: &'a mut HashMap<String, Atom>,
    consts: &HashMap<String, Vec<u8>>,
) -> Result<Atom, MinninError> {
    let (name, shape) = parse_atom_header(s)?;

    if let Some(existing) = atoms.get(&name) {
        if existing.shape != shape {
            return Err(MinninError::Parse(format!(
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
            .ok_or_else(|| MinninError::Parse(format!("missing const data for {name}")))?;
        AtomKind::Const(decode_f64(bytes, &shape)?)
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
) -> Result<Equation, MinninError> {
    let (outvar_str, expr) = line
        .split_once('=')
        .ok_or_else(|| MinninError::Parse(format!("bad equation: {line}")))?;
    let expr = expr.trim();

    let brace_open = expr
        .find('{')
        .ok_or_else(|| MinninError::Parse(format!("no options block: {expr}")))?;
    let brace_close = expr
        .find('}')
        .ok_or_else(|| MinninError::Parse(format!("no options block: {expr}")))?;

    let primitive = expr[..brace_open].trim().to_string();
    let opts_block = expr[brace_open + 1..brace_close].trim();
    let mut options = HashMap::new();
    if !opts_block.is_empty() {
        for pair in opts_block.split(';') {
            let (k, v) = pair
                .split_once(':')
                .ok_or_else(|| MinninError::Parse(format!("bad option: {pair}")))?;
            options.insert(k.trim().to_string(), v.trim().to_string());
        }
    }

    let outvar = parse_atom(outvar_str, atoms, consts)?;
    let inputs_str = expr[brace_close + 1..].trim();
    let inputs = inputs_str
        .split_whitespace()
        .map(|t| parse_atom(t, atoms, consts))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Equation {
        primitive,
        options,
        inputs,
        outvar,
    })
}

pub fn load_mininn(path: &Path) -> Result<ComputeGraph, MinninError> {
    let file = std::fs::File::open(path)?;
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
        .ok_or_else(|| MinninError::Parse("empty graph".into()))?
        .trim();
    let output_line = lines.last().unwrap().trim();
    let eqn_lines = &lines[1..lines.len() - 1];

    let mut atoms = HashMap::new();

    let invars = input_line
        .strip_prefix("input:")
        .ok_or_else(|| MinninError::Parse("missing input line".into()))?
        .split(';')
        .map(|s| parse_atom(s, &mut atoms, &consts))
        .collect::<Result<Vec<_>, _>>()?;

    let equations = eqn_lines
        .iter()
        .map(|l| parse_equation(l, &mut atoms, &consts))
        .collect::<Result<Vec<_>, _>>()?;

    let outvars = output_line
        .strip_prefix("output:")
        .ok_or_else(|| MinninError::Parse("missing output line".into()))?
        .split(';')
        .map(|s| parse_atom(s, &mut atoms, &consts))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ComputeGraph {
        invars,
        outvars,
        equations,
    })
}
