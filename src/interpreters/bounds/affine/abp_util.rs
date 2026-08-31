use ndarray::ArrayD;

use crate::{
    interpreters::{
        EvalError,
        bounds::ibp_util::IBPTensor,
        concrete::eval_util::{normcdf, normpdf},
    },
    mininn::Value,
};

pub fn lbp_inner<T: Value>(w: &T, x: &T) -> T {
    let product = w.clone() * x.clone();
    let axes: Vec<isize> = (0..product.ndim() as isize).collect();
    product.reduce_sum(&axes)
}

// ================================
// ABPTensor (Affine Bounds)
// ================================
#[derive(Debug, Clone)]
pub struct ABPTensor<T: Value> {
    pub weights: T,
    pub biases: T,
}

impl<T: Value> ABPTensor<T> {
    /// Concretize the affine bound given concrete lower/upper bounds on the input.
    pub fn concretize(&self, lb: &T, ub: &T) -> T {
        let pos_w = self.weights.relu();
        let neg_w = self.weights.clone() - pos_w.clone();
        self.biases.clone() + lbp_inner(&pos_w, lb) + lbp_inner(&neg_w, ub)
    }
}

// ================================
// Helpers for Bounding
// ================================
fn tangent_through(tx: f64, ty: f64, low: f64, high: f64) -> f64 {
    let g = |d: f64| normpdf(d) * (tx - d) + normcdf(d) - ty;
    let (mut a, mut b) = (low, high);
    for _ in 0..60 {
        let m = 0.5 * (a + b);
        if g(m) > 0.0 { b = m } else { a = m }
    }
    0.5 * (a + b)
}

fn secant(l: f64, u: f64, fl: f64, fu: f64) -> (f64, f64) {
    let slope = (fu - fl) / (u - l);
    let offset = fl - slope * l;

    (slope, offset)
}

pub(crate) fn bound_convex<T: Value>(
    x: &IBPTensor,
    w: &T,
    f: impl Fn(f64, f64) -> (f64, f64, f64, f64),
    bound_req: impl Fn(f64, f64) -> Result<(), EvalError>,
) -> Result<ABPTensor<T>, EvalError> {
    let (mut ls, mut lo, mut us, mut uo) = (vec![], vec![], vec![], vec![]);

    for (&l, &u) in x.lb.iter().zip(&x.ub) {
        bound_req(l, u)?;
        let (lower_slope, lower_offset, upper_slope, upper_offset) = f(l, u);

        ls.push(lower_slope);
        lo.push(lower_offset);
        us.push(upper_slope);
        uo.push(upper_offset);
    }

    let lower_slope = T::from(ArrayD::from_shape_vec(x.shape(), ls).unwrap());
    let lower_offset = T::from(ArrayD::from_shape_vec(x.shape(), lo).unwrap());
    let upper_slope = T::from(ArrayD::from_shape_vec(x.shape(), us).unwrap());
    let upper_offset = T::from(ArrayD::from_shape_vec(x.shape(), uo).unwrap());

    let pos_w = w.relu();
    let neg_w = w.clone() - pos_w.clone();

    Ok(ABPTensor {
        weights: lower_slope * pos_w.clone() + upper_slope * neg_w.clone(),
        biases: lbp_inner(&lower_offset, &pos_w) + lbp_inner(&upper_offset, &neg_w),
    })
}

pub(crate) fn normcdf_lines(l: f64, u: f64) -> (f64, f64, f64, f64) {
    if u - l < 1e-9 {
        // Degenerate interval: sandwich with flat lines
        (0.0, normcdf(l), 0.0, normcdf(u))
    } else if u <= 0.0 {
        // Convex region: upper = secant, lower = midpoint tangent.
        let (upper_slope, upper_offset) = secant(l, u, normcdf(l), normcdf(u));

        let mid = (u + l) * 0.5;
        let lower_slope = normpdf(mid);
        let lower_offset = normcdf(mid) - lower_slope * mid;

        (lower_slope, lower_offset, upper_slope, upper_offset)
    } else if l >= 0.0 {
        // Concave region: lower = secant, upper = midpoint tangent.
        let (lower_slope, lower_offset) = secant(l, u, normcdf(l), normcdf(u));

        let mid = (u + l) * 0.5;
        let upper_slope = normpdf(mid);
        let upper_offset = normcdf(mid) - upper_slope * mid;
        (lower_slope, lower_offset, upper_slope, upper_offset)
    } else {
        // Crossing: tangent through the far endpoint, unless the chord already works.
        let k = (normcdf(u) - normcdf(l)) / (u - l);

        let (lower_slope, lower_offset) = if k < normpdf(l) {
            (k, normcdf(l) - k * l)
        } else {
            // Lower tangent touches d ∈ [l, 0], passes through (u, Φ(u)).
            let d = tangent_through(u, normcdf(u), l, 0.0);
            (normpdf(d), normcdf(d) - normpdf(d) * d)
        };

        let (upper_slope, upper_offset) = if k < normpdf(u) {
            (k, normcdf(l) - k * l)
        } else {
            // Upper tangent touches d ∈ [0, u], passes through (l, Φ(l)).
            let d = tangent_through(l, normcdf(l), 0.0, u);
            (normpdf(d), normcdf(d) - normpdf(d) * d)
        };
        (lower_slope, lower_offset, upper_slope, upper_offset)
    }
}

pub(crate) fn reciprocal_lines(l: f64, u: f64) -> (f64, f64, f64, f64) {
    if u - l < 1e-9 {
        // degenerate: flat line through the point
        (0.0, l.recip(), 0.0, l.recip())
    } else {
        // https://openreview.net/pdf?id=BJxwPJHFwS
        let d = (l + u) / 2.0;
        let tan_slope = -1.0 / (d * d);
        let tan_offset = d.recip() - tan_slope * d;

        let (sec_slope, sec_offset) = secant(l, u, l.recip(), u.recip());

        if l > 0.0 {
            // convex: secant above, tangent below
            (tan_slope, tan_offset, sec_slope, sec_offset)
        } else {
            // concave: secant below, tangent above
            (sec_slope, sec_offset, tan_slope, tan_offset)
        }
    }
}

pub(crate) fn exp_lines(l: f64, u: f64) -> (f64, f64, f64, f64) {
    // https://openreview.net/pdf?id=BJxwPJHFwS
    let (upper_slope, upper_offset) = if u - l < 1e-9 {
        // degenerate: flat line through the point
        (0.0, l.exp())
    } else {
        secant(l, u, l.exp(), u.exp())
    };

    let d = ((l + u) / 2.0).min(l + 0.99); // for numerical stability
    let (lower_slope, lower_offset) = (d.exp(), d.exp() - d.exp() * d);

    (lower_slope, lower_offset, upper_slope, upper_offset)
}

pub(crate) fn square_lines(l: f64, u: f64) -> (f64, f64, f64, f64) {
    // σ(x) = x² is convex: upper = secant, lower = tangent at d.
    // https://openreview.net/pdf?id=BJxwPJHFwS
    let (upper_slope, upper_offset) = if u - l < 1e-9 {
        // degenerate: flat line through the point
        (0.0, l * l)
    } else {
        secant(l, u, l * l, u * u)
    };

    // The lower line feeds Sqrt (non-negative domain), so we pick the tangent
    // point d so that σ_L(x) = 2d·x − d² ≥ 0 across [l, u], preferring the
    // midpoint. The constraint requires d ∈ [2u, 0] for u ≤ 0 and d ∈ [0, 2l]
    // for l ≥ 0 (the paper's "max" for the l ≥ 0 case is a typo; d ≤ 2l ⇒ min).
    let mid = (l + u) / 2.0;
    let d = if u <= 0.0 {
        mid.max(2.0 * u)
    } else if l >= 0.0 {
        mid.min(2.0 * l)
    } else {
        // Any nonzero tangent dips below 0 at x = 0, so d = 0 (lower line y = 0).
        0.0
    };
    let (lower_slope, lower_offset) = (2.0 * d, -d * d);

    (lower_slope, lower_offset, upper_slope, upper_offset)
}

pub(crate) fn sqrt_lines(l: f64, u: f64) -> (f64, f64, f64, f64) {
    // σ(x) = √x is concave on x ≥ 0: lower = secant, upper = midpoint tangent.
    // Precondition: l ≥ 0 (enforced by the caller).
    // https://openreview.net/pdf?id=BJxwPJHFwS
    if u - l < 1e-9 {
        // degenerate: flat line through the point
        return (0.0, l.sqrt(), 0.0, l.sqrt());
    }

    let (lower_slope, lower_offset) = secant(l, u, l.sqrt(), u.sqrt());

    // tangent at the midpoint: σ(m) + σ'(m)(x − m), σ'(m) = 1/(2√m)
    let m = (l + u) / 2.0;
    let upper_slope = 0.5 / m.sqrt();
    let upper_offset = m.sqrt() - upper_slope * m;

    (lower_slope, lower_offset, upper_slope, upper_offset)
}
