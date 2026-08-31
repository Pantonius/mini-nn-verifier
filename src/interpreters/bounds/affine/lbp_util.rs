use std::f64::consts::SQRT_2;

use ndarray::{ArrayD, Zip};

use crate::{
    interpreters::{
        EvalError,
        bounds::ibp_util::IBPTensor,
        concrete::eval_util::{Tensor, elu, gelu, normcdf, normpdf},
    },
    mininn::Value,
};

pub fn lbp_inner<T: Value>(w: &T, x: &T) -> T {
    let product = w.clone() * x.clone();
    let axes: Vec<isize> = (0..product.ndim() as isize).collect();
    product.reduce_sum(&axes)
}

// ================================
// AffineBounds (Affine Bounds)
// ================================
#[derive(Debug, Clone)]
pub struct AffineBounds<T: Value> {
    pub weights: T,
    pub biases: T,
}

impl<T: Value> AffineBounds<T> {
    /// Concretize the affine bound given concrete lower/upper bounds on the input.
    pub fn concretize(&self, lb: &T, ub: &T) -> T {
        let pos_w = self.weights.relu();
        let neg_w = self.weights.clone() - pos_w.clone();
        self.biases.clone() + lbp_inner(&pos_w, lb) + lbp_inner(&neg_w, ub)
    }
}

impl AffineBounds<Tensor> {
    pub fn argmin_corner(&self, lb: &Tensor, ub: &Tensor) -> Tensor {
        Zip::from(&self.weights)
            .and(lb)
            .and(ub)
            .map_collect(|&w, &l, &u| if w > 0.0 { l } else { u })
            .into()
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

pub(super) fn bound_convex<T: Value>(
    x: &IBPTensor,
    w: &T,
    f: impl Fn(f64, f64) -> (f64, f64, f64, f64),
    bound_req: impl Fn(f64, f64) -> Result<(), EvalError>,
) -> Result<AffineBounds<T>, EvalError> {
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

    Ok(AffineBounds {
        weights: lower_slope * pos_w.clone() + upper_slope * neg_w.clone(),
        biases: lbp_inner(&lower_offset, &pos_w) + lbp_inner(&upper_offset, &neg_w),
    })
}

pub(super) fn normcdf_lines(l: f64, u: f64) -> (f64, f64, f64, f64) {
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

pub(super) fn gelu_lines(l: f64, u: f64) -> (f64, f64, f64, f64) {
    // TODO maybe do this through grad (better reuse)
    let dg = |x: f64| normcdf(x) + x * normpdf(x);

    let lower_slope;
    let lower_offset;
    let upper_slope;
    let upper_offset;

    if (-SQRT_2 <= l && u <= SQRT_2) || (l <= -SQRT_2 && u >= SQRT_2) {
        // fully convex ( -sqrt(2) <= l <= u <= sqrt(2) ) => lower = midpoint tangent, upper = secant
        let mid = (u + l) / 2.0;
        lower_slope = dg(mid);
        lower_offset = gelu(mid) - lower_slope * mid;

        (upper_slope, upper_offset) = secant(l, u, gelu(l), gelu(u));
    } else if u <= -SQRT_2 || l >= SQRT_2 {
        // fully concave ( u <= -sqrt(2) or l >= sqrt(2) ) => lower = secant, upper = midpoint tangent
        (lower_slope, lower_offset) = secant(l, u, gelu(l), gelu(u));

        let mid = (u + l) / 2.0;
        upper_slope = dg(mid);
        upper_offset = gelu(mid) - upper_slope * mid;
    } else {
        // let check_lower = |upper, d| -> bool {
        //     let k = dg(d);
        //
        //     return k * (upper - d) + gelu(d) <= gelu(upper);
        // };
        // let check_upper = |lower, d| -> bool {
        //     let k = dg(d);
        //
        //     return k * (lower - d) + gelu(d) >= gelu(lower);
        // };
        //
        // // ---------------------------------------------------------------
        // // Lower bound on the right side
        // // ---------------------------------------------------------------
        // const MAX_ITER: usize = 100;
        // const STEP_PRE: f64 = 0.01;
        // const X_LIMIT: f64 = 1000.0;
        // let num_points_pre = (X_LIMIT / STEP_PRE) as usize;
        // let smth = Tensor::from(
        //     Array1::from_vec((0..num_points_pre).map(|x| x as f64).collect::<Vec<_>>()).into_dyn(),
        // );
        // let upper = Tensor::from(STEP_PRE) * smth + SQRT_2.into();
        // let mut r = Tensor::from(ArrayD::ones(upper.shape()));
        // let mut l = Tensor::from(-ArrayD::ones(upper.shape()));
        //
        // loop {
        //     let checked = Zip::from(&upper)
        //         .and(&l)
        //         .map_collect(|&u, &x| check_lower(u, x));
        //
        //     l = Tensor::from(
        //         Zip::from(&l).and(&checked).map_collect(
        //             |&l_i, &ok| {
        //                 if !ok { l_i * 2.0 } else { l_i }
        //             },
        //         ),
        //     );
        //
        //     if checked.iter().all(|&x| x) {
        //         break;
        //     }
        // }
        //
        // // binary search
        // for _ in 0..MAX_ITER {
        //     let m = (l.clone() + r.clone()) / 2.0;
        //
        //     let checked = Zip::from(&upper)
        //         .and(&m)
        //         .map_collect(|&u, &x| check_lower(u, x));
        //
        //     l = Tensor::from(
        //         Zip::from(&l).and(&m).and(&checked).map_collect(
        //             |&l_i, &m_i, &ok| {
        //                 if ok { m_i } else { l_i }
        //             },
        //         ),
        //     );
        //     r = Tensor::from(
        //         Zip::from(&r).and(&m).and(&checked).map_collect(
        //             |&r_i, &m_i, &ok| {
        //                 if ok { r_i } else { m_i }
        //             },
        //         ),
        //     );
        // }
        //
        // let d_lower_right = l.clone();
        //
        // // ---------------------------------------------------------------
        // // Upper bound on the right side
        // // ---------------------------------------------------------------
        // let lower = Array1::from_iter(
        //     (0..(num_points_pre + 5)).map(|i| (-STEP_PRE * i as f64 + SQRT_2).max(0.01)),
        // );
        //
        // // l = sqrt_2
        // let mut l = Array1::from_elem(n, SQRT_2);
        //
        // // r = x_limit
        // let mut r = Array1::from_elem(n, x_limit);
        //
        // loop {
        //     let checked = Zip::from(&lower)
        //         .and(&r)
        //         .map_collect(|&lo, &x| check_upper(lo, x));
        //
        //     // r = checked * r + (1 - checked) * (r * 2)
        //     Zip::from(&mut r).and(&checked).for_each(|r_i, &ok| {
        //         if !ok {
        //             *r_i *= 2.0;
        //         }
        //     });
        //
        //     if checked.iter().all(|&x| x) {
        //         break;
        //     }
        // }
        //
        // for _ in 0..MAX_ITER {
        //     let m = (&l + &r) / 2.0;
        //
        //     let checked = Zip::from(&lower)
        //         .and(&m)
        //         .map_collect(|&lo, &x| check_upper(lo, x));
        //
        //     // l = (1 - checked) * m + checked * l
        //     // r = (1 - checked) * r + checked * m
        //     Zip::from(&mut l)
        //         .and(&mut r)
        //         .and(&m)
        //         .and(&checked)
        //         .for_each(|l_i, r_i, &m_i, &ok| {
        //             if ok {
        //                 *l_i = *l_i;
        //                 *r_i = m_i;
        //             } else {
        //                 *l_i = m_i;
        //             }
        //         });
        // }
        //
        // self.d_upper_right = r.clone();
        //
        // // ---------------------------------------------------------------
        // // Lower bound on the left side
        // // ---------------------------------------------------------------
        //
        // // PyTorch:
        // // upper = -step_pre * arange(...) - sqrt_2
        // let upper = Array1::from_iter((0..n).map(|i| -self.step_pre * i as f32 - sqrt_2));
        //
        // // r = -0.7517916
        // let mut r = Array1::from_elem(n, -0.7517916f32);
        //
        // // l = -sqrt_2
        // let mut l = Array1::from_elem(n, -sqrt_2);
        //
        // loop {
        //     let checked = Zip::from(&upper)
        //         .and(&r)
        //         .map_collect(|&u, &x| check_lower(u, x));
        //
        //     // r = checked * r + (1 - checked) * (r * 2)
        //     Zip::from(&mut r).and(&checked).for_each(|r_i, &ok| {
        //         if !ok {
        //             *r_i *= 2.0;
        //         }
        //     });
        //
        //     if checked.iter().all(|&x| x) {
        //         break;
        //     }
        // }
        //
        // for _ in 0..max_iter {
        //     let m = (&l + &r) / 2.0;
        //
        //     let checked = Zip::from(&upper)
        //         .and(&m)
        //         .map_collect(|&u, &x| check_lower(u, x));
        //
        //     // l = (1 - checked) * m + checked * l
        //     // r = (1 - checked) * r + checked * m
        //     Zip::from(&mut l)
        //         .and(&mut r)
        //         .and(&m)
        //         .and(&checked)
        //         .for_each(|l_i, r_i, &m_i, &ok| {
        //             if ok {
        //                 *r_i = m_i;
        //             } else {
        //                 *l_i = m_i;
        //             }
        //         });
        // }
        //
        // self.d_lower_left = r.clone();
        //
        // // ---------------------------------------------------------------
        // // Upper bound on the left side
        // // ---------------------------------------------------------------
        //
        // // PyTorch:
        // // lower = (step_pre * arange(...) - sqrt_2).clamp(max=0)
        // let lower = Array1::from_iter((0..n).map(|i| (self.step_pre * i as f32 - sqrt_2).min(0.0)));
        //
        // // l = -x_limit
        // let mut l = Array1::from_elem(n, -x_limit);
        //
        // // r = -sqrt_2
        // let mut r = Array1::from_elem(n, -sqrt_2);
        //
        // loop {
        //     let checked = Zip::from(&lower)
        //         .and(&l)
        //         .map_collect(|&lo, &x| check_upper(lo, x));
        //
        //     // l = checked * l + (1 - checked) * (l * 2)
        //     Zip::from(&mut l).and(&checked).for_each(|l_i, &ok| {
        //         if !ok {
        //             *l_i *= 2.0;
        //         }
        //     });
        //
        //     if checked.iter().all(|&x| x) {
        //         break;
        //     }
        // }
        //
        // for _ in 0..max_iter {
        //     let m = (&l + &r) / 2.0;
        //
        //     let checked = Zip::from(&lower)
        //         .and(&m)
        //         .map_collect(|&lo, &x| check_upper(lo, x));
        //
        //     // l = (1 - checked) * m + checked * l
        //     // r = (1 - checked) * r + checked * m
        //     Zip::from(&mut l)
        //         .and(&mut r)
        //         .and(&m)
        //         .and(&checked)
        //         .for_each(|l_i, r_i, &m_i, &ok| {
        //             if ok {
        //                 *r_i = m_i;
        //             } else {
        //                 *l_i = m_i;
        //             }
        //         });
        // }
        //
        // self.d_upper_left = r.clone();

        todo!()
    }

    (lower_slope, lower_offset, upper_slope, upper_offset)
}

pub(super) fn reciprocal_lines(l: f64, u: f64) -> (f64, f64, f64, f64) {
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

pub(super) fn exp_lines(l: f64, u: f64) -> (f64, f64, f64, f64) {
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

pub(super) fn elu_lines(l: f64, u: f64, slope: f64) -> (f64, f64, f64, f64) {
    let deriv = |x: f64| if x >= 0.0 { 1.0 } else { slope * x.exp() };

    let (upper_slope, upper_offset) = if u - l < 1e-9 {
        // degenerate: flat line through the point
        (0.0, elu(l, slope))
    } else {
        secant(l, u, elu(l, slope), elu(u, slope))
    };

    // arbitrary point inbetween (midpoint here)
    let d = (l + u) / 2.0;
    let (lower_slope, lower_offset) = (deriv(d), elu(d, slope) - deriv(d) * d);

    (lower_slope, lower_offset, upper_slope, upper_offset)
}

pub(super) fn square_lines(l: f64, u: f64) -> (f64, f64, f64, f64) {
    // https://openreview.net/pdf?id=BJxwPJHFwS
    let (upper_slope, upper_offset) = if u - l < 1e-9 {
        // degenerate: flat line through the point
        (0.0, l * l)
    } else {
        secant(l, u, l * l, u * u)
    };

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

pub(super) fn sqrt_lines(l: f64, u: f64) -> (f64, f64, f64, f64) {
    // https://openreview.net/pdf?id=BJxwPJHFwS
    if u - l < 1e-9 {
        // degenerate: flat line through the point
        return (0.0, l.sqrt(), 0.0, l.sqrt());
    }

    let (lower_slope, lower_offset) = secant(l, u, l.sqrt(), u.sqrt());

    let m = (l + u) / 2.0;
    let upper_slope = 0.5 / m.sqrt();
    let upper_offset = m.sqrt() - upper_slope * m;

    (lower_slope, lower_offset, upper_slope, upper_offset)
}
