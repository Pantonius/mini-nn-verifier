use std::{f64::consts::SQRT_2, iter::zip};

use ndarray::{Array1, Array2};
use rand::{RngExt, SeedableRng, rngs::Xoshiro128PlusPlus};
use rand_distr::Binomial;

use crate::mininn::MininnError;

pub struct Layer {
    pub w: Array2<f64>, // (fan_in, fan_out)
    pub b: Array1<f64>, // (fan_out,)
}

pub fn init_mlp(
    in_size: usize,
    layer_sizes: Vec<usize>,
    rng_key: u64,
) -> Result<Vec<Layer>, MininnError> {
    let mut sizes = layer_sizes.clone();
    sizes.insert(0, in_size);
    sizes.pop();

    let mut rng = Xoshiro128PlusPlus::seed_from_u64(rng_key);
    let distr = Binomial::new(layer_sizes.len().try_into().unwrap(), 0.5)?;

    let rng_keys: Vec<u64> = (0..layer_sizes.len()).map(|_| rng.sample(distr)).collect();

    zip(zip(sizes, layer_sizes), rng_keys)
        .map(
            |((in_, out), key): ((usize, usize), u64)| -> Result<Layer, MininnError> {
                let unif = kaiming_uniform((in_, out), in_, key)?;
                Ok(Layer {
                    w: unif,
                    b: Array1::zeros((out,)),
                })
            },
        )
        .collect()
}

fn kaiming_uniform(
    shape: (usize, usize),
    fan_in: usize,
    rng_key: u64,
) -> Result<Array2<f64>, MininnError> {
    let bound = SQRT_2 * (3.0 / fan_in as f64).sqrt();

    // arbitrary choice of RNG for now
    let mut rng = Xoshiro128PlusPlus::seed_from_u64(rng_key);

    let distr = rand::distr::Uniform::new_inclusive(-bound, bound)?;

    Ok(Array2::from_shape_fn(shape, |_| rng.sample(distr)))
}
