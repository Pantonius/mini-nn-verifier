use crate::interpreters::concrete::eval_util::*;
use crate::mininn::Value;
use crate::{
    interpreters::{EvalError, Interpreter},
    mininn::{Atom, ComputeGraph, Env, Primitive},
};

pub struct EvalInterpreter;

impl EvalInterpreter {
    pub fn new() -> Self {
        EvalInterpreter
    }

    pub fn process_primitive(
        primitive: &Primitive,
        env: &Env<Tensor>,
    ) -> Result<Tensor, EvalError> {
        let r = |a: &Atom| env.resolve(a);

        use Primitive::*;
        Ok(match primitive {
            // elementwise unary
            Neg(a) => -r(a)?,
            Reciprocal(a) => r(a)?.reciprocal(),
            Square(a) => r(a)?.square(),
            Sqrt(a) => r(a)?.sqrt(),
            Exp(a) => r(a)?.exp(),
            Log(a) => r(a)?.log(),
            // elementwise binary (numpy broadcasting)
            Add(a, b) => r(a)? + r(b)?,
            Mul(a, b) => r(a)? * r(b)?,
            Where(c, x, y) => Tensor::r#where(&r(c)?, &r(x)?, &r(y)?)?,
            // activations
            Relu(a) => r(a)?.relu(),
            LeakyRelu { operand, slope } => r(operand)?.leaky_relu(*slope),

            Elu { operand, slope } => r(operand)?.elu(*slope),
            Gelu(a) => r(a)?.gelu(),
            NormalCdf(a) => r(a)?.normcdf(),
            // linear algebra
            Dot(a, b) => r(a)?.dot(&r(b)?)?,
            // reduction
            ReduceSum { operand, axes } => r(operand)?.reduce_sum(axes),
            // shape manipulation
            ExpandDims { operand, axes } => r(operand)?.expand_dims(axes),
            MoveAxis {
                operand,
                source,
                destination,
            } => r(operand)?.moveaxis(*source, *destination),
            Reshape { operand, new_shape } => r(operand)?.reshape(&new_shape)?,
            // padding
            Pad { operand, options } => r(operand)?.pad(options),
            // 2d convolution
            Conv {
                input,
                kernel,
                options,
            } => r(&input)?.conv(&r(&kernel)?, options.stride)?,
            // pooling
            AvgPool { operand, options } => r(operand)?.pool(options, true)?,
            SumPool { operand, options } => r(operand)?.pool(options, false)?,
        })
    }
}

impl Interpreter<Tensor> for EvalInterpreter {
    /// Evaluate `graph` on `inputs` (one flat buffer per input var, in graph
    /// order) and return the output tensors flattened in row-major order.
    fn run(graph: &ComputeGraph, inputs: &Vec<Tensor>) -> Result<Vec<Tensor>, EvalError> {
        let mut env = Env::new();

        for (var, tensor) in graph.invars.iter().zip(inputs) {
            env.insert(var.name.clone(), tensor.clone());
        }

        for eqn in &graph.equations {
            let out = Self::process_primitive(&eqn.primitive, &env)?;
            env.insert(eqn.outvar.name.clone(), out);
        }

        graph
            .outvars
            .iter()
            .map(|var| {
                let tensor = env.get(&var.name).ok_or_else(|| {
                    EvalError::Eval(format!("output '{}' was never computed", var.name))
                })?;
                Ok(tensor.clone())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::mininn::{PaddingOptionConfig, PaddingOptions, PoolOptions};

    use super::*;
    use ndarray::{ArrayD, IxDyn, arr0, array, s};

    // --- norm_axis_index tests ---

    #[test]
    fn norm_axis_positive() {
        assert_eq!(norm_axis_index(0, 4), 0);
        assert_eq!(norm_axis_index(3, 4), 3);
    }

    #[test]
    fn norm_axis_negative() {
        assert_eq!(norm_axis_index(-1, 4), 3);
        assert_eq!(norm_axis_index(-4, 4), 0);
    }

    // --- broadcast_shape tests ---

    #[test]
    fn broadcast_same_shape() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn broadcast_left_pad() {
        assert_eq!(broadcast_shape(&[3], &[2, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn broadcast_size1_expansion() {
        assert_eq!(broadcast_shape(&[1, 3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[2, 1], &[1, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn broadcast_scalar() {
        assert_eq!(broadcast_shape(&[], &[2, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn broadcast_incompatible() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 4]), None);
    }

    // --- binary tests ---

    #[test]
    fn binary_add_elementwise() {
        let a: Tensor = array![1.0, 2.0, 3.0].into_dyn().into();
        let b: Tensor = array![4.0, 5.0, 6.0].into_dyn().into();
        let result = binary(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(result, array![5.0, 7.0, 9.0].into_dyn().into());
    }

    #[test]
    fn binary_broadcast() {
        // [2,1] + [3] → [2,3]
        let a: Tensor = array![[1.0], [2.0]].into_dyn().into();
        let b: Tensor = array![10.0, 20.0, 30.0].into_dyn().into();
        let result = binary(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 11.0);
        assert_eq!(result[[1, 2]], 32.0);
    }

    #[test]
    fn binary_incompatible_error() {
        let a: Tensor = ArrayD::zeros(IxDyn(&[2])).into();
        let b: Tensor = ArrayD::zeros(IxDyn(&[3])).into();
        assert!(binary(&a, &b, |x, y| x + y).is_err());
    }

    // --- where tests ---

    #[test]
    fn where_basic() {
        let cond: Tensor = array![1.0, 0.0, 1.0].into_dyn().into();
        let x: Tensor = array![10.0, 20.0, 30.0].into_dyn().into();
        let y: Tensor = array![40.0, 50.0, 60.0].into_dyn().into();
        let result = Tensor::r#where(&cond, &x, &y).unwrap();
        assert_eq!(result, array![10.0, 50.0, 30.0].into_dyn().into());
    }

    #[test]
    fn where_broadcast_scalar_branches() {
        let cond: Tensor = array![1.0, 0.0].into_dyn().into();
        let x: Tensor = arr0(5.0).into_dyn().into();
        let y: Tensor = arr0(0.0).into_dyn().into();
        let result = Tensor::r#where(&cond, &x, &y).unwrap();
        assert_eq!(result, array![5.0, 0.0].into_dyn().into());
    }

    // --- reshape_c tests ---

    #[test]
    fn reshape_c_flatten() {
        let a: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into_dyn().into();
        let result = reshape_c(&a, &[4]);
        assert_eq!(result, array![1.0, 2.0, 3.0, 4.0].into_dyn().into());
    }

    #[test]
    fn reshape_c_unflatten() {
        let a: Tensor = array![1.0, 2.0, 3.0, 4.0].into_dyn().into();
        let result = reshape_c(&a, &[2, 2]);
        assert_eq!(result, array![[1.0, 2.0], [3.0, 4.0]].into_dyn().into());
    }

    // --- moveaxis tests ---

    #[test]
    fn moveaxis_first_to_last() {
        // [2,3,4]: move axis 0 to 2 → permutation [1,2,0] → shape [3,4,2]
        let a: Tensor = ArrayD::zeros(IxDyn(&[2, 3, 4])).into();
        let result = a.moveaxis(0, 2);
        assert_eq!(result.shape(), &[3, 4, 2]);
    }

    #[test]
    fn moveaxis_last_to_first() {
        // [2,3,4]: move axis -1 to 0 → permutation [2,0,1] → shape [4,2,3]
        let a: Tensor = ArrayD::zeros(IxDyn(&[2, 3, 4])).into();
        let result = a.moveaxis(-1, 0);
        assert_eq!(result.shape(), &[4, 2, 3]);
    }

    #[test]
    fn moveaxis_preserves_values() {
        // verify element [0,1,2] ends up at [1,2,0] after moving axis 0 → 2
        let a: Tensor = ArrayD::from_shape_fn(IxDyn(&[2, 3, 4]), |idx| {
            (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
        })
        .into();
        let result = a.moveaxis(0, 2);
        // original [0,1,2] = 012.0 should now be at [1,2,0]
        assert_eq!(result[[1, 2, 0]], 12.0);
    }

    #[test]
    fn dot_1d_1d() {
        let a: Tensor = array![1.0, 2.0, 3.0].into_dyn().into();
        let b: Tensor = array![4.0, 5.0, 6.0].into_dyn().into();
        let result = a.dot(&b).unwrap();
        assert_eq!(result.ndim(), 0);
        assert_eq!(result[[]], 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn dot_2d_2d() {
        let a: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into_dyn().into();
        let b: Tensor = array![[5.0, 6.0], [7.0, 8.0]].into_dyn().into();
        let result = a.dot(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 19.0); // 1*5 + 2*7
        assert_eq!(result[[1, 1]], 50.0); // 3*6 + 4*8
    }

    #[test]
    fn dot_2d_1d() {
        let a: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into_dyn().into();
        let b: Tensor = array![1.0, 1.0].into_dyn().into();
        let result = a.dot(&b).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result[[0]], 3.0);
        assert_eq!(result[[1]], 7.0);
    }

    #[test]
    fn dot_nd_1d() {
        // a: [2, 3, 4], b: [4] → result: [2, 3]
        let a: Tensor = ArrayD::ones(IxDyn(&[2, 3, 4])).into();
        let b: Tensor = ArrayD::from_elem(IxDyn(&[4]), 2.0).into();
        let result = a.dot(&b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert!(result.iter().all(|&x| x == 8.0)); // sum of 4 * (1*2)
    }

    #[test]
    fn dot_nd_md() {
        // a: [2, 3], b: [3, 4] → result: [2, 4]
        let a: Tensor = ArrayD::ones(IxDyn(&[2, 3])).into();
        let b: Tensor = ArrayD::ones(IxDyn(&[3, 4])).into();
        let result = a.dot(&b).unwrap();
        assert_eq!(result.shape(), &[2, 4]);
        assert!(result.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn dot_scalar() {
        let a: Tensor = arr0(3.0).into_dyn().into();
        let b: Tensor = ArrayD::from_elem(IxDyn(&[2, 3]), 2.0).into();
        let result = a.dot(&b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert!(result.iter().all(|&x| x == 6.0));
    }

    #[test]
    fn dot_1d_1d_mismatch() {
        let a: Tensor = ArrayD::ones(IxDyn(&[3])).into();
        let b: Tensor = ArrayD::ones(IxDyn(&[4])).into();
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn dot_2d_1d_mismatch() {
        let a: Tensor = ArrayD::ones(IxDyn(&[2, 3])).into();
        let b: Tensor = ArrayD::ones(IxDyn(&[4])).into();
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn dot_2d_2d_mismatch() {
        let a: Tensor = ArrayD::ones(IxDyn(&[2, 3])).into();
        let b: Tensor = ArrayD::ones(IxDyn(&[4, 2])).into();
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn dot_nd_1d_mismatch() {
        let a: Tensor = ArrayD::ones(IxDyn(&[2, 3, 4])).into();
        let b: Tensor = ArrayD::ones(IxDyn(&[5])).into();
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn dot_nd_md_mismatch() {
        let a: Tensor = ArrayD::ones(IxDyn(&[2, 3])).into();
        let b: Tensor = ArrayD::ones(IxDyn(&[4, 5])).into(); // second-to-last is 4, not 3
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn reduce_sum_1d() {
        let a: Tensor = array![1.0, 2.0, 3.0].into_dyn().into();
        let result = a.reduce_sum(&[0]);
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(result[[]], 6.0);
    }

    #[test]
    fn reduce_sum_3d_middle_axis() {
        let a: Tensor = ArrayD::ones(IxDyn(&[2, 3, 4])).into();
        let result = a.reduce_sum(&[1]);
        assert_eq!(result.shape(), &[2, 4]);
        assert!(result.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn reduce_sum_3d_multi_axes() {
        let a: Tensor = ArrayD::ones(IxDyn(&[2, 3, 4])).into();
        let result = a.reduce_sum(&[0, 2]);
        assert_eq!(result.shape(), &[3]);
        assert!(result.iter().all(|&x| x == 8.0)); // 2 * 4 * 1
    }

    #[test]
    fn expand_dims_prepend() {
        let a: Tensor = ArrayD::ones(IxDyn(&[3, 4])).into();
        let result = a.expand_dims(&[0]);
        assert_eq!(result.shape(), &[1, 3, 4]);
    }

    #[test]
    fn expand_dims_insert() {
        let a: Tensor = ArrayD::ones(IxDyn(&[3, 4])).into();
        let result = a.expand_dims(&[1]);
        assert_eq!(result.shape(), &[3, 1, 4]);
    }

    #[test]
    fn expand_dims_multi_axes_neg() {
        let a: Tensor = ArrayD::ones(IxDyn(&[3, 4])).into();
        let result = a.expand_dims(&[0, -1]);
        assert_eq!(result.shape(), &[1, 3, 4, 1]);
    }

    #[test]
    fn reshape_flatten() {
        let a: Tensor = ArrayD::ones(IxDyn(&[2, 3])).into();
        assert!(a.reshape(&[6]).is_ok())
    }

    #[test]
    fn reshape_unflatten() {
        let a: Tensor = ArrayD::ones(IxDyn(&[6])).into();
        assert!(a.reshape(&[2, -1]).is_ok())
    }

    #[test]
    fn reshape_mismatch() {
        let a: Tensor = ArrayD::ones(IxDyn(&[6])).into();
        assert!(a.reshape(&[2, 4]).is_err())
    }

    // --- conv tests ---

    #[test]
    fn conv_identity_kernel() {
        // 1x1 all-ones kernel is the identity for a single channel
        let input: Tensor =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), (1..=9).map(|x| x as f64).collect())
                .unwrap()
                .into();
        let kernel: Tensor = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0])
            .unwrap()
            .into();
        let result = input.conv(&kernel, 1).unwrap();
        assert_eq!(result.shape(), &[1, 1, 3, 3]);
        assert_eq!(result, input);
    }

    #[test]
    fn conv_single_channel_sum_kernel() {
        // 2x2 all-ones kernel produces the sliding 2x2 sum
        let input: Tensor =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), (1..=9).map(|x| x as f64).collect())
                .unwrap()
                .into();
        let kernel: Tensor = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0; 4])
            .unwrap()
            .into();
        let result = input.conv(&kernel, 1).unwrap();
        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        assert_eq!(result[[0, 0, 0, 0]], 12.0); // 1+2+4+5
        assert_eq!(result[[0, 0, 0, 1]], 16.0); // 2+3+5+6
        assert_eq!(result[[0, 0, 1, 0]], 24.0); // 4+5+7+8
        assert_eq!(result[[0, 0, 1, 1]], 28.0); // 5+6+8+9
    }

    #[test]
    fn conv_multi_output_channels() {
        // kernel shape [2,1,2,2]: two different output filters on one input channel
        let input: Tensor =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), (1..=9).map(|x| x as f64).collect())
                .unwrap()
                .into();
        // output channel 0: all ones (sum); output channel 1: top-left only
        let kernel: Tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 1, 2, 2]),
            vec![
                1.0, 1.0, 1.0, 1.0, // oc=0
                1.0, 0.0, 0.0, 0.0, // oc=1
            ],
        )
        .unwrap()
        .into();
        let result = input.conv(&kernel, 1).unwrap();
        assert_eq!(result.shape(), &[1, 2, 2, 2]);
        // channel 0: sliding sum
        assert_eq!(result[[0, 0, 0, 0]], 12.0);
        assert_eq!(result[[0, 0, 0, 1]], 16.0);
        assert_eq!(result[[0, 0, 1, 0]], 24.0);
        assert_eq!(result[[0, 0, 1, 1]], 28.0);
        // channel 1: top-left element of each window
        assert_eq!(result[[0, 1, 0, 0]], 1.0);
        assert_eq!(result[[0, 1, 0, 1]], 2.0);
        assert_eq!(result[[0, 1, 1, 0]], 4.0);
        assert_eq!(result[[0, 1, 1, 1]], 5.0);
    }

    #[test]
    fn conv_multi_input_channels() {
        // kernel shape [1,2,2,2]: single output channel, sums over two input channels
        let ch0: Vec<f64> = (1..=9).map(|x| x as f64).collect(); // 1..9
        let ch1 = vec![1.0f64; 9]; // all ones
        let input_data: Vec<f64> = ch0.into_iter().chain(ch1).collect();
        let input: Tensor = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3, 3]), input_data)
            .unwrap()
            .into();
        // ic=0: pick only top-left; ic=1: all ones (2x2 sum over the all-ones channel)
        let kernel: Tensor = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 2, 2]),
            vec![
                1.0, 0.0, 0.0, 0.0, // ic=0
                1.0, 1.0, 1.0, 1.0, // ic=1
            ],
        )
        .unwrap()
        .into();
        let result = input.conv(&kernel, 1).unwrap();
        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        // [0,0]: ch0 contrib=1, ch1 contrib=4 → 5
        assert_eq!(result[[0, 0, 0, 0]], 5.0);
        // [0,1]: ch0=2, ch1=4 → 6
        assert_eq!(result[[0, 0, 0, 1]], 6.0);
        // [1,0]: ch0=4, ch1=4 → 8
        assert_eq!(result[[0, 0, 1, 0]], 8.0);
        // [1,1]: ch0=5, ch1=4 → 9
        assert_eq!(result[[0, 0, 1, 1]], 9.0);
    }

    #[test]
    fn conv_stride_2() {
        // stride=2 skips alternate positions
        let input: Tensor =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 4, 4]), (1..=16).map(|x| x as f64).collect())
                .unwrap()
                .into();
        // kernel picks only the top-left element of each window
        let kernel: Tensor = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 0.0, 0.0, 0.0])
            .unwrap()
            .into();
        let result = input.conv(&kernel, 2).unwrap();
        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        assert_eq!(result[[0, 0, 0, 0]], 1.0); // input[0,0,0,0]
        assert_eq!(result[[0, 0, 0, 1]], 3.0); // input[0,0,0,2]
        assert_eq!(result[[0, 0, 1, 0]], 9.0); // input[0,0,2,0]
        assert_eq!(result[[0, 0, 1, 1]], 11.0); // input[0,0,2,2]
    }

    #[test]
    fn conv_batch_size_2() {
        // two batch items are processed independently
        let input: Tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 1, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap()
        .into();
        let kernel: Tensor = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![2.0])
            .unwrap()
            .into();
        let result = input.conv(&kernel, 1).unwrap();
        assert_eq!(result.shape(), &[2, 1, 2, 2]);
        assert_eq!(result[[0, 0, 0, 0]], 2.0);
        assert_eq!(result[[0, 0, 1, 1]], 8.0);
        assert_eq!(result[[1, 0, 0, 0]], 10.0);
        assert_eq!(result[[1, 0, 1, 1]], 16.0);
    }

    #[test]
    fn conv_channel_mismatch_error() {
        let input: Tensor = ArrayD::zeros(IxDyn(&[1, 3, 4, 4])).into(); // 3 input channels
        let kernel: Tensor = ArrayD::zeros(IxDyn(&[2, 4, 2, 2])).into(); // expects 4 input channels
        assert!(input.conv(&kernel, 1).is_err());
    }

    #[test]
    fn conv_non_4d_input_error() {
        let input: Tensor = ArrayD::zeros(IxDyn(&[1, 3, 4])).into();
        let kernel: Tensor = ArrayD::zeros(IxDyn(&[2, 3, 2, 2])).into();
        assert!(input.conv(&kernel, 1).is_err());
    }

    #[test]
    fn conv_non_4d_kernel_error() {
        let input: Tensor = ArrayD::zeros(IxDyn(&[1, 3, 4, 4])).into();
        let kernel: Tensor = ArrayD::zeros(IxDyn(&[2, 3, 2])).into();
        assert!(input.conv(&kernel, 1).is_err());
    }

    // --- pool tests ---

    #[test]
    fn pool_1d_sum_stride1() {
        // [1,2,3,4] with window=2, stride=1 → [3,5,7]
        let a: Tensor = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .into();
        let opt = PoolOptions {
            window_size: vec![2],
            stride: vec![1],
        };
        let result = a.pool(&opt, false).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result[[0]], 3.0);
        assert_eq!(result[[1]], 5.0);
        assert_eq!(result[[2]], 7.0);
    }

    #[test]
    fn pool_1d_avg_stride1() {
        // same windows, averaged
        let a: Tensor = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .into();
        let opt = PoolOptions {
            window_size: vec![2],
            stride: vec![1],
        };
        let result = a.pool(&opt, true).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result[[0]], 1.5);
        assert_eq!(result[[1]], 2.5);
        assert_eq!(result[[2]], 3.5);
    }

    #[test]
    fn pool_1d_sum_stride2() {
        // [1,2,3,4] with window=2, stride=2 → [3,7]
        let a: Tensor = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .into();
        let opt = PoolOptions {
            window_size: vec![2],
            stride: vec![2],
        };
        let result = a.pool(&opt, false).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result[[0]], 3.0); // 1+2
        assert_eq!(result[[1]], 7.0); // 3+4
    }

    #[test]
    fn pool_2d_sum_stride1() {
        // 3x3 input, 2x2 window, stride 1 → 2x2 output; same values as conv sum-kernel test
        let a: Tensor = ArrayD::from_shape_vec(IxDyn(&[3, 3]), (1..=9).map(|x| x as f64).collect())
            .unwrap()
            .into();
        let opt = PoolOptions {
            window_size: vec![2, 2],
            stride: vec![1, 1],
        };
        let result = a.pool(&opt, false).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 12.0); // 1+2+4+5
        assert_eq!(result[[0, 1]], 16.0); // 2+3+5+6
        assert_eq!(result[[1, 0]], 24.0); // 4+5+7+8
        assert_eq!(result[[1, 1]], 28.0); // 5+6+8+9
    }

    #[test]
    fn pool_2d_avg_stride1() {
        let a: Tensor = ArrayD::from_shape_vec(IxDyn(&[3, 3]), (1..=9).map(|x| x as f64).collect())
            .unwrap()
            .into();
        let opt = PoolOptions {
            window_size: vec![2, 2],
            stride: vec![1, 1],
        };
        let result = a.pool(&opt, true).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 3.0); // 12/4
        assert_eq!(result[[0, 1]], 4.0); // 16/4
        assert_eq!(result[[1, 0]], 6.0); // 24/4
        assert_eq!(result[[1, 1]], 7.0); // 28/4
    }

    #[test]
    fn pool_2d_sum_stride2() {
        // 4x4 input, 2x2 window, stride 2 → 2x2 output
        let a: Tensor =
            ArrayD::from_shape_vec(IxDyn(&[4, 4]), (1..=16).map(|x| x as f64).collect())
                .unwrap()
                .into();
        let opt = PoolOptions {
            window_size: vec![2, 2],
            stride: vec![2, 2],
        };
        let result = a.pool(&opt, false).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 14.0); // 1+2+5+6
        assert_eq!(result[[0, 1]], 22.0); // 3+4+7+8
        assert_eq!(result[[1, 0]], 46.0); // 9+10+13+14
        assert_eq!(result[[1, 1]], 54.0); // 11+12+15+16
    }

    #[test]
    fn pad_single_axis() {
        let a: Tensor = ArrayD::ones(IxDyn(&[2, 3, 4])).into();
        let result = a.pad(&PaddingOptions {
            config: PaddingOptionConfig {
                left: 2,
                right: 2,
                interior: 1,
            },
            axes: vec![0],
            value: 0.0,
        });

        let mut expected_arr = ArrayD::from_elem(IxDyn(&[7, 3, 4]), 0.0);
        expected_arr.slice_mut(s![2, .., ..]).fill(1.0);
        expected_arr.slice_mut(s![4, .., ..]).fill(1.0);
        let expected: Tensor = expected_arr.into();

        assert_eq!(result.shape(), &[7, 3, 4]);
        assert_eq!(result, expected);
    }
}
