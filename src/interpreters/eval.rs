use crate::interpreters::eval_util::*;
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
            Neg(a) => r(a)?.mapv(|x| -x),
            Reciprocal(a) => r(a)?.recip(),
            Square(a) => r(a)?.mapv(|x| x * x),
            Sqrt(a) => r(a)?.sqrt(),
            Exp(a) => r(a)?.exp(),
            Log(a) => log(&r(a)?),
            // elementwise binary (numpy broadcasting)
            Add(a, b) => add(&r(a)?, &r(b)?)?,
            Mul(a, b) => mul(&r(a)?, &r(b)?)?,
            Where(c, x, y) => where_(&r(c)?, &r(x)?, &r(y)?)?,
            // activations
            Relu(a) => relu(&r(a)?),
            LeakyRelu { operand, slope } => leaky_relu(&r(operand)?, *slope),
            Elu { operand, slope } => elu(&r(operand)?, *slope),
            Gelu(a) => gelu(&r(a)?),
            NormalCdf(a) => r(a)?.mapv(|x| normcdf(x)),
            // linear algebra
            Dot(a, b) => dot(&r(a)?, &r(b)?)?,
            // reduction
            ReduceSum { operand, axes } => reduce_sum(&r(operand)?, axes),
            // shape manipulation
            ExpandDims { operand, axes } => expand_dims(&r(operand)?, axes),
            MoveAxis {
                operand,
                source,
                destination,
            } => moveaxis(&r(operand)?, *source, *destination),
            Reshape { operand, new_shape } => reshape(&r(operand)?, &new_shape)?,
            // padding
            Pad { operand, options } => pad(&r(operand)?, options),
            // 2d convolution
            Conv {
                input,
                kernel,
                options,
            } => conv(&r(&input)?, &r(&kernel)?, options.stride)?,
            // pooling
            AvgPool { operand, options } => pool(&r(operand)?, options, true)?,
            SumPool { operand, options } => pool(&r(operand)?, options, false)?,
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
        // shorter shape is left-padded with 1s
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
        let a = array![1.0, 2.0, 3.0].into_dyn();
        let b = array![4.0, 5.0, 6.0].into_dyn();
        let result = binary(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(result, array![5.0, 7.0, 9.0].into_dyn());
    }

    #[test]
    fn binary_broadcast() {
        // [2,1] + [3] → [2,3]
        let a = array![[1.0], [2.0]].into_dyn();
        let b = array![10.0, 20.0, 30.0].into_dyn();
        let result = binary(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 11.0);
        assert_eq!(result[[1, 2]], 32.0);
    }

    #[test]
    fn binary_incompatible_error() {
        let a = ArrayD::zeros(IxDyn(&[2]));
        let b = ArrayD::zeros(IxDyn(&[3]));
        assert!(binary(&a, &b, |x, y| x + y).is_err());
    }

    // --- where_ tests ---

    #[test]
    fn where_basic() {
        let cond = array![1.0, 0.0, 1.0].into_dyn();
        let x = array![10.0, 20.0, 30.0].into_dyn();
        let y = array![40.0, 50.0, 60.0].into_dyn();
        let result = where_(&cond, &x, &y).unwrap();
        assert_eq!(result, array![10.0, 50.0, 30.0].into_dyn());
    }

    #[test]
    fn where_broadcast_scalar_branches() {
        // scalar x and y broadcast to cond's shape
        let cond = array![1.0, 0.0].into_dyn();
        let x = arr0(5.0).into_dyn();
        let y = arr0(0.0).into_dyn();
        let result = where_(&cond, &x, &y).unwrap();
        assert_eq!(result, array![5.0, 0.0].into_dyn());
    }

    // --- reshape_c tests ---

    #[test]
    fn reshape_c_flatten() {
        let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let result = reshape_c(&a, &[4]);
        assert_eq!(result, array![1.0, 2.0, 3.0, 4.0].into_dyn());
    }

    #[test]
    fn reshape_c_unflatten() {
        let a = array![1.0, 2.0, 3.0, 4.0].into_dyn();
        let result = reshape_c(&a, &[2, 2]);
        assert_eq!(result, array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
    }

    // --- moveaxis tests ---

    #[test]
    fn moveaxis_first_to_last() {
        // [2,3,4]: move axis 0 to 2 → permutation [1,2,0] → shape [3,4,2]
        let a = ArrayD::zeros(IxDyn(&[2, 3, 4]));
        let result = moveaxis(&a, 0, 2);
        assert_eq!(result.shape(), &[3, 4, 2]);
    }

    #[test]
    fn moveaxis_last_to_first() {
        // [2,3,4]: move axis -1 to 0 → permutation [2,0,1] → shape [4,2,3]
        let a = ArrayD::zeros(IxDyn(&[2, 3, 4]));
        let result = moveaxis(&a, -1, 0);
        assert_eq!(result.shape(), &[4, 2, 3]);
    }

    #[test]
    fn moveaxis_preserves_values() {
        // verify element [0,1,2] ends up at [1,2,0] after moving axis 0 → 2
        let a = ArrayD::from_shape_fn(IxDyn(&[2, 3, 4]), |idx| {
            (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
        });
        let result = moveaxis(&a, 0, 2);
        // original [0,1,2] = 012.0 should now be at [1,2,0]
        assert_eq!(result[[1, 2, 0]], 12.0);
    }

    #[test]
    fn dot_1d_1d() {
        let a = array![1.0, 2.0, 3.0].into_dyn();
        let b = array![4.0, 5.0, 6.0].into_dyn();
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.ndim(), 0);
        assert_eq!(result[[]], 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn dot_2d_2d() {
        let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let b = array![[5.0, 6.0], [7.0, 8.0]].into_dyn();
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 19.0); // 1*5 + 2*7
        assert_eq!(result[[1, 1]], 50.0); // 3*6 + 4*8
    }

    #[test]
    fn dot_2d_1d() {
        let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let b = array![1.0, 1.0].into_dyn();
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result[[0]], 3.0);
        assert_eq!(result[[1]], 7.0);
    }

    #[test]
    fn dot_nd_1d() {
        // a: [2, 3, 4], b: [4] → result: [2, 3]
        let a = ArrayD::ones(IxDyn(&[2, 3, 4]));
        let b = ArrayD::from_elem(IxDyn(&[4]), 2.0);
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert!(result.iter().all(|&x| x == 8.0)); // sum of 4 * (1*2)
    }

    #[test]
    fn dot_nd_md() {
        // a: [2, 3], b: [3, 4] → result: [2, 4]
        let a = ArrayD::ones(IxDyn(&[2, 3]));
        let b = ArrayD::ones(IxDyn(&[3, 4]));
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 4]);
        assert!(result.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn dot_scalar() {
        let a = arr0(3.0).into_dyn();
        let b = ArrayD::from_elem(IxDyn(&[2, 3]), 2.0);
        let result = dot(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert!(result.iter().all(|&x| x == 6.0));
    }

    #[test]
    fn dot_1d_1d_mismatch() {
        let a = ArrayD::ones(IxDyn(&[3]));
        let b = ArrayD::ones(IxDyn(&[4]));
        assert!(dot(&a, &b).is_err());
    }

    #[test]
    fn dot_2d_1d_mismatch() {
        let a = ArrayD::ones(IxDyn(&[2, 3]));
        let b = ArrayD::ones(IxDyn(&[4]));
        assert!(dot(&a, &b).is_err());
    }

    #[test]
    fn dot_2d_2d_mismatch() {
        let a = ArrayD::ones(IxDyn(&[2, 3]));
        let b = ArrayD::ones(IxDyn(&[4, 2]));
        assert!(dot(&a, &b).is_err());
    }

    #[test]
    fn dot_nd_1d_mismatch() {
        let a = ArrayD::ones(IxDyn(&[2, 3, 4]));
        let b = ArrayD::ones(IxDyn(&[5]));
        assert!(dot(&a, &b).is_err());
    }

    #[test]
    fn dot_nd_md_mismatch() {
        let a = ArrayD::ones(IxDyn(&[2, 3]));
        let b = ArrayD::ones(IxDyn(&[4, 5])); // second-to-last is 4, not 3
        assert!(dot(&a, &b).is_err());
    }

    #[test]
    fn reduce_sum_1d() {
        let a = array![1.0, 2.0, 3.0].into_dyn();
        let result = reduce_sum(&a, &[0]);
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(result[[]], 6.0);
    }

    #[test]
    fn reduce_sum_3d_middle_axis() {
        let a = ArrayD::ones(IxDyn(&[2, 3, 4]));
        let result = reduce_sum(&a, &[1]);
        assert_eq!(result.shape(), &[2, 4]);
        assert!(result.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn reduce_sum_3d_multi_axes() {
        let a = ArrayD::ones(IxDyn(&[2, 3, 4]));
        let result = reduce_sum(&a, &[0, 2]);
        assert_eq!(result.shape(), &[3]);
        assert!(result.iter().all(|&x| x == 8.0)); // 2 * 4 * 1
    }

    #[test]
    fn expand_dims_prepend() {
        let a = ArrayD::ones(IxDyn(&[3, 4]));
        let result = expand_dims(&a, &[0]);
        assert_eq!(result.shape(), &[1, 3, 4]);
    }

    #[test]
    fn expand_dims_insert() {
        let a = ArrayD::ones(IxDyn(&[3, 4]));
        let result = expand_dims(&a, &[1]);
        assert_eq!(result.shape(), &[3, 1, 4]);
    }

    #[test]
    fn expand_dims_multi_axes_neg() {
        let a = ArrayD::ones(IxDyn(&[3, 4]));
        let result = expand_dims(&a, &[0, -1]);
        assert_eq!(result.shape(), &[1, 3, 4, 1]);
    }

    #[test]
    fn reshape_flatten() {
        let a = ArrayD::ones(IxDyn(&[2, 3]));
        assert!(reshape(&a, &[6]).is_ok())
    }

    #[test]
    fn reshape_unflatten() {
        let a = ArrayD::ones(IxDyn(&[6]));
        assert!(reshape(&a, &[2, -1]).is_ok())
    }

    #[test]
    fn reshape_mismatch() {
        let a = ArrayD::ones(IxDyn(&[6]));
        assert!(reshape(&a, &[2, 4]).is_err())
    }

    // --- conv tests ---

    #[test]
    fn conv_identity_kernel() {
        // 1x1 all-ones kernel is the identity for a single channel
        let input =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), (1..=9).map(|x| x as f64).collect())
                .unwrap();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0]).unwrap();
        let result = conv(&input, &kernel, 1).unwrap();
        assert_eq!(result.shape(), &[1, 1, 3, 3]);
        assert_eq!(result, input);
    }

    #[test]
    fn conv_single_channel_sum_kernel() {
        // 2x2 all-ones kernel produces the sliding 2x2 sum
        let input =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), (1..=9).map(|x| x as f64).collect())
                .unwrap();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0; 4]).unwrap();
        let result = conv(&input, &kernel, 1).unwrap();
        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        assert_eq!(result[[0, 0, 0, 0]], 12.0); // 1+2+4+5
        assert_eq!(result[[0, 0, 0, 1]], 16.0); // 2+3+5+6
        assert_eq!(result[[0, 0, 1, 0]], 24.0); // 4+5+7+8
        assert_eq!(result[[0, 0, 1, 1]], 28.0); // 5+6+8+9
    }

    #[test]
    fn conv_multi_output_channels() {
        // kernel shape [2,1,2,2]: two different output filters on one input channel
        let input =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), (1..=9).map(|x| x as f64).collect())
                .unwrap();
        // output channel 0: all ones (sum); output channel 1: top-left only
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[2, 1, 2, 2]),
            vec![
                1.0, 1.0, 1.0, 1.0, // oc=0
                1.0, 0.0, 0.0, 0.0, // oc=1
            ],
        )
        .unwrap();
        let result = conv(&input, &kernel, 1).unwrap();
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
        let input = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3, 3]), input_data).unwrap();

        // ic=0: pick only top-left; ic=1: all ones (2x2 sum over the all-ones channel)
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 2, 2]),
            vec![
                1.0, 0.0, 0.0, 0.0, // ic=0
                1.0, 1.0, 1.0, 1.0, // ic=1
            ],
        )
        .unwrap();
        let result = conv(&input, &kernel, 1).unwrap();
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
        let input =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 4, 4]), (1..=16).map(|x| x as f64).collect())
                .unwrap();
        // kernel picks only the top-left element of each window
        let kernel =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = conv(&input, &kernel, 2).unwrap();
        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        assert_eq!(result[[0, 0, 0, 0]], 1.0); // input[0,0,0,0]
        assert_eq!(result[[0, 0, 0, 1]], 3.0); // input[0,0,0,2]
        assert_eq!(result[[0, 0, 1, 0]], 9.0); // input[0,0,2,0]
        assert_eq!(result[[0, 0, 1, 1]], 11.0); // input[0,0,2,2]
    }

    #[test]
    fn conv_batch_size_2() {
        // two batch items are processed independently
        let input = ArrayD::from_shape_vec(
            IxDyn(&[2, 1, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![2.0]).unwrap();
        let result = conv(&input, &kernel, 1).unwrap();
        assert_eq!(result.shape(), &[2, 1, 2, 2]);
        assert_eq!(result[[0, 0, 0, 0]], 2.0);
        assert_eq!(result[[0, 0, 1, 1]], 8.0);
        assert_eq!(result[[1, 0, 0, 0]], 10.0);
        assert_eq!(result[[1, 0, 1, 1]], 16.0);
    }

    #[test]
    fn conv_channel_mismatch_error() {
        let input = ArrayD::zeros(IxDyn(&[1, 3, 4, 4])); // 3 input channels
        let kernel = ArrayD::zeros(IxDyn(&[2, 4, 2, 2])); // expects 4 input channels
        assert!(conv(&input, &kernel, 1).is_err());
    }

    #[test]
    fn conv_non_4d_input_error() {
        let input = ArrayD::zeros(IxDyn(&[1, 3, 4]));
        let kernel = ArrayD::zeros(IxDyn(&[2, 3, 2, 2]));
        assert!(conv(&input, &kernel, 1).is_err());
    }

    #[test]
    fn conv_non_4d_kernel_error() {
        let input = ArrayD::zeros(IxDyn(&[1, 3, 4, 4]));
        let kernel = ArrayD::zeros(IxDyn(&[2, 3, 2]));
        assert!(conv(&input, &kernel, 1).is_err());
    }

    // --- pool tests ---

    #[test]
    fn pool_1d_sum_stride1() {
        // [1,2,3,4] with window=2, stride=1 → [3,5,7]
        let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let opt = PoolOptions {
            window_size: vec![2],
            stride: vec![1],
        };
        let result = pool(&a, &opt, false).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result[[0]], 3.0);
        assert_eq!(result[[1]], 5.0);
        assert_eq!(result[[2]], 7.0);
    }

    #[test]
    fn pool_1d_avg_stride1() {
        // same windows, averaged
        let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let opt = PoolOptions {
            window_size: vec![2],
            stride: vec![1],
        };
        let result = pool(&a, &opt, true).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result[[0]], 1.5);
        assert_eq!(result[[1]], 2.5);
        assert_eq!(result[[2]], 3.5);
    }

    #[test]
    fn pool_1d_sum_stride2() {
        // [1,2,3,4] with window=2, stride=2 → [3,7]
        let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let opt = PoolOptions {
            window_size: vec![2],
            stride: vec![2],
        };
        let result = pool(&a, &opt, false).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result[[0]], 3.0); // 1+2
        assert_eq!(result[[1]], 7.0); // 3+4
    }

    #[test]
    fn pool_2d_sum_stride1() {
        // 3x3 input, 2x2 window, stride 1 → 2x2 output; same values as conv sum-kernel test
        let a =
            ArrayD::from_shape_vec(IxDyn(&[3, 3]), (1..=9).map(|x| x as f64).collect()).unwrap();
        let opt = PoolOptions {
            window_size: vec![2, 2],
            stride: vec![1, 1],
        };
        let result = pool(&a, &opt, false).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 12.0); // 1+2+4+5
        assert_eq!(result[[0, 1]], 16.0); // 2+3+5+6
        assert_eq!(result[[1, 0]], 24.0); // 4+5+7+8
        assert_eq!(result[[1, 1]], 28.0); // 5+6+8+9
    }

    #[test]
    fn pool_2d_avg_stride1() {
        let a =
            ArrayD::from_shape_vec(IxDyn(&[3, 3]), (1..=9).map(|x| x as f64).collect()).unwrap();
        let opt = PoolOptions {
            window_size: vec![2, 2],
            stride: vec![1, 1],
        };
        let result = pool(&a, &opt, true).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 3.0); // 12/4
        assert_eq!(result[[0, 1]], 4.0); // 16/4
        assert_eq!(result[[1, 0]], 6.0); // 24/4
        assert_eq!(result[[1, 1]], 7.0); // 28/4
    }

    #[test]
    fn pool_2d_sum_stride2() {
        // 4x4 input, 2x2 window, stride 2 → 2x2 output
        let a =
            ArrayD::from_shape_vec(IxDyn(&[4, 4]), (1..=16).map(|x| x as f64).collect()).unwrap();
        let opt = PoolOptions {
            window_size: vec![2, 2],
            stride: vec![2, 2],
        };
        let result = pool(&a, &opt, false).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 14.0); // 1+2+5+6
        assert_eq!(result[[0, 1]], 22.0); // 3+4+7+8
        assert_eq!(result[[1, 0]], 46.0); // 9+10+13+14
        assert_eq!(result[[1, 1]], 54.0); // 11+12+15+16
    }

    #[test]
    fn pad_single_axis() {
        let a = ArrayD::ones(IxDyn(&[2, 3, 4]));
        let result = pad(
            &a,
            &PaddingOptions {
                config: PaddingOptionConfig {
                    left: 2,
                    right: 2,
                    interior: 1,
                },
                axes: vec![0],
                value: 0.0,
            },
        );

        let mut expected = ArrayD::from_elem(IxDyn(&[7, 3, 4]), 0.0);
        expected.slice_mut(s![2, .., ..]).fill(1.0);
        expected.slice_mut(s![4, .., ..]).fill(1.0);

        assert_eq!(result.shape(), &[7, 3, 4]);
        assert_eq!(result, expected);
    }
}
