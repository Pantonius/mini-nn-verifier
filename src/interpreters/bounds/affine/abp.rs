use ndarray::{ArrayD, Zip};

use crate::{
    interpreters::{
        EvalError, Interpreter,
        bounds::{abp_util::ABPTensor, ibp::IBPInterpreter, ibp_util::IBPTensor},
        concrete::eval_util::{
            Tensor, add, conv, dot, elu, expand_dims, leaky_relu, log, moveaxis, mul, normcdf, pad,
            pool, reduce_sum, relu, reshape,
        },
    },
    mininn::{Atom, ComputeGraph, Env, Primitive},
};

pub struct ABPInterpreter {}

impl ABPInterpreter {
    pub fn process_primitive(
        primitive: &Primitive,
        env: &Env<ABPTensor>,
    ) -> Result<ABPTensor, EvalError> {
        todo!()
    }
}

impl Interpreter<IBPTensor> for ABPInterpreter {
    fn run(graph: &ComputeGraph, inputs: &Vec<IBPTensor>) -> Result<Vec<ABPTensor>, EvalError> {
        // === alpha-CROWN ===
        const ITERS: usize = 10;

        // --- Forward ---
        // - Bound all vars in the network via IBP
        // - Compute initial params (alpha per activation) given IBP bounds

        // --- Backward ---
        // Optimize affine bounds by optimizing over alpha
        //
        // Loss
        // ----
        // 1. compute linear lower bound by backward pass from W_out = ones, bias_out = zeros using
        //    IBP bounds and alpha params from forward pass
        // 2. concretize linear lower bound via "concrete" invar bounds

        // ------------------------------------------------------------
        // 1. init bounds (ibp forward pass)
        // 2. init params (alpha depending on mode of each activation given bounds)
        // 3. optimize alpha (gradient ascent over loss; loss is the concretization of linear lower bound
        //    from backprop starting at W_out = ones, bias_out = zeros, out_var_bounds from IBP
        //    pass; a few iterations of that)

        // 1. Init Bounds (IBP forward pass)
        let mut ibp_bounds = Env::new();
        let mut params: Env<Tensor> = Env::new();

        for (var, tensor) in graph.invars.iter().zip(inputs) {
            ibp_bounds.insert(var.name.clone(), tensor.clone());
        }

        for eqn in &graph.equations {
            let out = IBPInterpreter::process_primitive(&eqn.primitive, &ibp_bounds)?;
            ibp_bounds.insert(eqn.outvar.name.clone(), out);

            // 2. Init Params (Alpha per activation given IBP bounds on respective invar)
            match &eqn.primitive {
                Primitive::Relu(operand)
                | Primitive::LeakyRelu { operand, .. }
                | Primitive::Elu { operand, .. }
                | Primitive::Gelu(operand) => {
                    let bound = ibp_bounds.resolve(operand)?;
                    let alpha = Zip::from(&bound.lb)
                        .and(&bound.ub)
                        .map_collect(|&l, &u| if -l >= u { 0.0 } else { 1.0 });

                    params.insert(eqn.outvar.name.clone(), alpha);
                }
                _ => continue,
            }
        }

        // 3. Optimize Alpha (Gradient Ascent over alpha)
        if params.len() > 0 {
            for _ in 0..ITERS {}
        }

        todo!()
    }
}
