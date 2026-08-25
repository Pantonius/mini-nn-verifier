use ndarray::{ArrayD, Zip, arr0};

use crate::{
    interpreters::{
        EvalError, Interpreter,
        bounds::{abp_util::ABPTensor, ibp::IBPInterpreter, ibp_util::IBPTensor},
        concrete::eval_util::Tensor,
    },
    mininn::{Atom, ComputeGraph, Env, Primitive, Value},
};

pub struct ABPInterpreter {}

impl ABPInterpreter {
    pub fn process_primitive(
        primitive: &Primitive,
        env: &Env<ABPTensor>,
    ) -> Result<ABPTensor, EvalError> {
        todo!()
    }

    fn crown_relu(alpha: &Tensor, out_w: &Tensor, x: &IBPTensor) -> ABPTensor {
        let mut upper_slope = x.ub.clone() / (x.ub.clone() - x.lb.clone());
        let mut lower_slope = alpha.clone();

        let mut upper_offset = -x.ub.clone() * x.lb.clone() / (x.ub.clone() - x.lb.clone());

        upper_slope = Zip::from(&x.lb)
            .and(&x.ub)
            .and(&upper_slope)
            .map_collect(|&l, &u, &d| {
                if l >= 0.0 {
                    1.0
                } else if u <= 0.0 {
                    0.0
                } else {
                    d
                }
            })
            .into();
        lower_slope = Zip::from(&x.lb)
            .and(&x.ub)
            .and(&lower_slope)
            .map_collect(|&l, &u, &d| {
                if l >= 0.0 {
                    1.0
                } else if u <= 0.0 {
                    0.0
                } else {
                    d
                }
            })
            .into();
        upper_offset = Zip::from(&x.lb)
            .and(&x.ub)
            .and(&upper_offset)
            .map_collect(|&l, &u, &d| if (l >= 0.0) | (u <= 0.0) { 0.0 } else { d })
            .into();

        let pos_w = out_w.mapv(|x| if x >= 0.0 { x } else { 0.0 });
        let neg_w = out_w.mapv(|x| if x < 0.0 { x } else { 0.0 });

        let un = upper_offset * neg_w.clone();

        let axes: Vec<isize> = (0..un.ndim() as isize).collect();
        ABPTensor {
            weights: lower_slope * pos_w + upper_slope * neg_w,
            biases: un.reduce_sum(&axes),
        }
    }

    pub fn linear_lower_bound(
        graph: &ComputeGraph,
        var_bounds: Env<IBPTensor>,
        params: Env<Tensor>,
    ) -> Result<ABPTensor, EvalError> {
        // checks
        if graph.outvars.len() != 1 {
            return Err(EvalError::Eval(
                "Affine Bound Propogation only supports nets with a single outvar.".to_string(),
            ));
        }

        let outvar = &graph.outvars[0];

        if outvar.shape.is_empty() {
            return Err(EvalError::Eval(
                "Affine Bound Propogation only supports nets with a scalar output.".to_string(),
            ));
        }

        // actual start
        let mut weights = Env::new();
        weights.insert(
            outvar.name.clone(),
            Tensor::from(ArrayD::from_elem(outvar.shape.clone(), 1.0)),
        );
        let b = |var: &Atom| var_bounds.resolve(var);
        let w = |var: &Atom| weights.resolve(var);
        let p = |var: &Atom| params.resolve(var);

        let bias = Tensor::from(arr0(0.0).into_dyn());

        for eqn in graph.equations.iter().rev() {
            let aff = match eqn.primitive {
                Primitive::Neg(atom) => todo!(),
                Primitive::Reciprocal(atom) => todo!(),
                Primitive::Square(atom) => todo!(),
                Primitive::Sqrt(atom) => todo!(),
                Primitive::Exp(atom) => todo!(),
                Primitive::Log(atom) => todo!(),
                Primitive::Add(atom, atom1) => todo!(),
                Primitive::Mul(atom, atom1) => todo!(),
                Primitive::Where(atom, atom1, atom2) => todo!(),
                Primitive::Relu(atom) => {
                    Self::crown_relu(&p(&eqn.outvar)?, &w(&eqn.outvar)?, &b(&atom)?)
                }
                Primitive::LeakyRelu { operand, slope } => todo!(),
                Primitive::Elu { operand, slope } => todo!(),
                Primitive::Gelu(atom) => todo!(),
                Primitive::NormalCdf(atom) => todo!(),
                Primitive::Dot(atom, atom1) => todo!(),
                Primitive::ReduceSum { operand, axes } => todo!(),
                Primitive::ExpandDims { operand, axes } => todo!(),
                Primitive::MoveAxis {
                    operand,
                    source,
                    destination,
                } => todo!(),
                Primitive::Reshape { operand, new_shape } => todo!(),
                Primitive::Pad { operand, options } => todo!(),
                Primitive::Conv {
                    input,
                    kernel,
                    options,
                } => todo!(),
                Primitive::AvgPool { operand, options } => todo!(),
                Primitive::SumPool { operand, options } => todo!(),
            };
        }

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

                    params.insert(eqn.outvar.name.clone(), Tensor::from(alpha));
                }
                _ => continue,
            }
        }

        // 3. Optimize Alpha (Gradient Ascent over alpha)
        if params.len() > 0 {
            for _ in 0..ITERS {
                // grad(loss(cg, ibp_bounds, params))
            }
        }

        todo!()
    }
}
