use ndarray::ArrayD;

use crate::core::{Primitive, Value};
use crate::interpreters::Interpreter;

struct ComputeGraph {
    equations: Vec<Equation>,
    in_vars: Vec<Var>,  // input nodes
    out_vars: Vec<Var>, // output nodes
}

struct AbstractValue {
    var: Var,
}
impl Value for AbstractValue {}

struct Equation {
    // out_var = add in_var1 in_var2
    primitive: Primitive,
    in_vars: Vec<Var>,
    out_var: Var,
}

struct CGArray {
    value: ArrayD<f64>
}
impl Value for CGArray {}

#[derive(Debug, Clone)]
struct Var {}

struct ComputeGraphInterpreter {
    equations: Vec<Equation>,
}
impl Interpreter for ComputeGraphInterpreter {
    fn process_primitive(&mut self, primitive: Primitive) -> Box<dyn Value> {
        let out_var = Var {};
        let eqn = Equation {
            primitive,
            in_vars: Vec::new(),
            out_var,
        };

        self.equations.push(eqn);

        return Box::new(AbstractValue { var: out_var });
    }
}
impl ComputeGraphInterpreter {
    fn init() -> Self {
        Self {
            equations: Vec::new(),
        }
    }

    fn make_compute_graph(fn: ??, args: Vec<>) -> ComputeGraph {
        let in_vars = Vec::new();
        for _ in args {
            in_vars.push(Var {});
        }

        let in_vals = Vec::new();
        for var in in_vars {
            in_vals.push(AbstractValue { var });
        }

        let cg_interpreter = ComputeGraphInterpreter::init();

        return ComputeGraph {
            equations: cg_interpreter.equations,
            in_vars,
            out_vars,
        };
    }
}
