# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Random compute graph generation for fuzz testing.

Generates valid .mininn files (zip format) without depending on minijax.
Uses hypothesis.strategies for structured random generation.
"""

import io
import zipfile
from dataclasses import dataclass, field

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

# Strategies for structural decisions
_st_n_inputs = st.integers(min_value=1, max_value=10)
_st_ndim = st.integers(min_value=1, max_value=4)
_st_dim_size = st.integers(min_value=1, max_value=32)
_st_n_ops = st.integers(min_value=5, max_value=20)


# ---------------------------------------------------------------------------
# Graph representation
# ---------------------------------------------------------------------------


@dataclass
class Var:
    name: str
    shape: tuple[int, ...]
    is_const: bool = False


@dataclass
class Equation:
    primitive: str
    inputs: list[Var]
    output: Var
    options: dict = field(default_factory=dict)


@dataclass
class Graph:
    invars: list[Var]
    outvars: list[Var]
    equations: list[Equation]
    constants: dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Primitives and their shape rules
# ---------------------------------------------------------------------------

UNARY_ELEMENTWISE = ["neg", "reciprocal", "relu", "square", "sqrt", "exp", "log"]
UNARY_ACTIVATIONS = ["leaky_relu", "elu", "gelu"]
BINARY_ELEMENTWISE = ["add", "mul"]
LAST_AXES_OPS = ["pad", "conv", "avgpool"]
ALL_PRIMITIVES = [
    *UNARY_ELEMENTWISE,
    *UNARY_ACTIVATIONS,
    *BINARY_ELEMENTWISE,
    "dot",
    "where",
    "expand_dims",
    "moveaxis",
    "reshape",
    "reduce_sum",
    *LAST_AXES_OPS,
]
UNSAFE_PRIMITIVES = {"log", "sqrt", "reciprocal", "exp", "elu", "gelu"}
SAFE_PRIMITIVES = [p for p in ALL_PRIMITIVES if p not in UNSAFE_PRIMITIVES]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def generate_graph(draw, primitives=None, allow_bilinear_dot=True, n_inputs=None) -> Graph:
    """Generate a random valid compute graph.

    Args:
        draw: hypothesis draw function (e.g. _DrawAdapter instance).
        primitives: list of primitive names to use (defaults to ALL_PRIMITIVES).
        allow_bilinear_dot: when False, ``dot`` is only ever applied with a
            constant right operand (a linear application). The bounds fuzzer
            sets this so that ``dot`` is never a bilinear application of two
            bounded operands, which IBP does not handle.
        n_inputs: when given, fix the number of input variables instead of
            drawing it.  The ``affine_bounds`` fuzzer passes ``1`` because CROWN
            relaxes a single-input scalar-output network (see ``scalarize_graph``).

    Returns:
        A Graph with random structure and constant values.
    """
    if primitives is None:
        primitives = ALL_PRIMITIVES

    # Generate input variables
    if n_inputs is None:
        n_inputs = draw(_st_n_inputs)
    invars = []
    for i in range(n_inputs):
        ndim = draw(_st_ndim)
        shape = tuple(draw(_st_dim_size) for _ in range(ndim))
        invars.append(Var(name=_var_name(i), shape=shape))

    var_counter = n_inputs
    const_counter = 0
    available: list[Var] = list(invars)
    equations: list[Equation] = []
    constants: dict[str, np.ndarray] = {}

    n_ops = draw(_st_n_ops)

    for _ in range(n_ops):
        # Pick a random primitive and try to apply it
        order = draw(st.permutations(list(range(len(primitives)))))
        applied = False

        for prim_idx in order:
            prim_name = primitives[prim_idx]
            result = _try_apply(
                prim_name,
                available,
                draw,
                var_counter,
                const_counter,
                constants,
                allow_bilinear_dot=allow_bilinear_dot,
            )
            if result is not None:
                eqn, new_var, new_consts, var_counter, const_counter = result
                equations.append(eqn)
                available.append(new_var)
                constants.update(new_consts)
                applied = True
                break

        if not applied:
            # No primitive could be applied (shouldn't happen with constants)
            continue

    # Output: the last computed variable, or first input if no ops succeeded
    if equations:
        outvars = [equations[-1].output]
    else:
        outvars = [invars[0]]

    return Graph(invars=invars, outvars=outvars, equations=equations, constants=constants)


def scalarize_graph(graph: Graph) -> Graph:
    """Reduce ``graph``'s single output to a scalar (shape ``()``) in place.

    The ``affine_bounds`` CLI relaxes a *scalar*-output network, so the
    affine-bounds fuzzer appends a flatten + full ``reduce_sum`` to collapse the
    last computed variable (of any shape) to ``()``.  Both primitives are linear,
    so they add no relaxation gap of their own.  Returns the mutated graph.
    """
    out = graph.outvars[0]
    if out.shape == ():
        return graph

    total = 1
    for d in out.shape:
        total *= d

    used = (
        {v.name for v in graph.invars}
        | set(graph.constants)
        | {e.output.name for e in graph.equations}
    )

    def fresh(shape):
        i = 0
        while _var_name(i) in used:
            i += 1
        name = _var_name(i)
        used.add(name)
        return Var(name=name, shape=shape)

    cur = out
    if len(out.shape) != 1:
        flat = fresh((total,))
        graph.equations.append(Equation("reshape", [cur], flat, {"new_shape": (total,)}))
        cur = flat
    scalar = fresh(())
    graph.equations.append(Equation("reduce_sum", [cur], scalar, {"axes": (0,)}))
    graph.outvars = [scalar]
    return graph


def _try_apply(
    prim_name, available, draw, var_counter, const_counter, constants, allow_bilinear_dot=True
):
    """Try to apply a primitive. Returns (equation, new_var, new_consts, var_c, const_c) or None."""
    new_consts = {}

    def _pick(lst):
        """Pick a random element from a list using hypothesis."""
        return draw(st.sampled_from(lst))

    if prim_name in UNARY_ELEMENTWISE:
        inp = _pick(available)
        out_shape = inp.shape
        out_var = Var(name=_var_name(var_counter), shape=out_shape)
        eqn = Equation(prim_name, [inp], out_var)
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    elif prim_name == "conv":
        # NCHW × OIHW 2-D convolution (matches jax.lax.conv). Input: (N, C_in, H, W);
        # kernel: (C_out, C_in, kH, kW); output: (N, C_out, H', W').
        candidates = [v for v in available if len(v.shape) == 4]
        if not candidates:
            return None
        inp = _pick(candidates)
        n, c_in, h, w = inp.shape
        if h < 1 or w < 1 or c_in < 1:
            return None
        kh = draw(st.integers(min_value=1, max_value=min(h, 5)))
        kw = draw(st.integers(min_value=1, max_value=min(w, 5)))
        c_out = draw(st.integers(min_value=1, max_value=4))
        max_stride = min(h - kh + 1, w - kw + 1)
        if max_stride < 1:
            return None
        stride = draw(st.integers(min_value=1, max_value=max_stride))
        out_h = (h - kh) // stride + 1
        out_w = (w - kw) // stride + 1
        kernel, const_counter = _make_constant(
            (c_out, c_in, kh, kw), const_counter, draw, new_consts
        )
        out_shape = (n, c_out, out_h, out_w)
        out_var = Var(name=_var_name(var_counter), shape=out_shape)
        eqn = Equation(prim_name, [inp, kernel], out_var, {"stride": stride})
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    elif prim_name == "avgpool":
        # 2-D average pooling on the last two axes with a square window.
        candidates = [
            v for v in available if len(v.shape) >= 2 and v.shape[-1] >= 1 and v.shape[-2] >= 1
        ]
        if not candidates:
            return None
        inp = _pick(candidates)
        h, w = inp.shape[-2], inp.shape[-1]
        window = draw(st.integers(min_value=1, max_value=min(h, w, 5)))
        max_stride = min(h - window + 1, w - window + 1)
        if max_stride < 1:
            return None
        s = draw(st.integers(min_value=1, max_value=max_stride))
        out_h = (h - window) // s + 1
        out_w = (w - window) // s + 1
        out_shape = inp.shape[:-2] + (out_h, out_w)
        out_var = Var(name=_var_name(var_counter), shape=out_shape)
        window_size = (1,) * (len(inp.shape) - 2) + (window, window)
        stride = (1,) * (len(inp.shape) - 2) + (s, s)
        eqn = Equation(
            prim_name, [inp], out_var, {"window_size": window_size, "stride": stride}
        )
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    elif prim_name in UNARY_ACTIVATIONS:
        inp = _pick(available)
        out_var = Var(name=_var_name(var_counter), shape=inp.shape)
        options = {}
        if prim_name == "leaky_relu":
            options["slope"] = draw(
                st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False)
            )
        eqn = Equation(prim_name, [inp], out_var, options)
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    elif prim_name in BINARY_ELEMENTWISE:
        inp = _pick(available)
        # Find a matching var or create a constant
        matches = [v for v in available if v.shape == inp.shape and v is not inp]
        if matches:
            inp2 = _pick(matches)
        else:
            inp2, const_counter = _make_constant(inp.shape, const_counter, draw, new_consts)
        out_shape = inp.shape
        out_var = Var(name=_var_name(var_counter), shape=out_shape)
        eqn = Equation(prim_name, [inp, inp2], out_var)
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    elif prim_name == "dot":
        # Need two 2D inputs where first's last dim == second's first dim
        candidates_left = [v for v in available if len(v.shape) == 2]
        if not candidates_left:
            return None
        left = _pick(candidates_left)
        k = left.shape[-1]
        # Find a matching right operand. A bounded (variable) right operand
        # makes this a bilinear application; only use one when allowed,
        # otherwise the right operand is always a constant matrix.
        candidates_right = [v for v in available if len(v.shape) == 2 and v.shape[0] == k]
        if candidates_right and allow_bilinear_dot:
            right = _pick(candidates_right)
        else:
            n = draw(_st_dim_size)
            right, const_counter = _make_constant((k, n), const_counter, draw, new_consts)
        out_shape = (left.shape[0], right.shape[1])
        out_var = Var(name=_var_name(var_counter), shape=out_shape)
        eqn = Equation(prim_name, [left, right], out_var)
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    elif prim_name == "where":
        inp = _pick(available)
        shape = inp.shape
        # Find or create two more vars of same shape
        matches = [v for v in available if v.shape == shape and v is not inp]
        inputs = [inp]
        for _ in range(2):
            if matches:
                chosen = _pick(matches)
                matches = [v for v in matches if v is not chosen]
                inputs.append(chosen)
            else:
                c, const_counter = _make_constant(shape, const_counter, draw, new_consts)
                inputs.append(c)
        out_var = Var(name=_var_name(var_counter), shape=shape)
        eqn = Equation(prim_name, inputs, out_var)
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    elif prim_name == "expand_dims":
        inp = _pick(available)
        axis = draw(st.integers(min_value=0, max_value=len(inp.shape)))
        out_shape = inp.shape[:axis] + (1,) + inp.shape[axis:]
        out_var = Var(name=_var_name(var_counter), shape=out_shape)
        eqn = Equation(prim_name, [inp], out_var, {"axes": (axis,)})
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    elif prim_name == "moveaxis":
        candidates = [v for v in available if len(v.shape) >= 2]
        if not candidates:
            return None
        inp = _pick(candidates)
        ndim = len(inp.shape)
        source = draw(st.integers(min_value=0, max_value=ndim - 1))
        dest = draw(st.integers(min_value=0, max_value=ndim - 1))
        # Compute output shape
        dims = list(range(ndim))
        dims.pop(source)
        dims.insert(dest, source)
        out_shape = tuple(inp.shape[d] for d in dims)
        out_var = Var(name=_var_name(var_counter), shape=out_shape)
        eqn = Equation(prim_name, [inp], out_var, {"source": source, "destination": dest})
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    elif prim_name == "reshape":
        inp = _pick(available)
        total = 1
        for d in inp.shape:
            total *= d
        if total == 0:
            return None
        new_shape = _random_shape_with_total(total, draw)
        out_var = Var(name=_var_name(var_counter), shape=new_shape)
        eqn = Equation(prim_name, [inp], out_var, {"new_shape": new_shape})
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    elif prim_name == "reduce_sum":
        candidates = [v for v in available if len(v.shape) >= 1]
        if not candidates:
            return None
        inp = _pick(candidates)
        axis = draw(st.integers(min_value=0, max_value=len(inp.shape) - 1))
        out_shape = inp.shape[:axis] + inp.shape[axis + 1 :]
        if not out_shape:
            out_shape = ()
        out_var = Var(name=_var_name(var_counter), shape=out_shape)
        eqn = Equation(prim_name, [inp], out_var, {"axes": (axis,)})
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    elif prim_name == "pad":
        candidates = [v for v in available if len(v.shape) >= 1]
        if not candidates:
            return None
        inp = _pick(candidates)
        ndim = len(inp.shape)
        n_axes = min(ndim, draw(st.integers(min_value=1, max_value=2)))
        axes = tuple(range(-n_axes, 0))
        left = draw(st.integers(min_value=0, max_value=3))
        right = draw(st.integers(min_value=0, max_value=3))
        mid = draw(st.integers(min_value=0, max_value=2))
        value = draw(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        )
        out_shape = list(inp.shape)
        for ax in axes:
            s = out_shape[ax]
            out_shape[ax] = left + s + (s - 1) * mid + right
        out_var = Var(name=_var_name(var_counter), shape=tuple(out_shape))
        options = {"config": (left, right, mid), "axes": axes, "value": value}
        eqn = Equation(prim_name, [inp], out_var, options)
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    elif prim_name == "conv":
        candidates = [v for v in available if len(v.shape) >= 3]
        if not candidates:
            return None
        inp = _pick(candidates)
        n, c, *spatial = inp.shape
        max_k = min(min(spatial), 3)
        if max_k < 1:
            return None
        k_spatial = tuple(
            draw(st.integers(min_value=1, max_value=max_k)) for _ in spatial
        )
        f = draw(st.integers(min_value=1, max_value=4))
        kernel_shape = (f, c) + k_spatial
        kernel, const_counter = _make_constant(
            kernel_shape, const_counter, draw, new_consts
        )
        out_pre = tuple(s - k + 1 for s, k in zip(spatial, k_spatial))
        if any(d < 1 for d in out_pre):
            return None
        stride = draw(st.integers(min_value=1, max_value=min(out_pre)))
        out_spatial = tuple((d - 1) // stride + 1 for d in out_pre)
        out_shape = (n, f) + out_spatial
        out_var = Var(name=_var_name(var_counter), shape=out_shape)
        eqn = Equation(prim_name, [inp, kernel], out_var, {"stride": stride})
        return eqn, out_var, new_consts, var_counter + 1, const_counter

    return None


def _make_constant(shape, const_counter, draw, new_consts):
    """Create a constant variable with random values."""
    name = _const_name(const_counter)
    values = draw(
        np_arrays(
            dtype=np.float64,
            shape=shape,
            elements=st.floats(
                min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    var = Var(name=name, shape=shape, is_const=True)
    new_consts[name] = values
    return var, const_counter + 1


def _random_shape_with_total(total, draw):
    """Generate a random shape whose product equals total."""
    if total == 1:
        return (1,)
    # Factor total into 1-3 dimensions
    ndim = draw(_st_ndim)
    if ndim == 1:
        return (total,)

    # Find factors
    factors = _factorize(total)
    if len(factors) < ndim:
        # Not enough factors, use fewer dims
        ndim = len(factors)
    if ndim == 1:
        return (total,)

    # Distribute prime factors randomly across dimensions
    dims = [1] * ndim
    for f in factors:
        idx = draw(st.integers(min_value=0, max_value=ndim - 1))
        dims[idx] *= f
    return tuple(dims)


def _factorize(n):
    """Return prime factors of n."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def _var_name(index):
    """Lowercase variable name: a, b, ..., z, aa, ab, ..."""
    if index < 26:
        return chr(97 + index)
    return _var_name(index // 26 - 1) + chr(97 + (index % 26))


def _const_name(index):
    """Uppercase constant name: A, B, ..., Z, AA, AB, ..."""
    return _var_name(index).upper()


# ---------------------------------------------------------------------------
# Serialization to .mininn format
# ---------------------------------------------------------------------------


def serialize_graph(graph: Graph) -> bytes:
    """Serialize a Graph to .mininn zip format (as bytes)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        graph_txt = _build_graph_txt(graph)
        with zf.open("graph.txt", "w") as f:
            f.write(graph_txt.encode("utf-8"))

        for name, array in graph.constants.items():
            data = array.astype(np.float64).tobytes("C")
            with zf.open(f"{name}.bin", "w") as f:
                f.write(data)

    return buf.getvalue()


def _build_graph_txt(graph: Graph) -> str:
    lines = []

    # Input line
    input_parts = [_format_var(v) for v in graph.invars]
    lines.append("input: " + "; ".join(input_parts))

    # Equation lines
    for eqn in graph.equations:
        out_str = _format_var(eqn.output)
        opt_parts = [f"{k}: {v!r}" for k, v in eqn.options.items()]
        opts = "{" + "; ".join(opt_parts) + "}"
        in_strs = " ".join(_format_var(v) for v in eqn.inputs)
        lines.append(f"{out_str} = {eqn.primitive}{opts} {in_strs}")

    # Output line
    output_parts = [_format_var(v) for v in graph.outvars]
    lines.append("output: " + "; ".join(output_parts))

    return "\n".join(lines)


def _format_var(var: Var) -> str:
    shape_str = ",".join(str(d) for d in var.shape)
    return f"{var.name}[{shape_str}]"
