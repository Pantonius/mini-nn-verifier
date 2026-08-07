# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.fuzz.graph_builder."""

import io
import zipfile

import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from testrunner.fuzz.graph_builder import (
    ALL_PRIMITIVES,
    SAFE_PRIMITIVES,
    UNARY_ELEMENTWISE,
    UNARY_ACTIVATIONS,
    BINARY_ELEMENTWISE,
    LAST_AXES_OPS,
    Var,
    Equation,
    Graph,
    _const_name,
    _factorize,
    _format_var,
    _make_constant,
    _random_shape_with_total,
    _try_apply,
    _var_name,
    generate_graph,
    serialize_graph,
)


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def test_var_name_single_letters():
    assert _var_name(0) == "a"
    assert _var_name(25) == "z"


def test_var_name_double_letters():
    assert _var_name(26) == "aa"
    assert _var_name(27) == "ab"
    assert _var_name(51) == "az"
    assert _var_name(52) == "ba"


def test_const_name_uppercase():
    assert _const_name(0) == "A"
    assert _const_name(25) == "Z"
    assert _const_name(26) == "AA"


# ---------------------------------------------------------------------------
# _factorize
# ---------------------------------------------------------------------------


def test_factorize_prime():
    assert _factorize(7) == [7]


def test_factorize_composite():
    factors = _factorize(12)
    product = 1
    for f in factors:
        product *= f
    assert product == 12


def test_factorize_1():
    assert _factorize(1) == []


def test_factorize_power_of_2():
    factors = _factorize(8)
    assert factors == [2, 2, 2]


def test_factorize_large():
    n = 2 * 3 * 5 * 7
    factors = _factorize(n)
    product = 1
    for f in factors:
        product *= f
    assert product == n


# ---------------------------------------------------------------------------
# _format_var
# ---------------------------------------------------------------------------


def test_format_var_1d():
    v = Var("a", (5,))
    assert _format_var(v) == "a[5]"


def test_format_var_2d():
    v = Var("b", (3, 4))
    assert _format_var(v) == "b[3,4]"


def test_format_var_scalar():
    v = Var("c", ())
    assert _format_var(v) == "c[]"


# ---------------------------------------------------------------------------
# serialize_graph
# ---------------------------------------------------------------------------


def _make_simple_graph():
    """Construct a minimal valid Graph by hand."""
    x = Var("a", (3,))
    y = Var("b", (3,))
    eqn = Equation("neg", [x], y)
    return Graph(invars=[x], outvars=[y], equations=[eqn])


def test_serialize_graph_produces_zip():
    g = _make_simple_graph()
    data = serialize_graph(g)
    assert isinstance(data, bytes)
    # Must be a valid zip
    buf = io.BytesIO(data)
    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
    assert "graph.txt" in names


def test_serialize_graph_contains_graph_txt():
    g = _make_simple_graph()
    data = serialize_graph(g)
    buf = io.BytesIO(data)
    with zipfile.ZipFile(buf) as zf:
        txt = zf.read("graph.txt").decode()
    assert "input:" in txt
    assert "output:" in txt


def test_serialize_graph_with_constants():
    x = Var("a", (2,))
    c = Var("A", (2,), is_const=True)
    y = Var("b", (2,))
    eqn = Equation("add", [x, c], y)
    g = Graph(invars=[x], outvars=[y], equations=[eqn], constants={"A": np.array([1.0, 2.0])})
    data = serialize_graph(g)
    buf = io.BytesIO(data)
    with zipfile.ZipFile(buf) as zf:
        assert "A.bin" in zf.namelist()
        arr = np.frombuffer(zf.read("A.bin"), dtype=np.float64)
    np.testing.assert_array_equal(arr, [1.0, 2.0])


def test_serialize_graph_no_equations():
    x = Var("a", (4,))
    g = Graph(invars=[x], outvars=[x], equations=[])
    data = serialize_graph(g)
    buf = io.BytesIO(data)
    with zipfile.ZipFile(buf) as zf:
        txt = zf.read("graph.txt").decode()
    assert "input:" in txt
    assert "output:" in txt


# ---------------------------------------------------------------------------
# _random_shape_with_total
# ---------------------------------------------------------------------------


def test_random_shape_with_total_1():
    """Total=1 always returns (1,)."""

    def draw(strategy):
        # For total=1, ndim draw is irrelevant
        return 1

    shape = _random_shape_with_total(1, draw)
    assert shape == (1,)


def test_random_shape_with_total_prime():
    """Prime total with ndim>1 falls back to (total,)."""
    calls = []

    def draw(strategy):
        calls.append(strategy)
        return 2  # Request ndim=2

    shape = _random_shape_with_total(7, draw)  # 7 is prime, only 1 factor
    product = 1
    for d in shape:
        product *= d
    assert product == 7


def test_random_shape_with_total_composite():
    """Composite total with 1D request returns (total,)."""

    def draw(strategy):
        return 1  # Always pick ndim=1

    shape = _random_shape_with_total(12, draw)
    assert shape == (12,)


# ---------------------------------------------------------------------------
# _try_apply — each primitive branch
# ---------------------------------------------------------------------------


def _fixed_draw(*return_values):
    """Create a draw() that returns successive values from return_values."""
    it = iter(return_values)

    def draw(strategy):
        try:
            return next(it)
        except StopIteration:
            # Fallback: return something valid for the strategy
            if hasattr(strategy, "_draw_from_repr"):
                pass
            return 0

    return draw


def test_try_apply_unary_elementwise():
    x = Var("a", (3, 4))
    available = [x]
    draw = _fixed_draw(x)  # pick x
    result = _try_apply("neg", available, draw, 1, 0, {})
    assert result is not None
    eqn, out_var, new_consts, var_c, const_c = result
    assert eqn.primitive == "neg"
    assert out_var.shape == (3, 4)
    assert var_c == 2
    assert const_c == 0


def test_try_apply_binary_elementwise_matching_var():
    x = Var("a", (3,))
    y = Var("b", (3,))
    available = [x, y]
    # draw: pick x (inp), then pick y (matching)
    draw = _fixed_draw(x, y)
    result = _try_apply("add", available, draw, 2, 0, {})
    assert result is not None
    eqn, out_var, new_consts, var_c, const_c = result
    assert eqn.primitive == "add"
    assert len(new_consts) == 0  # no constant needed


def test_try_apply_binary_elementwise_creates_constant():
    x = Var("a", (3,))
    available = [x]
    # No matching var — needs a constant
    # draw: pick x (inp), then for constant data return zeros array
    import numpy as np
    from hypothesis.extra.numpy import arrays as np_arrays

    calls = []

    def draw(strategy):
        calls.append(strategy)
        # Return x for sampled_from, then a numpy array of zeros for the constant
        if len(calls) == 1:
            return x  # inp pick
        # Return a valid float64 array of shape (3,)
        return np.zeros(3, dtype=np.float64)

    result = _try_apply("add", available, draw, 1, 0, {})
    assert result is not None
    eqn, out_var, new_consts, var_c, const_c = result
    assert len(new_consts) == 1  # a constant was created
    assert const_c == 1


def test_try_apply_dot_creates_constant():
    left = Var("a", (2, 3))
    available = [left]
    import numpy as np

    calls = []

    def draw(strategy):
        calls.append(strategy)
        n = len(calls)
        if n == 1:
            return left  # pick candidates_left
        elif n == 2:
            return 4  # dim size for right
        else:
            return np.zeros((3, 4), dtype=np.float64)  # constant data

    result = _try_apply("dot", available, draw, 1, 0, {})
    assert result is not None
    eqn, out_var, new_consts, var_c, const_c = result
    assert eqn.primitive == "dot"
    assert out_var.shape == (2, 4)


def test_try_apply_dot_no_2d_candidates():
    x = Var("a", (3,))  # 1D — not a valid left operand
    result = _try_apply("dot", [x], _fixed_draw(), 0, 0, {})
    assert result is None


def test_try_apply_where():
    x = Var("a", (4,))
    y = Var("b", (4,))
    z = Var("c", (4,))
    available = [x, y, z]
    import numpy as np

    calls = []

    def draw(strategy):
        calls.append(strategy)
        n = len(calls)
        if n == 1:
            return x
        if n == 2:
            return y
        if n == 3:
            return z
        return np.zeros(4, dtype=np.float64)

    result = _try_apply("where", available, draw, 3, 0, {})
    assert result is not None
    eqn, out_var, new_consts, _, _ = result
    assert eqn.primitive == "where"
    assert out_var.shape == (4,)


def test_try_apply_expand_dims():
    x = Var("a", (3, 4))
    available = [x]

    def draw(strategy):
        # First call picks x, second call picks axis=1
        if not hasattr(draw, "_n"):
            draw._n = 0
        draw._n += 1
        if draw._n == 1:
            return x
        return 1

    result = _try_apply("expand_dims", available, draw, 1, 0, {})
    assert result is not None
    eqn, out_var, _, _, _ = result
    assert eqn.primitive == "expand_dims"
    assert out_var.shape == (3, 1, 4)


def test_try_apply_moveaxis_no_candidates():
    x = Var("a", (5,))  # 1D — not eligible
    result = _try_apply("moveaxis", [x], _fixed_draw(), 0, 0, {})
    assert result is None


def test_try_apply_moveaxis():
    x = Var("a", (3, 4))
    available = [x]

    calls = []

    def draw(strategy):
        calls.append(strategy)
        n = len(calls)
        if n == 1:
            return x
        if n == 2:
            return 0  # source
        if n == 3:
            return 1  # dest
        return 0

    result = _try_apply("moveaxis", available, draw, 1, 0, {})
    assert result is not None
    eqn, out_var, _, _, _ = result
    assert eqn.primitive == "moveaxis"
    assert out_var.shape == (4, 3)


def test_try_apply_reshape():
    x = Var("a", (2, 6))  # total = 12
    available = [x]

    calls = []

    def draw(strategy):
        calls.append(strategy)
        n = len(calls)
        if n == 1:
            return x
        if n == 2:
            return 1  # ndim=1
        return 0

    result = _try_apply("reshape", available, draw, 1, 0, {})
    assert result is not None
    eqn, out_var, _, _, _ = result
    assert eqn.primitive == "reshape"
    assert eqn.options["new_shape"] == (12,)


def test_try_apply_reduce_sum():
    x = Var("a", (3, 4))
    available = [x]

    calls = []

    def draw(strategy):
        calls.append(strategy)
        n = len(calls)
        if n == 1:
            return x
        return 0  # axis=0

    result = _try_apply("reduce_sum", available, draw, 1, 0, {})
    assert result is not None
    eqn, out_var, _, _, _ = result
    assert eqn.primitive == "reduce_sum"
    assert out_var.shape == (4,)


def test_try_apply_reduce_sum_to_scalar():
    x = Var("a", (5,))
    available = [x]

    calls = []

    def draw(strategy):
        calls.append(strategy)
        n = len(calls)
        if n == 1:
            return x
        return 0  # axis=0

    result = _try_apply("reduce_sum", available, draw, 1, 0, {})
    assert result is not None
    eqn, out_var, _, _, _ = result
    assert out_var.shape == ()


def test_try_apply_unknown_returns_none():
    result = _try_apply("nonexistent_op", [], _fixed_draw(), 0, 0, {})
    assert result is None


def test_try_apply_reshape_zero_total():
    """Var with shape containing 0 should return None."""
    x = Var("a", (0, 3))  # total = 0
    result = _try_apply("reshape", [x], _fixed_draw(x), 0, 0, {})
    assert result is None


def test_try_apply_reduce_sum_no_candidates():
    """reduce_sum with only scalar (0-dim) vars should return None."""
    x = Var("a", ())  # 0-dimensional — not a valid candidate
    result = _try_apply("reduce_sum", [x], _fixed_draw(x), 0, 0, {})
    assert result is None


def test_generate_graph_no_equations_fallback():
    """Empty primitives list produces a graph with first input as output."""
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st

    called = []

    @given(st.data())
    @settings(max_examples=1, suppress_health_check=list(HealthCheck))
    def inner(data):
        draw = data.draw
        graph = generate_graph(draw, primitives=[])
        called.append(graph)
        # With no primitives, equations should be empty and output = first input
        assert graph.outvars[0] is graph.invars[0]

    inner()


# ---------------------------------------------------------------------------
# generate_graph — property-based
# ---------------------------------------------------------------------------


@given(st.data())
@settings(max_examples=20, suppress_health_check=list(HealthCheck))
def test_generate_graph_valid(data):
    draw = data.draw
    graph = generate_graph(draw)
    assert len(graph.invars) >= 1
    assert len(graph.outvars) >= 1
    # Output var must come from equations or be an input
    output_names = {v.name for v in graph.outvars}
    all_names = {v.name for v in graph.invars} | {eqn.output.name for eqn in graph.equations}
    for name in output_names:
        assert name in all_names


@given(st.data())
@settings(max_examples=10, suppress_health_check=list(HealthCheck))
def test_generate_graph_with_safe_primitives(data):
    draw = data.draw
    graph = generate_graph(draw, primitives=SAFE_PRIMITIVES)
    for eqn in graph.equations:
        assert eqn.primitive in SAFE_PRIMITIVES


@given(st.data())
@settings(max_examples=10, suppress_health_check=list(HealthCheck))
def test_generate_graph_single_primitive(data):
    """Graph restricted to a single primitive still produces valid output."""
    draw = data.draw
    graph = generate_graph(draw, primitives=["neg"])
    assert graph.outvars


# ---------------------------------------------------------------------------
# Primitive sets
# ---------------------------------------------------------------------------


def test_all_primitives_contains_expected():
    for p in UNARY_ELEMENTWISE + BINARY_ELEMENTWISE + UNARY_ACTIVATIONS + LAST_AXES_OPS:
        assert p in ALL_PRIMITIVES
    for p in ["dot", "where", "expand_dims", "moveaxis", "reshape", "reduce_sum"]:
        assert p in ALL_PRIMITIVES


def test_safe_primitives_excludes_unsafe():
    unsafe = {"log", "sqrt", "reciprocal", "exp"}
    for p in SAFE_PRIMITIVES:
        assert p not in unsafe


def test_safe_primitives_includes_new_primitives():
    # leaky_relu and the last-axes ops are numerically safe.
    for p in ["leaky_relu"] + LAST_AXES_OPS:
        assert p in SAFE_PRIMITIVES
    # elu and gelu are exp-based, so they are classified unsafe.
    for p in ["elu", "gelu"]:
        assert p not in SAFE_PRIMITIVES


# ---------------------------------------------------------------------------
# New primitives: activations + last-axis ops
# ---------------------------------------------------------------------------


def test_try_apply_leaky_relu_sets_slope():
    x = Var("a", (3, 4))
    draw = _fixed_draw(x, 0.1)  # pick x, pick slope
    result = _try_apply("leaky_relu", [x], draw, 1, 0, {})
    assert result is not None
    eqn, out_var, _, _, _ = result
    assert out_var.shape == (3, 4)
    assert "slope" in eqn.options
    assert eqn.options["slope"] in [0.01, 0.05, 0.1, 0.2, 0.3]


def test_try_apply_elu_no_options():
    x = Var("a", (3,))
    result = _try_apply("elu", [x], _fixed_draw(x), 1, 0, {})
    assert result is not None
    eqn, out_var, _, _, _ = result
    assert eqn.options == {}
    assert out_var.shape == (3,)


def test_try_apply_gelu_no_options():
    x = Var("a", (2, 5))
    result = _try_apply("gelu", [x], _fixed_draw(x), 1, 0, {})
    assert result is not None
    eqn, out_var, _, _, _ = result
    assert eqn.options == {}
    assert out_var.shape == (2, 5)


def test_try_apply_pad_output_shape():
    x = Var("a", (4, 10))  # n = 10
    # draw order: pick x, n_axes=1, left=2, right=1, mid=2 (last axis only):
    #   2 + 10 + (10 - 1) * 2 + 1 = 31, leading dim preserved.
    draw = _fixed_draw(x, 1, 2, 1, 2)
    result = _try_apply("pad", [x], draw, 1, 0, {})
    assert result is not None
    eqn, out_var, _, _, _ = result
    assert out_var.shape == (4, 31)
    assert eqn.options["config"] == (2, 1, 2)
    assert eqn.options["axes"] == (-1,)
    assert eqn.options["value"] == 0.0


def test_try_apply_pad_no_1d_candidates():
    x = Var("a", ())  # 0-d
    result = _try_apply("pad", [x], _fixed_draw(x), 1, 0, {})
    assert result is None


def test_try_apply_conv_creates_kernel_constant():
    # NCHW input: (N=2, C_in=3, H=6, W=8)
    x = Var("a", (2, 3, 6, 8))
    # draw order: pick x, kh=3, kw=3, c_out=4, stride=2, then kernel data
    def draw(strategy):
        if not hasattr(draw, "n"):
            draw.n = 0
        draw.n += 1
        if draw.n == 1:
            return x
        elif draw.n == 2:
            return 3  # kernel height
        elif draw.n == 3:
            return 3  # kernel width
        elif draw.n == 4:
            return 4  # c_out
        elif draw.n == 5:
            return 2  # stride
        else:
            return np.zeros((4, 3, 3, 3), dtype=np.float64)

    result = _try_apply("conv", [x], draw, 1, 0, {})
    assert result is not None
    eqn, out_var, new_consts, _, const_c = result
    # out_h = (6 - 3) // 2 + 1 = 2; out_w = (8 - 3) // 2 + 1 = 3; c_out = 4
    assert out_var.shape == (2, 4, 2, 3)
    assert eqn.options == {"stride": 2}
    assert len(new_consts) == 1
    # Kernel must be OIHW (4-D)
    kernel_shape = next(iter(new_consts.values())).shape
    assert kernel_shape == (4, 3, 3, 3)
    assert const_c == 1


def test_try_apply_conv_no_1d_candidates():
    """conv requires 4-D NCHW input; anything else is rejected."""
    x1 = Var("a", ())
    x2 = Var("b", (8,))
    x3 = Var("c", (2, 8))
    x4 = Var("d", (2, 4, 8))
    result = _try_apply("conv", [x1, x2, x3, x4], _fixed_draw(x4), 1, 0, {})
    assert result is None


def test_try_apply_avgpool_output_shape():
    # NCHW input to exercise the typical case; avgpool operates on last two axes.
    x = Var("a", (2, 1, 10, 10))
    # draw order: pick x, window=4, stride=3
    draw = _fixed_draw(x, 4, 3)
    result = _try_apply("avgpool", [x], draw, 1, 0, {})
    assert result is not None
    eqn, out_var, _, _, _ = result
    # out_h = out_w = (10 - 4) // 3 + 1 = 3
    assert out_var.shape == (2, 1, 3, 3)
    assert eqn.options == {"window_size": (1, 1, 4, 4), "stride": (1, 1, 3, 3)}


def test_try_apply_avgpool_no_1d_candidates():
    """avgpool requires ndim >= 2 (at least 2 spatial axes); 1-D inputs are rejected."""
    x1 = Var("a", ())
    x2 = Var("b", (10,))
    result = _try_apply("avgpool", [x1, x2], _fixed_draw(x2), 1, 0, {})
    assert result is None


@given(st.data())
@settings(max_examples=20, suppress_health_check=list(HealthCheck))
def test_generate_graph_new_activations_only(data):
    draw = data.draw
    graph = generate_graph(draw, primitives=UNARY_ACTIVATIONS)
    for eqn in graph.equations:
        assert eqn.primitive in UNARY_ACTIVATIONS
        if eqn.primitive == "leaky_relu":
            assert "slope" in eqn.options


@given(st.data())
@settings(max_examples=20, suppress_health_check=list(HealthCheck))
def test_generate_graph_last_axis_ops_only(data):
    draw = data.draw
    graph = generate_graph(draw, primitives=LAST_AXES_OPS)
    for eqn in graph.equations:
        assert eqn.primitive in LAST_AXES_OPS
        if eqn.primitive == "pad":
            # pad grows up to the last two axes; leading dims are preserved.
            inp_shape, out_shape = eqn.inputs[0].shape, eqn.output.shape
            assert len(out_shape) == len(inp_shape)
            assert out_shape[:-2] == inp_shape[:-2]
            assert all(o >= i for o, i in zip(out_shape, inp_shape))
        elif eqn.primitive == "conv":
            # (N, C_in, *spatial) conv with an (C_out, C_in, *spatial) kernel,
            # >= 3-D input (>= 1 spatial axis); output shares N with the input.
            inp_shape = eqn.inputs[0].shape
            ker_shape = eqn.inputs[1].shape
            assert len(inp_shape) >= 3
            assert len(ker_shape) == len(inp_shape)
            assert ker_shape[1] == inp_shape[1]            # C_in matches
            assert eqn.output.shape[0] == inp_shape[0]     # N preserved
            assert eqn.output.shape[1] == ker_shape[0]     # C_out
        else:
            # avgpool operates on last two spatial axes, preserves leading dims
            assert len(eqn.inputs[0].shape) >= 2
            assert eqn.output.shape[:-2] == eqn.inputs[0].shape[:-2]


@given(st.data())
@settings(max_examples=10, suppress_health_check=list(HealthCheck))
def test_generate_graph_serializes_with_new_primitives(data):
    """Graphs built from the full primitive set serialize to a valid .mininn."""
    draw = data.draw
    graph = generate_graph(draw, primitives=ALL_PRIMITIVES)
    blob = serialize_graph(graph)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert "graph.txt" in zf.namelist()
        text = zf.read("graph.txt").decode()
        assert "input:" in text and "output:" in text
