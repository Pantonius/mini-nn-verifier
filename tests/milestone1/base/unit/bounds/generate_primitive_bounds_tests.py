#!/usr/bin/env python3
# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Generate ``bounds`` unit tests covering the IBP primitive rules.

For each primitive supported by ``mininnverifier.ibp.ibp`` we build a tiny
``.mininn`` network that exercises a single equation, then emit the
expected lower/upper bound files computed via a direct numpy ground
truth (independent of the implementation under test).

Run from the repository root:
    python tests/milestone1/base/unit/bounds/generate_primitive_bounds_tests.py
"""

import json
from pathlib import Path

import numpy as np

from minijax import core
from minijax.compute_graph import make_graph
from minijax.eval import Array
from minijax.serialize import dump


SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_TOLERANCE = 1e-4


# ----------------------------------------------------------------------
# Ground-truth IBP rules in pure numpy.
# Each spec describes:
#   - fn:           the minijax graph builder (lambda *vars: ...)
#   - inputs:       list of either ("box", lb_np, ub_np) or ("point", val_np)
#   - bound:        callable taking the lb/ub np arrays per input
#                   (point inputs get a single value) and returning
#                   a list of (out_lb_np, out_ub_np).
# ----------------------------------------------------------------------


def ibp_mono_non_decreasing(np_fn):
    """IBP for primitives that are non-decreasing in every argument."""

    def rule(*box_or_point):
        lbs = [b[0] for b in box_or_point]
        ubs = [b[1] for b in box_or_point]
        return [(np_fn(*lbs), np_fn(*ubs))]

    return rule


def ibp_mono_non_increasing(np_fn):
    """IBP for primitives that are non-increasing in every argument."""

    def rule(*box_or_point):
        lbs = [b[0] for b in box_or_point]
        ubs = [b[1] for b in box_or_point]
        return [(np_fn(*ubs), np_fn(*lbs))]

    return rule


def ibp_linear_with_one_point(np_fn):
    """IBP for a bilinear primitive where exactly one operand is a point.

    Output midpoint is ``np_fn(x_mid, y_mid)`` and the half-range is
    ``np_fn(x_ran, |y_const|)`` (or ``np_fn(|x_const|, y_ran)``).
    The current ``ibp.ibp_linear`` implementation uses the *non-abs*
    version of the constant operand; tests using non-negative constants
    agree with both formulas.
    """

    def rule(x_lu, y_lu):
        x_lb, x_ub = x_lu
        y_lb, y_ub = y_lu
        x_is_point = np.array_equal(x_lb, x_ub)
        y_is_point = np.array_equal(y_lb, y_ub)
        assert x_is_point ^ y_is_point, "exactly one operand must be a point"
        if y_is_point:
            x_mid = 0.5 * (x_lb + x_ub)
            x_ran = 0.5 * (x_ub - x_lb)
            mid = np_fn(x_mid, y_lb)
            ran = np_fn(x_ran, np.abs(y_lb))
        else:
            y_mid = 0.5 * (y_lb + y_ub)
            y_ran = 0.5 * (y_ub - y_lb)
            mid = np_fn(x_lb, y_mid)
            ran = np_fn(np.abs(x_lb), y_ran)
        return [(mid - ran, mid + ran)]

    return rule


def _np_dot(x, y):
    if y.ndim <= 1:
        return np.dot(x, y)
    return np.einsum("...j,...jk", x, y)


# ----------------------------------------------------------------------
# Fixture builder
# ----------------------------------------------------------------------


def _write_test(name, fn, inputs, expected):
    test_dir = SCRIPT_DIR / name
    resources_dir = test_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)

    # Trace the graph with arrays drawn from each input's lower bound.
    trace_arrays = [Array(lb) for kind, lb, *_ in inputs]
    graph = make_graph(fn)(*trace_arrays)
    network_file = f"{name}_network.mininn"
    dump(graph, resources_dir / network_file)

    # Lay out the test.json input stream: marker followed by file path(s).
    input_tokens = []
    for i, (kind, lb, ub) in enumerate(inputs):
        if kind == "box":
            lb_path = f"resources/input_{i}_lb.bin"
            ub_path = f"resources/input_{i}_ub.bin"
            np.asarray(lb, dtype=np.float64).tofile(test_dir / lb_path)
            np.asarray(ub, dtype=np.float64).tofile(test_dir / ub_path)
            input_tokens += ["box", lb_path, ub_path]
        elif kind == "point":
            val_path = f"resources/input_{i}.bin"
            np.asarray(lb, dtype=np.float64).tofile(test_dir / val_path)
            input_tokens += ["point", val_path]
        else:
            raise ValueError(f"unknown input kind: {kind}")

    expected_files = []
    for i, (out_lb, out_ub) in enumerate(expected):
        lb_name = f"expected_output_{i}_lb.bin"
        ub_name = f"expected_output_{i}_ub.bin"
        np.asarray(out_lb, dtype=np.float64).tofile(test_dir / lb_name)
        np.asarray(out_ub, dtype=np.float64).tofile(test_dir / ub_name)
        expected_files += [lb_name, ub_name]

    config = {
        "command": "bounds",
        "network": f"resources/{network_file}",
        "inputs": input_tokens,
        "expected_outputs": expected_files,
    }
    (test_dir / "test.json").write_text(json.dumps(config, indent=4) + "\n")
    print(f"  created {test_dir.relative_to(REPO_ROOT)}")


def _box(lb, ub):
    return ("box", np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64))


def _point(val):
    v = np.asarray(val, dtype=np.float64)
    return ("point", v, v)  # store as (kind, lb=ub) for uniform handling


def _bounds_for(rule, inputs):
    """Apply a numpy ground-truth IBP rule to the input bound pairs."""
    return rule(*[(lb, ub) for _, lb, ub in inputs])


# ----------------------------------------------------------------------
# Test specifications
# ----------------------------------------------------------------------


def main():
    # ------- monotonic non-decreasing primitives -------
    # add: box + box
    inputs = [
        _box([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        _box([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
             [[11.0, 21.0, 31.0], [41.0, 51.0, 61.0]]),
    ]
    _write_test(
        "add_box_box",
        lambda x, y: core.add(x, y),
        inputs,
        _bounds_for(ibp_mono_non_decreasing(np.add), inputs),
    )

    # add: point + point — exercises the all-points fast path
    inputs = [_point([1.0, 2.0, 3.0]), _point([10.0, 20.0, 30.0])]
    _write_test(
        "add_point_point",
        lambda x, y: core.add(x, y),
        inputs,
        _bounds_for(ibp_mono_non_decreasing(np.add), inputs),
    )

    # relu: spans negative and positive values
    inputs = [_box([-2.0, -1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 2.0, 3.0])]
    _write_test(
        "relu_box",
        lambda x: core.relu(x),
        inputs,
        _bounds_for(ibp_mono_non_decreasing(lambda x: np.maximum(x, 0.0)), inputs),
    )

    # exp: small range to keep tolerance comfortable
    inputs = [_box([-1.0, 0.0, 1.0], [0.0, 1.0, 2.0])]
    _write_test(
        "exp_box",
        lambda x: core.exp(x),
        inputs,
        _bounds_for(ibp_mono_non_decreasing(np.exp), inputs),
    )

    # reduce_sum with explicit axes option
    inputs = [_box([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                   [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])]
    _write_test(
        "reduce_sum_box",
        lambda x: core.reduce_sum(x, axes=1),
        inputs,
        _bounds_for(ibp_mono_non_decreasing(lambda x: x.sum(axis=1)), inputs),
    )

    # expand_dims — shape-only, identity on values
    inputs = [_box([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])]
    _write_test(
        "expand_dims_box",
        lambda x: core.expand_dims(x, axes=0),
        inputs,
        _bounds_for(
            ibp_mono_non_decreasing(lambda x: np.expand_dims(x, axis=0)), inputs
        ),
    )

    # moveaxis — shape-only, identity on values
    inputs = [_box(np.zeros((2, 3, 4)) + 1.0, np.zeros((2, 3, 4)) + 2.0)]
    _write_test(
        "moveaxis_box",
        lambda x: core.moveaxis(x, source=0, destination=-1),
        inputs,
        _bounds_for(
            ibp_mono_non_decreasing(lambda x: np.moveaxis(x, 0, -1)), inputs
        ),
    )

    # reshape — shape-only, identity on values
    inputs = [_box(np.linspace(0.0, 1.0, 12).reshape(3, 4),
                   np.linspace(1.0, 2.0, 12).reshape(3, 4))]
    _write_test(
        "reshape_box",
        lambda x: core.reshape(x, new_shape=(4, 3)),
        inputs,
        _bounds_for(
            ibp_mono_non_decreasing(lambda x: x.reshape(4, 3)), inputs
        ),
    )

    # ------- monotonic non-increasing primitives -------
    # neg: bounds flip
    inputs = [_box([-1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0])]
    _write_test(
        "neg_box",
        lambda x: core.neg(x),
        inputs,
        _bounds_for(ibp_mono_non_increasing(np.negative), inputs),
    )

    # ------- linear primitives with one point operand -------
    # mul: box * point (non-negative constant)
    inputs = [
        _box([-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]),
        _point([2.0, 3.0, 4.0]),
    ]
    _write_test(
        "mul_box_point",
        lambda x, y: core.mul(x, y),
        inputs,
        _bounds_for(ibp_linear_with_one_point(np.multiply), inputs),
    )

    # dot: box @ point (non-negative constant matrix)
    inputs = [
        _box([[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]),
        _point([[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]]),
    ]
    _write_test(
        "dot_box_point",
        lambda x, y: core.dot(x, y),
        inputs,
        _bounds_for(ibp_linear_with_one_point(_np_dot), inputs),
    )

    # dot: point @ box (non-negative constant matrix on the left)
    inputs = [
        _point([[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]]),
        _box([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
             [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
    ]
    _write_test(
        "dot_point_box",
        lambda x, y: core.dot(x, y),
        inputs,
        _bounds_for(ibp_linear_with_one_point(_np_dot), inputs),
    )

    # ------- small composite: a 3 -> 2 fully connected layer with ReLU -------
    # f(x, W, b) = relu(x @ W + b)
    # x is the box (input region); W, b are point (network weights).
    x_lb = np.array([0.0, 0.0, 0.0]); x_ub = np.array([1.0, 1.0, 1.0])
    W = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 1.0]])
    b = np.array([0.0, 0.0])

    # Manual IBP ground truth (W and b are non-negative, so impl matches theory).
    x_mid = 0.5 * (x_lb + x_ub); x_ran = 0.5 * (x_ub - x_lb)
    z_mid = x_mid @ W; z_ran = x_ran @ np.abs(W)
    z_lb = z_mid - z_ran + b; z_ub = z_mid + z_ran + b
    out_lb = np.maximum(z_lb, 0.0); out_ub = np.maximum(z_ub, 0.0)

    inputs = [_box(x_lb, x_ub), _point(W), _point(b)]
    _write_test(
        "fc_relu_box",
        lambda x, w, bias: core.relu(core.add(core.dot(x, w), bias)),
        inputs,
        [(out_lb, out_ub)],
    )

    print("Done.")


if __name__ == "__main__":
    main()
