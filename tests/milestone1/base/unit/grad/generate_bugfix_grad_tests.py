#!/usr/bin/env python3
# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Generate grad tests that expose known bugs in minijax's gradient logic.

Bug 1 – unbroadcast is incomplete (grad.py:73-75):
    unbroadcast only removes extra leading dimensions but does not reduce
    along size-1 broadcast dimensions.  Tests: mul_broadcast, add_broadcast

Bug 2 – dot VJP is wrong for 1D inputs (grad.py:90):
    transpose(x) @ t computes an inner product instead of an outer product
    when both x and t are 1D vectors.  Test: dot_1d

Expected gradients are computed analytically with numpy so they are
independent of the buggy _backwards implementation.

Run from the repository root:
    python tests/milestone1/base/unit/grad/generate_bugfix_grad_tests.py
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


def create_test(name, fn, inputs_data, expected_grads, *, tolerance=DEFAULT_TOLERANCE):
    """Write a grad test whose expected gradients are supplied externally."""
    test_dir = SCRIPT_DIR / name
    resources_dir = test_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)

    arrays = [Array(d) for d in inputs_data]
    graph = make_graph(fn)(*arrays)

    network_file = f"{name}_network.mininn"
    dump(graph, resources_dir / network_file)

    input_files = []
    for i, d in enumerate(inputs_data):
        fname = f"input_{i}.bin"
        np.asarray(d, dtype=np.float64).tofile(resources_dir / fname)
        input_files.append(f"resources/{fname}")

    expected_files = []
    for i, grad in enumerate(expected_grads):
        fname = f"expected_grad_{i}.bin"
        np.asarray(grad, dtype=np.float64).tofile(test_dir / fname)
        expected_files.append(fname)

    config = {
        "command": "grad",
        "network": f"resources/{network_file}",
        "inputs": input_files,
        "expected_outputs": expected_files,
    }
    if tolerance != DEFAULT_TOLERANCE:
        config["tolerance"] = tolerance
    (test_dir / "test.json").write_text(json.dumps(config, indent=4) + "\n")
    print(f"  created {test_dir.relative_to(REPO_ROOT)}")


def main():
    # ------------------------------------------------------------------
    # Bug 1: unbroadcast incomplete — mul with broadcasting
    # f(x, y) = x * y,  x shape (3, 4),  y shape (1, 4)
    # d(sum f)/dx = broadcast(y, (3,4))
    # d(sum f)/dy = x.sum(axis=0, keepdims=True)   ← requires reducing
    #              along the size-1 broadcast axis, which unbroadcast misses
    # ------------------------------------------------------------------
    rng = np.random.default_rng(100)
    x_data = rng.standard_normal((3, 4))
    y_data = rng.standard_normal((1, 4))

    create_test(
        "mul_broadcast",
        lambda x, y: core.mul(x, y),
        [x_data, y_data],
        expected_grads=[
            np.broadcast_to(y_data, (3, 4)),     # dL/dx
            x_data.sum(axis=0, keepdims=True),    # dL/dy
        ],
    )

    # Same bug, simpler case: add with broadcasting
    # f(x, y) = x + y,  x shape (3, 4),  y shape (1, 4)
    # d(sum f)/dx = ones(3, 4)
    # d(sum f)/dy = ones(1, 4) * 3  (sum of ones along broadcast axis)
    create_test(
        "add_broadcast",
        lambda x, y: core.add(x, y),
        [x_data, y_data],
        expected_grads=[
            np.ones((3, 4)),                      # dL/dx
            np.full((1, 4), 3.0),                 # dL/dy
        ],
    )

    # ------------------------------------------------------------------
    # Bug 2: dot VJP wrong for 1D inputs
    # f(x, y) = dot(x, y),  x shape (8,),  y shape (8, 5)
    # output shape (5,);  out_tangent = ones(5,)
    # d(sum f)/dx_i = sum_j y_{ij}  →  y.sum(axis=1),  shape (8,)
    # d(sum f)/dy_{ij} = x_i         →  x[:, None] * ones(1, 5), shape (8, 5)
    # The buggy VJP computes transpose(x) @ t = dot((8,), (5,)) which
    # raises an error (shape mismatch) or gives a scalar (if sizes equal).
    # ------------------------------------------------------------------
    x_dot = rng.standard_normal((8,))
    y_dot = rng.standard_normal((8, 5))

    create_test(
        "dot_1d",
        lambda x, y: core.dot(x, y),
        [x_dot, y_dot],
        expected_grads=[
            y_dot.sum(axis=1),                              # dL/dx, shape (8,)
            np.broadcast_to(x_dot[:, None], (8, 5)).copy(), # dL/dy, shape (8, 5)
        ],
    )

    print("Done.")


if __name__ == "__main__":
    main()
