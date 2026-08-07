#!/usr/bin/env python3
# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Generate eval tests for the same operations that expose grad bugs.

These eval tests verify that forward evaluation is correct for
broadcasting mul/add and 1D dot — the same operations whose VJP rules
are buggy.

Run from the repository root:
    python tests/milestone1/base/unit/eval/generate_bugfix_eval_tests.py
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


def create_test(name, fn, inputs_data, *, tolerance=DEFAULT_TOLERANCE):
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

    raw_out = fn(*arrays)
    if not isinstance(raw_out, (list, tuple)):
        raw_out = [raw_out]

    expected_files = []
    for i, out in enumerate(raw_out):
        fname = f"expected_output_{i}.bin"
        out.array.astype(np.float64).tofile(test_dir / fname)
        expected_files.append(fname)

    config = {
        "command": "eval",
        "network": f"resources/{network_file}",
        "inputs": input_files,
        "expected_outputs": expected_files,
    }
    if tolerance != DEFAULT_TOLERANCE:
        config["tolerance"] = tolerance
    (test_dir / "test.json").write_text(json.dumps(config, indent=4) + "\n")
    print(f"  created {test_dir.relative_to(REPO_ROOT)}")


def main():
    rng = np.random.default_rng(100)

    # mul with broadcasting: x (3, 4), y (1, 4)
    x_data = rng.standard_normal((3, 4))
    y_data = rng.standard_normal((1, 4))
    create_test(
        "mul_broadcast",
        lambda x, y: core.mul(x, y),
        [x_data, y_data],
    )

    # add with broadcasting: x (3, 4), y (1, 4)
    create_test(
        "add_broadcast",
        lambda x, y: core.add(x, y),
        [x_data, y_data],
    )

    # dot with 1D input: x (8,), y (8, 5)
    x_dot = rng.standard_normal((8,))
    y_dot = rng.standard_normal((8, 5))
    create_test(
        "dot_1d",
        lambda x, y: core.dot(x, y),
        [x_dot, y_dot],
    )

    print("Done.")


if __name__ == "__main__":
    main()
