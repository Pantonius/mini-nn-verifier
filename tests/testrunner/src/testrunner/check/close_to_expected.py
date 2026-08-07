# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Check that outputs are close to expected values."""

import numpy as np


def check_close_to_expected(test_dir, config, output_files, closed=False):
    expected_names = config.get("expected_outputs", [])
    tolerance = config.get("tolerance", 1e-4)

    if len(output_files) != len(expected_names):
        return {
            "passed": False,
            "error": (f"expected {len(expected_names)} output file(s), got {len(output_files)}"),
        }

    for out_file, exp_name in zip(output_files, expected_names):
        actual = np.fromfile(out_file, dtype=np.float64)
        expected = np.fromfile(test_dir / exp_name, dtype=np.float64)

        if actual.shape != expected.shape:
            return {
                "passed": False,
                "error": (
                    f"{out_file.name}: element count mismatch: "
                    f"actual {actual.size} vs expected {expected.size}"
                ),
            }

        if not np.allclose(actual, expected, atol=tolerance, rtol=0):
            if closed:
                error = f"{out_file.name}: max absolute diff exceeds tolerance threshold"
            else:
                diff = np.abs(actual - expected)
                worst = int(np.argmax(diff))
                error = (
                    f"{out_file.name}: max absolute diff "
                    f"{diff[worst]:.6e} > tolerance {tolerance:.1e} "
                    f"(index {worst}: expected {expected[worst]:.6e}, "
                    f"got {actual[worst]:.6e})"
                )
            return {"passed": False, "error": error}

    return {"passed": True, "error": None}
