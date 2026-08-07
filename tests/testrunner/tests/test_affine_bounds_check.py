# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.check.affine_bounds_within_range."""

import numpy as np

from testrunner.check import CHECKS, DEFAULT_CHECKS
from testrunner.check.affine_bounds_within_range import check_affine_bounds_within_range
from testrunner.commands import COMMANDS, command_sort_key


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry():
    assert "affine_bounds_within_range" in CHECKS
    assert DEFAULT_CHECKS["affine_bounds"] == "affine_bounds_within_range"
    assert "affine_bounds" in COMMANDS
    # affine_bounds is cheap (≈ bounds), so it sorts just after bounds.
    assert command_sort_key("affine_bounds") == command_sort_key("bounds") + 1


# ---------------------------------------------------------------------------
# Fixtures: a scalar affine target f(x) = sum(x) over the unit box [0,1]^3.
# ---------------------------------------------------------------------------


def _make(tmp_path, *, ref=(0.0, 3.0)):
    n = 3
    (tmp_path / "input_0_lb.bin").write_bytes(np.zeros(n).tobytes())
    (tmp_path / "input_0_ub.bin").write_bytes(np.ones(n).tobytes())
    xs = np.array([[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [0.2, 0.9, 0.4]], dtype=np.float64)
    ys = xs.sum(axis=1)
    xs.tofile(tmp_path / "sample_inputs_0.bin")
    ys.tofile(tmp_path / "sample_outputs_0.bin")
    np.array([ref[0]]).tofile(tmp_path / "reference_output_0_lb.bin")
    np.array([ref[1]]).tofile(tmp_path / "reference_output_0_ub.bin")
    return {
        "command": "affine_bounds",
        "inputs": ["box", "input_0_lb.bin", "input_0_ub.bin"],
        "check": "affine_bounds_within_range",
        "input_shape": [n],
        "output_shapes": [[1]],
        "sample_inputs": ["sample_inputs_0.bin"],
        "sample_outputs": ["sample_outputs_0.bin"],
        "reference_lb": ["reference_output_0_lb.bin"],
        "reference_ub": ["reference_output_0_ub.bin"],
        "tightness_factor": 1.5,
        "sample_atol": 1e-9,
    }


def _files(tmp_path, lbw, lbb, ubw, ubb):
    out = []
    for name, arr in (("lb_weight", lbw), ("lb_bias", [lbb]), ("ub_weight", ubw), ("ub_bias", [ubb])):
        p = tmp_path / f"output_0_{name}.bin"
        np.asarray(arr, dtype=np.float64).tofile(p)
        out.append(p)
    return out


def test_exact_affine_bound_passes(tmp_path):
    cfg = _make(tmp_path)
    files = _files(tmp_path, [1, 1, 1], 0.0, [1, 1, 1], 0.0)  # f(x) = sum(x) exactly
    r = check_affine_bounds_within_range(tmp_path, cfg, files)
    assert r["passed"] is True and r["error"] is None


def test_ibp_as_affine_bound_passes(tmp_path):
    """Zero-weight bounds equal to the reference interval are valid (if loose)."""
    cfg = _make(tmp_path)
    files = _files(tmp_path, [0, 0, 0], 0.0, [0, 0, 0], 3.0)
    r = check_affine_bounds_within_range(tmp_path, cfg, files)
    assert r["passed"] is True


def test_unsound_lower_bound_fails(tmp_path):
    cfg = _make(tmp_path)
    # same nonzero weight on both bounds keeps validity, but shifts lb above f.
    files = _files(tmp_path, [1, 1, 1], 0.5, [1, 1, 1], 3.0)
    r = check_affine_bounds_within_range(tmp_path, cfg, files)
    assert r["passed"] is False
    assert "lower affine bound unsound" in r["error"]


def test_unsound_upper_bound_fails(tmp_path):
    cfg = _make(tmp_path)
    files = _files(tmp_path, [1, 1, 1], -3.0, [1, 1, 1], -0.5)
    r = check_affine_bounds_within_range(tmp_path, cfg, files)
    assert r["passed"] is False
    assert "upper affine bound unsound" in r["error"]


def test_inverted_bounds_fail(tmp_path):
    cfg = _make(tmp_path)
    files = _files(tmp_path, [0, 0, 0], 3.0, [0, 0, 0], 0.0)  # lb_aff > ub_aff everywhere
    r = check_affine_bounds_within_range(tmp_path, cfg, files)
    assert r["passed"] is False
    assert "lower affine bound exceeds upper affine bound" in r["error"]


def test_too_loose_fails(tmp_path):
    cfg = _make(tmp_path)
    files = _files(tmp_path, [0, 0, 0], -10.0, [0, 0, 0], 10.0)  # ~6.6x the IBP width
    r = check_affine_bounds_within_range(tmp_path, cfg, files)
    assert r["passed"] is False
    assert "looser than" in r["error"]


def test_tightness_skipped_without_reference(tmp_path):
    cfg = _make(tmp_path)
    del cfg["reference_lb"]
    del cfg["reference_ub"]
    files = _files(tmp_path, [0, 0, 0], -10.0, [0, 0, 0], 10.0)  # loose but sound
    r = check_affine_bounds_within_range(tmp_path, cfg, files)
    assert r["passed"] is True


def test_wrong_file_count_fails(tmp_path):
    cfg = _make(tmp_path)
    files = _files(tmp_path, [1, 1, 1], 0.0, [1, 1, 1], 0.0)[:3]
    r = check_affine_bounds_within_range(tmp_path, cfg, files)
    assert r["passed"] is False
    assert "expected 4 affine-bound files" in r["error"]


def test_wrong_weight_size_fails(tmp_path):
    cfg = _make(tmp_path)
    files = _files(tmp_path, [1, 1], 0.0, [1, 1, 1], 0.0)  # lb_weight too short
    r = check_affine_bounds_within_range(tmp_path, cfg, files)
    assert r["passed"] is False
    assert "lb_weight" in r["error"]
