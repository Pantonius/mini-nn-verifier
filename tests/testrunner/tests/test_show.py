# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.show."""

import io
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from testrunner.show import (
    _print_bin,
    _print_network,
    _show_affine_bounds,
    _show_eval_grad,
    _show_fuzz,
    _show_test,
    _show_train,
    _show_verify,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mininn(tmp_path, name, graph_txt):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        with zf.open("graph.txt", "w") as f:
            f.write(graph_txt.encode())
    p = tmp_path / name
    p.write_bytes(buf.getvalue())
    return p


def _write_bin(path, arr):
    np.array(arr, dtype=np.float64).tofile(path)


def _capture_stdout(fn):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        fn()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _print_bin
# ---------------------------------------------------------------------------


def test_print_bin(tmp_path, capsys):
    p = tmp_path / "data.bin"
    _write_bin(p, [1.0, 2.0, 3.0])
    _print_bin(p, "test label")
    out = capsys.readouterr().out
    assert "test label" in out
    assert "data.bin" in out
    assert "shape" in out


# ---------------------------------------------------------------------------
# _print_network
# ---------------------------------------------------------------------------


def test_print_network(tmp_path, capsys):
    p = _make_mininn(tmp_path, "net.mininn", "input: a[3]\noutput: a[3]")
    _print_network(p)
    out = capsys.readouterr().out
    assert "Network" in out
    assert "input:" in out


# ---------------------------------------------------------------------------
# _show_eval_grad
# ---------------------------------------------------------------------------


def test_show_eval_grad_basic(tmp_path, capsys):
    net = _make_mininn(tmp_path, "net.mininn", "input: a[2]\noutput: a[2]")
    inp = tmp_path / "input.bin"
    _write_bin(inp, [1.0, 2.0])
    exp = tmp_path / "expected.bin"
    _write_bin(exp, [1.0, 2.0])

    config = {
        "command": "eval",
        "network": "net.mininn",
        "inputs": ["input.bin"],
        "expected_outputs": ["expected.bin"],
    }
    _show_eval_grad(tmp_path, config)
    out = capsys.readouterr().out
    assert "Command: eval" in out


def test_show_eval_grad_missing_expected(tmp_path, capsys):
    config = {
        "command": "eval",
        "network": None,
        "inputs": [],
        "expected_outputs": ["missing.bin"],
    }
    _show_eval_grad(tmp_path, config)
    out = capsys.readouterr().out
    assert "not found" in out


def test_show_eval_grad_with_actual_outputs(tmp_path, capsys):
    actual_dir = tmp_path / "actual"
    actual_dir.mkdir()
    p = actual_dir / "output.bin"
    _write_bin(p, [3.0])

    config = {"command": "grad", "network": None, "inputs": [], "expected_outputs": []}
    _show_eval_grad(tmp_path, config)
    out = capsys.readouterr().out
    assert "Actual outputs" in out


def test_show_eval_grad_no_network(tmp_path, capsys):
    config = {"command": "eval", "network": None, "inputs": [], "expected_outputs": []}
    _show_eval_grad(tmp_path, config)
    out = capsys.readouterr().out
    # No network — should not crash


def test_show_eval_grad_custom_tolerance(tmp_path, capsys):
    config = {
        "command": "eval",
        "network": None,
        "inputs": [],
        "expected_outputs": [],
        "tolerance": 1e-6,
    }
    _show_eval_grad(tmp_path, config)
    out = capsys.readouterr().out
    assert "1.0e-06" in out or "1e-06" in out


# ---------------------------------------------------------------------------
# _show_affine_bounds
# ---------------------------------------------------------------------------


def test_show_affine_bounds_basic(tmp_path, capsys):
    net = _make_mininn(tmp_path, "net.mininn", "input: a[3]\noutput: b[1]")
    _write_bin(tmp_path / "lb.bin", [0.0, 0.0, 0.0])
    _write_bin(tmp_path / "ub.bin", [1.0, 1.0, 1.0])
    _write_bin(tmp_path / "si.bin", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    _write_bin(tmp_path / "so.bin", [0.0, 3.0])
    _write_bin(tmp_path / "rlb.bin", [0.0])
    _write_bin(tmp_path / "rub.bin", [3.0])

    config = {
        "command": "affine_bounds",
        "network": "net.mininn",
        "inputs": ["box", "lb.bin", "ub.bin"],
        "input_shape": [3],
        "output_shapes": [[1]],
        "sample_inputs": ["si.bin"],
        "sample_outputs": ["so.bin"],
        "reference_lb": ["rlb.bin"],
        "reference_ub": ["rub.bin"],
        "tightness_factor": 1.5,
    }
    _show_affine_bounds(tmp_path, config)
    out = capsys.readouterr().out
    assert "Command: affine_bounds" in out
    assert "Input shape: [3]" in out
    assert "Tightness factor: 1.5" in out
    assert "input 0: box" in out
    assert "input 0 lb" in out
    assert "input 0 ub" in out
    assert "Sample inputs" in out
    assert "Reference lower bounds" in out


def test_show_affine_bounds_with_actual(tmp_path, capsys):
    _write_bin(tmp_path / "lb.bin", [0.0])
    _write_bin(tmp_path / "ub.bin", [1.0])
    actual_dir = tmp_path / "actual"
    actual_dir.mkdir()
    _write_bin(actual_dir / "output_0_lb_weight.bin", [1.0])

    config = {
        "command": "affine_bounds",
        "network": None,
        "inputs": ["box", "lb.bin", "ub.bin"],
    }
    _show_affine_bounds(tmp_path, config)
    out = capsys.readouterr().out
    assert "Actual affine bounds" in out
    assert "output_0_lb_weight" in out


def test_show_affine_bounds_no_reference(tmp_path, capsys):
    """Without reference bounds the tightness factor is not printed."""
    config = {"command": "affine_bounds", "network": None, "inputs": []}
    _show_affine_bounds(tmp_path, config)
    out = capsys.readouterr().out
    assert "Tightness factor" not in out


# ---------------------------------------------------------------------------
# _show_verify
# ---------------------------------------------------------------------------


def test_show_verify_basic(tmp_path, capsys):
    _make_mininn(tmp_path, "net.mininn", "input: a[3]\noutput: b[1]")
    _write_bin(tmp_path / "lb.bin", [0.0, 0.0, 0.0])
    _write_bin(tmp_path / "ub.bin", [1.0, 1.0, 1.0])

    config = {
        "command": "verify",
        "network": "net.mininn",
        "inputs": ["box", "lb.bin", "ub.bin"],
        "expected_verdict": "sat",
        "box_atol": 1e-9,
    }
    _show_verify(tmp_path, config)
    out = capsys.readouterr().out
    assert "Command: verify" in out
    assert "Expected verdict: sat" in out
    assert "input 0: box" in out


def test_show_verify2_with_counterexample(tmp_path, capsys):
    _write_bin(tmp_path / "lb.bin", [0.0, 0.0, 0.0])
    _write_bin(tmp_path / "ub.bin", [1.0, 1.0, 1.0])
    actual_dir = tmp_path / "actual"
    actual_dir.mkdir()
    _write_bin(actual_dir / "counterexample_0.bin", [0.5, 0.5, 0.5])

    config = {
        "command": "verify2",
        "network": None,
        "inputs": ["box", "lb.bin", "ub.bin"],
        "expected_verdict": "viol",
    }
    _show_verify(tmp_path, config)
    out = capsys.readouterr().out
    assert "Command: verify2" in out
    assert "Counterexamples" in out
    assert "counterexample 0" in out


def test_show_verify_point_input(tmp_path, capsys):
    """A 'point' input marker consumes a single file path."""
    _write_bin(tmp_path / "p.bin", [1.0, 2.0])
    config = {
        "command": "verify",
        "network": None,
        "inputs": ["point", "p.bin"],
        "expected_verdict": "sat",
    }
    _show_verify(tmp_path, config)
    out = capsys.readouterr().out
    assert "input 0: point" in out


# ---------------------------------------------------------------------------
# _show_train
# ---------------------------------------------------------------------------


def test_show_train(tmp_path, capsys):
    config = {
        "command": "train",
        "dataset": "mnist_mlp",
        "in_size": 784,
        "num_classes": 10,
    }
    _show_train(tmp_path, config)
    out = capsys.readouterr().out
    assert "Command: train" in out
    assert "dataset" in out
    assert "command" not in out.split("Command: train")[1][:50]


# ---------------------------------------------------------------------------
# _show_fuzz
# ---------------------------------------------------------------------------


def test_show_fuzz_no_failures(tmp_path, capsys):
    config = {"command": "fuzz_eval", "n_trials": 100}
    _show_fuzz(tmp_path, config)
    out = capsys.readouterr().out
    assert "Command: fuzz_eval" in out
    assert "No saved fuzz failures" in out


def test_show_fuzz_with_failures(tmp_path, capsys):
    failures_dir = tmp_path / "actual" / "fuzz_failures"
    failures_dir.mkdir(parents=True)

    # Create two case directories
    for i in range(2):
        case = failures_dir / str(i)
        case.mkdir()
        (case / "error.txt").write_text(f"error {i}")
        (case / "stderr.txt").write_text(f"stderr {i}")
        net = _make_mininn(case, "network.mininn", "input: a[2]\noutput: a[2]")
        inp = case / "a.bin"
        _write_bin(inp, [1.0, 2.0])

    config = {"command": "fuzz_eval", "n_trials": 50, "seed": 42}
    _show_fuzz(tmp_path, config)
    out = capsys.readouterr().out
    assert "Saved failing cases: 2" in out
    assert "error 0" in out
    assert "error 1" in out


def test_show_fuzz_empty_case_dirs(tmp_path, capsys):
    failures_dir = tmp_path / "actual" / "fuzz_failures"
    failures_dir.mkdir(parents=True)
    # Failures dir exists but no numbered case dirs
    config = {"command": "fuzz_eval", "n_trials": 10}
    _show_fuzz(tmp_path, config)
    out = capsys.readouterr().out
    assert "No saved fuzz failures" in out


def test_show_fuzz_case_with_empty_stderr(tmp_path, capsys):
    """Case with empty stderr.txt should not print stderr section."""
    failures_dir = tmp_path / "actual" / "fuzz_failures"
    failures_dir.mkdir(parents=True)
    case = failures_dir / "0"
    case.mkdir()
    (case / "error.txt").write_text("err")
    (case / "stderr.txt").write_text("")  # empty

    config = {"command": "fuzz_eval", "n_trials": 10}
    _show_fuzz(tmp_path, config)
    out = capsys.readouterr().out
    assert "--- stderr ---" not in out


def test_show_fuzz_no_seed(tmp_path, capsys):
    config = {"command": "fuzz_grad", "n_trials": 20}
    _show_fuzz(tmp_path, config)
    out = capsys.readouterr().out
    assert "Seed" not in out


# ---------------------------------------------------------------------------
# _show_test
# ---------------------------------------------------------------------------


def test_show_test_eval(tmp_path, capsys):
    config = {"command": "eval", "network": None, "inputs": [], "expected_outputs": []}
    (tmp_path / "test.json").write_text(json.dumps(config))
    _show_test(tmp_path)
    out = capsys.readouterr().out
    assert "Command: eval" in out


def test_show_test_grad(tmp_path, capsys):
    config = {"command": "grad", "network": None, "inputs": [], "expected_outputs": []}
    (tmp_path / "test.json").write_text(json.dumps(config))
    _show_test(tmp_path)
    out = capsys.readouterr().out
    assert "Command: grad" in out


def test_show_test_fuzz_eval(tmp_path, capsys):
    config = {"command": "fuzz_eval", "n_trials": 5}
    (tmp_path / "test.json").write_text(json.dumps(config))
    _show_test(tmp_path)
    out = capsys.readouterr().out
    assert "fuzz_eval" in out


def test_show_test_fuzz_grad(tmp_path, capsys):
    config = {"command": "fuzz_grad", "n_trials": 5}
    (tmp_path / "test.json").write_text(json.dumps(config))
    _show_test(tmp_path)
    out = capsys.readouterr().out
    assert "fuzz_grad" in out


def test_show_test_affine_bounds(tmp_path, capsys):
    config = {"command": "affine_bounds", "network": None, "inputs": []}
    (tmp_path / "test.json").write_text(json.dumps(config))
    _show_test(tmp_path)
    out = capsys.readouterr().out
    assert "Command: affine_bounds" in out


def test_show_test_verify(tmp_path, capsys):
    config = {"command": "verify", "network": None, "inputs": [], "expected_verdict": "sat"}
    (tmp_path / "test.json").write_text(json.dumps(config))
    _show_test(tmp_path)
    out = capsys.readouterr().out
    assert "Command: verify" in out
    assert "Expected verdict: sat" in out


def test_show_test_verify2(tmp_path, capsys):
    config = {"command": "verify2", "network": None, "inputs": [], "expected_verdict": "viol"}
    (tmp_path / "test.json").write_text(json.dumps(config))
    _show_test(tmp_path)
    out = capsys.readouterr().out
    assert "Command: verify2" in out


def test_show_test_train(tmp_path, capsys):
    config = {"command": "train", "dataset": "mnist", "in_size": 4, "num_classes": 2}
    (tmp_path / "test.json").write_text(json.dumps(config))
    _show_test(tmp_path)
    out = capsys.readouterr().out
    assert "Command: train" in out


def test_show_test_unknown_command(tmp_path, capsys):
    config = {"command": "mystery"}
    (tmp_path / "test.json").write_text(json.dumps(config))
    _show_test(tmp_path)
    out = capsys.readouterr().out
    assert "mystery" in out
    assert "no detailed display" in out


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_single_test(tmp_path, capsys):
    config = {"command": "eval", "network": None, "inputs": [], "expected_outputs": []}
    (tmp_path / "test.json").write_text(json.dumps(config))

    with patch("sys.argv", ["show", str(tmp_path)]):
        main()
    out = capsys.readouterr().out
    assert "Command: eval" in out


def test_main_directory_of_tests(tmp_path, capsys):
    for name in ["t1", "t2"]:
        d = tmp_path / name
        d.mkdir()
        (d / "test.json").write_text(json.dumps({"command": "eval", "network": None,
                                                  "inputs": [], "expected_outputs": []}))
    with patch("sys.argv", ["show", str(tmp_path)]):
        main()
    out = capsys.readouterr().out
    assert "t1" in out
    assert "t2" in out


def test_main_no_tests(tmp_path, capsys):
    with patch("sys.argv", ["show", str(tmp_path)]):
        main()
    out = capsys.readouterr().out
    assert "No tests found" in out


def test_module_guard(tmp_path):
    """The if __name__ == '__main__': guard in show.py calls main()."""
    import runpy
    with patch("sys.argv", ["show", str(tmp_path)]):
        runpy.run_module("testrunner.show", run_name="__main__", alter_sys=False)
