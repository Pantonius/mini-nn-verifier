# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.reproduce."""

import io
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from testrunner.reproduce import _print_failure_inputs, reproduce


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_failure_dir(tmp_path, mode="eval", check_nan_inf=False, expected_shapes=None,
                      inputs=None, include_error=True, include_stderr=True):
    """Create a minimal saved-failure directory."""
    fd = tmp_path / "failure"
    fd.mkdir()

    # Write network.mininn (a minimal zip)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        with zf.open("graph.txt", "w") as f:
            f.write(b"input: a[2]\na[2] = neg{} a[2]\noutput: a[2]")
    (fd / "network.mininn").write_bytes(buf.getvalue())

    # Write input bin files
    if inputs is None:
        inp = np.array([1.0, 2.0])
        inp.tofile(fd / "a.bin")
        input_names = ["a.bin"]
    else:
        input_names = []
        for name, arr in inputs.items():
            np.array(arr, dtype=np.float64).tofile(fd / name)
            input_names.append(name)

    # Write metadata
    metadata = {
        "mode": mode,
        "check_nan_inf": check_nan_inf,
        "expected_shapes": [list(s) for s in (expected_shapes or [(2,)])],
        "inputs": input_names,
    }
    (fd / "metadata.json").write_text(json.dumps(metadata))

    if include_error:
        (fd / "error.txt").write_text("crash (exit 1): error message")
    if include_stderr:
        (fd / "stderr.txt").write_text("stderr output")

    return fd


# ---------------------------------------------------------------------------
# reproduce()
# ---------------------------------------------------------------------------


def test_reproduce_missing_metadata(tmp_path):
    fd = tmp_path / "failure"
    fd.mkdir()
    result = reproduce(fd, "local", "sut")
    assert result["passed"] is False
    assert "no metadata.json" in result["error"]


def test_reproduce_missing_network(tmp_path):
    fd = tmp_path / "failure"
    fd.mkdir()
    (fd / "metadata.json").write_text(json.dumps({
        "mode": "eval", "check_nan_inf": False,
        "expected_shapes": [[2]], "inputs": []
    }))
    result = reproduce(fd, "local", "sut")
    assert result["passed"] is False
    assert "no network.mininn" in result["error"]


def test_reproduce_missing_input_file(tmp_path):
    fd = tmp_path / "failure"
    fd.mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        with zf.open("graph.txt", "w") as f:
            f.write(b"input: a[2]\noutput: a[2]")
    (fd / "network.mininn").write_bytes(buf.getvalue())
    (fd / "metadata.json").write_text(json.dumps({
        "mode": "eval", "check_nan_inf": False,
        "expected_shapes": [[2]], "inputs": ["missing.bin"]
    }))
    result = reproduce(fd, "local", "sut")
    assert result["passed"] is False
    assert "missing input file" in result["error"]


def test_reproduce_calls_run_and_check(tmp_path):
    fd = _make_failure_dir(tmp_path)
    expected = {"passed": True, "error": None}
    with patch("testrunner.reproduce.run_and_check", return_value=expected) as mock_rac:
        result = reproduce(fd, "local", "sut")
    mock_rac.assert_called_once()
    assert result["passed"] is True


def test_reproduce_passes_correct_args(tmp_path):
    fd = _make_failure_dir(tmp_path, mode="grad", check_nan_inf=True,
                           expected_shapes=[(3, 4)])
    calls_seen = []

    def mock_rac(network, inputs, backend, backend_arg, mode, expected_shapes,
                 check_nan_inf=False, timeout=60, extra_run_args=()):
        calls_seen.append({
            "mode": mode, "check_nan_inf": check_nan_inf,
            "expected_shapes": expected_shapes,
        })
        return {"passed": True, "error": None}

    with patch("testrunner.reproduce.run_and_check", side_effect=mock_rac):
        reproduce(fd, "local", "sut")

    assert len(calls_seen) == 1
    assert calls_seen[0]["mode"] == "grad"
    assert calls_seen[0]["check_nan_inf"] is True
    assert calls_seen[0]["expected_shapes"] == [(3, 4)]


# ---------------------------------------------------------------------------
# _print_failure_inputs
# ---------------------------------------------------------------------------


def _capture_stderr(fn):
    buf = io.StringIO()
    with patch("sys.stderr", buf):
        fn()
    return buf.getvalue()


def test_print_failure_inputs_with_network(tmp_path):
    fd = _make_failure_dir(tmp_path)
    out = _capture_stderr(lambda: _print_failure_inputs(fd))
    assert "Network" in out
    assert "input:" in out  # from graph.txt content


def test_print_failure_inputs_no_network(tmp_path):
    fd = tmp_path / "failure"
    fd.mkdir()
    # Create metadata and a bin file but no network
    (fd / "metadata.json").write_text(json.dumps({
        "mode": "eval", "check_nan_inf": False,
        "expected_shapes": [[2]], "inputs": ["a.bin"]
    }))
    np.array([1.0, 2.0]).tofile(fd / "a.bin")
    # Should not raise
    _capture_stderr(lambda: _print_failure_inputs(fd))


def test_print_failure_inputs_with_error_file(tmp_path):
    fd = _make_failure_dir(tmp_path)
    out = _capture_stderr(lambda: _print_failure_inputs(fd))
    assert "Original error" in out


def test_print_failure_inputs_no_metadata(tmp_path):
    """Without metadata.json, should scan for .bin files."""
    fd = tmp_path / "failure"
    fd.mkdir()
    np.array([1.0]).tofile(fd / "x.bin")
    # Should not raise
    _capture_stderr(lambda: _print_failure_inputs(fd))


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_pass(tmp_path, capsys):
    from testrunner.reproduce import main

    fd = _make_failure_dir(tmp_path)
    with patch("sys.argv", ["reproduce", "local", "sut", str(fd)]):
        with patch("testrunner.reproduce.run_and_check", return_value={"passed": True, "error": None}):
            main()  # should not raise


def test_main_fail_exits_1(tmp_path):
    from testrunner.reproduce import main

    fd = _make_failure_dir(tmp_path)
    with patch("sys.argv", ["reproduce", "local", "sut", str(fd)]):
        with patch("testrunner.reproduce.run_and_check",
                   return_value={"passed": False, "error": "crash",
                                 "stdout": "out", "stderr": "err"}):
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 1


def test_module_guard(tmp_path):
    """The if __name__ == '__main__': guard in reproduce.py calls main()."""
    import runpy
    from unittest.mock import MagicMock
    fd = _make_failure_dir(tmp_path)
    out_file = tmp_path / "result.bin"
    np.zeros(2, dtype=np.float64).tofile(out_file)

    with patch("sys.argv", ["reproduce", "local", "sut", str(fd)]):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=str(out_file) + "\n", stderr=""
            )
            runpy.run_module("testrunner.reproduce", run_name="__main__", alter_sys=False)
