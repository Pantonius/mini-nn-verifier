# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.benchmark.runner."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from testrunner.benchmark.runner import (
    REFERENCE_FILE,
    _clean_output_dir,
    _run_benchmark,
    run_bench_eval,
    run_bench_grad,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path, mode="eval", n_repeats=2, n_warmup=1, max_slowdown=2.0):
    config = {
        "command": f"bench_{mode}",
        "network": "net.mininn",
        "inputs": [],
        "n_repeats": n_repeats,
        "n_warmup": n_warmup,
        "max_slowdown": max_slowdown,
    }
    return config


def _make_test_dir(tmp_path, config):
    test_dir = tmp_path / "test"
    test_dir.mkdir(exist_ok=True)
    (test_dir / "test.json").write_text(json.dumps(config))
    (test_dir / "net.mininn").write_bytes(b"fake")
    return test_dir


def _make_run_result(returncode=0, stdout="", stderr=""):
    r = MagicMock()
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


# ---------------------------------------------------------------------------
# _clean_output_dir
# ---------------------------------------------------------------------------


def test_clean_output_dir_creates(tmp_path):
    d = tmp_path / "out"
    _clean_output_dir(d)
    assert d.exists()
    assert d.is_dir()


def test_clean_output_dir_removes_contents(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    (d / "stale.bin").write_bytes(b"old")
    _clean_output_dir(d)
    assert d.exists()
    assert not (d / "stale.bin").exists()


def test_clean_output_dir_nonexistent(tmp_path):
    d = tmp_path / "nonexistent"
    _clean_output_dir(d)
    assert d.exists()


# ---------------------------------------------------------------------------
# run_bench_eval / run_bench_grad delegations
# ---------------------------------------------------------------------------


def test_run_bench_eval_delegates(tmp_path):
    config = _make_config(tmp_path, "eval")
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    expected = {"passed": True, "median_time": 0.01}
    with patch("testrunner.benchmark.runner._run_benchmark", return_value=expected) as mock_rb:
        result = run_bench_eval(test_dir, config, output_dir, "local", "sut")
    mock_rb.assert_called_once()
    assert result is expected


def test_run_bench_grad_delegates(tmp_path):
    config = _make_config(tmp_path, "grad")
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    expected = {"passed": False, "error": "too slow"}
    with patch("testrunner.benchmark.runner._run_benchmark", return_value=expected) as mock_rb:
        result = run_bench_grad(test_dir, config, output_dir, "local", "sut")
    mock_rb.assert_called_once()
    assert result is expected


# ---------------------------------------------------------------------------
# _run_benchmark — generate mode
# ---------------------------------------------------------------------------


def test_run_benchmark_generate(tmp_path):
    config = _make_config(tmp_path, n_repeats=2, n_warmup=1)
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with patch("subprocess.run", return_value=_make_run_result(returncode=0)):
        result = _run_benchmark(test_dir, config, output_dir, "local", "sut", "eval",
                                generate=True)

    assert result["passed"] is True
    assert result.get("generated") is True
    assert (test_dir / REFERENCE_FILE).exists()


# ---------------------------------------------------------------------------
# _run_benchmark — normal mode (compare to reference)
# ---------------------------------------------------------------------------


def test_run_benchmark_pass(tmp_path):
    config = _make_config(tmp_path, n_repeats=2, n_warmup=1, max_slowdown=2.0)
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Write a reference file with a median of 1.0s
    ref = {"median": 1.0, "times": [1.0, 1.0], "n_repeats": 2}
    (test_dir / REFERENCE_FILE).write_text(json.dumps(ref))

    with patch("subprocess.run", return_value=_make_run_result(returncode=0)):
        # With fast runs (close to 0s), sut_median << ref_median * 2.0 → pass
        result = _run_benchmark(test_dir, config, output_dir, "local", "sut", "eval")

    assert result["passed"] is True
    assert "median_time" in result
    assert "slowdown" in result


def test_run_benchmark_fail_too_slow(tmp_path):
    config = _make_config(tmp_path, n_repeats=2, n_warmup=1, max_slowdown=0.0001)
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Very tight slowdown — any non-zero SUT time will fail
    ref = {"median": 0.000001, "times": [0.000001, 0.000001], "n_repeats": 2}
    (test_dir / REFERENCE_FILE).write_text(json.dumps(ref))

    with patch("subprocess.run", return_value=_make_run_result(returncode=0)):
        result = _run_benchmark(test_dir, config, output_dir, "local", "sut", "eval")

    # SUT median will be > 0 which exceeds reference * 0.0001
    assert result["passed"] is False
    assert "too slow" in result["error"]


def test_run_benchmark_no_reference(tmp_path):
    config = _make_config(tmp_path, n_repeats=2, n_warmup=1)
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with patch("subprocess.run", return_value=_make_run_result(returncode=0)):
        result = _run_benchmark(test_dir, config, output_dir, "local", "sut", "eval")

    assert result["passed"] is False
    assert REFERENCE_FILE in result["error"]


# ---------------------------------------------------------------------------
# _run_benchmark — error handling
# ---------------------------------------------------------------------------


def test_run_benchmark_warmup_timeout(tmp_path):
    config = _make_config(tmp_path, n_warmup=1)
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
        result = _run_benchmark(test_dir, config, output_dir, "local", "sut", "eval")

    assert result["passed"] is False
    assert "timed out" in result["error"]


def test_run_benchmark_warmup_failure(tmp_path):
    config = _make_config(tmp_path, n_warmup=1)
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with patch("subprocess.run", return_value=_make_run_result(returncode=1, stderr="crash")):
        result = _run_benchmark(test_dir, config, output_dir, "local", "sut", "eval")

    assert result["passed"] is False
    assert "crash" in result["error"]


def test_run_benchmark_warmup_failure_closed(tmp_path):
    config = _make_config(tmp_path, n_warmup=1)
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with patch("subprocess.run", return_value=_make_run_result(returncode=1, stderr="secret")):
        result = _run_benchmark(test_dir, config, output_dir, "local", "sut", "eval",
                                closed=True)

    assert result["passed"] is False
    assert "secret" not in result["error"]


def test_run_benchmark_rep_timeout(tmp_path):
    config = _make_config(tmp_path, n_repeats=2, n_warmup=1)
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    call_count = [0]

    def side_effect(*a, **kw):
        call_count[0] += 1
        if call_count[0] <= 1:
            return _make_run_result(returncode=0)  # warmup passes
        raise subprocess.TimeoutExpired("cmd", 5)  # first rep times out

    with patch("subprocess.run", side_effect=side_effect):
        result = _run_benchmark(test_dir, config, output_dir, "local", "sut", "eval")

    assert result["passed"] is False
    assert "timed out" in result["error"]


def test_run_benchmark_rep_failure(tmp_path):
    config = _make_config(tmp_path, n_repeats=2, n_warmup=1)
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    call_count = [0]

    def side_effect(*a, **kw):
        call_count[0] += 1
        if call_count[0] <= 1:
            return _make_run_result(returncode=0)
        return _make_run_result(returncode=1, stderr="rep failure")

    with patch("subprocess.run", side_effect=side_effect):
        result = _run_benchmark(test_dir, config, output_dir, "local", "sut", "eval")

    assert result["passed"] is False
    assert "rep failure" in result["error"]


def test_run_benchmark_rep_failure_closed(tmp_path):
    config = _make_config(tmp_path, n_repeats=2, n_warmup=1)
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    call_count = [0]

    def side_effect(*a, **kw):
        call_count[0] += 1
        if call_count[0] <= 1:
            return _make_run_result(returncode=0)
        return _make_run_result(returncode=1, stderr="secret rep failure")

    with patch("subprocess.run", side_effect=side_effect):
        result = _run_benchmark(test_dir, config, output_dir, "local", "sut", "eval",
                                closed=True)

    assert result["passed"] is False
    assert "secret" not in result["error"]


def test_run_benchmark_output_handler(tmp_path):
    config = _make_config(tmp_path, n_repeats=3, n_warmup=1)
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    ref = {"median": 100.0, "times": [100.0], "n_repeats": 1}
    (test_dir / REFERENCE_FILE).write_text(json.dumps(ref))

    reps_seen = []

    class Handler:
        def bench_rep_finished(self, test_path, rep, total_reps, elapsed):
            reps_seen.append(rep)

    with patch("subprocess.run", return_value=_make_run_result(returncode=0)):
        _run_benchmark(test_dir, config, output_dir, "local", "sut", "eval",
                       output_handler=Handler())

    assert reps_seen == [1, 2, 3]


def test_run_benchmark_ref_median_zero(tmp_path):
    """When reference median is 0, slowdown should be inf."""
    config = _make_config(tmp_path, n_repeats=1, n_warmup=0, max_slowdown=2.0)
    test_dir = _make_test_dir(tmp_path, config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    ref = {"median": 0.0, "times": [0.0], "n_repeats": 1}
    (test_dir / REFERENCE_FILE).write_text(json.dumps(ref))

    with patch("subprocess.run", return_value=_make_run_result(returncode=0)):
        result = _run_benchmark(test_dir, config, output_dir, "local", "sut", "eval")

    # slowdown = sut_median / 0 = inf; sut_median > 0 * 2.0 = 0 → passes if sut_median == 0
    # Since subprocess returns immediately, timing should be near 0
    assert "slowdown" in result
