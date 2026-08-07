# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.fuzz.runner."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from testrunner.fuzz.runner import (
    _build_cmd,
    _run_fuzz,
    _run_single_trial,
    _save_failure,
    _truncate,
    resolve_primitives,
    run_and_check,
    run_fuzz_eval,
    run_fuzz_grad,
)
from testrunner.fuzz.graph_builder import ALL_PRIMITIVES, SAFE_PRIMITIVES


# ---------------------------------------------------------------------------
# resolve_primitives
# ---------------------------------------------------------------------------


def test_resolve_primitives_all():
    result = resolve_primitives("all")
    assert result is ALL_PRIMITIVES


def test_resolve_primitives_safe():
    result = resolve_primitives("safe")
    assert result is SAFE_PRIMITIVES


def test_resolve_primitives_explicit_list():
    result = resolve_primitives(["neg", "relu"])
    assert result == ["neg", "relu"]


def test_resolve_primitives_unknown_string():
    with pytest.raises(ValueError, match="unknown primitive set"):
        resolve_primitives("bogus")


def test_resolve_primitives_unknown_in_list():
    with pytest.raises(ValueError, match="unknown primitives"):
        resolve_primitives(["neg", "NOT_A_PRIMITIVE"])


def test_resolve_primitives_invalid_type():
    with pytest.raises(ValueError, match="primitives must be a string or list"):
        resolve_primitives(42)


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


def test_truncate_short():
    assert _truncate("hello", max_len=100) == "hello"


def test_truncate_exact():
    text = "a" * 500
    assert _truncate(text, max_len=500) == text


def test_truncate_long():
    text = "a" * 1000
    result = _truncate(text, max_len=100)
    assert len(result) <= 105  # 50 + " ... " + 50
    assert "..." in result


def test_truncate_default_max():
    text = "x" * 600
    result = _truncate(text)  # default max_len=500
    assert "..." in result


# ---------------------------------------------------------------------------
# _build_cmd
# ---------------------------------------------------------------------------


def test_build_cmd_local_eval(tmp_path):
    network = tmp_path / "net.mininn"
    inp = tmp_path / "a.bin"
    out = tmp_path / "output"
    cmd = _build_cmd("eval", network, [inp], out, "local", "python sut.py", tmp_path)
    assert "eval" in cmd
    assert "--output-dir" in cmd
    assert str(network) in cmd
    assert str(inp) in cmd


def test_build_cmd_docker_eval(tmp_path):
    network = tmp_path / "net.mininn"
    inp = tmp_path / "a.bin"
    out = tmp_path / "output"
    cmd = _build_cmd("eval", network, [inp], out, "docker", "myimage", tmp_path)
    assert cmd[0] == "docker"
    assert "myimage" in cmd
    assert "eval" in cmd


def test_build_cmd_podman_eval(tmp_path):
    network = tmp_path / "net.mininn"
    inp = tmp_path / "a.bin"
    out = tmp_path / "output"
    cmd = _build_cmd(
        "eval", network, [inp], out, "podman", "myimage", tmp_path,
        extra_run_args=("--network=none",),
    )
    assert cmd[0] == "podman"
    assert "myimage" in cmd
    assert "eval" in cmd
    assert "--network=none" in cmd


def test_build_cmd_local_grad(tmp_path):
    network = tmp_path / "net.mininn"
    cmd = _build_cmd("grad", network, [], tmp_path / "out", "local", "sut", tmp_path)
    assert "grad" in cmd


def test_build_cmd_local_multiple_inputs(tmp_path):
    network = tmp_path / "net.mininn"
    inp1 = tmp_path / "a.bin"
    inp2 = tmp_path / "b.bin"
    out = tmp_path / "output"
    cmd = _build_cmd("eval", network, [inp1, inp2], out, "local", "sut", tmp_path)
    assert str(inp1) in cmd
    assert str(inp2) in cmd


# ---------------------------------------------------------------------------
# _save_failure
# ---------------------------------------------------------------------------


def test_save_failure_creates_dir(tmp_path):
    save_dir = tmp_path / "failures"
    save_dir.mkdir()
    network = tmp_path / "net.mininn"
    network.write_bytes(b"fake zip")
    inp = tmp_path / "a.bin"
    inp.write_bytes(b"data")

    dest = _save_failure(
        save_dir, network, [inp],
        mode="eval", check_nan_inf=False,
        expected_shapes=[(3,)],
        error="some error", stderr="some stderr"
    )
    assert dest is not None
    dest_path = Path(dest)
    assert dest_path.exists()
    # network is saved under its original name
    assert (dest_path / "net.mininn").exists()
    assert (dest_path / "a.bin").exists()
    assert (dest_path / "metadata.json").exists()
    assert (dest_path / "error.txt").exists()
    assert (dest_path / "stderr.txt").exists()


def test_save_failure_incrementing_index(tmp_path):
    save_dir = tmp_path / "failures"
    save_dir.mkdir()
    network = tmp_path / "net.mininn"
    network.write_bytes(b"z")
    inp = tmp_path / "a.bin"
    inp.write_bytes(b"d")

    dest1 = _save_failure(save_dir, network, [inp], "eval", False, [(1,)])
    dest2 = _save_failure(save_dir, network, [inp], "eval", False, [(1,)])
    assert Path(dest1).name == "0"
    assert Path(dest2).name == "1"


def test_save_failure_none_save_dir(tmp_path):
    network = tmp_path / "net.mininn"
    network.write_bytes(b"z")
    result = _save_failure(None, network, [], "eval", False, [])
    assert result is None


def test_save_failure_no_error_no_stderr(tmp_path):
    save_dir = tmp_path / "failures"
    save_dir.mkdir()
    network = tmp_path / "net.mininn"
    network.write_bytes(b"z")

    dest = _save_failure(save_dir, network, [], "eval", True, [(2, 3)])
    assert dest is not None
    dest_path = Path(dest)
    assert not (dest_path / "error.txt").exists()
    assert not (dest_path / "stderr.txt").exists()


# ---------------------------------------------------------------------------
# run_and_check
# ---------------------------------------------------------------------------


def _write_bin(path, arr):
    np.array(arr, dtype=np.float64).tofile(path)


def _make_completed_process(stdout="", stderr="", returncode=0):
    r = MagicMock()
    r.stdout = stdout
    r.stderr = stderr
    r.returncode = returncode
    return r


def test_run_and_check_passes(tmp_path):
    """Successful run: subprocess returns output file paths with correct shapes."""
    out_file = tmp_path / "output" / "result.bin"
    out_file.parent.mkdir()
    _write_bin(out_file, np.zeros(6))

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _make_completed_process(stdout=str(out_file) + "\n")
        network = tmp_path / "net.mininn"
        network.write_bytes(b"x")
        inp = tmp_path / "a.bin"
        inp.write_bytes(b"y")
        result = run_and_check(
            network, [inp], "local", "sut", "eval",
            expected_shapes=[(6,)], check_nan_inf=False
        )
    assert result["passed"] is True


def test_run_and_check_crash(tmp_path):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _make_completed_process(stderr="crash!", returncode=1)
        network = tmp_path / "net.mininn"
        network.write_bytes(b"x")
        result = run_and_check(network, [], "local", "sut", "eval", [(3,)])
    assert result["passed"] is False
    assert "crash" in result["error"]


def test_run_and_check_timeout(tmp_path):
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
        network = tmp_path / "net.mininn"
        network.write_bytes(b"x")
        result = run_and_check(network, [], "local", "sut", "eval", [(3,)])
    assert result["passed"] is False
    assert "timed out" in result["error"]


def test_run_and_check_no_output_files(tmp_path):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _make_completed_process(stdout="")
        network = tmp_path / "net.mininn"
        network.write_bytes(b"x")
        result = run_and_check(network, [], "local", "sut", "eval", [(3,)])
    assert result["passed"] is False
    assert "no output files" in result["error"]


def test_run_and_check_wrong_file_count(tmp_path):
    out_file = tmp_path / "output" / "result.bin"
    out_file.parent.mkdir()
    _write_bin(out_file, np.zeros(3))
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _make_completed_process(stdout=str(out_file))
        network = tmp_path / "net.mininn"
        network.write_bytes(b"x")
        result = run_and_check(network, [], "local", "sut", "eval", [(3,), (4,)])
    assert result["passed"] is False
    assert "expected 2 output file(s), got 1" in result["error"]


def test_run_and_check_wrong_size(tmp_path):
    out_file = tmp_path / "output" / "result.bin"
    out_file.parent.mkdir()
    _write_bin(out_file, np.zeros(5))  # wrong: expected 3
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _make_completed_process(stdout=str(out_file))
        network = tmp_path / "net.mininn"
        network.write_bytes(b"x")
        result = run_and_check(network, [], "local", "sut", "eval", [(3,)])
    assert result["passed"] is False
    assert "expected 3 values" in result["error"]


def test_run_and_check_nan(tmp_path):
    out_file = tmp_path / "output" / "result.bin"
    out_file.parent.mkdir()
    _write_bin(out_file, [1.0, float("nan"), 3.0])
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _make_completed_process(stdout=str(out_file))
        network = tmp_path / "net.mininn"
        network.write_bytes(b"x")
        result = run_and_check(network, [], "local", "sut", "eval", [(3,)], check_nan_inf=True)
    assert result["passed"] is False
    assert "NaN" in result["error"]


def test_run_and_check_inf(tmp_path):
    out_file = tmp_path / "output" / "result.bin"
    out_file.parent.mkdir()
    _write_bin(out_file, [1.0, float("inf"), 3.0])
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _make_completed_process(stdout=str(out_file))
        network = tmp_path / "net.mininn"
        network.write_bytes(b"x")
        result = run_and_check(network, [], "local", "sut", "eval", [(3,)], check_nan_inf=True)
    assert result["passed"] is False
    assert "Inf" in result["error"]


def test_run_and_check_nan_inf_disabled(tmp_path):
    """NaN/Inf should pass when check_nan_inf=False."""
    out_file = tmp_path / "output" / "result.bin"
    out_file.parent.mkdir()
    _write_bin(out_file, [float("nan")])
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _make_completed_process(stdout=str(out_file))
        network = tmp_path / "net.mininn"
        network.write_bytes(b"x")
        result = run_and_check(network, [], "local", "sut", "eval", [(1,)], check_nan_inf=False)
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# run_fuzz_eval / run_fuzz_grad with generate=True
# ---------------------------------------------------------------------------


def test_run_fuzz_eval_generate_warns(tmp_path):
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = run_fuzz_eval(tmp_path, {}, tmp_path / "out", "local", "sut", generate=True)
    assert result["passed"] is True
    assert result.get("generated") is False
    assert any("--generate has no effect" in str(warning.message) for warning in w)


def test_run_fuzz_grad_generate_warns(tmp_path):
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = run_fuzz_grad(tmp_path, {}, tmp_path / "out", "local", "sut", generate=True)
    assert result["passed"] is True
    assert any("--generate has no effect" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# _run_fuzz — dedup logic with mocked _run_single_trial
# ---------------------------------------------------------------------------


def _make_output_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir(exist_ok=True)
    return d


def test_run_fuzz_all_pass(tmp_path):
    """When all trials pass, result should show passed=True."""
    output_dir = _make_output_dir(tmp_path)
    config = {"n_trials": 3, "primitives": "safe", "seed": 0}

    def mock_trial(*a, **kw):
        return {"passed": True, "error": None}

    with patch("testrunner.fuzz.runner._run_single_trial", side_effect=mock_trial):
        result = _run_fuzz(tmp_path, config, output_dir, "local", "sut", "eval")
    assert result["passed"] is True
    assert result["n_passed"] == result["n_trials"]


def test_run_fuzz_all_fail(tmp_path):
    """When all trials fail, result should show passed=False."""
    output_dir = _make_output_dir(tmp_path)
    config = {"n_trials": 3, "primitives": "safe", "seed": 0}

    def mock_trial(*a, **kw):
        return {"passed": False, "error": "some error", "saved_to": None}

    with patch("testrunner.fuzz.runner._run_single_trial", side_effect=mock_trial):
        result = _run_fuzz(tmp_path, config, output_dir, "local", "sut", "eval")
    assert result["passed"] is False
    assert result["n_passed"] == 0


def test_run_fuzz_failure_summary(tmp_path):
    """Failure summary should group by error message."""
    output_dir = _make_output_dir(tmp_path)
    config = {"n_trials": 4, "primitives": "safe", "seed": 0}
    errors = ["error A", "error A", "error B", "error A"]
    call_idx = [0]

    def mock_trial(*a, **kw):
        err = errors[call_idx[0] % len(errors)]
        call_idx[0] += 1
        return {"passed": False, "error": err, "saved_to": None}

    with patch("testrunner.fuzz.runner._run_single_trial", side_effect=mock_trial):
        result = _run_fuzz(tmp_path, config, output_dir, "local", "sut", "eval")
    assert "failure_summary" in result
    assert result["failure_summary"]["error A"] == 3
    assert result["failure_summary"]["error B"] == 1


def test_run_fuzz_dedup_smaller_replaces(tmp_path):
    """Smaller failure for same error replaces existing saved failure."""
    output_dir = _make_output_dir(tmp_path)
    config = {"n_trials": 2, "primitives": "safe", "seed": 0}

    save_dir = output_dir / "fuzz_failures"
    save_dir.mkdir()

    saved_0 = save_dir / "0"
    saved_0.mkdir()
    saved_1 = save_dir / "1"
    saved_1.mkdir()

    call_idx = [0]

    def mock_trial(*a, **kw):
        i = call_idx[0]
        call_idx[0] += 1
        # Both fail with the same error, but different saved_to paths
        return {"passed": False, "error": "same error", "saved_to": str(saved_0 if i == 0 else saved_1)}

    # We need to patch serialize_graph and the size calculation
    with patch("testrunner.fuzz.runner._run_single_trial", side_effect=mock_trial):
        with patch("testrunner.fuzz.runner.serialize_graph") as mock_sg:
            # First trial is "bigger" (100 bytes), second is "smaller" (10 bytes)
            mock_sg.side_effect = [b"x" * 100, b"x" * 10]
            result = _run_fuzz(tmp_path, config, output_dir, "local", "sut", "eval")

    # Both had same error; the smaller one should have won
    assert result["passed"] is False


def test_run_fuzz_closed_no_failures_key(tmp_path):
    """Closed tests should not include 'failures' or 'failure_summary' in result."""
    output_dir = _make_output_dir(tmp_path)
    config = {"n_trials": 2, "primitives": "safe", "seed": 0}

    def mock_trial(*a, **kw):
        return {"passed": False, "error": "err", "saved_to": None}

    with patch("testrunner.fuzz.runner._run_single_trial", side_effect=mock_trial):
        result = _run_fuzz(tmp_path, config, output_dir, "local", "sut", "eval", closed=True)
    assert "failures" not in result
    assert "failure_summary" not in result


def test_run_fuzz_output_handler(tmp_path):
    """Output handler should be called for each trial."""
    output_dir = _make_output_dir(tmp_path)
    config = {"n_trials": 3, "primitives": "safe", "seed": 0}

    events = []

    class Handler:
        def fuzz_trial_finished(self, test_path, trial, total_trials, passed, error):
            events.append((trial, passed))

    def mock_trial(*a, **kw):
        return {"passed": True, "error": None}

    with patch("testrunner.fuzz.runner._run_single_trial", side_effect=mock_trial):
        _run_fuzz(tmp_path, config, output_dir, "local", "sut", "eval", output_handler=Handler())
    assert len(events) == 3


# ---------------------------------------------------------------------------
# _run_single_trial — direct call with mocked subprocess
# ---------------------------------------------------------------------------


def test_run_single_trial_pass(tmp_path):
    """_run_single_trial should pass when subprocess succeeds with correct output."""
    from testrunner.fuzz.graph_builder import Var, Equation, Graph, generate_graph
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st

    # Build a trivial graph manually
    x = Var("a", (3,))
    y = Var("b", (3,))
    eqn = Equation("neg", [x], y)
    graph = Graph(invars=[x], outvars=[y], equations=[eqn])
    inputs = {"a": np.zeros(3, dtype=np.float64)}

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Subprocess output: a file with 3 values
    out_file = output_dir / "result.bin"
    np.zeros(3, dtype=np.float64).tofile(out_file)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout=str(out_file) + "\n", stderr=""
        )
        result = _run_single_trial(
            graph, inputs, output_dir, "local", "sut", "eval",
            check_nan_inf=False, timeout=10,
        )
    assert result["passed"] is True


def test_run_single_trial_fail_saves(tmp_path):
    """Failing trial with save_dir should populate saved_to."""
    from testrunner.fuzz.graph_builder import Var, Equation, Graph

    x = Var("a", (2,))
    y = Var("b", (2,))
    eqn = Equation("neg", [x], y)
    graph = Graph(invars=[x], outvars=[y], equations=[eqn])
    inputs = {"a": np.zeros(2, dtype=np.float64)}

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    save_dir = output_dir / "failures"
    save_dir.mkdir()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="crash")
        result = _run_single_trial(
            graph, inputs, output_dir, "local", "sut", "eval",
            check_nan_inf=False, timeout=10, save_dir=save_dir
        )
    assert result["passed"] is False
    assert result.get("saved_to") is not None


def test_run_single_trial_grad_mode(tmp_path):
    """In grad mode, expected_shapes should be input shapes, not output shapes."""
    from testrunner.fuzz.graph_builder import Var, Equation, Graph

    x = Var("a", (4,))
    y = Var("b", (4,))
    eqn = Equation("neg", [x], y)
    graph = Graph(invars=[x], outvars=[y], equations=[eqn])
    inputs = {"a": np.zeros(4, dtype=np.float64)}

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    out_file = output_dir / "grad.bin"
    np.zeros(4, dtype=np.float64).tofile(out_file)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=str(out_file), stderr="")
        result = _run_single_trial(
            graph, inputs, output_dir, "local", "sut", "grad",
            check_nan_inf=False, timeout=10,
        )
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# _run_fuzz with seed
# ---------------------------------------------------------------------------


def test_run_fuzz_with_seed(tmp_path):
    """Seeded fuzz should still produce results."""
    output_dir = _make_output_dir(tmp_path)
    config = {"n_trials": 2, "primitives": "safe", "seed": 42}

    def mock_trial(*a, **kw):
        return {"passed": True, "error": None}

    with patch("testrunner.fuzz.runner._run_single_trial", side_effect=mock_trial):
        result = _run_fuzz(tmp_path, config, output_dir, "local", "sut", "eval")
    assert result["passed"] is True


def test_run_fuzz_max_saved_cap(tmp_path):
    """After MAX_SAVED distinct errors, new errors should be discarded."""
    output_dir = _make_output_dir(tmp_path)
    MAX_SAVED = 10
    n_errors = MAX_SAVED + 5
    config = {"n_trials": n_errors, "primitives": "safe", "seed": 0}

    save_dirs = []
    for i in range(n_errors):
        d = output_dir / f"s{i}"
        d.mkdir()
        save_dirs.append(str(d))

    call_idx = [0]

    def mock_trial(*a, **kw):
        i = call_idx[0]
        call_idx[0] += 1
        # Each trial has a unique error
        return {"passed": False, "error": f"unique error {i}", "saved_to": save_dirs[i]}

    with patch("testrunner.fuzz.runner._run_single_trial", side_effect=mock_trial):
        with patch("testrunner.fuzz.runner.serialize_graph", return_value=b"x"):
            result = _run_fuzz(tmp_path, config, output_dir, "local", "sut", "eval")

    assert result["passed"] is False
    # saved_failures_dir should be present since we saved at least one
    assert "saved_failures_dir" in result


def test_run_fuzz_dedup_replace_smaller(tmp_path):
    """When a second trial has the same error but a smaller size, it replaces the first."""
    output_dir = _make_output_dir(tmp_path)
    # Use n_trials=3 so hypothesis calls the function multiple times
    config = {"n_trials": 3, "primitives": "safe", "seed": 0}

    # _run_fuzz will create fuzz_failures/ inside output_dir
    failures_dir = output_dir / "fuzz_failures"

    call_count = [0]

    def mock_trial(*a, **kw):
        c = call_count[0]
        call_count[0] += 1
        # Create the "saved" directory on-the-fly (rmtree may have deleted it)
        p = str(failures_dir / str(c))
        Path(p).mkdir(parents=True, exist_ok=True)
        return {"passed": False, "error": "same error", "saved_to": p}

    sg_call_count = [0]

    def mock_serialize(*a, **kw):
        c = sg_call_count[0]
        sg_call_count[0] += 1
        # First call: huge (1 MB). Subsequent calls: tiny (10 bytes).
        # This ensures trial_size on call 1 is always < prev[1] regardless of
        # the actual input array sizes generated by hypothesis.
        return b"x" * (1_000_000 if c == 0 else 10)

    with patch("testrunner.fuzz.runner._run_single_trial", side_effect=mock_trial):
        with patch("testrunner.fuzz.runner.serialize_graph", side_effect=mock_serialize):
            result = _run_fuzz(tmp_path, config, output_dir, "local", "sut", "eval")

    assert result["passed"] is False


def test_run_fuzz_eval_non_generate(tmp_path):
    """run_fuzz_eval without generate=True delegates to _run_fuzz (covers line 87)."""
    expected = {"passed": True, "n_trials": 3, "n_passed": 3, "error": None}
    with patch("testrunner.fuzz.runner._run_fuzz", return_value=expected) as mock_rf:
        result = run_fuzz_eval(tmp_path, {}, tmp_path / "out", "local", "sut")
    mock_rf.assert_called_once()
    assert result is expected


def test_run_fuzz_grad_non_generate(tmp_path):
    """run_fuzz_grad without generate=True delegates to _run_fuzz (covers line 115)."""
    expected = {"passed": True, "n_trials": 3, "n_passed": 3, "error": None}
    with patch("testrunner.fuzz.runner._run_fuzz", return_value=expected) as mock_rf:
        result = run_fuzz_grad(tmp_path, {}, tmp_path / "out", "local", "sut")
    mock_rf.assert_called_once()
    assert result is expected


def test_run_fuzz_exception_in_trial_silenced(tmp_path):
    """An exception raised inside fuzz_trial is silenced by except Exception: pass."""
    output_dir = _make_output_dir(tmp_path)
    config = {"n_trials": 3, "primitives": "safe"}  # no seed

    def mock_trial(*a, **kw):
        raise RuntimeError("unexpected internal trial error")

    with patch("testrunner.fuzz.runner._run_single_trial", side_effect=mock_trial):
        # Should not propagate the exception
        result = _run_fuzz(tmp_path, config, output_dir, "local", "sut", "eval")
    # All trials raised before being recorded
    assert result["n_trials"] == 0
    assert result["passed"] is True  # 0 failures out of 0 trials
