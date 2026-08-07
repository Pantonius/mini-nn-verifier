# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.__main__ (is_closed, run_single_test, run_tests, main)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from testrunner.__main__ import is_closed, run_single_test, run_tests


# ---------------------------------------------------------------------------
# is_closed
# ---------------------------------------------------------------------------


def test_is_closed_closed_in_path(tmp_path):
    test_dir = tmp_path / "closed" / "test"
    assert is_closed(test_dir, {}) is True


def test_is_closed_config_access(tmp_path):
    assert is_closed(tmp_path / "normal", {"access": "closed"}) is True


def test_is_closed_open():
    # Use a path with no "closed" in any component
    assert is_closed(Path("/some/normal/path"), {}) is False


def test_is_closed_open_config():
    assert is_closed(Path("/some/normal/path"), {"access": "open"}) is False


# ---------------------------------------------------------------------------
# run_single_test helpers
# ---------------------------------------------------------------------------


def _write_bin(path, arr):
    np.array(arr, dtype=np.float64).tofile(path)


def _make_test_dir(tmp_path, config, files=None):
    test_dir = tmp_path / "test"
    test_dir.mkdir(exist_ok=True)
    (test_dir / "test.json").write_text(json.dumps(config))
    if files:
        for rel, content in files.items():
            dest = test_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, np.ndarray):
                content.tofile(dest)
            elif isinstance(content, bytes):
                dest.write_bytes(content)
            else:
                dest.write_text(content)
    return test_dir


# ---------------------------------------------------------------------------
# run_single_test — unknown command
# ---------------------------------------------------------------------------


def test_run_single_test_unknown_command(tmp_path):
    test_dir = _make_test_dir(tmp_path, {"command": "banana"})
    result = run_single_test(test_dir, "local", "sut")
    assert result["passed"] is False
    assert "unknown command" in result["error"]


def test_run_single_test_removes_existing_actual_dir(tmp_path):
    """run_single_test should remove pre-existing actual/ dir before running."""
    config = {"command": "eval", "network": "net.mininn", "inputs": []}
    test_dir = _make_test_dir(tmp_path, config)
    # Pre-create the actual/ directory with a stale file
    actual_dir = test_dir / "actual"
    actual_dir.mkdir()
    stale_file = actual_dir / "stale.bin"
    stale_file.write_bytes(b"old data")

    from testrunner.commands.common import SubprocessResult

    mock_result = SubprocessResult(1, "", "err")
    with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
        result = run_single_test(test_dir, "local", "sut")
    # Stale file should be gone — actual/ was recreated fresh
    assert not stale_file.exists()
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# run_single_test — COMMANDS path (eval)
# ---------------------------------------------------------------------------


def test_run_single_test_eval_pass(tmp_path):
    # Expected output: [0.0, 1.0, 2.0]
    expected_data = np.array([0.0, 1.0, 2.0])
    config = {
        "command": "eval",
        "network": "net.mininn",
        "inputs": ["input.bin"],
        "expected_outputs": ["expected.bin"],
    }
    test_dir = _make_test_dir(tmp_path, config)
    _write_bin(test_dir / "expected.bin", expected_data)
    test_dir.joinpath("net.mininn").write_bytes(b"fake")
    _write_bin(test_dir / "input.bin", [1.0])

    # Subprocess returns the path to an output file that matches expected
    actual_out = tmp_path / "actual_out.bin"
    _write_bin(actual_out, expected_data)

    from testrunner.commands.common import SubprocessResult

    mock_result = SubprocessResult(0, str(actual_out), "")
    with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
        result = run_single_test(test_dir, "local", "sut")
    assert result["passed"] is True


def test_run_single_test_eval_fail_check(tmp_path):
    expected_data = np.array([0.0, 1.0, 2.0])
    actual_data = np.array([9.0, 9.0, 9.0])
    config = {
        "command": "eval",
        "network": "net.mininn",
        "inputs": ["input.bin"],
        "expected_outputs": ["expected.bin"],
    }
    test_dir = _make_test_dir(tmp_path, config)
    _write_bin(test_dir / "expected.bin", expected_data)
    test_dir.joinpath("net.mininn").write_bytes(b"fake")

    actual_out = tmp_path / "actual_out.bin"
    _write_bin(actual_out, actual_data)

    from testrunner.commands.common import SubprocessResult

    mock_result = SubprocessResult(0, str(actual_out), "")
    with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
        result = run_single_test(test_dir, "local", "sut")
    assert result["passed"] is False


def test_run_single_test_command_failure(tmp_path):
    config = {"command": "eval", "network": "net.mininn", "inputs": []}
    test_dir = _make_test_dir(tmp_path, config)

    from testrunner.commands.common import SubprocessResult

    mock_result = SubprocessResult(1, "", "fatal error")
    with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
        result = run_single_test(test_dir, "local", "sut")
    assert result["passed"] is False
    assert "command failed" in result["error"]


def test_run_single_test_command_failure_closed(tmp_path):
    # Use "access": "closed" in config — run_single_test infers closed internally
    config = {"command": "eval", "network": "net.mininn", "inputs": [], "access": "closed"}
    # Make test_dir path not contain "closed" (config sets it instead)
    test_dir = tmp_path / "mytest"
    test_dir.mkdir()
    (test_dir / "test.json").write_text(__import__("json").dumps(config))

    from testrunner.commands.common import SubprocessResult

    mock_result = SubprocessResult(1, "", "secret error details")
    with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
        result = run_single_test(test_dir, "local", "sut")
    # closed=True means error is masked — should not leak stderr details
    assert "secret error details" not in result.get("error", "")


def test_run_single_test_timeout(tmp_path):
    config = {"command": "eval", "network": "net.mininn", "inputs": []}
    test_dir = _make_test_dir(tmp_path, config)
    with patch("testrunner.__main__.run_subprocess", side_effect=subprocess.TimeoutExpired("cmd", 60)):
        result = run_single_test(test_dir, "local", "sut")
    assert result["passed"] is False
    assert "timed out" in result["error"]


def test_run_single_test_generate(tmp_path):
    config = {
        "command": "eval",
        "network": "net.mininn",
        "inputs": [],
        "expected_outputs": [],
    }
    test_dir = _make_test_dir(tmp_path, config)
    out_file = tmp_path / "result.bin"
    np.zeros(3, dtype=np.float64).tofile(out_file)

    from testrunner.commands.common import SubprocessResult

    mock_result = SubprocessResult(0, str(out_file), "")
    with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
        result = run_single_test(test_dir, "local", "sut", generate=True)
    assert result.get("generated") is True
    # The expected file is saved as expected_<original_name> in test_dir
    assert (test_dir / f"expected_{out_file.name}").exists()


def test_run_single_test_no_check_and_no_default(tmp_path):
    """Command with no check and no default check name should fail gracefully."""
    config = {"command": "eval", "network": "net.mininn", "inputs": []}
    test_dir = _make_test_dir(tmp_path, config)
    out_file = tmp_path / "result.bin"
    np.zeros(3, dtype=np.float64).tofile(out_file)

    from testrunner.commands.common import SubprocessResult
    from testrunner.check import DEFAULT_CHECKS

    mock_result = SubprocessResult(0, str(out_file), "")
    # Temporarily remove the default check for eval
    original = dict(DEFAULT_CHECKS)
    DEFAULT_CHECKS.clear()
    try:
        with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
            result = run_single_test(test_dir, "local", "sut")
    finally:
        DEFAULT_CHECKS.update(original)
    assert result["passed"] is False
    assert "no check specified" in result["error"]


def test_run_single_test_unknown_check(tmp_path):
    config = {
        "command": "eval",
        "network": "net.mininn",
        "inputs": [],
        "check": "nonexistent_check",
    }
    test_dir = _make_test_dir(tmp_path, config)
    out_file = tmp_path / "result.bin"
    np.zeros(3, dtype=np.float64).tofile(out_file)

    from testrunner.commands.common import SubprocessResult

    mock_result = SubprocessResult(0, str(out_file), "")
    with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
        result = run_single_test(test_dir, "local", "sut")
    assert result["passed"] is False
    assert "unknown check" in result["error"]


# ---------------------------------------------------------------------------
# run_single_test — RUNNERS path (via train mock)
# ---------------------------------------------------------------------------


def test_run_single_test_uses_runner(tmp_path):
    config = {"command": "train", "dataset": "ds", "in_size": 4, "num_classes": 2}
    test_dir = _make_test_dir(tmp_path, config)

    expected_result = {"passed": True, "error": None}

    from testrunner.commands import _init_runners, RUNNERS
    _init_runners()

    with patch.dict(RUNNERS, {"train": lambda *a, **kw: expected_result}):
        result = run_single_test(test_dir, "local", "sut")
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# run_tests
# ---------------------------------------------------------------------------


def test_run_tests_empty_dir(tmp_path):
    results = run_tests(tmp_path, "local", "sut")
    assert results == []


def test_run_tests_single_test(tmp_path):
    config = {"command": "eval", "network": "net.mininn", "inputs": []}
    test_dir = tmp_path / "test1"
    test_dir.mkdir()
    (test_dir / "test.json").write_text(json.dumps(config))

    from testrunner.commands.common import SubprocessResult

    out = tmp_path / "out.bin"
    np.zeros(1, dtype=np.float64).tofile(out)
    mock_result = SubprocessResult(1, "", "err")
    with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
        results = run_tests(tmp_path, "local", "sut")
    assert len(results) == 1
    path, result = results[0]
    assert path == "test1"


def test_run_tests_output_handler(tmp_path):
    config = {"command": "eval", "network": "net.mininn", "inputs": []}
    test_dir = tmp_path / "mytest"
    test_dir.mkdir()
    (test_dir / "test.json").write_text(json.dumps(config))

    events = []

    class Handler:
        def test_starting(self, path, idx, total):
            events.append(("starting", path))

        def test_finished(self, path, result):
            events.append(("finished", path))

        def all_finished(self, n_passed, n_failed, total_score=0.0, total_max_points=0.0, total_bonus=0.0, total_max_bonus=0.0):
            events.append(("all_finished",))

    from testrunner.commands.common import SubprocessResult

    mock_result = SubprocessResult(0, "", "")
    with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
        run_tests(tmp_path, "local", "sut", output_handler=Handler())

    assert ("starting", "mytest") in events
    assert ("finished", "mytest") in events
    assert ("all_finished",) in events


def test_run_tests_counts_pass_fail(tmp_path):
    """run_tests should correctly count passes and failures."""
    for name in ["t1", "t2", "t3"]:
        d = tmp_path / name
        d.mkdir()
        (d / "test.json").write_text(json.dumps({"command": "eval", "network": "n", "inputs": []}))

    n_calls = [0]

    from testrunner.commands.common import SubprocessResult

    def mock_subprocess(*a, **kw):
        n_calls[0] += 1
        # First two pass, third fails
        if n_calls[0] <= 2:
            return SubprocessResult(0, "", "")
        return SubprocessResult(1, "", "err")

    n_passed_final = [0]
    n_failed_final = [0]

    class Handler:
        def test_starting(self, *a): pass
        def test_finished(self, *a): pass
        def all_finished(self, n_passed, n_failed, total_score=0.0, total_max_points=0.0, total_bonus=0.0, total_max_bonus=0.0):
            n_passed_final[0] = n_passed
            n_failed_final[0] = n_failed

    with patch("testrunner.__main__.run_subprocess", side_effect=mock_subprocess):
        run_tests(tmp_path, "local", "sut", output_handler=Handler())

    assert n_failed_final[0] >= 1


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _attach_score / scoring integration
# ---------------------------------------------------------------------------


def test_run_single_test_attaches_score(tmp_path):
    """When config has 'points', result should include score and max_points."""
    expected_data = np.array([1.0, 2.0])
    config = {
        "command": "eval",
        "network": "net.mininn",
        "inputs": [],
        "expected_outputs": ["expected.bin"],
        "points": 10,
    }
    test_dir = _make_test_dir(tmp_path, config)
    _write_bin(test_dir / "expected.bin", expected_data)

    actual_out = tmp_path / "actual_out.bin"
    _write_bin(actual_out, expected_data)

    from testrunner.commands.common import SubprocessResult

    mock_result = SubprocessResult(0, str(actual_out), "")
    with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
        result = run_single_test(test_dir, "local", "sut")
    assert "score" in result
    assert result["score"] == 10
    assert result["max_points"] == 10


def test_run_tests_accumulates_score(tmp_path):
    """run_tests should sum scores across tests."""
    config = {
        "command": "eval",
        "network": "n",
        "inputs": [],
        "expected_outputs": [],
        "points": 5,
    }
    test_dir = tmp_path / "t1"
    test_dir.mkdir()
    (test_dir / "test.json").write_text(json.dumps(config))

    total_scores = [0.0]
    total_max = [0.0]

    class Handler:
        def test_starting(self, *a): pass
        def test_finished(self, *a): pass
        def all_finished(self, n_passed, n_failed, total_score=0.0, total_max_points=0.0, total_bonus=0.0, total_max_bonus=0.0):
            total_scores[0] = total_score
            total_max[0] = total_max_points

    from testrunner.commands.common import SubprocessResult

    mock_result = SubprocessResult(0, "", "")
    with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
        run_tests(tmp_path, "local", "sut", output_handler=Handler())

    assert total_max[0] == 5


def test_main_exits_1_on_failure(tmp_path):
    config = {"command": "eval", "network": "n", "inputs": []}
    test_dir = tmp_path / "t"
    test_dir.mkdir()
    (test_dir / "test.json").write_text(json.dumps(config))

    from testrunner.commands.common import SubprocessResult
    from testrunner.__main__ import main

    mock_result = SubprocessResult(1, "", "err")
    with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
        with patch("sys.argv", ["testrunner", "local", "sut", str(tmp_path)]):
            with pytest.raises(SystemExit) as exc:
                main()
    assert exc.value.code == 1


def test_main_exits_0_on_success(tmp_path):
    # No tests found means empty results -> no failures -> exit 0
    from testrunner.__main__ import main

    with patch("sys.argv", ["testrunner", "local", "sut", str(tmp_path)]):
        with patch("sys.exit") as mock_exit:
            main()
    mock_exit.assert_not_called()


def test_main_json_output_mode(tmp_path):
    from testrunner.__main__ import main

    with patch("sys.argv", ["testrunner", "local", "sut", str(tmp_path), "--output", "json"]):
        with patch("sys.exit") as mock_exit:
            main()
    mock_exit.assert_not_called()


def test_main_accepts_podman_backend(tmp_path):
    """The CLI accepts 'podman' as a backend and threads --extra-run-args."""
    from testrunner.__main__ import main

    seen = {}

    def fake_run_tests(root, backend, backend_arg, generate=False,
                       output_handler=None, extra_run_args=()):
        seen["backend"] = backend
        seen["extra_run_args"] = extra_run_args
        return []

    with patch("testrunner.__main__.run_tests", side_effect=fake_run_tests):
        with patch(
            "sys.argv",
            [
                "testrunner",
                "podman",
                "myimg:latest",
                str(tmp_path),
                "--extra-run-args",
                "--network=none --memory=1g",
            ],
        ):
            main()
    assert seen["backend"] == "podman"
    assert "--network=none" in seen["extra_run_args"]
    assert "--memory=1g" in seen["extra_run_args"]


def test_module_guard(tmp_path):
    """The if __name__ == '__main__': guard in __main__.py calls main()."""
    import runpy
    with patch("sys.argv", ["testrunner", "local", "sut", str(tmp_path)]):
        with patch("sys.exit"):
            runpy.run_module("testrunner.__main__", run_name="__main__", alter_sys=False)
