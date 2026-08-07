# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.commands.common."""

import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from testrunner.commands.common import (
    DEFAULT_TIMEOUTS,
    SubprocessResult,
    build_eval_grad_cmd,
    container_run_prefix,
    get_timeout,
    is_container_backend,
    parse_output_paths,
    run_subprocess,
)


# ---------------------------------------------------------------------------
# get_timeout
# ---------------------------------------------------------------------------


def test_get_timeout_explicit():
    assert get_timeout({"timeout": 30}) == 30


def test_get_timeout_default_eval():
    assert get_timeout({"command": "eval"}) == DEFAULT_TIMEOUTS["eval"]


def test_get_timeout_default_grad():
    assert get_timeout({"command": "grad"}) == DEFAULT_TIMEOUTS["grad"]


def test_get_timeout_default_train():
    assert get_timeout({"command": "train"}) == DEFAULT_TIMEOUTS["train"]


def test_get_timeout_default_fuzz_eval():
    assert get_timeout({"command": "fuzz_eval"}) == DEFAULT_TIMEOUTS["fuzz_eval"]


def test_get_timeout_default_fuzz_grad():
    assert get_timeout({"command": "fuzz_grad"}) == DEFAULT_TIMEOUTS["fuzz_grad"]


def test_get_timeout_unknown_command():
    assert get_timeout({"command": "unknown"}) == 60


def test_get_timeout_no_command():
    assert get_timeout({}) == 60


# ---------------------------------------------------------------------------
# parse_output_paths
# ---------------------------------------------------------------------------


def test_parse_output_paths_existing(tmp_path):
    p = tmp_path / "out.bin"
    p.write_bytes(b"data")
    paths, warnings = parse_output_paths(str(p))
    assert paths == [p]
    assert warnings == []


def test_parse_output_paths_nonexistent(tmp_path):
    missing = tmp_path / "missing.bin"
    paths, warnings = parse_output_paths(str(missing))
    assert paths == []
    assert len(warnings) == 1
    assert "ignoring non-existent" in warnings[0]


def test_parse_output_paths_empty_string():
    paths, warnings = parse_output_paths("")
    assert paths == []
    assert warnings == []


def test_parse_output_paths_blank_lines(tmp_path):
    p = tmp_path / "out.bin"
    p.write_bytes(b"")
    stdout = f"\n  \n{p}\n\n"
    paths, warnings = parse_output_paths(stdout)
    assert paths == [p]
    assert warnings == []


def test_parse_output_paths_blank_middle_lines(tmp_path):
    """Blank lines in the MIDDLE of stdout are skipped via `continue`."""
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    p1.write_bytes(b"x")
    p2.write_bytes(b"y")
    # Two real paths with a blank line between them
    stdout = f"{p1}\n\n{p2}"
    paths, warnings = parse_output_paths(stdout)
    assert paths == [p1, p2]
    assert warnings == []


def test_parse_output_paths_mixed(tmp_path):
    p = tmp_path / "real.bin"
    p.write_bytes(b"x")
    stdout = f"{p}\n/totally/nonexistent/path.bin"
    paths, warnings = parse_output_paths(stdout)
    assert paths == [p]
    assert len(warnings) == 1


def test_parse_output_paths_container_root_rewrite(tmp_path):
    """In-container /data paths are rewritten to the host bind-mount source."""
    (tmp_path / "actual").mkdir()
    out = tmp_path / "actual" / "output_0.bin"
    out.write_bytes(b"x")
    stdout = "/data/actual/output_0.bin\n"
    paths, warnings = parse_output_paths(stdout, container_root=tmp_path)
    assert paths == [out]
    assert warnings == []


def test_parse_output_paths_container_root_bare_data(tmp_path):
    """Exact '/data' maps to the container_root itself."""
    paths, warnings = parse_output_paths("/data", container_root=tmp_path)
    assert paths == [tmp_path]
    assert warnings == []


def test_parse_output_paths_container_root_missing_file(tmp_path):
    """Rewritten paths that don't exist on host still become warnings."""
    stdout = "/data/actual/missing.bin"
    paths, warnings = parse_output_paths(stdout, container_root=tmp_path)
    assert paths == []
    assert len(warnings) == 1
    # The warning quotes the original (in-container) path.
    assert "/data/actual/missing.bin" in warnings[0]


# ---------------------------------------------------------------------------
# SubprocessResult
# ---------------------------------------------------------------------------


def test_subprocess_result_attributes():
    r = SubprocessResult(0, "out", "err")
    assert r.returncode == 0
    assert r.stdout == "out"
    assert r.stderr == "err"


# ---------------------------------------------------------------------------
# run_subprocess
# ---------------------------------------------------------------------------


def _make_popen_mock(stdout_lines, stderr_lines, returncode=0):
    """Build a mock Popen object."""
    mock = MagicMock()
    mock.stdout = iter(stdout_lines)
    mock.stderr = iter(stderr_lines)
    mock.returncode = returncode
    mock.wait.return_value = None
    mock.kill.return_value = None
    return mock


def test_run_subprocess_success():
    with patch("subprocess.Popen") as popen:
        popen.return_value = _make_popen_mock(["hello\n", "world\n"], [""], 0)
        result = run_subprocess(["echo", "hello world"], timeout=5)
    assert result.returncode == 0
    assert "hello" in result.stdout
    assert "world" in result.stdout


def test_run_subprocess_nonzero_return():
    with patch("subprocess.Popen") as popen:
        popen.return_value = _make_popen_mock([], ["error msg\n"], 1)
        result = run_subprocess(["false"], timeout=5)
    assert result.returncode == 1
    assert "error msg" in result.stderr


def test_run_subprocess_writes_log_file(tmp_path):
    log = tmp_path / "stdout.log"
    with patch("subprocess.Popen") as popen:
        popen.return_value = _make_popen_mock(["line1\n"], [], 0)
        run_subprocess(["echo"], timeout=5, log_file=log)
    assert log.exists()
    assert "line1" in log.read_text()


def test_run_subprocess_output_handler():
    lines_seen = []

    class Handler:
        def test_stdout_line(self, line):
            lines_seen.append(line)

    with patch("subprocess.Popen") as popen:
        popen.return_value = _make_popen_mock(["a\n", "b\n"], [], 0)
        run_subprocess(["cmd"], timeout=5, output_handler=Handler())
    assert lines_seen == ["a", "b"]


def test_run_subprocess_timeout():
    """When the timer fires, run_subprocess raises TimeoutExpired."""
    # Patch Timer so its _kill callback fires immediately when start() is called.
    # We use real threading.Event so is_set() correctly reflects the kill.
    class ImmediateTimer:
        """Fires the callback synchronously on start(), like a zero-delay timer."""
        def __init__(self, interval, fn, *args, **kwargs):
            self._fn = fn
        def start(self):
            self._fn()
        def cancel(self):
            pass
        # threading.Timer sets daemon=True before start
        daemon = True

    with patch("subprocess.Popen") as popen, \
         patch("testrunner.commands.common.threading.Timer", ImmediateTimer):
        popen.return_value = _make_popen_mock([], [], -9)
        with pytest.raises(subprocess.TimeoutExpired):
            run_subprocess(["sleep", "100"], timeout=0.001)


# ---------------------------------------------------------------------------
# build_eval_grad_cmd
# ---------------------------------------------------------------------------


def test_build_eval_grad_cmd_docker(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = test_dir / "actual"
    output_dir.mkdir()
    config = {"command": "eval", "network": "net.mininn", "inputs": ["input.bin"]}
    cmd, cwd = build_eval_grad_cmd(config, test_dir, output_dir, "docker", "myimage:latest")
    assert cmd[0] == "docker"
    assert "myimage:latest" in cmd
    assert "eval" in cmd
    assert "--output-dir" in cmd
    assert cwd is None


def test_build_eval_grad_cmd_docker_no_inputs(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = test_dir / "actual"
    output_dir.mkdir()
    config = {"command": "eval", "network": "net.mininn", "inputs": []}
    cmd, cwd = build_eval_grad_cmd(config, test_dir, output_dir, "docker", "img")
    assert "/data/net.mininn" in cmd


def test_build_eval_grad_cmd_local(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = test_dir / "actual"
    output_dir.mkdir()
    config = {"command": "eval", "network": "net.mininn", "inputs": ["input.bin"]}
    cmd, cwd = build_eval_grad_cmd(config, test_dir, output_dir, "local", "./sut eval_runner")
    assert "./sut" in cmd
    assert "eval_runner" in cmd
    assert str(test_dir / "net.mininn") in cmd
    assert cwd is None


def test_build_eval_grad_cmd_podman(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = test_dir / "actual"
    output_dir.mkdir()
    config = {"command": "eval", "network": "net.mininn", "inputs": ["input.bin"]}
    cmd, cwd = build_eval_grad_cmd(config, test_dir, output_dir, "podman", "myimage:latest")
    assert cmd[0] == "podman"
    assert cmd[1] == "run"
    assert "myimage:latest" in cmd
    assert "eval" in cmd
    assert f"{test_dir.resolve()}:/data" in cmd
    assert "/data/net.mininn" in cmd
    assert "/data/input.bin" in cmd
    assert cwd is None


def test_build_eval_grad_cmd_podman_extra_run_args(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = test_dir / "actual"
    output_dir.mkdir()
    config = {"command": "eval", "network": "n.mininn", "inputs": []}
    extra = ("--network=none", "--memory=1g", "--read-only")
    cmd, _ = build_eval_grad_cmd(
        config, test_dir, output_dir, "podman", "img", extra_run_args=extra
    )
    # Extra args appear between ``run --rm`` and the ``-v`` mount
    for flag in extra:
        assert flag in cmd
    # Image name still follows the mount
    assert cmd.index("img") > cmd.index(f"{test_dir.resolve()}:/data")


def test_build_eval_grad_cmd_docker_extra_run_args(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = test_dir / "actual"
    output_dir.mkdir()
    config = {"command": "eval", "network": "n.mininn", "inputs": []}
    cmd, _ = build_eval_grad_cmd(
        config, test_dir, output_dir, "docker", "img", extra_run_args=("--network=none",)
    )
    assert "--network=none" in cmd


def test_build_eval_grad_cmd_local_ignores_extra_run_args(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = test_dir / "actual"
    output_dir.mkdir()
    config = {"command": "eval", "network": "n.mininn", "inputs": []}
    cmd, _ = build_eval_grad_cmd(
        config, test_dir, output_dir, "local", "./sut", extra_run_args=("--network=none",)
    )
    assert "--network=none" not in cmd


def test_is_container_backend():
    assert is_container_backend("docker")
    assert is_container_backend("podman")
    assert not is_container_backend("local")
    assert not is_container_backend("other")


def test_container_run_prefix_shape(tmp_path):
    prefix = container_run_prefix("podman", tmp_path, ("--network=none",))
    assert prefix[0] == "podman"
    assert prefix[1] == "run"
    assert prefix[2] == "--rm"
    assert "--network=none" in prefix
    assert "-v" in prefix
    assert f"{tmp_path.resolve()}:/data" in prefix


def test_build_eval_grad_cmd_local_multiple_inputs(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    output_dir = test_dir / "actual"
    output_dir.mkdir()
    config = {
        "command": "grad",
        "network": "net.mininn",
        "inputs": ["a.bin", "b.bin"],
    }
    cmd, cwd = build_eval_grad_cmd(config, test_dir, output_dir, "local", "python sut.py")
    assert str(test_dir / "a.bin") in cmd
    assert str(test_dir / "b.bin") in cmd
