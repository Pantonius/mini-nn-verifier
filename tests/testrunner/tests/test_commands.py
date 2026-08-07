# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.commands (eval, grad, train __init__)."""

from pathlib import Path

import pytest

from testrunner.commands import COMMANDS, RUNNERS, _init_runners
from testrunner.commands.eval import build_eval_cmd
from testrunner.commands.grad import build_grad_cmd
from testrunner.commands.train import build_train_cmd


# ---------------------------------------------------------------------------
# build_eval_cmd / build_grad_cmd — thin wrappers
# ---------------------------------------------------------------------------


def test_build_eval_cmd_delegates(tmp_path):
    test_dir = tmp_path / "t"
    test_dir.mkdir()
    out = test_dir / "actual"
    out.mkdir()
    config = {"command": "eval", "network": "n.mininn", "inputs": []}
    cmd, cwd = build_eval_cmd(config, test_dir, out, "local", "sut")
    assert "eval" in cmd


def test_build_grad_cmd_delegates(tmp_path):
    test_dir = tmp_path / "t"
    test_dir.mkdir()
    out = test_dir / "actual"
    out.mkdir()
    config = {"command": "grad", "network": "n.mininn", "inputs": []}
    cmd, cwd = build_grad_cmd(config, test_dir, out, "local", "sut")
    assert "grad" in cmd


# ---------------------------------------------------------------------------
# COMMANDS registry
# ---------------------------------------------------------------------------


def test_commands_registry():
    assert "eval" in COMMANDS
    assert "grad" in COMMANDS


# ---------------------------------------------------------------------------
# command_sort_key
# ---------------------------------------------------------------------------


def test_command_sort_key_known():
    from testrunner.commands import command_sort_key, COMMAND_ORDER
    assert command_sort_key("eval") == 0
    assert command_sort_key("grad") == 1
    assert command_sort_key("bounds") == 2
    assert command_sort_key("affine_bounds") == 3
    assert command_sort_key("verify") == 4
    assert command_sort_key("verify2") == 5
    assert command_sort_key("train") == 6
    # cheap commands always sort before expensive ones
    assert command_sort_key("eval") < command_sort_key("train")


def test_command_sort_key_unknown():
    from testrunner.commands import command_sort_key, COMMAND_ORDER
    # Unknown commands sort after all known ones
    assert command_sort_key("unknown_cmd") == len(COMMAND_ORDER)


# ---------------------------------------------------------------------------
# _init_runners
# ---------------------------------------------------------------------------


def test_init_runners_populates():
    RUNNERS.clear()
    _init_runners()
    assert "train" in RUNNERS
    assert "fuzz_eval" in RUNNERS
    assert "fuzz_grad" in RUNNERS


def test_init_runners_idempotent():
    _init_runners()
    first = dict(RUNNERS)
    _init_runners()
    assert RUNNERS == first


# ---------------------------------------------------------------------------
# build_train_cmd — docker and local, absolute and relative inputs
# ---------------------------------------------------------------------------


def test_build_train_cmd_docker(tmp_path):
    test_dir = tmp_path / "t"
    test_dir.mkdir()
    out = test_dir / "actual"
    out.mkdir()
    config = {
        "command": "train",
        "dataset": "mnist_mlp",
        "train_inputs": ["resources/train_images.bin", "resources/train_labels.bin"],
    }
    cmd, cwd = build_train_cmd(config, test_dir, out, "docker", "img")
    assert cmd[0] == "docker"
    assert "train" in cmd
    assert "mnist_mlp" in cmd
    assert cwd is None


def test_build_train_cmd_podman(tmp_path):
    test_dir = tmp_path / "t"
    test_dir.mkdir()
    out = test_dir / "actual"
    out.mkdir()
    config = {
        "command": "train",
        "dataset": "mnist_mlp",
        "train_inputs": ["resources/train_images.bin"],
    }
    cmd, cwd = build_train_cmd(
        config, test_dir, out, "podman", "img", extra_run_args=("--network=none",)
    )
    assert cmd[0] == "podman"
    assert cmd[1] == "run"
    assert "--network=none" in cmd
    assert "mnist_mlp" in cmd
    assert "/data/resources/train_images.bin" in cmd
    assert cwd is None


def test_build_train_cmd_local_relative_inputs(tmp_path):
    test_dir = tmp_path / "t"
    test_dir.mkdir()
    out = test_dir / "actual"
    out.mkdir()
    config = {
        "command": "train",
        "dataset": "ds",
        "train_inputs": ["resources/train.bin"],
    }
    cmd, cwd = build_train_cmd(config, test_dir, out, "local", "python run.py")
    # Relative path should be resolved to test_dir / path
    assert str(test_dir / "resources/train.bin") in cmd
    assert cwd is None


def test_build_train_cmd_local_absolute_inputs(tmp_path):
    test_dir = tmp_path / "t"
    test_dir.mkdir()
    out = test_dir / "actual"
    out.mkdir()
    abs_path = str(tmp_path / "cache" / "train.bin")
    config = {
        "command": "train",
        "dataset": "ds",
        "train_inputs": [abs_path],
    }
    cmd, cwd = build_train_cmd(config, test_dir, out, "local", "sut")
    assert abs_path in cmd
