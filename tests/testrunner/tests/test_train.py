# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.commands.train."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from testrunner.commands.train import (
    _eval_accuracy,
    build_train_cmd,
    run_train_test,
)
from testrunner.commands.common import SubprocessResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_bin(path, arr):
    arr = np.array(arr, dtype=np.float64)
    arr.tofile(path)
    return path


def _write_labels(path, labels_list, num_classes):
    """Write one-hot encoded labels as float64."""
    n = len(labels_list)
    one_hot = np.zeros((n, num_classes), dtype=np.float64)
    for i, label in enumerate(labels_list):
        one_hot[i, label] = 1.0
    one_hot.tofile(path)
    return path


def _make_test_dir(tmp_path, config):
    d = tmp_path / "test"
    d.mkdir(exist_ok=True)
    (d / "test.json").write_text(json.dumps(config))
    return d


# ---------------------------------------------------------------------------
# _eval_accuracy
# ---------------------------------------------------------------------------


def _make_completed_process(stdout="", stderr="", returncode=0):
    r = MagicMock()
    r.stdout = stdout
    r.stderr = stderr
    r.returncode = returncode
    return r


def test_eval_accuracy_basic(tmp_path):
    """_eval_accuracy should correctly compute accuracy."""
    n_samples = 4
    eval_batch_size = 2
    in_size = 3
    num_classes = 2

    # Images: 4 samples, 3 features each
    images = np.random.rand(n_samples, in_size).astype(np.float64)
    _write_bin(tmp_path / "images.bin", images)

    # Labels: alternating 0,1,0,1
    labels = [0, 1, 0, 1]
    _write_labels(tmp_path / "labels.bin", labels, num_classes)

    # Mock eval subprocess: output predicts correctly
    output_dir = tmp_path / "eval_out"
    output_dir.mkdir()

    call_count = [0]

    def mock_subprocess_run(cmd, capture_output, text, timeout):
        i = call_count[0]
        call_count[0] += 1
        # Batch i: 2 samples
        start = i * eval_batch_size
        batch_labels = labels[start : start + eval_batch_size]
        # Perfect predictions: one-hot class
        preds = np.zeros((eval_batch_size, num_classes), dtype=np.float64)
        for j, lbl in enumerate(batch_labels):
            preds[j, lbl] = 1.0
        out_path = output_dir / f"pred_{i}.bin"
        preds.tofile(out_path)
        return _make_completed_process(stdout=str(out_path))

    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"fake")

    with patch("subprocess.run", side_effect=mock_subprocess_run):
        accuracy, error = _eval_accuracy(
            checkpoint,
            tmp_path / "images.bin",
            tmp_path / "labels.bin",
            eval_batch_size,
            in_size,
            num_classes,
            output_dir,
            "local",
            "sut",
            tmp_path,
        )
    assert error is None
    assert accuracy == 1.0


def test_eval_accuracy_non_divisible(tmp_path):
    """Non-divisible sample count should return an error."""
    n_samples = 5
    eval_batch_size = 2
    in_size = 3
    num_classes = 2

    images = np.zeros((n_samples, in_size), dtype=np.float64)
    _write_bin(tmp_path / "images.bin", images)
    _write_labels(tmp_path / "labels.bin", [0] * n_samples, num_classes)

    accuracy, error = _eval_accuracy(
        tmp_path / "cp.bin",
        tmp_path / "images.bin",
        tmp_path / "labels.bin",
        eval_batch_size,
        in_size,
        num_classes,
        tmp_path / "out",
        "local",
        "sut",
        tmp_path,
    )
    assert accuracy is None
    assert "not divisible" in error


def test_eval_accuracy_subprocess_failure(tmp_path):
    n_samples = 2
    eval_batch_size = 2
    in_size = 2
    num_classes = 2

    images = np.zeros((n_samples, in_size), dtype=np.float64)
    _write_bin(tmp_path / "images.bin", images)
    _write_labels(tmp_path / "labels.bin", [0, 1], num_classes)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with patch("subprocess.run", return_value=_make_completed_process(returncode=1, stderr="crash")):
        accuracy, error = _eval_accuracy(
            tmp_path / "cp.bin",
            tmp_path / "images.bin",
            tmp_path / "labels.bin",
            eval_batch_size,
            in_size,
            num_classes,
            out_dir,
            "local",
            "sut",
            tmp_path,
        )
    assert accuracy is None
    assert "eval failed" in error


def test_eval_accuracy_timeout(tmp_path):
    n_samples = 2
    eval_batch_size = 2
    in_size = 2
    num_classes = 2

    images = np.zeros((n_samples, in_size), dtype=np.float64)
    _write_bin(tmp_path / "images.bin", images)
    _write_labels(tmp_path / "labels.bin", [0, 1], num_classes)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
        accuracy, error = _eval_accuracy(
            tmp_path / "cp.bin",
            tmp_path / "images.bin",
            tmp_path / "labels.bin",
            eval_batch_size,
            in_size,
            num_classes,
            out_dir,
            "local",
            "sut",
            tmp_path,
        )
    assert accuracy is None
    assert "timed out" in error


def test_eval_accuracy_no_output(tmp_path):
    n_samples = 2
    eval_batch_size = 2
    in_size = 2
    num_classes = 2

    images = np.zeros((n_samples, in_size), dtype=np.float64)
    _write_bin(tmp_path / "images.bin", images)
    _write_labels(tmp_path / "labels.bin", [0, 1], num_classes)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with patch("subprocess.run", return_value=_make_completed_process(stdout="")):
        accuracy, error = _eval_accuracy(
            tmp_path / "cp.bin",
            tmp_path / "images.bin",
            tmp_path / "labels.bin",
            eval_batch_size,
            in_size,
            num_classes,
            out_dir,
            "local",
            "sut",
            tmp_path,
        )
    assert accuracy is None
    assert "no output" in error


def test_eval_accuracy_docker_external_checkpoint_raises(tmp_path):
    """_eval_accuracy with docker backend raises when checkpoint is outside test_dir.

    Checkpoints must be saved inside the test directory when using docker,
    because only that directory is mounted into the container.
    """
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    test_dir = tmp_path / "test"
    test_dir.mkdir()

    checkpoint = external_dir / "checkpoint.bin"
    checkpoint.write_bytes(b"cp")

    out_dir = test_dir / "out"
    out_dir.mkdir()

    with pytest.raises(ValueError, match="outside the test directory"):
        _eval_accuracy(
            checkpoint,
            external_dir / "images.bin",
            external_dir / "labels.bin",
            eval_batch_size=2,
            in_size=2,
            num_classes=2,
            output_dir=out_dir,
            backend="docker",
            backend_arg="myimage",
            test_dir=test_dir,
        )


def test_eval_accuracy_docker_external_batch(tmp_path):
    """Docker + checkpoint under test_dir + batch_paths outside test_dir uses mounted volumes."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    external_dir = tmp_path / "external"
    external_dir.mkdir()

    eval_batch_size = 2
    in_size = 2
    num_classes = 2

    # Pre-split batch file lives outside test_dir (simulates a cached dataset)
    batch_file = external_dir / "batch_0.bin"
    np.zeros((eval_batch_size, in_size), dtype=np.float64).tofile(batch_file)
    _write_labels(external_dir / "labels.bin", [0, 0], num_classes)

    checkpoint = test_dir / "checkpoint.bin"
    checkpoint.write_bytes(b"cp")

    out_dir = test_dir / "out"
    out_dir.mkdir()

    preds = np.zeros((eval_batch_size, num_classes), dtype=np.float64)
    preds[:, 0] = 1.0
    out_file = out_dir / "pred.bin"
    preds.tofile(out_file)

    with patch("subprocess.run",
               return_value=_make_completed_process(stdout=str(out_file))) as mock_run:
        accuracy, error = _eval_accuracy(
            checkpoint,
            None,
            external_dir / "labels.bin",
            eval_batch_size,
            in_size,
            num_classes,
            out_dir,
            "docker",
            "myimage",
            test_dir,
            batch_paths=[batch_file],
        )
    assert error is None
    assert accuracy == 1.0
    # Docker command should mount the external batch file via a separate -v flag
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "docker"
    assert "/input/" in " ".join(cmd)


def test_eval_accuracy_local_external_batch(tmp_path):
    """Local backend + batch_paths outside test_dir uses absolute paths directly."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    external_dir = tmp_path / "external"
    external_dir.mkdir()

    eval_batch_size = 2
    in_size = 2
    num_classes = 2

    batch_file = external_dir / "batch_0.bin"
    np.zeros((eval_batch_size, in_size), dtype=np.float64).tofile(batch_file)
    _write_labels(external_dir / "labels.bin", [0, 0], num_classes)

    checkpoint = test_dir / "checkpoint.bin"
    checkpoint.write_bytes(b"cp")

    out_dir = test_dir / "out"
    out_dir.mkdir()

    preds = np.zeros((eval_batch_size, num_classes), dtype=np.float64)
    preds[:, 0] = 1.0
    out_file = out_dir / "pred.bin"
    preds.tofile(out_file)

    with patch("subprocess.run",
               return_value=_make_completed_process(stdout=str(out_file))) as mock_run:
        accuracy, error = _eval_accuracy(
            checkpoint,
            None,
            external_dir / "labels.bin",
            eval_batch_size,
            in_size,
            num_classes,
            out_dir,
            "local",
            "sut",
            test_dir,
            batch_paths=[batch_file],
        )
    assert error is None
    assert accuracy == 1.0
    # Local command should use the absolute path to the external batch file
    cmd = mock_run.call_args[0][0]
    assert str(batch_file) in cmd


def test_eval_accuracy_with_batch_paths(tmp_path):
    """When batch_paths is provided, should use pre-split batches."""
    n_batches = 2
    eval_batch_size = 2
    in_size = 3
    num_classes = 2
    n_samples = n_batches * eval_batch_size

    # Create batch .bin files
    batch_paths = []
    for i in range(n_batches):
        bp = tmp_path / f"batch_{i}.bin"
        np.zeros((eval_batch_size, in_size), dtype=np.float64).tofile(bp)
        batch_paths.append(bp)

    _write_labels(tmp_path / "labels.bin", [0] * n_samples, num_classes)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    call_count = [0]

    def mock_run(cmd, capture_output, text, timeout):
        i = call_count[0]
        call_count[0] += 1
        preds = np.zeros((eval_batch_size, num_classes), dtype=np.float64)
        preds[:, 0] = 1.0  # predict class 0 for all
        out_path = out_dir / f"pred_{i}.bin"
        preds.tofile(out_path)
        return _make_completed_process(stdout=str(out_path))

    with patch("subprocess.run", side_effect=mock_run):
        accuracy, error = _eval_accuracy(
            tmp_path / "cp.bin",
            None,  # image bin not needed when batch_paths provided
            tmp_path / "labels.bin",
            eval_batch_size,
            in_size,
            num_classes,
            out_dir,
            "local",
            "sut",
            tmp_path,
            batch_paths=batch_paths,
        )
    assert error is None
    assert accuracy == 1.0  # all predict class 0, labels are class 0


# ---------------------------------------------------------------------------
# run_train_test — happy path
# ---------------------------------------------------------------------------


def _make_stdout_for_train(checkpoint_paths, eval_batch_size=2):
    lines = [f"eval_batch_size: {eval_batch_size}"] + [str(p) for p in checkpoint_paths]
    return "\n".join(lines)


def test_run_train_test_generate(tmp_path):
    test_dir = _make_test_dir(tmp_path, {
        "command": "train",
        "dataset": "ds",
        "in_size": 4,
        "num_classes": 2,
        "train_inputs": ["t.bin"],
        "train_labels": "l.bin",
        "test_inputs": ["te.bin"],
        "test_labels": "tl.bin",
    })
    cp = tmp_path / "checkpoint.bin"
    cp.write_bytes(b"cp")

    with patch("testrunner.commands.train.run_subprocess",
               return_value=SubprocessResult(0, _make_stdout_for_train([cp]), "")):
        result = run_train_test(test_dir, json.loads((test_dir / "test.json").read_text()),
                                test_dir / "actual", "local", "sut", generate=True)
    assert result["passed"] is True
    assert result.get("generated") is True


def test_run_train_test_no_output(tmp_path):
    test_dir = _make_test_dir(tmp_path, {
        "command": "train",
        "dataset": "ds",
        "in_size": 4,
        "num_classes": 2,
        "train_inputs": ["t.bin"],
        "train_labels": "l.bin",
        "test_inputs": ["te.bin"],
        "test_labels": "tl.bin",
    })
    (test_dir / "actual").mkdir()
    with patch("testrunner.commands.train.run_subprocess",
               return_value=SubprocessResult(0, "", "")):
        result = run_train_test(test_dir, json.loads((test_dir / "test.json").read_text()),
                                test_dir / "actual", "local", "sut")
    assert result["passed"] is False
    assert "no output" in result["error"]


def test_run_train_test_missing_eval_batch_size(tmp_path):
    test_dir = _make_test_dir(tmp_path, {
        "command": "train",
        "dataset": "ds",
        "in_size": 4,
        "num_classes": 2,
        "train_inputs": ["t.bin"],
        "train_labels": "l.bin",
        "test_inputs": ["te.bin"],
        "test_labels": "tl.bin",
    })
    (test_dir / "actual").mkdir()
    with patch("testrunner.commands.train.run_subprocess",
               return_value=SubprocessResult(0, "bad first line", "")):
        result = run_train_test(test_dir, json.loads((test_dir / "test.json").read_text()),
                                test_dir / "actual", "local", "sut")
    assert result["passed"] is False
    assert "eval_batch_size" in result["error"]


def test_run_train_test_no_checkpoints(tmp_path):
    test_dir = _make_test_dir(tmp_path, {
        "command": "train",
        "dataset": "ds",
        "in_size": 4,
        "num_classes": 2,
        "train_inputs": ["t.bin"],
        "train_labels": "l.bin",
        "test_inputs": ["te.bin"],
        "test_labels": "tl.bin",
    })
    (test_dir / "actual").mkdir()
    # Only the eval_batch_size line, no checkpoint paths
    with patch("testrunner.commands.train.run_subprocess",
               return_value=SubprocessResult(0, "eval_batch_size: 2\n/nonexistent/cp.bin", "")):
        result = run_train_test(test_dir, json.loads((test_dir / "test.json").read_text()),
                                test_dir / "actual", "local", "sut")
    assert result["passed"] is False
    assert "no checkpoint" in result["error"]


def test_run_train_test_command_failure(tmp_path):
    test_dir = _make_test_dir(tmp_path, {
        "command": "train",
        "dataset": "ds",
        "in_size": 4,
        "num_classes": 2,
        "train_inputs": ["t.bin"],
        "train_labels": "l.bin",
        "test_inputs": ["te.bin"],
        "test_labels": "tl.bin",
    })
    (test_dir / "actual").mkdir()
    with patch("testrunner.commands.train.run_subprocess",
               return_value=SubprocessResult(1, "", "fatal")):
        result = run_train_test(test_dir, json.loads((test_dir / "test.json").read_text()),
                                test_dir / "actual", "local", "sut")
    assert result["passed"] is False
    assert "train command failed" in result["error"]


def test_run_train_test_command_failure_closed(tmp_path):
    test_dir = _make_test_dir(tmp_path, {
        "command": "train",
        "dataset": "ds",
        "in_size": 4,
        "num_classes": 2,
        "train_inputs": ["t.bin"],
        "train_labels": "l.bin",
        "test_inputs": ["te.bin"],
        "test_labels": "tl.bin",
    })
    (test_dir / "actual").mkdir()
    with patch("testrunner.commands.train.run_subprocess",
               return_value=SubprocessResult(1, "", "secret details")):
        result = run_train_test(test_dir, json.loads((test_dir / "test.json").read_text()),
                                test_dir / "actual", "local", "sut", closed=True)
    assert result["passed"] is False
    assert "secret details" not in result["error"]


def test_run_train_test_timeout(tmp_path):
    test_dir = _make_test_dir(tmp_path, {
        "command": "train",
        "dataset": "ds",
        "in_size": 4,
        "num_classes": 2,
        "train_inputs": ["t.bin"],
        "train_labels": "l.bin",
        "test_inputs": ["te.bin"],
        "test_labels": "tl.bin",
    })
    (test_dir / "actual").mkdir()
    with patch("testrunner.commands.train.run_subprocess",
               side_effect=subprocess.TimeoutExpired("cmd", 600)):
        result = run_train_test(test_dir, json.loads((test_dir / "test.json").read_text()),
                                test_dir / "actual", "local", "sut")
    assert result["passed"] is False
    assert "timed out" in result["error"]


def test_run_train_test_eval_error_propagates(tmp_path):
    """If _eval_accuracy returns an error, run_train_test should fail."""
    test_dir = _make_test_dir(tmp_path, {
        "command": "train",
        "dataset": "ds",
        "in_size": 2,
        "num_classes": 2,
        "train_inputs": [str(tmp_path / "t.bin")],
        "train_labels": str(tmp_path / "l.bin"),
        "test_inputs": [str(tmp_path / "te.bin")],
        "test_labels": str(tmp_path / "tl.bin"),
    })
    actual_dir = test_dir / "actual"
    actual_dir.mkdir()

    cp = tmp_path / "checkpoint.bin"
    cp.write_bytes(b"cp")

    # Create stub data files
    np.zeros(4, dtype=np.float64).tofile(tmp_path / "t.bin")
    np.zeros(4, dtype=np.float64).tofile(tmp_path / "l.bin")
    np.zeros(4, dtype=np.float64).tofile(tmp_path / "te.bin")
    np.zeros(4, dtype=np.float64).tofile(tmp_path / "tl.bin")

    train_out = _make_stdout_for_train([cp], eval_batch_size=2)
    with patch("testrunner.commands.train.run_subprocess",
               return_value=SubprocessResult(0, train_out, "")):
        with patch("testrunner.commands.train._eval_accuracy",
                   return_value=(None, "eval exploded")):
            result = run_train_test(
                test_dir,
                json.loads((test_dir / "test.json").read_text()),
                actual_dir,
                "local",
                "sut",
            )
    assert result["passed"] is False
    assert "eval exploded" in result["error"]


def test_run_train_test_full_pass(tmp_path):
    """Full passing train test run."""
    test_dir = _make_test_dir(tmp_path, {
        "command": "train",
        "dataset": "ds",
        "in_size": 2,
        "num_classes": 2,
        "train_inputs": [str(tmp_path / "t.bin")],
        "train_labels": str(tmp_path / "l.bin"),
        "test_inputs": [str(tmp_path / "te.bin")],
        "test_labels": str(tmp_path / "tl.bin"),
    })
    actual_dir = test_dir / "actual"
    actual_dir.mkdir()

    cp = tmp_path / "checkpoint.bin"
    cp.write_bytes(b"cp")

    np.zeros(4, dtype=np.float64).tofile(tmp_path / "t.bin")
    np.zeros(4, dtype=np.float64).tofile(tmp_path / "l.bin")
    np.zeros(4, dtype=np.float64).tofile(tmp_path / "te.bin")
    np.zeros(4, dtype=np.float64).tofile(tmp_path / "tl.bin")

    train_out = _make_stdout_for_train([cp], eval_batch_size=2)
    with patch("testrunner.commands.train.run_subprocess",
               return_value=SubprocessResult(0, train_out, "")):
        with patch("testrunner.commands.train._eval_accuracy", return_value=(0.9, None)):
            result = run_train_test(
                test_dir,
                json.loads((test_dir / "test.json").read_text()),
                actual_dir,
                "local",
                "sut",
            )
    assert result["passed"] is True
    assert result["best_test_accuracy"] == 0.9


def test_run_train_test_with_source_dataset(tmp_path):
    """run_train_test should resolve paths from source_dataset."""
    # Create fake dataset files
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    n_samples = 4
    in_size = 2
    num_classes = 2
    eval_batch_size = 2

    np.zeros((n_samples, in_size), dtype=np.float64).tofile(cache_dir / "train_images.bin")
    np.zeros((n_samples, in_size), dtype=np.float64).tofile(cache_dir / "test_images.bin")
    # Labels: one-hot
    lbls = np.zeros((n_samples, num_classes), dtype=np.float64)
    lbls[:, 0] = 1.0
    lbls.tofile(cache_dir / "train_labels.bin")
    lbls.tofile(cache_dir / "test_labels.bin")

    fake_paths = {
        "train_images": str(cache_dir / "train_images.bin"),
        "train_labels": str(cache_dir / "train_labels.bin"),
        "test_images": str(cache_dir / "test_images.bin"),
        "test_labels": str(cache_dir / "test_labels.bin"),
    }

    test_dir = _make_test_dir(tmp_path, {
        "command": "train",
        "dataset": "mnist_mlp",
        "in_size": in_size,
        "num_classes": num_classes,
        "source_dataset": "mnist",
    })
    actual_dir = test_dir / "actual"
    actual_dir.mkdir()

    cp = tmp_path / "checkpoint.bin"
    cp.write_bytes(b"cp")
    train_out = _make_stdout_for_train([cp], eval_batch_size=eval_batch_size)

    with patch("testrunner.commands.train.run_subprocess",
               return_value=SubprocessResult(0, train_out, "")):
        with patch("testrunner.commands.train._eval_accuracy", return_value=(0.8, None)):
            with patch("testrunner.datasets.prepare_dataset", return_value=fake_paths):
                with patch("testrunner.datasets.mnist.prepare_batches", return_value=[]):
                    result = run_train_test(
                        test_dir,
                        json.loads((test_dir / "test.json").read_text()),
                        actual_dir,
                        "local",
                        "sut",
                    )
    assert result["passed"] is True


def test_run_train_test_docker_checkpoint_path_translation(tmp_path):
    """Docker backend: checkpoint paths printed as /data/... must resolve to host paths."""
    test_dir = _make_test_dir(tmp_path, {
        "command": "train",
        "dataset": "ds",
        "in_size": 2,
        "num_classes": 2,
        "train_inputs": ["t.bin"],
        "train_labels": "l.bin",
        "test_inputs": ["te.bin"],
        "test_labels": "tl.bin",
    })
    actual_dir = test_dir / "actual"
    actual_dir.mkdir()

    np.zeros(4, dtype=np.float64).tofile(test_dir / "t.bin")
    np.zeros(4, dtype=np.float64).tofile(test_dir / "l.bin")
    np.zeros(4, dtype=np.float64).tofile(test_dir / "te.bin")
    np.zeros(4, dtype=np.float64).tofile(test_dir / "tl.bin")

    # Checkpoint lives under actual/ on the host; student code (running
    # inside the container) prints the in-container path /data/actual/cp.bin.
    cp_host = actual_dir / "cp.bin"
    cp_host.write_bytes(b"cp")
    container_cp_path = f"/data/{cp_host.relative_to(test_dir)}"
    train_stdout = f"eval_batch_size: 2\n{container_cp_path}\n"

    captured = {}

    def fake_eval(cp, *a, **kw):
        captured["cp"] = cp
        return (0.75, None)

    with patch("testrunner.commands.train.run_subprocess",
               return_value=SubprocessResult(0, train_stdout, "")):
        with patch("testrunner.commands.train._eval_accuracy", side_effect=fake_eval):
            result = run_train_test(
                test_dir,
                json.loads((test_dir / "test.json").read_text()),
                actual_dir,
                "docker",
                "myimage",
            )
    assert result["passed"] is True
    assert captured["cp"] == cp_host


def test_run_train_test_no_output_closed(tmp_path):
    """closed=True with no output should mask the message."""
    test_dir = _make_test_dir(tmp_path, {
        "command": "train",
        "dataset": "ds",
        "in_size": 4,
        "num_classes": 2,
        "train_inputs": ["t.bin"],
        "train_labels": "l.bin",
        "test_inputs": ["te.bin"],
        "test_labels": "tl.bin",
    })
    (test_dir / "actual").mkdir()
    with patch("testrunner.commands.train.run_subprocess",
               return_value=SubprocessResult(0, "", "")):
        result = run_train_test(test_dir, json.loads((test_dir / "test.json").read_text()),
                                test_dir / "actual", "local", "sut", closed=True)
    assert result["passed"] is False
    assert result["error"] == "command failed"
