# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Command builder and custom runner for train.

Expected test.json format (with source_dataset for cached datasets):
    {
        "command": "train",
        "dataset": "mnist_mlp",
        "in_size": 784,
        "num_classes": 10,
        "source_dataset": "mnist"
    }

Or with explicit paths (no caching):
    {
        "command": "train",
        "dataset": "mnist_mlp",
        "in_size": 784,
        "num_classes": 10,
        "train_inputs": ["resources/train_images.bin", "resources/train_labels.bin"],
        "train_labels": "resources/train_labels.bin",
        "test_inputs": ["resources/test_images.bin"],
        "test_labels": "resources/test_labels.bin"
    }

Checkpoint placement
--------------------
The training command receives the test directory (mounted at ``/data`` for
Docker) via ``--output-dir``.  **Checkpoints must be saved inside that
directory.**  When using the docker backend the test runner mounts only the
test directory into the container, so any checkpoint written elsewhere is
inaccessible and will cause an error during accuracy evaluation.
"""

import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from testrunner.commands.common import (
    build_eval_grad_cmd,
    container_run_prefix,
    get_timeout,
    is_container_backend,
    parse_output_paths,
    run_subprocess,
)


def _translate_container_path(path_str, container_root):
    """Map an in-container /data path to its host equivalent."""
    if container_root is None:
        return path_str
    if path_str == "/data":
        return str(container_root)
    if path_str.startswith("/data/"):
        return str(Path(container_root) / path_str[len("/data/"):])
    return path_str


class _TrainProgressHandler:
    """Wrap an output handler to show a live checkpoint-count indicator.

    The train command prints ``eval_batch_size: N`` followed by one
    checkpoint path per line. We forward every line to the inner handler
    and, on a TTY, overwrite a single status line showing how many
    checkpoints have been written so far.
    """

    def __init__(self, inner):
        self._inner = inner
        self._count = 0
        self._seen_header = False
        self._tty = sys.stderr.isatty()
        self._active = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def test_stdout_line(self, line):
        if self._inner is not None:
            self._inner.test_stdout_line(line)
        stripped = line.strip()
        if not stripped:
            return
        if not self._seen_header and stripped.startswith("eval_batch_size:"):
            self._seen_header = True
            return
        self._count += 1
        if self._tty:
            suffix = "" if self._count == 1 else "s"
            msg = f"  train [{self._count} checkpoint{suffix} written]"
            prefix = "\033[1A\r\033[2K" if self._active else ""
            sys.stderr.write(f"{prefix}{msg}\n")
            sys.stderr.flush()
            self._active = True


def build_train_cmd(config, test_dir, output_dir, backend, backend_arg, extra_run_args=()):
    """Build the CLI command for the train step."""
    dataset = config["dataset"]
    train_inputs = config["train_inputs"]

    if is_container_backend(backend):
        # Inputs under test_dir are reachable via /data (the test_dir bind
        # mount). Absolute inputs outside test_dir (e.g. cached dataset
        # files injected by run_train_test when source_dataset is set)
        # need their own read-only bind mount.
        extra_mounts = []
        cache_mounts = {}  # host parent dir -> in-container mount point
        rewritten_inputs = []
        for i in train_inputs:
            p = Path(i)
            if p.is_absolute() and not p.is_relative_to(test_dir):
                parent = str(p.parent)
                if parent not in cache_mounts:
                    mount_point = f"/datasets_{len(cache_mounts)}"
                    cache_mounts[parent] = mount_point
                    extra_mounts.extend(["-v", f"{parent}:{mount_point}:ro"])
                rewritten_inputs.append(f"{cache_mounts[parent]}/{p.name}")
            else:
                rel = p.relative_to(test_dir) if p.is_absolute() else p
                rewritten_inputs.append(f"/data/{rel}")

        cmd = [
            *container_run_prefix(backend, test_dir, extra_run_args),
            *extra_mounts,
            backend_arg,
            "train",
            "--output-dir",
            f"/data/{output_dir.relative_to(test_dir)}",
            dataset,
            *rewritten_inputs,
        ]
        return cmd, None
    else:
        resolved_inputs = []
        for i in train_inputs:
            p = Path(i)
            resolved_inputs.append(str(p if p.is_absolute() else test_dir / p))
        cmd = [
            *shlex.split(backend_arg),
            "train",
            "--output-dir",
            str(output_dir),
            dataset,
            *resolved_inputs,
        ]
        return cmd, None


def _eval_accuracy(
    checkpoint_path,
    input_bin,
    labels_bin,
    eval_batch_size,
    in_size,
    num_classes,
    output_dir,
    backend,
    backend_arg,
    test_dir,
    batch_paths=None,
    timeout=60,
    extra_run_args=(),
):
    """Evaluate a checkpoint on a dataset and return accuracy.

    If batch_paths is provided, uses those pre-split .bin files directly.
    Otherwise splits the input into batches of eval_batch_size on the fly.

    When using a container backend (docker/podman) the checkpoint must be
    inside test_dir, because only that directory is mounted into the
    container.
    """
    if is_container_backend(backend) and not checkpoint_path.is_relative_to(test_dir):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' is outside the test directory "
            f"'{test_dir}'. When using a container backend ({backend}), the "
            f"training command must save all checkpoints inside the test "
            f"directory (i.e. inside the directory passed via --output-dir)."
        )

    labels = np.fromfile(labels_bin, dtype=np.float64)
    num_labels = labels.size // num_classes
    labels = labels.reshape(num_labels, num_classes)

    if batch_paths is not None:
        num_batches = len(batch_paths)
        num_samples = num_batches * eval_batch_size
    else:
        images = np.fromfile(input_bin, dtype=np.float64)
        num_samples = images.size // in_size
        images = images.reshape(num_samples, in_size)
        if num_samples % eval_batch_size != 0:
            return None, (
                f"number of samples ({num_samples}) is not divisible "
                f"by eval_batch_size ({eval_batch_size})"
            )
        num_batches = num_samples // eval_batch_size

    all_preds = []

    for batch_idx in range(num_batches):
        if batch_paths is not None:
            batch_input_path = batch_paths[batch_idx]
        else:
            start = batch_idx * eval_batch_size
            end = start + eval_batch_size
            batch_data = images[start:end]
            batch_input_path = output_dir / f"batch_{batch_idx}_input.bin"
            batch_data.tofile(batch_input_path)

        batch_output_dir = output_dir / f"batch_{batch_idx}_output"
        batch_output_dir.mkdir(exist_ok=True)

        # When batch_input_path is from a cached dataset outside test_dir,
        # we can't use build_eval_grad_cmd (which assumes paths relative to
        # test_dir). Build the command directly with absolute paths instead.
        all_under_test_dir = checkpoint_path.is_relative_to(
            test_dir
        ) and batch_input_path.is_relative_to(test_dir)
        if all_under_test_dir:
            eval_config = {
                "command": "eval",
                "network": str(checkpoint_path.relative_to(test_dir)),
                "inputs": [str(batch_input_path.relative_to(test_dir))],
            }
            cmd, _ = build_eval_grad_cmd(
                eval_config,
                test_dir,
                batch_output_dir,
                backend,
                backend_arg,
                extra_run_args=extra_run_args,
            )
        elif is_container_backend(backend):
            cmd = [
                *container_run_prefix(backend, test_dir, extra_run_args),
                "-v",
                f"{batch_input_path.resolve()}:/input/{batch_input_path.name}",
                backend_arg,
                "eval",
                "--output-dir",
                f"/data/{batch_output_dir.relative_to(test_dir)}",
                f"/data/{checkpoint_path.relative_to(test_dir)}",
                f"/input/{batch_input_path.name}",
            ]
        else:
            cmd = [
                *shlex.split(backend_arg),
                "eval",
                "--output-dir",
                str(batch_output_dir),
                str(checkpoint_path),
                str(batch_input_path),
            ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, (
                f"eval timed out on {checkpoint_path.name} batch {batch_idx} after {timeout}s"
            )
        if result.returncode != 0:
            return None, (
                f"eval failed on {checkpoint_path.name} batch {batch_idx}: {result.stderr.strip()}"
            )

        # Parse output file paths from stdout. In container mode the SUT
        # prints /data-rooted paths; translate them to host paths.
        output_files, _ = parse_output_paths(
            result.stdout,
            container_root=test_dir if is_container_backend(backend) else None,
        )

        if not output_files:
            return None, f"eval produced no output for batch {batch_idx}"

        batch_preds = np.fromfile(output_files[0], dtype=np.float64)
        all_preds.append(batch_preds)

    preds = np.concatenate(all_preds).reshape(num_samples, num_classes)
    # Only compare up to the number of samples we evaluated
    accuracy = float(np.mean(np.argmax(preds, axis=-1) == np.argmax(labels[:num_samples], axis=-1)))
    return accuracy, None


def run_train_test(
    test_dir,
    config,
    output_dir,
    backend,
    backend_arg,
    generate=False,
    output_handler=None,
    closed=False,
    extra_run_args=(),
):
    """Custom runner for train tests.

    Runs the train command, evaluates each checkpoint on train and test sets,
    and computes classification accuracy.
    """
    in_size = config["in_size"]
    num_classes = config["num_classes"]

    # Resolve dataset paths from cache if source_dataset is present
    dataset_cache_dir = None
    if "source_dataset" in config:
        from testrunner.datasets import prepare_dataset

        dataset_paths = prepare_dataset(config["source_dataset"])
        dataset_cache_dir = Path(dataset_paths["train_images"]).parent
        config = {
            **config,
            "train_inputs": [dataset_paths["train_images"], dataset_paths["train_labels"]],
            "train_labels": dataset_paths["train_labels"],
            "test_inputs": [dataset_paths["test_images"]],
            "test_labels": dataset_paths["test_labels"],
        }

    # Step 1: Run train command
    timeout = get_timeout(config)
    cmd, cwd = build_train_cmd(
        config, test_dir, output_dir, backend, backend_arg, extra_run_args=extra_run_args
    )
    progress_handler = (
        _TrainProgressHandler(output_handler) if output_handler is not None else None
    )
    try:
        result = run_subprocess(
            cmd,
            cwd=cwd,
            timeout=timeout,
            log_file=None if closed else output_dir / "stdout.log",
            output_handler=progress_handler,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"train command timed out after {timeout}s"}

    if result.returncode != 0:
        return {
            "passed": False,
            "error": "command failed"
            if closed
            else f"train command failed (exit {result.returncode}): {result.stderr.strip()}",
        }

    # Step 2: Parse stdout — first line is eval_batch_size, rest are checkpoint paths
    lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        return {
            "passed": False,
            "error": "command failed" if closed else "train command produced no output",
        }

    first_line = lines[0]
    if not first_line.startswith("eval_batch_size:"):
        return {
            "passed": False,
            "error": "command failed"
            if closed
            else f"expected first output line to be 'eval_batch_size: N', got: {first_line}",
        }
    eval_batch_size = int(first_line.split(":", 1)[1].strip())

    container_root = test_dir if is_container_backend(backend) else None
    checkpoint_paths = []
    for line in lines[1:]:
        host_path = _translate_container_path(line, container_root)
        p = Path(host_path)
        if p.exists():
            checkpoint_paths.append(p)
        else:
            print(f"warning: ignoring non-existent checkpoint path: {line}", file=sys.stderr)

    if not checkpoint_paths:
        return {
            "passed": False,
            "error": "command failed" if closed else "train produced no checkpoint files",
        }

    if generate:
        return {
            "passed": True,
            "error": None,
            "generated": True,
            "checkpoints": [str(p) for p in checkpoint_paths],
        }

    # Step 3: Resolve input/label paths
    #   When source_dataset is used, paths are already absolute.
    #   Otherwise they are relative to test_dir.
    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else test_dir / p

    test_images = _resolve(config["test_inputs"][0])
    test_labels = _resolve(config["test_labels"])
    train_images = _resolve(config["train_inputs"][0])
    train_labels = _resolve(config["train_labels"])

    # Prepare cached batch splits if we have a dataset cache
    cached_batches = {}
    if dataset_cache_dir is not None:
        from testrunner.datasets.mnist import prepare_batches

        for split_name, images_path in [("test", test_images), ("train", train_images)]:
            cached_batches[split_name] = prepare_batches(
                dataset_cache_dir, split_name, in_size, eval_batch_size
            )

    # Step 4: Evaluate each checkpoint on both splits
    checkpoint_results = {}
    best_test_accuracy = 0.0

    for cp_path in checkpoint_paths:
        cp_name = cp_path.name
        cp_result = {}

        for split_name, images_path, labels_path in [
            ("test", test_images, test_labels),
            ("train", train_images, train_labels),
        ]:
            split_output_dir = output_dir / f"eval_{cp_name}_{split_name}"
            split_output_dir.mkdir(parents=True, exist_ok=True)

            accuracy, error = _eval_accuracy(
                cp_path,
                images_path,
                labels_path,
                eval_batch_size,
                in_size,
                num_classes,
                split_output_dir,
                backend,
                backend_arg,
                test_dir,
                batch_paths=cached_batches.get(split_name),
                timeout=get_timeout({**config, "command": "eval"}),
                extra_run_args=extra_run_args,
            )
            if error:
                return {"passed": False, "error": "command failed" if closed else error}

            cp_result[f"{split_name}_accuracy"] = accuracy

        checkpoint_results[cp_name] = cp_result
        if cp_result["test_accuracy"] > best_test_accuracy:
            best_test_accuracy = cp_result["test_accuracy"]

    return {
        "passed": True,
        "error": None,
        "best_test_accuracy": best_test_accuracy,
        "checkpoint_results": checkpoint_results,
    }
