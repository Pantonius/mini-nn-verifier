# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Reproduce a saved fuzz failure.

Usage:
    python -m testrunner.reproduce <backend> <backend_arg> <failure_dir>

Where <failure_dir> is a directory previously saved by the fuzz runner
(containing network.mininn, input .bin files, and metadata.json).
"""

import argparse
import json
import shlex
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from testrunner.fuzz.runner import run_and_check


def reproduce(failure_dir, backend, backend_arg, extra_run_args=()):
    """Reproduce a single saved fuzz failure.

    Copies the saved network and inputs into a temporary directory and
    reruns the fuzz checking logic.

    Returns a result dict with ``passed`` and ``error`` keys.
    """
    failure_dir = Path(failure_dir).resolve()

    metadata_path = failure_dir / "metadata.json"
    if not metadata_path.exists():
        return {"passed": False, "error": f"no metadata.json in {failure_dir}"}

    metadata = json.loads(metadata_path.read_text())
    mode = metadata["mode"]
    check_nan_inf = metadata["check_nan_inf"]
    expected_shapes = [tuple(s) for s in metadata["expected_shapes"]]
    input_names = metadata["inputs"]

    network_path = failure_dir / "network.mininn"
    if not network_path.exists():
        return {"passed": False, "error": f"no network.mininn in {failure_dir}"}

    input_paths = []
    for name in input_names:
        p = failure_dir / name
        if not p.exists():
            return {"passed": False, "error": f"missing input file: {name}"}
        input_paths.append(p)

    # Work in a temporary directory so output files don't pollute the failure dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        tmp_network = tmp_dir / "network.mininn"
        shutil.copy2(network_path, tmp_network)
        tmp_inputs = []
        for p in input_paths:
            dest = tmp_dir / p.name
            shutil.copy2(p, dest)
            tmp_inputs.append(dest)

        if mode == "bounds":
            from testrunner.fuzz import run_and_check_bounds

            if len(tmp_inputs) % 2 != 0:
                return {
                    "passed": False,
                    "error": f"bounds saved failure has odd input count ({len(tmp_inputs)})",
                }
            box_paths = [
                (tmp_inputs[2 * i], tmp_inputs[2 * i + 1])
                for i in range(len(tmp_inputs) // 2)
            ]
            sound_cfg = metadata.get("sound_cfg", {})
            return run_and_check_bounds(
                tmp_network, box_paths, backend, backend_arg, expected_shapes,
                check_nan_inf=check_nan_inf, extra_run_args=extra_run_args,
                n_eval_samples=sound_cfg.get("n_eval_samples", 3),
                sample_atol=sound_cfg.get("sample_atol", 1e-6),
                sample_rtol=sound_cfg.get("sample_rtol", 1e-6),
            )

        return run_and_check(
            tmp_network,
            tmp_inputs,
            backend,
            backend_arg,
            mode,
            expected_shapes,
            check_nan_inf=check_nan_inf,
            extra_run_args=extra_run_args,
        )


def main():
    parser = argparse.ArgumentParser(description="Reproduce a saved fuzz failure.")
    parser.add_argument("backend", choices=["docker", "podman", "local"])
    parser.add_argument(
        "backend_arg", type=str, help="Container image name (docker/podman) or local command"
    )
    parser.add_argument("failure_dir", type=str, help="Path to saved failure directory")
    parser.add_argument(
        "--extra-run-args",
        type=str,
        default="",
        help=(
            "Extra arguments passed through to 'docker run'/'podman run'. "
            "Shell-quoted; ignored for the 'local' backend."
        ),
    )
    args = parser.parse_args()
    extra_run_args = tuple(shlex.split(args.extra_run_args)) if args.extra_run_args else ()

    failure_dir = Path(args.failure_dir).resolve()

    _DIM = "\033[2m"
    _RED = "\033[31m"
    _GREEN = "\033[32m"
    _RESET = "\033[0m"

    # Print network and inputs before running
    _print_failure_inputs(failure_dir)

    result = reproduce(failure_dir, args.backend, args.backend_arg, extra_run_args=extra_run_args)

    if result["passed"]:
        print(f"{_GREEN}PASS: failure did not reproduce{_RESET}", file=sys.stderr)
    else:
        print(f"{_RED}FAIL: {result['error']}{_RESET}", file=sys.stderr)
        if result.get("stdout"):
            print(f"\n{_DIM}--- stdout ---{_RESET}", file=sys.stderr)
            print(result["stdout"], file=sys.stderr)
            print(f"{_DIM}--- end stdout ---{_RESET}", file=sys.stderr)
        if result.get("stderr"):
            print(f"\n{_DIM}--- stderr ---{_RESET}", file=sys.stderr)
            print(result["stderr"], file=sys.stderr)
            print(f"{_DIM}--- end stderr ---{_RESET}", file=sys.stderr)
        sys.exit(1)


def _print_failure_inputs(failure_dir):
    """Print the network and input arrays from a saved failure directory."""
    network_path = failure_dir / "network.mininn"
    if network_path.exists():
        print(f"Network ({network_path}):", file=sys.stderr)
        with zipfile.ZipFile(network_path) as z:
            graph = z.read("graph.txt").decode()
        for line in graph.strip().splitlines():
            print(f"  {line}", file=sys.stderr)
        print(file=sys.stderr)

    metadata_path = failure_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        input_names = metadata.get("inputs", [])
    else:
        input_names = sorted(p.name for p in failure_dir.iterdir() if p.suffix == ".bin")

    if input_names:
        print("Inputs:", file=sys.stderr)
        for i, name in enumerate(input_names):
            p = failure_dir / name
            if p.exists():
                arr = np.fromfile(p, dtype=np.float64)
                print(f"  input {i} ({name}):", file=sys.stderr)
                print(f"    shape: {arr.shape}", file=sys.stderr)
                for line in np.array2string(arr, max_line_width=100).splitlines():
                    print(f"    {line}", file=sys.stderr)
                print(file=sys.stderr)

    error_path = failure_dir / "error.txt"
    if error_path.exists():
        print(f"Original error: {error_path.read_text().strip()}", file=sys.stderr)
        print(file=sys.stderr)


if __name__ == "__main__":
    main()
