# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Show test inputs, expected outputs, and actual outputs for a test directory.

Usage:
    python -m testrunner.show <test_dir>
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

_DIM = "\033[2m"
_RESET = "\033[0m"


def _print_bin(path, label, indent="  "):
    """Load a float64 .bin file and print its shape and values."""
    arr = np.fromfile(path, dtype=np.float64)
    print(f"{indent}{label} ({path.name}):")
    print(f"{indent}  shape: {arr.shape}")
    # np.array2string truncates large arrays automatically
    for line in np.array2string(arr, max_line_width=100).splitlines():
        print(f"{indent}  {line}")
    print()


def _print_network(path):
    """Print the graph.txt from a .mininn ZIP file."""
    print(f"  Network ({path}):")
    with zipfile.ZipFile(path) as z:
        graph = z.read("graph.txt").decode()
    for line in graph.strip().splitlines():
        print(f"    {line}")
    print()


def _show_eval_grad(test_dir, config):
    command = config["command"]
    tolerance = config.get("tolerance", 1e-4)
    network = config.get("network")
    inputs = config.get("inputs", [])
    expected_outputs = config.get("expected_outputs", [])

    print(f"Command: {command}")
    print(f"Tolerance: {tolerance:.1e}")
    print()

    if network:
        _print_network(test_dir / network)

    if inputs:
        print("Inputs:")
        for i, rel in enumerate(inputs):
            if rel in ("box", "point"):
                print(f"  marker: {rel}")
                continue
            _print_bin(test_dir / rel, f"input {i}")

    if expected_outputs:
        print("Expected outputs:")
        for i, rel in enumerate(expected_outputs):
            p = test_dir / rel
            if p.exists():
                _print_bin(p, f"output {i}")
            else:
                print(f"  output {i} ({rel}): [not found]")
                print()

    actual_dir = test_dir / "actual"
    if actual_dir.exists():
        actual_bins = sorted(actual_dir.glob("*.bin"))
        if actual_bins:
            print("Actual outputs:")
            for i, p in enumerate(actual_bins):
                _print_bin(p, f"output {i}")


def _print_box_inputs(test_dir, inputs):
    """Print the interleaved ``box``/``point`` input spec used by bounds/verify.

    ``inputs`` is a flat list of markers (``"box"``, ``"point"``) interleaved
    with file paths: ``"box"`` consumes the next two paths (lb, ub), ``"point"``
    the next one.
    """
    print("Inputs:")
    i = 0
    idx = 0
    while i < len(inputs):
        marker = inputs[i]
        if marker == "box":
            print(f"  input {idx}: box")
            _print_bin(test_dir / inputs[i + 1], f"input {idx} lb", indent="    ")
            _print_bin(test_dir / inputs[i + 2], f"input {idx} ub", indent="    ")
            i += 3
        elif marker == "point":
            print(f"  input {idx}: point")
            _print_bin(test_dir / inputs[i + 1], f"input {idx}", indent="    ")
            i += 2
        else:
            _print_bin(test_dir / inputs[i], f"input {idx}")
            i += 1
        idx += 1


def _show_affine_bounds(test_dir, config):
    command = config["command"]
    network = config.get("network")
    inputs = config.get("inputs", [])

    print(f"Command: {command}")
    if "input_shape" in config:
        print(f"Input shape: {config['input_shape']}")
    if "output_shapes" in config:
        print(f"Output shapes: {config['output_shapes']}")
    print(f"Sample atol: {float(config.get('sample_atol', 1e-9)):.1e}")
    if config.get("reference_lb") and config.get("reference_ub"):
        print(f"Tightness factor: {float(config.get('tightness_factor', 1.5))}")
    print()

    if network:
        _print_network(test_dir / network)

    if inputs:
        _print_box_inputs(test_dir, inputs)

    for label, key in (("Sample inputs", "sample_inputs"), ("Sample outputs", "sample_outputs")):
        paths = config.get(key, [])
        if paths:
            print(f"{label}:")
            for i, rel in enumerate(paths):
                _print_bin(test_dir / rel, f"{key} {i}")

    for label, key in (("Reference lower bounds", "reference_lb"),
                        ("Reference upper bounds", "reference_ub")):
        paths = config.get(key, [])
        if paths:
            print(f"{label}:")
            for i, rel in enumerate(paths):
                _print_bin(test_dir / rel, f"{key} {i}")

    actual_dir = test_dir / "actual"
    if actual_dir.exists():
        actual_bins = sorted(actual_dir.glob("*.bin"))
        if actual_bins:
            print("Actual affine bounds:")
            for p in actual_bins:
                _print_bin(p, p.stem)


def _show_verify(test_dir, config):
    command = config["command"]
    network = config.get("network")
    inputs = config.get("inputs", [])

    print(f"Command: {command}")
    print(f"Expected verdict: {config.get('expected_verdict', '[not set]')}")
    print(f"Box atol: {float(config.get('box_atol', 1e-9)):.1e}")
    print(f"Margin atol: {float(config.get('margin_atol', 0.0)):.1e}")
    print()

    if network:
        _print_network(test_dir / network)

    if inputs:
        _print_box_inputs(test_dir, inputs)

    actual_dir = test_dir / "actual"
    if actual_dir.exists():
        cx_bins = sorted(actual_dir.glob("counterexample_*.bin"))
        if cx_bins:
            print("Counterexamples:")
            for i, p in enumerate(cx_bins):
                _print_bin(p, f"counterexample {i}")


def _show_train(test_dir, config):
    print("Command: train")
    skip = {"command"}
    for key, value in config.items():
        if key not in skip:
            print(f"  {key}: {value}")


def _show_fuzz(test_dir, config):
    command = config["command"]
    n_trials = config.get("n_trials", 100)
    seed = config.get("seed")

    print(f"Command: {command}")
    print(f"Trials: {n_trials}")
    if seed is not None:
        print(f"Seed: {seed}")
    print()

    failures_dir = test_dir / "actual" / "fuzz_failures"
    if not failures_dir.exists():
        print("No saved fuzz failures (run the test first, or all trials passed).")
        return

    case_dirs = sorted(
        (p for p in failures_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not case_dirs:
        print("No saved fuzz failures.")
        return

    print(f"Saved failing cases: {len(case_dirs)}")
    print()

    for case_dir in case_dirs:
        print(f"--- Failure {case_dir.name} ---")

        error_file = case_dir / "error.txt"
        if error_file.exists():
            print(f"  Error: {error_file.read_text().strip()}")
            print()

        stderr_file = case_dir / "stderr.txt"
        if stderr_file.exists():
            stderr = stderr_file.read_text().strip()
            if stderr:
                print(f"  {_DIM}--- stderr ---{_RESET}")
                for line in stderr.splitlines():
                    print(f"    {line}")
                print(f"  {_DIM}--- end stderr ---{_RESET}")
                print()

        network = case_dir / "network.mininn"
        if network.exists():
            _print_network(network)

        input_bins = sorted(p for p in case_dir.iterdir() if p.suffix == ".bin")
        if input_bins:
            print("  Inputs:")
            for i, p in enumerate(input_bins):
                _print_bin(p, f"input {i}", indent="    ")


def _show_test(test_dir):
    config = json.loads((test_dir / "test.json").read_text())
    command = config.get("command", "unknown")

    print(f"=== {test_dir} ===")
    print()

    if command in ("eval", "grad", "bounds"):
        _show_eval_grad(test_dir, config)
    elif command == "affine_bounds":
        _show_affine_bounds(test_dir, config)
    elif command in ("verify", "verify2"):
        _show_verify(test_dir, config)
    elif command in ("fuzz_eval", "fuzz_grad", "fuzz_bounds", "fuzz_affine_bounds"):
        _show_fuzz(test_dir, config)
    elif command == "train":
        _show_train(test_dir, config)
    else:
        print(f"Command: {command}")
        print("(no detailed display for this command type)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Show test inputs and expected/actual outputs.")
    parser.add_argument("test_dir", type=str)
    args = parser.parse_args()

    root = Path(args.test_dir).resolve()

    if (root / "test.json").exists():
        _show_test(root)
    else:
        test_dirs = sorted(p.parent for p in root.rglob("test.json"))
        if not test_dirs:
            print(f"No tests found under {root}")
            return
        for td in test_dirs:
            _show_test(td)


if __name__ == "__main__":
    main()
