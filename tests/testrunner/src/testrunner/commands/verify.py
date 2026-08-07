# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Command builder and custom runner for ``verify``.

A ``verify`` test asks the implementation to decide whether a network's single
scalar output (a *margin* with the property baked in) satisfies ``margin(x) >= 0``
for all ``x`` in an input box. The implementation is invoked exactly like
``bounds`` (inline ``box``/``point`` markers interleaved with file paths):

    <impl> verify --output-dir <dir> <network.mininn> box <lb.bin> <ub.bin>

Implementation output protocol (stdout):
  * The **last** line is the verdict, ``sat`` (property holds) or ``viol``
    (property violated). Arbitrary logging may precede it.
  * For ``viol`` the implementation writes one ``counterexample_<i>.bin`` per
    network input into ``--output-dir`` and prints their path(s) on the line(s)
    immediately before the verdict.

test.json fields::

    {
        "command": "verify",
        "network": "resources/net.mininn",
        "inputs": ["box", "resources/input_0_lb.bin", "resources/input_0_ub.bin"],
        "expected_verdict": "sat",   # or "viol"
        "box_atol": 1e-9,            # in-box slack for counterexample check
        "points": 10
    }

The runner checks the verdict against ``expected_verdict``. For a ``viol`` verdict
it additionally re-evaluates the network at the reported counterexample (via the
``eval`` command) and confirms the witness is inside the box and has ``margin < 0``
— any valid witness is accepted, not just a stored one.
"""

import shlex
import subprocess
from pathlib import Path

import numpy as np

from .common import (
    build_eval_grad_cmd,
    container_run_prefix,
    get_timeout,
    is_container_backend,
    parse_output_paths,
    run_subprocess,
)

INPUT_MARKERS = ("box", "point")
VERDICTS = ("sat", "viol")


def build_verify_cmd(
    config, test_dir, output_dir, backend, backend_arg, extra_run_args=(), command="verify"
):
    """Build the CLI command for ``verify``/``verify2`` (same shape as ``bounds``).

    *command* is the SUT subcommand to invoke (``verify`` or ``verify2``); the
    two share an identical input/output protocol and differ only in which
    verifier the implementation runs.
    """
    network = config["network"]
    inputs = config.get("inputs", [])

    if is_container_backend(backend):
        cmd = [
            *container_run_prefix(backend, test_dir, extra_run_args),
            backend_arg,
            command,
            "--output-dir",
            f"/data/{output_dir.relative_to(test_dir)}",
            f"/data/{network}",
            *[tok if tok in INPUT_MARKERS else f"/data/{tok}" for tok in inputs],
        ]
        return cmd, None
    cmd = [
        *shlex.split(backend_arg),
        command,
        "--output-dir",
        str(output_dir),
        str(test_dir / network),
        *[tok if tok in INPUT_MARKERS else str(test_dir / tok) for tok in inputs],
    ]
    return cmd, None


def _box_input_index_and_bounds(config, test_dir):
    """Return (box_position_among_inputs, lb, ub) for the single box input."""
    tokens = config.get("inputs", [])
    pos = -1
    input_index = -1
    i = 0
    while i < len(tokens):
        marker = tokens[i]
        input_index += 1
        if marker == "box":
            pos = input_index
            lb = np.fromfile(test_dir / tokens[i + 1], dtype=np.float64)
            ub = np.fromfile(test_dir / tokens[i + 2], dtype=np.float64)
            return pos, lb, ub
        elif marker == "point":
            i += 2
        else:
            i += 1
    return pos, None, None


def _eval_margin(cx_files, config, test_dir, output_dir, backend, backend_arg, extra_run_args, timeout):
    """Evaluate the network at the counterexample via the ``eval`` command.

    Returns (margin_array, error). ``cx_files`` are host paths under ``test_dir``,
    one per network input, in network-input order.
    """
    eval_out = output_dir / "cx_eval"
    eval_out.mkdir(parents=True, exist_ok=True)
    eval_config = {
        "command": "eval",
        "network": config["network"],
        "inputs": [str(p.relative_to(test_dir)) for p in cx_files],
    }
    cmd, cwd = build_eval_grad_cmd(
        eval_config, test_dir, eval_out, backend, backend_arg, extra_run_args=extra_run_args
    )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        return None, "counterexample eval timed out"
    if result.returncode != 0:
        return None, f"counterexample eval failed: {result.stderr.strip()}"
    out_files, _ = parse_output_paths(
        result.stdout, container_root=test_dir if is_container_backend(backend) else None
    )
    if not out_files:
        return None, "counterexample eval produced no output"
    return np.fromfile(out_files[0], dtype=np.float64), None


def run_verify_test(
    test_dir, config, output_dir, backend, backend_arg,
    generate=False, output_handler=None, closed=False, extra_run_args=(),
):
    command = config.get("command", "verify")
    cmd, cwd = build_verify_cmd(
        config, test_dir, output_dir, backend, backend_arg,
        extra_run_args=extra_run_args, command=command,
    )
    timeout = get_timeout(config)
    try:
        result = run_subprocess(
            cmd, cwd=cwd, timeout=timeout,
            log_file=None if closed else output_dir / "stdout.log",
            output_handler=output_handler,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"{command} timed out after {timeout}s"}
    if result.returncode != 0:
        return {
            "passed": False,
            "error": "command failed"
            if closed
            else f"{command} failed (exit {result.returncode}): {result.stderr.strip()}",
        }

    lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return {"passed": False, "error": f"{command} produced no output"}
    verdict = lines[-1]
    if verdict not in VERDICTS:
        return {
            "passed": False,
            "error": "command failed"
            if closed
            else f"last stdout line is not a verdict {VERDICTS}: {verdict!r}",
        }

    # Counterexample paths (viol only) are the line(s) just before the verdict.
    container_root = test_dir if is_container_backend(backend) else None
    cx_files, _ = parse_output_paths("\n".join(lines[:-1]), container_root=container_root)

    if generate:
        return {"passed": True, "error": None, "generated": True, "verdict": verdict}

    expected = config.get("expected_verdict")
    if expected not in VERDICTS:
        return {"passed": False, "error": f"test.json missing expected_verdict {VERDICTS}"}

    if verdict != expected:
        return {
            "passed": False,
            "error": "wrong verdict"
            if closed
            else f"verdict mismatch: got {verdict!r}, expected {expected!r}",
        }

    if verdict == "sat":
        return {"passed": True, "error": None, "verdict": verdict}

    # verdict == "viol": validate the counterexample independently.
    box_pos, lb, ub = _box_input_index_and_bounds(config, test_dir)
    if not cx_files:
        return {"passed": False, "error": "viol verdict but no counterexample file produced"}

    box_atol = float(config.get("box_atol", 1e-9))
    if 0 <= box_pos < len(cx_files) and lb is not None:
        cx = np.fromfile(cx_files[box_pos], dtype=np.float64)
        if cx.size != lb.size:
            return {"passed": False, "error": "counterexample size does not match box"}
        out_of_box = float(np.maximum(lb - box_atol - cx, cx - (ub + box_atol)).max())
        if out_of_box > 0:
            return {
                "passed": False,
                "error": "counterexample outside the input box"
                if closed
                else f"counterexample outside the input box by {out_of_box:.3e}",
            }

    margin, err = _eval_margin(
        cx_files, config, test_dir, output_dir, backend, backend_arg, extra_run_args,
        get_timeout({**config, "command": "eval"}),
    )
    if err is not None:
        return {"passed": False, "error": "command failed" if closed else err}

    margin_atol = float(config.get("margin_atol", 0.0))
    worst = float(margin.max())
    if worst >= margin_atol:
        return {
            "passed": False,
            "error": "counterexample does not violate the property"
            if closed
            else f"counterexample margin {worst:.3e} is not < {margin_atol:.1e}",
        }

    return {"passed": True, "error": None, "verdict": verdict}
