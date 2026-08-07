# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Benchmark runners that measure runtime and compare against a reference.

Runs a command (eval, grad, …) repeatedly and computes the median
wall-clock time.  In ``--generate`` mode the reference median is saved
to ``reference_time.json`` inside the test directory.  In normal mode
the SUT median is compared to the reference: the test passes when

    sut_median <= reference_median * max_slowdown

test.json format::

    {
        "command": "bench_eval",
        "network": "network.mininn",
        "inputs": ["input.bin"],
        "n_repeats": 50,
        "max_slowdown": 2.0
    }

``n_repeats`` defaults to 30.  ``max_slowdown`` defaults to 2.0.
``n_warmup`` (default 3) controls how many untimed warm-up runs
precede the measured repetitions.
"""

import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from testrunner.commands.common import (
    build_eval_grad_cmd,
    get_timeout,
    parse_output_paths,
    run_subprocess,
)
from testrunner.commands.bounds import build_bounds_cmd
from testrunner.commands.verify import build_verify_cmd

REFERENCE_FILE = "reference_time.json"

# Per-mode SUT command builders. eval/grad share one builder; bounds and verify
# use the box/point input convention.
_SUT_BUILDERS = {
    "eval": build_eval_grad_cmd,
    "grad": build_eval_grad_cmd,
    "bounds": build_bounds_cmd,
    "verify": build_verify_cmd,
}


def run_bench_eval(
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
    return _run_benchmark(
        test_dir, config, output_dir, backend, backend_arg,
        mode="eval", generate=generate, output_handler=output_handler, closed=closed,
        extra_run_args=extra_run_args,
    )


def run_bench_grad(
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
    return _run_benchmark(
        test_dir, config, output_dir, backend, backend_arg,
        mode="grad", generate=generate, output_handler=output_handler, closed=closed,
        extra_run_args=extra_run_args,
    )


def run_bench_bounds(
    test_dir, config, output_dir, backend, backend_arg,
    generate=False, output_handler=None, closed=False, extra_run_args=(),
):
    return _run_benchmark(
        test_dir, config, output_dir, backend, backend_arg,
        mode="bounds", generate=generate, output_handler=output_handler, closed=closed,
        extra_run_args=extra_run_args,
    )


def run_bench_verify(
    test_dir, config, output_dir, backend, backend_arg,
    generate=False, output_handler=None, closed=False, extra_run_args=(),
):
    return _run_benchmark(
        test_dir, config, output_dir, backend, backend_arg,
        mode="verify", generate=generate, output_handler=output_handler, closed=closed,
        extra_run_args=extra_run_args,
    )


def _run_benchmark(
    test_dir, config, output_dir, backend, backend_arg, mode,
    generate=False, output_handler=None, closed=False, extra_run_args=(),
):
    n_repeats = config.get("n_repeats", 30)
    n_warmup = config.get("n_warmup", 3)
    max_slowdown = config.get("max_slowdown", 2.0)
    timeout = get_timeout(config)

    # Build the command once — all repetitions use the same arguments.
    # Use the underlying mode (eval/grad/bounds/verify) as the SUT subcommand,
    # not "bench_<mode>".
    sut_config = {**config, "command": mode}
    cmd, cwd = _SUT_BUILDERS[mode](
        sut_config, test_dir, output_dir, backend, backend_arg, extra_run_args=extra_run_args
    )

    # Warm-up runs (untimed, also validates the command works).
    for i in range(n_warmup):
        _clean_output_dir(output_dir)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"warm-up run {i + 1} timed out after {timeout}s"}
        if result.returncode != 0:
            return {
                "passed": False,
                "error": "command failed"
                if closed
                else f"warm-up run failed (exit {result.returncode}): {result.stderr.strip()}",
            }

    # Timed repetitions.
    times = []
    for i in range(n_repeats):
        _clean_output_dir(output_dir)
        start = time.perf_counter()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "error": f"repetition {i + 1}/{n_repeats} timed out after {timeout}s",
            }
        elapsed = time.perf_counter() - start

        if result.returncode != 0:
            return {
                "passed": False,
                "error": "command failed"
                if closed
                else (
                    f"repetition {i + 1}/{n_repeats} failed "
                    f"(exit {result.returncode}): {result.stderr.strip()}"
                ),
            }
        times.append(elapsed)

        if output_handler is not None:
            output_handler.bench_rep_finished(
                str(test_dir.name), i + 1, n_repeats, elapsed,
            )

    sut_median = statistics.median(times)

    if generate:
        ref = {"median": sut_median, "times": times, "n_repeats": n_repeats}
        (test_dir / REFERENCE_FILE).write_text(json.dumps(ref, indent=2))
        return {
            "passed": True,
            "error": None,
            "generated": True,
            "median_time": sut_median,
            "times": times,
        }

    # Load reference.
    ref_path = test_dir / REFERENCE_FILE
    if not ref_path.exists():
        return {
            "passed": False,
            "error": f"no reference file ({REFERENCE_FILE}); run with --generate first",
        }
    ref = json.loads(ref_path.read_text())
    ref_median = ref["median"]

    passed = sut_median <= ref_median * max_slowdown
    slowdown = sut_median / ref_median if ref_median > 0 else float("inf")

    return {
        "passed": passed,
        "error": None
        if passed
        else (
            f"too slow: {sut_median:.4f}s median vs {ref_median:.4f}s reference "
            f"({slowdown:.2f}x, limit {max_slowdown:.2f}x)"
        ),
        "median_time": sut_median,
        "reference_median": ref_median,
        "slowdown": round(slowdown, 4),
        "max_slowdown": max_slowdown,
        "times": times,
    }


def _clean_output_dir(output_dir):
    """Remove and recreate the output directory so each run starts clean."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
