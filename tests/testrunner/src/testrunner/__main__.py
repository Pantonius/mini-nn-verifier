# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Test runner for mininnverifier-compatible implementations.

Usage:
    python -m testrunner docker <image> <test_dir>
    python -m testrunner local "<command>" <test_dir>
    python -m testrunner docker <image> <test_dir> --generate
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from testrunner.commands import COMMANDS, RUNNERS, _init_runners, command_sort_key
from testrunner.commands.common import (
    get_timeout,
    is_container_backend,
    parse_output_paths,
    run_subprocess,
)
from testrunner.check import CHECKS, DEFAULT_CHECKS
from testrunner.output import CliOutputHandler, JsonOutputHandler
from testrunner.scoring import compute_score


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _attach_score(config, result):
    """Add score, max_points, bonus, and max_bonus to a test result if configured."""
    scoring = compute_score(config, result)
    if scoring is not None:
        result.update(scoring)
    return result


def is_closed(test_dir, config):
    """Return True if this test is marked as closed (confidential)."""
    return "closed" in str(test_dir) or config.get("access") == "closed"


def run_single_test(
    test_dir, backend, backend_arg, generate=False, output_handler=None, extra_run_args=()
):
    """Run a single test from a directory containing test.json."""
    config = json.loads((test_dir / "test.json").read_text())

    command = config["command"]
    closed = is_closed(test_dir, config)

    output_dir = test_dir / "actual"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    # Custom runners handle the full test lifecycle
    _init_runners()
    if command in RUNNERS:
        test_result = RUNNERS[command](
            test_dir,
            config,
            output_dir,
            backend,
            backend_arg,
            generate=generate,
            output_handler=output_handler,
            closed=closed,
            extra_run_args=extra_run_args,
        )
        return _attach_score(config, test_result)

    if command not in COMMANDS:
        return {"passed": False, "error": f"unknown command: {command}"}

    cmd, cwd = COMMANDS[command](
        config, test_dir, output_dir, backend, backend_arg, extra_run_args=extra_run_args
    )
    timeout = get_timeout(config)
    try:
        result = run_subprocess(
            cmd,
            cwd=cwd,
            timeout=timeout,
            log_file=None if closed else output_dir / "stdout.log",
            output_handler=output_handler,
        )
    except subprocess.TimeoutExpired:
        return _attach_score(
            config, {"passed": False, "output_files": [], "error": f"command timed out after {timeout}s"}
        )

    if result.returncode != 0:
        return _attach_score(
            config,
            {
                "passed": False,
                "output_files": [],
                "error": "command failed"
                if closed
                else f"command failed (exit {result.returncode}): {result.stderr.strip()}",
            },
        )

    output_files, warnings = parse_output_paths(
        result.stdout,
        container_root=test_dir if is_container_backend(backend) else None,
    )

    if generate:
        for out_file in output_files:
            dest = test_dir / f"expected_{out_file.name}"
            shutil.copy2(out_file, dest)
        return {
            "passed": True,
            "error": None,
            "generated": True,
            "output_files": [str(f) for f in output_files],
            "warnings": warnings,
        }

    check_name = config.get("check", DEFAULT_CHECKS.get(command))
    if check_name is None:
        return _attach_score(
            config,
            {
                "passed": False,
                "error": f"no check specified and no default for command '{command}'",
            },
        )
    if check_name not in CHECKS:
        return _attach_score(config, {"passed": False, "error": f"unknown check: {check_name}"})

    check_result = CHECKS[check_name](test_dir, config, output_files, closed=closed)
    return _attach_score(
        config,
        {
            "passed": check_result["passed"],
            "output_files": [str(f) for f in output_files],
            "error": check_result["error"],
            "warnings": warnings,
        },
    )


def run_tests(
    root_dir, backend, backend_arg, generate=False, output_handler=None, extra_run_args=()
):
    """Discover and run all tests under root_dir.

    Returns a flat list of (test_path, result_dict) tuples.
    """
    root_dir = Path(root_dir).resolve()

    # Discover all test directories and sort by command cost (cheap first)
    test_dirs = sorted(p.parent for p in root_dir.rglob("test.json"))
    if not test_dirs:
        if output_handler is not None:
            output_handler.all_finished(0, 0)
        return []

    def _sort_key(td):
        cfg = json.loads((td / "test.json").read_text())
        return (command_sort_key(cfg["command"]), td)

    test_dirs.sort(key=_sort_key)

    total = len(test_dirs)
    results = []
    n_passed = 0
    n_failed = 0
    total_score = 0.0
    total_max_points = 0.0
    total_bonus = 0.0
    total_max_bonus = 0.0

    for i, test_dir in enumerate(test_dirs):
        rel = test_dir.relative_to(root_dir)
        test_path = test_dir.name if str(rel) == "." else str(rel)

        if output_handler is not None:
            output_handler.test_starting(test_path, i + 1, total)

        test_result = run_single_test(
            test_dir,
            backend,
            backend_arg,
            generate,
            output_handler=output_handler,
            extra_run_args=extra_run_args,
        )
        results.append((test_path, test_result))

        if test_result.get("passed", False):
            n_passed += 1
        else:
            n_failed += 1

        if "score" in test_result:
            total_score += test_result["score"]
            total_max_points += test_result["max_points"]
        if "bonus" in test_result:
            total_bonus += test_result["bonus"]
            total_max_bonus += test_result["max_bonus"]

        if output_handler is not None:
            output_handler.test_finished(test_path, test_result)

    if output_handler is not None:
        output_handler.all_finished(
            n_passed, n_failed, total_score, total_max_points, total_bonus, total_max_bonus
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run a mininnverifier test.")
    parser.add_argument("backend", choices=["docker", "podman", "local"])
    parser.add_argument(
        "backend_arg", type=str, help="Container image name (docker/podman) or local command"
    )
    parser.add_argument("test_dir", type=str)
    parser.add_argument(
        "--generate", action="store_true", help="Generate expected outputs instead of checking"
    )
    parser.add_argument(
        "--output",
        choices=["cli", "json"],
        default="cli",
        help="Output mode: cli (default) for interactive display, json for JSONL",
    )
    parser.add_argument(
        "--extra-run-args",
        type=str,
        default="",
        help=(
            "Extra arguments passed through to 'docker run'/'podman run' "
            "(e.g. '--network=none --memory=1g --read-only'). "
            "Shell-quoted; ignored for the 'local' backend."
        ),
    )
    args = parser.parse_args()

    if args.output == "json":
        handler = JsonOutputHandler()
    else:
        handler = CliOutputHandler()

    extra_run_args = tuple(shlex.split(args.extra_run_args)) if args.extra_run_args else ()

    results = run_tests(
        args.test_dir,
        args.backend,
        args.backend_arg,
        args.generate,
        output_handler=handler,
        extra_run_args=extra_run_args,
    )
    if not results and args.output == "cli":
        print(f"No tests found in {args.test_dir}", file=sys.stderr)
    if any(not r["passed"] for _, r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
