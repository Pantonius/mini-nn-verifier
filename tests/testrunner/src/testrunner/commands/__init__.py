# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from .affine_bounds import build_affine_bounds_cmd
from .bounds import build_bounds_cmd
from .eval import build_eval_cmd
from .grad import build_grad_cmd
from .train import run_train_test
from .verify import run_verify_test

COMMANDS = {
    "eval": build_eval_cmd,
    "grad": build_grad_cmd,
    "bounds": build_bounds_cmd,
    "affine_bounds": build_affine_bounds_cmd,
}

# RUNNERS is populated lazily to break a circular import:
# commands/__init__ -> fuzz -> fuzz.runner -> commands.common -> commands/__init__
RUNNERS = {}

# Cheapest commands first. Tests are run in this order so that fast/cheap
# tests fail early before expensive ones are attempted.
# Append new commands at the position that matches their cost.
COMMAND_ORDER = [
    "eval", "grad", "bounds", "affine_bounds", "verify", "verify2", "train",
    "fuzz_eval", "fuzz_grad", "fuzz_bounds", "fuzz_affine_bounds",
    "bench_eval", "bench_grad", "bench_bounds", "bench_verify",
]


def command_sort_key(command: str) -> int:
    """Return a numeric priority for *command* (lower = run earlier).

    Unknown commands sort after all known ones, preserving alphabetical
    order among themselves.
    """
    try:
        return COMMAND_ORDER.index(command)
    except ValueError:
        return len(COMMAND_ORDER)


def _init_runners():
    if not RUNNERS:
        from testrunner.fuzz import (
            run_fuzz_eval, run_fuzz_grad, run_fuzz_bounds, run_fuzz_affine_bounds
        )
        from testrunner.benchmark import (
            run_bench_eval, run_bench_grad, run_bench_bounds, run_bench_verify
        )

        RUNNERS.update(
            {
                "train": run_train_test,
                "verify": run_verify_test,
                "verify2": run_verify_test,
                "fuzz_eval": run_fuzz_eval,
                "fuzz_grad": run_fuzz_grad,
                "fuzz_bounds": run_fuzz_bounds,
                "fuzz_affine_bounds": run_fuzz_affine_bounds,
                "bench_eval": run_bench_eval,
                "bench_grad": run_bench_grad,
                "bench_bounds": run_bench_bounds,
                "bench_verify": run_bench_verify,
            }
        )
