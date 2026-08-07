# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Command builder for grad."""

from .common import build_eval_grad_cmd


def build_grad_cmd(config, test_dir, output_dir, backend, backend_arg, extra_run_args=()):
    return build_eval_grad_cmd(
        config, test_dir, output_dir, backend, backend_arg, extra_run_args=extra_run_args
    )
