# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Command builder for ``affine_bounds`` (CROWN affine output bounds).

The SUT invocation is identical to ``bounds`` — a single scalar-output network
plus inline ``box``/``point`` input markers — so this reuses the ``bounds``
builder verbatim. The implementation writes affine bound files
(``output_<i>_{lb,ub}_{weight,bias}.bin``) instead of constant interval bounds.
"""

from .bounds import build_bounds_cmd


def build_affine_bounds_cmd(config, test_dir, output_dir, backend, backend_arg, extra_run_args=()):
    return build_bounds_cmd(
        config, test_dir, output_dir, backend, backend_arg, extra_run_args=extra_run_args
    )
