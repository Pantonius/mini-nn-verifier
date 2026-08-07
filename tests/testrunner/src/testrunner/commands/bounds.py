# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Command builder for ``bounds``.

The ``inputs`` field in ``test.json`` is a flat list interleaving inline
markers (``"box"``, ``"point"``) with file paths.  Markers are forwarded
verbatim; only the file-path entries are rewritten relative to the test
directory (or to ``/data`` for container backends).
"""

import shlex

from .common import container_run_prefix, is_container_backend


INPUT_MARKERS = ("box", "point")


def _rewrite_tokens(inputs, root):
    return [tok if tok in INPUT_MARKERS else f"{root}/{tok}" for tok in inputs]


def build_bounds_cmd(config, test_dir, output_dir, backend, backend_arg, extra_run_args=()):
    command = config["command"]
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
            *_rewrite_tokens(inputs, "/data"),
        ]
        return cmd, None
    else:
        cmd = [
            *shlex.split(backend_arg),
            command,
            "--output-dir",
            str(output_dir),
            str(test_dir / network),
            *[tok if tok in INPUT_MARKERS else str(test_dir / tok) for tok in inputs],
        ]
        return cmd, None
