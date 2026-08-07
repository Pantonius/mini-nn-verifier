# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Shared helpers and command builders for eval and grad."""

import subprocess
import shlex
import threading
from pathlib import Path


DEFAULT_TIMEOUTS = {
    "eval": 60, "grad": 60, "bounds": 60, "affine_bounds": 120,
    "verify": 300, "verify2": 600, "train": 600,
    "fuzz_eval": 600, "fuzz_grad": 600, "fuzz_bounds": 600, "fuzz_affine_bounds": 600,
    "bench_eval": 600, "bench_grad": 600, "bench_bounds": 600, "bench_verify": 900,
}

# Backends that run the implementation inside a container (mounting the
# test directory at /data). ``docker`` and ``podman`` use identical CLIs
# for the features we need (``run --rm -v src:/data image cmd args``).
CONTAINER_BACKENDS = ("docker", "podman")


def is_container_backend(backend):
    """Return True if *backend* runs the SUT inside a container."""
    return backend in CONTAINER_BACKENDS


def container_run_prefix(backend, test_dir, extra_run_args=()):
    """Return the ``<exe> run --rm [extra] -v <test_dir>:/data`` prefix.

    *extra_run_args* is an optional sequence of additional ``run`` flags
    (e.g. ``--network=none``, ``--memory=1g``) placed before the image
    argument.  Paths inside the container are always rewritten to
    ``/data/<relative>`` by callers.
    """
    return [
        backend,
        "run",
        "--rm",
        *extra_run_args,
        "-v",
        f"{Path(test_dir).resolve()}:/data",
    ]


def get_timeout(config):
    """Return timeout in seconds from config, falling back to per-command defaults."""
    if "timeout" in config:
        return config["timeout"]
    return DEFAULT_TIMEOUTS.get(config.get("command", ""), 60)


def parse_output_paths(stdout, container_root=None):
    """Parse file paths from subprocess stdout.

    When running inside a container, the student implementation prints
    in-container paths (rooted at ``/data``) since it has no knowledge of
    the host-side bind-mount source. *container_root*, when provided,
    maps ``/data`` -> that host path so the resulting :class:`Path` can be
    resolved on the host.

    Returns (existing_paths, warnings) where warnings list non-existent paths.
    """
    paths = []
    warnings = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        path_str = line
        if container_root is not None:
            if path_str == "/data":
                path_str = str(container_root)
            elif path_str.startswith("/data/"):
                path_str = str(Path(container_root) / path_str[len("/data/"):])
        p = Path(path_str)
        if p.exists():
            paths.append(p)
        else:
            warnings.append(f"ignoring non-existent output path: {line}")
    return paths, warnings


class SubprocessResult:
    """Drop-in replacement for subprocess.CompletedProcess."""

    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_subprocess(cmd, cwd=None, timeout=60, log_file=None, output_handler=None):
    """Run a command with line-by-line stdout streaming.

    Streams stdout to *log_file* (a Path) and *output_handler* (via
    ``test_stdout_line``) while accumulating the full text so that
    ``parse_output_paths`` can be called on it afterwards.

    Returns a :class:`SubprocessResult` compatible with
    ``subprocess.CompletedProcess``.  Raises ``subprocess.TimeoutExpired``
    on timeout (after killing the process).
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)

    # Drain stderr in a background thread to avoid deadlocks
    stderr_chunks = []

    def _drain_stderr():
        for line in proc.stderr:
            stderr_chunks.append(line)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    # Timeout via a timer that kills the process
    timed_out = threading.Event()

    def _kill():
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, _kill)
    timer.daemon = True
    timer.start()

    # Read stdout line by line
    stdout_lines = []
    log_fh = None
    try:
        if log_file is not None:
            log_fh = open(log_file, "w")
        for line in proc.stdout:
            stripped = line.rstrip("\n")
            stdout_lines.append(stripped)
            if log_fh is not None:
                log_fh.write(line)
                log_fh.flush()
            if output_handler is not None:
                output_handler.test_stdout_line(stripped)
    finally:
        if log_fh is not None:
            log_fh.close()

    proc.wait()
    timer.cancel()
    stderr_thread.join(timeout=5)

    if timed_out.is_set():
        raise subprocess.TimeoutExpired(cmd, timeout)

    stdout_text = "\n".join(stdout_lines)
    stderr_text = "".join(stderr_chunks)
    return SubprocessResult(proc.returncode, stdout_text, stderr_text)


def build_eval_grad_cmd(config, test_dir, output_dir, backend, backend_arg, extra_run_args=()):
    """Shared builder for eval and grad — same argument structure."""
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
            *[f"/data/{i}" for i in inputs],
        ]
        return cmd, None
    else:
        cmd = [
            *shlex.split(backend_arg),
            command,
            "--output-dir",
            str(output_dir),
            *[str(test_dir / i) for i in [network, *inputs]],
        ]
        return cmd, None
