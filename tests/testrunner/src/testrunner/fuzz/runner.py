# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Fuzz test runners for eval and grad commands.

Each runner generates random compute graphs, runs the SUT, and checks
for crashes, NaN/Inf, and shape correctness.

The test.json for each fuzz test specifies which primitives to use and
whether to check for NaN/Inf in the output:

    {
        "command": "fuzz_eval",
        "n_trials": 500,
        "seed": 0,
        "primitives": "safe",
        "check_nan_inf": true
    }

``primitives`` can be ``"all"`` (default), ``"safe"`` (excludes
log/sqrt/reciprocal), or an explicit list of primitive names.
``check_nan_inf`` defaults to ``false``.
"""

import hashlib
import json as json_mod
import shlex
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from hypothesis import (
    given,
    settings,
    HealthCheck,
    Phase,
    seed as hypothesis_seed,
    strategies as st,
)
from hypothesis.extra.numpy import arrays as np_arrays

from testrunner.commands.common import (
    container_run_prefix,
    get_timeout,
    is_container_backend,
    parse_output_paths,
)
from .graph_builder import (
    generate_graph,
    scalarize_graph,
    serialize_graph,
    ALL_PRIMITIVES,
    SAFE_PRIMITIVES,
)

PRIMITIVE_SETS = {"all": ALL_PRIMITIVES, "safe": SAFE_PRIMITIVES}


def resolve_primitives(spec):
    """Resolve a primitives spec from test.json to a list of primitive names.

    ``spec`` can be a string key (``"all"``, ``"safe"``) or an explicit
    list of primitive names.  Raises ``ValueError`` on unknown keys or
    invalid primitive names.
    """
    if isinstance(spec, str):
        if spec not in PRIMITIVE_SETS:
            raise ValueError(
                f"unknown primitive set {spec!r}, "
                f"expected one of {sorted(PRIMITIVE_SETS)} or an explicit list"
            )
        return PRIMITIVE_SETS[spec]
    if isinstance(spec, list):
        unknown = set(spec) - set(ALL_PRIMITIVES)
        if unknown:
            raise ValueError(f"unknown primitives: {sorted(unknown)}")
        return spec
    raise ValueError(f"primitives must be a string or list, got {type(spec).__name__}")


def run_fuzz_eval(
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
    """Fuzz runner for eval command."""
    if generate:
        import warnings

        warnings.warn("--generate has no effect for fuzz tests")
        return {"passed": True, "error": None, "generated": False}
    return _run_fuzz(
        test_dir,
        config,
        output_dir,
        backend,
        backend_arg,
        mode="eval",
        output_handler=output_handler,
        closed=closed,
        extra_run_args=extra_run_args,
    )


def run_fuzz_grad(
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
    """Fuzz runner for grad command."""
    if generate:
        import warnings

        warnings.warn("--generate has no effect for fuzz tests")
        return {"passed": True, "error": None, "generated": False}
    return _run_fuzz(
        test_dir,
        config,
        output_dir,
        backend,
        backend_arg,
        mode="grad",
        output_handler=output_handler,
        closed=closed,
        extra_run_args=extra_run_args,
    )


def run_fuzz_bounds(
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
    """Fuzz runner for the bounds command.

    Each trial generates a random graph and a random center input per invar.
    The center is widened by a fuzzed half-width ``eps`` — drawn per trial from
    ``[eps_min, eps_max]`` (default ``[0.0, 0.5]``; configurable in
    ``test.json``) — into an axis-aligned box.  The bounds CLI is invoked with
    one ``box`` marker per invar, and the impl's lb/ub output pairs are
    validated structurally (count, shape, lb <= ub, and optionally finiteness
    if ``check_nan_inf`` is set) and for **soundness**: random points inside the
    box are run through the ``eval`` command and must fall within the reported
    bounds (``n_eval_samples``, ``sample_atol``, ``sample_rtol`` configurable).
    """
    if generate:
        import warnings

        warnings.warn("--generate has no effect for fuzz tests")
        return {"passed": True, "error": None, "generated": False}
    return _run_fuzz(
        test_dir,
        config,
        output_dir,
        backend,
        backend_arg,
        mode="bounds",
        output_handler=output_handler,
        closed=closed,
        extra_run_args=extra_run_args,
    )


def run_fuzz_affine_bounds(
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
    """Fuzz runner for the ``affine_bounds`` (CROWN) command.

    Like ``fuzz_bounds``, but each trial generates a random *single-input,
    scalar-output* graph (the form CROWN's ``affine_bounds`` entry point
    requires) and the impl returns an affine relaxation ``lb_weight @ x + lb_bias
    <= f(x) <= ub_weight @ x + ub_bias`` rather than a constant interval.  The
    box half-width ``eps`` is fuzzed per trial from ``[eps_min, eps_max]``.  The
    affine bounds are validated structurally (four files, sizes, ``lb_aff <=
    ub_aff`` over the box, optionally finiteness) and for **soundness**: random
    points inside the box are run through ``eval`` and their output must lie
    between the lower and upper affine bounds.  As with ``fuzz_bounds``, ``dot``
    is never a bilinear application of two bounded operands.
    """
    if generate:
        import warnings

        warnings.warn("--generate has no effect for fuzz tests")
        return {"passed": True, "error": None, "generated": False}
    return _run_fuzz(
        test_dir,
        config,
        output_dir,
        backend,
        backend_arg,
        mode="affine_bounds",
        output_handler=output_handler,
        closed=closed,
        extra_run_args=extra_run_args,
    )


@st.composite
def _graph_with_inputs(
    draw, primitives=None, eps_range=None, allow_bilinear_dot=True, scalar=False
):
    """Composite strategy that generates a graph and matching input arrays.

    When ``eps_range`` is given as ``(lo, hi)``, an ``eps`` half-width is
    drawn last and returned as the third tuple element; otherwise ``eps`` is
    ``None``.  Drawing ``eps`` only when requested keeps the graph/input draw
    sequence byte-for-byte identical for the ``eval``/``grad`` fuzz runners.

    ``allow_bilinear_dot`` is forwarded to ``generate_graph``; the bounds
    runner sets it to ``False`` so ``dot`` is never applied to two bounded
    operands (which IBP cannot bound).

    ``scalar`` (set by the ``affine_bounds`` runner) forces a single input and
    reduces the output to a scalar, the form CROWN's ``affine_bounds`` entry
    point requires.
    """
    graph = generate_graph(
        draw,
        primitives=primitives,
        allow_bilinear_dot=allow_bilinear_dot,
        n_inputs=1 if scalar else None,
    )
    if scalar:
        scalarize_graph(graph)
    inputs = {}
    for var in graph.invars:
        data = draw(
            np_arrays(
                dtype=np.float64,
                shape=var.shape,
                elements=st.floats(
                    min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False
                ),
            )
        )
        inputs[var.name] = data
    eps = None
    if eps_range is not None:
        lo, hi = eps_range
        eps = draw(
            st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)
        )
    return graph, inputs, eps


def _run_fuzz(
    test_dir,
    config,
    output_dir,
    backend,
    backend_arg,
    mode,
    output_handler=None,
    closed=False,
    extra_run_args=(),
):
    n_trials = config.get("n_trials", 100)
    timeout = get_timeout(config)
    primitives = resolve_primitives(config.get("primitives", "all"))
    check_nan_inf = config.get("check_nan_inf", False)
    is_bounds = mode in ("bounds", "affine_bounds")
    eps = float(config.get("eps", 0.05)) if is_bounds else None

    # Bounds soundness: the input box half-width ``eps`` is itself fuzzed,
    # drawn per trial from ``[eps_min, eps_max]``.  After computing bounds we
    # sample random points inside the box, run ``eval`` on each, and check the
    # bounds contain the concrete outputs (SUT bounds ⊇ SUT eval).
    eps_range = None
    sound_cfg = {}
    if is_bounds:
        eps_min = float(config.get("eps_min", 0.0))
        eps_max = float(config.get("eps_max", config.get("eps", 0.5)))
        eps_range = (eps_min, eps_max)
        sound_cfg = {
            "n_eval_samples": int(config.get("n_eval_samples", 3)),
            "sample_atol": float(config.get("sample_atol", 1e-6)),
            "sample_rtol": float(config.get("sample_rtol", 1e-6)),
        }

    # For open tests, save up to MAX_SAVED failing cases to disk,
    # deduplicated by error message (one example per distinct error).
    # When multiple trials hit the same error, keep the smallest example
    # (by serialized network length + total input size).
    MAX_SAVED = 10
    save_dir = None
    saved_count = [0]
    # error -> (saved_path, size)
    saved_errors = {}
    if not closed:
        save_dir = output_dir / "fuzz_failures"
        save_dir.mkdir(exist_ok=True)

    results = []
    trial_counter = [0]
    test_path = str(test_dir.name)

    @given(
        data=_graph_with_inputs(
            primitives=primitives,
            eps_range=eps_range,
            allow_bilinear_dot=not is_bounds,
            scalar=(mode == "affine_bounds"),
        )
    )
    @settings(
        max_examples=n_trials,
        database=None,
        deadline=None,
        phases=[Phase.explicit, Phase.generate],
        suppress_health_check=list(HealthCheck),
    )
    def fuzz_trial(data):
        graph, inputs, eps = data
        # Always attempt to save when save_dir is set — the dedup
        # logic below will discard or replace as needed.
        trial_save_dir = save_dir
        result = _run_single_trial(
            graph,
            inputs,
            output_dir,
            backend,
            backend_arg,
            mode,
            check_nan_inf=check_nan_inf,
            timeout=timeout,
            save_dir=trial_save_dir,
            extra_run_args=extra_run_args,
            eps=eps,
            sound_cfg=sound_cfg,
        )
        if not result["passed"] and result.get("saved_to"):
            trial_size = len(serialize_graph(graph)) + sum(arr.nbytes for arr in inputs.values())
            prev = saved_errors.get(result["error"])
            if prev is not None and trial_size < prev[1]:
                # Smaller example for the same error — replace
                shutil.rmtree(prev[0], ignore_errors=True)
                saved_errors[result["error"]] = (result["saved_to"], trial_size)
            elif prev is not None:
                # Duplicate but not smaller — discard
                shutil.rmtree(result["saved_to"], ignore_errors=True)
                result["saved_to"] = None
            elif saved_count[0] < MAX_SAVED:
                # New error and under the cap — keep
                saved_errors[result["error"]] = (result["saved_to"], trial_size)
                saved_count[0] += 1
            else:
                # New error but already at cap — discard
                shutil.rmtree(result["saved_to"], ignore_errors=True)
                result["saved_to"] = None
        results.append(result)
        trial_counter[0] += 1
        if output_handler is not None:
            output_handler.fuzz_trial_finished(
                test_path, trial_counter[0], n_trials, result["passed"], result["error"]
            )

    if "seed" in config:
        fuzz_trial = hypothesis_seed(config["seed"])(fuzz_trial)

    try:
        fuzz_trial()
    except Exception:
        pass

    n_passed = sum(1 for r in results if r["passed"])
    n_failed = len(results) - n_passed

    result = {
        "passed": n_failed == 0,
        "error": None if n_failed == 0 else f"{n_failed} trial(s) failed",
        "n_trials": len(results),
        "n_passed": n_passed,
    }

    if not closed:
        failures = [
            {"trial": i, "error": r["error"]} for i, r in enumerate(results) if not r["passed"]
        ]
        result["failures"] = failures

        # Build grouped failure summary: error message -> count
        error_counts = Counter(r["error"] for r in results if not r["passed"])
        if error_counts:
            result["failure_summary"] = dict(error_counts.most_common())

        if save_dir is not None and saved_count[0] > 0:
            result["saved_failures_dir"] = str(save_dir)

    return result


def _run_single_trial(
    graph,
    inputs,
    output_dir,
    backend,
    backend_arg,
    mode,
    check_nan_inf=True,
    timeout=60,
    save_dir=None,
    extra_run_args=(),
    eps=None,
    sound_cfg=None,
):
    """Run a single fuzz trial.

    Args:
        save_dir: If not None and the trial fails, copy the network and
            input files to a subdirectory of save_dir for reproduction.
        eps: Half-width of the input box for ``mode="bounds"``.
        sound_cfg: Soundness-sampling config for ``mode="bounds"``
            (``n_eval_samples``, ``sample_atol``, ``sample_rtol``).
    """
    sound_cfg = sound_cfg or {}
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Write .mininn file
        network_path = tmp_dir / "network.mininn"
        network_path.write_bytes(serialize_graph(graph))

        if mode in ("bounds", "affine_bounds"):
            box_paths = []
            for var in graph.invars:
                center = inputs[var.name]
                lb = (center - eps).astype(np.float64)
                ub = (center + eps).astype(np.float64)
                lb_path = tmp_dir / f"{var.name}_lb.bin"
                ub_path = tmp_dir / f"{var.name}_ub.bin"
                lb.tofile(lb_path)
                ub.tofile(ub_path)
                box_paths.append((lb_path, ub_path))

            expected_shapes = [v.shape for v in graph.outvars]
            input_paths_for_save = [p for pair in box_paths for p in pair]
            if mode == "affine_bounds":
                result = run_and_check_affine_bounds(
                    network_path,
                    box_paths,
                    backend,
                    backend_arg,
                    graph.invars[0].shape,
                    check_nan_inf=check_nan_inf,
                    timeout=timeout,
                    extra_run_args=extra_run_args,
                    n_eval_samples=sound_cfg.get("n_eval_samples", 3),
                    sample_atol=sound_cfg.get("sample_atol", 1e-6),
                    sample_rtol=sound_cfg.get("sample_rtol", 1e-6),
                )
            else:
                result = run_and_check_bounds(
                    network_path,
                    box_paths,
                    backend,
                    backend_arg,
                    expected_shapes,
                    check_nan_inf=check_nan_inf,
                    timeout=timeout,
                    extra_run_args=extra_run_args,
                    n_eval_samples=sound_cfg.get("n_eval_samples", 3),
                    sample_atol=sound_cfg.get("sample_atol", 1e-6),
                    sample_rtol=sound_cfg.get("sample_rtol", 1e-6),
                )
        else:
            # Write input .bin files
            input_paths = []
            for var in graph.invars:
                input_path = tmp_dir / f"{var.name}.bin"
                inputs[var.name].tofile(input_path)
                input_paths.append(input_path)

            # Determine expected shapes
            if mode == "eval":
                expected_shapes = [v.shape for v in graph.outvars]
            else:
                expected_shapes = [v.shape for v in graph.invars]

            input_paths_for_save = input_paths
            result = run_and_check(
                network_path,
                input_paths,
                backend,
                backend_arg,
                mode,
                expected_shapes,
                check_nan_inf=check_nan_inf,
                timeout=timeout,
                extra_run_args=extra_run_args,
            )

        if not result["passed"] and save_dir is not None:
            result["saved_to"] = _save_failure(
                save_dir,
                network_path,
                input_paths_for_save,
                mode=mode,
                check_nan_inf=check_nan_inf,
                expected_shapes=expected_shapes,
                error=result.get("error"),
                stderr=result.get("stderr"),
                sound_cfg=sound_cfg if mode in ("bounds", "affine_bounds") else None,
            )

    return result


def run_and_check(
    network_path,
    input_paths,
    backend,
    backend_arg,
    mode,
    expected_shapes,
    check_nan_inf=True,
    timeout=60,
    extra_run_args=(),
):
    """Run the SUT on a network/inputs and check the output.

    This is the core fuzz-checking logic, usable both from fuzz trials
    and from reproduction of saved failures.  The network and input files
    must already exist on disk.  An ``output`` subdirectory is created
    next to the network file for SUT output.

    Returns a result dict with at least ``passed`` and ``error`` keys.
    """
    network_path = Path(network_path)
    input_paths = [Path(p) for p in input_paths]
    work_dir = network_path.parent

    cmd = _build_cmd(
        mode,
        network_path,
        input_paths,
        work_dir / "output",
        backend,
        backend_arg,
        work_dir,
        extra_run_args=extra_run_args,
    )
    (work_dir / "output").mkdir(exist_ok=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        error = f"timed out after {timeout}s"
        return {"passed": False, "error": error}

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        error = f"crash (exit {result.returncode}): {_truncate(stderr)}"
        return {"passed": False, "error": error, "stdout": stdout, "stderr": stderr}

    # Parse output file paths from stdout. In container mode the SUT prints
    # in-container paths rooted at /data; map them back to the host work_dir.
    output_files, _ = parse_output_paths(
        result.stdout,
        container_root=work_dir if is_container_backend(backend) else None,
    )

    if not output_files:
        error = "no output files produced"
        return {"passed": False, "error": error, "stdout": stdout, "stderr": stderr}

    if len(output_files) != len(expected_shapes):
        error = f"expected {len(expected_shapes)} output file(s), got {len(output_files)}"
        return {"passed": False, "error": error, "stdout": stdout, "stderr": stderr}

    # Check each output
    for out_file, expected_shape in zip(output_files, expected_shapes):
        data = np.fromfile(out_file, dtype=np.float64)
        expected_size = 1
        for d in expected_shape:
            expected_size *= d

        if data.size != expected_size:
            error = (
                f"{out_file.name}: expected {expected_size} values "
                f"(shape {expected_shape}), got {data.size}"
            )
            return {"passed": False, "error": error, "stdout": stdout, "stderr": stderr}

        if check_nan_inf:
            if np.any(np.isnan(data)):
                error = f"{out_file.name}: output contains NaN"
                return {"passed": False, "error": error, "stdout": stdout, "stderr": stderr}
            if np.any(np.isinf(data)):
                error = f"{out_file.name}: output contains Inf"
                return {"passed": False, "error": error, "stdout": stdout, "stderr": stderr}

    return {"passed": True, "error": None}


def run_and_check_bounds(
    network_path,
    box_paths,
    backend,
    backend_arg,
    expected_shapes,
    check_nan_inf=False,
    timeout=60,
    extra_run_args=(),
    n_eval_samples=3,
    sample_atol=1e-6,
    sample_rtol=1e-6,
):
    """Run the SUT's bounds command on a network + per-input box, validate outputs.

    ``box_paths`` is a list of ``(lb_path, ub_path)`` per input variable.
    The impl is expected to emit two output files per network output (lb
    first, then ub).  We verify:

      * exit code 0
      * exactly ``2 * len(expected_shapes)`` output files
      * each output file has the right element count
      * for every (lb, ub) pair, ``lb <= ub`` element-wise
      * optionally, no NaN/Inf
      * **soundness**: for ``n_eval_samples`` random points inside the input
        box (plus the box corners and centre), the SUT's ``eval`` output must
        lie within the SUT's bounds, i.e. ``lb - tol <= eval(x) <= ub + tol``
        with ``tol = sample_atol + sample_rtol * max(|lb|, |ub|)``.

    The soundness check is *self-consistency* (SUT bounds ⊇ SUT eval), not a
    comparison against a ground-truth oracle.  Non-finite ``eval`` outputs
    (e.g. from numerically unsafe primitives) are skipped, and ``eval`` runs
    that crash on a sampled point are ignored — those are the province of the
    eval fuzz tests, not bounds soundness.
    """
    network_path = Path(network_path)
    work_dir = network_path.parent

    cmd = _build_bounds_cmd(
        network_path,
        box_paths,
        work_dir / "output",
        backend,
        backend_arg,
        work_dir,
        extra_run_args=extra_run_args,
    )
    (work_dir / "output").mkdir(exist_ok=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"timed out after {timeout}s"}

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        return {
            "passed": False,
            "error": f"crash (exit {result.returncode}): {_truncate(stderr)}",
            "stdout": stdout,
            "stderr": stderr,
        }

    output_files, _ = parse_output_paths(
        result.stdout,
        container_root=work_dir if is_container_backend(backend) else None,
    )

    n_outputs = len(expected_shapes)
    if not output_files:
        return {
            "passed": False,
            "error": "no output files produced",
            "stdout": stdout, "stderr": stderr,
        }
    if len(output_files) != 2 * n_outputs:
        return {
            "passed": False,
            "error": (
                f"expected {2 * n_outputs} bound files (lb,ub per output), "
                f"got {len(output_files)}"
            ),
            "stdout": stdout, "stderr": stderr,
        }

    impl_bounds = []
    for i, expected_shape in enumerate(expected_shapes):
        expected_size = 1
        for d in expected_shape:
            expected_size *= d
        lb_path, ub_path = output_files[2 * i], output_files[2 * i + 1]
        lb = np.fromfile(lb_path, dtype=np.float64)
        ub = np.fromfile(ub_path, dtype=np.float64)
        if lb.size != expected_size or ub.size != expected_size:
            return {
                "passed": False,
                "error": (
                    f"output {i}: expected {expected_size} values per bound "
                    f"(shape {expected_shape}), got lb={lb.size}, ub={ub.size}"
                ),
                "stdout": stdout, "stderr": stderr,
            }
        if (ub < lb).any():
            worst = float((lb - ub).max())
            return {
                "passed": False,
                "error": f"output {i}: impl_ub < impl_lb by up to {worst:.3e}",
                "stdout": stdout, "stderr": stderr,
            }
        if check_nan_inf:
            if not np.isfinite(lb).all():
                return {
                    "passed": False,
                    "error": f"{lb_path.name}: lb contains NaN/Inf",
                    "stdout": stdout, "stderr": stderr,
                }
            if not np.isfinite(ub).all():
                return {
                    "passed": False,
                    "error": f"{ub_path.name}: ub contains NaN/Inf",
                    "stdout": stdout, "stderr": stderr,
                }
        impl_bounds.append((lb, ub))

    sound = _check_bounds_soundness(
        network_path,
        box_paths,
        impl_bounds,
        backend,
        backend_arg,
        n_outputs,
        n_eval_samples=n_eval_samples,
        sample_atol=sample_atol,
        sample_rtol=sample_rtol,
        timeout=timeout,
        extra_run_args=extra_run_args,
    )
    if not sound["passed"]:
        sound.setdefault("stdout", stdout)
        sound.setdefault("stderr", stderr)
        return sound

    return {"passed": True, "error": None}


def run_and_check_affine_bounds(
    network_path,
    box_paths,
    backend,
    backend_arg,
    input_shape,
    check_nan_inf=False,
    timeout=60,
    extra_run_args=(),
    n_eval_samples=3,
    sample_atol=1e-6,
    sample_rtol=1e-6,
):
    """Run the SUT's ``affine_bounds`` command on a single-input scalar network.

    CROWN returns an affine relaxation of the scalar output as a function of the
    (single) input ``x``.  ``box_paths`` is a one-element list ``[(lb, ub)]``.
    The impl writes four files (``lb_weight``, ``lb_bias``, ``ub_weight``,
    ``ub_bias``) and prints their paths in that order.  We verify:

      * exit code 0 and exactly 4 output files,
      * the weight files have ``prod(input_shape)`` elements and the bias files
        one element each,
      * optionally, no NaN/Inf,
      * **validity**: ``lb_aff(x) <= ub_aff(x)`` for all ``x`` in the box (the
        difference is affine, so its max is attained at a box vertex), and
      * **soundness**: for sampled points in the box, the SUT's ``eval`` output
        lies between the lower and upper affine bounds.

    As with ``run_and_check_bounds`` the soundness check is self-consistency
    (SUT affine bounds ⊇ SUT eval), not a comparison against a ground truth.
    """
    network_path = Path(network_path)
    work_dir = network_path.parent

    cmd = _build_bounds_cmd(
        network_path,
        box_paths,
        work_dir / "output",
        backend,
        backend_arg,
        work_dir,
        extra_run_args=extra_run_args,
        command="affine_bounds",
    )
    (work_dir / "output").mkdir(exist_ok=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"timed out after {timeout}s"}

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        return {
            "passed": False,
            "error": f"crash (exit {result.returncode}): {_truncate(stderr)}",
            "stdout": stdout, "stderr": stderr,
        }

    output_files, _ = parse_output_paths(
        result.stdout,
        container_root=work_dir if is_container_backend(backend) else None,
    )
    if not output_files:
        return {
            "passed": False,
            "error": "no output files produced",
            "stdout": stdout, "stderr": stderr,
        }
    if len(output_files) != 4:
        return {
            "passed": False,
            "error": (
                "expected 4 affine-bound files (lb_weight, lb_bias, ub_weight, "
                f"ub_bias), got {len(output_files)}"
            ),
            "stdout": stdout, "stderr": stderr,
        }

    n_in = 1
    for d in input_shape:
        n_in *= int(d)

    lbw = np.fromfile(output_files[0], dtype=np.float64)
    lbb = np.fromfile(output_files[1], dtype=np.float64)
    ubw = np.fromfile(output_files[2], dtype=np.float64)
    ubb = np.fromfile(output_files[3], dtype=np.float64)
    for name, arr, want in (
        ("lb_weight", lbw, n_in),
        ("lb_bias", lbb, 1),
        ("ub_weight", ubw, n_in),
        ("ub_bias", ubb, 1),
    ):
        if arr.size != want:
            return {
                "passed": False,
                "error": f"{name}: expected {want} double(s), got {arr.size}",
                "stdout": stdout, "stderr": stderr,
            }
        if check_nan_inf and not np.isfinite(arr).all():
            return {
                "passed": False,
                "error": f"{name}: contains NaN/Inf",
                "stdout": stdout, "stderr": stderr,
            }

    # Non-finite affine bounds (e.g. an ``inf`` slope from ``exp`` overflowing on a
    # wide box) are vacuously sound and cannot be meaningfully range-checked.  When
    # ``check_nan_inf`` is off we accept them, mirroring how the interval ``bounds``
    # fuzzer tolerates ``inf`` intervals; when it is on they have already failed above.
    if not all(np.isfinite(a).all() for a in (lbw, lbb, ubw, ubb)):
        return {"passed": True, "error": None}

    lbw = lbw.reshape(n_in)
    ubw = ubw.reshape(n_in)
    lbb = float(lbb.reshape(-1)[0])
    ubb = float(ubb.reshape(-1)[0])

    # --- Validity: max_{x in box} (lb_aff - ub_aff)(x) <= tol. ---
    box_lb = np.fromfile(box_paths[0][0], dtype=np.float64)
    box_ub = np.fromfile(box_paths[0][1], dtype=np.float64)
    center = 0.5 * (box_lb + box_ub)
    radius = 0.5 * (box_ub - box_lb)
    diff_w = lbw - ubw
    max_diff = (lbb - ubb) + float(diff_w @ center) + float(np.abs(diff_w) @ radius)
    # Scale the validity tolerance to the bound magnitude: fuzz graphs reach
    # large weights/biases where pure-absolute float slack is too strict.
    scale = abs(lbb) + abs(ubb) + float((np.abs(lbw) + np.abs(ubw)) @ (np.abs(center) + radius))
    if max_diff > sample_atol + sample_rtol * scale:
        return {
            "passed": False,
            "error": f"lower affine bound exceeds upper affine bound within the box "
            f"by up to {max_diff:.3e}",
            "stdout": stdout, "stderr": stderr,
        }

    sound = _check_affine_soundness(
        network_path,
        box_paths,
        (lbw, lbb, ubw, ubb),
        backend,
        backend_arg,
        n_eval_samples=n_eval_samples,
        sample_atol=sample_atol,
        sample_rtol=sample_rtol,
        timeout=timeout,
        extra_run_args=extra_run_args,
    )
    if not sound["passed"]:
        sound.setdefault("stdout", stdout)
        sound.setdefault("stderr", stderr)
        return sound

    return {"passed": True, "error": None}


def _check_affine_soundness(
    network_path,
    box_paths,
    affine,
    backend,
    backend_arg,
    n_eval_samples,
    sample_atol,
    sample_rtol,
    timeout,
    extra_run_args,
):
    """Validate ``lb_aff(x) <= eval(x) <= ub_aff(x)`` at points sampled in the box.

    ``affine`` is ``(lb_weight, lb_bias, ub_weight, ub_bias)``.  Mirrors
    ``_check_bounds_soundness``: sampled ``eval`` runs that crash or yield a
    non-finite value are skipped; only a finite output escaping the affine bounds
    beyond tolerance is a failure.
    """
    if n_eval_samples <= 0:
        return {"passed": True, "error": None}

    lbw, lbb, ubw, ubb = affine
    box = [
        (np.fromfile(lb_path, dtype=np.float64), np.fromfile(ub_path, dtype=np.float64))
        for lb_path, ub_path in box_paths
    ]
    h = hashlib.sha256()
    for lb_path, ub_path in box_paths:
        h.update(Path(lb_path).read_bytes())
        h.update(Path(ub_path).read_bytes())
    seed = int.from_bytes(h.digest()[:4], "big")

    work_dir = Path(network_path).parent
    eval_out_dir = work_dir / "eval_output"
    eval_out_dir.mkdir(exist_ok=True)

    for point in _sample_points(box, n_eval_samples, seed):
        outputs = _eval_point(
            network_path,
            point,
            eval_out_dir,
            backend,
            backend_arg,
            1,
            timeout=timeout,
            extra_run_args=extra_run_args,
        )
        if outputs is None:
            continue  # eval crashed / wrong shape — not a bounds bug
        out = outputs[0]
        if out.size != 1:
            continue
        fv = float(out.reshape(-1)[0])
        if not np.isfinite(fv):
            continue
        x = point[0]  # flat input for the single invar
        lb_aff = float(x @ lbw + lbb)
        ub_aff = float(x @ ubw + ubb)
        tol = sample_atol + sample_rtol * max(abs(lb_aff), abs(ub_aff), abs(fv))
        if lb_aff - fv > tol:
            return {
                "passed": False,
                "error": (
                    f"lower affine bound unsound — exceeds an eval sample by "
                    f"{lb_aff - fv:.3e} (atol {sample_atol:.1e}, rtol {sample_rtol:.1e})"
                ),
            }
        if fv - ub_aff > tol:
            return {
                "passed": False,
                "error": (
                    f"upper affine bound unsound — an eval sample exceeds it by "
                    f"{fv - ub_aff:.3e} (atol {sample_atol:.1e}, rtol {sample_rtol:.1e})"
                ),
            }

    return {"passed": True, "error": None}


def _sample_points(box, n_eval_samples, seed):
    """Yield flat per-invar input points inside ``box``.

    ``box`` is a list of ``(lb, ub)`` flat float64 arrays, one per invar.
    Always yields the two box corners (all-lb, all-ub) and the centre, then
    ``n_eval_samples`` independent uniform-random interior points.  The RNG is
    seeded from ``seed`` (derived from the box bytes) so the same saved box
    reproduces the same points.
    """
    yield [lb.copy() for lb, _ in box]
    yield [ub.copy() for _, ub in box]
    yield [0.5 * (lb + ub) for lb, ub in box]
    rng = np.random.RandomState(seed)
    for _ in range(n_eval_samples):
        yield [lb + rng.random_sample(lb.shape) * (ub - lb) for lb, ub in box]


def _check_bounds_soundness(
    network_path,
    box_paths,
    impl_bounds,
    backend,
    backend_arg,
    n_outputs,
    n_eval_samples,
    sample_atol,
    sample_rtol,
    timeout,
    extra_run_args,
):
    """Validate ``impl_lb <= eval(x) <= impl_ub`` for points sampled in the box.

    Returns a result dict with ``passed``/``error``.  Sampled ``eval`` runs
    that crash or yield non-finite values are skipped (see
    ``run_and_check_bounds`` docstring); only a finite output that escapes the
    bounds beyond tolerance is a failure.
    """
    if n_eval_samples <= 0:
        return {"passed": True, "error": None}

    box = [
        (np.fromfile(lb_path, dtype=np.float64), np.fromfile(ub_path, dtype=np.float64))
        for lb_path, ub_path in box_paths
    ]
    # Seed the sample RNG from the box bytes so reproduction is deterministic.
    h = hashlib.sha256()
    for lb_path, ub_path in box_paths:
        h.update(Path(lb_path).read_bytes())
        h.update(Path(ub_path).read_bytes())
    seed = int.from_bytes(h.digest()[:4], "big")

    work_dir = Path(network_path).parent
    eval_out_dir = work_dir / "eval_output"
    eval_out_dir.mkdir(exist_ok=True)

    for point in _sample_points(box, n_eval_samples, seed):
        outputs = _eval_point(
            network_path,
            point,
            eval_out_dir,
            backend,
            backend_arg,
            n_outputs,
            timeout=timeout,
            extra_run_args=extra_run_args,
        )
        if outputs is None:
            continue  # eval crashed / produced wrong shape — not a bounds bug
        for i, out in enumerate(outputs):
            lb, ub = impl_bounds[i]
            if out.size != lb.size:
                continue
            finite = np.isfinite(out)
            if not finite.any():
                continue
            o = out[finite]
            lo = lb[finite]
            hi = ub[finite]
            tol = sample_atol + sample_rtol * np.maximum(np.abs(lo), np.abs(hi))
            lb_viol = float(np.max(lo - o - tol))
            ub_viol = float(np.max(o - hi - tol))
            if lb_viol > 0:
                return {
                    "passed": False,
                    "error": (
                        f"output {i}: lb unsound — impl_lb exceeds an eval sample "
                        f"by {lb_viol:.3e} (atol {sample_atol:.1e}, rtol {sample_rtol:.1e})"
                    ),
                }
            if ub_viol > 0:
                return {
                    "passed": False,
                    "error": (
                        f"output {i}: ub unsound — an eval sample exceeds impl_ub "
                        f"by {ub_viol:.3e} (atol {sample_atol:.1e}, rtol {sample_rtol:.1e})"
                    ),
                }

    return {"passed": True, "error": None}


def _eval_point(
    network_path,
    point,
    output_dir,
    backend,
    backend_arg,
    n_outputs,
    timeout,
    extra_run_args,
):
    """Run the SUT's ``eval`` command on a single concrete input point.

    ``point`` is a list of flat float64 arrays in invar order.  Returns a list
    of flat output arrays (in outvar order), or ``None`` if eval crashed, timed
    out, or produced the wrong number of outputs.
    """
    network_path = Path(network_path)
    work_dir = network_path.parent
    input_paths = []
    for j, arr in enumerate(point):
        p = output_dir / f"sample_in_{j}.bin"
        arr.astype(np.float64).tofile(p)
        input_paths.append(p)

    cmd = _build_cmd(
        "eval",
        network_path,
        input_paths,
        output_dir,
        backend,
        backend_arg,
        work_dir,
        extra_run_args=extra_run_args,
    )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None

    output_files, _ = parse_output_paths(
        result.stdout,
        container_root=work_dir if is_container_backend(backend) else None,
    )
    if len(output_files) != n_outputs:
        return None
    return [np.fromfile(f, dtype=np.float64) for f in output_files]


def _save_failure(
    save_dir,
    network_path,
    input_paths,
    mode,
    check_nan_inf,
    expected_shapes,
    error=None,
    stderr=None,
    sound_cfg=None,
):
    """Copy the network and input files to save_dir for reproduction.

    Returns the path to the saved directory, or None if saving is disabled.
    """
    if save_dir is None:
        return None
    # Use incrementing directory names: 0, 1, 2, ...
    existing = [int(p.name) for p in save_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    idx = max(existing, default=-1) + 1
    dest = save_dir / str(idx)
    dest.mkdir()
    shutil.copy2(network_path, dest / network_path.name)
    for inp in input_paths:
        shutil.copy2(inp, dest / inp.name)
    # Save metadata needed for reproduction
    metadata = {
        "mode": mode,
        "check_nan_inf": check_nan_inf,
        "expected_shapes": [list(s) for s in expected_shapes],
        "inputs": [inp.name for inp in input_paths],
    }
    if sound_cfg:
        # Soundness-sampling params so reproduction redoes the same eval check.
        metadata["sound_cfg"] = dict(sound_cfg)
    (dest / "metadata.json").write_text(json_mod.dumps(metadata, indent=2))
    if error:
        (dest / "error.txt").write_text(error)
    if stderr:
        (dest / "stderr.txt").write_text(stderr)
    return str(dest)


def _truncate(text, max_len=500):
    """Truncate text to max_len, keeping the first and last halves."""
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return text[:half] + " ... " + text[-half:]


def _build_cmd(
    mode,
    network_path,
    input_paths,
    output_dir,
    backend,
    backend_arg,
    test_dir,
    extra_run_args=(),
):
    """Build the CLI command for eval or grad."""
    if is_container_backend(backend):
        cmd = [
            *container_run_prefix(backend, test_dir, extra_run_args),
            backend_arg,
            mode,
            "--output-dir",
            f"/data/{output_dir.relative_to(test_dir)}",
            f"/data/{network_path.relative_to(test_dir)}",
            *[f"/data/{p.relative_to(test_dir)}" for p in input_paths],
        ]
    else:
        cmd = [
            *shlex.split(backend_arg),
            mode,
            "--output-dir",
            str(output_dir),
            str(network_path),
            *[str(p) for p in input_paths],
        ]
    return cmd


def _build_bounds_cmd(
    network_path,
    box_paths,
    output_dir,
    backend,
    backend_arg,
    test_dir,
    extra_run_args=(),
    command="bounds",
):
    """Build the CLI command for ``command`` (``bounds`` or ``affine_bounds``)
    with one ``box`` marker per invar."""
    if is_container_backend(backend):
        tokens = []
        for lb_path, ub_path in box_paths:
            tokens += [
                "box",
                f"/data/{lb_path.relative_to(test_dir)}",
                f"/data/{ub_path.relative_to(test_dir)}",
            ]
        cmd = [
            *container_run_prefix(backend, test_dir, extra_run_args),
            backend_arg,
            command,
            "--output-dir",
            f"/data/{output_dir.relative_to(test_dir)}",
            f"/data/{network_path.relative_to(test_dir)}",
            *tokens,
        ]
    else:
        tokens = []
        for lb_path, ub_path in box_paths:
            tokens += ["box", str(lb_path), str(ub_path)]
        cmd = [
            *shlex.split(backend_arg),
            command,
            "--output-dir",
            str(output_dir),
            str(network_path),
            *tokens,
        ]
    return cmd
