# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Check that CROWN *affine* output bounds are valid, sound, and not too loose.

Unlike ``bounds_within_range`` (which validates constant interval bounds), the
implementation under test here produces an affine relaxation of a scalar-output
network ``f`` as a function of its (single) input ``x``:

    lb_weight @ x + lb_bias  <=  f(x)  <=  ub_weight @ x + ub_bias    (∀ x ∈ box)

The implementation writes four files per output and prints their paths in this
order: ``output_<i>_lb_weight.bin``, ``output_<i>_lb_bias.bin``,
``output_<i>_ub_weight.bin``, ``output_<i>_ub_bias.bin``.  This check validates
them against:

* **Validity.** The lower affine bound must not exceed the upper affine bound
  anywhere in the box: ``max_{x∈box} (lb_aff(x) - ub_aff(x)) <= atol``.
* **Soundness.** At every pre-evaluated sample point ``x`` in the box the affine
  bounds must contain the true output: ``lb_aff(x) <= f(x) <= ub_aff(x)`` (with
  absolute slack ``sample_atol``).
* **Tightness (optional).** The affine bounds, *concretized* over the box to a
  scalar interval ``[min lb_aff, max ub_aff]``, must be no looser than the
  reference IBP interval by more than ``(tightness_factor - 1) * ref_range``.

Config fields:
    inputs            inline ["box", lb.bin, ub.bin] spec for the single input
                      (required; "point" inputs are not supported here)
    input_shape       shape of the network input (required)
    output_shapes     [[<out>]] — exactly one scalar output (required)
    sample_inputs     list with one path to a flat (N, *input_shape) float64
                      file of sample points inside the box (required)
    sample_outputs    list with one path to a flat (N, *output_shape) float64
                      file of the matching network outputs (required)
    reference_lb      list with one path to the reference IBP lb bin (optional)
    reference_ub      list with one path to the reference IBP ub bin (optional)
    tightness_factor  float, default 1.5 (only used if reference_lb/ub set)
    sample_atol       float, default 1e-9
"""

import numpy as np


def _prod(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _read_box(test_dir, config, input_shape):
    """Parse the single ``box`` input spec and return (lb, ub) flattened."""
    inputs = config.get("inputs", [])
    if len(inputs) != 3 or inputs[0] != "box":
        return None, "affine_bounds expects a single 'box' input spec"
    n = _prod(input_shape)
    lb = np.fromfile(test_dir / inputs[1], dtype=np.float64)
    ub = np.fromfile(test_dir / inputs[2], dtype=np.float64)
    if lb.size != n or ub.size != n:
        return None, f"input box size {lb.size}/{ub.size} != prod(input_shape)={n}"
    return (lb, ub), None


def check_affine_bounds_within_range(test_dir, config, output_files, closed=False):
    output_shapes = [tuple(s) for s in config.get("output_shapes", [])]
    if len(output_shapes) != 1:
        return {"passed": False, "error": "affine_bounds requires exactly one (scalar) output"}
    out_shape = output_shapes[0]
    n_out = _prod(out_shape)

    input_shape = tuple(config.get("input_shape", ()))
    if not input_shape:
        return {"passed": False, "error": "input_shape missing in test.json"}
    n_in = _prod(input_shape)

    if len(output_files) != 4:
        return {
            "passed": False,
            "error": f"expected 4 affine-bound files (lb_weight, lb_bias, ub_weight, ub_bias), "
            f"got {len(output_files)}",
        }

    lbw = np.fromfile(output_files[0], dtype=np.float64)
    lbb = np.fromfile(output_files[1], dtype=np.float64)
    ubw = np.fromfile(output_files[2], dtype=np.float64)
    ubb = np.fromfile(output_files[3], dtype=np.float64)
    for name, arr, want in (
        ("lb_weight", lbw, n_in * n_out),
        ("lb_bias", lbb, n_out),
        ("ub_weight", ubw, n_in * n_out),
        ("ub_bias", ubb, n_out),
    ):
        if arr.size != want:
            return {
                "passed": False,
                "error": f"{name}: expected {want} doubles, got {arr.size}",
            }
    # Scalar output: weights are vectors over the input, biases are scalars.
    lbw = lbw.reshape(n_in)
    ubw = ubw.reshape(n_in)
    lbb = float(lbb.reshape(-1)[0])
    ubb = float(ubb.reshape(-1)[0])

    box, err = _read_box(test_dir, config, input_shape)
    if err is not None:
        return {"passed": False, "error": err}
    box_lb, box_ub = box
    center = 0.5 * (box_lb + box_ub)
    radius = 0.5 * (box_ub - box_lb)

    # --- Validity: lb_aff(x) <= ub_aff(x) for all x in the box. ---
    # (lb_aff - ub_aff)(x) is affine; its max over the box is at a vertex.
    diff_w = lbw - ubw
    diff_b = lbb - ubb
    max_diff = diff_b + float(diff_w @ center) + float(np.abs(diff_w) @ radius)
    sample_atol = float(config.get("sample_atol", 1e-9))
    if max_diff > sample_atol:
        return {
            "passed": False,
            "error": f"lower affine bound exceeds upper affine bound within the box by "
            f"up to {max_diff:.3e}",
        }

    # --- Soundness: lb_aff(x) <= f(x) <= ub_aff(x) at every sample point. ---
    sample_in_paths = config.get("sample_inputs", [])
    sample_out_paths = config.get("sample_outputs", [])
    if len(sample_in_paths) != 1 or len(sample_out_paths) != 1:
        return {
            "passed": False,
            "error": "affine_bounds requires exactly one sample_inputs and one sample_outputs entry",
        }
    xs = np.fromfile(test_dir / sample_in_paths[0], dtype=np.float64)
    ys = np.fromfile(test_dir / sample_out_paths[0], dtype=np.float64)
    if xs.size % n_in != 0:
        return {"passed": False, "error": f"sample_inputs size {xs.size} not a multiple of {n_in}"}
    n_samples = xs.size // n_in
    xs = xs.reshape(n_samples, n_in)
    if ys.size != n_samples * n_out:
        return {
            "passed": False,
            "error": f"sample_outputs size {ys.size} != n_samples*{n_out}={n_samples * n_out}",
        }
    ys = ys.reshape(n_samples)  # scalar output

    lb_aff = xs @ lbw + lbb
    ub_aff = xs @ ubw + ubb
    lb_viol = float((lb_aff - ys).max())
    ub_viol = float((ys - ub_aff).max())
    if lb_viol > sample_atol:
        return {
            "passed": False,
            "error": f"lower affine bound unsound — exceeds a sample output by {lb_viol:.3e} "
            f"(atol {sample_atol:.1e}, {n_samples} samples)",
        }
    if ub_viol > sample_atol:
        return {
            "passed": False,
            "error": f"upper affine bound unsound — a sample output exceeds it by {ub_viol:.3e} "
            f"(atol {sample_atol:.1e}, {n_samples} samples)",
        }

    # --- Tightness: concretize the affine bounds over the box, compare to IBP. ---
    ref_lb_paths = config.get("reference_lb")
    ref_ub_paths = config.get("reference_ub")
    if ref_lb_paths and ref_ub_paths:
        if len(ref_lb_paths) != 1 or len(ref_ub_paths) != 1:
            return {"passed": False, "error": "reference_lb/reference_ub must have one entry"}
        ref_lb = float(np.fromfile(test_dir / ref_lb_paths[0], dtype=np.float64).reshape(-1)[0])
        ref_ub = float(np.fromfile(test_dir / ref_ub_paths[0], dtype=np.float64).reshape(-1)[0])
        # Concretized scalar interval certified by the affine bounds.
        lb_concrete = lbb + float(lbw @ center) - float(np.abs(lbw) @ radius)
        ub_concrete = ubb + float(ubw @ center) + float(np.abs(ubw) @ radius)
        factor = float(config.get("tightness_factor", 1.5))
        slack = max(0.0, factor - 1.0) * max(ref_ub - ref_lb, 0.0)
        if (ref_lb - slack) - lb_concrete > 0:
            return {
                "passed": False,
                "error": f"lower bound looser than {factor:.2f}x reference IBP by "
                f"{(ref_lb - slack) - lb_concrete:.3e}",
            }
        if ub_concrete - (ref_ub + slack) > 0:
            return {
                "passed": False,
                "error": f"upper bound looser than {factor:.2f}x reference IBP by "
                f"{ub_concrete - (ref_ub + slack):.3e}",
            }

    return {"passed": True, "error": None}
