# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Check that output bounds are sound and not too loose.

The implementation under test produces, for each network output, a pair of
files ``output_{i}_lb.bin`` and ``output_{i}_ub.bin`` printed in that order
(``lb`` first, then ``ub``).  This check validates them against:

* **Soundness.** For each pre-evaluated network output at a sample input
  point inside the input box, the impl's bounds must contain it:
  ``impl_lb <= sample_output <= impl_ub`` (with absolute slack
  ``sample_atol``).
* **Tightness (optional).** If a reference IBP bound pair is supplied, the
  impl's box width must not exceed the reference width by more than
  ``(tightness_factor - 1) * ref_width`` on either side, i.e.
  ``impl_lb >= ref_lb - (factor - 1) * ref_range`` and
  ``impl_ub <= ref_ub + (factor - 1) * ref_range``.

Config fields:
    output_shapes      list of shape lists, one per network output (required)
    sample_outputs     list of paths to flat (N, *out_shape) float64 sample-
                       output files, one per network output (required)
    reference_lb       list of paths to reference lb bins (optional)
    reference_ub       list of paths to reference ub bins (optional)
    tightness_factor   float, default 1.5 (only used if reference_lb/ub set)
    sample_atol        float, default 1e-9
"""

import numpy as np


def _prod(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def check_bounds_within_range(test_dir, config, output_files, closed=False):
    output_shapes = [tuple(s) for s in config.get("output_shapes", [])]
    n_outputs = len(output_shapes)
    if n_outputs == 0:
        return {"passed": False, "error": "output_shapes missing or empty in test.json"}

    if len(output_files) != 2 * n_outputs:
        return {
            "passed": False,
            "error": f"expected {2 * n_outputs} bound files (lb,ub per output), got {len(output_files)}",
        }

    impl_bounds = []
    for i, shape in enumerate(output_shapes):
        lb_path, ub_path = output_files[2 * i], output_files[2 * i + 1]
        lb = np.fromfile(lb_path, dtype=np.float64)
        ub = np.fromfile(ub_path, dtype=np.float64)
        n = _prod(shape)
        if lb.size != n or ub.size != n:
            return {
                "passed": False,
                "error": (
                    f"output {i}: expected {n} doubles per bound for shape {shape}, "
                    f"got lb={lb.size}, ub={ub.size}"
                ),
            }
        impl_bounds.append((lb.reshape(shape), ub.reshape(shape)))

    if (impl_bounds[0][1] < impl_bounds[0][0]).any():
        worst = float((impl_bounds[0][0] - impl_bounds[0][1]).max())
        return {
            "passed": False,
            "error": f"output 0: impl_ub < impl_lb (inverted box) by up to {worst:.3e}",
        }

    sample_atol = float(config.get("sample_atol", 1e-9))
    sample_out_paths = config.get("sample_outputs", [])
    if len(sample_out_paths) != n_outputs:
        return {
            "passed": False,
            "error": f"expected {n_outputs} sample_outputs entries, got {len(sample_out_paths)}",
        }

    for i, (lb, ub) in enumerate(impl_bounds):
        flat = np.fromfile(test_dir / sample_out_paths[i], dtype=np.float64)
        n_per = _prod(output_shapes[i])
        if flat.size % n_per != 0:
            return {
                "passed": False,
                "error": (
                    f"sample_outputs[{i}]: size {flat.size} not a multiple of "
                    f"prod(output_shape)={n_per}"
                ),
            }
        n_samples = flat.size // n_per
        samples_out = flat.reshape((n_samples,) + output_shapes[i])

        # impl_lb above some sample output? impl_ub below some sample output?
        lb_viol = float((lb[None, ...] - samples_out).max())
        ub_viol = float((samples_out - ub[None, ...]).max())
        if lb_viol > sample_atol:
            return {
                "passed": False,
                "error": (
                    f"output {i}: lb unsound — impl_lb exceeds a sample output by "
                    f"{lb_viol:.3e} (atol {sample_atol:.1e}, {n_samples} samples)"
                ),
            }
        if ub_viol > sample_atol:
            return {
                "passed": False,
                "error": (
                    f"output {i}: ub unsound — a sample output exceeds impl_ub by "
                    f"{ub_viol:.3e} (atol {sample_atol:.1e}, {n_samples} samples)"
                ),
            }

    ref_lb_paths = config.get("reference_lb")
    ref_ub_paths = config.get("reference_ub")
    if ref_lb_paths and ref_ub_paths:
        if len(ref_lb_paths) != n_outputs or len(ref_ub_paths) != n_outputs:
            return {
                "passed": False,
                "error": "reference_lb/reference_ub must have one entry per output",
            }
        factor = float(config.get("tightness_factor", 1.5))
        extra_frac = max(0.0, factor - 1.0)
        for i, (impl_lb, impl_ub) in enumerate(impl_bounds):
            ref_lb = np.fromfile(test_dir / ref_lb_paths[i], dtype=np.float64).reshape(
                output_shapes[i]
            )
            ref_ub = np.fromfile(test_dir / ref_ub_paths[i], dtype=np.float64).reshape(
                output_shapes[i]
            )
            ref_range = ref_ub - ref_lb
            slack = extra_frac * np.maximum(ref_range, 0.0)
            lb_loose = float((ref_lb - slack - impl_lb).max())
            ub_loose = float((impl_ub - (ref_ub + slack)).max())
            if lb_loose > 0:
                return {
                    "passed": False,
                    "error": (
                        f"output {i}: lb is looser than {factor:.2f}x reference IBP "
                        f"by {lb_loose:.3e}"
                    ),
                }
            if ub_loose > 0:
                return {
                    "passed": False,
                    "error": (
                        f"output {i}: ub is looser than {factor:.2f}x reference IBP "
                        f"by {ub_loose:.3e}"
                    ),
                }

    return {"passed": True, "error": None}
