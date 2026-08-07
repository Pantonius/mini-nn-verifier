# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from .affine_bounds_within_range import check_affine_bounds_within_range
from .bounds_within_range import check_bounds_within_range
from .close_to_expected import check_close_to_expected

CHECKS = {
    "close_to_expected": check_close_to_expected,
    "bounds_within_range": check_bounds_within_range,
    "affine_bounds_within_range": check_affine_bounds_within_range,
}
DEFAULT_CHECKS = {
    "eval": "close_to_expected",
    "grad": "close_to_expected",
    "bounds": "close_to_expected",
    "affine_bounds": "affine_bounds_within_range",
}
