# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.check.close_to_expected."""

import numpy as np
import pytest

from testrunner.check import CHECKS, DEFAULT_CHECKS
from testrunner.check.close_to_expected import check_close_to_expected


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_checks_registry():
    assert "close_to_expected" in CHECKS


def test_default_checks():
    assert DEFAULT_CHECKS["eval"] == "close_to_expected"
    assert DEFAULT_CHECKS["grad"] == "close_to_expected"


# ---------------------------------------------------------------------------
# check_close_to_expected
# ---------------------------------------------------------------------------


def _write_bin(path, arr):
    np.array(arr, dtype=np.float64).tofile(path)


def test_close_to_expected_passes(tmp_path):
    expected = np.array([1.0, 2.0, 3.0])
    actual = expected + 1e-6  # within default tolerance
    _write_bin(tmp_path / "actual.bin", actual)
    _write_bin(tmp_path / "expected.bin", expected)
    config = {"expected_outputs": ["expected.bin"], "tolerance": 1e-4}
    result = check_close_to_expected(tmp_path, config, [tmp_path / "actual.bin"])
    assert result["passed"] is True
    assert result["error"] is None


def test_close_to_expected_fails_diff(tmp_path):
    expected = np.array([1.0, 2.0, 3.0])
    actual = expected + 1.0  # way beyond tolerance
    _write_bin(tmp_path / "actual.bin", actual)
    _write_bin(tmp_path / "expected.bin", expected)
    config = {"expected_outputs": ["expected.bin"], "tolerance": 1e-4}
    result = check_close_to_expected(tmp_path, config, [tmp_path / "actual.bin"])
    assert result["passed"] is False
    assert "max absolute diff" in result["error"]


def test_close_to_expected_fails_diff_closed(tmp_path):
    expected = np.array([1.0])
    actual = expected + 1.0
    _write_bin(tmp_path / "actual.bin", actual)
    _write_bin(tmp_path / "expected.bin", expected)
    config = {"expected_outputs": ["expected.bin"]}
    result = check_close_to_expected(tmp_path, config, [tmp_path / "actual.bin"], closed=True)
    assert result["passed"] is False
    # closed=True should not reveal exact diff values
    assert "max absolute diff" in result["error"]
    assert "expected" not in result["error"]


def test_close_to_expected_shape_mismatch(tmp_path):
    _write_bin(tmp_path / "actual.bin", [1.0, 2.0])
    _write_bin(tmp_path / "expected.bin", [1.0, 2.0, 3.0])
    config = {"expected_outputs": ["expected.bin"]}
    result = check_close_to_expected(tmp_path, config, [tmp_path / "actual.bin"])
    assert result["passed"] is False
    assert "element count mismatch" in result["error"]


def test_close_to_expected_wrong_file_count(tmp_path):
    config = {"expected_outputs": ["e1.bin", "e2.bin"]}
    result = check_close_to_expected(tmp_path, config, [tmp_path / "only_one.bin"])
    assert result["passed"] is False
    assert "expected 2 output file(s), got 1" in result["error"]


def test_close_to_expected_default_tolerance(tmp_path):
    expected = np.array([0.0])
    actual = expected + 1e-5  # within default 1e-4
    _write_bin(tmp_path / "actual.bin", actual)
    _write_bin(tmp_path / "expected.bin", expected)
    config = {"expected_outputs": ["expected.bin"]}  # no tolerance key
    result = check_close_to_expected(tmp_path, config, [tmp_path / "actual.bin"])
    assert result["passed"] is True


def test_close_to_expected_exact_match(tmp_path):
    data = np.array([1.0, -2.0, 0.5])
    _write_bin(tmp_path / "actual.bin", data)
    _write_bin(tmp_path / "expected.bin", data)
    config = {"expected_outputs": ["expected.bin"]}
    result = check_close_to_expected(tmp_path, config, [tmp_path / "actual.bin"])
    assert result["passed"] is True


def test_close_to_expected_zero_outputs(tmp_path):
    config = {"expected_outputs": []}
    result = check_close_to_expected(tmp_path, config, [])
    assert result["passed"] is True


def test_close_to_expected_at_tolerance_boundary(tmp_path):
    """Value exactly at tolerance should pass."""
    expected = np.array([0.0])
    actual = expected + 1e-4  # exactly at tolerance
    _write_bin(tmp_path / "actual.bin", actual)
    _write_bin(tmp_path / "expected.bin", expected)
    config = {"expected_outputs": ["expected.bin"], "tolerance": 1e-4}
    result = check_close_to_expected(tmp_path, config, [tmp_path / "actual.bin"])
    assert result["passed"] is True


def test_close_to_expected_just_over_tolerance(tmp_path):
    """Value just beyond tolerance should fail."""
    expected = np.array([0.0])
    actual = expected + 1e-4 + 1e-10
    _write_bin(tmp_path / "actual.bin", actual)
    _write_bin(tmp_path / "expected.bin", expected)
    config = {"expected_outputs": ["expected.bin"], "tolerance": 1e-4}
    result = check_close_to_expected(tmp_path, config, [tmp_path / "actual.bin"])
    assert result["passed"] is False
