# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.scoring."""

import math

import pytest

from testrunner.scoring import (
    SCORING_FUNCTIONS,
    DEFAULT_SCORING,
    binary,
    compute_score,
    exponential,
    proportional,
    speed,
)


# ---------------------------------------------------------------------------
# binary
# ---------------------------------------------------------------------------


def test_binary_passed():
    assert binary(10, {"passed": True}) == (10, 0.0)


def test_binary_failed():
    assert binary(10, {"passed": False}) == (0.0, 0.0)


def test_binary_zero_points():
    assert binary(0, {"passed": True}) == (0, 0.0)


def test_binary_fractional_points():
    assert binary(7.5, {"passed": True}) == (7.5, 0.0)


def test_binary_bonus_passed():
    score, bonus = binary(10, {"passed": True}, bonus_points=5)
    assert score == 10
    assert bonus == 5.0


def test_binary_bonus_failed():
    score, bonus = binary(10, {"passed": False}, bonus_points=5)
    assert score == 0.0
    assert bonus == 0.0


# ---------------------------------------------------------------------------
# exponential
# ---------------------------------------------------------------------------


def test_exponential_perfect_accuracy():
    score, bonus = exponential(10, {"best_test_accuracy": 1.0})
    assert score == 10.0


def test_exponential_zero_accuracy():
    score, bonus = exponential(10, {"best_test_accuracy": 0.0})
    assert score == 0.0


def test_exponential_negative_accuracy():
    score, bonus = exponential(10, {"best_test_accuracy": -0.1})
    assert score == 0.0


def test_exponential_missing_accuracy():
    score, bonus = exponential(10, {})
    assert score == 0.0


def test_exponential_midpoint_is_between():
    score, bonus = exponential(10, {"best_test_accuracy": 0.5})
    assert 0.0 < score < 10.0


def test_exponential_higher_k_penalizes_mid_accuracy():
    """With high k, the points are concentrated near 1.0.
    At mid-range accuracy (0.9), lower k yields higher score.
    """
    s_low_k, _ = exponential(10, {"best_test_accuracy": 0.9}, k=1)
    s_high_k, _ = exponential(10, {"best_test_accuracy": 0.9}, k=10)
    # High k concentrates rewards toward 1.0, so mid-range accuracy scores lower
    assert s_low_k > s_high_k


def test_exponential_custom_k():
    k = 3
    acc = 0.5
    max_pts = 5
    expected = max_pts * (math.exp(k * acc) - 1) / (math.exp(k) - 1)
    score, bonus = exponential(max_pts, {"best_test_accuracy": acc}, k=k)
    assert abs(score - expected) < 1e-10


def test_exponential_bonus_below_threshold():
    score, bonus = exponential(
        10, {"best_test_accuracy": 0.8}, bonus_points=5, bonus_threshold=0.9
    )
    assert score > 0.0
    assert bonus == 0.0


def test_exponential_bonus_above_threshold():
    score, bonus = exponential(
        10, {"best_test_accuracy": 0.95}, bonus_points=5, bonus_threshold=0.9
    )
    assert score == 10.0  # at threshold -> full base
    assert 0.0 < bonus < 5.0


def test_exponential_bonus_perfect():
    score, bonus = exponential(
        10, {"best_test_accuracy": 1.0}, bonus_points=5, bonus_threshold=0.9
    )
    assert score == 10.0
    assert bonus == 5.0


# ---------------------------------------------------------------------------
# proportional
# ---------------------------------------------------------------------------


def test_proportional_all_pass():
    score, bonus = proportional(10, {"n_trials": 100, "n_passed": 100})
    assert score == 10.0


def test_proportional_none_pass():
    score, bonus = proportional(10, {"n_trials": 100, "n_passed": 0})
    assert score == 0.0


def test_proportional_half_pass():
    score, bonus = proportional(10, {"n_trials": 100, "n_passed": 50})
    assert score == 5.0


def test_proportional_zero_trials():
    score, bonus = proportional(10, {"n_trials": 0})
    assert score == 0.0


def test_proportional_missing_fields():
    score, bonus = proportional(10, {})
    assert score == 0.0


def test_proportional_bonus_below_threshold():
    score, bonus = proportional(
        10, {"n_trials": 100, "n_passed": 80}, bonus_points=5, bonus_threshold=0.9
    )
    assert abs(score - 10 * 80 / 90) < 1e-10  # 80% / 90% of base
    assert bonus == 0.0


def test_proportional_bonus_above_threshold():
    score, bonus = proportional(
        10, {"n_trials": 100, "n_passed": 95}, bonus_points=5, bonus_threshold=0.9
    )
    assert score == 10.0  # at or above threshold -> full base
    assert 0.0 < bonus < 5.0


def test_proportional_bonus_all_pass():
    score, bonus = proportional(
        10, {"n_trials": 100, "n_passed": 100}, bonus_points=5, bonus_threshold=0.9
    )
    assert score == 10.0
    assert bonus == 5.0


# ---------------------------------------------------------------------------
# exponential with baseline/ceiling
# ---------------------------------------------------------------------------


def test_exponential_baseline_clips_below():
    score, bonus = exponential(
        10, {"best_test_accuracy": 0.5}, baseline=0.92
    )
    assert score == 0.0
    assert bonus == 0.0


def test_exponential_baseline_at_baseline_is_zero():
    score, bonus = exponential(
        10, {"best_test_accuracy": 0.92}, baseline=0.92
    )
    assert score == 0.0


def test_exponential_baseline_at_target_is_full():
    score, bonus = exponential(
        10, {"best_test_accuracy": 1.0}, baseline=0.92
    )
    assert abs(score - 10.0) < 1e-10


def test_exponential_baseline_with_bonus_threshold():
    # baseline=0.92, target=0.98 (=bonus_threshold), ceiling=0.999
    score, bonus = exponential(
        10,
        {"best_test_accuracy": 0.98},
        bonus_points=5,
        bonus_threshold=0.98,
        baseline=0.92,
        ceiling=0.999,
    )
    assert abs(score - 10.0) < 1e-10
    assert bonus == 0.0


def test_exponential_bonus_ceiling_cap():
    score, bonus = exponential(
        10,
        {"best_test_accuracy": 0.999},
        bonus_points=5,
        bonus_threshold=0.98,
        baseline=0.92,
        ceiling=0.999,
    )
    assert abs(score - 10.0) < 1e-10
    assert abs(bonus - 5.0) < 1e-10


# ---------------------------------------------------------------------------
# speed
# ---------------------------------------------------------------------------


def test_speed_passed_no_speedup():
    score, bonus = speed(
        5, {"passed": True, "slowdown": 1.5}, tier1_bonus=4, tier2_bonus=10
    )
    assert score == 5
    assert bonus == 0.0


def test_speed_failed():
    score, bonus = speed(
        5, {"passed": False, "slowdown": 3.0}, tier1_bonus=4, tier2_bonus=10
    )
    assert score == 0.0
    assert bonus == 0.0


def test_speed_tier1_only():
    score, bonus = speed(
        5, {"passed": True, "slowdown": 0.90}, tier1_bonus=4, tier2_bonus=10
    )
    assert score == 5
    assert bonus == 4.0


def test_speed_both_tiers():
    score, bonus = speed(
        5, {"passed": True, "slowdown": 0.50}, tier1_bonus=4, tier2_bonus=10
    )
    assert score == 5
    assert bonus == 14.0


def test_speed_just_above_tier1_threshold():
    # slowdown=0.96 is not 5% faster; no bonus.
    score, bonus = speed(
        5, {"passed": True, "slowdown": 0.96}, tier1_bonus=4, tier2_bonus=10
    )
    assert bonus == 0.0


def test_speed_missing_slowdown():
    score, bonus = speed(
        5, {"passed": True}, tier1_bonus=4, tier2_bonus=10
    )
    assert score == 5
    assert bonus == 0.0


# ---------------------------------------------------------------------------
# SCORING_FUNCTIONS / DEFAULT_SCORING
# ---------------------------------------------------------------------------


def test_scoring_functions_keys():
    assert "binary" in SCORING_FUNCTIONS
    assert "exponential" in SCORING_FUNCTIONS
    assert "proportional" in SCORING_FUNCTIONS


def test_default_scoring_keys():
    for cmd in ["eval", "grad", "train", "fuzz_eval", "fuzz_grad"]:
        assert cmd in DEFAULT_SCORING


# ---------------------------------------------------------------------------
# compute_score
# ---------------------------------------------------------------------------


def test_compute_score_no_points():
    result = compute_score({"command": "eval"}, {"passed": True})
    assert result is None


def test_compute_score_binary():
    s = compute_score({"command": "eval", "points": 10}, {"passed": True})
    assert s["score"] == 10
    assert s["max_points"] == 10
    assert s["bonus"] == 0.0
    assert s["max_bonus"] == 0


def test_compute_score_binary_fail():
    s = compute_score({"command": "eval", "points": 5}, {"passed": False})
    assert s["score"] == 0.0
    assert s["max_points"] == 5


def test_compute_score_exponential():
    config = {"command": "train", "points": 10}
    result = {"best_test_accuracy": 1.0}
    s = compute_score(config, result)
    assert s["score"] == 10.0
    assert s["max_points"] == 10


def test_compute_score_proportional():
    config = {"command": "fuzz_eval", "points": 8}
    result = {"n_trials": 10, "n_passed": 5}
    s = compute_score(config, result)
    assert s["score"] == 4.0
    assert s["max_points"] == 8


def test_compute_score_explicit_scoring_function():
    config = {"command": "eval", "points": 6, "scoring": {"function": "proportional"}}
    result = {"n_trials": 10, "n_passed": 10}
    s = compute_score(config, result)
    assert s["score"] == 6.0


def test_compute_score_string_scoring():
    config = {"command": "eval", "points": 5, "scoring": "binary"}
    s = compute_score(config, {"passed": True})
    assert s["score"] == 5


def test_compute_score_with_params():
    config = {"command": "train", "points": 10, "scoring": {"function": "exponential", "k": 3}}
    result = {"best_test_accuracy": 1.0}
    s = compute_score(config, result)
    assert s["score"] == 10.0


def test_compute_score_unknown_function():
    config = {"command": "eval", "points": 5, "scoring": {"function": "bogus"}}
    with pytest.raises(ValueError, match="unknown scoring function"):
        compute_score(config, {"passed": True})


def test_compute_score_default_fallback_binary():
    """Unknown command should fall back to 'binary'."""
    config = {"command": "unknown_cmd", "points": 4}
    s = compute_score(config, {"passed": True})
    assert s["score"] == 4
    assert s["max_points"] == 4


def test_compute_score_with_bonus():
    config = {
        "command": "train",
        "points": 10,
        "bonus_points": 5,
        "scoring": {"function": "exponential", "bonus_threshold": 0.9},
    }
    result = {"best_test_accuracy": 1.0}
    s = compute_score(config, result)
    assert s["score"] == 10.0
    assert s["max_points"] == 10
    assert s["bonus"] == 5.0
    assert s["max_bonus"] == 5


def test_compute_score_bonus_only():
    """A test with only bonus_points and no base points."""
    config = {
        "command": "eval",
        "bonus_points": 3,
    }
    s = compute_score(config, {"passed": True})
    assert s["score"] == 0
    assert s["max_points"] == 0
    assert s["bonus"] == 3.0
    assert s["max_bonus"] == 3
