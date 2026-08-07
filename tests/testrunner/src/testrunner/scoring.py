# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Scoring functions for test results.

Each scoring function maps a test result to a score in [0, max_points],
with optional bonus points for exceeding expectations.
Signature: (max_points, result, **params) -> float

test.json format:
    "points": 10,
    "scoring": {"function": "binary"}           # default for eval/grad
    "scoring": {"function": "exponential", "k": 5}  # for train
    "scoring": {"function": "proportional"}      # for fuzz
    "scoring": {"function": "speed",            # default for bench_eval/grad
                "tier1_bonus": 4, "tier2_bonus": 10}

Bonus points:
    "points": 10,
    "bonus_points": 5,
    "scoring": {"function": "exponential", "k": 5, "bonus_threshold": 0.9,
                "baseline": 0.92, "ceiling": 0.999}

    For exponential scoring, accuracy is rescaled within [baseline, target]
    where target is bonus_threshold when set (else 1.0).  Base points are
    awarded up to the target; bonus points scale from the target up to
    ceiling.  A bonus-only test uses "points": 0 (or omits it) with
    "bonus_points": 5.
"""

import math


def binary(max_points, result, bonus_points=0, bonus_threshold=None):
    """Full points if passed, else 0. Bonus awarded in full if passed."""
    passed = result.get("passed")
    score = max_points if passed else 0.0
    bonus = float(bonus_points) if passed else 0.0
    return score, bonus


def _exp_curve(x, k):
    """Exponential curve mapping [0, 1] -> [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return (math.exp(k * x) - 1) / (math.exp(k) - 1)


def exponential(
    max_points,
    result,
    k=5,
    bonus_points=0,
    bonus_threshold=None,
    baseline=0.0,
    ceiling=1.0,
):
    """Exponential scoring based on best_test_accuracy.

    Accuracy is rescaled within [baseline, target], where ``target`` is
    ``bonus_threshold`` if bonus scoring applies, otherwise ``1.0``.
    Accuracy at or below ``baseline`` gives 0 points; accuracy at or above
    ``target`` gives full ``max_points``.  The rescaled fraction is passed
    through an exponential curve ``(exp(k*x) - 1) / (exp(k) - 1)`` so that
    higher k concentrates more of the reward near the target.

    When ``bonus_points`` and ``bonus_threshold`` are set, accuracy beyond
    ``bonus_threshold`` maps through the same exponential curve onto
    ``[0, bonus_points]``, reaching full bonus at ``ceiling``.
    """
    accuracy = result.get("best_test_accuracy", 0.0)

    has_bonus = (
        bonus_points > 0 and bonus_threshold is not None and bonus_threshold < 1.0
    )
    target = bonus_threshold if has_bonus else 1.0

    if target > baseline:
        base_frac = max(0.0, min(1.0, (accuracy - baseline) / (target - baseline)))
    else:
        base_frac = 1.0 if accuracy >= target else 0.0
    score = max_points * _exp_curve(base_frac, k)

    if has_bonus and accuracy > bonus_threshold:
        if ceiling > bonus_threshold:
            bonus_frac = max(
                0.0, min(1.0, (accuracy - bonus_threshold) / (ceiling - bonus_threshold))
            )
        else:
            bonus_frac = 1.0
        bonus = bonus_points * _exp_curve(bonus_frac, k)
    else:
        bonus = 0.0

    return score, bonus


def speed(
    max_points,
    result,
    bonus_points=0,
    bonus_threshold=None,
    tier1_bonus=0,
    tier2_bonus=0,
    tier1_speedup=0.05,
    tier2_speedup=0.25,
):
    """Speed scoring with two-tier speedup bonus.

    Base points are awarded in full if the benchmark passed (the SUT ran
    within ``max_slowdown`` of the reference).  Two independent bonuses are
    awarded on top when the SUT is faster than the reference:

    * tier 1 (default ``tier1_speedup=0.05``): slowdown <= 0.95
      (at least 5% faster than the reference)
    * tier 2 (default ``tier2_speedup=0.25``): slowdown <= 0.75
      (at least 25% faster than the reference)

    Bonuses are cumulative — a 30% speedup earns ``tier1_bonus + tier2_bonus``.
    ``bonus_points`` is carried through for display as the maximum possible
    bonus and should equal ``tier1_bonus + tier2_bonus``.
    """
    passed = result.get("passed")
    score = max_points if passed else 0.0
    slowdown = result.get("slowdown")
    bonus = 0.0
    if slowdown is not None:
        if slowdown <= 1.0 - tier1_speedup:
            bonus += tier1_bonus
        if slowdown <= 1.0 - tier2_speedup:
            bonus += tier2_bonus
    return score, bonus


def proportional(max_points, result, bonus_points=0, bonus_threshold=None):
    """Score proportional to the fraction of passed trials.

    When bonus_points and bonus_threshold are set, base points are awarded
    for pass rates up to bonus_threshold, bonus points for rates beyond it.
    """
    n_trials = result.get("n_trials", 0)
    if n_trials == 0:
        return 0.0, 0.0
    n_passed = result.get("n_passed", 0)
    frac = n_passed / n_trials

    if bonus_points > 0 and bonus_threshold is not None and bonus_threshold < 1.0:
        base_frac = min(frac / bonus_threshold, 1.0) if bonus_threshold > 0 else 1.0
        score = max_points * base_frac
        if frac > bonus_threshold:
            bonus_frac = (frac - bonus_threshold) / (1.0 - bonus_threshold)
            bonus = bonus_points * bonus_frac
        else:
            bonus = 0.0
    else:
        score = max_points * frac
        bonus = 0.0

    return score, bonus


SCORING_FUNCTIONS = {
    "binary": binary,
    "exponential": exponential,
    "proportional": proportional,
    "speed": speed,
}

DEFAULT_SCORING = {
    "eval": "binary",
    "grad": "binary",
    "bounds": "binary",
    "affine_bounds": "binary",
    "verify": "binary",
    "verify2": "binary",
    "train": "exponential",
    "fuzz_eval": "proportional",
    "fuzz_grad": "proportional",
    "fuzz_bounds": "proportional",
    "fuzz_affine_bounds": "proportional",
    "bench_eval": "speed",
    "bench_grad": "speed",
    "bench_bounds": "speed",
    "bench_verify": "speed",
}


def compute_score(config, result):
    """Compute the score for a test result based on its config.

    Returns a dict with keys ``score``, ``max_points``, ``bonus``,
    ``max_bonus``, or *None* if the test has neither points nor bonus_points.
    """
    max_points = config.get("points", 0)
    max_bonus = config.get("bonus_points", 0)
    if not max_points and not max_bonus:
        return None

    scoring_spec = config.get("scoring", {})
    if isinstance(scoring_spec, str):
        func_name = scoring_spec
        params = {}
    else:
        func_name = scoring_spec.get("function", DEFAULT_SCORING.get(config.get("command"), "binary"))
        params = {k: v for k, v in scoring_spec.items() if k != "function"}

    if func_name not in SCORING_FUNCTIONS:
        raise ValueError(
            f"unknown scoring function {func_name!r}, "
            f"expected one of {sorted(SCORING_FUNCTIONS)}"
        )

    score, bonus = SCORING_FUNCTIONS[func_name](
        max_points, result, bonus_points=max_bonus, **params
    )
    return {"score": score, "max_points": max_points, "bonus": bonus, "max_bonus": max_bonus}
