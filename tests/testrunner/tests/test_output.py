# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.output (JsonOutputHandler and CliOutputHandler)."""

import io
import json
import sys
from unittest.mock import patch, MagicMock

import pytest

from testrunner.output import CliOutputHandler, JsonOutputHandler


# ---------------------------------------------------------------------------
# JsonOutputHandler
# ---------------------------------------------------------------------------


def _capture_stdout(fn):
    """Run fn() and return everything printed to stdout as a string."""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        fn()
    return buf.getvalue()


def test_json_test_starting_is_noop():
    handler = JsonOutputHandler()
    # Should not raise and produce no output
    out = _capture_stdout(lambda: handler.test_starting("my/test", 1, 10))
    assert out == ""


def test_json_test_stdout_line_is_noop():
    handler = JsonOutputHandler()
    out = _capture_stdout(lambda: handler.test_stdout_line("some line"))
    assert out == ""


def test_json_fuzz_trial_finished_passed():
    handler = JsonOutputHandler()
    out = _capture_stdout(
        lambda: handler.fuzz_trial_finished("t/foo", 3, 100, True, None)
    )
    record = json.loads(out.strip())
    assert record["event"] == "fuzz_trial"
    assert record["trial"] == 3
    assert record["total_trials"] == 100
    assert record["passed"] is True
    assert "error" not in record


def test_json_fuzz_trial_finished_with_error():
    handler = JsonOutputHandler()
    out = _capture_stdout(
        lambda: handler.fuzz_trial_finished("t/foo", 1, 5, False, "crash!")
    )
    record = json.loads(out.strip())
    assert record["error"] == "crash!"


def test_json_test_finished_pass():
    handler = JsonOutputHandler()
    result = {"passed": True, "error": None, "output_files": []}
    out = _capture_stdout(lambda: handler.test_finished("t/bar", result))
    record = json.loads(out.strip())
    assert record["test"] == "t/bar"
    assert record["passed"] is True


def test_json_test_finished_drops_failures():
    handler = JsonOutputHandler()
    result = {"passed": False, "error": "e", "failures": [{"trial": 0}]}
    out = _capture_stdout(lambda: handler.test_finished("t/bar", result))
    record = json.loads(out.strip())
    assert "failures" not in record


def test_json_all_finished_no_points():
    handler = JsonOutputHandler()
    out = _capture_stdout(lambda: handler.all_finished(5, 2))
    record = json.loads(out.strip())
    assert record["event"] == "summary"
    assert record["total_passed"] == 5
    assert record["total_failed"] == 2
    assert record["total_score"] == 0.0
    assert record["total_max_points"] == 0.0
    assert "total_bonus" not in record


# ---------------------------------------------------------------------------
# CliOutputHandler — non-TTY mode (isatty() = False)
# ---------------------------------------------------------------------------


def _make_cli_non_tty():
    handler = CliOutputHandler.__new__(CliOutputHandler)
    handler._tty = False
    handler._stdout_buf = []
    return handler


def _capture_stderr(fn):
    buf = io.StringIO()
    with patch("sys.stderr", buf):
        fn()
    return buf.getvalue()


def test_cli_non_tty_test_starting():
    handler = _make_cli_non_tty()
    out = _capture_stderr(lambda: handler.test_starting("t/foo", 1, 5))
    assert "Running t/foo" in out


def test_cli_non_tty_test_finished_pass():
    handler = _make_cli_non_tty()
    handler.test_starting("t/foo", 1, 5)
    out = _capture_stderr(lambda: handler.test_finished("t/foo", {"passed": True}))
    assert "PASS" in out


def test_cli_non_tty_test_finished_fail_non_fuzz():
    handler = _make_cli_non_tty()
    out = _capture_stderr(
        lambda: handler.test_finished("t/foo", {"passed": False, "error": "bad result"})
    )
    assert "bad result" in out


def test_cli_non_tty_test_finished_fail_with_stdout_buf():
    handler = _make_cli_non_tty()
    handler._stdout_buf = ["captured line 1", "captured line 2"]
    out = _capture_stderr(
        lambda: handler.test_finished("t/foo", {"passed": False, "error": "err"})
    )
    assert "captured line 1" in out
    assert "captured line 2" in out


def test_cli_non_tty_test_finished_fail_with_warnings():
    handler = _make_cli_non_tty()
    out = _capture_stderr(
        lambda: handler.test_finished(
            "t/foo", {"passed": False, "error": "err", "warnings": ["w1", "w2"]}
        )
    )
    assert "w1" in out
    assert "w2" in out


def test_cli_non_tty_test_finished_fail_fuzz():
    handler = _make_cli_non_tty()
    result = {
        "passed": False,
        "error": "2 trial(s) failed",
        "n_trials": 10,
        "n_passed": 8,
        "failure_summary": {"crash": 2},
    }
    out = _capture_stderr(lambda: handler.test_finished("t/fuzz", result))
    assert "2/10" in out


def test_cli_non_tty_test_finished_fuzz_with_saved_dir(tmp_path):
    handler = _make_cli_non_tty()
    result = {
        "passed": False,
        "n_trials": 5,
        "n_passed": 3,
        "failure_summary": {"e": 2},
        "saved_failures_dir": str(tmp_path / "failures"),
    }
    out = _capture_stderr(lambda: handler.test_finished("t/fuzz", result))
    assert "Failing cases saved to" in out


def test_cli_non_tty_all_finished_all_pass():
    handler = _make_cli_non_tty()
    out = _capture_stderr(lambda: handler.all_finished(5, 0))
    assert "5 passed" in out
    assert "0 failed" in out


def test_cli_non_tty_all_finished_some_fail():
    handler = _make_cli_non_tty()
    out = _capture_stderr(lambda: handler.all_finished(3, 2))
    assert "3 passed" in out
    assert "2 failed" in out


def test_cli_non_tty_fuzz_trial_finished_is_noop():
    handler = _make_cli_non_tty()
    # No TTY — fuzz_trial_finished should not crash
    out = _capture_stderr(lambda: handler.fuzz_trial_finished("t", 1, 10, True, None))
    assert out == ""


def test_cli_stdout_buf_cleared_after_test_finished():
    handler = _make_cli_non_tty()
    handler._stdout_buf = ["line"]
    handler.test_finished("t", {"passed": True})
    assert handler._stdout_buf == []


def test_cli_stdout_line_buffered():
    handler = _make_cli_non_tty()
    handler.test_stdout_line("hello")
    handler.test_stdout_line("world")
    assert handler._stdout_buf == ["hello", "world"]


def test_cli_test_starting_clears_buf():
    handler = _make_cli_non_tty()
    handler._stdout_buf = ["old"]
    handler.test_starting("t", 1, 1)
    assert handler._stdout_buf == []


# ---------------------------------------------------------------------------
# CliOutputHandler — TTY mode (isatty() = True)
# ---------------------------------------------------------------------------


def _make_cli_tty():
    handler = CliOutputHandler.__new__(CliOutputHandler)
    handler._tty = True
    handler._stdout_buf = []
    return handler


def test_cli_tty_test_starting():
    handler = _make_cli_tty()
    out = _capture_stderr(lambda: handler.test_starting("t/foo", 2, 10))
    assert "Running t/foo" in out


def test_cli_tty_test_finished_pass():
    handler = _make_cli_tty()
    out = _capture_stderr(lambda: handler.test_finished("t/foo", {"passed": True}))
    assert "t/foo" in out


def test_cli_tty_test_finished_fail_non_fuzz():
    handler = _make_cli_tty()
    out = _capture_stderr(
        lambda: handler.test_finished("t/foo", {"passed": False, "error": "oops"})
    )
    assert "oops" in out


def test_cli_tty_test_finished_fail_fuzz():
    handler = _make_cli_tty()
    result = {
        "passed": False,
        "n_trials": 10,
        "n_passed": 7,
        "failure_summary": {"e": 3},
    }
    out = _capture_stderr(lambda: handler.test_finished("t/fuzz", result))
    assert "3/10" in out


def test_cli_tty_fuzz_trial_finished():
    handler = _make_cli_tty()
    out = _capture_stderr(lambda: handler.fuzz_trial_finished("t", 5, 100, True, None))
    assert "fuzz [5/100]" in out


# ---------------------------------------------------------------------------
# JsonOutputHandler — all_finished with scoring
# ---------------------------------------------------------------------------


def test_json_all_finished_with_score():
    handler = JsonOutputHandler()
    out = _capture_stdout(lambda: handler.all_finished(3, 1, total_score=7.5, total_max_points=10.0))
    record = json.loads(out.strip())
    assert record["event"] == "summary"
    assert record["total_score"] == 7.5
    assert record["total_max_points"] == 10.0


def test_json_all_finished_zero_max_points():
    """all_finished with total_max_points=0 should still emit a summary event."""
    handler = JsonOutputHandler()
    out = _capture_stdout(lambda: handler.all_finished(5, 0, total_score=0.0, total_max_points=0.0))
    record = json.loads(out.strip())
    assert record["event"] == "summary"
    assert record["total_passed"] == 5
    assert record["total_score"] == 0.0


# ---------------------------------------------------------------------------
# CliOutputHandler — score display
# ---------------------------------------------------------------------------


def test_cli_non_tty_test_finished_with_score():
    handler = _make_cli_non_tty()
    result = {"passed": True, "score": 10, "max_points": 10}
    out = _capture_stderr(lambda: handler.test_finished("t", result))
    assert "(10/10)" in out


def test_cli_non_tty_test_finished_score_float():
    handler = _make_cli_non_tty()
    result = {"passed": True, "score": 7.5, "max_points": 10.0}
    out = _capture_stderr(lambda: handler.test_finished("t", result))
    assert "(7.5/10)" in out


def test_cli_non_tty_all_finished_with_score_integers():
    handler = _make_cli_non_tty()
    out = _capture_stderr(lambda: handler.all_finished(2, 1, total_score=8.0, total_max_points=10.0))
    assert "Score: 8/10" in out


def test_cli_non_tty_all_finished_with_score_floats():
    handler = _make_cli_non_tty()
    out = _capture_stderr(lambda: handler.all_finished(1, 0, total_score=7.5, total_max_points=10.0))
    assert "Score: 7.5/10" in out


def test_cli_non_tty_all_finished_no_score():
    handler = _make_cli_non_tty()
    out = _capture_stderr(lambda: handler.all_finished(2, 0, total_score=0.0, total_max_points=0.0))
    assert "Score:" not in out


def test_cli_tty_test_finished_with_score():
    handler = _make_cli_tty()
    result = {"passed": True, "score": 5, "max_points": 5}
    out = _capture_stderr(lambda: handler.test_finished("t", result))
    assert "(5/5)" in out


def test_cli_write_fuzz_summary_singular_error_type(tmp_path):
    handler = _make_cli_non_tty()
    result = {
        "passed": False,
        "n_trials": 3,
        "n_passed": 2,
        "failure_summary": {"only_one_type": 1},
    }
    out = _capture_stderr(lambda: handler.test_finished("t/fuzz", result))
    # Singular: "1 distinct error type" (no trailing 's')
    assert "1 distinct error type" in out


# ---------------------------------------------------------------------------
# Bonus display (output.py lines 52-53, 148, 192)
# ---------------------------------------------------------------------------


def test_json_bench_rep_finished():
    handler = JsonOutputHandler()
    out = _capture_stdout(lambda: handler.bench_rep_finished("t/foo", 2, 10, 0.123456))
    record = json.loads(out.strip())
    assert record["event"] == "bench_rep"
    assert record["rep"] == 2
    assert record["total_reps"] == 10
    assert abs(record["elapsed"] - 0.123456) < 1e-4


def test_cli_tty_bench_rep_finished():
    handler = _make_cli_tty()
    out = _capture_stderr(lambda: handler.bench_rep_finished("t/foo", 3, 10, 0.5))
    assert "bench [3/10]" in out


def test_cli_non_tty_bench_rep_finished_is_noop():
    handler = _make_cli_non_tty()
    out = _capture_stderr(lambda: handler.bench_rep_finished("t/foo", 1, 10, 0.1))
    assert out == ""


def test_json_all_finished_with_bonus():
    """all_finished with total_max_bonus > 0 should include bonus fields in JSON."""
    handler = JsonOutputHandler()
    out = _capture_stdout(
        lambda: handler.all_finished(3, 1, total_score=7.5, total_max_points=10.0,
                                     total_bonus=3.0, total_max_bonus=5.0)
    )
    record = json.loads(out.strip())
    assert record["total_bonus"] == 3.0
    assert record["total_max_bonus"] == 5.0


def test_cli_non_tty_test_finished_with_bonus():
    """test_finished with max_bonus > 0 should show bonus in score suffix."""
    handler = _make_cli_non_tty()
    result = {"passed": True, "score": 7.5, "max_points": 10.0, "bonus": 2.5, "max_bonus": 5.0}
    out = _capture_stderr(lambda: handler.test_finished("t", result))
    assert "7.5/10" in out
    assert "bonus" in out


def test_cli_non_tty_all_finished_with_bonus():
    """all_finished with total_max_bonus > 0 should show bonus in score part."""
    handler = _make_cli_non_tty()
    out = _capture_stderr(
        lambda: handler.all_finished(2, 0, total_score=10.0, total_max_points=10.0,
                                     total_bonus=3.0, total_max_bonus=5.0)
    )
    assert "bonus" in out
    assert "3" in out
    assert "5" in out
