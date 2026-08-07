# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Output handlers for the test runner.

Two modes:
- JsonOutputHandler: emits one JSON line per test result to stdout (JSONL)
- CliOutputHandler: live progress display with colored pass/fail on stderr
"""

import json
import sys


class JsonOutputHandler:
    """Emit one JSON line per completed test to stdout."""

    def test_starting(self, test_path, index, total):
        pass

    def test_stdout_line(self, line):
        pass

    def fuzz_trial_finished(self, test_path, trial, total_trials, passed, error):
        record = {
            "test": test_path,
            "event": "fuzz_trial",
            "trial": trial,
            "total_trials": total_trials,
            "passed": passed,
        }
        if error is not None:
            record["error"] = error
        print(json.dumps(record), flush=True)

    def bench_rep_finished(self, test_path, rep, total_reps, elapsed):
        record = {
            "test": test_path,
            "event": "bench_rep",
            "rep": rep,
            "total_reps": total_reps,
            "elapsed": round(elapsed, 6),
        }
        print(json.dumps(record), flush=True)

    def test_finished(self, test_path, result):
        record = {"test": test_path, **result}
        # Drop individual failures list from JSON — the summary is sufficient
        record.pop("failures", None)
        print(json.dumps(record), flush=True)

    def all_finished(
        self, total_passed, total_failed, total_score=0.0, total_max_points=0.0,
        total_bonus=0.0, total_max_bonus=0.0,
    ):
        summary = {
            "event": "summary",
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_score": total_score,
            "total_max_points": total_max_points,
        }
        if total_max_bonus > 0:
            summary["total_bonus"] = total_bonus
            summary["total_max_bonus"] = total_max_bonus
        print(json.dumps(summary), flush=True)


class CliOutputHandler:
    """Live progress display on stderr with colored pass/fail indicators.

    Uses a simple two-phase approach: while a test runs, a single
    "Running ..." status line is shown (overwritten in-place for fuzz
    progress).  When the test finishes the status line is replaced with
    the final result.  Stdout lines and failure details are buffered and
    only printed on failure — no multi-line cursor navigation needed.
    """

    # BAM10 colorblind-safe palette (purple fail / green pass)
    _GREEN = "\033[38;2;124;167;86m"
    _RED = "\033[38;2;157;60;132m"
    _BOLD = "\033[1m"
    _DIM = "\033[2m"
    _RESET = "\033[0m"

    def __init__(self):
        self._tty = sys.stderr.isatty()
        self._stdout_buf: list[str] = []

    def test_starting(self, test_path, index, total):
        self._stdout_buf = []
        msg = f"[{index}/{total}] Running {test_path}..."
        if self._tty:
            # Print status line (will be overwritten by fuzz progress or result)
            sys.stderr.write(f"{msg}\n")
            sys.stderr.flush()
        else:
            print(msg, end="", file=sys.stderr, flush=True)

    def test_stdout_line(self, line):
        # Buffer stdout — only shown on failure
        self._stdout_buf.append(line)

    def fuzz_trial_finished(self, test_path, trial, total_trials, passed, error):
        if self._tty:
            line = f"  fuzz [{trial}/{total_trials}]"
            # Overwrite the status line above
            sys.stderr.write(f"\033[1A\r\033[2K{line}\n")
            sys.stderr.flush()

    def bench_rep_finished(self, test_path, rep, total_reps, elapsed):
        if self._tty:
            line = f"  bench [{rep}/{total_reps}] {elapsed:.4f}s"
            sys.stderr.write(f"\033[1A\r\033[2K{line}\n")
            sys.stderr.flush()

    def test_finished(self, test_path, result):
        passed = result.get("passed", False)
        saved_dir = result.get("saved_failures_dir")
        is_fuzz = "n_trials" in result
        score_suffix = self._format_score(result)

        if self._tty:
            # Clear the status/progress line and replace with result
            sys.stderr.write("\033[1A\r\033[2K")
            if passed:
                sys.stderr.write(
                    f"  {self._GREEN}\u2713{self._RESET} {test_path}{score_suffix}\n"
                )
            else:
                sys.stderr.write(
                    f"  {self._RED}\u2717{self._RESET} {test_path}{score_suffix}\n"
                )
                if is_fuzz:
                    self._write_fuzz_summary(result, test_path, saved_dir)
                else:
                    self._write_test_failure(result)
            sys.stderr.flush()
        else:
            # Non-TTY: simple sequential output
            if passed:
                print(f" PASS{score_suffix}", file=sys.stderr, flush=True)
            else:
                if is_fuzz:
                    self._write_fuzz_summary(result, test_path, saved_dir)
                else:
                    self._write_test_failure(result)
        self._stdout_buf = []

    @staticmethod
    def _fmt_num(n):
        """Format a number: integer if whole, one decimal otherwise."""
        return str(int(n)) if n == int(n) else f"{n:.1f}"

    @classmethod
    def _format_score(cls, result):
        """Format a score suffix like ' (7.5/10)' or ' (7/10 + 3/5 bonus)'."""
        if "score" not in result:
            return ""
        score = result["score"]
        max_pts = result["max_points"]
        bonus = result.get("bonus", 0)
        max_bonus = result.get("max_bonus", 0)
        base = f"{cls._fmt_num(score)}/{cls._fmt_num(max_pts)}"
        if max_bonus > 0:
            return f" ({base} + {cls._fmt_num(bonus)}/{cls._fmt_num(max_bonus)} bonus)"
        return f" ({base})"

    def _write_test_failure(self, result):
        """Print failure details for a non-fuzz test."""
        error = result.get("error", "unknown error")
        warnings = result.get("warnings", [])
        if self._stdout_buf:
            sys.stderr.write(f"    {self._DIM}--- stdout ---{self._RESET}\n")
            for line in self._stdout_buf:
                sys.stderr.write(f"    {line}\n")
            sys.stderr.write(f"    {self._DIM}--- end stdout ---{self._RESET}\n")
        for w in warnings:
            sys.stderr.write(f"    warning: {w}\n")
        sys.stderr.write(f"    {self._RED}{error}{self._RESET}\n")

    def _write_fuzz_summary(self, result, test_path, saved_dir):
        """Print a short fuzz failure summary with a pointer to testrunner.show."""
        n_failed = result.get("n_trials", 0) - result.get("n_passed", 0)
        n_trials = result.get("n_trials", 0)
        failure_summary = result.get("failure_summary", {})
        n_types = len(failure_summary)
        sys.stderr.write(
            f"    {self._RED}{n_failed}/{n_trials} trials failed"
            f" ({n_types} distinct error type{'s' if n_types != 1 else ''})"
            f"{self._RESET}\n"
        )
        if saved_dir:
            sys.stderr.write(f"    Failing cases saved to: {saved_dir}\n")
        sys.stderr.write(f"    Run `python -m testrunner.show {test_path}` for details\n")

    def all_finished(
        self, total_passed, total_failed, total_score=0.0, total_max_points=0.0,
        total_bonus=0.0, total_max_bonus=0.0,
    ):
        total = total_passed + total_failed
        if total_failed == 0:
            color = self._GREEN
        else:
            color = self._RED
        score_part = ""
        if total_max_points > 0 or total_max_bonus > 0:
            score_part = f" | Score: {self._fmt_num(total_score)}/{self._fmt_num(total_max_points)}"
            if total_max_bonus > 0:
                score_part += (
                    f" + {self._fmt_num(total_bonus)}/{self._fmt_num(total_max_bonus)} bonus"
                )
        print(
            f"\n{self._BOLD}{color}{total_passed} passed, {total_failed} failed{self._RESET}"
            f" ({total} total){score_part}",
            file=sys.stderr,
            flush=True,
        )
