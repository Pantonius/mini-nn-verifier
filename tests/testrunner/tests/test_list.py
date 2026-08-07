# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.list (discover_tests, print_cli, print_json, main)."""

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from testrunner.list import discover_tests, main, print_cli, print_json


# ---------------------------------------------------------------------------
# discover_tests
# ---------------------------------------------------------------------------


def _make_test(directory, name, command):
    """Create a test.json in directory/name/."""
    d = directory / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "test.json").write_text(json.dumps({"command": command}))
    return d


def test_discover_tests_empty_dir(tmp_path):
    assert discover_tests(tmp_path) == {}


def test_discover_tests_single_top_level(tmp_path):
    _make_test(tmp_path, "t1", "eval")
    groups = discover_tests(tmp_path)
    assert "." in groups
    assert any(t["name"] == "t1" for t in groups["."])


def test_discover_tests_grouped(tmp_path):
    _make_test(tmp_path / "group1", "t1", "eval")
    _make_test(tmp_path / "group1", "t2", "grad")
    groups = discover_tests(tmp_path)
    assert "group1" in groups
    assert len(groups["group1"]) == 2


def test_discover_tests_includes_command(tmp_path):
    _make_test(tmp_path, "mytest", "train")
    groups = discover_tests(tmp_path)
    test = groups["."][0]
    assert test["command"] == "train"
    assert test["path"] == "mytest"


# ---------------------------------------------------------------------------
# discover_tests — max_points / max_bonus / access / command
# ---------------------------------------------------------------------------


def _make_test_with_config(directory, name, config):
    d = directory / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "test.json").write_text(json.dumps(config))
    return d


def test_discover_tests_includes_max_points_and_bonus(tmp_path):
    _make_test_with_config(
        tmp_path,
        "t1",
        {"command": "eval", "points": 7, "bonus_points": 3},
    )
    info = discover_tests(tmp_path)["."][0]
    assert info["max_points"] == 7
    assert info["max_bonus"] == 3
    assert info["command"] == "eval"


def test_discover_tests_defaults_max_points_to_zero(tmp_path):
    _make_test(tmp_path, "t1", "eval")
    info = discover_tests(tmp_path)["."][0]
    assert info["max_points"] == 0
    assert info["max_bonus"] == 0


def test_discover_tests_access_open_and_closed(tmp_path):
    _make_test_with_config(tmp_path, "open_test", {"command": "eval"})
    _make_test_with_config(
        tmp_path, "secret", {"command": "eval", "access": "closed"}
    )
    # Also a test under a "closed" subdirectory — access inferred from path.
    _make_test_with_config(
        tmp_path / "closed", "t", {"command": "eval"}
    )
    groups = discover_tests(tmp_path)
    by_path = {t["path"]: t for g in groups.values() for t in g}
    assert by_path["open_test"]["access"] == "open"
    assert by_path["secret"]["access"] == "closed"
    assert by_path["closed/t"]["access"] == "closed"


def test_discover_tests_max_points_sum_matches_full_run(tmp_path):
    """Per-directory sum of max_points equals what a full run reports."""
    from unittest.mock import patch
    from testrunner.commands.common import SubprocessResult
    from testrunner.__main__ import run_tests

    # Build a small tree: two subdirs, each with multiple tests of various points
    for sub, specs in [
        ("ms1", [("a", 5), ("b", 0), ("c", 10)]),
        ("ms2", [("x", 3), ("y", 7)]),
    ]:
        for name, pts in specs:
            _make_test_with_config(
                tmp_path / sub,
                name,
                {
                    "command": "eval",
                    "network": "n.mininn",
                    "inputs": [],
                    "expected_outputs": [],
                    "points": pts,
                },
            )

    groups = discover_tests(tmp_path)

    # Sum max_points per top-level milestone.
    def sum_for_prefix(prefix):
        total = 0
        for group, tests in groups.items():
            if group == prefix or group.startswith(prefix + "/"):
                total += sum(t["max_points"] for t in tests)
        return total

    for sub, expected in [("ms1", 15), ("ms2", 10)]:
        # What a full run of just this subdir would report.
        totals = {}

        class Handler:
            def test_starting(self, *a): pass
            def test_finished(self, *a): pass
            def all_finished(
                self, n_passed, n_failed,
                total_score=0.0, total_max_points=0.0,
                total_bonus=0.0, total_max_bonus=0.0,
            ):
                totals["max_points"] = total_max_points

        mock_result = SubprocessResult(0, "", "")
        with patch("testrunner.__main__.run_subprocess", return_value=mock_result):
            run_tests(tmp_path / sub, "local", "sut", output_handler=Handler())

        assert totals["max_points"] == expected
        assert sum_for_prefix(sub) == expected


def test_discover_tests_nested_group(tmp_path):
    _make_test(tmp_path / "a" / "b", "t1", "eval")
    groups = discover_tests(tmp_path)
    # group should be "a/b"
    assert any("a" in g for g in groups.keys())


# ---------------------------------------------------------------------------
# print_cli
# ---------------------------------------------------------------------------


def _capture_stdout(fn):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        fn()
    return buf.getvalue()


def _capture_stderr(fn):
    buf = io.StringIO()
    with patch("sys.stderr", buf):
        fn()
    return buf.getvalue()


def test_print_cli_shows_groups_and_tests():
    groups = {
        "group1": [{"name": "t1", "path": "group1/t1", "command": "eval"}],
        "group2": [{"name": "t2", "path": "group2/t2", "command": "train"}],
    }
    out = _capture_stdout(lambda: print_cli(groups))
    assert "group1/" in out
    assert "t1" in out
    assert "eval" in out


def test_print_cli_total_count():
    groups = {
        ".": [
            {"name": "a", "path": "a", "command": "eval"},
            {"name": "b", "path": "b", "command": "grad"},
        ]
    }
    err = _capture_stderr(lambda: print_cli(groups))
    assert "2 tests total" in err


# ---------------------------------------------------------------------------
# print_json
# ---------------------------------------------------------------------------


def test_print_json_is_valid_json():
    groups = {"g": [{"name": "t", "path": "g/t", "command": "eval"}]}
    out = _capture_stdout(lambda: print_json(groups))
    parsed = json.loads(out)
    assert "g" in parsed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_cli_output(tmp_path):
    _make_test(tmp_path, "t1", "eval")
    with patch("sys.argv", ["testrunner.list", str(tmp_path)]):
        # Should succeed (prints to stdout/stderr, exits 0 implicitly)
        main()


def test_main_json_output(tmp_path):
    _make_test(tmp_path, "t1", "eval")
    out = _capture_stdout(
        lambda: (
            setattr(__import__("sys"), "argv",
                    ["testrunner.list", str(tmp_path), "--output", "json"]),
            main(),
        )
    )
    # Output might be empty if main() used sys.stdout directly; use patch instead
    with patch("sys.argv", ["testrunner.list", str(tmp_path), "--output", "json"]):
        out = _capture_stdout(main)
    parsed = json.loads(out)
    assert "." in parsed


def test_main_no_tests_exits_1(tmp_path):
    with patch("sys.argv", ["testrunner.list", str(tmp_path)]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_main_module_guard(tmp_path):
    """The if __name__ == '__main__': guard in list.py calls main()."""
    import runpy
    _make_test(tmp_path, "t", "eval")
    with patch("sys.argv", ["testrunner.list", str(tmp_path)]):
        runpy.run_module("testrunner.list", run_name="__main__", alter_sys=False)
