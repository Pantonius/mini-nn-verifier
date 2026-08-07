# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""List tests grouped by directory.

Usage:
    python -m testrunner.list <test_dir>
    python -m testrunner.list <test_dir> --output json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _resolve_access(test_dir, config):
    """Return ``"closed"`` or ``"open"`` for a test.

    Mirrors ``testrunner.__main__.is_closed``: a test is closed when its
    path contains the substring ``"closed"`` or when its config sets
    ``"access": "closed"``.
    """
    if "closed" in str(test_dir) or config.get("access") == "closed":
        return "closed"
    return "open"


def discover_tests(root_dir):
    """Discover all tests under root_dir and group by parent directory.

    Returns a dict mapping group names to lists of test info dicts.  Each
    info dict carries:

    - ``name``, ``path``, ``command``
    - ``max_points``, ``max_bonus`` — the denominators a full run would
      report (read from ``test.json``: ``points``/``bonus_points``,
      defaulting to ``0``)
    - ``access`` — ``"open"`` or ``"closed"``
    - ``description`` — free-form human-readable text from ``test.json``
      (empty string if the test has no ``description`` field)
    """
    root_dir = Path(root_dir).resolve()
    test_dirs = sorted(p.parent for p in root_dir.rglob("test.json"))

    groups = defaultdict(list)
    for test_dir in test_dirs:
        rel = test_dir.relative_to(root_dir)
        parts = rel.parts
        if len(parts) > 1:
            group = str(Path(*parts[:-1]))
            name = parts[-1]
        else:
            group = "."
            name = str(rel)

        config = json.loads((test_dir / "test.json").read_text())
        groups[group].append({
            "name": name,
            "path": str(rel),
            "command": config.get("command"),
            "max_points": config.get("points", 0),
            "max_bonus": config.get("bonus_points", 0),
            "access": _resolve_access(test_dir, config),
            "description": config.get("description", ""),
        })

    return dict(groups)


def print_cli(groups):
    """Print tests grouped by directory in a human-readable format."""
    total = sum(len(tests) for tests in groups.values())
    for group, tests in sorted(groups.items()):
        print(f"{group}/ ({len(tests)} tests)")
        for t in tests:
            cmd = t["command"]
            print(f"  {t['name']}  [{cmd}]")
    print(f"\n{total} tests total", file=sys.stderr)


def print_json(groups):
    """Print tests grouped by directory as JSON."""
    print(json.dumps(groups, indent=2))


def main():
    parser = argparse.ArgumentParser(description="List tests grouped by directory.")
    parser.add_argument("test_dir", type=str)
    parser.add_argument(
        "--output",
        choices=["cli", "json"],
        default="cli",
        help="Output mode: cli (default) for human-readable, json for JSON",
    )
    args = parser.parse_args()

    groups = discover_tests(args.test_dir)
    if not groups:
        print("No tests found.", file=sys.stderr)
        sys.exit(1)

    if args.output == "json":
        print_json(groups)
    else:
        print_cli(groups)


if __name__ == "__main__":
    main()
