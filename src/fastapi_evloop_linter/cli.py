"""CLI entry point for fastapi-evloop-linter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .checker import EventLoopChecker, format_violation, LintResult


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fastapi-evloop-linter",
        description="Detect event loop blocking calls in FastAPI async endpoints",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to check",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Maximum call graph traversal depth (default: 20)",
    )
    parser.add_argument(
        "--severity",
        choices=["error", "warning", "all"],
        default="all",
        help="Which severities to report (default: all)",
    )

    args = parser.parse_args(argv)

    checker = EventLoopChecker(max_depth=args.max_depth)
    result = LintResult()

    for path_str in args.paths:
        path = Path(path_str)
        if path.is_dir():
            result.merge(checker.check_directory(path))
        elif path.is_file():
            result.merge(checker.check_file(path))
        else:
            print(f"Warning: {path_str} not found", file=sys.stderr)

    # Filter by severity
    violations = result.violations
    if args.severity != "all":
        violations = [v for v in violations if v.severity == args.severity]

    if args.format == "json":
        output = {
            "files_checked": result.files_checked,
            "endpoints_found": result.endpoints_found,
            "violations": [
                {
                    "file": v.filepath,
                    "line": v.line,
                    "col": v.col,
                    "severity": v.severity,
                    "message": v.message,
                    "depth": v.depth,
                    "call_chain": v.call_chain,
                    "reason": v.reason,
                }
                for v in violations
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        for v in sorted(violations, key=lambda v: (v.filepath, v.line)):
            print(format_violation(v))

        if violations:
            print(f"\nFound {len(violations)} violation(s) in {result.files_checked} file(s) "
                  f"({result.endpoints_found} endpoint(s) checked)", file=sys.stderr)

    return 1 if any(v.severity == "error" for v in violations) else 0


if __name__ == "__main__":
    sys.exit(main())
