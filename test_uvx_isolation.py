#!/usr/bin/env python3
"""Regression: endpoint detection must work when the framework module isn't
installed in the linter's Python environment (the typical ``uvx --from git+``
scenario).

Before the fix, the new generic-detection rewrite required ``fastapi`` /
``starlette`` / etc. to be importable in the same env as the linter, so
``resolve_module_origin("fastapi") -> NOT_INSTALLED`` caused 0 endpoints to
be found and 0 violations to be reported, regardless of how many blocking
calls the user's code contained.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter import endpoints, introspect
from fastapi_evloop_linter.checker import EventLoopChecker


SOURCE = """\
from pathlib import Path
from fastapi import APIRouter

router = APIRouter()


def _get_pdf_content_and_name(pdf_path):
    p = Path(pdf_path)
    if p.exists():
        return p.read_bytes(), p.name
    return None, None


def _prepare_and_execute_job(pdf_path):
    return _get_pdf_content_and_name(pdf_path)


@router.post("/specs-matrix/jobs")
async def create_specs_matrix_job(pdf_path: str):
    return _prepare_and_execute_job(pdf_path)
"""


def _origin_with_fastapi_uninstalled(module_name: str):
    """resolve_module_origin replacement: pretend fastapi & friends are not installed."""
    if module_name.split(".")[0] in {"fastapi", "starlette", "litestar", "aiohttp"}:
        return introspect.ModuleOrigin.NOT_INSTALLED
    return _real_resolve(module_name)


_real_resolve = introspect.resolve_module_origin


def main() -> int:
    # Patch the symbol used by endpoints.py at import time.
    endpoints.resolve_module_origin = _origin_with_fastapi_uninstalled
    try:
        checker = EventLoopChecker(max_depth=20)
        result = checker.check_source(SOURCE, filepath="<isolated_uvx_env>")
    finally:
        endpoints.resolve_module_origin = _real_resolve

    expected_violation_lines = {9, 10}  # Path.exists() and Path.read_bytes()
    found_lines = {v.line for v in result.violations}

    print(f"endpoints_found = {result.endpoints_found}")
    print(f"violations      = {len(result.violations)}")
    for v in result.violations:
        print(f"  {v.filepath}:{v.line} {v.message}")

    failed = []
    if result.endpoints_found != 1:
        failed.append(
            f"expected 1 endpoint, got {result.endpoints_found} "
            "(endpoint detection must work without fastapi installed)"
        )
    if not expected_violation_lines.issubset(found_lines):
        failed.append(
            f"expected violations at lines {sorted(expected_violation_lines)}, "
            f"got {sorted(found_lines)}"
        )

    if failed:
        print("\nFAILED:")
        for f in failed:
            print(f"  - {f}")
        return 1

    print("\nOK: endpoint detection survives an isolated linter env.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
