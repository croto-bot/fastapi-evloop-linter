#!/usr/bin/env python3
"""Regression: common false-positive sources from real fastapi apps.

Each case is a snippet the user's linter flagged on TEC-7 / their bimbuddy
batchjobs repo. The linter should report NO violations for any of these.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker

CASES = [
    (
        "asyncio_create_task",
        """\
import asyncio
from fastapi import FastAPI
app = FastAPI()

async def _bg(): pass

@app.post("/x")
async def ep():
    asyncio.create_task(_bg())
    return {}
""",
    ),
    (
        "datetime_now_and_utc",
        """\
from datetime import datetime, UTC
from fastapi import FastAPI
app = FastAPI()

@app.post("/x")
async def ep():
    return {"now": datetime.now(UTC)}
""",
    ),
    (
        "logger_calls",
        """\
import logging
from fastapi import FastAPI
app = FastAPI()
log = logging.getLogger(__name__)

@app.post("/x")
async def ep():
    log.info("hi")
    log.error("bad")
    return {}
""",
    ),
    (
        "exception_class_in_raise",
        """\
from fastapi import FastAPI, HTTPException
app = FastAPI()

@app.post("/x")
async def ep():
    raise HTTPException(status_code=404)
""",
    ),
    (
        "project_local_helpers",
        """\
from fastapi import FastAPI
from src.utils.logging import get_logger
from src.services.job_tracker import init_job
from app.api.health.schemas import HealthCheckResponse

app = FastAPI()
log = get_logger(__name__)

@app.post("/x")
async def ep():
    log.info("starting")
    init_job("k")
    return HealthCheckResponse(status="ok")
""",
    ),
    (
        "pathlib_pure_methods",
        """\
from pathlib import Path
from fastapi import FastAPI
app = FastAPI()

@app.post("/x")
async def ep(s: str):
    p = Path(s)
    return {
        "abs": p.is_absolute(),
        "name": p.name,
        "parent": str(p.parent),
        "with_suffix": str(p.with_suffix(".txt")),
    }
""",
    ),
]


# Cases that MUST still be flagged — guard against over-suppression.
POSITIVE_CASES = [
    (
        "pathlib_real_io_still_flagged",
        """\
from pathlib import Path
from fastapi import FastAPI
app = FastAPI()

@app.post("/x")
async def ep(s: str):
    p = Path(s)
    if p.exists():
        return p.read_bytes()
""",
        2,
    ),
    (
        "requests_still_flagged_when_uninstalled",
        """\
import requests
from fastapi import FastAPI
app = FastAPI()

@app.post("/x")
async def ep():
    return requests.get("https://example.com").json()
""",
        1,
    ),
]


def main() -> int:
    checker = EventLoopChecker(max_depth=20)
    failures: list[str] = []

    for name, src in CASES:
        result = checker.check_source(src, filepath=f"<{name}>")
        if result.violations:
            failures.append(
                f"[{name}] expected 0 violations, got {len(result.violations)}: "
                + "; ".join(v.message for v in result.violations)
            )

    for name, src, expected in POSITIVE_CASES:
        result = checker.check_source(src, filepath=f"<{name}>")
        if len(result.violations) < expected:
            failures.append(
                f"[{name}] expected at least {expected} violation(s), got "
                f"{len(result.violations)} — over-suppression regression"
            )

    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    total = len(CASES) + len(POSITIVE_CASES)
    print(f"OK: {total} cases pass ({len(CASES)} silent, {len(POSITIVE_CASES)} flagged).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
