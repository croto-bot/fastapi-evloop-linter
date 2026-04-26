# Autoresearch: fastapi-evloop-linter

## Objective
Optimize the fastapi-evloop-linter to detect ALL event loop blocking calls in FastAPI async endpoints. The linter must trace blocking calls at any depth in the call tree, handle aliased imports, and correctly resolve method calls. The adversarial benchmark generates tricky patterns to challenge the linter.

## Metrics
- **Primary**: missed (count, lower is better) — number of expected violations not detected
- **Secondary**: detection_rate (%), cases_missed, false_positives, elapsed_ms

## How to Run
`uv run python benchmark.py` — outputs METRIC lines.

## Files in Scope
- `src/fastapi_evloop_linter/blockers.py` — Registry of known blocking patterns
- `src/fastapi_evloop_linter/callgraph.py` — AST visitor, call graph builder, import resolver
- `src/fastapi_evloop_linter/checker.py` — Main checker that traces call trees and reports violations
- `src/fastapi_evloop_linter/cli.py` — CLI entry point
- `src/fastapi_evloop_linter/__init__.py` — Package init
- `tests/adversarial/generator.py` — Adversarial test case generator
- `benchmark.py` — Benchmark script

## Off Limits
- `pyproject.toml` — don't change project config
- `benchmark.py` — don't change the scoring logic

## Constraints
- Pure Python, no external dependencies (only stdlib ast module)
- Must be fast (< 100ms for the full benchmark suite)
- False positives must stay at 0

## What's Been Tried
- **Baseline**: 3 missed violations (91.7% detection rate)
  - `aliased_from_import`: `from subprocess import run as execute` — aliased from-imports not resolved
  - `chained_session`: `requests.Session().get()` — chained calls: only Session detected, not .get()
  - `callback_blocking`: callback pattern — blocking call inside function passed as argument (depth 8, very hard)
