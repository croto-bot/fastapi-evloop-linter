# Autoresearch: fastapi-evloop-linter

## Objective
Optimize the fastapi-evloop-linter to detect ALL event loop blocking calls in FastAPI async endpoints. The linter must trace blocking calls at any depth in the call tree, handle aliased imports, and correctly resolve method calls. The adversarial benchmark generates tricky patterns to challenge the linter.

## Metrics
- **Primary**: missed (count, lower is better) — number of expected violations not detected
- **Secondary**: detection_rate (%), cases_missed, false_positives, elapsed_ms

## How to Run
`uv run python benchmark.py` or `./autoresearch.sh` — outputs METRIC lines.

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
  - `aliased_from_import`: from-import with alias not resolved
  - `chained_session`: requests.Session().get() not detected
  - `callback_blocking`: callback pattern not detected
- **Fix 1**: Return original import name (not alias) in call resolution → fixed aliased_from_import
- **Fix 2**: Match module-level patterns even when object_type is set → fixed chained_session
- **Fix 3**: Positional arg tracking with None placeholders for callback flow → fixed callback_blocking
- **Result**: 100% detection rate (0 missed, 0 false positives) across 44 adversarial test cases
- **Hardened**: Added 11 more adversarial cases (router decorators, depth 7 chains, map callbacks, builtin open, aliased modules) — all pass
