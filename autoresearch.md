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

## New Session: Blind Spot Hardening (13 new adversarial cases)

Added 13 adversarial cases that exploit blind spots the linter currently misses:

### Missed categories (from testing):
1. **variable_alias** (4 cases): `f = time.sleep; f(1)`, walrus, cross-func alias, functools.partial
2. **class_dunder** (5 cases): `__call__`, `__enter__`, `__init__`, `__post_init__`, operator overloading
3. **blocking_decorator** (1 case): decorator wrapper contains time.sleep
4. **dict_dispatch** (1 case): `HANDLERS["wait"](5)` with `time.sleep` value
5. **higher_order** (1 case): `map(slow_func, items)`
6. **blocking_property** (1 case): `@property` that does blocking I/O

### Analysis of what needs to change:

**Variable aliasing**: Track variable assignments where RHS is a known blocking function/module.attr.
  - `f = time.sleep` → track `f` as alias for `time.sleep`
  - `partial(time.sleep, 1)` → detect `partial` of blocking function
  - `get_sleeper()` → too dynamic, skip for now (function returns blocking ref)
  - Walrus: `(wait := time.sleep)` → same as variable aliasing

**Class dunder methods**: When a constructor/dunder is called, trace into the dunder method.
  - `s = Sleeper(); s(5)` → `Sleeper.__call__` contains blocking call
  - `with Timer():` → `Timer.__enter__` contains blocking call
  - `ApiClient(url)` → `ApiClient.__init__` contains blocking call
  - `UserInfo(url)` → `UserInfo.__post_init__` contains blocking call
  - `d * 5` → `Delay.__mul__` contains blocking call

**Blocking decorators**: The decorator pattern replaces `compute` with `wrapper`.
  - When `compute()` is called, actually calls `wrapper()` which contains `time.sleep`
  - Need to detect that `compute` is reassigned by decorator application

**Dict dispatch**: `HANDLERS["wait"](5)` → subscript on a module-level dict
  - Track dict literals at module level that map to blocking functions

**Higher-order**: `map(slow_transform, items)` → `map` calls `slow_transform` for each item
  - Treat `map`/`filter`/`sorted` as higher-order callers (like callback pattern)

**Property**: `cfg.data` → property access calls getter which contains blocking call
  - Track property definitions and flag them when accessed on objects

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
- **New baseline**: 13 missed out of 63 violations (79.4% detection rate) with 13 new blind-spot cases
- **Fix blind spots**: Added variable alias tracking (var_aliases), cross-function return tracking (returns_blocking), class dunder resolution (__call__, __enter__, __init__, __post_init__, operator overloading), decorator replacement detection, dict dispatch, map() higher-order, property access tracking, and module-level var type tracking
- **Result**: 100% detection rate (0 missed, 0 false positives) across 57 adversarial test cases (63 violations)
