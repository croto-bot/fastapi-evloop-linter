# fastapi-evloop-linter

Detect event loop blocking calls in async Python entrypoints.

A static analysis tool that traces call graphs from async functions, including FastAPI endpoints and background jobs, to find blocking synchronous calls (like `time.sleep()`, `requests.get()`, `subprocess.run()`, etc.) at any depth in the execution tree.

## Installation

```bash
uv add --dev fastapi-evloop-linter
```

## Usage

### Command Line

```bash
# Check files or directories
fastapi-evloop-linter src/

# Check with JSON output
fastapi-evloop-linter --format json src/

# Check specific files
fastapi-evloop-linter app/main.py app/routes.py
```

### Integration with ruff

Add to your `ruff.toml` or `pyproject.toml`:

```toml
[tool.ruff]
# Run fastapi-evloop-linter alongside ruff
# In your CI/CD pipeline or pre-commit:
# ruff check . && fastapi-evloop-linter src/
```

Or use as a pre-commit hook:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: fastapi-evloop-linter
        name: fastapi-evloop-linter
        entry: fastapi-evloop-linter
        language: system
        types: [python]
```

### Integration with uv

```bash
# Run with uvx (no install needed)
uvx fastapi-evloop-linter src/

# Or add to your project
uv add --dev fastapi-evloop-linter
uv run fastapi-evloop-linter src/
```

## What It Detects

- **Blocking I/O**: `requests.get()`, `urllib.request.urlopen()`, `http.client`
- **CPU-bound operations**: `time.sleep()`, `bcrypt.hashpw()`
- **Subprocess calls**: `subprocess.run()`, `os.system()`
- **File I/O**: `open()`, `pathlib.Path.read_text()`
- **Sync database drivers**: `psycopg2.connect()`, `pymysql.connect()`
- **Sync Redis**: `redis.Redis()`

All detected through deep call chain analysis from async entrypoints:
- Traces through helper functions
- Resolves aliased imports
- Handles callback patterns
- Tracks variable types
- Propagates local return values, including `await helper()` results
- Resolves bound methods, context-manager `as` variables, properties, descriptors, dunders, and superclass calls

## Semantic Analysis

The linter uses a small semantic layer rather than matching only individual examples.
It assigns stable local symbols to functions and methods, tracks best-effort value
types, emits semantic operations, then classifies those operations with the EVL001
policy.

For example, these shapes all reduce to "call a blocking operation reachable from an
async endpoint":

```python
conn = await get_conn()
conn.execute("SELECT 1")

fn = worker.process
fn()

with sqlite3.connect("db.sqlite") as conn:
    conn.execute("SELECT 1")

value = proxy.data  # local property, descriptor, or __getattr__
```

Offloaded callbacks are treated differently from direct callbacks. A blocking function
passed to `asyncio.to_thread()` or `loop.run_in_executor()` is not reported as
event-loop blocking, while callbacks invoked by synchronous helpers are still traced.

## Non-goals

- This is not a missing-`await` rule.
- It is not a full Python type checker.
- It does not attempt cross-file whole-project call graph analysis.
- Unknown dynamic Python stays low-noise by default; only high-confidence blocking
  operations become EVL001 errors.

## Error Code

**EVL001** — Event loop blocking call detected

## Output Format

Compatible with ruff's output format:

```
path/to/file.py:42:8: EVL001 error: time.sleep() blocks the event loop (depth=3) via endpoint -> helper -> wait
```
