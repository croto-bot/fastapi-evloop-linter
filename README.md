# fastapi-evloop-linter

Detect event loop blocking calls in FastAPI async endpoints.

A static analysis tool that traces call graphs from async FastAPI endpoints to find blocking synchronous calls (like `time.sleep()`, `requests.get()`, `subprocess.run()`, etc.) at any depth in the execution tree.

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

All detected through deep call chain analysis:
- Traces through helper functions
- Resolves aliased imports
- Handles callback patterns
- Tracks variable types

## Error Code

**EVL001** — Event loop blocking call detected

## Output Format

Compatible with ruff's output format:

```
path/to/file.py:42:8: EVL001 error: time.sleep() blocks the event loop (depth=3) via endpoint -> helper -> wait
```
