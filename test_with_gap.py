#!/usr/bin/env python3
"""Verify the with-statement type inference blind spot in detail."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker

checker = EventLoopChecker(max_depth=20)

# Case 1: zipfile.ZipFile - ZERO violations (constructor SAFE, methods UNKNOWN)
print("=== Case 1: zipfile.ZipFile in with statement ===")
result = checker.check_source('''
import zipfile
from fastapi import FastAPI
app = FastAPI()

@app.get("/zip")
async def read_zip():
    with zipfile.ZipFile("data.zip") as zf:
        files = zf.namelist()
        data = zf.read("entry.txt")
    return {"files": files, "data": data}
''')
print(f"Violations: {len(result.violations)}")
for v in result.violations:
    print(f"  {v}")
print()

# Case 2: Direct assignment DOES track type → should catch zf.namelist()
print("=== Case 2: Direct assignment (no with) - should catch methods ===")
result = checker.check_source('''
import zipfile
from fastapi import FastAPI
app = FastAPI()

@app.get("/zip")
async def read_zip():
    zf = zipfile.ZipFile("data.zip")
    files = zf.namelist()
    data = zf.read("entry.txt")
    return {"files": files, "data": data}
''')
print(f"Violations: {len(result.violations)}")
for v in result.violations:
    print(f"  {v}")
print()

# Case 3: httpx.Client sync - ZERO violations in real usage
print("=== Case 3: httpx sync Client in with statement ===")
result = checker.check_source('''
import httpx
from fastapi import FastAPI
app = FastAPI()

@app.get("/fetch")
async def fetch_data():
    with httpx.Client() as client:
        response = client.get("https://example.com")
    return {"data": response.json()}
''')
print(f"Violations: {len(result.violations)}")
for v in result.violations:
    print(f"  {v}")
print()

# Case 4: custom class context manager - _handle_context_manager misses Attribute func
print("=== Case 4: with variable (zf = ZipFile(); with zf as archive:) ===")
result = checker.check_source('''
import zipfile
from fastapi import FastAPI
app = FastAPI()

@app.get("/zip2")
async def read_zip2():
    zf = zipfile.ZipFile("data.zip")
    with zf as archive:
        data = archive.read("entry.txt")
    return {"data": data}
''')
print(f"Violations: {len(result.violations)}")
for v in result.violations:
    print(f"  {v}")
print()

# Case 5: sqlite3.connect is caught, but conn.execute() is NOT
print("=== Case 5: sqlite3.connect caught, but conn.execute() missed ===")
result = checker.check_source('''
import sqlite3
from fastapi import FastAPI
app = FastAPI()

@app.get("/db")
async def db_endpoint():
    with sqlite3.connect("data.db") as conn:
        rows = conn.execute("SELECT * FROM users").fetchall()
    return {"rows": rows}
''')
print(f"Violations: {len(result.violations)}")
for v in result.violations:
    print(f"  {v}")
print()

# Case 6: with + open() - f.read() missed (but open() caught)
print("=== Case 6: open() caught, f.read() missed ===")
result = checker.check_source('''
from fastapi import FastAPI
app = FastAPI()

@app.get("/read")
async def read_file():
    with open("data.txt") as f:
        content = f.read()
    return {"content": content}
''')
print(f"Violations: {len(result.violations)}")
for v in result.violations:
    print(f"  {v}")
