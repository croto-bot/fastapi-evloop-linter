"""Test: asyncio.run() / loop.run_until_complete() blind spot.

The linter marks the entire asyncio module as safe (SAFE_MODULES), but
asyncio.run() and loop.run_until_complete() are dangerous when called
from within an already-running async context:
  - asyncio.run() raises RuntimeError
  - loop.run_until_complete() deadlocks or raises RuntimeError

This is one of the most common mistakes in async Python.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker

checker = EventLoopChecker(max_depth=20)

CASES = {
    "asyncio.run() direct": '''
import asyncio
from fastapi import FastAPI
app = FastAPI()

async def fetch_user(user_id: int):
    await asyncio.sleep(0.1)
    return {"id": user_id, "name": "Test"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    result = asyncio.run(fetch_user(user_id))
    return result
''',
    "loop.run_until_complete()": '''
import asyncio
from fastapi import FastAPI
app = FastAPI()

async def query_db(sql: str):
    await asyncio.sleep(0)
    return {"rows": []}

@app.get("/data")
async def get_data():
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(query_db("SELECT * FROM users"))
    return result
''',
    "from asyncio import run": '''
from asyncio import run
from fastapi import FastAPI
app = FastAPI()

async def helper():
    return 42

@app.get("/test")
async def test():
    result = run(helper())
    return {"result": result}
''',
    "get_running_loop + run_until_complete": '''
import asyncio
from fastapi import FastAPI
app = FastAPI()

async def process():
    await asyncio.sleep(0)
    return "done"

@app.get("/test")
async def test():
    loop = asyncio.get_running_loop()
    result = loop.run_until_complete(process())
    return {"result": result}
''',
    "asyncio.run() in sync helper": '''
import asyncio
from fastapi import FastAPI
app = FastAPI()

async def inner():
    return 42

def sync_caller():
    return asyncio.run(inner())

@app.get("/test")
async def test():
    result = sync_caller()
    return {"result": result}
''',
}

# Negative cases that should NOT be flagged
NEGATIVE_CASES = {
    "await asyncio.sleep()": '''
import asyncio
from fastapi import FastAPI
app = FastAPI()

@app.get("/ok")
async def ok_endpoint():
    await asyncio.sleep(1)
    return {"ok": True}
''',
    "asyncio.create_task + gather": '''
import asyncio
from fastapi import FastAPI
app = FastAPI()

@app.get("/ok")
async def ok_endpoint():
    task = asyncio.create_task(asyncio.sleep(1))
    await task
    results = await asyncio.gather(asyncio.sleep(0.1))
    return {"ok": True}
''',
    "asyncio.Lock + async with": '''
import asyncio
from fastapi import FastAPI
app = FastAPI()

@app.get("/ok")
async def ok_endpoint():
    lock = asyncio.Lock()
    async with lock:
        pass
    return {"ok": True}
''',
}

print("=" * 70)
print("BLIND SPOT: asyncio.run() / run_until_complete() in async endpoints")
print("=" * 70)

missed = 0
for name, source in CASES.items():
    r = checker.check_source(source, f"<{name}>")
    status = "MISSED ✗" if len(r.violations) == 0 else f"caught ({len(r.violations)})"
    if len(r.violations) == 0:
        missed += 1
    print(f"  {name}: {status}")

print()
false_positives = 0
for name, source in NEGATIVE_CASES.items():
    r = checker.check_source(source, f"<{name}>")
    status = "OK ✓" if len(r.violations) == 0 else f"FALSE POSITIVE ({len(r.violations)})"
    if len(r.violations) > 0:
        false_positives += 1
    print(f"  [negative] {name}: {status}")

print(f"\nResult: {missed}/{len(CASES)} positive cases MISSED, {false_positives}/{len(NEGATIVE_CASES)} false positives")
