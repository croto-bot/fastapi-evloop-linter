#!/usr/bin/env python3
"""Hunt for NEW blind spots not covered by existing issues TEC-5, TEC-8, TEC-9."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker

checker = EventLoopChecker(max_depth=20)

cases = [
    # 1. WITH statement: as-variable type not tracked → methods missed
    (
        "with_as_variable",
        """\
import sqlite3
from fastapi import FastAPI
app = FastAPI()

@app.get("/db")
async def db_endpoint():
    with sqlite3.connect("data.db") as conn:
        rows = conn.execute("SELECT * FROM users").fetchall()
    return {"rows": rows}
""",
        "with sqlite3.connect() as conn: conn.execute() - methods on as-variable missed",
    ),
    # 2. WITH statement using module.Class() - _handle_context_manager only handles ast.Name
    (
        "with_module_class",
        """\
import zipfile
from fastapi import FastAPI
app = FastAPI()

@app.get("/zip")
async def read_zip():
    with zipfile.ZipFile("data.zip") as zf:
        files = zf.namelist()
        data = zf.read("entry.txt")
    return {"files": files, "data": data}
""",
        "with zipfile.ZipFile() as zf: zf.read() - constructor SAFE, methods UNKNOWN",
    ),
    # 3. Method reference passed as callback (ast.Attribute arg to HOF)
    (
        "method_as_callback",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class Processor:
    def transform(self, data):
        time.sleep(0.1)
        return data.upper()

def apply_all(items, func):
    return [func(item) for item in items]

@app.get("/process")
async def process_items():
    proc = Processor()
    results = apply_all(["a", "b"], proc.transform)
    return {"results": results}
""",
        "proc.transform passed as callback to apply_all - method ref as ast.Attribute arg",
    ),
    # 4. Await expression type propagation
    (
        "await_type_propagation",
        """\
import sqlite3
from fastapi import FastAPI
app = FastAPI()

async def get_conn():
    return sqlite3.connect("data.db")

@app.get("/db2")
async def db2_endpoint():
    conn = await get_conn()
    rows = conn.execute("SELECT 1").fetchall()
    return {"rows": rows}
""",
        "conn = await get_conn() → conn.execute() missed because await doesn't propagate type",
    ),
    # 5. AugAssign operator (+=, -=, *=) triggers __iadd__, __isub__, __imul__
    (
        "augmented_assign",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class Accumulator:
    def __init__(self):
        self.total = 0
    def __iadd__(self, other):
        time.sleep(0.5)
        self.total += other
        return self

acc = Accumulator()

@app.get("/acc")
async def acc_endpoint():
    acc += 5
    return {"total": acc.total}
""",
        "acc += 5 triggers __iadd__ with time.sleep - AugAssign not handled",
    ),
    # 6. super() call resolution
    (
        "super_call",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class BaseService:
    def fetch(self):
        time.sleep(1)
        return "data"

class MyService(BaseService):
    def fetch(self):
        return super().fetch()

svc = MyService()

@app.get("/data")
async def get_data():
    return {"data": svc.fetch()}
""",
        "super().fetch() can't resolve to BaseService.fetch which blocks",
    ),
    # 7. __getattr__ dynamic attribute
    (
        "getattr_dunder",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class LazyProxy:
    def __getattr__(self, name):
        time.sleep(1)
        return f"value_{name}"

proxy = LazyProxy()

@app.get("/proxy")
async def proxy_endpoint():
    return {"data": proxy.data}
""",
        "proxy.data triggers __getattr__ with time.sleep",
    ),
    # 8. Chained attribute from constructor: obj = Module.Class(); with obj as x:
    (
        "with_variable_context_mgr",
        """\
import zipfile
from fastapi import FastAPI
app = FastAPI()

@app.get("/zip2")
async def read_zip2():
    zf = zipfile.ZipFile("data.zip")
    with zf as archive:
        data = archive.read("entry.txt")
    return {"data": data}
""",
        "zf = zipfile.ZipFile(); with zf as archive: archive.read() missed",
    ),
    # 9. asyncio.to_thread wrapping (should be safe) but using loop.run_in_executor
    (
        "run_in_executor",
        """\
import asyncio
import time
from fastapi import FastAPI
app = FastAPI()

def blocking_task():
    time.sleep(2)
    return 42

@app.get("/executor")
async def executor_endpoint():
    loop = asyncio.get_event_loop()
    result = loop.run_in_executor(None, blocking_task)
    return {"result": result}
""",
        "loop.run_in_executor - should be safe but interesting to check",
    ),
    # 10. walrus in list comprehension
    (
        "walrus_comp",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def slow_calc(x):
    time.sleep(0.1)
    return x * 2

@app.get("/walrus-comp")
async def walrus_comp_endpoint():
    results = [(y := slow_calc(x), y + 1) for x in range(3)]
    return {"results": results}
""",
        "Walrus in list comprehension calling blocking function",
    ),
    # 11. With statement with open() - f.read() missed
    (
        "with_open_read",
        """\
from fastapi import FastAPI
app = FastAPI()

@app.get("/read")
async def read_file():
    with open("data.txt") as f:
        content = f.read()
    return {"content": content}
""",
        "with open() as f: f.read() - open() caught but f.read() missed",
    ),
    # 12. Multiple return values unpacking
    (
        "tuple_unpack",
        """\
import subprocess
from fastapi import FastAPI
app = FastAPI()

def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr

@app.get("/cmd")
async def cmd_endpoint():
    stdout, stderr = run_cmd(["ls"])
    return {"stdout": stdout}
""",
        "subprocess.run inside helper with tuple unpacking",
    ),
    # 13. Instance method stored in variable then called
    (
        "bound_method_var",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class Worker:
    def process(self):
        time.sleep(1)
        return "done"

w = Worker()

@app.get("/bound")
async def bound_endpoint():
    fn = w.process
    result = fn()
    return {"result": result}
""",
        "fn = w.process; fn() - bound method stored in variable",
    ),
    # 14. hasattr / getattr dynamic patterns
    (
        "hasattr_pattern",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class Obj:
    def fetch(self):
        time.sleep(1)
        return "data"

obj = Obj()

@app.get("/hasattr")
async def hasattr_endpoint():
    if hasattr(obj, "fetch"):
        return {"data": obj.fetch()}
    return {"error": "no fetch"}
""",
        "hasattr + obj.fetch() - the fetch() call should be caught",
    ),
    # 15. Descriptor __get__ triggered by attribute access
    (
        "descriptor_get",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class BlockingDescriptor:
    def __get__(self, obj, objtype=None):
        time.sleep(1)
        return "value"

class Service:
    data = BlockingDescriptor()

svc = Service()

@app.get("/desc")
async def desc_endpoint():
    return {"data": svc.data}
""",
        "svc.data triggers BlockingDescriptor.__get__ with time.sleep",
    ),
]

print("=" * 80)
print("BLIND SPOT HUNT - NEW patterns not covered by TEC-5, TEC-8, TEC-9")
print("=" * 80)

detected = 0
missed = 0
missed_cases = []

for name, source, description in cases:
    result = checker.check_source(source, filepath=f"<{name}>")
    found = len(result.violations)
    
    if found > 0:
        status = "✅ DETECTED"
        detected += 1
        for v in result.violations:
            print(f"  Found: line {v.line}: {v.message} (depth={v.depth})")
    else:
        status = "❌ MISSED"
        missed += 1
        missed_cases.append((name, description))
    
    print(f"{status} [{name}] {description}")
    print()

print("=" * 80)
print(f"SUMMARY: {detected} detected, {missed} missed out of {len(cases)} cases")
if missed_cases:
    print(f"\nMISSED CASES ({missed}):")
    for name, desc in missed_cases:
        print(f"  - {name}: {desc}")
