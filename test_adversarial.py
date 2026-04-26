#!/usr/bin/env python3
"""Test adversarial examples against the linter to find blind spots."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker

checker = EventLoopChecker(max_depth=20)

cases = [
    # Case 1: Variable aliasing of blocking functions
    (
        "variable_alias",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

@app.get("/alias")
async def alias_endpoint():
    f = time.sleep
    f(1)
    return {"ok": True}
""",
        "Variable aliasing: f = time.sleep; f(1)",
    ),
    # Case 2: functools.partial
    (
        "functools_partial",
        """\
import time
from functools import partial
from fastapi import FastAPI
app = FastAPI()

@app.get("/partial")
async def partial_endpoint():
    wait = partial(time.sleep, 1)
    wait()
    return {"ok": True}
""",
        "functools.partial wrapping time.sleep",
    ),
    # Case 3: Global dict dispatch
    (
        "dict_dispatch",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

HANDLERS = {"wait": time.sleep}

@app.get("/dispatch")
async def dispatch_endpoint():
    HANDLERS["wait"](5)
    return {"ok": True}
""",
        "Global dict dispatch: HANDLERS['wait'](5)",
    ),
    # Case 4: getattr dynamic access
    (
        "getattr_dynamic",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

@app.get("/getattr")
async def getattr_endpoint():
    getattr(time, "sleep")(1)
    return {"ok": True}
""",
        "getattr(time, 'sleep')(1)",
    ),
    # Case 5: Cross-function variable alias
    (
        "cross_func_alias",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def get_sleeper():
    return time.sleep

@app.get("/cross-alias")
async def cross_alias_endpoint():
    sleeper = get_sleeper()
    sleeper(3)
    return {"ok": True}
""",
        "Cross-function variable alias via return value",
    ),
    # Case 6: Class __call__
    (
        "class_call",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class Sleeper:
    def __call__(self, seconds):
        time.sleep(seconds)

s = Sleeper()

@app.get("/callable")
async def callable_endpoint():
    s(5)
    return {"ok": True}
""",
        "Class with __call__ that blocks",
    ),
    # Case 7: Property that blocks
    (
        "blocking_property",
        """\
import requests
from fastapi import FastAPI
app = FastAPI()

class Config:
    @property
    def data(self):
        return requests.get("https://example.com").json()

cfg = Config()

@app.get("/config")
async def config_endpoint():
    return cfg.data
""",
        "Property that makes blocking request",
    ),
    # Case 8: Decorator that blocks
    (
        "blocking_decorator",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def slow_decorator(func):
    def wrapper(*args, **kwargs):
        time.sleep(2)
        return func(*args, **kwargs)
    return wrapper

@slow_decorator
def compute():
    return 42

@app.get("/decorated")
async def decorated_endpoint():
    result = compute()
    return {"result": result}
""",
        "Function wrapped in a blocking decorator",
    ),
    # Case 9: Dict/list comprehension with indirect call
    (
        "comprehension_indirect",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def delay(x):
    time.sleep(0.1)
    return x

@app.get("/comp")
async def comp_endpoint():
    results = [delay(x) for x in range(5)]
    return {"count": len(results)}
""",
        "Blocking call via list comprehension - should be detected",
    ),
    # Case 10: Blocking call in except block
    (
        "except_block",
        """\
import requests
from fastapi import FastAPI
app = FastAPI()

@app.get("/except")
async def except_endpoint():
    try:
        pass
    except Exception:
        requests.get("https://error-handler.example.com")
    return {"ok": True}
""",
        "Blocking call in except block - should be detected",
    ),
    # Case 11: Blocking call in finally block
    (
        "finally_block",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

@app.get("/finally")
async def finally_endpoint():
    try:
        pass
    finally:
        time.sleep(1)
    return {"ok": True}
""",
        "Blocking call in finally block - should be detected",
    ),
    # Case 12: Walrus operator
    (
        "walrus_operator",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

@app.get("/walrus")
async def walrus_endpoint():
    if (wait := time.sleep):
        wait(1)
    return {"ok": True}
""",
        "Walrus operator: (wait := time.sleep); wait(1)",
    ),
    # Case 13: Module imported inside function
    (
        "local_import",
        """\
from fastapi import FastAPI
app = FastAPI()

@app.get("/local-import")
async def local_import_endpoint():
    import time
    time.sleep(1)
    return {"ok": True}
""",
        "time imported inside function body",
    ),
    # Case 14: try/except around import with alias
    (
        "try_import",
        """\
from fastapi import FastAPI
app = FastAPI()

try:
    import requests as http
except ImportError:
    http = None

@app.get("/try-import")
async def try_import_endpoint():
    if http:
        http.get("https://example.com")
    return {"ok": True}
""",
        "requests imported inside try block with alias",
    ),
    # Case 15: Rebinding a local name to a blocking function
    (
        "rebinding",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def wait():
    pass

@app.get("/rebind")
async def rebind_endpoint():
    global wait
    wait = time.sleep
    wait(1)
    return {"ok": True}
""",
        "Rebinding function name to time.sleep via global",
    ),
    # Case 16: Star import
    (
        "star_import",
        """\
from time import *
from fastapi import FastAPI
app = FastAPI()

@app.get("/star")
async def star_endpoint():
    sleep(1)
    return {"ok": True}
""",
        "Star import: from time import *",
    ),
    # Case 17: Blocking in __init__
    (
        "blocking_init",
        """\
import requests
from fastapi import FastAPI
app = FastAPI()

class ApiClient:
    def __init__(self, url):
        self.response = requests.get(url)

@app.get("/init")
async def init_endpoint():
    client = ApiClient("https://example.com")
    return {"ok": True}
""",
        "Blocking call in __init__ via constructor",
    ),
    # Case 18: Blocking call via map()
    (
        "map_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def slow_transform(x):
    time.sleep(0.1)
    return x * 2

@app.get("/map")
async def map_endpoint():
    results = list(map(slow_transform, [1, 2, 3]))
    return {"results": results}
""",
        "Blocking call via map() - should be detected via call graph",
    ),
    # Case 19: Blocking in a lambda (assigned to variable)
    (
        "lambda_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

@app.get("/lambda")
async def lambda_endpoint():
    wait = lambda s: time.sleep(s)
    wait(5)
    return {"ok": True}
""",
        "Lambda wrapping time.sleep",
    ),
    # Case 20: Blocking in context manager __enter__
    (
        "context_manager",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class Timer:
    def __enter__(self):
        time.sleep(1)
        return self
    def __exit__(self, *args):
        pass

@app.get("/ctx")
async def ctx_endpoint():
    with Timer():
        pass
    return {"ok": True}
""",
        "Blocking call in context manager __enter__",
    ),
    # Case 21: Multiple files - call to unknown function
    (
        "external_module",
        """\
from fastapi import FastAPI
from my_helpers import blocking_fetch
app = FastAPI()

@app.get("/external")
async def external_endpoint():
    data = blocking_fetch("https://example.com")
    return {"data": data}
""",
        "Call to external module function - can't resolve",
    ),
    # Case 22: Indirect via dataclass __post_init__
    (
        "dataclass_postinit",
        """\
import requests
from dataclasses import dataclass
from fastapi import FastAPI
app = FastAPI()

@dataclass
class UserInfo:
    url: str

    def __post_init__(self):
        self.data = requests.get(self.url).json()

@app.get("/dataclass")
async def dataclass_endpoint():
    user = UserInfo("https://api.example.com/user")
    return user.data
""",
        "Blocking in dataclass __post_init__",
    ),
    # Case 23: Blocking via operator overloading
    (
        "operator_overload",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class Delay:
    def __mul__(self, seconds):
        time.sleep(seconds)
        return self

d = Delay()

@app.get("/operator")
async def operator_endpoint():
    d * 5
    return {"ok": True}
""",
        "Blocking via operator overloading (__mul__)",
    ),
    # Case 24: eval/exec
    (
        "eval_exec",
        """\
from fastapi import FastAPI
app = FastAPI()

@app.get("/eval")
async def eval_endpoint():
    import time
    eval("time.sleep(1)")
    return {"ok": True}
""",
        "Blocking via eval()",
    ),
    # Case 25: Nested class method
    (
        "nested_class_method",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class Service:
    class Inner:
        def process(self):
            time.sleep(1)

svc = Service.Inner()

@app.get("/nested-class")
async def nested_class_endpoint():
    svc.process()
    return {"ok": True}
""",
        "Nested class method with blocking call",
    ),
    # Case 26: Blocking in default argument evaluation  
    (
        "default_arg_eval",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def get_default():
    time.sleep(1)
    return "default"

def helper(val=get_default()):
    return val

@app.get("/default-arg")
async def default_arg_endpoint():
    result = helper()
    return {"result": result}
""",
        "Blocking in default argument (module-level eval)",
    ),
    # Case 27: os.system via os.path (confusing name)
    (
        "misleading_name",
        """\
from fastapi import FastAPI
app = FastAPI()

def data():
    import os
    os.system("rm -rf /")

@app.get("/mislead")
async def mislead_endpoint():
    data()
    return {"ok": True}
""",
        "Blocking call in function named 'data' - should be detected",
    ),
    # Case 28: Triple indirection through globals
    (
        "globals_lookup",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

FN = time.sleep

@app.get("/globals")
async def globals_endpoint():
    fn = globals()["FN"]
    fn(3)
    return {"ok": True}
""",
        "Blocking via globals() dict lookup",
    ),
]

print("=" * 80)
print("ADVERSARIAL TEST RESULTS - Finding linter blind spots")
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
        missed_cases.append(name)
    
    print(f"{status} [{name}] {description}")
    print()

print("=" * 80)
print(f"SUMMARY: {detected} detected, {missed} missed out of {len(cases)} cases")
print(f"Blind spot rate: {missed}/{len(cases)} = {missed/len(cases)*100:.1f}%")
print()
print("MISSED CASES (blind spots):")
for name in missed_cases:
    print(f"  - {name}")
