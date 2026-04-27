#!/usr/bin/env python3
"""Test 10+ new adversarial patterns against the linter to find remaining blind spots."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker

checker = EventLoopChecker(max_depth=20)

cases = [
    # 1. filter() as higher-order caller (like map but filter)
    (
        "filter_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def is_ready(item):
    time.sleep(0.1)
    return item > 0

@app.get("/filter")
async def filter_endpoint():
    results = list(filter(is_ready, [1, -2, 3]))
    return {"count": len(results)}
""",
        "filter() with blocking predicate",
    ),
    # 2. sorted() with blocking key function
    (
        "sorted_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def slow_key(item):
    time.sleep(0.1)
    return item["score"]

@app.get("/sorted")
async def sorted_endpoint():
    results = sorted([{"score": 1}, {"score": 2}], key=slow_key)
    return {"results": results}
""",
        "sorted() with blocking key function",
    ),
    # 3. Blocking call in __aenter__ (async context manager - but sync blocking inside)
    (
        "async_context_manager",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class AsyncTimer:
    async def __aenter__(self):
        time.sleep(1)
        return self
    async def __aexit__(self, *args):
        pass

@app.get("/actx")
async def actx_endpoint():
    async with AsyncTimer():
        pass
    return {"ok": True}
""",
        "Blocking call in __aenter__ of async context manager",
    ),
    # 4. Blocking in list.sort() with key
    (
        "list_sort_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def slow_compare(x):
    time.sleep(0.01)
    return x

@app.get("/sort")
async def sort_endpoint():
    items = [3, 1, 2]
    items.sort(key=slow_compare)
    return {"items": items}
""",
        "list.sort() with blocking key function",
    ),
    # 5. Blocking call in __repr__ or __str__ (implicit call during string formatting)
    (
        "blocking_repr",
        """\
import requests
from fastapi import FastAPI
app = FastAPI()

class User:
    def __init__(self, uid):
        self.uid = uid
    
    def __repr__(self):
        data = requests.get(f"https://api.example.com/users/{self.uid}").json()
        return data["name"]

u = User(1)

@app.get("/repr")
async def repr_endpoint():
    name = f"{u}"
    return {"name": name}
""",
        "Blocking call in __repr__ triggered by f-string formatting",
    ),
    # 6. Blocking in ternary expression
    (
        "ternary_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

@app.get("/ternary")
async def ternary_endpoint(flag: bool):
    time.sleep(1) if flag else None
    return {"ok": True}
""",
        "Blocking call in ternary expression",
    ),
    # 7. Blocking call in assert
    (
        "assert_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

@app.get("/assert")
async def assert_endpoint():
    time.sleep(1)
    assert True
    return {"ok": True}
""",
        "Blocking call before assert",
    ),
    # 8. Async function that calls sync (not an endpoint but called from endpoint)
    (
        "async_helper_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

async def async_helper():
    time.sleep(1)

@app.get("/async-helper")
async def async_helper_endpoint():
    await async_helper()
    return {"ok": True}
""",
        "Blocking call inside an async helper (not endpoint)",
    ),
    # 9. Blocking call in generator function
    (
        "generator_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def slow_gen():
    for i in range(3):
        time.sleep(0.5)
        yield i

@app.get("/gen")
async def gen_endpoint():
    results = list(slow_gen())
    return {"results": results}
""",
        "Blocking call inside generator function",
    ),
    # 10. Nested class with blocking in outer class method
    (
        "nested_class_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class Outer:
    class Inner:
        pass
    
    def process(self):
        time.sleep(1)

o = Outer()

@app.get("/nested")
async def nested_endpoint():
    o.process()
    return {"ok": True}
""",
        "Blocking call in outer class method (non-dunder)",
    ),
    # 11. Blocking via threading.Lock (acquire is blocking)
    (
        "lock_acquire",
        """\
import threading
from fastapi import FastAPI
app = FastAPI()

lock = threading.Lock()

@app.get("/lock")
async def lock_endpoint():
    lock.acquire()
    lock.release()
    return {"ok": True}
""",
        "threading.Lock.acquire() is blocking",
    ),
    # 12. Blocking via concurrent.futures (submit and result)
    (
        "concurrent_result",
        """\
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
app = FastAPI()

pool = ThreadPoolExecutor()

@app.get("/concurrent")
async def concurrent_endpoint():
    future = pool.submit(lambda: 42)
    result = future.result()
    return {"result": result}
""",
        "future.result() blocks the event loop",
    ),
    # 13. Blocking in __del__
    (
        "blocking_del",
        """\
import requests
from fastapi import FastAPI
app = FastAPI()

class Cleanup:
    def __del__(self):
        requests.get("https://cleanup.example.com")

@app.get("/del")
async def del_endpoint():
    c = Cleanup()
    del c
    return {"ok": True}
""",
        "Blocking call in __del__ (destructor)",
    ),
    # 14. Multiple imports on one line
    (
        "multi_import_line",
        """\
import time, requests
from fastapi import FastAPI
app = FastAPI()

@app.get("/multi-imp")
async def multi_imp_endpoint():
    time.sleep(1)
    requests.get("https://example.com")
    return {"ok": True}
""",
        "Multiple modules on one import line",
    ),
    # 15. Blocking in a lambda passed to sorted
    (
        "lambda_sorted",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

@app.get("/lambda-sort")
async def lambda_sort_endpoint():
    results = sorted([1, 2, 3], key=lambda x: time.sleep(0.1) or x)
    return {"results": results}
""",
        "Lambda with time.sleep passed to sorted()",
    ),
    # 16. Blocking via queue.get (stdlib queue)
    (
        "queue_get",
        """\
import queue
from fastapi import FastAPI
app = FastAPI()

q = queue.Queue()

@app.get("/queue")
async def queue_endpoint():
    item = q.get()
    return {"item": item}
""",
        "queue.Queue.get() blocks the event loop",
    ),
    # 17. Blocking call via any() / all() with blocking predicate
    (
        "any_all_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def slow_check(x):
    time.sleep(0.1)
    return x > 0

@app.get("/any")
async def any_endpoint():
    result = any(slow_check(x) for x in [1, 2, 3])
    return {"result": result}
""",
        "any() with blocking predicate via generator expression",
    ),
    # 18. Deep nesting: endpoint calls async func which calls sync func which blocks
    (
        "deep_async_sync_mix",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def sync_blocker():
    time.sleep(1)

def sync_mid():
    sync_blocker()

async def async_mid():
    sync_mid()

@app.get("/deep-mix")
async def deep_mix_endpoint():
    await async_mid()
    return {"ok": True}
""",
        "Deep chain: endpoint -> async helper -> sync helper -> blocking call",
    ),
    # 19. Blocking in a set comprehension
    (
        "set_comp_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def slow_hash(x):
    time.sleep(0.1)
    return hash(x)

@app.get("/set-comp")
async def set_comp_endpoint():
    results = {slow_hash(x) for x in range(3)}
    return {"count": len(results)}
""",
        "Blocking call inside set comprehension",
    ),
    # 20. Star-args unpacking of blocking function references
    (
        "star_args_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def call_all(*funcs):
    for f in funcs:
        f()

@app.get("/star-args")
async def star_args_endpoint():
    call_all(lambda: time.sleep(1))
    return {"ok": True}
""",
        "Blocking lambda passed via *args to helper",
    ),
    # 21. Type annotation as string with blocking pattern
    (
        "string_annotation",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def helper() -> "dict":
    time.sleep(1)
    return {}

@app.get("/str-ann")
async def str_ann_endpoint():
    data = helper()
    return {"data": data}
""",
        "Blocking call in function with string type annotation",
    ),
    # 22. walrus in while condition
    (
        "walrus_while",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def get_val():
    time.sleep(0.1)
    return None

@app.get("/walrus-while")
async def walrus_while_endpoint():
    while (val := get_val()) is not None:
        pass
    return {"ok": True}
""",
        "Walrus in while condition calling blocking function",
    ),
    # 23. Blocking call in dict.get() default
    (
        "dict_get_default",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def slow_default():
    time.sleep(1)
    return "default"

@app.get("/dict-default")
async def dict_default_endpoint():
    d = {}
    result = d.get("key", slow_default())
    return {"result": result}
""",
        "Blocking call as default argument to dict.get()",
    ),
    # 24. Blocking in f-string expression
    (
        "fstring_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def slow_fmt():
    time.sleep(1)
    return "done"

@app.get("/fstring")
async def fstring_endpoint():
    msg = f"status: {slow_fmt()}"
    return {"msg": msg}
""",
        "Blocking call inside f-string expression",
    ),
    # 25. Module-level blocking function called indirectly via class attribute
    (
        "class_attr_blocking",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def wait():
    time.sleep(1)

class Runner:
    action = wait

r = Runner()

@app.get("/class-attr")
async def class_attr_endpoint():
    r.action()
    return {"ok": True}
""",
        "Blocking function stored as class attribute",
    ),
    # 26. Decorator with args
    (
        "decorator_with_args",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def retry(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                time.sleep(1)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@retry(3)
def fetch():
    return 42

@app.get("/retry")
async def retry_endpoint():
    result = fetch()
    return {"result": result}
""",
        "Decorator with arguments that adds blocking calls",
    ),
]

print("=" * 80)
print("NEW ADVERSARIAL TEST RESULTS")
print("=" * 80)

detected = 0
missed = 0
missed_cases = []
detected_cases = []

for name, source, description in cases:
    result = checker.check_source(source, filepath=f"<{name}>")
    found = len(result.violations)
    
    if found > 0:
        status = "✅ DETECTED"
        detected += 1
        detected_cases.append(name)
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
if missed > 0:
    print(f"\nMISSED CASES ({missed}):")
    for name in missed_cases:
        print(f"  - {name}")
