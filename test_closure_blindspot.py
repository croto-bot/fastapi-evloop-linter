"""Test: Linter misses blocking calls in closures/factory functions that return inner functions.

The linter can trace into the outer function (catching direct blocking calls there),
but cannot resolve the returned inner function as a callable to trace into.

This is a generic pattern: factory functions, closures, partial application alternatives,
decorator builders, and event handler factories all use this pattern.
"""


# Case 1: Basic factory returning inner function — 0 violations (should be 1)
case1 = """
import time
from fastapi import FastAPI
app = FastAPI()

def make_waiter(seconds):
    def waiter():
        time.sleep(seconds)
        return "done"
    return waiter

@app.get("/test")
async def test():
    wait = make_waiter(1)
    result = wait()
    return {"result": result}
"""

# Case 2: Nested closures (3 levels) — 0 violations (should be 1)
case2 = """
import time
from fastapi import FastAPI
app = FastAPI()

def outer(x):
    def middle(y):
        def inner():
            time.sleep(x + y)
            return "done"
        return inner
    return middle(1)

@app.get("/test")
async def test():
    fn = outer(1)
    result = fn()
    return {"result": result}
"""

# Case 3: Module-level factory variable — 0 violations (should be 1)
case3 = """
import time
from fastapi import FastAPI
app = FastAPI()

def create_query_runner(db_path):
    def run_query(sql):
        time.sleep(1)
        return [{"id": 1}]
    return run_query

query_runner = create_query_runner("/data/app.db")

@app.get("/users")
async def get_users():
    results = query_runner("SELECT * FROM users")
    return {"users": results}
"""

# Case 4: Class method returning closure — 0 violations (should be 1)
case4 = """
import time
from fastapi import FastAPI
app = FastAPI()

class QueryBuilder:
    def make_query(self, table):
        def execute():
            time.sleep(1)
            return f"SELECT * FROM {table}"
        return execute

builder = QueryBuilder()

@app.get("/data")
async def get_data():
    query = builder.make_query("users")
    result = query()
    return {"sql": result}
"""

# Case 5: Outer function also blocks — catches outer's direct call but misses inner — 1 violation (should be 2)
case5 = """
import time
from fastapi import FastAPI
app = FastAPI()

def setup_and_run():
    time.sleep(0.1)  # CAUGHT
    def runner():
        time.sleep(1)  # MISSED
        return "done"
    return runner

@app.get("/test")
async def test():
    fn = setup_and_run()
    result = fn()
    return {"result": result}
"""

# Case 6: Real-world pattern — DB connection factory — 0 violations (should be 1)
case6 = """
import time
from fastapi import FastAPI
app = FastAPI()

def create_db_connection(db_path):
    def execute_query(sql):
        time.sleep(1)
        return [{"id": 1}]
    return execute_query

@app.get("/users")
async def get_users():
    query_fn = create_db_connection("app.db")
    results = query_fn("SELECT * FROM users")
    return {"users": results}
"""

# Case 7: Event handler factory — 0 violations (should be 1)
case7 = """
import time
from fastapi import FastAPI
app = FastAPI()

def create_handler(event_type):
    def handle(data):
        time.sleep(0.5)
        return {"processed": event_type, "data": data}
    return handle

handle_click = create_handler("click")

@app.get("/event")
async def process_event():
    result = handle_click({"x": 100, "y": 200})
    return result
"""

# Case 8: Multiple inner functions returned conditionally — 0 violations (should be 2)
case8 = """
import time
from fastapi import FastAPI
app = FastAPI()

def get_processor(mode):
    if mode == "slow":
        def process():
            time.sleep(2)
            return "slow_result"
        return process
    else:
        def process():
            time.sleep(1)
            return "fast_result"
        return process

@app.get("/process/{mode}")
async def process_data(mode: str):
    proc = get_processor(mode)
    result = proc()
    return {"result": result}
"""

# CONTRAST: Direct call to local function — WORKS correctly (1 violation)
contrast = """
import time
from fastapi import FastAPI
app = FastAPI()

def slow_func():
    time.sleep(1)
    return "done"

@app.get("/test")
async def test():
    result = slow_func()
    return {"result": result}
"""

# CONTRAST: Lambda returned from function — WORKS (1 violation, via generic_visit into lambda)
contrast_lambda = """
import time
from fastapi import FastAPI
app = FastAPI()

def make_delayer(seconds):
    return lambda: time.sleep(seconds)

@app.get("/test")
async def test():
    delay = make_delayer(1)
    delay()
    return {"done": True}
"""
