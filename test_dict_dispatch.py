#!/usr/bin/env python3
"""Test cases for function-local dictionary dispatch blind spot (TEC-11).

Demonstrates that the linter misses blocking calls invoked through
function-local dict dispatch (strategy/registry pattern). The linter
only tracks module-level dicts with imported function references.

Run:
    python3 test_dict_dispatch.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker

checker = EventLoopChecker(max_depth=20)

cases = [
    # Case 1: Basic function-local dict dispatch
    (
        "basic_local_dict",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def action_wait():
    time.sleep(1)
    return "waited"

def action_compute():
    time.sleep(2)
    return "computed"

@app.get("/dispatch")
async def dispatch_endpoint(action: str):
    actions = {"wait": action_wait, "compute": action_compute}
    fn = actions[action]
    return {"result": fn()}
""",
        "Function-local dict: actions[action]() calls blocking functions",
        True,  # Should be caught (expected blocking)
    ),
    # Case 2: Direct subscript call: actions["go"]()
    (
        "subscript_call",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def slow_action():
    time.sleep(1)
    return "done"

@app.get("/subscript")
async def subscript_endpoint():
    actions = {"go": slow_action}
    return {"result": actions["go"]()}
""",
        "Direct subscript call: actions['go']() - dict lookup + call",
        True,
    ),
    # Case 3: dict.get() pattern
    (
        "dict_get_call",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def slow_action():
    time.sleep(1)
    return "done"

@app.get("/dict-get")
async def dict_get_endpoint():
    actions = {"go": slow_action}
    fn = actions.get("go")
    if fn:
        return {"result": fn()}
    return {"error": "not found"}
""",
        "actions.get('go')() - dict.get pattern",
        True,
    ),
    # Case 4: CRUD handler registry (very common FastAPI pattern)
    (
        "crud_registry",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def handle_create():
    time.sleep(1)
    return "created"

def handle_update():
    time.sleep(1)
    return "updated"

def handle_delete():
    time.sleep(1)
    return "deleted"

@app.get("/crud/{operation}")
async def crud_endpoint(operation: str):
    handlers = {
        "create": handle_create,
        "update": handle_update,
        "delete": handle_delete,
    }
    handler = handlers.get(operation)
    if handler:
        return {"result": handler()}
    return {"error": "unknown"}
""",
        "CRUD handler dict: handlers.get(operation)()",
        True,
    ),
    # Case 5: Module-level dict with local function refs (also broken)
    (
        "module_dict_local_fn",
        """\
import time
from fastapi import FastAPI

def action_wait():
    time.sleep(1)
    return "waited"

HANDLERS = {"wait": action_wait}

app = FastAPI()

@app.get("/module-dict")
async def module_dict_endpoint():
    fn = HANDLERS["wait"]
    return {"result": fn()}
""",
        "Module-level dict with local function action_wait (should work but doesn't)",
        True,
    ),
    # Case 6: Module-level dict with imported refs (contrast - works)
    (
        "module_dict_imported",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

HANDLERS = {"wait": time.sleep}

@app.get("/mod-imp")
async def mod_imp_endpoint():
    HANDLERS["wait"](1)
    return {"done": True}
""",
        "Module-level dict with time.sleep (imported ref) - should be caught",
        True,
    ),
    # Case 7: Class instances in function-local dict
    (
        "class_strategy_dict",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

class WaitStrategy:
    def execute(self):
        time.sleep(1)
        return "waited"

class ComputeStrategy:
    def execute(self):
        time.sleep(2)
        return "computed"

@app.get("/strategy")
async def strategy_endpoint(name: str):
    strategies = {
        "wait": WaitStrategy(),
        "compute": ComputeStrategy(),
    }
    strategy = strategies.get(name)
    return {"result": strategy.execute()}
""",
        "Class instances in function-local dict: strategies[name].execute()",
        True,
    ),
    # Case 8: Incrementally built dict (registry pattern)
    (
        "incremental_registry",
        """\
import time
from fastapi import FastAPI
app = FastAPI()

def plugin_a():
    time.sleep(0.5)
    return "plugin_a"

def plugin_b():
    time.sleep(0.3)
    return "plugin_b"

@app.get("/run-plugin/{name}")
async def run_plugin(name: str):
    registry = {}
    registry["a"] = plugin_a
    registry["b"] = plugin_b
    
    if name in registry:
        return {"result": registry[name]()}
    return {"error": "not found"}
""",
        "Registry built incrementally: registry[name]()",
        True,
    ),
]

print("=" * 80)
print("TEC-11: Function-local Dictionary Dispatch Blind Spot Tests")
print("=" * 80)

detected = 0
missed = 0

for name, source, description, should_block in cases:
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
    
    expected = "(should catch)" if should_block else "(safe)"
    print(f"{status} [{name}] {description} {expected}")
    print()

print("=" * 80)
print(f"SUMMARY: {detected} detected, {missed} missed out of {len(cases)} cases")
print(f"Miss rate: {missed}/{len(cases)} = {missed/len(cases)*100:.0f}%")
