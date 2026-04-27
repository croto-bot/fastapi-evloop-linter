from __future__ import annotations

from fastapi_evloop_linter.checker import EventLoopChecker


def messages(source: str) -> list[str]:
    result = EventLoopChecker(max_depth=20).check_source(source, filepath="<semantic>")
    return [violation.message for violation in result.violations]


def assert_reports(source: str, expected: str) -> None:
    found = messages(source)
    assert any(expected in message for message in found), found


def assert_clean(source: str) -> None:
    assert messages(source) == []


def test_qualified_method_identity_avoids_same_name_collision() -> None:
    source = """\
import time
from fastapi import FastAPI
app = FastAPI()

class Slow:
    def process(self):
        time.sleep(1)

class Fast:
    def process(self):
        return "ok"

slow = Slow()
fast = Fast()

@app.get("/slow")
async def slow_ep():
    slow.process()

@app.get("/fast")
async def fast_ep():
    fast.process()
"""
    found = messages(source)
    assert any("time.sleep" in message for message in found), found
    assert len(found) == 1


def test_awaited_local_return_type_propagates_to_receiver_methods() -> None:
    source = """\
import sqlite3
from fastapi import FastAPI
app = FastAPI()

async def get_conn():
    return sqlite3.connect("data.db")

@app.get("/db")
async def db_ep():
    conn = await get_conn()
    return conn.execute("SELECT 1").fetchall()
"""
    assert_reports(source, "sqlite3.connect.execute")


def test_bound_method_variable_reports_blocking_body() -> None:
    source = """\
import time
from fastapi import FastAPI
app = FastAPI()

class Worker:
    def process(self):
        time.sleep(1)

w = Worker()

@app.get("/bound")
async def ep():
    fn = w.process
    fn()
"""
    assert_reports(source, "time.sleep")


def test_bound_method_callback_reports_for_sync_local_helper() -> None:
    source = """\
import time
from fastapi import FastAPI
app = FastAPI()

class Worker:
    def process(self, item):
        time.sleep(1)
        return item

def apply_one(item, fn):
    return fn(item)

w = Worker()

@app.get("/callback")
async def ep():
    return apply_one("x", w.process)
"""
    assert_reports(source, "time.sleep")


def test_offloaded_bound_method_does_not_report_event_loop_blocking_body() -> None:
    source = """\
import asyncio
import time
from fastapi import FastAPI
app = FastAPI()

class Worker:
    def process(self):
        time.sleep(1)

w = Worker()

@app.get("/thread")
async def ep():
    await asyncio.to_thread(w.process)
"""
    assert_clean(source)


def test_asyncio_run_is_reported_but_normal_asyncio_helpers_are_safe() -> None:
    source = """\
import asyncio
from fastapi import FastAPI
app = FastAPI()

async def bg():
    return None

@app.get("/bad")
async def bad():
    asyncio.run(bg())

@app.get("/good")
async def good():
    asyncio.create_task(bg())
    await asyncio.gather(bg())
"""
    found = messages(source)
    assert any("asyncio.run" in message for message in found), found
    assert len(found) == 1


def test_context_manager_as_variable_propagates_resource_type() -> None:
    source = """\
import sqlite3
from fastapi import FastAPI
app = FastAPI()

@app.get("/db")
async def ep():
    with sqlite3.connect("data.db") as conn:
        return conn.execute("SELECT 1").fetchall()
"""
    assert_reports(source, "sqlite3.connect.execute")


def test_open_context_manager_file_methods_report() -> None:
    source = """\
from fastapi import FastAPI
app = FastAPI()

@app.get("/file")
async def ep():
    with open("data.txt") as f:
        return f.read()
"""
    assert_reports(source, "io.TextIOWrapper.read")


def test_descriptor_get_reports_blocking_body() -> None:
    source = """\
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

@app.get("/descriptor")
async def ep():
    return svc.data
"""
    assert_reports(source, "time.sleep")


def test_augmented_assignment_reports_iadd_body() -> None:
    source = """\
import time
from fastapi import FastAPI
app = FastAPI()

class Accumulator:
    def __iadd__(self, value):
        time.sleep(1)
        return self

acc = Accumulator()

@app.get("/acc")
async def ep():
    global acc
    acc += 1
"""
    assert_reports(source, "time.sleep")


def test_iterator_next_reports_blocking_body() -> None:
    source = """\
import time
from fastapi import FastAPI
app = FastAPI()

class SlowIterator:
    def __iter__(self):
        return self

    def __next__(self):
        time.sleep(1)
        raise StopIteration

items = SlowIterator()

@app.get("/items")
async def ep():
    for item in items:
        return item
"""
    assert_reports(source, "time.sleep")


def test_super_call_resolves_to_base_method_body() -> None:
    source = """\
import time
from fastapi import FastAPI
app = FastAPI()

class BaseService:
    def fetch(self):
        time.sleep(1)
        return "data"

class Service(BaseService):
    def fetch(self):
        return super().fetch()

svc = Service()

@app.get("/super")
async def ep():
    return svc.fetch()
"""
    assert_reports(source, "time.sleep")


def test_function_local_dict_dispatch_reports_callable_values() -> None:
    source = """\
import time
from fastapi import FastAPI
app = FastAPI()

def wait():
    time.sleep(1)

@app.get("/dict")
async def ep(action: str):
    actions = {"go": wait}
    actions.get(action)()
"""
    assert_reports(source, "time.sleep")


def test_self_method_call_resolves_within_local_class() -> None:
    source = """\
import time
from fastapi import FastAPI
app = FastAPI()

class Service:
    def handle(self):
        self.slow()

    def slow(self):
        time.sleep(1)

svc = Service()

@app.get("/self")
async def ep():
    svc.handle()
"""
    assert_reports(source, "time.sleep")


def test_async_with_does_not_trace_sync_enter_method() -> None:
    source = """\
import time
from fastapi import FastAPI
app = FastAPI()

class Manager:
    def __enter__(self):
        time.sleep(1)
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

manager = Manager()

@app.get("/async-with")
async def ep():
    async with manager:
        return "ok"
"""
    assert_clean(source)


def test_class_attribute_imported_callable_reports() -> None:
    source = """\
import time
from fastapi import FastAPI
app = FastAPI()

class Service:
    slow = time.sleep

svc = Service()

@app.get("/attr-callable")
async def ep():
    svc.slow(1)
"""
    assert_reports(source, "time.sleep")


def test_local_function_shadows_imported_name() -> None:
    source = """\
from time import sleep
from fastapi import FastAPI
app = FastAPI()

def sleep(seconds):
    return None
"""
    assert_clean(source)


def test_later_local_function_shadows_imported_name() -> None:
    source = """\
from time import sleep
from fastapi import FastAPI
app = FastAPI()

@app.get("/shadow")
async def ep():
    sleep(1)

def sleep(seconds):
    return None
"""
    assert_clean(source)


def test_noise_suppression_still_ignores_common_safe_operations() -> None:
    source = """\
import asyncio
import logging
from datetime import datetime, UTC
from pathlib import Path
from fastapi import FastAPI, HTTPException
app = FastAPI()
log = logging.getLogger(__name__)

async def bg():
    return None

@app.get("/safe")
async def ep(s: str):
    asyncio.create_task(bg())
    log.info("ok")
    p = Path(s)
    if not p.is_absolute():
        raise HTTPException(status_code=400)
    return {"now": datetime.now(UTC), "name": p.name}
"""
    assert_clean(source)
