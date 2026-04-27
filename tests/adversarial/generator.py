"""Adversarial test case generator for fastapi-evloop-linter.

Generates tricky FastAPI code patterns that the linter should detect,
organized by difficulty level. Each test case has:
- A Python source file with blocking calls
- Metadata about expected violations (how many, at what depth, etc.)

The adversarial generator tries to create patterns that are hard to detect:
- Deep call chains
- Indirect imports
- Wrapper functions
- Dynamic patterns (partial: we can't detect truly dynamic calls, but should detect static patterns)
- Conditional branches with blocking calls
- Helper functions in other "modules" (simulated via analysis of multiple functions)
"""

from __future__ import annotations

import json
import random
import string
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestCase:
    """A single adversarial test case."""
    name: str
    source: str
    difficulty: int  # 1-10
    expected_violations: int  # How many blocking calls the linter SHOULD find
    expected_min_depth: int  # Minimum expected depth of deepest violation
    category: str  # e.g., "deep_chain", "indirect_import", "wrapper"
    description: str = ""


def generate_all_cases() -> list[TestCase]:
    """Generate all adversarial test cases."""
    cases = []
    cases.extend(_simple_blocking_cases())
    cases.extend(_deep_chain_cases())
    cases.extend(_indirect_import_cases())
    cases.extend(_wrapper_function_cases())
    cases.extend(_conditional_cases())
    cases.extend(_method_call_cases())
    cases.extend(_multi_endpoint_cases())
    cases.extend(_class_method_cases())
    cases.extend(_chained_calls_cases())
    cases.extend(_nested_import_cases())
    cases.extend(_type_inference_cases())
    cases.extend(_callback_pattern_cases())
    cases.extend(_router_decorator_cases())
    cases.extend(_multiple_blocking_cases())
    cases.extend(_deep_chain_7_cases())
    cases.extend(_callback_deep_cases())
    cases.extend(_open_builtin_cases())
    cases.extend(_aliased_module_cases())
    cases.extend(_variable_alias_cases())
    cases.extend(_class_dunder_cases())
    cases.extend(_blocking_decorator_cases())
    cases.extend(_blocking_constructor_cases())
    cases.extend(_higher_order_cases())
    cases.extend(_property_blocking_cases())
    cases.extend(_higher_order_extra_cases())
    cases.extend(_async_dunder_cases())
    cases.extend(_class_attr_cases())
    cases.extend(_decorator_factory_cases())
    cases.extend(_stdlib_blocking_cases())
    return cases


def _simple_blocking_cases() -> list[TestCase]:
    """Basic cases: direct blocking calls in async endpoints."""
    return [
        TestCase(
            name="simple_time_sleep",
            source=textwrap.dedent("""\
                import time
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/slow")
                async def slow_endpoint():
                    time.sleep(10)
                    return {"ok": True}
            """),
            difficulty=1,
            expected_violations=1,
            expected_min_depth=0,
            category="simple",
            description="Direct time.sleep() in endpoint",
        ),
        TestCase(
            name="simple_requests_get",
            source=textwrap.dedent("""\
                import requests
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/fetch")
                async def fetch_data():
                    resp = requests.get("https://api.example.com/data")
                    return resp.json()
            """),
            difficulty=1,
            expected_violations=1,
            expected_min_depth=0,
            category="simple",
            description="Direct requests.get() in endpoint",
        ),
        TestCase(
            name="simple_subprocess",
            source=textwrap.dedent("""\
                import subprocess
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/run")
                async def run_cmd():
                    result = subprocess.run(["ls", "-la"])
                    return {"exit_code": result.returncode}
            """),
            difficulty=1,
            expected_violations=1,
            expected_min_depth=0,
            category="simple",
            description="Direct subprocess.run() in endpoint",
        ),
    ]


def _deep_chain_cases() -> list[TestCase]:
    """Cases with blocking calls deep in the call chain."""
    return [
        TestCase(
            name="depth_2_chain",
            source=textwrap.dedent("""\
                import time
                from fastapi import FastAPI
                app = FastAPI()

                def helper():
                    time.sleep(1)

                @app.get("/chain")
                async def chain_endpoint():
                    helper()
                    return {"ok": True}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=1,
            category="deep_chain",
            description="time.sleep() at depth 2 (endpoint -> helper)",
        ),
        TestCase(
            name="depth_3_chain",
            source=textwrap.dedent("""\
                import requests
                from fastapi import FastAPI
                app = FastAPI()

                def fetcher():
                    requests.get("http://example.com")

                def processor():
                    fetcher()

                @app.get("/deep")
                async def deep_endpoint():
                    processor()
                    return {"ok": True}
            """),
            difficulty=4,
            expected_violations=1,
            expected_min_depth=2,
            category="deep_chain",
            description="requests.get() at depth 3 (endpoint -> processor -> fetcher)",
        ),
        TestCase(
            name="depth_4_chain",
            source=textwrap.dedent("""\
                import subprocess
                from fastapi import FastAPI
                app = FastAPI()

                def run_command():
                    subprocess.run(["echo", "hello"])

                def execute_pipeline():
                    run_command()

                def process_data():
                    execute_pipeline()

                @app.get("/pipeline")
                async def pipeline_endpoint():
                    process_data()
                    return {"ok": True}
            """),
            difficulty=5,
            expected_violations=1,
            expected_min_depth=3,
            category="deep_chain",
            description="subprocess.run() at depth 4",
        ),
        TestCase(
            name="depth_5_chain",
            source=textwrap.dedent("""\
                import time
                from fastapi import FastAPI
                app = FastAPI()

                def level5():
                    time.sleep(1)

                def level4():
                    level5()

                def level3():
                    level4()

                def level2():
                    level3()

                def level1():
                    level2()

                @app.get("/deep5")
                async def deep5_endpoint():
                    level1()
                    return {"ok": True}
            """),
            difficulty=6,
            expected_violations=1,
            expected_min_depth=5,
            category="deep_chain",
            description="time.sleep() at depth 5",
        ),
    ]


def _indirect_import_cases() -> list[TestCase]:
    """Cases with aliased imports."""
    return [
        TestCase(
            name="aliased_import",
            source=textwrap.dedent("""\
                import requests as req
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/aliased")
                async def aliased_endpoint():
                    resp = req.get("https://api.example.com")
                    return resp.json()
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="indirect_import",
            description="requests imported as req, then req.get() called",
        ),
        TestCase(
            name="from_import",
            source=textwrap.dedent("""\
                from time import sleep
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/from-import")
                async def from_import_endpoint():
                    sleep(5)
                    return {"ok": True}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="indirect_import",
            description="from time import sleep, then sleep() called",
        ),
        TestCase(
            name="aliased_from_import",
            source=textwrap.dedent("""\
                from subprocess import run as execute
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/aliased-from")
                async def aliased_from_endpoint():
                    execute(["echo", "test"])
                    return {"ok": True}
            """),
            difficulty=4,
            expected_violations=1,
            expected_min_depth=0,
            category="indirect_import",
            description="from subprocess import run as execute",
        ),
        TestCase(
            name="aliased_import_deep",
            source=textwrap.dedent("""\
                import requests as http_client
                from fastapi import FastAPI
                app = FastAPI()

                def do_request():
                    http_client.post("https://api.example.com", json={"key": "val"})

                @app.get("/aliased-deep")
                async def aliased_deep_endpoint():
                    do_request()
                    return {"ok": True}
            """),
            difficulty=5,
            expected_violations=1,
            expected_min_depth=1,
            category="indirect_import",
            description="Aliased import used deep in call chain",
        ),
    ]


def _wrapper_function_cases() -> list[TestCase]:
    """Cases where blocking calls are wrapped in helper functions."""
    return [
        TestCase(
            name="sync_wrapper",
            source=textwrap.dedent("""\
                import requests
                from fastapi import FastAPI
                app = FastAPI()

                def make_request(url):
                    return requests.get(url)

                @app.get("/wrapper")
                async def wrapper_endpoint():
                    data = make_request("https://api.example.com")
                    return data.json()
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=1,
            category="wrapper",
            description="requests.get() wrapped in sync helper",
        ),
        TestCase(
            name="multi_wrapper",
            source=textwrap.dedent("""\
                import time
                import requests
                from fastapi import FastAPI
                app = FastAPI()

                def wait():
                    time.sleep(2)

                def fetch():
                    requests.get("http://example.com")

                @app.get("/multi-wrap")
                async def multi_wrapper_endpoint():
                    wait()
                    fetch()
                    return {"ok": True}
            """),
            difficulty=3,
            expected_violations=2,
            expected_min_depth=1,
            category="wrapper",
            description="Two separate wrappers, each with blocking calls",
        ),
    ]


def _conditional_cases() -> list[TestCase]:
    """Cases with blocking calls inside conditionals."""
    return [
        TestCase(
            name="conditional_blocking",
            source=textwrap.dedent("""\
                import time
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/cond")
                async def conditional_endpoint(flag: bool):
                    if flag:
                        time.sleep(5)
                    return {"ok": True}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="conditional",
            description="time.sleep() inside an if block",
        ),
        TestCase(
            name="conditional_deep",
            source=textwrap.dedent("""\
                import requests
                from fastapi import FastAPI
                app = FastAPI()

                def fetch(url):
                    return requests.get(url)

                @app.get("/cond-deep")
                async def conditional_deep_endpoint(mode: str):
                    if mode == "sync":
                        fetch("https://api.example.com")
                    return {"ok": True}
            """),
            difficulty=4,
            expected_violations=1,
            expected_min_depth=1,
            category="conditional",
            description="Blocking call inside conditional + deep chain",
        ),
    ]


def _method_call_cases() -> list[TestCase]:
    """Cases with blocking method calls on objects."""
    return [
        TestCase(
            name="pathlib_methods",
            source=textwrap.dedent("""\
                from pathlib import Path
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/read-file")
                async def read_file():
                    content = Path("/tmp/data.txt").read_text()
                    return {"content": content}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="method_call",
            description="Path.read_text() on a pathlib object",
        ),
        TestCase(
            name="pathlib_variable",
            source=textwrap.dedent("""\
                from pathlib import Path
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/path-var")
                async def path_variable():
                    p = Path("/tmp/data.txt")
                    content = p.read_text()
                    return {"content": content}
            """),
            difficulty=4,
            expected_violations=1,
            expected_min_depth=0,
            category="method_call",
            description="Path variable, then .read_text() on it",
        ),
    ]


def _multi_endpoint_cases() -> list[TestCase]:
    """Cases with multiple endpoints, some clean, some not."""
    return [
        TestCase(
            name="mixed_endpoints",
            source=textwrap.dedent("""\
                import time
                import requests
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/clean")
                async def clean_endpoint():
                    return {"status": "ok"}

                @app.get("/blocking")
                async def blocking_endpoint():
                    time.sleep(1)
                    return {"status": "blocked"}

                @app.post("/also-blocking")
                async def also_blocking():
                    requests.post("http://example.com", json={"a": 1})
                    return {"status": "blocked"}
            """),
            difficulty=2,
            expected_violations=2,
            expected_min_depth=0,
            category="multi_endpoint",
            description="Three endpoints: one clean, two with blocking calls",
        ),
    ]


def _class_method_cases() -> list[TestCase]:
    """Cases where blocking calls are in class methods."""
    return [
        TestCase(
            name="class_method_helper",
            source=textwrap.dedent("""\
                import time
                from fastapi import FastAPI
                app = FastAPI()

                class Service:
                    def process(self):
                        time.sleep(2)

                svc = Service()

                @app.get("/svc")
                async def service_endpoint():
                    svc.process()
                    return {"ok": True}
            """),
            difficulty=7,
            expected_violations=1,
            expected_min_depth=1,
            category="class_method",
            description="Blocking call inside class method (may not be detectable without cross-class analysis)",
        ),
    ]


def _chained_calls_cases() -> list[TestCase]:
    """Cases with chained method calls."""
    return [
        TestCase(
            name="chained_session",
            source=textwrap.dedent("""\
                import requests
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/chained")
                async def chained_endpoint():
                    resp = requests.Session().get("https://api.example.com")
                    return resp.json()
            """),
            difficulty=4,
            expected_violations=2,
            expected_min_depth=0,
            category="chained",
            description="requests.Session() and .get() chained",
        ),
    ]


def _nested_import_cases() -> list[TestCase]:
    """Cases with nested/dotted imports."""
    return [
        TestCase(
            name="nested_module",
            source=textwrap.dedent("""\
                import urllib.request
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/urllib")
                async def urllib_endpoint():
                    resp = urllib.request.urlopen("https://example.com")
                    return {"ok": True}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="nested_import",
            description="urllib.request.urlopen() with dotted module",
        ),
    ]


def _type_inference_cases() -> list[TestCase]:
    """Cases requiring type inference to detect."""
    return [
        TestCase(
            name="pathlib_from_import",
            source=textwrap.dedent("""\
                from pathlib import Path
                from fastapi import FastAPI
                app = FastAPI()

                def read_config():
                    return Path("/etc/config.yaml").read_text()

                @app.get("/config")
                async def config_endpoint():
                    data = read_config()
                    return {"config": data}
            """),
            difficulty=5,
            expected_violations=1,
            expected_min_depth=1,
            category="type_inference",
            description="Path.read_text() inside helper called from endpoint",
        ),
    ]


def _callback_pattern_cases() -> list[TestCase]:
    """Cases with callback-like patterns."""
    return [
        TestCase(
            name="callback_blocking",
            source=textwrap.dedent("""\
                import time
                from fastapi import FastAPI
                app = FastAPI()

                def apply_callback(data, callback):
                    result = callback(data)
                    return result

                def slow_transform(x):
                    time.sleep(1)
                    return x * 2

                @app.get("/callback")
                async def callback_endpoint():
                    result = apply_callback(42, slow_transform)
                    return {"result": result}
            """),
            difficulty=8,
            expected_violations=1,
            expected_min_depth=2,
            category="callback",
            description="Blocking call inside callback passed to helper (very hard to detect statically)",
        ),
    ]


def _router_decorator_cases() -> list[TestCase]:
    """Cases using APIRouter instead of app."""
    return [
        TestCase(
            name="router_decorator",
            source=textwrap.dedent("""\
                import time
                from fastapi import APIRouter
                router = APIRouter()

                @router.get("/items")
                async def list_items():
                    time.sleep(1)
                    return []
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="router_decorator",
            description="Blocking call in router-decorated endpoint",
        ),
        TestCase(
            name="router_deep",
            source=textwrap.dedent("""\
                import requests
                from fastapi import APIRouter
                router = APIRouter()

                def get_data():
                    return requests.get("https://api.example.com")

                @router.post("/submit")
                async def submit():
                    data = get_data()
                    return {"ok": True}
            """),
            difficulty=4,
            expected_violations=1,
            expected_min_depth=1,
            category="router_decorator",
            description="Deep blocking call in router endpoint",
        ),
    ]


def _multiple_blocking_cases() -> list[TestCase]:
    """Cases with multiple blocking calls in a single function."""
    return [
        TestCase(
            name="multi_blocking_one_func",
            source=textwrap.dedent("""\
                import time
                import requests
                import subprocess
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/heavy")
                async def heavy():
                    time.sleep(1)
                    resp = requests.get("https://example.com")
                    subprocess.run(["echo", "hello"])
                    return {"ok": True}
            """),
            difficulty=1,
            expected_violations=3,
            expected_min_depth=0,
            category="multi_blocking",
            description="Three different blocking calls in one endpoint",
        ),
        TestCase(
            name="multi_blocking_deep",
            source=textwrap.dedent("""\
                import time
                import requests
                from fastapi import FastAPI
                app = FastAPI()

                def wait():
                    time.sleep(1)

                def fetch():
                    requests.get("https://example.com")

                @app.get("/multi-deep")
                async def multi_deep():
                    wait()
                    fetch()
                    return {"ok": True}
            """),
            difficulty=3,
            expected_violations=2,
            expected_min_depth=1,
            category="multi_blocking",
            description="Two blocking wrappers called from one endpoint",
        ),
    ]


def _deep_chain_7_cases() -> list[TestCase]:
    """Very deep call chains (7+ levels)."""
    return [
        TestCase(
            name="depth_7_chain",
            source=textwrap.dedent("""\
                import os
                from fastapi import FastAPI
                app = FastAPI()

                def l7(): os.system("echo deep")
                def l6(): l7()
                def l5(): l6()
                def l4(): l5()
                def l3(): l4()
                def l2(): l3()
                def l1(): l2()

                @app.get("/d7")
                async def d7():
                    l1()
                    return {"ok": True}
            """),
            difficulty=7,
            expected_violations=1,
            expected_min_depth=7,
            category="deep_chain_7",
            description="os.system at depth 7",
        ),
    ]


def _callback_deep_cases() -> list[TestCase]:
    """Callback patterns with deeper nesting."""
    return [
        TestCase(
            name="callback_deep",
            source=textwrap.dedent("""\
                import time
                from fastapi import FastAPI
                app = FastAPI()

                def runner(fn, val):
                    return fn(val)

                def blocker(x):
                    time.sleep(x)
                    return x

                @app.get("/cb-deep")
                async def cb_deep():
                    result = runner(blocker, 5)
                    return {"result": result}
            """),
            difficulty=8,
            expected_violations=1,
            expected_min_depth=2,
            category="callback_deep",
            description="time.sleep in callback passed through runner",
        ),
        TestCase(
            name="callback_with_extra_args",
            source=textwrap.dedent("""\
                import requests
                from fastapi import FastAPI
                app = FastAPI()

                def map_fn(items, transform):
                    return [transform(i) for i in items]

                def fetch_item(i):
                    return requests.get(f"https://api.example.com/items/{i}")

                @app.get("/map-cb")
                async def map_cb():
                    results = map_fn([1, 2, 3], fetch_item)
                    return {"count": len(results)}
            """),
            difficulty=9,
            expected_violations=1,
            expected_min_depth=2,
            category="callback_deep",
            description="requests.get inside callback used in map-like function",
        ),
    ]


def _open_builtin_cases() -> list[TestCase]:
    """Cases using builtin open() for file I/O."""
    return [
        TestCase(
            name="builtin_open",
            source=textwrap.dedent("""\
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/read")
                async def read_data():
                    with open("/tmp/data.txt") as f:
                        return f.read()
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="builtin_open",
            description="builtin open() in endpoint",
        ),
        TestCase(
            name="open_in_helper",
            source=textwrap.dedent("""\
                from fastapi import FastAPI
                app = FastAPI()

                def load_file(path):
                    with open(path) as f:
                        return f.read()

                @app.get("/load")
                async def load():
                    data = load_file("/tmp/config.yaml")
                    return {"data": data}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=1,
            category="builtin_open",
            description="open() inside helper function",
        ),
    ]


def _aliased_module_cases() -> list[TestCase]:
    """More aliased import patterns."""
    return [
        TestCase(
            name="aliased_os",
            source=textwrap.dedent("""\
                import os as operating_system
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/cmd")
                async def run_cmd():
                    operating_system.system("ls")
                    return {"ok": True}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="aliased_module",
            description="os imported as operating_system",
        ),
        TestCase(
            name="aliased_time",
            source=textwrap.dedent("""\
                import time as t
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/wait")
                async def wait():
                    t.sleep(10)
                    return {"ok": True}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="aliased_module",
            description="time imported as t",
        ),
    ]


def generate_random_cases(count: int = 10, seed: int | None = None) -> list[TestCase]:
    """Generate random adversarial test cases by combining patterns."""
    if seed is not None:
        random.seed(seed)

    blocking_calls = [
        ("time", "sleep({n})", "time"),
        ("requests", "requests.get('{url}')", "requests"),
        ("requests", "requests.post('{url}', json=data)", "requests"),
        ("subprocess", "subprocess.run(['{cmd}', '{arg}'])", "subprocess"),
        ("os", "os.system('{cmd}')", "os"),
    ]

    wrapper_names = [
        "helper", "process", "handle", "execute", "run_task",
        "do_work", "perform", "compute", "resolve", "fetch_data",
    ]

    cases = []
    for i in range(count):
        depth = random.randint(1, 5)
        blocking = random.choice(blocking_calls)
        import_module = blocking[2]
        blocking_code = blocking[1].format(
            n=random.randint(1, 10),
            url=f"http://example.com/api/v{random.randint(1, 5)}",
            cmd=random.choice(["ls", "cat", "echo", "grep"]),
            arg=random.choice(["test", "data", "hello"]),
        )

        # Build a chain of wrapper functions
        wrappers = random.sample(wrapper_names, min(depth, len(wrapper_names)))
        source_lines = [f"import {import_module}"]
        source_lines.append("from fastapi import FastAPI")
        source_lines.append("app = FastAPI()")
        source_lines.append("")

        # Build wrappers from deepest to shallowest
        for j, name in enumerate(wrappers):
            if j == 0:
                # Deepest: contains the blocking call
                source_lines.append(f"def {name}():")
                source_lines.append(f"    {blocking_code}")
                source_lines.append("")
            else:
                source_lines.append(f"def {name}():")
                source_lines.append(f"    {wrappers[j-1]}()")
                source_lines.append("")

        # Build endpoint
        endpoint_name = f"endpoint_{i}"
        source_lines.append(f"@app.get('/{endpoint_name}')")
        source_lines.append("async def {}():".format(endpoint_name))
        if wrappers:
            source_lines.append(f"    {wrappers[-1]}()")
        else:
            source_lines.append(f"    {blocking_code}")
        source_lines.append("    return {'ok': True}")

        source = "\n".join(source_lines)

        cases.append(TestCase(
            name=f"random_{i}_d{depth}",
            source=source,
            difficulty=min(depth + 2, 10),
            expected_violations=1,
            expected_min_depth=depth - 1 if depth > 1 else 0,
            category="random",
            description=f"Random case with {import_module} at depth {depth}",
        ))

    return cases


def write_test_cases(cases: list[TestCase], output_dir: str | Path) -> None:
    """Write test cases as Python files to a directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for case in cases:
        filename = f"{case.name}.py"
        (output_dir / filename).write_text(case.source)
        manifest.append({
            "name": case.name,
            "filename": filename,
            "difficulty": case.difficulty,
            "expected_violations": case.expected_violations,
            "expected_min_depth": case.expected_min_depth,
            "category": case.category,
            "description": case.description,
        })

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _variable_alias_cases() -> list[TestCase]:
    """Cases where blocking functions are accessed via variable aliasing."""
    return [
        TestCase(
            name="variable_alias_sleep",
            source=textwrap.dedent("""\
                import time
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/alias")
                async def alias_endpoint():
                    f = time.sleep
                    f(1)
                    return {"ok": True}
            """),
            difficulty=6,
            expected_violations=1,
            expected_min_depth=0,
            category="variable_alias",
            description="Variable aliasing: f = time.sleep; f(1)",
        ),
        TestCase(
            name="walrus_alias",
            source=textwrap.dedent("""\
                import time
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/walrus")
                async def walrus_endpoint():
                    if (wait := time.sleep):
                        wait(1)
                    return {"ok": True}
            """),
            difficulty=7,
            expected_violations=1,
            expected_min_depth=0,
            category="variable_alias",
            description="Walrus operator: (wait := time.sleep); wait(1)",
        ),
        TestCase(
            name="cross_func_alias",
            source=textwrap.dedent("""\
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
            """),
            difficulty=8,
            expected_violations=1,
            expected_min_depth=0,
            category="variable_alias",
            description="Cross-function variable alias via return value",
        ),
        TestCase(
            name="functools_partial",
            source=textwrap.dedent("""\
                import time
                from functools import partial
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/partial")
                async def partial_endpoint():
                    wait = partial(time.sleep, 1)
                    wait()
                    return {"ok": True}
            """),
            difficulty=7,
            expected_violations=1,
            expected_min_depth=0,
            category="variable_alias",
            description="functools.partial wrapping time.sleep",
        ),
    ]


def _class_dunder_cases() -> list[TestCase]:
    """Cases where blocking calls happen in __call__, __enter__, __init__, etc."""
    return [
        TestCase(
            name="class_callable",
            source=textwrap.dedent("""\
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
            """),
            difficulty=8,
            expected_violations=1,
            expected_min_depth=1,
            category="class_dunder",
            description="Class with __call__ that contains blocking call",
        ),
        TestCase(
            name="context_manager_enter",
            source=textwrap.dedent("""\
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
            """),
            difficulty=8,
            expected_violations=1,
            expected_min_depth=1,
            category="class_dunder",
            description="Blocking call in context manager __enter__",
        ),
        TestCase(
            name="blocking_init",
            source=textwrap.dedent("""\
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
            """),
            difficulty=8,
            expected_violations=1,
            expected_min_depth=1,
            category="class_dunder",
            description="Blocking call in __init__ via constructor",
        ),
        TestCase(
            name="dataclass_postinit",
            source=textwrap.dedent("""\
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
            """),
            difficulty=9,
            expected_violations=1,
            expected_min_depth=1,
            category="class_dunder",
            description="Blocking in dataclass __post_init__",
        ),
        TestCase(
            name="operator_overload",
            source=textwrap.dedent("""\
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
            """),
            difficulty=9,
            expected_violations=1,
            expected_min_depth=1,
            category="class_dunder",
            description="Blocking via operator overloading (__mul__)",
        ),
    ]


def _blocking_decorator_cases() -> list[TestCase]:
    """Cases where a decorator introduces blocking behavior."""
    return [
        TestCase(
            name="blocking_decorator",
            source=textwrap.dedent("""\
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
            """),
            difficulty=8,
            expected_violations=1,
            expected_min_depth=1,
            category="blocking_decorator",
            description="Function wrapped in a blocking decorator (time.sleep in decorator wrapper)",
        ),
    ]


def _blocking_constructor_cases() -> list[TestCase]:
    """Cases where object construction triggers blocking calls."""
    return [
        TestCase(
            name="dict_dispatch",
            source=textwrap.dedent("""\
                import time
                from fastapi import FastAPI
                app = FastAPI()

                HANDLERS = {"wait": time.sleep}

                @app.get("/dispatch")
                async def dispatch_endpoint():
                    HANDLERS["wait"](5)
                    return {"ok": True}
            """),
            difficulty=7,
            expected_violations=1,
            expected_min_depth=0,
            category="dict_dispatch",
            description="Global dict dispatch: HANDLERS['wait'](5) calling time.sleep",
        ),
    ]


def _higher_order_cases() -> list[TestCase]:
    """Cases with map/filter and other higher-order function patterns."""
    return [
        TestCase(
            name="map_blocking",
            source=textwrap.dedent("""\
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
            """),
            difficulty=7,
            expected_violations=1,
            expected_min_depth=1,
            category="higher_order",
            description="Blocking call via map() builtin",
        ),
    ]


def _property_blocking_cases() -> list[TestCase]:
    """Cases where property access triggers blocking calls."""
    return [
        TestCase(
            name="blocking_property",
            source=textwrap.dedent("""\
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
            """),
            difficulty=8,
            expected_violations=1,
            expected_min_depth=0,
            category="blocking_property",
            description="Property that makes blocking HTTP request",
        ),
    ]


def _higher_order_extra_cases() -> list[TestCase]:
    """Cases with filter/sorted/list.sort as higher-order callers."""
    return [
        TestCase(
            name="filter_blocking",
            source=textwrap.dedent("""\
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
            """),
            difficulty=7,
            expected_violations=1,
            expected_min_depth=1,
            category="higher_order_extra",
            description="filter() with blocking predicate",
        ),
        TestCase(
            name="sorted_blocking",
            source=textwrap.dedent("""\
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
            """),
            difficulty=7,
            expected_violations=1,
            expected_min_depth=1,
            category="higher_order_extra",
            description="sorted() with blocking key function",
        ),
    ]


def _async_dunder_cases() -> list[TestCase]:
    """Cases with async context managers (__aenter__)."""
    return [
        TestCase(
            name="async_context_manager",
            source=textwrap.dedent("""\
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
            """),
            difficulty=8,
            expected_violations=1,
            expected_min_depth=1,
            category="async_dunder",
            description="Blocking call in __aenter__ of async context manager",
        ),
    ]


def _class_attr_cases() -> list[TestCase]:
    """Cases where blocking functions are stored as class attributes."""
    return [
        TestCase(
            name="class_attr_blocking",
            source=textwrap.dedent("""\
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
            """),
            difficulty=8,
            expected_violations=1,
            expected_min_depth=1,
            category="class_attr",
            description="Blocking function stored as class attribute",
        ),
    ]


def _decorator_factory_cases() -> list[TestCase]:
    """Cases with decorator factories (decorators with arguments)."""
    return [
        TestCase(
            name="decorator_with_args",
            source=textwrap.dedent("""\
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
            """),
            difficulty=9,
            expected_violations=1,
            expected_min_depth=1,
            category="decorator_factory",
            description="Decorator with arguments that adds blocking calls",
        ),
    ]


def _stdlib_blocking_cases() -> list[TestCase]:
    """Cases with stdlib blocking patterns not yet in the blockers registry."""
    return [
        TestCase(
            name="queue_get",
            source=textwrap.dedent("""\
                import queue
                from fastapi import FastAPI
                app = FastAPI()

                q = queue.Queue()

                @app.get("/queue")
                async def queue_endpoint():
                    item = q.get()
                    return {"item": item}
            """),
            difficulty=5,
            expected_violations=1,
            expected_min_depth=0,
            category="stdlib_blocking",
            description="queue.Queue.get() blocks the event loop",
        ),
        TestCase(
            name="lock_acquire",
            source=textwrap.dedent("""\
                import threading
                from fastapi import FastAPI
                app = FastAPI()

                lock = threading.Lock()

                @app.get("/lock")
                async def lock_endpoint():
                    lock.acquire()
                    lock.release()
                    return {"ok": True}
            """),
            difficulty=5,
            expected_violations=1,
            expected_min_depth=0,
            category="stdlib_blocking",
            description="threading.Lock.acquire() blocks the event loop",
        ),
    ]


if __name__ == "__main__":
    cases = generate_all_cases()
    cases.extend(generate_random_cases(10, seed=42))
    write_test_cases(cases, "tests/adversarial/cases")
    print(f"Generated {len(cases)} test cases")
