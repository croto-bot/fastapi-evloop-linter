#!/usr/bin/env python3
"""Test: getattr() dynamic dispatch completely bypasses blocking-call detection.

The linter resolves `requests.get(url)` correctly but produces zero violations
when the same call is made via `getattr(requests, 'get')(url)`.  This is a
generic gap: ANY blocking module + function combination is invisible to the
linter when accessed through getattr().

This is one of Python's most common patterns for:
- Generic HTTP proxies (dispatch on method name)
- Plugin / strategy systems
- Dynamic method selection
- Configuration-driven dispatch
- Test monkey-patching patterns

Run:
    python3 test_getattr_blindspot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker

checker = EventLoopChecker(max_depth=20)

# ── Helpers ──────────────────────────────────────────────────────────
passed = 0
failed = 0

def _test(name: str, code: str, expected_min: int = 1):
    global passed, failed
    r = checker.check_source(code)
    ok = len(r.violations) >= expected_min
    tag = "CAUGHT" if ok else "MISSED"
    print(f"  [{tag}] {name}: {len(r.violations)} violations (expected >= {expected_min})")
    for v in r.violations:
        print(f"         {v.message}")
    if ok:
        passed += 1
    else:
        failed += 1


# ── Case 1: getattr(time, 'sleep') — the simplest case ──────────────
print("=== Case 1: getattr(time, 'sleep') ===")
_test("getattr(time, 'sleep')()", """\
import time
from fastapi import FastAPI
app = FastAPI()

@app.get('/x')
async def endpoint():
    fn = getattr(time, 'sleep')
    fn(5)
    return {}
""")

# ── Case 2: getattr(requests, method) — real-world proxy pattern ─────
print("=== Case 2: getattr(requests, method) — HTTP proxy ===")
_test("getattr(requests, method.lower())(url)", """\
import requests
from fastapi import FastAPI
app = FastAPI()

@app.api_route('/proxy/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE'])
async def generic_proxy(path: str, method: str = 'GET'):
    http_fn = getattr(requests, method.lower())
    response = http_fn(f'http://backend-service:8080/{path}')
    return {'status': response.status_code, 'body': response.text}
""")

# ── Case 3: getattr(subprocess, 'run') ──────────────────────────────
print("=== Case 3: getattr(subprocess, 'run') ===")
_test("getattr(subprocess, 'run')([...])", """\
import subprocess
from fastapi import FastAPI
app = FastAPI()

@app.get('/run')
async def run_cmd():
    fn = getattr(subprocess, 'run')
    result = fn(['ls', '-la'], capture_output=True, text=True)
    return {'output': result.stdout}
""")

# ── Case 4: getattr on from-imported module alias ───────────────────
print("=== Case 4: getattr on module alias ===")
_test("getattr(urllib.request, 'urlopen')", """\
import urllib.request
from fastapi import FastAPI
app = FastAPI()

@app.get('/fetch')
async def fetch_url():
    fn = getattr(urllib.request, 'urlopen')
    resp = fn('http://example.com')
    return {}
""")

# ── Case 5: getattr with string variable ────────────────────────────
print("=== Case 5: getattr with string variable ===")
_test("getattr(time, method_name)", """\
import time
from fastapi import FastAPI
app = FastAPI()

@app.get('/delay')
async def delay_endpoint():
    method_name = 'sleep'
    fn = getattr(time, method_name)
    fn(1)
    return {}
""")

# ── Case 6: getattr + immediate call on result ──────────────────────
print("=== Case 6: getattr + immediate call ===")
_test("getattr(time, 'sleep')(5) — no intermediate var", """\
import time
from fastapi import FastAPI
app = FastAPI()

@app.get('/inline')
async def inline_endpoint():
    getattr(time, 'sleep')(5)
    return {}
""")

# ── Case 7: getattr on class constructor ────────────────────────────
print("=== Case 7: getattr for class constructor ===")
_test("getattr(smtplib, 'SMTP') + method calls", """\
import smtplib
from fastapi import FastAPI
app = FastAPI()

@app.get('/email')
async def send_email():
    cls = getattr(smtplib, 'SMTP')
    server = cls('smtp.example.com')
    server.sendmail('from@x.com', 'to@x.com', 'Hello')
    return {}
""")

# ── Case 8: getattr on local variable holding module ─────────────────
print("=== Case 8: getattr on local var (module object) ===")
_test("getattr(mod, 'sleep') where mod = time", """\
import time
from fastapi import FastAPI
app = FastAPI()

@app.get('/local-mod')
async def local_mod_endpoint():
    mod = time
    fn = getattr(mod, 'sleep')
    fn(1)
    return {}
""")

# ── Contrast: direct call IS caught ──────────────────────────────────
print("=== Contrast: direct call IS caught ===")
_test("requests.get() directly", """\
import requests
from fastapi import FastAPI
app = FastAPI()

@app.get('/direct')
async def direct_endpoint():
    response = requests.get('http://example.com')
    return {'status': response.status_code}
""")

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"Results: {passed} caught, {failed} missed out of {passed + failed} tests")
if failed > 0:
    print("BLIND SPOT CONFIRMED: getattr() dynamic dispatch bypasses all blocking detection")
