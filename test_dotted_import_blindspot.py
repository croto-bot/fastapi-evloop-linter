"""Test: Linter misses blocking method calls on objects from dotted module imports.

Bug: When using dotted imports like `import http.client`, the linter fails to
track var_module for objects constructed via `module.submodule.Class()`. This
causes subsequent method calls (e.g., conn.request()) to resolve as UNKNOWN
instead of BLOCKING.

The same pattern works correctly with `from` imports because those are tracked
via import_froms, allowing var_module to be set.

Root cause: In callgraph.py, _track_assignment_type only checks
isinstance(func.value, ast.Name) when setting var_module. For dotted imports,
func.value is an ast.Attribute (nested), so var_module is never set.

Issue: TEC-15
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker

checker = EventLoopChecker()


def test_http_client_dotted_import():
    """import http.client + HTTPConnection() + conn.request() → should flag as BLOCKING."""
    code = """\
import http.client
from fastapi import FastAPI
app = FastAPI()

@app.get("/fetch")
async def fetch_data():
    conn = http.client.HTTPConnection("api.example.com")
    conn.request("GET", "/data")
    response = conn.getresponse()
    body = response.read()
    return {"body": body}
"""
    result = checker.check_source(code, "test_http_client_dotted.py")
    # BUG: Currently returns 0 violations
    # Expected: at least conn.request() and conn.getresponse() should be flagged
    assert len(result.violations) > 0, (
        f"Expected blocking violations for http.client method calls, got {len(result.violations)}"
    )
    print(f"PASS: http.client dotted import → {len(result.violations)} violations")


def test_http_client_from_import():
    """from http.client import HTTPConnection → should work correctly."""
    code = """\
from http.client import HTTPConnection
from fastapi import FastAPI
app = FastAPI()

@app.get("/fetch")
async def fetch_data():
    conn = HTTPConnection("api.example.com")
    conn.request("GET", "/data")
    response = conn.getresponse()
    body = response.read()
    return {"body": body}
"""
    result = checker.check_source(code, "test_http_client_from.py")
    # This works correctly — the from-import allows var_module tracking
    assert len(result.violations) > 0, (
        f"Expected blocking violations for http.client from-import, got {len(result.violations)}"
    )
    print(f"PASS: http.client from-import → {len(result.violations)} violations")


def test_xmlrpc_client_dotted_import():
    """import xmlrpc.client + ServerProxy() + method calls → should flag as BLOCKING."""
    code = """\
import xmlrpc.client
from fastapi import FastAPI
app = FastAPI()

@app.get("/rpc")
async def rpc_endpoint():
    proxy = xmlrpc.client.ServerProxy("http://example.com/RPC2")
    result = proxy.system.listMethods()
    return {"result": result}
"""
    result = checker.check_source(code, "test_xmlrpc_dotted.py")
    # BUG: Currently returns 0 violations
    assert len(result.violations) > 0, (
        f"Expected blocking violations for xmlrpc.client method calls, got {len(result.violations)}"
    )
    print(f"PASS: xmlrpc.client dotted import → {len(result.violations)} violations")


def test_email_mime_dotted_import():
    """import email.mime.text + MIMEText() → should flag as blocking."""
    code = """\
import email.mime.text
from fastapi import FastAPI
app = FastAPI()

@app.get("/email")
async def email_endpoint():
    msg = email.mime.text.MIMEText("Hello World")
    return {"ok": True}
"""
    result = checker.check_source(code, "test_email_dotted.py")
    # BUG: Currently returns 0 violations
    assert len(result.violations) > 0, (
        f"Expected blocking violations for email.mime.text, got {len(result.violations)}"
    )
    print(f"PASS: email.mime.text dotted import → {len(result.violations)} violations")


if __name__ == "__main__":
    tests = [
        test_http_client_from_import,  # This one should pass
        test_http_client_dotted_import,  # Bug cases
        test_xmlrpc_client_dotted_import,
        test_email_mime_dotted_import,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
