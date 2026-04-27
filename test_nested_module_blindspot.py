"""Blind spot test: nested module class constructors (TEC-13).

The linter fails to detect blocking method calls when an object is
constructed from a nested module attribute (e.g., http.client.HTTPSConnection()).
The var_module tracking only handles simple `module.Class()` patterns but not
`a.b.Class()` patterns where the AST has nested Attribute nodes.

See: TEC-13
"""

import textwrap
import sys

sys.path.insert(0, "src")
from fastapi_evloop_linter.checker import EventLoopChecker

checker = EventLoopChecker()

# ── Case 1: http.client — all method calls missed ────────────────────
code_http_client = textwrap.dedent("""\
    import http.client
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/fetch")
    async def fetch_url():
        conn = http.client.HTTPSConnection("example.com")
        conn.request("GET", "/")
        resp = conn.getresponse()
        return {"status": resp.status}
""")
r = checker.check_source(code_http_client, "repro_http_client.py")
assert r.endpoints_found == 1, f"Expected 1 endpoint, got {r.endpoints_found}"
assert r.error_count == 0, (
    f"http.client: expected 0 violations (blind spot confirmed), "
    f"got {r.error_count}. "
    f"If this fails, the blind spot has been fixed!"
)
print(f"✓ Case 1 (http.client): {r.error_count} violations — blind spot confirmed")

# ── Case 2: xmlrpc.client — all method calls missed ──────────────────
code_xmlrpc = textwrap.dedent("""\
    import xmlrpc.client
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/rpc")
    async def call_rpc():
        proxy = xmlrpc.client.ServerProxy("http://example.com/RPC2")
        result = proxy.system.listMethods()
        return {"methods": result}
""")
r = checker.check_source(code_xmlrpc, "repro_xmlrpc.py")
assert r.endpoints_found == 1, f"Expected 1 endpoint, got {r.endpoints_found}"
assert r.error_count == 0, (
    f"xmlrpc.client: expected 0 violations (blind spot confirmed), "
    f"got {r.error_count}. "
    f"If this fails, the blind spot has been fixed!"
)
print(f"✓ Case 2 (xmlrpc.client): {r.error_count} violations — blind spot confirmed")

# ── Case 3: concurrent.futures — future.result() missed ──────────────
code_concurrent = textwrap.dedent("""\
    import concurrent.futures
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/compute")
    async def compute():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(pow, 2, 10)
            result = future.result()
        return {"result": result}
""")
r = checker.check_source(code_concurrent, "repro_concurrent.py")
assert r.endpoints_found == 1, f"Expected 1 endpoint, got {r.endpoints_found}"
assert r.error_count == 0, (
    f"concurrent.futures: expected 0 violations (blind spot confirmed), "
    f"got {r.error_count}. "
    f"If this fails, the blind spot has been fixed!"
)
print(f"✓ Case 3 (concurrent.futures): {r.error_count} violations — blind spot confirmed")

# ── Case 4: email.mime.text — constructor missed ─────────────────────
code_email = textwrap.dedent("""\
    import email.mime.text
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/email")
    async def send_email():
        msg = email.mime.text.MIMEText("Hello World")
        return {"ok": True}
""")
r = checker.check_source(code_email, "repro_email_mime.py")
assert r.endpoints_found == 1, f"Expected 1 endpoint, got {r.endpoints_found}"
assert r.error_count == 0, (
    f"email.mime.text: expected 0 violations (blind spot confirmed), "
    f"got {r.error_count}. "
    f"If this fails, the blind spot has been fixed!"
)
print(f"✓ Case 4 (email.mime.text): {r.error_count} violations — blind spot confirmed")

# ── Comparison: simple module works correctly ─────────────────────────
code_smtp = textwrap.dedent("""\
    import smtplib
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/email2")
    async def send_email2():
        server = smtplib.SMTP("smtp.example.com")
        server.sendmail("from@x.com", "to@x.com", "Hello")
        server.quit()
        return {"ok": True}
""")
r = checker.check_source(code_smtp, "repro_smtp_simple.py")
assert r.error_count >= 2, (
    f"smtplib (simple module): expected >=2 violations, got {r.error_count}. "
    f"Simple module patterns should work!"
)
print(f"✓ Comparison (smtplib simple): {r.error_count} violations — correctly detected")

print("\n" + "=" * 60)
print("BLIND SPOT CONFIRMED: Nested module constructors (a.b.Class())")
print("are not tracked, so method calls on their instances are UNKNOWN.")
print("Simple module constructors (a.Class()) work correctly.")
print("=" * 60)
