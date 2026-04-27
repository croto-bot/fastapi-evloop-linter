#!/usr/bin/env python3
"""
Test for TEC-16: self.xxx instance attribute blind spot.

Verifies that blocking calls through self.xxx instance attributes are
detected. Before the fix, _track_assignment_type() only handled
isinstance(target, ast.Name), so self.conn = sqlite3.connect(...) was
completely invisible to the type tracker.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker

checker = EventLoopChecker(max_depth=20)

# ── Real-world pattern: repository/service class ──
# Uses unique endpoint names to avoid overwriting class methods in
# the analysis functions dict.
code = """\
import sqlite3
import smtplib
import subprocess
from fastapi import FastAPI

app = FastAPI()


class UserRepository:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

    def get_user(self, user_id: int):
        return self.conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()

    def create_user(self, name: str):
        self.conn.execute(
            "INSERT INTO users (name) VALUES (?)", (name,)
        )
        self.conn.commit()


class NotificationService:
    def __init__(self):
        self.server = smtplib.SMTP("smtp.example.com", 587)

    def send(self, to: str, body: str):
        self.server.sendmail("noreply@example.com", to, body)
        self.server.quit()


class CommandRunner:
    def run(self, cmd: str):
        self.proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        self.stdout, self.stderr = self.proc.communicate()


repo = UserRepository("data.db")
notifier = NotificationService()
runner = CommandRunner()


@app.get("/users/{user_id}")
async def handle_get_user(user_id: int):
    user = repo.get_user(user_id)
    return {"user": user}


@app.post("/users")
async def handle_create_user(name: str):
    repo.create_user(name)
    return {"created": True}


@app.post("/notify")
async def send_notification(to: str):
    notifier.send(to, "Hello!")
    return {"sent": True}


@app.post("/run")
async def run_command(cmd: str):
    runner.run(cmd)
    return {"done": True}
"""

result = checker.check_source(code, "test_instance_attr.py")

print("=" * 70)
print("INSTANCE ATTRIBUTE BLIND SPOT TEST (TEC-16)")
print("Pattern: self.conn = sqlite3.connect(...); self.conn.execute(...)")
print("=" * 70)
print(f"\nTotal violations: {len(result.violations)}")
print()
for v in result.violations:
    print(f"  Line {v.line}: {v.message}")

# Verify expected violations
# Reachable through call chain from endpoints:
#   handle_get_user -> repo.get_user -> self.conn.execute + .fetchone (2)
#   handle_create_user -> repo.create_user -> self.conn.execute + self.conn.commit (2)
#   send_notification -> notifier.send -> self.server.sendmail + self.server.quit (2)
#   run_command -> runner.run -> self.proc.communicate (1)
#   (subprocess.Popen is a stdlib class constructor → classified as SAFE)
#   (smtplib.SMTP is a stdlib class constructor → classified as SAFE)
# Total expected: 7
expected_violations = 7

print("\n" + "-" * 70)
if len(result.violations) >= expected_violations:
    print(f"PASS: {len(result.violations)} violations detected (expected >= {expected_violations})")
    sys.exit(0)
else:
    print(f"FAIL: {len(result.violations)} violations detected (expected >= {expected_violations})")
    sys.exit(1)
