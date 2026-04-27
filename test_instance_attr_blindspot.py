#!/usr/bin/env python3
"""
Comprehensive test: The linter misses ALL blocking calls through self.xxx
instance attributes because _track_assignment_type only handles ast.Name targets.

This is distinct from TEC-14 (variable reassignment) because:
- TEC-14: local var = local var → type propagation missing
- THIS: self.attr = ModuleClass() → type NEVER tracked in the first place

Root cause: _track_assignment_type checks `isinstance(target, ast.Name)` which
is False for `self.conn` (an ast.Attribute node).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker

checker = EventLoopChecker(max_depth=20)

# ── Real-world pattern: repository/service class ──
code = """\
import sqlite3
import smtplib
import subprocess
from fastapi import FastAPI

app = FastAPI()


class UserRepository:
    '''Classic repository pattern - holds DB connection as self.conn.'''
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)        # BLOCKING (I/O)
        self.conn.row_factory = None                  # property access (safe but unresolved)
    
    def get_user(self, user_id: int):
        return self.conn.execute(                     # BLOCKING (DB I/O)
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()                                  # BLOCKING (DB I/O)
    
    def list_users(self):
        return self.conn.execute(                     # BLOCKING (DB I/O)
            "SELECT * FROM users"
        ).fetchall()                                  # BLOCKING (DB I/O)
    
    def create_user(self, name: str):
        self.conn.execute(                            # BLOCKING (DB I/O)
            "INSERT INTO users (name) VALUES (?)", (name,)
        )
        self.conn.commit()                            # BLOCKING (DB I/O)
    
    def close(self):
        self.conn.close()                             # BLOCKING (I/O)


class NotificationService:
    '''Service class - holds SMTP connection as self.server.'''
    
    def __init__(self):
        self.server = smtplib.SMTP("smtp.example.com", 587)  # constructor (safe but I/O setup)
        self.server.login("user", "pass")                      # BLOCKING (network I/O)
    
    def send(self, to: str, body: str):
        self.server.sendmail("noreply@example.com", to, body)  # BLOCKING (network I/O)
        self.server.quit()                                      # BLOCKING (network I/O)


class CommandRunner:
    '''Holds subprocess as self.proc.'''
    
    def run(self, cmd: str):
        self.proc = subprocess.Popen(                          # BLOCKING (process spawn)
            cmd, shell=True, stdout=subprocess.PIPE
        )
        self.stdout, self.stderr = self.proc.communicate()     # BLOCKING (wait for process)


repo = UserRepository("data.db")
notifier = NotificationService()
runner = CommandRunner()


@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = repo.get_user(user_id)
    return {"user": user}


@app.get("/users")
async def list_users():
    users = repo.list_users()
    return {"users": users}


@app.post("/users")
async def create_user(name: str):
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
print("INSTANCE ATTRIBUTE BLIND SPOT TEST")
print("Pattern: self.conn = sqlite3.connect(...); self.conn.execute(...)")
print("=" * 70)
print(f"\nTotal violations: {len(result.violations)}")
print()
for v in result.violations:
    print(f"  Line {v.line}: {v.message}")

print("\n" + "-" * 70)
print("ANALYSIS:")
print("-" * 70)

# Count expected vs found blocking calls
blocking_in_code = [
    (8, "sqlite3.connect()"),
    (14, "self.conn.execute() in get_user"),
    (15, ".fetchone()"),
    (19, "self.conn.execute() in list_users"),
    (20, ".fetchall()"),
    (24, "self.conn.execute() in create_user"),
    (27, "self.conn.commit()"),
    (31, "self.conn.close()"),
    (38, "self.server.login()"),
    (41, "self.server.sendmail()"),
    (42, "self.server.quit()"),
    (49, "subprocess.Popen()"),
    (51, "self.proc.communicate()"),
]

caught_lines = {v.line for v in result.violations}
missed = [(line, desc) for line, desc in blocking_in_code if line not in caught_lines]
caught = [(line, desc) for line, desc in blocking_in_code if line in caught_lines]

print(f"\nExpected blocking calls: {len(blocking_in_code)}")
print(f"Caught: {len(caught)}")
print(f"Missed: {len(missed)} ({100*len(missed)/len(blocking_in_code):.0f}%)")

print("\nCaught:")
for line, desc in caught:
    print(f"  ✓ Line {line}: {desc}")

print("\nMISSED (self.xxx blind spot):")
for line, desc in missed:
    print(f"  ✗ Line {line}: {desc}")

print("\n" + "=" * 70)
print("ROOT CAUSE:")
print("  _track_assignment_type() only handles isinstance(target, ast.Name)")
print("  For 'self.conn = sqlite3.connect(...)', target is ast.Attribute, not ast.Name")
print("  → self.conn never gets type/module info")
print("  → self.conn.execute() resolves to (None, 'execute', UNKNOWN) → UNKNOWN")
print("=" * 70)
