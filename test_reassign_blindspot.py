"""Test: Variable reassignment and method return type inference blind spot.

The linter fails to propagate type and module information through two common patterns:

1. Variable reassignment: `conn = server` loses all type info from `server`
2. Method return values: `cursor = conn.cursor()` stores the method name as
   the type instead of inferring the return type, and doesn't propagate the
   module from `conn` to `cursor`

Both patterns cause all subsequent method calls on the affected variable to
be classified as UNKNOWN instead of BLOCKING.
"""

import sqlite3
import smtplib
import xml.etree.ElementTree as ET
import imaplib
from fastapi import FastAPI

app = FastAPI()


# ============================================================
# BLIND SPOT 1: Method return values lose module info
# ============================================================

@app.get("/sqlite-cursor")
async def sqlite_cursor_endpoint():
    """The most common real-world case: sqlite3 cursor pattern."""
    conn = sqlite3.connect("data.db")       # CAUGHT ✅
    cursor = conn.cursor()                   # CAUGHT ✅ (call itself)
    cursor.execute("SELECT * FROM users")    # MISSED ❌ — cursor has no module
    rows = cursor.fetchall()                 # MISSED ❌ — cursor has no module
    conn.commit()                            # CAUGHT ✅ — conn still has module
    return {"rows": rows}


@app.get("/xml-tree")
async def xml_tree_endpoint():
    """xml.etree.ElementTree: parse() → getroot() → find() chain."""
    tree = ET.parse("data.xml")              # CAUGHT ✅
    root = tree.getroot()                    # CAUGHT ✅ (call itself)
    elem = root.find("item")                 # MISSED ❌ — root has no module
    return {"tag": elem.tag if elem else None}


@app.get("/imap-search")
async def imap_search_endpoint():
    """IMAP4_SSL: search() returns tuple, but we demonstrate the
    intermediate variable issue on the mail object itself."""
    mail = imaplib.IMAP4_SSL("imap.gmail.com")  # CAUGHT ✅ (constructor)
    mail.login("user", "pass")                   # CAUGHT ✅
    mail.select("INBOX")                         # CAUGHT ✅
    status, data = mail.search(None, "ALL")      # CAUGHT ✅
    # Now data is bytes, but if we used a method on the search result:
    return {"status": status}


# ============================================================
# BLIND SPOT 2: Variable reassignment loses ALL type info
# ============================================================

@app.get("/reassign-smtplib")
async def reassign_smtplib():
    """Simple variable reassignment: conn = server loses type/module info."""
    server = smtplib.SMTP("smtp.example.com", 587)  # type tracked, module="smtplib"
    conn = server                                     # type LOST, module LOST
    conn.login("user", "pass")                        # MISSED ❌
    conn.sendmail("a@b.com", "c@d.com", "Hello")     # MISSED ❌
    return {"sent": True}


@app.get("/reassign-sqlite")
async def reassign_sqlite():
    """Reassignment of a sqlite3 connection."""
    connection = sqlite3.connect("data.db")   # type="connect", module="sqlite3"
    db = connection                            # type LOST, module LOST
    db.execute("SELECT 1")                    # MISSED ❌
    db.commit()                               # MISSED ❌
    return {"ok": True}


# ============================================================
# CONTRAST: These patterns ARE correctly detected
# ============================================================

@app.get("/direct-smtplib")
async def direct_smtplib():
    """Direct usage of tracked variable — works correctly."""
    server = smtplib.SMTP("smtp.example.com", 587)
    server.login("user", "pass")              # CAUGHT ✅
    server.sendmail("a@b.com", "c@d.com", "Hello")  # CAUGHT ✅
    return {"sent": True}


@app.get("/direct-sqlite")
async def direct_sqlite():
    """Direct usage of sqlite3 connection — works correctly."""
    conn = sqlite3.connect("data.db")
    conn.execute("SELECT 1")                  # CAUGHT ✅
    conn.commit()                             # CAUGHT ✅
    return {"ok": True}
