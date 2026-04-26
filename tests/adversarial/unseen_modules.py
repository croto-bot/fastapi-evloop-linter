"""Adversarial test cases using modules NOT in BLOCKING_PATTERNS.

These cases are designed to validate that the generic rewrite of the linter
generalises beyond its hardcoded list. Every module used here is absent from
blockers.py at the time these cases were authored. A correct generic detector
must flag all positive cases and must NOT flag the negative cases.

Categories
----------
- direct_call      : unseen module, single blocking call at depth 0
- aliased_import   : module imported under an alias
- deep_chain       : blocking call buried N levels below the endpoint
- method_on_instance : method called on a variable (instance-tracking required)
- negative         : legitimate patterns that must NOT be flagged
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass


@dataclass
class TestCase:
    """A single adversarial test case (mirrors generator.TestCase)."""
    name: str
    source: str
    difficulty: int      # 1-10
    expected_violations: int
    expected_min_depth: int
    category: str
    description: str = ""


# ---------------------------------------------------------------------------
# 1. DIRECT CALL (one blocking call, depth 0)
# ---------------------------------------------------------------------------

def _direct_call_cases() -> list[TestCase]:
    return [
        TestCase(
            name="paramiko_exec_command_direct",
            source=textwrap.dedent("""\
                import paramiko
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/ssh")
                async def run_ssh_command(host: str, cmd: str):
                    client = paramiko.SSHClient()
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    client.connect(host)
                    stdin, stdout, stderr = client.exec_command(cmd)
                    return {"output": stdout.read().decode()}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="paramiko.SSHClient.exec_command() is synchronous I/O",
        ),
        TestCase(
            name="boto3_s3_get_object_direct",
            source=textwrap.dedent("""\
                import boto3
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/s3-object")
                async def get_s3_object(bucket: str, key: str):
                    s3 = boto3.client("s3")
                    obj = s3.get_object(Bucket=bucket, Key=key)
                    return {"body": obj["Body"].read().decode()}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="boto3 S3 get_object() is synchronous I/O",
        ),
        TestCase(
            name="boto3_s3_put_object_direct",
            source=textwrap.dedent("""\
                import boto3
                from fastapi import FastAPI
                app = FastAPI()

                @app.put("/s3-object")
                async def put_s3_object(bucket: str, key: str, body: str):
                    s3 = boto3.client("s3")
                    s3.put_object(Bucket=bucket, Key=key, Body=body.encode())
                    return {"status": "uploaded"}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="boto3 S3 put_object() is synchronous I/O",
        ),
        TestCase(
            name="sqlite3_execute_direct",
            source=textwrap.dedent("""\
                import sqlite3
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/users")
                async def list_users():
                    conn = sqlite3.connect("/tmp/app.db")
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM users")
                    rows = cur.fetchall()
                    conn.close()
                    return {"users": rows}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="sqlite3.connect() + cursor.execute() are synchronous DB I/O",
        ),
        TestCase(
            name="smtplib_sendmail_direct",
            source=textwrap.dedent("""\
                import smtplib
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/send-email")
                async def send_email(to: str, subject: str, body: str):
                    server = smtplib.SMTP("smtp.example.com", 587)
                    server.starttls()
                    server.login("user@example.com", "secret")
                    server.sendmail("user@example.com", to, f"Subject: {subject}\\n\\n{body}")
                    server.quit()
                    return {"status": "sent"}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="smtplib.SMTP.sendmail() blocks on network I/O",
        ),
        TestCase(
            name="ftplib_retrbinary_direct",
            source=textwrap.dedent("""\
                import ftplib
                import io
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/ftp-file")
                async def download_ftp_file(host: str, path: str):
                    buf = io.BytesIO()
                    ftp = ftplib.FTP(host)
                    ftp.login()
                    ftp.retrbinary(f"RETR {path}", buf.write)
                    ftp.quit()
                    return {"size": buf.tell()}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="ftplib.FTP.retrbinary() blocks on network I/O",
        ),
        TestCase(
            name="lxml_etree_parse_direct",
            source=textwrap.dedent("""\
                from lxml import etree
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/xml")
                async def parse_xml_file(path: str):
                    tree = etree.parse(path)
                    root = tree.getroot()
                    return {"tag": root.tag}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="lxml.etree.parse() reads a file synchronously",
        ),
        TestCase(
            name="pil_image_open_direct",
            source=textwrap.dedent("""\
                from PIL import Image
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/thumbnail")
                async def create_thumbnail(path: str):
                    img = Image.open(path)
                    img.thumbnail((128, 128))
                    img.save(path + ".thumb.jpg")
                    return {"status": "done"}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="PIL.Image.open() reads a file synchronously",
        ),
        TestCase(
            name="pandas_read_csv_direct",
            source=textwrap.dedent("""\
                import pandas as pd
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/csv-stats")
                async def csv_stats(path: str):
                    df = pd.read_csv(path)
                    return {"rows": len(df), "cols": len(df.columns)}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="pandas.read_csv() blocks on file I/O",
        ),
        TestCase(
            name="pandas_read_sql_direct",
            source=textwrap.dedent("""\
                import sqlite3
                import pandas as pd
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/query")
                async def query_db(sql: str):
                    conn = sqlite3.connect("/tmp/app.db")
                    df = pd.read_sql(sql, conn)
                    conn.close()
                    return {"rows": len(df)}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="pandas.read_sql() blocks on DB I/O",
        ),
        TestCase(
            name="numpy_load_direct",
            source=textwrap.dedent("""\
                import numpy as np
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/array")
                async def load_array(path: str):
                    arr = np.load(path)
                    return {"shape": list(arr.shape)}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="numpy.load() blocks on file I/O",
        ),
        TestCase(
            name="tarfile_extractall_direct",
            source=textwrap.dedent("""\
                import tarfile
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/extract-tar")
                async def extract_tar(archive: str, dest: str):
                    with tarfile.open(archive) as tf:
                        tf.extractall(dest)
                    return {"status": "extracted"}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="tarfile.TarFile.extractall() blocks on file I/O",
        ),
        TestCase(
            name="zipfile_extractall_direct",
            source=textwrap.dedent("""\
                import zipfile
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/extract-zip")
                async def extract_zip(archive: str, dest: str):
                    with zipfile.ZipFile(archive, "r") as zf:
                        zf.extractall(dest)
                    return {"status": "extracted"}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="zipfile.ZipFile.extractall() blocks on file I/O",
        ),
        TestCase(
            name="gzip_open_direct",
            source=textwrap.dedent("""\
                import gzip
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/gz-content")
                async def read_gzip(path: str):
                    with gzip.open(path, "rt") as f:
                        return {"content": f.read()}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="gzip.open() blocks on file I/O",
        ),
        TestCase(
            name="pyodbc_execute_direct",
            source=textwrap.dedent("""\
                import pyodbc
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/odbc-query")
                async def odbc_query(dsn: str, sql: str):
                    conn = pyodbc.connect(dsn)
                    cur = conn.cursor()
                    cur.execute(sql)
                    rows = cur.fetchall()
                    conn.close()
                    return {"rows": [list(r) for r in rows]}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="pyodbc.connect() + cursor.execute() are synchronous DB I/O",
        ),
        TestCase(
            name="kafka_producer_send_direct",
            source=textwrap.dedent("""\
                from kafka import KafkaProducer
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/publish")
                async def publish_event(topic: str, message: str):
                    producer = KafkaProducer(bootstrap_servers=["localhost:9092"])
                    future = producer.send(topic, message.encode())
                    future.get(timeout=10)
                    producer.close()
                    return {"status": "published"}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="KafkaProducer.send().get() blocks waiting for broker ack",
        ),
        TestCase(
            name="pymongo_find_direct",
            source=textwrap.dedent("""\
                from pymongo import MongoClient
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/documents")
                async def list_documents(db: str, collection: str):
                    client = MongoClient("mongodb://localhost:27017")
                    col = client[db][collection]
                    docs = list(col.find({}))
                    client.close()
                    return {"count": len(docs)}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="pymongo.MongoClient collection.find() is synchronous",
        ),
        TestCase(
            name="concurrent_future_result_direct",
            source=textwrap.dedent("""\
                from concurrent.futures import ThreadPoolExecutor, Future
                from fastapi import FastAPI
                app = FastAPI()

                executor = ThreadPoolExecutor(max_workers=4)

                @app.get("/compute")
                async def compute(value: int):
                    future: Future = executor.submit(lambda v: v * 2, value)
                    result = future.result()  # blocks the event loop!
                    return {"result": result}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="concurrent.futures.Future.result() blocks the event loop",
        ),
        TestCase(
            name="tempfile_mkstemp_direct",
            source=textwrap.dedent("""\
                import tempfile
                import os
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/upload-temp")
                async def upload_to_temp(content: str):
                    fd, path = tempfile.mkstemp(suffix=".txt")
                    try:
                        os.write(fd, content.encode())
                    finally:
                        os.close(fd)
                    return {"path": path}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="tempfile.mkstemp() performs synchronous file creation",
        ),
        TestCase(
            name="tempfile_named_tempfile_direct",
            source=textwrap.dedent("""\
                import tempfile
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/temp-write")
                async def temp_write(content: str):
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                        f.write(content)
                        return {"path": f.name}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="tempfile.NamedTemporaryFile() opens a file synchronously",
        ),
        TestCase(
            name="socket_recv_direct",
            source=textwrap.dedent("""\
                import socket
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/raw-socket")
                async def raw_recv(host: str, port: int):
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((host, port))
                    data = sock.recv(4096)
                    sock.close()
                    return {"data": data.decode()}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="socket().recv() and connect() block the event loop",
        ),
        TestCase(
            name="ssl_wrap_socket_direct",
            source=textwrap.dedent("""\
                import ssl
                import socket
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/tls-connect")
                async def tls_connect(host: str, port: int):
                    ctx = ssl.create_default_context()
                    raw = socket.socket()
                    raw.connect((host, port))
                    tls_sock = ctx.wrap_socket(raw, server_hostname=host)
                    tls_sock.send(b"GET / HTTP/1.0\\r\\n\\r\\n")
                    resp = tls_sock.recv(4096)
                    return {"bytes": len(resp)}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="ssl.SSLContext.wrap_socket() performs synchronous TLS handshake",
        ),
        TestCase(
            name="selectors_select_direct",
            source=textwrap.dedent("""\
                import selectors
                import socket
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/select-io")
                async def select_io(host: str, port: int):
                    sel = selectors.DefaultSelector()
                    sock = socket.socket()
                    sock.connect((host, port))
                    sel.register(sock, selectors.EVENT_READ)
                    events = sel.select(timeout=5.0)  # blocks!
                    sel.close()
                    sock.close()
                    return {"events": len(events)}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="selectors.DefaultSelector.select() blocks the event loop",
        ),
        TestCase(
            name="mmap_direct",
            source=textwrap.dedent("""\
                import mmap
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/mmap-read")
                async def mmap_read(path: str):
                    with open(path, "rb") as f:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        data = mm[:1024]
                        mm.close()
                    return {"preview": data.decode(errors="replace")}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="mmap.mmap() maps a file synchronously (also open() present)",
        ),
        TestCase(
            name="input_builtin_direct",
            source=textwrap.dedent("""\
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/interactive")
                async def interactive_endpoint():
                    answer = input("Enter something: ")  # waits for stdin — blocks event loop
                    return {"answer": answer}
            """),
            difficulty=2,
            expected_violations=1,
            expected_min_depth=0,
            category="direct_call",
            description="builtin input() blocks waiting for stdin indefinitely",
        ),
    ]


# ---------------------------------------------------------------------------
# 2. ALIASED IMPORT (~5 cases)
# ---------------------------------------------------------------------------

def _aliased_import_cases() -> list[TestCase]:
    return [
        TestCase(
            name="paramiko_aliased",
            source=textwrap.dedent("""\
                import paramiko as pk
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/ssh-alias")
                async def ssh_alias(host: str, cmd: str):
                    client = pk.SSHClient()
                    client.set_missing_host_key_policy(pk.AutoAddPolicy())
                    client.connect(host)
                    _, stdout, _ = client.exec_command(cmd)
                    return {"output": stdout.read().decode()}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="aliased_import",
            description="paramiko imported as pk; pk.SSHClient().exec_command() called",
        ),
        TestCase(
            name="boto3_aliased",
            source=textwrap.dedent("""\
                import boto3 as aws
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/aws-s3")
                async def fetch_from_s3(bucket: str, key: str):
                    s3 = aws.client("s3")
                    obj = s3.get_object(Bucket=bucket, Key=key)
                    return {"size": len(obj["Body"].read())}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="aliased_import",
            description="boto3 imported as aws; aws.client().get_object() called",
        ),
        TestCase(
            name="pandas_aliased_to_csv",
            source=textwrap.dedent("""\
                import pandas as pd
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/export-csv")
                async def export_csv(path: str):
                    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
                    df = pd.DataFrame(data)
                    df.to_csv(path, index=False)
                    return {"status": "exported"}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="aliased_import",
            description="pandas imported as pd (standard alias); df.to_csv() blocks on I/O",
        ),
        TestCase(
            name="pymongo_aliased",
            source=textwrap.dedent("""\
                import pymongo as mg
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/mongo-alias")
                async def mongo_alias(db_name: str, coll_name: str):
                    client = mg.MongoClient("mongodb://localhost:27017")
                    docs = list(client[db_name][coll_name].find({}))
                    return {"count": len(docs)}
            """),
            difficulty=3,
            expected_violations=1,
            expected_min_depth=0,
            category="aliased_import",
            description="pymongo imported as mg; mg.MongoClient().find() called",
        ),
        TestCase(
            name="smtplib_from_import_aliased",
            source=textwrap.dedent("""\
                from smtplib import SMTP as MailServer
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/send-alias")
                async def send_alias(to: str, msg: str):
                    server = MailServer("smtp.example.com", 587)
                    server.starttls()
                    server.sendmail("bot@example.com", to, msg)
                    server.quit()
                    return {"status": "sent"}
            """),
            difficulty=4,
            expected_violations=1,
            expected_min_depth=0,
            category="aliased_import",
            description="from smtplib import SMTP as MailServer; MailServer().sendmail() called",
        ),
    ]


# ---------------------------------------------------------------------------
# 3. DEEP CHAIN (~5 cases — blocking buried 2-4 levels below endpoint)
# ---------------------------------------------------------------------------

def _deep_chain_cases() -> list[TestCase]:
    return [
        TestCase(
            name="sqlite3_deep_chain_2",
            source=textwrap.dedent("""\
                import sqlite3
                from fastapi import FastAPI
                app = FastAPI()

                def query_db(sql: str):
                    conn = sqlite3.connect("/tmp/app.db")
                    cur = conn.cursor()
                    cur.execute(sql)
                    rows = cur.fetchall()
                    conn.close()
                    return rows

                @app.get("/users")
                async def list_users():
                    rows = query_db("SELECT * FROM users")
                    return {"users": rows}
            """),
            difficulty=4,
            expected_violations=1,
            expected_min_depth=1,
            category="deep_chain",
            description="sqlite3 at depth 2: endpoint -> query_db -> sqlite3.connect/execute",
        ),
        TestCase(
            name="boto3_deep_chain_3",
            source=textwrap.dedent("""\
                import boto3
                from fastapi import FastAPI
                app = FastAPI()

                def download_object(bucket: str, key: str) -> bytes:
                    s3 = boto3.client("s3")
                    obj = s3.get_object(Bucket=bucket, Key=key)
                    return obj["Body"].read()

                def load_config(env: str) -> dict:
                    raw = download_object("my-configs", f"{env}.json")
                    import json
                    return json.loads(raw)

                @app.get("/config")
                async def get_config(env: str):
                    cfg = load_config(env)
                    return cfg
            """),
            difficulty=5,
            expected_violations=1,
            expected_min_depth=2,
            category="deep_chain",
            description="boto3 at depth 3: endpoint -> load_config -> download_object -> s3.get_object",
        ),
        TestCase(
            name="paramiko_deep_chain_3",
            source=textwrap.dedent("""\
                import paramiko
                from fastapi import FastAPI
                app = FastAPI()

                def exec_remote(client: paramiko.SSHClient, cmd: str) -> str:
                    _, stdout, _ = client.exec_command(cmd)
                    return stdout.read().decode()

                def run_deploy(host: str, cmd: str) -> str:
                    client = paramiko.SSHClient()
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    client.connect(host)
                    result = exec_remote(client, cmd)
                    client.close()
                    return result

                @app.post("/deploy")
                async def deploy(host: str, script: str):
                    output = run_deploy(host, script)
                    return {"output": output}
            """),
            difficulty=5,
            expected_violations=1,
            expected_min_depth=2,
            category="deep_chain",
            description="paramiko at depth 3: endpoint -> run_deploy -> exec_remote -> exec_command",
        ),
        TestCase(
            name="kafka_deep_chain_4",
            source=textwrap.dedent("""\
                from kafka import KafkaProducer
                from fastapi import FastAPI
                app = FastAPI()

                def _flush_producer(producer: KafkaProducer) -> None:
                    producer.flush()

                def _send_and_wait(producer: KafkaProducer, topic: str, value: bytes) -> None:
                    future = producer.send(topic, value)
                    future.get(timeout=10)
                    _flush_producer(producer)

                def publish_event(topic: str, payload: str) -> None:
                    producer = KafkaProducer(bootstrap_servers=["localhost:9092"])
                    _send_and_wait(producer, topic, payload.encode())
                    producer.close()

                @app.post("/event")
                async def emit_event(topic: str, payload: str):
                    publish_event(topic, payload)
                    return {"status": "emitted"}
            """),
            difficulty=6,
            expected_violations=1,
            expected_min_depth=3,
            category="deep_chain",
            description="kafka future.get() at depth 4 through 3 helper layers",
        ),
        TestCase(
            name="pymongo_deep_chain_4",
            source=textwrap.dedent("""\
                from pymongo import MongoClient
                from fastapi import FastAPI
                app = FastAPI()

                def _find_docs(collection, query: dict) -> list:
                    return list(collection.find(query))

                def _get_collection(client: MongoClient, db: str, coll: str):
                    return client[db][coll]

                def fetch_records(db: str, coll: str, query: dict) -> list:
                    client = MongoClient("mongodb://localhost:27017")
                    col = _get_collection(client, db, coll)
                    docs = _find_docs(col, query)
                    client.close()
                    return docs

                @app.get("/records")
                async def get_records(db: str, coll: str):
                    records = fetch_records(db, coll, {})
                    return {"count": len(records)}
            """),
            difficulty=6,
            expected_violations=1,
            expected_min_depth=3,
            category="deep_chain",
            description="pymongo collection.find() at depth 4 through 3 helper layers",
        ),
    ]


# ---------------------------------------------------------------------------
# 4. METHOD ON INSTANCE (~5 cases)
# ---------------------------------------------------------------------------

def _method_on_instance_cases() -> list[TestCase]:
    return [
        TestCase(
            name="boto3_instance_get_object",
            source=textwrap.dedent("""\
                import boto3
                from fastapi import FastAPI
                app = FastAPI()

                s3_client = boto3.client("s3")

                @app.get("/s3-instance")
                async def s3_instance(bucket: str, key: str):
                    response = s3_client.get_object(Bucket=bucket, Key=key)
                    data = response["Body"].read()
                    return {"size": len(data)}
            """),
            difficulty=4,
            expected_violations=1,
            expected_min_depth=0,
            category="method_on_instance",
            description="Module-level boto3 client instance; .get_object() called in endpoint",
        ),
        TestCase(
            name="pymongo_instance_find",
            source=textwrap.dedent("""\
                from pymongo import MongoClient
                from fastapi import FastAPI
                app = FastAPI()

                mongo_client = MongoClient("mongodb://localhost:27017")
                db = mongo_client["mydb"]
                users_col = db["users"]

                @app.get("/users-instance")
                async def users_instance(name: str):
                    docs = list(users_col.find({"name": name}))
                    return {"results": len(docs)}
            """),
            difficulty=4,
            expected_violations=1,
            expected_min_depth=0,
            category="method_on_instance",
            description="Module-level MongoClient collection; .find() called in endpoint",
        ),
        TestCase(
            name="paramiko_client_instance_exec",
            source=textwrap.dedent("""\
                import paramiko
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/ssh-instance")
                async def ssh_instance(host: str, cmd: str):
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(host, username="deploy")
                    _, out, err = ssh.exec_command(cmd)
                    exit_code = out.channel.recv_exit_status()
                    ssh.close()
                    return {"exit_code": exit_code}
            """),
            difficulty=4,
            expected_violations=1,
            expected_min_depth=0,
            category="method_on_instance",
            description="paramiko SSHClient local variable; .exec_command() + channel.recv_exit_status() both block",
        ),
        TestCase(
            name="pandas_df_to_csv_instance",
            source=textwrap.dedent("""\
                import pandas as pd
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/df-export")
                async def df_export(path: str):
                    df = pd.read_csv("/tmp/input.csv")
                    df["processed"] = df["value"] * 2
                    df.to_csv(path, index=False)
                    return {"rows": len(df)}
            """),
            difficulty=3,
            expected_violations=2,
            expected_min_depth=0,
            category="method_on_instance",
            description="pandas read_csv() + DataFrame.to_csv() both block — two violations",
        ),
        TestCase(
            name="kafka_consumer_instance",
            source=textwrap.dedent("""\
                from kafka import KafkaConsumer
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/consume-one")
                async def consume_one(topic: str):
                    consumer = KafkaConsumer(
                        topic,
                        bootstrap_servers=["localhost:9092"],
                        consumer_timeout_ms=5000,
                    )
                    messages = []
                    for msg in consumer:  # iteration is synchronous and blocking
                        messages.append(msg.value.decode())
                        if len(messages) >= 1:
                            break
                    consumer.close()
                    return {"messages": messages}
            """),
            difficulty=4,
            expected_violations=1,
            expected_min_depth=0,
            category="method_on_instance",
            description="KafkaConsumer iteration blocks the event loop",
        ),
    ]


# ---------------------------------------------------------------------------
# 5. NEGATIVE CASES (must NOT be flagged)
# ---------------------------------------------------------------------------

def _negative_cases() -> list[TestCase]:
    return [
        TestCase(
            name="negative_math_sqrt",
            source=textwrap.dedent("""\
                import math
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/sqrt")
                async def compute_sqrt(value: float):
                    result = math.sqrt(value)
                    return {"result": result}
            """),
            difficulty=1,
            expected_violations=0,
            expected_min_depth=0,
            category="negative",
            description="math.sqrt() is pure computation — must NOT be flagged",
        ),
        TestCase(
            name="negative_json_loads",
            source=textwrap.dedent("""\
                import json
                from fastapi import FastAPI
                app = FastAPI()

                @app.post("/parse-json")
                async def parse_json(raw: str):
                    data = json.loads(raw)
                    return {"parsed": data}
            """),
            difficulty=1,
            expected_violations=0,
            expected_min_depth=0,
            category="negative",
            description="json.loads() is pure CPU — must NOT be flagged",
        ),
        TestCase(
            name="negative_await_async_func",
            source=textwrap.dedent("""\
                import asyncio
                from fastapi import FastAPI
                app = FastAPI()

                async def fetch_data_async(url: str) -> dict:
                    await asyncio.sleep(0)
                    return {"url": url, "data": "mock"}

                @app.get("/async-fetch")
                async def async_fetch(url: str):
                    result = await fetch_data_async(url)
                    return result
            """),
            difficulty=1,
            expected_violations=0,
            expected_min_depth=0,
            category="negative",
            description="Awaited async helper — must NOT be flagged",
        ),
        TestCase(
            name="negative_list_comprehension",
            source=textwrap.dedent("""\
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/squares")
                async def squares(n: int):
                    result = [x * x for x in range(n)]
                    return {"squares": result}
            """),
            difficulty=1,
            expected_violations=0,
            expected_min_depth=0,
            category="negative",
            description="Pure list comprehension — must NOT be flagged",
        ),
        TestCase(
            name="negative_asyncio_sleep",
            source=textwrap.dedent("""\
                import asyncio
                from fastapi import FastAPI
                app = FastAPI()

                @app.get("/delay")
                async def delay_response(seconds: float):
                    await asyncio.sleep(seconds)
                    return {"status": "ok"}
            """),
            difficulty=1,
            expected_violations=0,
            expected_min_depth=0,
            category="negative",
            description="await asyncio.sleep() is the correct async sleep — must NOT be flagged",
        ),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_unseen_cases() -> list[TestCase]:
    """Return all adversarial test cases that use modules absent from BLOCKING_PATTERNS."""
    cases: list[TestCase] = []
    cases.extend(_direct_call_cases())
    cases.extend(_aliased_import_cases())
    cases.extend(_deep_chain_cases())
    cases.extend(_method_on_instance_cases())
    cases.extend(_negative_cases())
    return cases


if __name__ == "__main__":
    all_cases = generate_unseen_cases()
    positive = [c for c in all_cases if c.expected_violations > 0]
    negative = [c for c in all_cases if c.expected_violations == 0]
    print(f"Total cases: {len(all_cases)}")
    print(f"  Positive (must flag): {len(positive)}")
    print(f"  Negative (must NOT flag): {len(negative)}")
    by_category: dict[str, int] = {}
    for c in all_cases:
        by_category[c.category] = by_category.get(c.category, 0) + 1
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")
