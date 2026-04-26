"""Registry of known blocking patterns that should not be called from async code."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BlockingPattern:
    """A single blocking function/method pattern to detect."""
    module: str  # e.g. "time", "requests", "os"
    name: str  # e.g. "sleep", "get", "system"
    kind: str  # "function" or "method"
    severity: str = "error"  # "error" or "warning"
    message: str = ""
    # For methods: the object type pattern (e.g. "Path" for Path.read_text)
    object_type: str = ""

    @property
    def full_name(self) -> str:
        if self.object_type:
            return f"{self.module}.{self.object_type}.{self.name}"
        return f"{self.module}.{self.name}"


# Known blocking functions/methods
BLOCKING_PATTERNS: list[BlockingPattern] = [
    # time module
    BlockingPattern("time", "sleep", "function", "error", "time.sleep() blocks the event loop"),
    BlockingPattern("time", "time", "function", "warning", "time.time() is OK but consider monotonic"),
    # requests module (all HTTP methods)
    BlockingPattern("requests", "get", "function", "error", "requests.get() is synchronous; use httpx.AsyncClient"),
    BlockingPattern("requests", "post", "function", "error", "requests.post() is synchronous; use httpx.AsyncClient"),
    BlockingPattern("requests", "put", "function", "error", "requests.put() is synchronous; use httpx.AsyncClient"),
    BlockingPattern("requests", "delete", "function", "error", "requests.delete() is synchronous; use httpx.AsyncClient"),
    BlockingPattern("requests", "patch", "function", "error", "requests.patch() is synchronous; use httpx.AsyncClient"),
    BlockingPattern("requests", "head", "function", "error", "requests.head() is synchronous; use httpx.AsyncClient"),
    BlockingPattern("requests", "options", "function", "error", "requests.options() is synchronous; use httpx.AsyncClient"),
    BlockingPattern("requests", "request", "function", "error", "requests.request() is synchronous; use httpx.AsyncClient"),
    BlockingPattern("requests", "Session", "function", "error", "requests.Session is synchronous; use httpx.AsyncClient"),
    # subprocess module
    BlockingPattern("subprocess", "run", "function", "error", "subprocess.run() blocks; use asyncio.create_subprocess_exec"),
    BlockingPattern("subprocess", "call", "function", "error", "subprocess.call() blocks; use asyncio.create_subprocess_exec"),
    BlockingPattern("subprocess", "check_output", "function", "error", "subprocess.check_output() blocks; use asyncio.create_subprocess_exec"),
    BlockingPattern("subprocess", "check_call", "function", "error", "subprocess.check_call() blocks; use asyncio.create_subprocess_exec"),
    BlockingPattern("subprocess", "Popen", "function", "error", "subprocess.Popen blocks; use asyncio.create_subprocess_exec"),
    # os module
    BlockingPattern("os", "system", "function", "error", "os.system() blocks; use asyncio.create_subprocess_exec"),
    BlockingPattern("os", "popen", "function", "error", "os.popen() blocks"),
    # http.client
    BlockingPattern("http.client", "HTTPConnection", "function", "error", "http.client is synchronous; use httpx.AsyncClient"),
    BlockingPattern("http.client", "HTTPSConnection", "function", "error", "http.client is synchronous; use httpx.AsyncClient"),
    # urllib.request
    BlockingPattern("urllib.request", "urlopen", "function", "error", "urllib.request.urlopen is synchronous; use httpx.AsyncClient"),
    BlockingPattern("urllib.request", "Request", "function", "error", "urllib.request.Request is synchronous"),
    # socket module (raw socket ops)
    BlockingPattern("socket", "socket", "function", "error", "socket.socket is synchronous; use asyncio.open_connection"),
    # file I/O via builtins.open (detected separately)
    BlockingPattern("builtins", "open", "function", "error", "open() for file I/O blocks; use aiofiles or run in executor"),
    # pathlib blocking methods
    BlockingPattern("pathlib", "read_text", "method", "error", "Path.read_text() blocks; use aiofiles", "Path"),
    BlockingPattern("pathlib", "write_text", "method", "error", "Path.write_text() blocks; use aiofiles", "Path"),
    BlockingPattern("pathlib", "read_bytes", "method", "error", "Path.read_bytes() blocks; use aiofiles", "Path"),
    BlockingPattern("pathlib", "write_bytes", "method", "error", "Path.write_bytes() blocks; use aiofiles", "Path"),
    BlockingPattern("pathlib", "exists", "method", "warning", "Path.exists() does I/O; consider aiofiles.os.path.exists", "Path"),
    BlockingPattern("pathlib", "is_file", "method", "warning", "Path.is_file() does I/O", "Path"),
    BlockingPattern("pathlib", "is_dir", "method", "warning", "Path.is_dir() does I/O", "Path"),
    BlockingPattern("pathlib", "mkdir", "method", "error", "Path.mkdir() blocks; use aiofiles.os.mkdir", "Path"),
    BlockingPattern("pathlib", "rmdir", "method", "error", "Path.rmdir() blocks", "Path"),
    BlockingPattern("pathlib", "unlink", "method", "error", "Path.unlink() blocks", "Path"),
    BlockingPattern("pathlib", "rename", "method", "error", "Path.rename() blocks", "Path"),
    BlockingPattern("pathlib", "glob", "method", "error", "Path.glob() blocks", "Path"),
    BlockingPattern("pathlib", "rglob", "method", "error", "Path.rglob() blocks", "Path"),
    BlockingPattern("pathlib", "iterdir", "method", "error", "Path.iterdir() blocks", "Path"),
    # json load/dump with file (detected when first arg is file)
    BlockingPattern("json", "load", "function", "error", "json.load() with file I/O blocks; use aiofiles + json.loads"),
    BlockingPattern("json", "dump", "function", "error", "json.dump() with file I/O blocks; use aiofiles + json.dumps"),
    # cpu-intensive pure-python functions (warning level)
    BlockingPattern("hashlib", "pbkdf2_hmac", "function", "warning", "CPU-intensive hash; consider running in executor"),
    BlockingPattern("bcrypt", "hashpw", "function", "warning", "CPU-intensive bcrypt; consider running in executor"),
    BlockingPattern("bcrypt", "gensalt", "function", "warning", "CPU-intensive bcrypt; consider running in executor"),
    BlockingPattern("bcrypt", "checkpw", "function", "warning", "CPU-intensive bcrypt; consider running in executor"),
    # pickle
    BlockingPattern("pickle", "loads", "function", "warning", "pickle.loads can be CPU-intensive for large objects"),
    BlockingPattern("pickle", "dumps", "function", "warning", "pickle.dumps can be CPU-intensive for large objects"),
    BlockingPattern("pickle", "load", "function", "error", "pickle.load does file I/O and can be CPU-intensive"),
    BlockingPattern("pickle", "dump", "function", "error", "pickle.dump does file I/O and can be CPU-intensive"),
    # xml parsing
    BlockingPattern("xml.etree.ElementTree", "parse", "function", "error", "XML parsing reads from file; use async"),
    BlockingPattern("xml.etree.ElementTree", "fromstring", "function", "warning", "XML parsing can be CPU-intensive for large docs"),
    # psycopg2 (sync postgres)
    BlockingPattern("psycopg2", "connect", "function", "error", "psycopg2 is synchronous; use asyncpg or psycopg3 async"),
    # pymysql / mysql
    BlockingPattern("pymysql", "connect", "function", "error", "pymysql is synchronous; use aiomysql"),
    # redis sync
    BlockingPattern("redis", "Redis", "function", "error", "redis.Redis is synchronous; use redis.asyncio.Redis"),
    BlockingPattern("redis", "StrictRedis", "function", "error", "redis.StrictRedis is synchronous"),
    # queue module (blocking)
    BlockingPattern("queue", "Queue", "function", "error", "queue.Queue is synchronous; use asyncio.Queue"),
    BlockingPattern("queue", "get", "method", "error", "queue.get() blocks; use asyncio.Queue.get", "Queue"),
    BlockingPattern("queue", "put", "method", "error", "queue.put() blocks; use asyncio.Queue.put", "Queue"),
    # threading blocking primitives
    BlockingPattern("threading", "Lock", "function", "error", "threading.Lock blocks; use asyncio.Lock"),
    BlockingPattern("threading", "acquire", "method", "error", "Lock.acquire() blocks the event loop; use asyncio.Lock.acquire", "Lock"),
    BlockingPattern("threading", "RLock", "function", "error", "threading.RLock blocks; use asyncio.Lock"),
    BlockingPattern("threading", "Event", "function", "error", "threading.Event blocks; use asyncio.Event"),
    BlockingPattern("threading", "Condition", "function", "error", "threading.Condition blocks; use asyncio.Condition"),
    BlockingPattern("threading", "Semaphore", "function", "error", "threading.Semaphore blocks; use asyncio.Semaphore"),
    BlockingPattern("threading", "Barrier", "function", "error", "threading.Barrier blocks"),
    BlockingPattern("threading", "join", "method", "error", "Thread.join() blocks; use asyncio tasks", "Thread"),
]

# Build lookup dicts for fast matching
_BLOCKING_BY_NAME: dict[str, list[BlockingPattern]] = {}
for bp in BLOCKING_PATTERNS:
    _BLOCKING_BY_NAME.setdefault(bp.name, []).append(bp)


def find_blocking_by_name(name: str) -> list[BlockingPattern]:
    """Find all blocking patterns matching a function/method name."""
    return _BLOCKING_BY_NAME.get(name, [])


def is_blocking_call(
    func_name: str,
    module: str | None = None,
    object_type: str | None = None,
) -> BlockingPattern | None:
    """Check if a call matches a known blocking pattern.

    Returns the matching BlockingPattern or None.
    """
    candidates = _BLOCKING_BY_NAME.get(func_name)
    if not candidates:
        return None

    for bp in candidates:
        if object_type and bp.object_type:
            # Method call: match on object type
            if bp.object_type == object_type:
                return bp
        elif not object_type and not bp.object_type:
            # Function call: match on module if we have it
            if module is None or bp.module == module:
                return bp
        elif object_type and not bp.object_type:
            # Call has object_type but pattern doesn't (e.g., requests.Session().get() vs requests.get)
            # If module matches, still consider it a match since the method is on
            # an object from a known blocking module
            if module is not None and bp.module == module:
                return bp

    return None
