"""Generic blocking-call classifier.

Decides whether a call is SAFE / BLOCKING / ASYNC / UNKNOWN based on
module origin and runtime introspection. No hardcoded blocking-module
enumerations.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from .introspect import (
    ModuleOrigin,
    attr_exists,
    is_async_callable,
    resolve_module_origin,
)

if TYPE_CHECKING:
    from .callgraph import ModuleAnalysis


class Verdict(Enum):
    SAFE = "safe"
    BLOCKING = "blocking"
    ASYNC = "async"
    UNKNOWN = "unknown"


# Computational stdlib modules that never do I/O — or whose I/O is too
# fast / too pervasive to flag in an async endpoint (datetime/logging).
#
# - asyncio: it IS the async framework. `asyncio.create_task`,
#   `asyncio.gather`, `asyncio.Lock` etc. are non-blocking. The only
#   pathological call is `asyncio.run()`, which nobody writes from inside
#   an already-running endpoint.
# - datetime: `datetime.now()` is a microsecond-level syscall in practice;
#   `datetime.UTC` is a constant. Flagging these is pure noise.
# - logging: `logger.info(...)` is the standard pattern in async Python.
#   Yes, default handlers are sync, but the cost is microseconds for the
#   stderr handler that 99% of apps use; flagging every log call would
#   bury real signal.
SAFE_MODULES: frozenset[str] = frozenset({
    "math", "operator", "itertools", "functools", "collections",
    "dataclasses", "typing", "abc", "enum", "decimal", "fractions",
    "statistics", "base64", "secrets", "uuid", "copy", "string",
    "re", "types", "contextlib",
    "asyncio", "datetime", "logging", "traceback", "warnings",
})

# Per-function safe exceptions inside otherwise-unsafe modules.
SAFE_STDLIB_FUNCTIONS: frozenset[tuple[str, str]] = frozenset({
    ("json", "loads"), ("json", "dumps"),
    ("hashlib", "md5"), ("hashlib", "sha1"), ("hashlib", "sha256"),
    ("hashlib", "sha512"), ("hashlib", "sha384"), ("hashlib", "sha224"),
    ("hashlib", "blake2b"), ("hashlib", "blake2s"), ("hashlib", "new"),
})

# Method/property accesses on stdlib classes that are pure (no I/O).
# Format: (module, class, attr).
SAFE_STDLIB_METHODS: frozenset[tuple[str, str, str]] = frozenset({
    # str methods on values returned by known-safe serializers.
    ("json", "dumps", "encode"),

    # pathlib.Path: pure path-string manipulation, no syscalls.
    ("pathlib", "Path", "is_absolute"),
    ("pathlib", "Path", "is_reserved"),
    ("pathlib", "Path", "as_posix"),
    ("pathlib", "Path", "as_uri"),
    ("pathlib", "Path", "match"),
    ("pathlib", "Path", "joinpath"),
    ("pathlib", "Path", "with_name"),
    ("pathlib", "Path", "with_suffix"),
    ("pathlib", "Path", "with_stem"),
    ("pathlib", "Path", "relative_to"),
    ("pathlib", "Path", "is_relative_to"),
    ("pathlib", "Path", "name"),
    ("pathlib", "Path", "stem"),
    ("pathlib", "Path", "suffix"),
    ("pathlib", "Path", "suffixes"),
    ("pathlib", "Path", "parent"),
    ("pathlib", "Path", "parents"),
    ("pathlib", "Path", "parts"),
    ("pathlib", "Path", "anchor"),
    ("pathlib", "Path", "drive"),
    ("pathlib", "Path", "root"),
})


def classify_call(
    module: str | None,
    func_name: str | None,
    object_type: str | None,
    analysis: "ModuleAnalysis",
) -> tuple[Verdict, str]:
    """Classify a call site as SAFE / BLOCKING / ASYNC / UNKNOWN.

    Returns (Verdict, reason_string).
    """
    if not func_name:
        return Verdict.UNKNOWN, "missing function name"

    # asyncio itself is mostly safe, but these APIs try to drive an event loop
    # from inside code that may already be running on one.
    if module and module.split(".")[0] == "asyncio" and func_name in {
        "run",
        "run_until_complete",
    }:
        return Verdict.BLOCKING, f"asyncio.{func_name} drives the event loop synchronously"

    # 1a. getattr(module, <dynamic>) — unknown function on a known module.
    # Flag as warning if the module is not safe (potential blocking call).
    if func_name == "<getattr>" and module:
        top_module = module.split(".")[0]
        if top_module in SAFE_MODULES:
            return Verdict.SAFE, f"{module} is a pure-computation module"
        return (
            Verdict.BLOCKING,
            f"getattr() on {module} — dynamic dispatch may call blocking function",
        )

    # 1. Local function (defined in the analyzed module) — handled by tracer; safe here.
    if module is None and func_name in analysis.functions:
        return Verdict.SAFE, "local function"

    # 2. Builtin open() — language-level builtin, blocks for file I/O.
    if module is None and func_name == "open":
        return Verdict.BLOCKING, "builtin open() does file I/O"

    # 3. SAFE_MODULES — computational stdlib, never blocks.
    if module:
        top_module = module.split(".")[0]
        if top_module in SAFE_MODULES:
            return Verdict.SAFE, f"{module} is a pure-computation module"

    # 4. Per-function safe exceptions inside larger modules.
    if module and (module, func_name) in SAFE_STDLIB_FUNCTIONS:
        return Verdict.SAFE, f"{module}.{func_name} is non-blocking"

    # 4a. Per-method safe exceptions on stdlib classes (pure path manipulation, etc.)
    if (
        object_type
        and module
        and (module, object_type, func_name) in SAFE_STDLIB_METHODS
    ):
        return Verdict.SAFE, f"{module}.{object_type}.{func_name} is non-blocking"

    # 4b. Class constructor handling.
    #   - Stdlib class constructor (e.g. Path(), socket()): SAFE. The blocking
    #     happens in its methods, which the checker traces via object_type.
    #   - Exception class constructor (any origin): SAFE. `raise HTTPException(...)`
    #     does not block; instantiating an exception is pure object creation.
    #   - Other third-party class constructor (e.g. requests.Session()): leave
    #     to the BLOCKING fall-through, since those represent synchronous
    #     resource acquisition.
    if module and not object_type:
        _origin_check = resolve_module_origin(module)
        if _origin_check in (
            ModuleOrigin.STDLIB,
            ModuleOrigin.BUILTIN,
            ModuleOrigin.THIRD_PARTY,
        ):
            try:
                import importlib as _il
                _m = _il.import_module(module)
                _attr = getattr(_m, func_name, None)
                if _attr is not None and isinstance(_attr, type):
                    if _origin_check in (ModuleOrigin.STDLIB, ModuleOrigin.BUILTIN):
                        return (
                            Verdict.SAFE,
                            f"{module}.{func_name} is a stdlib class constructor",
                        )
                    if func_name.startswith("Async"):
                        return (
                            Verdict.SAFE,
                            f"{module}.{func_name} is an async client constructor",
                        )
                    if issubclass(_attr, BaseException):
                        return (
                            Verdict.SAFE,
                            f"{module}.{func_name} is an exception class",
                        )
            except BaseException:
                pass

    # 5. Async callable — verified via inspect.iscoroutinefunction.
    if module and is_async_callable(module, func_name, object_type):
        return Verdict.ASYNC, f"{module}.{func_name} is a coroutine function"

    # 6. Module origin checks.
    if not module:
        # Try to recover module from object_type via import_froms.
        # e.g. `from pathlib import Path; p = Path(...); p.read_text()`
        # resolves to module=None, func=read_text, object_type=Path.
        if object_type and object_type in analysis.import_froms:
            imp = analysis.import_froms[object_type]
            module = imp.module
        elif object_type and object_type in analysis.imports:
            imp = analysis.imports[object_type]
            module = imp.module
        if not module:
            return Verdict.UNKNOWN, f"unresolved call: {func_name}"

    # Re-check SAFE_* tables now that the module is resolved. The early
    # checks above may have run with module=None (e.g. `p.is_absolute()`
    # where `p = Path(...)`), in which case neither SAFE_MODULES nor
    # SAFE_STDLIB_METHODS could match.
    top_module = module.split(".")[0]
    if top_module in SAFE_MODULES:
        return Verdict.SAFE, f"{module} is a pure-computation module"
    if (
        object_type
        and (module, object_type, func_name) in SAFE_STDLIB_METHODS
    ):
        return Verdict.SAFE, f"{module}.{object_type}.{func_name} is non-blocking"

    origin = resolve_module_origin(module)

    # If the module is importable, verify the attribute actually exists. A
    # chained call like requests.get(url).json() resolves to module=requests,
    # func=json, object_type=get — but `requests.json` does not exist; `json`
    # is a method on the Response object, whose type we don't statically know.
    # Without proof that the attribute belongs to the module (or to a class
    # named `object_type` on that module), we cannot assert it's blocking.
    if origin in (
        ModuleOrigin.STDLIB,
        ModuleOrigin.BUILTIN,
        ModuleOrigin.THIRD_PARTY,
    ):
        if object_type:
            # Chained / method-on-instance form. Try to verify the class
            # lives in the module AND exposes the method.
            try:
                import importlib as _il
                m = _il.import_module(module)
                cls = getattr(m, object_type, None)
                if cls is None:
                    # For stdlib/builtin modules, the object_type may be a
                    # method name (e.g. cursor from conn.cursor()) rather
                    # than an exported class. Trust the module origin and
                    # fall through to BLOCKING — the call IS on a return
                    # value of a stdlib function.
                    if origin in (
                        ModuleOrigin.STDLIB,
                        ModuleOrigin.BUILTIN,
                    ):
                        pass  # fall through to BLOCKING below
                    else:
                        return (
                            Verdict.UNKNOWN,
                            f"{module}.{object_type} unresolved",
                        )
                if not hasattr(cls, func_name):
                    # The class/factory exists but we can't confirm the
                    # method. For stdlib non-safe modules, factories like
                    # threading.Lock() return opaque objects whose methods
                    # block (acquire, get, recv, etc.) — trust the module
                    # origin and fall through to BLOCKING. For third-party,
                    # the method likely belongs to a return-type we don't
                    # know about (e.g. requests.get().json()) — leave UNKNOWN.
                    if origin == ModuleOrigin.THIRD_PARTY:
                        return (
                            Verdict.UNKNOWN,
                            f"{module}.{object_type}.{func_name} unresolved",
                        )
            except BaseException:
                return Verdict.UNKNOWN, f"could not import {module}"
        else:
            # Direct call form: module.func_name. Require the attribute to
            # exist on stdlib modules before classifying as blocking. For
            # third-party modules, a failed import/attribute probe commonly
            # means the target project's venv is not importable by the linter
            # interpreter (for example Python 3.11 wheels under Python 3.12).
            # The static shape is still a direct third-party call.
            if not attr_exists(module, func_name):
                if origin == ModuleOrigin.THIRD_PARTY:
                    if func_name and func_name[0].isupper() and any(
                        func_name.endswith(suffix)
                        for suffix in ("Error", "Exception", "Warning")
                    ):
                        return (
                            Verdict.SAFE,
                            f"{module}.{func_name} is an exception class (by name)",
                        )
                    if func_name.startswith("Async"):
                        return (
                            Verdict.SAFE,
                            f"{module}.{func_name} is an async client constructor (by name)",
                        )
                    return (
                        Verdict.BLOCKING,
                        f"{module}.{func_name} is synchronous (third-party); could not verify async",
                    )
                return (
                    Verdict.UNKNOWN,
                    f"{module}.{func_name} not exported by module",
                )

    if origin in (ModuleOrigin.STDLIB, ModuleOrigin.BUILTIN):
        # Stdlib that wasn't whitelisted — assume blocking.
        return (
            Verdict.BLOCKING,
            f"{module}.{func_name} is stdlib and not async-safe",
        )

    if origin == ModuleOrigin.THIRD_PARTY:
        # Third-party, not async — blocking.
        return (
            Verdict.BLOCKING,
            f"{module}.{func_name} is synchronous (third-party); use an async equivalent",
        )

    if origin == ModuleOrigin.NOT_INSTALLED:
        # Not installed in the linter's environment. Two very different
        # populations land here:
        #
        #   (a) Third-party deps the user has but the linter doesn't see
        #       because it runs in an isolated `uvx --from git+...` venv —
        #       e.g. `requests`, `httpx`, `psycopg2`. These are typically
        #       single top-level package names.
        #   (b) Project-local packages — `src.utils.logging`,
        #       `app.api.schemas`, `mycompany.foo.bar`. These are dotted
        #       paths that resolve nowhere because the linter doesn't have
        #       the user's project on its sys.path.
        #
        # Heuristic: a dotted module path (>= 2 segments) is almost always
        # project-local; flagging it produces noise on every helper call
        # (logger.info, schema construction, internal services). A single
        # segment is much more likely to be a real third-party dependency,
        # which is the population we want to keep flagging — that's what
        # makes the linter useful when run standalone via uvx.
        if "." in module:
            return (
                Verdict.UNKNOWN,
                f"{module}.{func_name} looks project-local; cannot verify",
            )
        # Single-segment uninstalled module: still suppress exception-class
        # constructors by name (raising exceptions doesn't block).
        if func_name and func_name[0].isupper() and any(
            func_name.endswith(suffix)
            for suffix in ("Error", "Exception", "Warning")
        ):
            return (
                Verdict.SAFE,
                f"{module}.{func_name} is an exception class (by name)",
            )
        return (
            Verdict.BLOCKING,
            f"{module}.{func_name} is from an uninstalled module; cannot verify async",
        )

    return Verdict.UNKNOWN, f"could not classify {module}.{func_name}"
