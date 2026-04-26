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


# Computational stdlib modules that never do I/O.
SAFE_MODULES: frozenset[str] = frozenset({
    "math", "operator", "itertools", "functools", "collections",
    "dataclasses", "typing", "abc", "enum", "decimal", "fractions",
    "statistics", "base64", "secrets", "uuid", "copy", "string",
    "re", "types", "contextlib",
})

# Per-function safe exceptions inside otherwise-unsafe modules.
SAFE_STDLIB_FUNCTIONS: frozenset[tuple[str, str]] = frozenset({
    ("json", "loads"), ("json", "dumps"),
    ("hashlib", "md5"), ("hashlib", "sha1"), ("hashlib", "sha256"),
    ("hashlib", "sha512"), ("hashlib", "sha384"), ("hashlib", "sha224"),
    ("hashlib", "blake2b"), ("hashlib", "blake2s"), ("hashlib", "new"),
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

    # 4b. Stdlib class constructor: calling a stdlib class (e.g. Path(), socket())
    # is not itself blocking — the blocking is in its methods, which the checker
    # traces via object_type resolution. Don't flag the constructor call itself.
    # Note: third-party class constructors (e.g. requests.Session()) ARE flagged
    # because they represent synchronous third-party resource acquisition.
    if module and not object_type:
        _origin_check = resolve_module_origin(module)
        if _origin_check in (ModuleOrigin.STDLIB, ModuleOrigin.BUILTIN):
            try:
                import importlib as _il
                _m = _il.import_module(module)
                _attr = getattr(_m, func_name, None)
                if _attr is not None and isinstance(_attr, type):
                    return Verdict.SAFE, f"{module}.{func_name} is a stdlib class constructor"
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
            # exist on the module before classifying as blocking.
            if not attr_exists(module, func_name):
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
        # Not installed — cannot verify; conservatively flag as blocking.
        return (
            Verdict.BLOCKING,
            f"{module}.{func_name} is from an uninstalled module; cannot verify async",
        )

    return Verdict.UNKNOWN, f"could not classify {module}.{func_name}"
