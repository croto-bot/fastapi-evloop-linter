"""Module resolution and introspection layer.

Resolves module origins (stdlib / third-party / not installed) and
introspects callables to determine if they're coroutine functions.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from enum import Enum
from functools import lru_cache


class ModuleOrigin(Enum):
    STDLIB = "stdlib"
    BUILTIN = "builtin"
    THIRD_PARTY = "third_party"
    NOT_INSTALLED = "not_installed"
    UNKNOWN = "unknown"


@lru_cache(maxsize=1)
def _stdlib_path_prefix() -> str:
    """Filesystem prefix where stdlib .py files live (used as fallback on Python <3.10)."""
    try:
        import sysconfig as _sc
        return _sc.get_path("stdlib") or ""
    except Exception:
        return ""


@lru_cache(maxsize=512)
def resolve_module_origin(module_name: str) -> ModuleOrigin:
    """Determine the origin (stdlib / third-party / not installed) of a module."""
    if not module_name:
        return ModuleOrigin.UNKNOWN

    top_level = module_name.split(".")[0]

    # Pure C builtin modules are always stdlib.
    if top_level in sys.builtin_module_names:
        return ModuleOrigin.BUILTIN

    # Python 3.10+: sys.stdlib_module_names is authoritative.
    stdlib_names = getattr(sys, "stdlib_module_names", frozenset())
    if stdlib_names and top_level in stdlib_names:
        return ModuleOrigin.STDLIB

    # Try to find the module spec.
    try:
        spec = importlib.util.find_spec(top_level)
    except (ImportError, ValueError, ModuleNotFoundError, AttributeError):
        return ModuleOrigin.NOT_INSTALLED
    except Exception:
        return ModuleOrigin.UNKNOWN

    if spec is None:
        return ModuleOrigin.NOT_INSTALLED

    # No origin file → built-in C extension.
    if spec.origin is None or spec.origin == "built-in":
        return ModuleOrigin.BUILTIN

    # Python 3.9 and earlier fallback: check whether the module's file lives
    # inside the stdlib directory (not site-packages).
    if not stdlib_names:
        stdlib_prefix = _stdlib_path_prefix()
        if stdlib_prefix and spec.origin.startswith(stdlib_prefix):
            return ModuleOrigin.STDLIB

    return ModuleOrigin.THIRD_PARTY


@lru_cache(maxsize=512)
def attr_exists(module: str, attr_name: str) -> bool:
    """True if `module` is importable and has `attr_name` as a top-level attribute."""
    if not module or not attr_name:
        return False
    try:
        mod = importlib.import_module(module)
    except BaseException:
        return False
    try:
        return hasattr(mod, attr_name)
    except BaseException:
        return False


@lru_cache(maxsize=512)
def is_async_callable(
    module: str, func_name: str, object_type: str | None
) -> bool:
    """Check whether (module, func_name [, object_type]) refers to a coroutine function.

    Returns False on any import or attribute error.
    """
    if not module or not func_name:
        return False

    try:
        mod = importlib.import_module(module)
    except BaseException:
        return False

    try:
        # If we have an object_type, look up the class first then the method
        if object_type:
            cls = getattr(mod, object_type, None)
            if cls is None:
                return False
            target = getattr(cls, func_name, None)
        else:
            # Try direct attribute by name; if it's a class and func_name was the class
            # itself, fall back to None.
            target = getattr(mod, func_name, None)

        if target is None:
            return False

        if inspect.iscoroutinefunction(target):
            return True

        # If target is a class, check if calling it produces an awaitable (rare).
        # Skip — we only consider explicit coroutine functions.
        return False
    except BaseException:
        return False
