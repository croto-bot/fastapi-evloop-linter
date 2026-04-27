"""Generic async-endpoint detection.

A function is treated as an async endpoint if it is decorated by something
that resolves to a third-party (non-stdlib, non-project-local) class or
callable. This handles fastapi/starlette/litestar/aiohttp/quart/flask
without naming any of them.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .classifier import SAFE_MODULES
from .introspect import ModuleOrigin, resolve_module_origin

if TYPE_CHECKING:
    from .callgraph import ModuleAnalysis


def _get_decorator_base(dec_ast: ast.expr) -> str | None:
    """Extract the base name from a decorator AST.

    Examples:
        app.get("/")            -> "app"
        router.post("/x")       -> "router"
        api.options             -> "api"
        my_dec                  -> "my_dec"
        a.b.c                   -> "a"
    """
    # Decorator written as a call: app.get(...) or my_dec(...)
    if isinstance(dec_ast, ast.Call):
        return _get_decorator_base(dec_ast.func)

    # Attribute access: app.get / router.post / a.b.c
    if isinstance(dec_ast, ast.Attribute):
        # Walk down to the leftmost Name
        cur: ast.expr = dec_ast.value
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if isinstance(cur, ast.Name):
            return cur.id
        return None

    # Bare name: @some_decorator
    if isinstance(dec_ast, ast.Name):
        return dec_ast.id

    return None


def _is_third_party_module(module_name: str | None) -> bool:
    """True if the module is third-party (not stdlib, not project-local).

    Accepts both installed third-party modules and not-installed ones: the
    linter often runs in an isolated environment (e.g. ``uvx --from git+...``)
    where the user's web framework (fastapi/starlette/litestar/...) isn't
    importable. We still want to recognize ``@router.post(...)`` style
    decorators as endpoints in that case, because the user code clearly
    imports a non-stdlib name.
    """
    if not module_name:
        return False

    top = module_name.split(".")[0]
    if top in SAFE_MODULES:
        return False

    origin = resolve_module_origin(module_name)
    return origin in (ModuleOrigin.THIRD_PARTY, ModuleOrigin.NOT_INSTALLED)


def _resolve_base_to_module(
    base_name: str, analysis: "ModuleAnalysis"
) -> str | None:
    """Resolve a decorator base name to its origin module.

    The base name may refer to:
      * a module-level variable typed as a class imported from a third-party module
        (e.g. `app = FastAPI()`)
      * an imported module (e.g. `import flask` and `@flask.route(...)`)
      * an imported name (e.g. `from fastapi import APIRouter`)
    """
    # Module-level variable holding a class instance.
    if base_name in analysis.module_var_types:
        class_name = analysis.module_var_types[base_name]
        # Find where this class came from.
        if class_name in analysis.import_froms:
            return analysis.import_froms[class_name].module
        if class_name in analysis.imports:
            return analysis.imports[class_name].module

    # Direct import: `import flask` -> base "flask".
    if base_name in analysis.imports:
        return analysis.imports[base_name].module

    # From-import: `from fastapi import APIRouter` -> base "APIRouter".
    if base_name in analysis.import_froms:
        return analysis.import_froms[base_name].module

    return None


def _decorator_is_method_call(dec_ast: ast.expr) -> bool:
    """Decorator is `<base>.<verb>(...)` (a method call on an attribute)."""
    if not isinstance(dec_ast, ast.Call):
        return False
    return isinstance(dec_ast.func, ast.Attribute)


def is_async_endpoint(
    func_node: ast.AST,
    decorators_ast: list[ast.expr],
    analysis: "ModuleAnalysis",
) -> bool:
    """Return True if this function looks like an async web-framework endpoint."""
    if not isinstance(func_node, ast.AsyncFunctionDef):
        return False

    for dec in decorators_ast:
        base = _get_decorator_base(dec)
        if base is None:
            continue

        module = _resolve_base_to_module(base, analysis)
        if module and _is_third_party_module(module):
            # Method-call decorator: app.get("/path")
            if _decorator_is_method_call(dec):
                return True
            # Bare-name decorator imported from a third-party framework.
            if isinstance(dec, ast.Name) or isinstance(dec, ast.Call):
                return True

    return False
