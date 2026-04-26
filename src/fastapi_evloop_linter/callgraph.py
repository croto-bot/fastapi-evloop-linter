"""Inter-procedural call graph analysis for detecting deep blocking calls.

This module builds a call graph from async entry points (FastAPI endpoints)
and traces through helper functions to find blocking calls buried deep in
the call tree.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CallSite:
    """A call site in the source code."""
    name: str  # The function/method being called
    line: int
    col: int
    module: str | None = None  # Module prefix (e.g., "requests" for requests.get)
    object_type: str | None = None  # Object type for method calls (e.g., "Path")
    # The function this call is inside
    caller_function: str | None = None
    # Function names passed as positional arguments (None for non-Name args)
    arg_names: list[str | None] = field(default_factory=list)


@dataclass
class FuncDef:
    """A function definition found in the source."""
    name: str
    is_async: bool
    node: ast.AsyncFunctionDef | ast.FunctionDef
    calls: list[CallSite] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    # Variables tracked for type inference
    var_types: dict[str, str] = field(default_factory=dict)
    # Parameter names (for callback resolution)
    params: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Tracked import information."""
    module: str  # e.g., "requests"
    name: str  # e.g., "get" or "Path"
    alias: str | None = None  # e.g., "req" for `import requests as req`


@dataclass
class ModuleAnalysis:
    """Analysis results for a single module."""
    filepath: str
    functions: dict[str, FuncDef] = field(default_factory=dict)
    imports: dict[str, ImportInfo] = field(default_factory=dict)  # alias/name -> ImportInfo
    import_froms: dict[str, ImportInfo] = field(default_factory=dict)  # name -> ImportInfo
    # Call graph: caller_name -> [callee_names]
    call_graph: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


class CallGraphBuilder(ast.NodeVisitor):
    """AST visitor that builds a call graph with import resolution."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.analysis = ModuleAnalysis(filepath=filepath)
        self._current_func: FuncDef | None = None
        self._scope_stack: list[FuncDef] = []

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Extract decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_decorator_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return ""

    def _is_fastapi_endpoint(self, decorators: list[ast.expr]) -> bool:
        """Check if function decorators indicate a FastAPI endpoint."""
        fastapi_patterns = {
            "app.get", "app.post", "app.put", "app.delete", "app.patch",
            "app.head", "app.options", "app.route",
            "router.get", "router.post", "router.put", "router.delete",
            "router.patch", "router.head", "router.options", "router.route",
            "api.get", "api.post", "api.put", "api.delete", "api.patch",
        }
        # Also match any .get/.post etc. on any variable (heuristic)
        http_methods = {"get", "post", "put", "delete", "patch", "head", "options"}

        for dec in decorators:
            name = self._get_decorator_name(dec)
            if name in fastapi_patterns:
                return True
            # Check for any_var.method pattern
            parts = name.split(".")
            if len(parts) == 2 and parts[1] in http_methods:
                return True
            # APIRouter() or FastAPI() instances with decorators
            if len(parts) >= 2 and parts[-1] in http_methods:
                return True
        return False

    def _resolve_call_name(self, node: ast.Call) -> tuple[str | None, str | None, str | None]:
        """Resolve a call to (module, func_name, object_type).

        Returns (module, name, object_type) or (None, name, None) for unresolvable.
        """
        func = node.func

        if isinstance(func, ast.Name):
            # Simple function call: foo()
            name = func.id
            # Check if it's an imported name
            if name in self.analysis.imports:
                imp = self.analysis.imports[name]
                return imp.module, None, None  # module.method()
            if name in self.analysis.import_froms:
                imp = self.analysis.import_froms[name]
                # Return the ORIGINAL name (not alias) so blocker lookup works
                return imp.module, imp.name, None
            return None, name, None

        elif isinstance(func, ast.Attribute):
            # Method/module call: obj.method() or module.func()
            attr_name = func.attr
            value = func.value

            if isinstance(value, ast.Name):
                base_name = value.id
                # Check if base is an imported module
                if base_name in self.analysis.imports:
                    imp = self.analysis.imports[base_name]
                    return imp.module, attr_name, None
                # Check if base is an imported object
                if base_name in self.analysis.import_froms:
                    imp = self.analysis.import_froms[base_name]
                    return imp.module, attr_name, imp.name
                # Check if we know the type of this variable
                if self._current_func and base_name in self._current_func.var_types:
                    obj_type = self._current_func.var_types[base_name]
                    return None, attr_name, obj_type
                return None, attr_name, base_name

            elif isinstance(value, ast.Attribute):
                # Nested: a.b.method()
                inner_attr = value.attr
                inner_value = value.value
                if isinstance(inner_value, ast.Name):
                    base = inner_value.id
                    if base in self.analysis.imports:
                        imp = self.analysis.imports[base]
                        return f"{imp.module}.{inner_attr}", attr_name, None
                return None, attr_name, None

            elif isinstance(value, ast.Call):
                # Chained: SomeClass().method() or module.Func().method()
                # Try to resolve the call target
                inner = self._resolve_call_name(value)
                if inner and inner[1]:
                    # If inner is a known blocking module's function, the method
                    # call on its result is also likely blocking (e.g., requests.Session().get())
                    if inner[0] and inner[0] in ("requests",):
                        # e.g., requests.Session().get() -> module=requests, func=get
                        return inner[0], attr_name, inner[1]
                    return None, attr_name, inner[1]

            return None, attr_name, None

        return None, None, None

    def _track_assignment_type(self, target: ast.expr, value: ast.expr) -> None:
        """Track variable types from assignments for method call resolution."""
        if not self._current_func:
            return

        if isinstance(target, ast.Name) and isinstance(value, ast.Call):
            # Track: var = SomeClass()
            func = value.func
            if isinstance(func, ast.Name):
                name = func.id
                if name in self.analysis.import_froms:
                    self._current_func.var_types[target.id] = self.analysis.import_froms[name].name
                elif name in self.analysis.imports:
                    self._current_func.var_types[target.id] = name
                else:
                    self._current_func.var_types[target.id] = name
            elif isinstance(func, ast.Attribute):
                self._current_func.var_types[target.id] = func.attr

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            info = ImportInfo(
                module=alias.name,
                name=alias.name.split(".")[-1],
                alias=alias.asname,
            )
            key = alias.asname if alias.asname else alias.name
            self.analysis.imports[key] = info
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            info = ImportInfo(
                module=module,
                name=alias.name,
                alias=alias.asname,
            )
            key = alias.asname if alias.asname else alias.name
            self.analysis.import_froms[key] = info
        self.generic_visit(node)

    def _extract_params(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Extract parameter names from a function definition."""
        params = []
        args = node.args
        for arg in args.args:
            params.append(arg.arg)
        if args.vararg:
            params.append(args.vararg.arg)
        for arg in args.kwonlyargs:
            params.append(arg.arg)
        if args.kwarg:
            params.append(args.kwarg.arg)
        return params

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        func = FuncDef(
            name=node.name,
            is_async=True,
            node=node,
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            params=self._extract_params(node),
        )
        is_endpoint = self._is_fastapi_endpoint(node.decorator_list)

        prev = self._current_func
        self._current_func = func
        self._scope_stack.append(func)

        self.generic_visit(node)

        self._scope_stack.pop()
        self._current_func = prev

        self.analysis.functions[node.name] = func

        if is_endpoint:
            func.decorators.append("__fastapi_endpoint__")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit sync function definitions (helpers called from async)."""
        func = FuncDef(
            name=node.name,
            is_async=False,
            node=node,
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            params=self._extract_params(node),
        )

        prev = self._current_func
        self._current_func = func
        self._scope_stack.append(func)

        self.generic_visit(node)

        self._scope_stack.pop()
        self._current_func = prev

        self.analysis.functions[node.name] = func

    def visit_Call(self, node: ast.Call) -> None:
        if self._current_func is None:
            self.generic_visit(node)
            return

        module, func_name, obj_type = self._resolve_call_name(node)

        if func_name is None:
            self.generic_visit(node)
            return

        # Extract Name arguments (for callback/function-reference tracking)
        # Use None for non-Name args to preserve positional alignment
        arg_names: list[str | None] = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                arg_names.append(arg.id)
            else:
                arg_names.append(None)
        call_site = CallSite(
            name=func_name,
            line=node.lineno,
            col=node.col_offset,
            module=module,
            object_type=obj_type,
            caller_function=self._current_func.name,
            arg_names=arg_names,
        )

        self._current_func.calls.append(call_site)

        # Also record in call graph
        self.analysis.call_graph[self._current_func.name].append(func_name)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track variable assignments for type inference."""
        for target in node.targets:
            self._track_assignment_type(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Track annotated assignments for type inference."""
        if node.target and node.value:
            self._track_assignment_type(node.target, node.value)
        self.generic_visit(node)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        """Track loop variable types."""
        if self._current_func and isinstance(node.target, ast.Name):
            # e.g., for item in some_list: - can't easily infer type
            pass
        self.generic_visit(node)

    visit_AsyncFor = visit_For


def analyze_file(filepath: str | Path) -> ModuleAnalysis:
    """Analyze a single Python file and return its call graph."""
    filepath = Path(filepath)
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))

    builder = CallGraphBuilder(str(filepath))
    builder.visit(tree)

    return builder.analysis


def analyze_source(source: str, filepath: str = "<test>") -> ModuleAnalysis:
    """Analyze source code string."""
    tree = ast.parse(source, filename=filepath)
    builder = CallGraphBuilder(filepath)
    builder.visit(tree)
    return builder.analysis
