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
    # Whether this is a property access (not a direct call)
    is_property_access: bool = False


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
    # Variable aliases: var_name -> (module, func_name) for blocking function refs
    var_aliases: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    # Whether this function is a decorator wrapper that replaces the original
    is_decorator_wrapper: bool = False
    # The original function name this wrapper replaces (set when used as decorator)
    replaces_func: str | None = None
    # Blocking reference returned by this function: (module, func_name) or None
    returns_blocking: tuple[str | None, str | None] | None = None


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
    # Class definitions: class_name -> set of method names
    class_methods: dict[str, set[str]] = field(default_factory=dict)
    # Dunder method to class mapping: method_name (e.g. "__call__") -> {class_name: func_def_name}
    dunder_map: dict[str, dict[str, str]] = field(default_factory=dict)
    # Decorator replacements: original_func_name -> wrapper_func_name
    decorator_replacements: dict[str, str] = field(default_factory=dict)
    # Module-level dict literals that map to blocking functions: dict_name -> {key: (module, func_name)}
    blocking_dicts: dict[str, dict[str, tuple[str | None, str | None]]] = field(default_factory=dict)
    # Properties: class_name -> set of property names
    class_properties: dict[str, set[str]] = field(default_factory=dict)
    # Module-level variable types: var_name -> class_name
    module_var_types: dict[str, str] = field(default_factory=dict)
    # Class attributes that reference known functions: class_name -> {attr_name: func_name}
    class_func_attrs: dict[str, dict[str, str]] = field(default_factory=dict)


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
            # Check if it's a variable alias for a blocking function
            if self._current_func and name in self._current_func.var_aliases:
                alias_mod, alias_name = self._current_func.var_aliases[name]
                return alias_mod, alias_name, None
            # Check if this is a callable object (has __call__)
            if self._current_func and name in self._current_func.var_types:
                var_type = self._current_func.var_types[name]
                call_key = f"{var_type}.__call__"
                if call_key in self.analysis.functions:
                    # Return the __call__ key so the checker traces into it
                    return None, call_key, var_type
            # Check module-level var types too
            if name in self.analysis.module_var_types:
                var_type = self.analysis.module_var_types[name]
                call_key = f"{var_type}.__call__"
                if call_key in self.analysis.functions:
                    return None, call_key, var_type
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
                    # Check if this is a class function attribute
                    if obj_type in self.analysis.class_func_attrs:
                        if attr_name in self.analysis.class_func_attrs[obj_type]:
                            real_func = self.analysis.class_func_attrs[obj_type][attr_name]
                            return None, real_func, None
                    return None, attr_name, obj_type
                # Check module-level var types
                if base_name in self.analysis.module_var_types:
                    obj_type = self.analysis.module_var_types[base_name]
                    if obj_type in self.analysis.class_func_attrs:
                        if attr_name in self.analysis.class_func_attrs[obj_type]:
                            real_func = self.analysis.class_func_attrs[obj_type][attr_name]
                            return None, real_func, None
                    return None, attr_name, obj_type
                # Check if we know the type of this variable
                if self._current_func and base_name in self._current_func.var_types:
                    obj_type = self._current_func.var_types[base_name]
                    # Check if this is a class function attribute
                    if obj_type in self.analysis.class_func_attrs:
                        if attr_name in self.analysis.class_func_attrs[obj_type]:
                            real_func = self.analysis.class_func_attrs[obj_type][attr_name]
                            return None, real_func, None
                    return None, attr_name, obj_type
                # Check module-level var types
                if base_name in self.analysis.module_var_types:
                    obj_type = self.analysis.module_var_types[base_name]
                    if obj_type in self.analysis.class_func_attrs:
                        if attr_name in self.analysis.class_func_attrs[obj_type]:
                            real_func = self.analysis.class_func_attrs[obj_type][attr_name]
                            return None, real_func, None
                    return None, attr_name, obj_type
                # Check if base is a known variable from imports (e.g., lock = threading.Lock())
                # Use the base_name as a fallback obj_type for blocking method detection
                return None, attr_name, base_name

            if isinstance(value, ast.Attribute):
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

        if isinstance(target, ast.Name):
            # Track: var = SomeClass()
            if isinstance(value, ast.Call):
                func = value.func
                if isinstance(func, ast.Name):
                    name = func.id
                    if name in self.analysis.import_froms:
                        self._current_func.var_types[target.id] = self.analysis.import_froms[name].name
                    elif name in self.analysis.imports:
                        self._current_func.var_types[target.id] = name
                    else:
                        self._current_func.var_types[target.id] = name

                    # Check if the called function returns a blocking reference
                    callee = self.analysis.functions.get(name)
                    if callee and callee.returns_blocking:
                        self._current_func.var_aliases[target.id] = callee.returns_blocking

                elif isinstance(func, ast.Attribute):
                    self._current_func.var_types[target.id] = func.attr
                    # Also check if this is module.ClassName() pattern
                    # e.g., q = queue.Queue() -> need to track module context
                    if isinstance(func.value, ast.Name):
                        base = func.value.id
                        if base in self.analysis.imports:
                            # Store as "module.ClassName" for richer matching
                            pass

            # Track variable aliasing to blocking functions
            # e.g. f = time.sleep  or  f = requests.get
            self._track_var_alias(target.id, value)

    def _track_var_alias(self, var_name: str, value: ast.expr) -> None:
        """Track when a variable is assigned a reference to a blocking function."""
        if not self._current_func:
            return

        # Direct module.attr reference: f = time.sleep
        if isinstance(value, ast.Attribute):
            attr_name = value.attr
            if isinstance(value.value, ast.Name):
                base = value.value.id
                if base in self.analysis.imports:
                    imp = self.analysis.imports[base]
                    self._current_func.var_aliases[var_name] = (imp.module, attr_name)
                    return
                if base in self.analysis.import_froms:
                    imp = self.analysis.import_froms[base]
                    self._current_func.var_aliases[var_name] = (imp.module, attr_name)
                    return

        # Direct name reference: f = sleep  (where sleep was from-imported)
        if isinstance(value, ast.Name):
            ref = value.id
            if ref in self.analysis.import_froms:
                imp = self.analysis.import_froms[ref]
                self._current_func.var_aliases[var_name] = (imp.module, imp.name)
                return
            if ref in self.analysis.imports:
                imp = self.analysis.imports[ref]
                self._current_func.var_aliases[var_name] = (imp.module, None)
                return
            # Check if ref is itself a var alias (propagation)
            if ref in self._current_func.var_aliases:
                self._current_func.var_aliases[var_name] = self._current_func.var_aliases[ref]
                return

        # Walrus operator in NamedExpr: (x := time.sleep)
        if isinstance(value, ast.NamedExpr):
            self._track_var_alias(var_name, value.value)

        # functools.partial: wait = partial(time.sleep, 1)
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name) and value.func.id == "partial":
                if value.args:
                    first_arg = value.args[0]
                    if isinstance(first_arg, ast.Attribute):
                        attr_name = first_arg.attr
                        if isinstance(first_arg.value, ast.Name):
                            base = first_arg.value.id
                            if base in self.analysis.imports:
                                imp = self.analysis.imports[base]
                                self._current_func.var_aliases[var_name] = (imp.module, attr_name)
                                return
                    if isinstance(first_arg, ast.Name):
                        ref = first_arg.id
                        if ref in self.analysis.import_froms:
                            imp = self.analysis.import_froms[ref]
                            self._current_func.var_aliases[var_name] = (imp.module, imp.name)
                            return

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

        # Store function - but check if a decorator wrapper already took this name
        self.analysis.functions[node.name] = func

        # Check if this function is a decorator wrapper that replaces the original
        # Pattern: a function defined inside another function and returned as decorator result
        if self._current_func and node.name == "wrapper":
            # This wrapper might replace a decorated function
            self._current_func.is_decorator_wrapper = True

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class definitions and their methods."""
        methods = set()
        properties = set()
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if it's a property
                is_prop = any(
                    self._get_decorator_name(d) == "property"
                    for d in item.decorator_list
                )
                if is_prop:
                    properties.add(item.name)
                    # Add property getter as a synthetic function for tracing
                    prop_key = f"{node.name}.<prop:{item.name}>"
                    prop_func = FuncDef(
                        name=prop_key,
                        is_async=isinstance(item, ast.AsyncFunctionDef),
                        node=item,
                        decorators=[self._get_decorator_name(d) for d in item.decorator_list],
                        params=self._extract_params(item),
                    )
                    prev = self._current_func
                    self._current_func = prop_func
                    self._scope_stack.append(prop_func)
                    self.generic_visit(item)
                    self._scope_stack.pop()
                    self._current_func = prev
                    self.analysis.functions[prop_key] = prop_func
                else:
                    methods.add(item.name)
                    # Map dunder methods: e.g. __call__ -> {ClassName: __call___ClassName}
                    if item.name.startswith("__") and item.name.endswith("__"):
                        if item.name not in self.analysis.dunder_map:
                            self.analysis.dunder_map[item.name] = {}
                        # Use a synthetic name to avoid collisions
                        dunder_key = f"{node.name}.{item.name}"
                        self.analysis.dunder_map[item.name][node.name] = dunder_key

                        # Also add the dunder method as a regular function so the checker can trace into it
                        dunder_func = FuncDef(
                            name=dunder_key,
                            is_async=isinstance(item, ast.AsyncFunctionDef),
                            node=item,
                            decorators=[self._get_decorator_name(d) for d in item.decorator_list],
                            params=self._extract_params(item),
                        )
                        # Visit the dunder body
                        prev = self._current_func
                        self._current_func = dunder_func
                        self._scope_stack.append(dunder_func)
                        self.generic_visit(item)
                        self._scope_stack.pop()
                        self._current_func = prev

                        self.analysis.functions[dunder_key] = dunder_func
                    else:
                        # Non-dunder, non-property methods - still visit their bodies
                        prev = self._current_func
                        method_func = FuncDef(
                            name=item.name,
                            is_async=isinstance(item, ast.AsyncFunctionDef),
                            node=item,
                            decorators=[self._get_decorator_name(d) for d in item.decorator_list],
                            params=self._extract_params(item),
                        )
                        self._current_func = method_func
                        self._scope_stack.append(method_func)
                        self.generic_visit(item)
                        self._scope_stack.pop()
                        self._current_func = prev
                        self.analysis.functions[item.name] = method_func

        self.analysis.class_methods[node.name] = methods
        self.analysis.class_properties[node.name] = properties

        # Track class-level attributes that reference known functions
        func_attrs: dict[str, str] = {}
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and isinstance(item.value, ast.Name):
                        # Class attr referencing a function: action = wait
                        ref = item.value.id
                        if ref in self.analysis.functions:
                            func_attrs[target.id] = ref
        if func_attrs:
            self.analysis.class_func_attrs[node.name] = func_attrs

        # Visit the class body for nested items (but don't re-visit methods)
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(item)

        # Track module-level variable assignments to this class
        # (e.g., cfg = Config() at module level)
        self._track_module_level_class_vars(node.name)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Track walrus operator assignments: (x := value)."""
        if self._current_func and isinstance(node.target, ast.Name):
            self._track_var_alias(node.target.id, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._current_func is None:
            self.generic_visit(node)
            return

        module, func_name, obj_type = self._resolve_call_name(node)

        if func_name is None:
            # Check for dict dispatch: HANDLERS["wait"](5)
            dict_result = self._resolve_dict_dispatch_call(node)
            if dict_result:
                module, func_name = dict_result
                obj_type = None

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

        # Handle constructor calls: ClassName() -> trace into ClassName.__init__
        if isinstance(node.func, ast.Name) and node.func.id in self.analysis.class_methods:
            class_name = node.func.id
            init_key = f"{class_name}.__init__"
            if init_key in self.analysis.functions:
                init_call = CallSite(
                    name=init_key,
                    line=node.lineno,
                    col=node.col_offset,
                    module=None,
                    object_type=class_name,
                    caller_function=self._current_func.name,
                    arg_names=arg_names,
                )
                self._current_func.calls.append(init_call)
                self.analysis.call_graph[self._current_func.name].append(init_key)

            # Also check __post_init__ (dataclass pattern)
            postinit_key = f"{class_name}.__post_init__"
            if postinit_key in self.analysis.functions:
                pi_call = CallSite(
                    name=postinit_key,
                    line=node.lineno,
                    col=node.col_offset,
                    module=None,
                    object_type=class_name,
                    caller_function=self._current_func.name,
                    arg_names=arg_names,
                )
                self._current_func.calls.append(pi_call)
                self.analysis.call_graph[self._current_func.name].append(postinit_key)

        # Handle map(func, items) as higher-order caller
        if isinstance(node.func, ast.Name) and node.func.id == "map" and len(node.args) >= 2:
            func_arg = node.args[0]
            if isinstance(func_arg, ast.Name):
                map_callee_call = CallSite(
                    name=func_arg.id,
                    line=node.lineno,
                    col=node.col_offset,
                    module=None,
                    object_type=None,
                    caller_function=self._current_func.name,
                    arg_names=None,
                )
                # Add as a proper call site so the checker traces into it
                self._current_func.calls.append(map_callee_call)
                self.analysis.call_graph[self._current_func.name].append(func_arg.id)

        # Handle filter(func, items) as higher-order caller
        if isinstance(node.func, ast.Name) and node.func.id == "filter" and len(node.args) >= 2:
            func_arg = node.args[0]
            if isinstance(func_arg, ast.Name):
                filter_callee_call = CallSite(
                    name=func_arg.id,
                    line=node.lineno,
                    col=node.col_offset,
                    module=None,
                    object_type=None,
                    caller_function=self._current_func.name,
                    arg_names=None,
                )
                self._current_func.calls.append(filter_callee_call)
                self.analysis.call_graph[self._current_func.name].append(func_arg.id)

        # Handle sorted(items, key=func) as higher-order caller
        if isinstance(node.func, ast.Name) and node.func.id == "sorted":
            for kw in node.keywords:
                if kw.arg == "key" and isinstance(kw.value, ast.Name):
                    sorted_callee_call = CallSite(
                        name=kw.value.id,
                        line=node.lineno,
                        col=node.col_offset,
                        module=None,
                        object_type=None,
                        caller_function=self._current_func.name,
                        arg_names=None,
                    )
                    self._current_func.calls.append(sorted_callee_call)
                    self.analysis.call_graph[self._current_func.name].append(kw.value.id)

        self.generic_visit(node)

    def _resolve_dict_dispatch_call(self, node: ast.Call) -> tuple[str | None, str | None] | None:
        """Resolve dict dispatch calls like HANDLERS[\"wait\"](5)."""
        # The func of the Call should be a Subscript
        if not isinstance(node.func, ast.Subscript):
            return None

        subscript = node.func
        # The value should be a Name referencing a known blocking dict
        if not isinstance(subscript.value, ast.Name):
            return None

        dict_name = subscript.value.id
        if dict_name not in self.analysis.blocking_dicts:
            return None

        # Get the key
        key = None
        if isinstance(subscript.slice, ast.Constant):
            key = subscript.slice.value

        if key and isinstance(key, str) and key in self.analysis.blocking_dicts[dict_name]:
            return self.analysis.blocking_dicts[dict_name][key]

        # If we can't resolve the exact key, report the first value as a heuristic
        # (covers cases where the key is a variable)
        if self.analysis.blocking_dicts[dict_name]:
            first_val = next(iter(self.analysis.blocking_dicts[dict_name].values()))
            return first_val

        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track variable assignments for type inference and aliasing."""
        for target in node.targets:
            self._track_assignment_type(target, node.value)
        # Track module-level assignments
        if self._current_func is None:
            self._track_blocking_dict(node)
            # Track module-level variable types
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Name):
                        self.analysis.module_var_types[target.id] = func.id
                    elif isinstance(func, ast.Attribute):
                        # e.g., q = queue.Queue() -> type=Queue
                        self.analysis.module_var_types[target.id] = func.attr
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Track with-statement context manager calls."""
        self._handle_context_manager(node)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Track async with-statement context manager calls."""
        self._handle_context_manager(node)
        self.generic_visit(node)

    def _handle_context_manager(self, node: ast.With | ast.AsyncWith) -> None:
        """Track __enter__ and __aenter__ calls in with statements."""
        if not self._current_func:
            return

        for item in node.items:
            if not item.context_expr or not isinstance(item.context_expr, ast.Call):
                continue

            ctx_call = item.context_expr
            if not isinstance(ctx_call.func, ast.Name):
                continue

            class_name = ctx_call.func.id
            # Check for __enter__ (sync context manager)
            enter_key = f"{class_name}.__enter__"
            if enter_key in self.analysis.functions:
                self._add_context_manager_call(ctx_call, enter_key, class_name)
            # Check for __aenter__ (async context manager)
            aenter_key = f"{class_name}.__aenter__"
            if aenter_key in self.analysis.functions:
                self._add_context_manager_call(ctx_call, aenter_key, class_name)

    def _add_context_manager_call(self, ctx_call: ast.Call, key: str, class_name: str) -> None:
        arg_names: list[str | None] = []
        for arg in ctx_call.args:
            if isinstance(arg, ast.Name):
                arg_names.append(arg.id)
            else:
                arg_names.append(None)
        enter_call = CallSite(
            name=key,
            line=ctx_call.lineno,
            col=ctx_call.col_offset,
            module=None,
            object_type=class_name,
            caller_function=self._current_func.name,
            arg_names=arg_names,
        )
        self._current_func.calls.append(enter_call)
        self.analysis.call_graph[self._current_func.name].append(key)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Track annotated assignments for type inference."""
        if node.target and node.value:
            self._track_assignment_type(node.target, node.value)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        """Track return statements for blocking function references."""
        if node.value and self._current_func:
            resolved = self._resolve_blocking_ref(node.value)
            if resolved:
                self._current_func.returns_blocking = resolved
        self.generic_visit(node)

    def _track_blocking_dict(self, node: ast.Assign) -> None:
        """Track module-level dict literals that map string keys to blocking functions."""
        if not isinstance(node.value, ast.Dict):
            return

        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue

            dict_name = target.id
            key_map: dict[str, tuple[str | None, str | None]] = {}

            for key, val in zip(node.value.keys, node.value.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue

                # Resolve the value to a (module, func_name) pair
                resolved = self._resolve_blocking_ref(val)
                if resolved:
                    key_map[key.value] = resolved

            if key_map:
                self.analysis.blocking_dicts[dict_name] = key_map

    def _resolve_blocking_ref(self, node: ast.expr) -> tuple[str | None, str | None] | None:
        """Resolve an AST expression to a (module, func_name) pair if it references a known module."""
        # module.attr pattern: time.sleep
        if isinstance(node, ast.Attribute):
            attr_name = node.attr
            if isinstance(node.value, ast.Name):
                base = node.value.id
                if base in self.analysis.imports:
                    imp = self.analysis.imports[base]
                    return (imp.module, attr_name)
                if base in self.analysis.import_froms:
                    imp = self.analysis.import_froms[base]
                    return (imp.module, attr_name)
        # Direct name: sleep (from-imported)
        if isinstance(node, ast.Name):
            ref = node.id
            if ref in self.analysis.import_froms:
                imp = self.analysis.import_froms[ref]
                return (imp.module, imp.name)
            if ref in self.analysis.imports:
                imp = self.analysis.imports[ref]
                return (imp.module, None)
        return None

    def _track_module_level_class_vars(self, class_name: str) -> None:
        """Track module-level variables of this class type (best effort)."""
        # This is called after the class is defined
        # We look for patterns like: cfg = Config() at module level
        # These were already tracked in visit_Assign if at module level
        pass

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Track attribute access for property detection."""
        if self._current_func and isinstance(node.value, ast.Name):
            var_name = node.value.id
            attr_name = node.attr

            # Check if this is a property access on a known class
            # First check function-local var types
            class_name = None
            if var_name in self._current_func.var_types:
                class_name = self._current_func.var_types[var_name]
            elif var_name in self.analysis.module_var_types:
                class_name = self.analysis.module_var_types[var_name]

            if class_name and class_name in self.analysis.class_properties:
                if attr_name in self.analysis.class_properties[class_name]:
                    prop_key = f"{class_name}.<prop:{attr_name}>"
                    if prop_key in self.analysis.functions:
                        prop_call = CallSite(
                            name=prop_key,
                            line=node.lineno,
                            col=node.col_offset,
                            module=None,
                            object_type=class_name,
                            caller_function=self._current_func.name,
                            arg_names=None,
                            is_property_access=True,
                        )
                        self._current_func.calls.append(prop_call)
                        self.analysis.call_graph[self._current_func.name].append(prop_key)

            # Also check for dunder call patterns: s(5) -> resolve s type to class, trace __call__
            # This is handled in visit_Call via _resolve_call_name

        self.generic_visit(node)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        """Track loop variable types."""
        if self._current_func and isinstance(node.target, ast.Name):
            # e.g., for item in some_list: - can't easily infer type
            pass
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Track loop variable types."""
        self.visit_For(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Track operator overloading: d * 5 -> Delay.__mul__."""
        if self._current_func and isinstance(node.left, ast.Name):
            var_name = node.left.id
            # Check if this variable has a known type with dunder methods
            class_name = None
            if var_name in self._current_func.var_types:
                class_name = self._current_func.var_types[var_name]
            elif var_name in self.analysis.module_var_types:
                class_name = self.analysis.module_var_types[var_name]

            if class_name:
                op_dunders = {
                    ast.Add: "__add__",
                    ast.Sub: "__sub__",
                    ast.Mult: "__mul__",
                    ast.Div: "__truediv__",
                    ast.FloorDiv: "__floordiv__",
                    ast.Mod: "__mod__",
                    ast.Pow: "__pow__",
                    ast.MatMult: "__matmul__",
                }
                dunder = op_dunders.get(type(node.op))
                if dunder:
                    dunder_key = f"{class_name}.{dunder}"
                    if dunder_key in self.analysis.functions:
                        binop_call = CallSite(
                            name=dunder_key,
                            line=node.lineno,
                            col=node.col_offset,
                            module=None,
                            object_type=class_name,
                            caller_function=self._current_func.name,
                            arg_names=None,
                        )
                        self._current_func.calls.append(binop_call)
                        self.analysis.call_graph[self._current_func.name].append(dunder_key)
        self.generic_visit(node)


def _post_process_decorator_replacements(analysis: ModuleAnalysis, tree: ast.Module) -> None:
    """Detect decorator patterns where a function is replaced by a wrapper.

    Handles two patterns:
    1. Simple decorator: @slow_decorator -> compute is replaced by wrapper
    2. Decorator factory: @retry(3) -> fetch is replaced by innermost wrapper
    """
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Check if this function is a decorator factory (defines and returns a nested func)
        # Supports both direct return and factory pattern (nested returns)
        wrapper_name = _find_wrapper_name(node)

        if not wrapper_name:
            continue

        # Now find all functions decorated with this decorator
        decorator_name = node.name
        for target in ast.iter_child_nodes(tree):
            if not isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if target.name == decorator_name:
                continue

            for dec in target.decorator_list:
                dec_name = None
                if isinstance(dec, ast.Name):
                    dec_name = dec.id
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    dec_name = dec.func.id

                if dec_name == decorator_name:
                    _apply_decorator_replacement(analysis, target.name, wrapper_name)


def _find_wrapper_name(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Find the name of the innermost wrapper function that's returned.

    Handles both simple decorators and decorator factories:
    - Simple: def dec(f): def wrapper(...): ...; return wrapper
    - Factory: def dec(n): def decorator(f): def wrapper(...): ...; return wrapper; return decorator
    """
    # Look for a returned function name
    for stmt in func_node.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
            returned_name = stmt.value.id
            # Check that this name is a function defined in this body
            for child in func_node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == returned_name:
                    # Check if this returned function itself returns another wrapper (factory pattern)
                    inner_wrapper = _find_wrapper_name(child)
                    if inner_wrapper:
                        return inner_wrapper
                    return returned_name
    return None


def _apply_decorator_replacement(analysis: ModuleAnalysis, target_name: str, wrapper_name: str) -> None:
    """Apply decorator replacement: copy wrapper's calls to the decorated function."""
    analysis.decorator_replacements[target_name] = wrapper_name

    if wrapper_name in analysis.functions and target_name in analysis.functions:
        target_func = analysis.functions[target_name]
        wrapper_func = analysis.functions[wrapper_name]
        for call in wrapper_func.calls:
            target_func.calls.append(call)


def analyze_file(filepath: str | Path) -> ModuleAnalysis:
    """Analyze a single Python file and return its call graph."""
    filepath = Path(filepath)
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))

    builder = CallGraphBuilder(str(filepath))
    builder.visit(tree)
    _post_process_decorator_replacements(builder.analysis, tree)

    return builder.analysis


def analyze_source(source: str, filepath: str = "<test>") -> ModuleAnalysis:
    """Analyze source code string."""
    tree = ast.parse(source, filename=filepath)
    builder = CallGraphBuilder(filepath)
    builder.visit(tree)
    _post_process_decorator_replacements(builder.analysis, tree)
    return builder.analysis
