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

from .semantic import (
    CallOperation,
    Confidence,
    ExprResult,
    FunctionSummary,
    KnownCallable,
    KnownInstance,
    KnownModule,
    SymbolId,
    UnknownValue,
    ValueInfo,
)


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
    arg_names: list[str | None] | None = field(default_factory=list)
    # Whether this is a property access (not a direct call)
    is_property_access: bool = False
    # Semantic confidence for this resolved operation.
    confidence: Confidence = Confidence.HIGH
    # True when the callable is deliberately offloaded from the event loop.
    offloaded: bool = False
    # True for the outer factory call in `async with factory() as value`.
    async_context_manager_factory: bool = False


@dataclass
class FuncDef:
    """A function definition found in the source."""
    name: str
    is_async: bool
    node: ast.AsyncFunctionDef | ast.FunctionDef
    # Stable qualified identity. `name` remains the compatibility lookup key.
    symbol: str | None = None
    calls: list[CallSite] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    # Variables tracked for type inference
    var_types: dict[str, str] = field(default_factory=dict)
    # Parameter names (for callback resolution)
    params: list[str] = field(default_factory=list)
    # Variable aliases: var_name -> (module, func_name) for blocking function refs
    var_aliases: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    # Source module for function-local vars constructed from `mod.Cls()`.
    # var_name -> module name (e.g. server = smtplib.SMTP() -> "server" -> "smtplib").
    var_module: dict[str, str] = field(default_factory=dict)
    # Whether this function is a decorator wrapper that replaces the original
    is_decorator_wrapper: bool = False
    # The original function name this wrapper replaces (set when used as decorator)
    replaces_func: str | None = None
    # Blocking reference returned by this function: (module, func_name) or None
    returns_blocking: tuple[str | None, str | None] | None = None
    # Inner functions defined in this function's body (set of names)
    inner_functions: set[str] = field(default_factory=set)
    # Best-effort return value summary for local function calls.
    return_type: str | None = None
    return_module: str | None = None
    return_alias: tuple[str | None, str | None] | None = None
    # Function-local bound method aliases: var_name -> qualified method key.
    bound_methods: dict[str, str] = field(default_factory=dict)
    # Function-local dict literals that map string keys to callables.
    blocking_dicts: dict[str, dict[str, tuple[str | None, str | None]]] = field(default_factory=dict)
    summary: FunctionSummary | None = None


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
    # Class inheritance: class_name -> [base_class_names]
    class_bases: dict[str, list[str]] = field(default_factory=dict)
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
    # Source module for module-level vars constructed from `mod.Cls()`.
    # var_name -> module name (e.g. q = queue.Queue() -> "q" -> "queue").
    module_var_module: dict[str, str] = field(default_factory=dict)
    # Class attributes that reference known functions: class_name -> {attr_name: func_name}
    class_func_attrs: dict[str, dict[str, str]] = field(default_factory=dict)
    # Class attributes that reference imported callables: class_name -> {attr_name -> (module, func_name)}
    class_callable_attrs: dict[str, dict[str, tuple[str | None, str | None]]] = field(default_factory=dict)
    # Instance attribute source modules: class_name -> {attr_name -> module}
    # e.g. self.conn = sqlite3.connect(...) -> {"UserRepository": {"conn": "sqlite3"}}
    class_attr_modules: dict[str, dict[str, str]] = field(default_factory=dict)
    # Instance attribute types: class_name -> {attr_name -> type_name}
    # e.g. self.conn = sqlite3.connect(...) -> {"UserRepository": {"conn": "connect"}}
    class_attr_types: dict[str, dict[str, str]] = field(default_factory=dict)
    # Module-level variable aliases: var_name -> (module, func_name)
    # e.g. query_runner = create_query_runner(...) -> {"query_runner": (None, "run_query")}
    module_var_aliases: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    # Stable symbol lookup for local functions/methods.
    symbols: dict[str, FuncDef] = field(default_factory=dict)
    # Backwards-compatible aliases from short names to qualified symbols.
    symbol_aliases: dict[str, str] = field(default_factory=dict)
    # Module-level bound method aliases: var_name -> qualified method key.
    module_bound_methods: dict[str, str] = field(default_factory=dict)


class CallGraphBuilder(ast.NodeVisitor):
    """AST visitor that builds a call graph with import resolution."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.analysis = ModuleAnalysis(filepath=filepath)
        self._current_func: FuncDef | None = None
        self._current_class: str | None = None
        self._scope_stack: list[FuncDef] = []
        self._async_context_factory_calls: set[int] = set()

    def _symbol_for(
        self,
        name: str,
        class_name: str | None = None,
        parent: FuncDef | None = None,
    ) -> str:
        """Return the stable local symbol for a definition."""
        if class_name:
            return SymbolId.class_member(class_name, name).name
        if parent:
            parent_symbol = parent.symbol or parent.name
            return SymbolId.nested(parent_symbol, name).name
        return SymbolId.module_function(name).name

    def _register_function(self, func: FuncDef, compatibility_name: str | None = None) -> None:
        """Register a function by stable symbol and compatibility aliases."""
        symbol = func.symbol or func.name
        func.symbol = symbol
        self.analysis.functions[symbol] = func
        self.analysis.symbols[symbol] = func
        if compatibility_name:
            self.analysis.symbol_aliases.setdefault(compatibility_name, symbol)
            self.analysis.functions.setdefault(compatibility_name, func)

    def _lookup_function(self, name: str | None) -> FuncDef | None:
        if not name:
            return None
        func = self.analysis.functions.get(name)
        if func is not None:
            return func
        alias = self.analysis.symbol_aliases.get(name)
        if alias:
            return self.analysis.functions.get(alias)
        return None

    def _local_method_key(self, class_name: str | None, method_name: str) -> str | None:
        if not class_name:
            return None
        key = SymbolId.class_member(class_name, method_name).name
        if key in self.analysis.functions:
            return key
        return None

    def precollect_definitions(self, tree: ast.Module) -> None:
        """Collect local symbol identities before analyzing executable bodies."""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = self._symbol_for(node.name)
                func = FuncDef(
                    name=node.name,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    node=node,
                    symbol=symbol,
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                    params=self._extract_params(node),
                )
                self._register_function(func, compatibility_name=node.name)
            elif isinstance(node, ast.ClassDef):
                self._precollect_class(node)

    def _precollect_class(self, node: ast.ClassDef) -> None:
        methods: set[str] = set()
        properties: set[str] = set()
        self.analysis.class_bases[node.name] = [
            base.id for base in node.bases if isinstance(base, ast.Name)
        ]
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            is_prop = any(
                self._get_decorator_name(d) == "property"
                for d in item.decorator_list
            )
            if is_prop:
                properties.add(item.name)
                symbol = SymbolId.class_member(node.name, f"<prop:{item.name}>").name
            else:
                methods.add(item.name)
                symbol = SymbolId.class_member(node.name, item.name).name
                if item.name.startswith("__") and item.name.endswith("__"):
                    self.analysis.dunder_map.setdefault(item.name, {})[node.name] = symbol
            func = FuncDef(
                name=symbol,
                is_async=isinstance(item, ast.AsyncFunctionDef),
                node=item,
                symbol=symbol,
                decorators=[self._get_decorator_name(d) for d in item.decorator_list],
                params=self._extract_params(item),
            )
            self._register_function(func)
        self.analysis.class_methods[node.name] = methods
        self.analysis.class_properties[node.name] = properties

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

    def _infer_bare_name_module(self, name: str) -> str | None:
        """For an unresolved bare name, see if any imported module exposes it.

        Returns the module name if exactly one import has the attribute,
        else None. Uses runtime introspection — no hardcoding.
        """
        from .introspect import attr_exists

        candidates: list[str] = []
        seen: set[str] = set()
        for imp in self.analysis.imports.values():
            if imp.module in seen:
                continue
            seen.add(imp.module)
            if attr_exists(imp.module, name):
                candidates.append(imp.module)
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _resolve_call_name(self, node: ast.Call) -> tuple[str | None, str | None, str | None]:
        """Resolve a call to (module, func_name, object_type).

        Returns (module, name, object_type) or (None, name, None) for unresolvable.
        """
        func = node.func

        if isinstance(func, ast.Name):
            # Simple function call: foo()
            name = func.id
            if self._lookup_function(name):
                return None, name, None
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
            if self._current_func and name in self._current_func.bound_methods:
                method_key = self._current_func.bound_methods[name]
                return None, method_key, method_key.rsplit(".", 1)[0]
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
            # Check module-level var aliases (closure/factory pattern)
            if name in self.analysis.module_var_aliases:
                alias_mod, alias_name = self.analysis.module_var_aliases[name]
                return alias_mod, alias_name, None
            if name in self.analysis.module_bound_methods:
                method_key = self.analysis.module_bound_methods[name]
                return None, method_key, method_key.rsplit(".", 1)[0]
            # Bare-name fallback: search imported modules via runtime introspection
            # for an exported callable matching this name. e.g. `import time` then
            # bare `sleep(5)` (semantically wrong but legal-looking) — we infer
            # `time.sleep` if exactly one imported module exposes a `sleep` attr.
            if name not in self.analysis.functions:
                inferred = self._infer_bare_name_module(name)
                if inferred is not None:
                    return inferred, name, None
            return None, name, None

        elif isinstance(func, ast.Attribute):
            # Method/module call: obj.method() or module.func()
            attr_name = func.attr
            value = func.value

            if isinstance(value, ast.Name):
                base_name = value.id
                if base_name in {"self", "cls"} and self._current_class:
                    local_method = self._local_method_key(self._current_class, attr_name)
                    if local_method:
                        return None, local_method, self._current_class
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
                    local_method = self._local_method_key(obj_type, attr_name)
                    if local_method:
                        return None, local_method, obj_type
                    # Check if this is a class function attribute
                    if obj_type in self.analysis.class_func_attrs:
                        if attr_name in self.analysis.class_func_attrs[obj_type]:
                            real_func = self.analysis.class_func_attrs[obj_type][attr_name]
                            return None, real_func, None
                    if obj_type in self.analysis.class_callable_attrs:
                        if attr_name in self.analysis.class_callable_attrs[obj_type]:
                            alias_mod, alias_name = self.analysis.class_callable_attrs[obj_type][attr_name]
                            return alias_mod, alias_name, None
                    # Propagate source module if tracked (e.g. smtplib.SMTP() -> "smtplib")
                    src_mod = self._current_func.var_module.get(base_name)
                    return src_mod, attr_name, obj_type
                # Check module-level var types
                if base_name in self.analysis.module_var_types:
                    obj_type = self.analysis.module_var_types[base_name]
                    local_method = self._local_method_key(obj_type, attr_name)
                    if local_method:
                        return None, local_method, obj_type
                    if obj_type in self.analysis.class_func_attrs:
                        if attr_name in self.analysis.class_func_attrs[obj_type]:
                            real_func = self.analysis.class_func_attrs[obj_type][attr_name]
                            return None, real_func, None
                    if obj_type in self.analysis.class_callable_attrs:
                        if attr_name in self.analysis.class_callable_attrs[obj_type]:
                            alias_mod, alias_name = self.analysis.class_callable_attrs[obj_type][attr_name]
                            return alias_mod, alias_name, None
                    # If we know the source module of this var (from
                    # `q = queue.Queue()` or `from queue import Queue`), propagate
                    # it so the classifier can verify the method on that class.
                    src_mod = self.analysis.module_var_module.get(base_name)
                    return src_mod, attr_name, obj_type
                # Fallback: use base_name as obj_type for blocking method detection.
                return None, attr_name, base_name

            elif isinstance(value, ast.Attribute):
                # Nested: a.b.method() or a.b.c.method()
                # Walk down to leftmost Name and reconstruct the dotted path.
                parts: list[str] = [value.attr]
                cur: ast.expr = value.value
                while isinstance(cur, ast.Attribute):
                    parts.insert(0, cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    base = cur.id

                    # Resolve self.xxx.method() and obj.xxx.method()
                    # where self.xxx was tracked via class_attr_modules.
                    if len(parts) == 1:
                        if base == "self" and self._current_class:
                            class_modules = self.analysis.class_attr_modules.get(
                                self._current_class, {}
                            )
                            class_types = self.analysis.class_attr_types.get(
                                self._current_class, {}
                            )
                            if parts[0] in class_modules:
                                module = class_modules[parts[0]]
                                obj_type = class_types.get(parts[0])
                                return module, attr_name, obj_type
                        elif (
                            self._current_func
                            and base in self._current_func.var_types
                        ):
                            obj_class = self._current_func.var_types[base]
                            class_modules = self.analysis.class_attr_modules.get(
                                obj_class, {}
                            )
                            class_types = self.analysis.class_attr_types.get(
                                obj_class, {}
                            )
                            if parts[0] in class_modules:
                                module = class_modules[parts[0]]
                                obj_type = class_types.get(parts[0])
                                return module, attr_name, obj_type

                    # Try matching the longest dotted-import prefix:
                    # `import urllib.request` registers key 'urllib.request',
                    # so for urllib.request.urlopen we want module='urllib.request'.
                    candidate_keys = []
                    if base in self.analysis.imports:
                        candidate_keys.append((base, []))
                    # Also consider keys like 'urllib.request' that start with this base.
                    for key in self.analysis.imports:
                        if key.split(".")[0] == base and "." in key:
                            candidate_keys.append((key, key.split(".")[1:]))
                    # Prefer the longest matching prefix that aligns with parts.
                    best: tuple[str, list[str]] | None = None
                    for key, key_tail in candidate_keys:
                        if key_tail == parts[: len(key_tail)]:
                            if best is None or len(key_tail) > len(best[1]):
                                best = (key, key_tail)
                    if best is not None:
                        key, key_tail = best
                        imp = self.analysis.imports[key]
                        remaining = parts[len(key_tail) :]
                        if remaining:
                            mod_path = imp.module + "." + ".".join(remaining)
                        else:
                            mod_path = imp.module
                        return mod_path, attr_name, None
                return None, attr_name, None

            elif isinstance(value, ast.Call):
                if (
                    isinstance(value.func, ast.Name)
                    and value.func.id == "super"
                    and self._current_class
                ):
                    for base_class in self.analysis.class_bases.get(self._current_class, []):
                        method_key = self._local_method_key(base_class, attr_name)
                        if method_key:
                            return None, method_key, base_class
                # Chained: SomeClass().method() or module.Func().method()
                # Try to resolve the call target
                inner = self._resolve_call_name(value)
                if inner and inner[1]:
                    # Propagate the module unconditionally: if the inner call has
                    # a module, the chained method is called on its return value —
                    # preserve module+class so the classifier can introspect.
                    # e.g., requests.Session().get() -> (requests, get, Session)
                    return inner[0], attr_name, inner[1]

            return None, attr_name, None

        # Handle inline getattr call: getattr(module, 'func')()
        if isinstance(func, ast.Call):
            getattr_result = self._resolve_getattr_call(func)
            if getattr_result:
                return getattr_result[0], getattr_result[1], None

        return None, None, None

    def _track_assignment_type(self, target: ast.expr, value: ast.expr) -> None:
        """Track variable types from assignments for method call resolution."""
        if not self._current_func:
            return

        if isinstance(target, ast.Name):
            self._apply_value_to_name(target.id, self._infer_expr_value(value))

            # Track: var = SomeClass()
            if isinstance(value, ast.Call):
                func = value.func
                if isinstance(func, ast.Name):
                    name = func.id
                    if name in self.analysis.import_froms:
                        self._current_func.var_types[target.id] = self.analysis.import_froms[name].name
                    elif name in self.analysis.imports:
                        self._current_func.var_types[target.id] = name
                    elif name in self._current_func.var_aliases:
                        # Variable alias from getattr or similar: cls = getattr(smtplib, 'SMTP')
                        alias_mod, alias_name = self._current_func.var_aliases[name]
                        if alias_name:
                            self._current_func.var_types[target.id] = alias_name
                        else:
                            self._current_func.var_types[target.id] = name
                        if alias_mod:
                            self._current_func.var_module[target.id] = alias_mod
                    else:
                        self._current_func.var_types[target.id] = name

                    # Check if the called function returns a blocking reference
                    callee = self._lookup_function(name)
                    if callee and callee.returns_blocking:
                        self._current_func.var_aliases[target.id] = callee.returns_blocking
                    if callee and callee.return_type:
                        self._current_func.var_types[target.id] = callee.return_type
                        if callee.return_module:
                            self._current_func.var_module[target.id] = callee.return_module

                elif isinstance(func, ast.Attribute):
                    self._current_func.var_types[target.id] = func.attr
                    # Track the source module so method calls on this variable
                    # can be resolved back to the module for classification.
                    # e.g., server = smtplib.SMTP() -> var_module["server"] = "smtplib"
                    if isinstance(func.value, ast.Name):
                        base = func.value.id
                        if base in self.analysis.imports:
                            self._current_func.var_module[target.id] = (
                                self.analysis.imports[base].module
                            )
                        elif base in self.analysis.import_froms:
                            self._current_func.var_module[target.id] = (
                                self.analysis.import_froms[base].module
                            )
                        elif base in self._current_func.var_module:
                            # Propagate from tracked local variable
                            # e.g., cursor = conn.cursor() where conn has module "sqlite3"
                            self._current_func.var_module[target.id] = (
                                self._current_func.var_module[base]
                            )
                    # Check if the method returns a blocking reference (closure/factory)
                    callee = self._lookup_function(func.attr)
                    if callee and callee.returns_blocking:
                        self._current_func.var_aliases[target.id] = callee.returns_blocking
                    if callee and callee.return_type:
                        self._current_func.var_types[target.id] = callee.return_type
                        if callee.return_module:
                            self._current_func.var_module[target.id] = callee.return_module

            # Handle variable reassignment: propagate type/module from source
            # e.g., conn = server  ->  copy server's type/module to conn
            elif isinstance(value, ast.Name):
                src = value.id
                if src in self._current_func.var_types:
                    self._current_func.var_types[target.id] = (
                        self._current_func.var_types[src]
                    )
                if src in self._current_func.var_module:
                    self._current_func.var_module[target.id] = (
                        self._current_func.var_module[src]
                    )
                if src in self._current_func.var_aliases:
                    self._current_func.var_aliases[target.id] = (
                        self._current_func.var_aliases[src]
                    )

            # Also propagate from module-level vars for reassignment
            if isinstance(value, ast.Name):
                src = value.id
                if src in self.analysis.module_var_types and target.id not in self._current_func.var_types:
                    self._current_func.var_types[target.id] = (
                        self.analysis.module_var_types[src]
                    )
                if src in self.analysis.module_var_module and target.id not in self._current_func.var_module:
                    self._current_func.var_module[target.id] = (
                        self.analysis.module_var_module[src]
                    )

            # Track variable aliasing to blocking functions
            # e.g. f = time.sleep  or  f = requests.get
            self._track_var_alias(target.id, value)

        # Track: self.xxx = Call() or obj.xxx = Call()
        # Stores the source module per class so self.xxx.method() can resolve.
        if isinstance(target, ast.Attribute) and isinstance(value, ast.Call):
            attr_name = target.attr
            obj = target.value

            if isinstance(obj, ast.Name):
                obj_name = obj.id
                class_name = None

                if obj_name == "self" and self._current_class:
                    class_name = self._current_class
                elif obj_name in self._current_func.var_types:
                    class_name = self._current_func.var_types[obj_name]

                if class_name:
                    module, func_name, _ = self._resolve_call_name(value)
                    if module:
                        self.analysis.class_attr_modules.setdefault(class_name, {})[attr_name] = module
                    if func_name:
                        self.analysis.class_attr_types.setdefault(class_name, {})[attr_name] = func_name

    def _infer_expr_value(self, value: ast.expr) -> ValueInfo:
        """Infer a reusable semantic value from an expression."""
        if isinstance(value, ast.Await):
            return self._infer_expr_value(value.value)

        if isinstance(value, ast.Name):
            name = value.id
            if self._current_func:
                if name in self._current_func.bound_methods:
                    symbol = self._current_func.bound_methods[name]
                    return KnownCallable(symbol=symbol, name=symbol)
                if name in self._current_func.var_aliases:
                    module, func_name = self._current_func.var_aliases[name]
                    return KnownCallable(module=module, name=func_name)
                if name in self._current_func.var_types:
                    return KnownInstance(
                        module=self._current_func.var_module.get(name),
                        type_name=self._current_func.var_types[name],
                    )
            if name in self.analysis.module_bound_methods:
                symbol = self.analysis.module_bound_methods[name]
                return KnownCallable(symbol=symbol, name=symbol)
            if name in self.analysis.module_var_aliases:
                module, func_name = self.analysis.module_var_aliases[name]
                return KnownCallable(module=module, name=func_name)
            if name in self.analysis.module_var_types:
                return KnownInstance(
                    module=self.analysis.module_var_module.get(name),
                    type_name=self.analysis.module_var_types[name],
                )
            if name in self.analysis.imports:
                return KnownModule(module=self.analysis.imports[name].module)
            if name in self.analysis.import_froms:
                imp = self.analysis.import_froms[name]
                return KnownCallable(module=imp.module, name=imp.name)
            return UnknownValue(reason=f"unresolved name {name}")

        if isinstance(value, ast.Attribute):
            method = self._resolve_bound_method(value)
            if method:
                return KnownCallable(symbol=method, name=method)
            ref = self._resolve_blocking_ref(value)
            if ref:
                return KnownCallable(module=ref[0], name=ref[1])
            return UnknownValue(reason="unresolved attribute")

        if isinstance(value, ast.Call):
            module, func_name, obj_type = self._resolve_call_name(value)
            if module and func_name:
                return KnownInstance(module=module, type_name=func_name)
            if module and obj_type:
                return KnownInstance(module=module, type_name=obj_type)
            callee = self._lookup_function(func_name)
            if callee:
                if callee.return_alias:
                    return KnownCallable(module=callee.return_alias[0], name=callee.return_alias[1])
                if callee.return_type:
                    return KnownInstance(module=callee.return_module, type_name=callee.return_type)
            if func_name and obj_type:
                return KnownInstance(module=module, type_name=obj_type)
            if func_name:
                return KnownInstance(module=module, type_name=func_name)

        return UnknownValue()

    def _apply_value_to_name(self, name: str, value: ValueInfo) -> None:
        """Bind a semantic value to a local name."""
        if not self._current_func:
            return
        if isinstance(value, KnownInstance) and value.type_name:
            self._current_func.var_types[name] = value.type_name
            if value.module:
                self._current_func.var_module[name] = value.module
        elif isinstance(value, KnownCallable):
            if value.symbol:
                self._current_func.bound_methods[name] = value.symbol
            elif value.module or value.name:
                self._current_func.var_aliases[name] = (value.module, value.name)

    def _resolve_bound_method(self, value: ast.Attribute) -> str | None:
        """Resolve `obj.method` to a qualified local method symbol when possible."""
        if not isinstance(value.value, ast.Name):
            return None
        base = value.value.id
        class_name = None
        if self._current_func and base in self._current_func.var_types:
            class_name = self._current_func.var_types[base]
        elif base in self.analysis.module_var_types:
            class_name = self.analysis.module_var_types[base]
        if not class_name:
            return None
        return self._local_method_key(class_name, value.attr)

    def _resolve_getattr_call(self, node: ast.Call) -> tuple[str | None, str | None] | None:
        """Resolve getattr(module_or_var, 'name') to (module, func_name).

        Handles both:
        - getattr(imported_module, 'func_name')  -> (module, func_name)
        - getattr(local_var, 'func_name')         -> resolves var to module
        - getattr(module, string_var)              -> (module, None) as conservative
        Returns None if the pattern doesn't match.
        """
        if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
            return None
        if len(node.args) < 2:
            return None

        obj_arg, name_arg = node.args[0], node.args[1]

        # Resolve the module from the first argument
        module: str | None = None
        if isinstance(obj_arg, ast.Name):
            base = obj_arg.id
            if base in self.analysis.imports:
                module = self.analysis.imports[base].module
            elif base in self.analysis.import_froms:
                module = self.analysis.import_froms[base].module
            elif self._current_func and base in self._current_func.var_module:
                module = self._current_func.var_module[base]
            elif self._current_func and base in self._current_func.var_aliases:
                # Variable assigned an imported module: mod = time
                alias_mod, alias_name = self._current_func.var_aliases[base]
                if alias_mod and alias_name is None:
                    # mod = time  -> var_aliases["mod"] = ("time", None)
                    module = alias_mod
        elif isinstance(obj_arg, ast.Attribute):
            # getattr(urllib.request, 'urlopen') - dotted module
            resolved = self._resolve_dotted_module(obj_arg)
            if resolved:
                module = resolved

        if module is None:
            return None

        # Resolve the attribute name from the second argument
        if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
            return (module, name_arg.value)

        # Variable name arg (e.g., getattr(time, method_name)) —
        # try to resolve the variable to a string literal if it was assigned earlier.
        if isinstance(name_arg, ast.Name) and self._current_func:
            resolved_name = self._try_resolve_string_var(name_arg.id)
            if resolved_name:
                return (module, resolved_name)

        # Dynamic second arg (e.g., method.lower(), computed expr) —
        # we know the module but not the exact function. Flag with a generic
        # marker so the classifier can warn about potential blocking getattr.
        return (module, "<getattr>")

    def _try_resolve_string_var(self, var_name: str) -> str | None:
        """Try to resolve a variable to a string literal by scanning assignments.

        Looks through the current function's AST body for simple assignments like
        `method_name = 'sleep'` and returns the string value if found.
        """
        if not self._current_func or not self._current_func.node:
            return None
        for stmt in ast.walk(self._current_func.node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == var_name
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)
                    ):
                        return stmt.value.value
        return None

    def _resolve_dotted_module(self, node: ast.Attribute) -> str | None:
        """Resolve a dotted attribute chain to a module path.

        E.g., urllib.request -> 'urllib.request' if imported.
        """
        parts: list[str] = [node.attr]
        cur: ast.expr = node.value
        while isinstance(cur, ast.Attribute):
            parts.insert(0, cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            base = cur.id
            # Build candidate keys to match against imports
            candidate_keys = []
            if base in self.analysis.imports:
                candidate_keys.append((base, []))
            for key in self.analysis.imports:
                if key.split(".")[0] == base and "." in key:
                    candidate_keys.append((key, key.split(".")[1:]))
            best: tuple[str, list[str]] | None = None
            for key, key_tail in candidate_keys:
                if key_tail == parts[: len(key_tail)]:
                    if best is None or len(key_tail) > len(best[1]):
                        best = (key, key_tail)
            if best is not None:
                key, key_tail = best
                imp = self.analysis.imports[key]
                remaining = parts[len(key_tail) :]
                if remaining:
                    return imp.module + "." + ".".join(remaining)
                return imp.module
        return None

    def _track_var_alias(self, var_name: str, value: ast.expr) -> None:
        """Track when a variable is assigned a reference to a blocking function."""
        if not self._current_func:
            return

        # getattr(module, 'func_name') -> alias to (module, func_name)
        if isinstance(value, ast.Call):
            getattr_result = self._resolve_getattr_call(value)
            if getattr_result:
                self._current_func.var_aliases[var_name] = getattr_result
                # Also track var_module for constructor patterns:
                # cls = getattr(smtplib, 'SMTP'); server = cls(...)
                mod, name = getattr_result
                if name:
                    self._current_func.var_module[var_name] = mod
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

        # Generic callable wrapper: any ast.Call whose first positional arg
        # resolves to a (module, func_name) reference.
        # Covers functools.partial, custom wrappers, etc.
        if isinstance(value, ast.Call):
            if value.args:
                first_arg = value.args[0]
                resolved = self._resolve_blocking_ref(first_arg)
                if resolved:
                    self._current_func.var_aliases[var_name] = resolved
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

    def _visit_function_body(self, node: ast.AsyncFunctionDef | ast.FunctionDef) -> None:
        """Visit function args + body but NOT decorator_list.

        Decorators are part of the surrounding scope, not the function body —
        recording them as calls inside the function would attribute decorator
        side effects (e.g. `@app.get("/x")`) to the function being decorated.
        """
        for arg_default in node.args.defaults:
            self.visit(arg_default)
        for arg_default in node.args.kw_defaults:
            if arg_default is not None:
                self.visit(arg_default)
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        parent = self._current_func
        class_name = self._current_class
        symbol = self._symbol_for(node.name, class_name=class_name, parent=parent if not class_name else None)
        func = FuncDef(
            name=node.name,
            is_async=True,
            node=node,
            symbol=symbol,
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            params=self._extract_params(node),
        )
        from .endpoints import is_async_endpoint as _is_async_endpoint
        is_endpoint = _is_async_endpoint(node, node.decorator_list, self.analysis)

        prev = self._current_func
        self._current_func = func
        self._scope_stack.append(func)

        self._visit_function_body(node)

        self._scope_stack.pop()
        self._current_func = prev

        self._register_function(
            func,
            compatibility_name=node.name if not class_name and not parent else None,
        )

        if is_endpoint:
            func.decorators.append("__fastapi_endpoint__")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit sync function definitions (helpers called from async)."""
        parent = self._current_func
        class_name = self._current_class
        symbol = self._symbol_for(node.name, class_name=class_name, parent=parent if not class_name else None)
        func = FuncDef(
            name=node.name,
            is_async=False,
            node=node,
            symbol=symbol,
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            params=self._extract_params(node),
        )

        prev = self._current_func
        self._current_func = func
        self._scope_stack.append(func)

        self._visit_function_body(node)

        self._scope_stack.pop()
        self._current_func = prev

        # Store function - but check if a decorator wrapper already took this name
        self._register_function(
            func,
            compatibility_name=node.name if not class_name and not parent else None,
        )

        # Register this function as an inner function of its parent scope
        if self._current_func:
            self._current_func.inner_functions.add(symbol)
            self._current_func.is_decorator_wrapper = True

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class definitions and their methods."""
        prev_class = self._current_class
        self._current_class = node.name
        self.analysis.class_bases[node.name] = [
            base.id for base in node.bases if isinstance(base, ast.Name)
        ]

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
                    prop_key = SymbolId.class_member(node.name, f"<prop:{item.name}>").name
                    prop_func = FuncDef(
                        name=prop_key,
                        is_async=isinstance(item, ast.AsyncFunctionDef),
                        node=item,
                        symbol=prop_key,
                        decorators=[self._get_decorator_name(d) for d in item.decorator_list],
                        params=self._extract_params(item),
                    )
                    prev = self._current_func
                    self._current_func = prop_func
                    self._scope_stack.append(prop_func)
                    self._visit_function_body(item)
                    self._scope_stack.pop()
                    self._current_func = prev
                    self._register_function(prop_func)
                else:
                    methods.add(item.name)
                    # Map dunder methods: e.g. __call__ -> {ClassName: __call___ClassName}
                    if item.name.startswith("__") and item.name.endswith("__"):
                        if item.name not in self.analysis.dunder_map:
                            self.analysis.dunder_map[item.name] = {}
                        # Use a synthetic name to avoid collisions
                        dunder_key = SymbolId.class_member(node.name, item.name).name
                        self.analysis.dunder_map[item.name][node.name] = dunder_key

                        # Also add the dunder method as a regular function so the checker can trace into it
                        dunder_func = FuncDef(
                            name=dunder_key,
                            is_async=isinstance(item, ast.AsyncFunctionDef),
                            node=item,
                            symbol=dunder_key,
                            decorators=[self._get_decorator_name(d) for d in item.decorator_list],
                            params=self._extract_params(item),
                        )
                        # Visit the dunder body
                        prev = self._current_func
                        self._current_func = dunder_func
                        self._scope_stack.append(dunder_func)
                        self._visit_function_body(item)
                        self._scope_stack.pop()
                        self._current_func = prev

                        self._register_function(dunder_func)
                    else:
                        # Non-dunder, non-property methods - still visit their bodies
                        prev = self._current_func
                        method_key = SymbolId.class_member(node.name, item.name).name
                        method_func = FuncDef(
                            name=method_key,
                            is_async=isinstance(item, ast.AsyncFunctionDef),
                            node=item,
                            symbol=method_key,
                            decorators=[self._get_decorator_name(d) for d in item.decorator_list],
                            params=self._extract_params(item),
                        )
                        self._current_func = method_func
                        self._scope_stack.append(method_func)
                        self._visit_function_body(item)
                        self._scope_stack.pop()
                        self._current_func = prev
                        self._register_function(method_func)

        self.analysis.class_methods[node.name] = methods
        self.analysis.class_properties[node.name] = properties

        # Track class-level attributes that reference known functions
        func_attrs: dict[str, str] = {}
        callable_attrs: dict[str, tuple[str | None, str | None]] = {}
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and isinstance(item.value, ast.Name):
                        # Class attr referencing a function: action = wait
                        ref = item.value.id
                        if ref in self.analysis.functions:
                            func_attrs[target.id] = ref
                        elif ref in self.analysis.import_froms:
                            imp = self.analysis.import_froms[ref]
                            callable_attrs[target.id] = (imp.module, imp.name)
                    elif isinstance(target, ast.Name) and isinstance(item.value, ast.Call):
                        if isinstance(item.value.func, ast.Name):
                            self.analysis.class_attr_types.setdefault(node.name, {})[
                                target.id
                            ] = item.value.func.id
                    elif isinstance(target, ast.Name) and isinstance(item.value, ast.Attribute):
                        resolved = self._resolve_blocking_ref(item.value)
                        if resolved:
                            callable_attrs[target.id] = resolved
        if func_attrs:
            self.analysis.class_func_attrs[node.name] = func_attrs
        if callable_attrs:
            self.analysis.class_callable_attrs[node.name] = callable_attrs

        # Visit the class body for nested items (but don't re-visit methods)
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(item)

        self._current_class = prev_class


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
            elif isinstance(arg, ast.Attribute):
                arg_names.append(self._resolve_bound_method(arg))
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
            async_context_manager_factory=id(node) in self._async_context_factory_calls,
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

        # Generic higher-order call: for any call, if a positional ast.Name arg
        # refers to a known function or imported name, add a synthetic CallSite
        # so the checker traces into it (covers map, filter, executor.submit, etc.).
        # - For local functions: only add when the call target itself is NOT a
        #   local function (builtins like map/filter won't trigger _trace_callback_args).
        # - For imported names: always add (they are never reached via arg_names).
        caller_is_local = (
            isinstance(node.func, ast.Name)
            and self._lookup_function(node.func.id) is not None
        )
        offloaded_positions = self._offloaded_callback_positions(node, module, func_name)
        for index, hof_arg in enumerate(node.args):
            if index in offloaded_positions:
                continue
            if isinstance(hof_arg, ast.Attribute):
                bound_method = self._resolve_bound_method(hof_arg)
                if bound_method and not caller_is_local:
                    hof_call = CallSite(
                        name=bound_method,
                        line=node.lineno,
                        col=node.col_offset,
                        module=None,
                        object_type=bound_method.rsplit(".", 1)[0],
                        caller_function=self._current_func.name,
                        arg_names=None,
                    )
                    self._current_func.calls.append(hof_call)
                    self.analysis.call_graph[self._current_func.name].append(bound_method)
                continue
            if not isinstance(hof_arg, ast.Name):
                continue
            hof_name = hof_arg.id
            if hof_name in self.analysis.import_froms:
                imp = self.analysis.import_froms[hof_name]
                hof_call = CallSite(
                    name=imp.name,
                    line=node.lineno,
                    col=node.col_offset,
                    module=imp.module,
                    object_type=None,
                    caller_function=self._current_func.name,
                    arg_names=None,
                )
                self._current_func.calls.append(hof_call)
                self.analysis.call_graph[self._current_func.name].append(imp.name)
            elif self._lookup_function(hof_name) is not None and not caller_is_local:
                # Local function passed to a non-local caller (e.g. map, list, etc.)
                hof_call = CallSite(
                    name=hof_name,
                    line=node.lineno,
                    col=node.col_offset,
                    module=None,
                    object_type=None,
                    caller_function=self._current_func.name,
                    arg_names=None,
                )
                self._current_func.calls.append(hof_call)
                self.analysis.call_graph[self._current_func.name].append(hof_name)

        # Generic higher-order keyword args: catch sorted(items, key=fn),
        # min(items, key=fn), max(items, key=fn) etc., where the callable is
        # passed as a keyword. Same shape as the positional case above.
        for kw in node.keywords:
            if kw.arg in {"func"} and 0 in offloaded_positions:
                continue
            if isinstance(kw.value, ast.Attribute):
                bound_method = self._resolve_bound_method(kw.value)
                if bound_method and not caller_is_local:
                    self._current_func.calls.append(
                        CallSite(
                            name=bound_method,
                            line=node.lineno,
                            col=node.col_offset,
                            module=None,
                            object_type=bound_method.rsplit(".", 1)[0],
                            caller_function=self._current_func.name,
                            arg_names=None,
                        )
                    )
                    self.analysis.call_graph[self._current_func.name].append(bound_method)
                continue
            if not isinstance(kw.value, ast.Name):
                continue
            kw_name = kw.value.id
            if kw_name in self.analysis.import_froms:
                imp = self.analysis.import_froms[kw_name]
                self._current_func.calls.append(
                    CallSite(
                        name=imp.name,
                        line=node.lineno,
                        col=node.col_offset,
                        module=imp.module,
                        object_type=None,
                        caller_function=self._current_func.name,
                        arg_names=None,
                    )
                )
                self.analysis.call_graph[self._current_func.name].append(imp.name)
            elif self._lookup_function(kw_name) is not None and not caller_is_local:
                self._current_func.calls.append(
                    CallSite(
                        name=kw_name,
                        line=node.lineno,
                        col=node.col_offset,
                        module=None,
                        object_type=None,
                        caller_function=self._current_func.name,
                        arg_names=None,
                    )
                )
                self.analysis.call_graph[self._current_func.name].append(kw_name)

        self.generic_visit(node)

    def _offloaded_callback_positions(
        self,
        node: ast.Call,
        module: str | None,
        func_name: str | None,
    ) -> set[int]:
        """Return callback argument indexes that run off the event loop."""
        if module == "asyncio" and func_name == "to_thread":
            return {0}
        if func_name == "run_in_executor":
            return {1}
        if isinstance(node.func, ast.Attribute) and node.func.attr == "submit":
            # concurrent.futures executors run callbacks in worker threads/processes.
            return {0}
        return set()

    def _resolve_dict_dispatch_call(self, node: ast.Call) -> tuple[str | None, str | None] | None:
        """Resolve dict dispatch calls like HANDLERS[\"wait\"](5)."""
        dict_name: str | None = None
        key = None

        # HANDLERS["wait"]()
        if isinstance(node.func, ast.Subscript) and isinstance(node.func.value, ast.Name):
            dict_name = node.func.value.id
            if isinstance(node.func.slice, ast.Constant):
                key = node.func.slice.value

        # HANDLERS.get("wait")()
        elif (
            isinstance(node.func, ast.Call)
            and isinstance(node.func.func, ast.Attribute)
            and node.func.func.attr == "get"
            and isinstance(node.func.func.value, ast.Name)
        ):
            dict_name = node.func.func.value.id
            if node.func.args and isinstance(node.func.args[0], ast.Constant):
                key = node.func.args[0].value

        if not dict_name:
            return None

        dicts: list[dict[str, tuple[str | None, str | None]]] = []
        if self._current_func and dict_name in self._current_func.blocking_dicts:
            dicts.append(self._current_func.blocking_dicts[dict_name])
        if dict_name in self.analysis.blocking_dicts:
            dicts.append(self.analysis.blocking_dicts[dict_name])

        for key_map in dicts:
            if key and isinstance(key, str) and key in key_map:
                return key_map[key]
            if key_map:
                return next(iter(key_map.values()))

        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track variable assignments for type inference and aliasing."""
        for target in node.targets:
            self._track_assignment_type(target, node.value)
        self._track_blocking_dict(node)
        # Track module-level assignments
        if self._current_func is None:
            # Track module-level variable types
            for target in node.targets:
                if isinstance(target, ast.Name):
                    inferred = self._infer_expr_value(node.value)
                    if isinstance(inferred, KnownInstance) and inferred.type_name:
                        self.analysis.module_var_types[target.id] = inferred.type_name
                        if inferred.module:
                            self.analysis.module_var_module[target.id] = inferred.module
                    elif isinstance(inferred, KnownCallable):
                        if inferred.symbol:
                            self.analysis.module_bound_methods[target.id] = inferred.symbol
                        elif inferred.module or inferred.name:
                            self.analysis.module_var_aliases[target.id] = (
                                inferred.module,
                                inferred.name,
                            )
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Name):
                        self.analysis.module_var_types[target.id] = func.id
                        # If the called name was from-imported, remember its module
                        # so a method call on `target` can resolve to that module.
                        if func.id in self.analysis.import_froms:
                            self.analysis.module_var_module[target.id] = (
                                self.analysis.import_froms[func.id].module
                            )
                        # Check if the called function returns a blocking reference
                        # (closure/factory pattern at module level)
                        callee = self._lookup_function(func.id)
                        if callee and callee.returns_blocking:
                            self.analysis.module_var_aliases[target.id] = callee.returns_blocking
                        if callee and callee.return_type:
                            self.analysis.module_var_types[target.id] = callee.return_type
                            if callee.return_module:
                                self.analysis.module_var_module[target.id] = callee.return_module
                    elif isinstance(func, ast.Attribute):
                        # e.g., q = queue.Queue() -> type=Queue, source module=queue
                        self.analysis.module_var_types[target.id] = func.attr
                        if isinstance(func.value, ast.Name):
                            base = func.value.id
                            if base in self.analysis.imports:
                                self.analysis.module_var_module[target.id] = (
                                    self.analysis.imports[base].module
                                )
                        # Check if the method returns a blocking reference
                        callee = self._lookup_function(func.attr)
                        if callee and callee.returns_blocking:
                            self.analysis.module_var_aliases[target.id] = callee.returns_blocking
                        if callee and callee.return_type:
                            self.analysis.module_var_types[target.id] = callee.return_type
                            if callee.return_module:
                                self.analysis.module_var_module[target.id] = callee.return_module
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Track with-statement context manager calls."""
        self._handle_context_manager(node)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Track async with-statement context manager calls."""
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                self._async_context_factory_calls.add(id(item.context_expr))
        self._handle_context_manager(node)
        self.generic_visit(node)

    def _handle_context_manager(self, node: ast.With | ast.AsyncWith) -> None:
        """Track __enter__/__aenter__ calls and as-variable type in with statements."""
        if not self._current_func:
            return

        for item in node.items:
            ctx_expr = item.context_expr

            # --- Case A: context expression is a Call (most common) ---
            if isinstance(ctx_expr, ast.Call):
                class_name: str | None = None
                source_module: str | None = None

                if isinstance(ctx_expr.func, ast.Name):
                    class_name = ctx_expr.func.id
                    if class_name == "open":
                        class_name = "TextIOWrapper"
                        source_module = "io"
                elif isinstance(ctx_expr.func, ast.Attribute):
                    class_name = ctx_expr.func.attr
                    # Resolve source module from module.Class pattern
                    if isinstance(ctx_expr.func.value, ast.Name):
                        base = ctx_expr.func.value.id
                        if base in self.analysis.imports:
                            source_module = self.analysis.imports[base].module
                        elif base in self.analysis.import_froms:
                            source_module = self.analysis.import_froms[base].module

                if class_name:
                    # Track the as-variable type and source module
                    self._track_with_var(item, class_name, source_module)

                    self._add_context_manager_dunders(ctx_expr, class_name, is_async=isinstance(node, ast.AsyncWith))

            # --- Case B: context expression is a plain variable (with zf as archive:) ---
            elif isinstance(ctx_expr, ast.Name):
                var_name = ctx_expr.id
                # Look up the variable's type from var_types or module_var_types
                class_name = None
                source_module = None
                if var_name in self._current_func.var_types:
                    class_name = self._current_func.var_types[var_name]
                    source_module = self._current_func.var_module.get(var_name)
                elif var_name in self.analysis.module_var_types:
                    class_name = self.analysis.module_var_types[var_name]
                    source_module = self.analysis.module_var_module.get(var_name)
                if class_name:
                    self._track_with_var(item, class_name, source_module)
                    self._add_context_manager_dunders(ctx_expr, class_name, is_async=isinstance(node, ast.AsyncWith))

    def _track_with_var(
        self,
        item: ast.withitem,
        class_name: str,
        source_module: str | None,
    ) -> None:
        """Store the type (and optionally module) of the `as` variable in a with statement."""
        if not self._current_func:
            return
        if item.optional_vars and isinstance(item.optional_vars, ast.Name):
            var_name = item.optional_vars.id
            self._current_func.var_types[var_name] = class_name
            if source_module:
                self._current_func.var_module[var_name] = source_module

    def _add_context_manager_call(self, ctx_call: ast.Call, key: str, class_name: str) -> None:
        self._add_context_manager_call_from_expr(ctx_call, key, class_name)

    def _add_context_manager_dunders(
        self,
        ctx_expr: ast.expr,
        class_name: str,
        is_async: bool,
    ) -> None:
        names = ("__aenter__", "__aexit__") if is_async else ("__enter__", "__exit__")
        for dunder_name in names:
            key = f"{class_name}.{dunder_name}"
            if key in self.analysis.functions:
                self._add_context_manager_call_from_expr(ctx_expr, key, class_name)

    def _add_context_manager_call_from_expr(
        self,
        ctx_expr: ast.expr,
        key: str,
        class_name: str,
    ) -> None:
        arg_names: list[str | None] = []
        if isinstance(ctx_expr, ast.Call):
            for arg in ctx_expr.args:
                if isinstance(arg, ast.Name):
                    arg_names.append(arg.id)
                else:
                    arg_names.append(None)
        enter_call = CallSite(
            name=key,
            line=ctx_expr.lineno,
            col=ctx_expr.col_offset,
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
            inferred = self._infer_expr_value(node.value)
            if isinstance(inferred, KnownInstance):
                self._current_func.return_type = inferred.type_name
                self._current_func.return_module = inferred.module
            elif isinstance(inferred, KnownCallable):
                if inferred.symbol:
                    self._current_func.return_alias = (None, inferred.symbol)
                else:
                    self._current_func.return_alias = (inferred.module, inferred.name)

            resolved = self._resolve_blocking_ref(node.value)
            if resolved:
                self._current_func.returns_blocking = resolved
                self._current_func.return_alias = resolved
            elif isinstance(node.value, ast.Name) and (
                node.value.id in self._current_func.inner_functions
                or any(sym.endswith(f".<locals>.{node.value.id}") for sym in self._current_func.inner_functions)
            ):
                # Returning a locally-defined inner function (closure/factory pattern)
                inner_symbol = next(
                    (
                        sym
                        for sym in self._current_func.inner_functions
                        if sym.endswith(f".<locals>.{node.value.id}")
                    ),
                    node.value.id,
                )
                self._current_func.returns_blocking = (None, inner_symbol)
                self._current_func.return_alias = (None, inner_symbol)
            elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                # Returning a call to a locally-defined inner function:
                #   return inner_func(args)
                # The inner function may itself return a closure.
                callee_name = node.value.func.id
                inner_symbol = next(
                    (
                        sym
                        for sym in self._current_func.inner_functions
                        if sym == callee_name or sym.endswith(f".<locals>.{callee_name}")
                    ),
                    None,
                )
                if inner_symbol:
                    callee = self._lookup_function(inner_symbol)
                    if callee and callee.returns_blocking:
                        self._current_func.returns_blocking = callee.returns_blocking
                # Also handle returning a call to a regular local function
                # that has returns_blocking set (covers indirect factory chains)
                else:
                    callee = self._lookup_function(callee_name)
                    if callee and callee.returns_blocking:
                        self._current_func.returns_blocking = callee.returns_blocking
                    if callee and callee.return_type:
                        self._current_func.return_type = callee.return_type
                        self._current_func.return_module = callee.return_module
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
                if self._current_func:
                    self._current_func.blocking_dicts[dict_name] = key_map
                else:
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
            local_func = self._lookup_function(ref)
            if local_func:
                return (None, local_func.symbol or local_func.name)
            if ref in self.analysis.import_froms:
                imp = self.analysis.import_froms[ref]
                return (imp.module, imp.name)
            if ref in self.analysis.imports:
                imp = self.analysis.imports[ref]
                return (imp.module, None)
        return None

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

            if class_name:
                descriptor_type = self.analysis.class_attr_types.get(class_name, {}).get(attr_name)
                if descriptor_type:
                    descriptor_get = f"{descriptor_type}.__get__"
                    if descriptor_get in self.analysis.functions:
                        descriptor_call = CallSite(
                            name=descriptor_get,
                            line=node.lineno,
                            col=node.col_offset,
                            module=None,
                            object_type=descriptor_type,
                            caller_function=self._current_func.name,
                            arg_names=None,
                            is_property_access=True,
                        )
                        self._current_func.calls.append(descriptor_call)
                        self.analysis.call_graph[self._current_func.name].append(descriptor_get)

            # Attribute access can execute local dynamic hooks. Keep this
            # evidence-based to avoid turning normal data reads into noise.
            if class_name:
                for dunder_name in ("__getattr__", "__getattribute__"):
                    dunder_key = f"{class_name}.{dunder_name}"
                    if dunder_key in self.analysis.functions:
                        dynamic_call = CallSite(
                            name=dunder_key,
                            line=node.lineno,
                            col=node.col_offset,
                            module=None,
                            object_type=class_name,
                            caller_function=self._current_func.name,
                            arg_names=None,
                            is_property_access=True,
                        )
                        self._current_func.calls.append(dynamic_call)
                        self.analysis.call_graph[self._current_func.name].append(dunder_key)

            # Also check for dunder call patterns: s(5) -> resolve s type to class, trace __call__
            # This is handled in visit_Call via _resolve_call_name

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Track augmented assignment: acc += 1 -> Accumulator.__iadd__."""
        if self._current_func and isinstance(node.target, ast.Name):
            class_name = None
            var_name = node.target.id
            if var_name in self._current_func.var_types:
                class_name = self._current_func.var_types[var_name]
            elif var_name in self.analysis.module_var_types:
                class_name = self.analysis.module_var_types[var_name]

            op_dunders = {
                ast.Add: "__iadd__",
                ast.Sub: "__isub__",
                ast.Mult: "__imul__",
                ast.Div: "__itruediv__",
                ast.FloorDiv: "__ifloordiv__",
                ast.Mod: "__imod__",
                ast.Pow: "__ipow__",
                ast.MatMult: "__imatmul__",
            }
            dunder = op_dunders.get(type(node.op))
            if class_name and dunder:
                dunder_key = f"{class_name}.{dunder}"
                if dunder_key in self.analysis.functions:
                    aug_call = CallSite(
                        name=dunder_key,
                        line=node.lineno,
                        col=node.col_offset,
                        module=None,
                        object_type=class_name,
                        caller_function=self._current_func.name,
                        arg_names=None,
                    )
                    self._current_func.calls.append(aug_call)
                    self.analysis.call_graph[self._current_func.name].append(dunder_key)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Track local iterator dunders used by synchronous for loops."""
        self._handle_sync_iteration(node.iter)
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        for generator in node.generators:
            self._handle_sync_iteration(generator.iter)
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        for generator in node.generators:
            self._handle_sync_iteration(generator.iter)
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        for generator in node.generators:
            self._handle_sync_iteration(generator.iter)
        self.generic_visit(node)

    def _handle_sync_iteration(self, iter_expr: ast.expr) -> None:
        if not self._current_func or not isinstance(iter_expr, ast.Name):
            return
        var_name = iter_expr.id
        class_name = None
        if var_name in self._current_func.var_types:
            class_name = self._current_func.var_types[var_name]
        elif var_name in self.analysis.module_var_types:
            class_name = self.analysis.module_var_types[var_name]
        if not class_name:
            return
        for dunder in ("__iter__", "__next__"):
            key = f"{class_name}.{dunder}"
            if key in self.analysis.functions:
                iter_call = CallSite(
                    name=key,
                    line=iter_expr.lineno,
                    col=iter_expr.col_offset,
                    module=None,
                    object_type=class_name,
                    caller_function=self._current_func.name,
                    arg_names=None,
                )
                self._current_func.calls.append(iter_call)
                self.analysis.call_graph[self._current_func.name].append(key)

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
                    _apply_decorator_replacement(
                        analysis,
                        target.name,
                        wrapper_name,
                        decorator_name,
                    )


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


def _apply_decorator_replacement(
    analysis: ModuleAnalysis,
    target_name: str,
    wrapper_name: str,
    decorator_name: str | None = None,
) -> None:
    """Apply decorator replacement: copy wrapper's calls to the decorated function."""
    wrapper_key = wrapper_name
    if wrapper_key not in analysis.functions and decorator_name:
        candidates = [
            symbol
            for symbol in analysis.functions
            if (
                symbol.startswith(f"{decorator_name}.<locals>.")
                and symbol.endswith(f".<locals>.{wrapper_name}")
            )
            or symbol == f"{decorator_name}.<locals>.{wrapper_name}"
        ]
        if candidates:
            wrapper_key = sorted(candidates, key=len)[-1]

    analysis.decorator_replacements[target_name] = wrapper_key

    if wrapper_key in analysis.functions and target_name in analysis.functions:
        target_func = analysis.functions[target_name]
        wrapper_func = analysis.functions[wrapper_key]
        for call in wrapper_func.calls:
            target_func.calls.append(call)


def _build_function_summaries(analysis: ModuleAnalysis, max_iterations: int = 8) -> None:
    """Compute lightweight function summaries after all symbols are collected."""
    for func in analysis.functions.values():
        func.summary = FunctionSummary(
            symbol=func.symbol or func.name,
            params=func.params.copy(),
            return_values=[],
            operations=[
                CallOperation(
                    line=call.line,
                    col=call.col,
                    confidence=call.confidence,
                    callable=KnownCallable(
                        module=call.module,
                        name=call.name,
                        receiver_type=call.object_type,
                        symbol=call.name if call.module is None else None,
                        offloaded=call.offloaded,
                    ),
                )
                for call in func.calls
            ],
        )
        if func.return_alias:
            func.summary.return_values.append(
                KnownCallable(module=func.return_alias[0], name=func.return_alias[1])
            )
        elif func.return_type:
            func.summary.return_values.append(
                KnownInstance(module=func.return_module, type_name=func.return_type)
            )

    # A small fixed point propagates return summaries through local factory chains.
    for _ in range(max_iterations):
        changed = False
        for func in analysis.functions.values():
            if func.return_type or func.return_alias:
                continue
            for call in func.calls:
                if call.module is not None:
                    continue
                callee = analysis.functions.get(call.name)
                if callee is None and call.name in analysis.symbol_aliases:
                    callee = analysis.functions.get(analysis.symbol_aliases[call.name])
                if not callee:
                    continue
                if callee.return_alias:
                    func.return_alias = callee.return_alias
                    changed = True
                    break
                if callee.return_type:
                    func.return_type = callee.return_type
                    func.return_module = callee.return_module
                    changed = True
                    break
        if not changed:
            break


def analyze_file(filepath: str | Path) -> ModuleAnalysis:
    """Analyze a single Python file and return its call graph."""
    filepath = Path(filepath)
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))

    builder = CallGraphBuilder(str(filepath))
    builder.precollect_definitions(tree)
    builder.visit(tree)
    _post_process_decorator_replacements(builder.analysis, tree)
    _build_function_summaries(builder.analysis)

    return builder.analysis


def analyze_source(source: str, filepath: str = "<test>") -> ModuleAnalysis:
    """Analyze source code string."""
    tree = ast.parse(source, filename=filepath)
    builder = CallGraphBuilder(filepath)
    builder.precollect_definitions(tree)
    builder.visit(tree)
    _post_process_decorator_replacements(builder.analysis, tree)
    _build_function_summaries(builder.analysis)
    return builder.analysis
