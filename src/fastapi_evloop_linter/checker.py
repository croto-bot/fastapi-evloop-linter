"""Main checker: detects event loop blocking calls in FastAPI async code.

This is the core of the linter. It:
1. Parses Python files and builds a call graph
2. Identifies async FastAPI endpoints
3. Traces the call tree to find blocking calls at any depth
4. Reports violations with precise locations
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .classifier import classify_call, Verdict
from .callgraph import (
    ModuleAnalysis,
    FuncDef,
    CallSite,
    analyze_file,
    analyze_source,
)


@dataclass
class Violation:
    """A detected blocking call violation."""
    filepath: str
    line: int
    col: int
    message: str
    severity: str  # "error" or "warning"
    reason: str = ""
    # Chain from endpoint to the blocking call
    call_chain: list[str] = field(default_factory=list)
    # How deep in the call tree this was found
    depth: int = 0


@dataclass
class LintResult:
    """Result of linting one or more files."""
    violations: list[Violation] = field(default_factory=list)
    files_checked: int = 0
    endpoints_found: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")

    def merge(self, other: LintResult) -> None:
        self.violations.extend(other.violations)
        self.files_checked += other.files_checked
        self.endpoints_found += other.endpoints_found


class EventLoopChecker:
    """Detects event loop blocking calls in FastAPI async endpoints."""

    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth

    def check_source(self, source: str, filepath: str = "<stdin>") -> LintResult:
        """Check a source code string for violations."""
        try:
            analysis = analyze_source(source, filepath)
        except SyntaxError:
            return LintResult(files_checked=1)

        return self._check_analysis(analysis, source)

    def check_file(self, filepath: str | Path) -> LintResult:
        """Check a single file for violations."""
        filepath = Path(filepath)
        try:
            source = filepath.read_text()
            analysis = analyze_file(filepath)
        except (SyntaxError, OSError):
            return LintResult(files_checked=1)

        return self._check_analysis(analysis, source)

    def _check_analysis(self, analysis: ModuleAnalysis, source: str) -> LintResult:
        """Run violation detection on a module analysis."""
        result = LintResult(files_checked=1)

        # Find all FastAPI endpoints
        endpoints: list[FuncDef] = []
        for func in analysis.functions.values():
            if self._is_endpoint(func):
                endpoints.append(func)
                result.endpoints_found += 1

        # For each endpoint, trace the call tree
        for endpoint in endpoints:
            visited: set[str] = set()
            self._trace_calls(
                analysis=analysis,
                func=endpoint,
                result=result,
                call_chain=[endpoint.name],
                depth=0,
                visited=visited,
            )

        return result

    def _is_endpoint(self, func: FuncDef) -> bool:
        """Check if a function is a FastAPI endpoint."""
        return "__fastapi_endpoint__" in func.decorators and func.is_async

    def _trace_calls(
        self,
        analysis: ModuleAnalysis,
        func: FuncDef,
        result: LintResult,
        call_chain: list[str],
        depth: int,
        visited: set[str],
    ) -> None:
        """Recursively trace calls from a function, looking for blocking patterns."""
        if depth > self.max_depth:
            return

        func_key = func.name
        if func_key in visited:
            return
        visited.add(func_key)

        for call in func.calls:
            # Classify this call generically.
            verdict, reason = classify_call(
                call.module,
                call.name,
                call.object_type,
                analysis,
            )

            if verdict == Verdict.BLOCKING:
                if call.module:
                    full_name = (
                        f"{call.module}.{call.object_type}.{call.name}"
                        if call.object_type
                        else f"{call.module}.{call.name}"
                    )
                else:
                    full_name = call.name
                violation = Violation(
                    filepath=analysis.filepath,
                    line=call.line,
                    col=call.col,
                    message=f"Blocking call: {full_name} ({reason})",
                    severity="error",
                    reason=reason,
                    call_chain=call_chain.copy(),
                    depth=depth,
                )
                result.violations.append(violation)

            # Try to resolve this call to a local function and recurse
            callee_name = call.name
            # Don't try to follow calls that are clearly external (module.name patterns)
            if call.module is not None:
                continue

            callee_func = analysis.functions.get(callee_name)
            if callee_func is not None:
                new_chain = call_chain + [callee_name]
                self._trace_calls(
                    analysis=analysis,
                    func=callee_func,
                    result=result,
                    call_chain=new_chain,
                    depth=depth + 1,
                    visited=visited.copy(),  # Allow re-visiting from different paths
                )

                # Also trace any function names passed as arguments (callback pattern)
                # If callee has parameters and one of the arg_names is a known function,
                # and callee calls one of its parameters, follow the argument function too.
                if callee_func.params and call.arg_names:
                    self._trace_callback_args(
                        analysis=analysis,
                        callee_func=callee_func,
                        arg_names=call.arg_names,
                        call_chain=new_chain,
                        depth=depth + 1,
                        visited=visited.copy(),
                        result=result,
                    )

    def check_directory(self, dirpath: str | Path) -> LintResult:
        """Check all Python files in a directory recursively."""
        dirpath = Path(dirpath)
        result = LintResult()

        for py_file in sorted(dirpath.rglob("*.py")):
            file_result = self.check_file(py_file)
            result.merge(file_result)

        return result

    def _trace_callback_args(
        self,
        analysis: ModuleAnalysis,
        callee_func: FuncDef,
        arg_names: list[str | None],
        call_chain: list[str],
        depth: int,
        visited: set[str],
        result: LintResult,
    ) -> None:
        """Trace function references passed as arguments into a callee.

        If callee_func calls one of its parameters, and that parameter maps to
        a function name in arg_names, trace through that function too.
        """
        if depth > self.max_depth:
            return

        # Build a mapping of parameter position to arg_name
        params = callee_func.params
        param_to_arg: dict[str, str] = {}
        for i, param in enumerate(params):
            if i < len(arg_names) and arg_names[i] is not None:
                param_to_arg[param] = arg_names[i]

        # Check if callee calls any of its parameters
        for call in callee_func.calls:
            # If the call name matches a parameter, follow the mapped arg function
            if call.name in param_to_arg:
                actual_func_name = param_to_arg[call.name]
                actual_func = analysis.functions.get(actual_func_name)
                if actual_func is not None:
                    new_chain = call_chain + [actual_func_name]
                    self._trace_calls(
                        analysis=analysis,
                        func=actual_func,
                        result=result,
                        call_chain=new_chain,
                        depth=depth + 1,
                        visited=visited.copy(),
                    )


def format_violation(v: Violation) -> str:
    """Format a violation for output (ruff-compatible style)."""
    chain_str = " -> ".join(v.call_chain) if v.call_chain else ""
    depth_info = f" (depth={v.depth})" if v.depth > 0 else ""
    chain_info = f" via {chain_str}" if chain_str and v.depth > 0 else ""
    return f"{v.filepath}:{v.line}:{v.col}: EVL001 {v.severity}: {v.message}{chain_info}{depth_info}"
