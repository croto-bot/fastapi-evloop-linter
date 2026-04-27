"""Semantic value/effect models for EVL001 analysis.

The call graph keeps the public linting surface small, but internally we need
stable names and value summaries so different AST shapes can collapse to the
same operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class SymbolId:
    """Stable identity for local symbols inside one analyzed module."""

    name: str

    @classmethod
    def module_function(cls, name: str) -> "SymbolId":
        return cls(name)

    @classmethod
    def class_member(cls, class_name: str, member: str) -> "SymbolId":
        return cls(f"{class_name}.{member}")

    @classmethod
    def nested(cls, parent: str, child: str) -> "SymbolId":
        return cls(f"{parent}.<locals>.{child}")


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class ValueInfo:
    confidence: Confidence = Confidence.HIGH


@dataclass(frozen=True)
class UnknownValue(ValueInfo):
    reason: str = "unknown"


@dataclass(frozen=True)
class KnownModule(ValueInfo):
    module: str = ""


@dataclass(frozen=True)
class KnownType(ValueInfo):
    module: str | None = None
    name: str = ""


@dataclass(frozen=True)
class KnownInstance(ValueInfo):
    module: str | None = None
    type_name: str = ""


@dataclass(frozen=True)
class KnownCallable(ValueInfo):
    module: str | None = None
    name: str | None = None
    receiver_type: str | None = None
    symbol: str | None = None
    offloaded: bool = False


@dataclass(frozen=True)
class TupleValue(ValueInfo):
    items: tuple[ValueInfo, ...] = ()


@dataclass(frozen=True)
class MaybeBlockingValue(ValueInfo):
    module: str | None = None
    name: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class Operation:
    line: int
    col: int
    confidence: Confidence = Confidence.HIGH


@dataclass(frozen=True)
class CallOperation(Operation):
    callable: KnownCallable = field(default_factory=KnownCallable)


@dataclass(frozen=True)
class AttributeOperation(Operation):
    owner: ValueInfo = field(default_factory=UnknownValue)
    attr: str = ""


@dataclass(frozen=True)
class ContextOperation(Operation):
    manager: ValueInfo = field(default_factory=UnknownValue)
    enter_symbol: str | None = None
    is_async: bool = False


@dataclass(frozen=True)
class IteratorOperation(Operation):
    iterator: ValueInfo = field(default_factory=UnknownValue)


@dataclass(frozen=True)
class OperatorOperation(Operation):
    symbol: str | None = None


@dataclass(frozen=True)
class AwaitOperation(Operation):
    value: ValueInfo = field(default_factory=UnknownValue)


@dataclass(frozen=True)
class ReturnOperation(Operation):
    value: ValueInfo = field(default_factory=UnknownValue)


@dataclass
class ExprResult:
    value: ValueInfo = field(default_factory=UnknownValue)
    operations: list[Operation] = field(default_factory=list)
    confidence: Confidence = Confidence.HIGH
    notes: list[str] = field(default_factory=list)


@dataclass
class ScopeState:
    values: dict[str, ValueInfo] = field(default_factory=dict)
    direct_event_loop: bool = True


@dataclass
class FunctionSummary:
    symbol: str
    params: list[str] = field(default_factory=list)
    return_values: list[ValueInfo] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    direct_blocking_effects: list[CallOperation] = field(default_factory=list)
    transitive_blocking_effects: list[CallOperation] = field(default_factory=list)
    confidence: Confidence = Confidence.HIGH
    partial: bool = False
