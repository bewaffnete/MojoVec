"""Compile public Chroma-style filters into native bitmap predicates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._runtime import _native
from ._types import Where


def _compile_predicate(operation: str, key: str, value: Any) -> Any:
    if not isinstance(value, (str, int, float, bool)):
        raise TypeError("where values must be str, int, float, or bool")
    return _native._where_predicate(operation, key, value)


def _scalar_kind(value: Any) -> type[Any]:
    # bool is an int subclass in Python but a separate MetadataValue kind.
    return bool if isinstance(value, bool) else type(value)


def _combine(operation: str, conditions: Sequence[Any]) -> Any:
    if not conditions:
        raise ValueError(f"${operation} requires at least one condition")
    return _native._where_combine(operation, list(conditions))


def _compile_field(key: str, expression: Any) -> Any:
    if not isinstance(expression, Mapping):
        return _compile_predicate("eq", key, expression)
    if not expression:
        raise ValueError(f"where expression for {key!r} cannot be empty")

    conditions: list[Any] = []
    operators = {
        "$eq": "eq",
        "$ne": "ne",
        "$gt": "gt",
        "$gte": "gte",
        "$lt": "lt",
        "$lte": "lte",
    }
    for operator, value in expression.items():
        if operator in operators:
            conditions.append(_compile_predicate(operators[operator], key, value))
        elif operator in ("$in", "$nin", "$not_in"):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"{operator} requires a sequence of scalar values")
            if not value:
                raise ValueError(f"{operator} requires at least one value")
            first_kind = _scalar_kind(value[0])
            if first_kind not in (str, int, float, bool) or any(
                _scalar_kind(item) is not first_kind for item in value
            ):
                raise TypeError(
                    f"{operator} values must be scalar and have one type"
                )
            predicate = "eq" if operator == "$in" else "ne"
            combine = "any" if operator == "$in" else "all"
            conditions.append(
                _combine(
                    combine,
                    [_compile_predicate(predicate, key, item) for item in value],
                )
            )
        else:
            raise ValueError(f"unsupported where operator: {operator!r}")
    return conditions[0] if len(conditions) == 1 else _combine("all", conditions)


def _compile_where(where: Where) -> Any:
    if not isinstance(where, Mapping):
        raise TypeError("where must be a mapping")
    if not where:
        raise ValueError("where cannot be empty")

    conditions: list[Any] = []
    for key, expression in where.items():
        if not isinstance(key, str):
            raise TypeError("where keys must be strings")
        if key in ("$and", "$or"):
            if (
                isinstance(expression, (str, bytes))
                or not isinstance(expression, Sequence)
            ):
                raise TypeError(f"{key} requires a sequence of where mappings")
            operation = "all" if key == "$and" else "any"
            conditions.append(
                _combine(operation, [_compile_where(item) for item in expression])
            )
        elif key == "$not":
            conditions.append(_combine("not", [_compile_where(expression)]))
        elif key.startswith("$"):
            raise ValueError(f"unsupported logical where operator: {key!r}")
        else:
            conditions.append(_compile_field(key, expression))
    return conditions[0] if len(conditions) == 1 else _combine("all", conditions)
