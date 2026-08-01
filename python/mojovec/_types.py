"""Public type aliases and result schemas for the managed API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict


WAL_ASYNC = 1
"""Append WAL records asynchronously for maximum write throughput."""

WAL_SYNC = 2
"""Synchronously flush each WAL mutation before returning."""

DEFAULT_MMAP_THRESHOLD_BYTES = 64 * 1024 * 1024
"""Default minimum mapped region size used by collection loading."""

Metadata = Mapping[str, str | int | float | bool]
"""A metadata mapping containing scalar string, integer, float, or bool values."""

Where = Mapping[str, Any]
"""A Chroma-style typed metadata filter expression."""


class QueryResult(TypedDict):
    """Aligned batched results returned by vector, text, and hybrid queries."""

    ids: list[list[int]]
    distances: list[list[float]]
    metadatas: list[list[dict[str, str | int | float | bool]]]
    documents: list[list[str]]
    scores: list[list[float]]
