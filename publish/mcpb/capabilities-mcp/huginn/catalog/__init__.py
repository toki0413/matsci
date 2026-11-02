"""Catalog 统一接入清单控制面."""
from __future__ import annotations

from huginn.catalog.manager import CatalogManager, discover_handles
from huginn.catalog.models import (
    KINDS,
    ORIGIN_PRIORITY,
    ORIGINS,
    CatalogEntry,
    make_entry_id,
)

__all__ = [
    "CatalogEntry",
    "CatalogManager",
    "KINDS",
    "ORIGINS",
    "ORIGIN_PRIORITY",
    "discover_handles",
    "make_entry_id",
]
