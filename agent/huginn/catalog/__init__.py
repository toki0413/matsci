"""Catalog 统一接入清单控制面."""
from __future__ import annotations

from huginn.catalog.manager import CatalogManager
from huginn.catalog.models import (
    CatalogEntry,
    KINDS,
    ORIGINS,
    ORIGIN_PRIORITY,
    make_entry_id,
)

__all__ = [
    "CatalogEntry",
    "CatalogManager",
    "KINDS",
    "ORIGINS",
    "ORIGIN_PRIORITY",
    "make_entry_id",
]