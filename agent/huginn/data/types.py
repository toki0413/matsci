"""Data type definitions for the Huginn data dictionary."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DataType(str, Enum):
    CRYSTAL_STRUCTURE = "crystal_structure"
    MOLECULAR_DYNAMICS = "molecular_dynamics"
    DFT_RESULT = "dft_result"
    POTENTIAL = "potential"
    JOB_RECORD = "job_record"
    EXPERIMENTAL_DATA = "experimental_data"
    DESCRIPTOR = "descriptor"


@dataclass
class DataField:
    name: str
    dtype: str  # "float", "int", "str", "array", "dict"
    required: bool = False
    description: str = ""
    unit: str | None = None


@dataclass
class DataSchema:
    """Schema for a data type in the dictionary."""

    type_name: DataType
    description: str
    fields: list[DataField] = field(default_factory=list)
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)
