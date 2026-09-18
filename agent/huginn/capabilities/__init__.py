"""capaibilities 包 — 能力维度体系 (P1).
"""
from huginn.capabilities.capability import (
    DIMENSIONS,
    CapabilityMetadata,
    capabilities_to_metadata,
    capability,
    get_capabilities,
)
from huginn.capabilities.registry import CapabilityRegistry

__all__ = [
    "DIMENSIONS", "CapabilityMetadata",
    "capability", "get_capabilities", "capabilities_to_metadata",
    "CapabilityRegistry",
]
