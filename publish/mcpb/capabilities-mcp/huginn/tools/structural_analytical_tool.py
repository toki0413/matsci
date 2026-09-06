"""shim: 实现已移至 huginn.tools.structural_analytical 包.

9 action: beam_static/modal/buckling, plate_static/modal/buckling,
shell_buckling/modal, stress_concentration. 纯 numpy/scipy 后端.
"""
from huginn.tools.structural_analytical import (
    StructuralAnalyticalInput,
    StructuralAnalyticalTool,
)

__all__ = ["StructuralAnalyticalInput", "StructuralAnalyticalTool"]
