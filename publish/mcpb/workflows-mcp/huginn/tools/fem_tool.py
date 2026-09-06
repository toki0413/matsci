"""shim: 实现已移至 huginn.tools.fem 包.

4 action: mesh_from_geometry, static_linear, modal, buckling.
scikit-fem 缺失时 optional_modules 自动跳过注册.
"""
from huginn.tools.fem import FEMInput, FEMTool

__all__ = ["FEMInput", "FEMTool"]
