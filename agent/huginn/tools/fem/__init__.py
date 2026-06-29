"""轻量 FEM 工具包 — scikit-fem 封装.

线性 2D 平面应力/应变: 静力/模态/屈曲. 主体在 tool, 各分析类型分模块:
  mesh    - mesh_from_geometry (矩形/圆形/立方体网格)
  static  - static_linear (线弹性 K u = f)
  modal   - modal (一致质量 + eigsh)
  buckling - buckling (几何刚度 + eigsh)
"""
from .tool import FEMInput, FEMTool

__all__ = ["FEMInput", "FEMTool"]
