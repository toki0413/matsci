"""结构力学解析求解工具包 — 梁/板/壳/应力集中.

纯 numpy/scipy 实现, 不依赖外部求解器. 主体在 tool, 各结构类型分模块:
  beams                - beam_static / beam_modal / beam_buckling
  plates               - plate_static / plate_modal / plate_buckling
  shells               - shell_buckling / shell_modal
  stress_concentration - stress_concentration (Kt)
"""
from .tool import StructuralAnalyticalInput, StructuralAnalyticalTool

__all__ = ["StructuralAnalyticalInput", "StructuralAnalyticalTool"]
