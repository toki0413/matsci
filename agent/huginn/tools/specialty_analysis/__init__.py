"""结构力学专门分析工具包 — 屈曲/模态/疲劳/断裂.

纯 numpy/scipy 后端. 主体在 tool, 各分析类型分模块:
  buckling  - eigenvalue_buckling (广义特征值)
  modal     - modal_lanczos (shift-invert Lanczos)
  fatigue   - fatigue_sn / fatigue_crack_growth (Basquin + Paris)
  fracture  - fracture_lefm (K_I / J / G / K_IC 判据)
"""
from .tool import SpecialtyAnalysisInput, SpecialtyAnalysisTool

__all__ = ["SpecialtyAnalysisInput", "SpecialtyAnalysisTool"]
