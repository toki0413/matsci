"""shim: 实现已移至 huginn.tools.specialty_analysis 包.

5 action: eigenvalue_buckling, modal_lanczos, fatigue_sn,
fatigue_crack_growth, fracture_lefm. 纯 numpy/scipy 后端.
"""
from huginn.tools.specialty_analysis import (
    SpecialtyAnalysisInput,
    SpecialtyAnalysisTool,
)

__all__ = ["SpecialtyAnalysisInput", "SpecialtyAnalysisTool"]
