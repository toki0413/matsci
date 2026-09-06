"""符号数学工具包 — SymPy 驱动的材料科学符号计算.

解析器在 _parsers, 微积分在 calculus, 代数在 algebra, 张量在 tensor,
FEM 弱形式在 fem, 物理量在 physics, SymbolicMathTool 主体在 tool.
"""
from .tool import SymbolicMathInput, SymbolicMathTool

__all__ = ["SymbolicMathInput", "SymbolicMathTool"]
