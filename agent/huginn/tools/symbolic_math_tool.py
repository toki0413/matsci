"""shim: 实现已移至 huginn.tools.symbolic_math 包.

原 2150 行单文件拆为 8 模块:
  _parsers   - 符号声明 + 表达式安全解析 + Einstein 指标 token
  calculus    - differentiate / integrate / simplify / taylor / series
  algebra     - solve / eigenvalue / linear_algebra
  tensor      - tensor_ops / tensor_calculus / einstein_sum
  fem         - constitutive / weak_form
  physics     - dimensional_analysis / dft / thermodynamics / probability / unified
  tool        - SymbolicMathTool 主体 + SymbolicMathInput
"""
from huginn.tools.symbolic_math import SymbolicMathInput, SymbolicMathTool

__all__ = ["SymbolicMathInput", "SymbolicMathTool"]
