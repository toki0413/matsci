# 保留: 仍有生产代码引用, 升级路径: 迁移到 huginn.tools.sci.gp_tool 后删除
"""shim: 文件已移至 huginn.tools.sci.gp_tool."""
from huginn.tools.sci.gp_tool import (  # noqa: F401
    CalibrationVariableSpec,
    GPTool,
    GPToolInput,
    NumPyGP,
)

__all__ = ["GPTool", "GPToolInput", "NumPyGP", "CalibrationVariableSpec"]
