"""shim: 文件已移至 huginn.tools.sci.gp_tool."""
from huginn.tools.sci.gp_tool import (  # noqa: F401
    CalibrationVariableSpec,
    GPTool,
    GPToolInput,
    NumPyGP,
)

__all__ = ["GPTool", "GPToolInput", "NumPyGP", "CalibrationVariableSpec"]
