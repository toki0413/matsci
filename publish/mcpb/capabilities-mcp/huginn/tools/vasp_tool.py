# 保留: 仍有生产代码引用, 升级路径: 迁移到 huginn.tools.sim.vasp_tool 后删除
"""shim: 文件已移至 huginn.tools.sim.vasp_tool."""
from huginn.tools.sim.vasp_tool import (  # noqa: F401
    _HAS_HUGINN_EXT,
    VaspTool,
    VaspToolInput,
    VaspToolOutput,
)

__all__ = ["VaspTool", "VaspToolInput", "VaspToolOutput"]
