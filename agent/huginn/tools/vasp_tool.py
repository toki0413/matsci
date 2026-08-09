"""shim: 文件已移至 huginn.tools.sim.vasp_tool."""
from huginn.tools.sim.vasp_tool import (  # noqa: F401
    _HAS_HUGINN_EXT,
    VaspTool,
    VaspToolInput,
    VaspToolOutput,
)

__all__ = ["VaspTool", "VaspToolInput", "VaspToolOutput"]
