"""shim: 文件已移至 huginn.tools.sim.vasp_tool."""
from huginn.tools.sim.vasp_tool import (  # noqa: F401
    VaspTool,
    VaspToolInput,
    VaspToolOutput,
    _HAS_HUGINN_EXT,
)

__all__ = ["VaspTool", "VaspToolInput", "VaspToolOutput"]
