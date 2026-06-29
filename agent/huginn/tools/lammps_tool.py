"""shim: 文件已移至 huginn.tools.sim.lammps_tool."""
from huginn.tools.sim.lammps_tool import (  # noqa: F401
    LammpsTool,
    LammpsToolInput,
    LammpsToolOutput,
    _HAS_HUGINN_EXT,
)

__all__ = ["LammpsTool", "LammpsToolInput", "LammpsToolOutput"]
