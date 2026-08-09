"""shim: 文件已移至 huginn.tools.sim.lammps_tool."""
from huginn.tools.sim.lammps_tool import (  # noqa: F401
    _HAS_HUGINN_EXT,
    LammpsTool,
    LammpsToolInput,
    LammpsToolOutput,
)

__all__ = ["LammpsTool", "LammpsToolInput", "LammpsToolOutput"]
