"""shim: 文件已移至 huginn.tools.sci.multi_fidelity_tool."""
from huginn.tools.sci.multi_fidelity_tool import (  # noqa: F401
    FidelitySource,
    MultiFidelityInput,
    MultiFidelitySurrogate,
    MultiFidelityTool,
)

__all__ = [
    "MultiFidelityTool",
    "MultiFidelityInput",
    "FidelitySource",
    "MultiFidelitySurrogate",
]
