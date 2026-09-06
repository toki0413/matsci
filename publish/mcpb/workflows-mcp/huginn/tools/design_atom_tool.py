# 保留: 仍有生产代码引用, 升级路径: 迁移到 huginn.tools.design.design_atom_tool 后删除
"""shim: 文件已移至 huginn.tools.design.design_atom_tool."""
from huginn.tools.design.design_atom_tool import (  # noqa: F401
    _ATOM_REGISTRY,
    _RENDERERS,
    DesignAtomInput,
    DesignAtomTool,
)

__all__ = ["DesignAtomTool", "DesignAtomInput"]
