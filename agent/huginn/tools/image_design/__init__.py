"""材料科学输出设计工具包 — 7 个场景: 粒度分布 / XRD / 应力-应变 / 能带 / 显微标注 / EDS / TEM FFT.

共享 matplotlib 工具在 _mpl_utils, 各场景在 scenes_*, ImageDesignTool 主体在 tool.
"""
from .tool import ImageDesignInput, ImageDesignTool

__all__ = ["ImageDesignInput", "ImageDesignTool"]
