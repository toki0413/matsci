"""shim: 实现已移至 huginn.tools.image_design 包.

原 1020 行单文件拆为 9 模块:
  _mpl_utils       - matplotlib 工具 (字体/存图/图像加载/颜色/参数)
  scenes_particles - 粒度分布
  scenes_xrd       - XRD 谱图
  scenes_stress    - 应力-应变
  scenes_band      - 能带结构
  scenes_micro     - 显微标注
  scenes_eds       - EDS 叠加
  scenes_tem       - TEM FFT 标注
  tool             - ImageDesignTool 主体 + ImageDesignInput
"""
from huginn.tools.image_design import ImageDesignInput, ImageDesignTool

__all__ = ["ImageDesignInput", "ImageDesignTool"]
