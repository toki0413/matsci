"""shim: 实现已移至 huginn.tools.image_analysis 包.

原 960 行单文件拆为 10 模块:
  _utils               - load_gray/rgb, parse_color, auto_detect_colors, otsu_numpy
  scenes_sem           - sem_analysis
  scenes_tem           - tem_lattice (含 _D_SPACING_TABLE)
  scenes_eds           - eds_mapping
  scenes_particles     - particle_stats
  scenes_defect        - defect_detect
  scenes_phase_field   - phase_field
  scenes_plot_extract  - plot_extract
  tool                 - ImageAnalysisTool 主体 + ImageAnalysisInput
"""
from huginn.tools.image_analysis import ImageAnalysisInput, ImageAnalysisTool

__all__ = ["ImageAnalysisInput", "ImageAnalysisTool"]
