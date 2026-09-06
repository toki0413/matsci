"""材料科学图像分析工具包.

模块布局:
  _utils                - load_gray/rgb, parse_color, auto_detect_colors, otsu_numpy
  scenes_sem            - sem_analysis
  scenes_tem            - tem_lattice (含 _D_SPACING_TABLE)
  scenes_eds            - eds_mapping
  scenes_particles      - particle_stats
  scenes_defect         - defect_detect
  scenes_phase_field    - phase_field
  scenes_plot_extract   - plot_extract (单/多曲线 + 自动轴检测 + 可选 plotdigitizer)
  scenes_deplot         - deplot_chart (图表转表格, 需 transformers+torch)
  tool                  - ImageAnalysisTool 主体 + ImageAnalysisInput
"""
from huginn.tools.image_analysis.tool import ImageAnalysisInput, ImageAnalysisTool

__all__ = ["ImageAnalysisInput", "ImageAnalysisTool"]
