"""NEB / PES 工具包 — 势函数地形图能力.

模块布局:
  _io          - read_structure / read_xyz_fallback / write_xyz / write_poscar
  _evaluators  - eval_images / eval_single / eval_lj / eval_via_ml_potential / eval_via_vasp
  _neb_core    - idpp_initial_path / compute_neb_forces / compute_barriers / compute_path_length
  _dimer       - dimer_rotate / estimate_hessian_along_mode
  _topology    - topology_via_tda / basin_analysis / find_local_minima/saddles/extrema / count_connected_basins
  tool         - NEBTool 主体 + NEBToolInput + NEBToolOutput
"""
from huginn.tools.neb.tool import NEBTool, NEBToolInput, NEBToolOutput

__all__ = ["NEBTool", "NEBToolInput", "NEBToolOutput"]
