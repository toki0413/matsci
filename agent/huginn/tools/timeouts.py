"""工具超时分级 —— 按工具类型给不同的超时上限。

之前所有工具都套同一个硬编码超时，查个结构文件和跑 VASP 用一样的时间
显然不合理。这里把工具分成几档，快的快杀，慢的给够。

注意：用户显式设的超时（环境变量 / 工具自带 timeout）优先于这里的预设。
"""

from __future__ import annotations

import os

# 各档超时上限（秒）
TIMEOUT_PRESETS = {
    "fast": 15,           # 本地查询、kb_question 这类基本秒回的
    "medium": 60,         # web_search、rag 检索，需要联网但不会太久
    "slow": 300,          # vasp / lammps 单次计算，几分钟很正常
    "external_api": 120,  # materials_database / structure 调外部 API
    "long_running": 1800, # 大 workflow、高通量批量任务
}

# 工具名 -> 档位
TOOL_TIMEOUT_MAP = {
    # 快档：本地操作
    "kb_tool": "fast",
    "structure_tool": "fast",
    "symmetry_tool": "fast",
    "unit_tool": "fast",
    "descriptor_tool": "fast",
    "memory_tool": "fast",
    "file_read_tool": "fast",
    "file_edit_tool": "fast",
    "file_write_tool": "fast",
    "validate_tool": "fast",
    "diff_tool": "fast",
    "extract_tool": "fast",
    # 中档：联网检索
    "web_search_tool": "medium",
    "rag_tool": "medium",
    "browser_tool": "medium",
    "characterization_tool": "medium",
    "experimental_data_tool": "medium",
    "knowledge_tool": "medium",
    # 外部 API 档：查材料数据库
    "materials_database_tool": "external_api",
    "database_tool": "external_api",
    # 慢档：单次仿真计算
    "vasp_tool": "slow",
    "qe_tool": "slow",
    "cp2k_tool": "slow",
    "lammps_tool": "slow",
    "abaqus_tool": "slow",
    "comsol_tool": "slow",
    "openfoam_tool": "slow",
    "bash_tool": "slow",
    "code_tool": "slow",
    "ml_potential_tool": "slow",
    "potential_tool": "slow",
    "packing_tool": "slow",
    "numerical_tool": "slow",
    "symbolic_math_tool": "slow",
    "symbolic_regression_tool": "slow",
    "debugger_tool": "slow",
    "autodiff_tool": "slow",
    "image_analysis_tool": "slow",
    "gap_analysis_tool": "slow",
    "hypothesis_generator_tool": "slow",
    "evidence_fusion_tool": "slow",
    "doe_tool": "slow",
    "uq_tool": "slow",
    "tda_tool": "slow",
    "gp_tool": "slow",
    "active_learning_tool": "slow",
    "generative_design_tool": "slow",
    "design_atom_tool": "slow",
    "lean_tool": "slow",
    "report_tool": "slow",
    "visualize_tool": "slow",
    "review_committee_tool": "slow",
    # 长档：大 workflow
    "high_throughput_tool": "long_running",
    "orchestrate_tool": "long_running",
    "materials_autoresearch_tool": "long_running",
    "design_plan_tool": "long_running",
    "diagnose_tool": "long_running",
    "job_tool": "long_running",
    "skill_tool": "long_running",
}


def get_timeout(tool_name: str, default: int = 60) -> int:
    """按工具名查超时上限（秒）。

    优先级:
        1. 环境变量 HUGINN_TOOL_TIMEOUT_<NAME> （用户显式覆盖）
        2. 全局环境变量 HUGINN_TOOL_TIMEOUT （统一覆盖所有工具）
        3. TOOL_TIMEOUT_MAP 里按档位查到的预设
        4. default 兜底

    tool_name 里的横杠会转成下划线拼环境变量名。
    """
    # 1. 单工具级别环境变量覆盖
    env_key = "HUGINN_TOOL_TIMEOUT_" + tool_name.upper().replace("-", "_")
    env_val = os.environ.get(env_key)
    if env_val:
        try:
            return max(int(env_val), 1)
        except ValueError:
            pass

    # 2. 全局环境变量覆盖
    global_val = os.environ.get("HUGINN_TOOL_TIMEOUT")
    if global_val:
        try:
            return max(int(global_val), 1)
        except ValueError:
            pass

    # 3. 按档位查预设
    tier = TOOL_TIMEOUT_MAP.get(tool_name)
    if tier and tier in TIMEOUT_PRESETS:
        return TIMEOUT_PRESETS[tier]

    # 4. 兜底
    return default


class ToolTimeoutPresets:
    """方便外部按类属性访问的薄封装，保持跟需求文档一致的接口。"""

    PRESETS = TIMEOUT_PRESETS
    MAP = TOOL_TIMEOUT_MAP

    @staticmethod
    def get(tool_name: str, default: int = 60) -> int:
        return get_timeout(tool_name, default)
