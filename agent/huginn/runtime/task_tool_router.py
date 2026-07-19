"""Task-Dynamic Tool Router (P4).

根据当前 task 语义动态决定暴露哪些工具子集. 复用已有 tool_filter 机制
(agent/core.py:77 + subagent.py:49 + factory.py:178 都是 profile 静态),
补 task-dynamic 路由.

设计 (ponytail):
- 规则版: task keyword → tool category 映射, 零 LLM 成本
- 复用 HuginnAgent.tool_filter, 不新建 ToolRegistry
- 默认关, HUGINN_TASK_TOOL_ROUTER=1 开启
- 无命中 → 返回空 list, caller fallback 到原 tool_filter (不破坏现状)
- 多关键词命中 → union 工具集
- 工具不存在于 available → 静默跳过 (跟 core.py:642 行为一致)

接入:
  from huginn.runtime.task_tool_router import route_tools
  subset = route_tools(task_message, available_tool_names)
  if subset:
      self.tool_filter = set(subset)

升级路径:
- LLM 版: 小模型给工具打 0-1 分 (覆盖未知模式)
- phase-aware: 同 task 不同 phase 可能需要不同工具子集
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# === Tool Category 定义 ===
#
# ponytail: category 名跟工具名前缀对齐 (vasp_tool → "vasp"). 这样
# categorize_tool 直接按前缀匹配, 不需 lookup table.
# 升级路径: 工具 metadata 显式标 category (更准, 但要改 ToolRegistry).

TOOL_CATEGORIES: dict[str, list[str]] = {
    # 第一性原理 (DFT)
    "vasp": ["vasp_tool", "structure_tool", "file_read_tool", "file_write_tool",
             "analysis_tool", "literature_tool"],
    "qe": ["qe_tool", "structure_tool", "file_read_tool", "file_write_tool",
           "analysis_tool", "literature_tool"],
    "cp2k": ["cp2k_tool", "structure_tool", "file_read_tool", "file_write_tool",
             "analysis_tool", "literature_tool"],
    "gaussian": ["gaussian_tool", "structure_tool", "file_read_tool",
                 "file_write_tool", "analysis_tool", "literature_tool"],
    "orca": ["orca_tool", "structure_tool", "file_read_tool", "file_write_tool",
             "analysis_tool", "literature_tool"],
    # 分子动力学
    "lammps": ["lammps_tool", "structure_tool", "file_read_tool",
               "file_write_tool", "analysis_tool"],
    "gromacs": ["gromacs_tool", "structure_tool", "file_read_tool",
                "file_write_tool", "analysis_tool"],
    # 有限元
    "abaqus": ["abaqus_tool", "file_read_tool", "file_write_tool",
               "analysis_tool"],
    # 机器学习势函数
    "ml": ["ml_potential_tool", "structure_tool", "file_read_tool",
           "file_write_tool", "analysis_tool", "literature_tool"],
    # 通用分析 (无计算工具)
    "analysis": ["analysis_tool", "file_read_tool", "literature_tool",
                 "web_search_tool"],
    # 通用文件操作
    "file": ["file_read_tool", "file_write_tool", "file_edit_tool"],
    # 文献检索
    "web": ["web_search_tool", "literature_tool"],
}


# === Task keyword → category 映射 ===
#
# ponytail: 中英文双语, 覆盖常见说法. 多个关键词命中同一 category 没关系,
# union 时自然去重.

TASK_KEYWORD_MAP: dict[str, str] = {
    # VASP (具体软件名, 不用模糊的 "dft" / "first-principles" 因为 Gaussian/QE
    # 也做 DFT, 模糊关键词会误路由. 用户应说具体软件名.)
    "vasp": "vasp",
    "band structure": "vasp",
    "能带": "vasp",
    "band gap": "vasp",
    "带隙": "vasp",
    "dos": "vasp",
    "态密度": "vasp",
    "pbe": "vasp",
    "hse": "vasp",
    "encut": "vasp",

    # Quantum ESPRESSO
    "quantum espresso": "qe",
    "qe": "qe",
    "pwscf": "qe",

    # CP2K
    "cp2k": "cp2k",

    # Gaussian / ORCA (量子化学)
    "gaussian": "gaussian",
    "高斯": "gaussian",
    "orca": "orca",
    "quantum chemistry": "gaussian",
    "量子化学": "gaussian",
    "dftb": "gaussian",

    # LAMMPS / MD
    "lammps": "lammps",
    "molecular dynamics": "lammps",
    "分子动力学": "lammps",
    " md ": "lammps",
    "md simulation": "lammps",
    "nve": "lammps",
    "nvt": "lammps",
    "npt": "lammps",
    "rdf": "lammps",
    "msd": "lammps",
    "diffusion": "lammps",
    "扩散": "lammps",

    # GROMACS (生物 MD)
    "gromacs": "gromacs",
    "force field": "gromacs",
    "力场": "gromacs",

    # Abaqus (FEA)
    "abaqus": "abaqus",
    "fea": "abaqus",
    "有限元": "abaqus",
    "stress": "abaqus",
    "strain": "abaqus",
    "应力": "abaqus",
    "应变": "abaqus",

    # ML potential
    "ml potential": "ml",
    "machine learning potential": "ml",
    "neural network potential": "ml",
    "mtp": "ml",
    "nnp": "ml",
    "graph neural network": "ml",
    "机器学习势": "ml",

    # 分析
    "analyze": "analysis",
    "analysis": "analysis",
    "分析": "analysis",
    "plot": "analysis",
    "可视化": "analysis",
    "visualize": "analysis",

    # 文件
    "read file": "file",
    "write file": "file",
    "edit file": "file",
    "文件": "file",

    # 文献
    "search": "web",
    "检索": "web",
    "literature": "web",
    "文献": "web",
    "arxiv": "web",
    "paper": "web",
    "论文": "web",
}


def categorize_tool(tool_name: str) -> str:
    """从工具名推断 category.

    ponytail: 按工具名前缀匹配. vasp_tool → "vasp", lammps_tool → "lammps".
    升级路径: 工具 metadata 显式标 category (更准, 但要改 ToolRegistry).

    Returns:
        category 名, 无匹配返回 "unknown".
    """
    name_lower = tool_name.lower()
    # 精确前缀匹配
    for prefix in ("vasp", "lammps", "gromacs", "gaussian", "orca",
                    "qe", "cp2k", "abaqus", "ml", "analysis",
                    "structure", "file", "literature", "web_search"):
        if name_lower.startswith(prefix) or name_lower == prefix:
            return prefix if prefix != "web_search" else "web"
    return "unknown"


def route_tools(
    task_message: str,
    available_tool_names: list[str],
) -> list[str]:
    """根据 task 语义路由工具子集.

    Args:
        task_message: 当前 task 的用户消息/任务描述
        available_tool_names: ToolRegistry 里实际可用的工具名列表
            (caller 从 ToolRegistry.list_tools() 拿)

    Returns:
        路由后的工具名子集. 无命中时返回空 list (caller fallback 到全塞).
        工具名不存在于 available 时静默跳过.

    ponytail: 规则版, 零 LLM 成本. 多关键词命中 union.
    """
    if not task_message or not available_tool_names:
        return []

    msg_lower = task_message.lower()

    # 1. 找所有命中的 category
    matched_categories: set[str] = set()
    for keyword, category in TASK_KEYWORD_MAP.items():
        # " md " 这种带空格的关键词特殊处理 (避免 match "md5" / "cmd")
        if keyword.startswith(" ") and keyword.endswith(" "):
            if keyword in f" {msg_lower} ":
                matched_categories.add(category)
        elif keyword in msg_lower:
            matched_categories.add(category)

    if not matched_categories:
        # 无命中 → 空列表, caller fallback
        logger.debug("task_tool_router: no keyword match, fallback to all")
        return []

    # 2. union 工具集
    routed_set: set[str] = set()
    for category in matched_categories:
        tools = TOOL_CATEGORIES.get(category, [])
        routed_set.update(tools)

    # 3. 跟 available 取交集 (静默跳过不存在的工具)
    available_set = set(available_tool_names)
    routed = [t for t in routed_set if t in available_set]

    logger.debug(
        "task_tool_router: matched categories=%s, routed %d/%d tools",
        matched_categories, len(routed), len(available_tool_names),
    )
    return routed


# === 自检 ===

if __name__ == "__main__":
    # 模拟 ToolRegistry 里实际可用的工具
    AVAILABLE = [
        "vasp_tool", "lammps_tool", "qe_tool", "cp2k_tool", "gromacs_tool",
        "gaussian_tool", "orca_tool", "abaqus_tool", "ml_potential_tool",
        "structure_tool", "file_read_tool", "file_write_tool", "file_edit_tool",
        "analysis_tool", "literature_tool", "web_search_tool",
        "diagnose_tool", "knowledge_tool",  # 杂项
    ]

    # 1. 单关键词命中 — VASP
    routed = route_tools("compute band structure of Si", AVAILABLE)
    assert "vasp_tool" in routed, "VASP task 应暴露 vasp_tool"
    assert "structure_tool" in routed
    assert "file_read_tool" in routed
    assert "analysis_tool" in routed
    assert "lammps_tool" not in routed, "VASP task 不应暴露 lammps_tool"
    assert "gromacs_tool" not in routed
    assert "abaqus_tool" not in routed
    print(f"1. VASP: {len(routed)} tools")

    # 2. 中文关键词 — 能带 (不再用模糊的 "第一性原理", 改用具体关键词)
    routed = route_tools("用 vasp 算一下 Si 的能带", AVAILABLE)
    assert "vasp_tool" in routed, "vasp 应路由到 vasp"
    assert "lammps_tool" not in routed

    # 2b. 模糊关键词 (第一性原理) 已删除 → 应 fallback 到空列表
    routed = route_tools("用第一性原理算一下", AVAILABLE)
    assert routed == [], "模糊关键词删除后应 fallback 到空列表"

    # 3. LAMMPS / MD
    routed = route_tools("run molecular dynamics simulation", AVAILABLE)
    assert "lammps_tool" in routed
    assert "vasp_tool" not in routed, "MD task 不应暴露 vasp_tool"

    # 4. 中文 — 分子动力学
    routed = route_tools("做分子动力学模拟", AVAILABLE)
    assert "lammps_tool" in routed

    # 5. 多关键词 union — VASP + LAMMPS
    routed = route_tools("run vasp calc and lammps md", AVAILABLE)
    assert "vasp_tool" in routed, "多关键词应 union"
    assert "lammps_tool" in routed
    assert "gromacs_tool" not in routed

    # 6. 无关键词命中 → 空列表 (caller fallback)
    routed = route_tools("hello world what's up", AVAILABLE)
    assert routed == [], f"无命中应返回空, got {routed}"

    # 7. 工具不存在于 available → 静默跳过
    routed = route_tools("vasp calc", ["vasp_tool", "structure_tool"])
    assert "qe_tool" not in routed, "不在 available 的工具应跳过"
    # available 只有 vasp_tool + structure_tool, 其他工具应被静默跳过
    assert set(routed) == {"vasp_tool", "structure_tool"}, \
        f"应只保留 available 里的工具, got {routed}"

    # 8. 空 task → 空列表
    assert route_tools("", AVAILABLE) == []
    assert route_tools(None, AVAILABLE) == []  # type: ignore[arg-type]

    # 9. 空 available → 空列表
    assert route_tools("vasp calc", []) == []

    # 10. MD 带空格关键词 (" md ")
    routed = route_tools("run md simulation of water", AVAILABLE)
    assert "lammps_tool" in routed, "' md ' 关键词应命中"

    # 11. categorize_tool
    assert categorize_tool("vasp_tool") == "vasp"
    assert categorize_tool("lammps_tool") == "lammps"
    assert categorize_tool("ml_potential_tool") == "ml"
    assert categorize_tool("web_search_tool") == "web"
    assert categorize_tool("unknown_tool") == "unknown"

    # 12. Gaussian / ORCA
    routed = route_tools("gaussian dft optimization", AVAILABLE)
    assert "gaussian_tool" in routed
    assert "vasp_tool" not in routed

    routed = route_tools("orca single point energy", AVAILABLE)
    assert "orca_tool" in routed
    assert "gaussian_tool" not in routed

    # 13. Abaqus / FEA
    routed = route_tools("abaqus stress analysis", AVAILABLE)
    assert "abaqus_tool" in routed
    assert "vasp_tool" not in routed

    routed = route_tools("有限元应力分析", AVAILABLE)
    assert "abaqus_tool" in routed, "有限元应路由到 abaqus"

    # 14. ML potential
    routed = route_tools("train ml potential for Si", AVAILABLE)
    assert "ml_potential_tool" in routed
    assert "structure_tool" in routed

    # 15. 文献检索 (多关键词 union: search + literature + vasp)
    routed = route_tools("search literature for vasp papers", AVAILABLE)
    # 多关键词: search (web) + literature (web) + vasp (vasp) → union
    assert "web_search_tool" in routed or "literature_tool" in routed
    assert "vasp_tool" in routed, "vasp 关键词应 union 进来"

    print(f"15. 文献检索 union: {len(routed)} tools")

    # 额外: 所有 TOOL_CATEGORIES 的工具名都有效
    for cat, tools in TOOL_CATEGORIES.items():
        assert isinstance(tools, list)
        assert all(isinstance(t, str) for t in tools)

    # 额外: TASK_KEYWORD_MAP 所有 category 都在 TOOL_CATEGORIES 里
    for kw, cat in TASK_KEYWORD_MAP.items():
        assert cat in TOOL_CATEGORIES, f"keyword {kw} → category {cat} 不在 TOOL_CATEGORIES"

    print(f"task_tool_router selfcheck All passed "
          f"({len(TOOL_CATEGORIES)} categories, {len(TASK_KEYWORD_MAP)} keywords)")
