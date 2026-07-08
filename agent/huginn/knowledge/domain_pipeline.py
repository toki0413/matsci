"""领域标签与计算管线联动 — 根据查询领域推荐标准计算流程.

把知识库的 DOMAIN_TAG_TREE (合金/半导体/催化/能源材料/生物材料/机械工程)
跟 PipelineStage 联动. agent 搜索知识库时, 根据查询内容识别领域,
同时给出该领域推荐的计算管线和注意事项, 让 agent 知道 "这类材料该走什么流程".

和 provenance/pipeline.py 的 SimulationPipeline 区别:
  - SimulationPipeline: 事件驱动, 跟踪实际执行了什么, 建议下一步
  - DomainPipeline: 领域驱动, 根据材料类型推荐完整流程, 纯指导性
"""

from __future__ import annotations

import logging
from typing import Any

from huginn.hooks import HookContext
from huginn.provenance.pipeline import PipelineStage

logger = logging.getLogger(__name__)

# 每个领域推荐的标准计算阶段
# 合金/机械工程走力学线 (MECHANICAL), 半导体走电子结构线 (STATIC→PROPERTIES),
# 催化走表面性质线, 能源材料/生物材料走 MD 线
DOMAIN_PIPELINE_MAP: dict[str, list[PipelineStage]] = {
    "合金": [
        PipelineStage.STRUCTURE,
        PipelineStage.RELAX,
        PipelineStage.MECHANICAL,
        PipelineStage.ANALYSIS,
    ],
    "半导体": [
        PipelineStage.STRUCTURE,
        PipelineStage.RELAX,
        PipelineStage.STATIC,
        PipelineStage.PROPERTIES,
    ],
    "催化": [
        PipelineStage.STRUCTURE,
        PipelineStage.RELAX,
        PipelineStage.PROPERTIES,
    ],
    "能源材料": [
        PipelineStage.STRUCTURE,
        PipelineStage.RELAX,
        PipelineStage.MD,
        PipelineStage.PROPERTIES,
    ],
    "生物材料": [
        PipelineStage.STRUCTURE,
        PipelineStage.RELAX,
        PipelineStage.MD,
        PipelineStage.ANALYSIS,
    ],
    "机械工程": [
        PipelineStage.STRUCTURE,
        PipelineStage.RELAX,
        PipelineStage.MECHANICAL,
        PipelineStage.ANALYSIS,
    ],
}

# 查询关键词到领域的映射, 用于从 RAG 查询中识别用户关心的领域
# 只收一级领域的关键词, 子领域由 store.py 的 auto_tag 处理
DOMAIN_KEYWORD_HINTS: dict[str, list[str]] = {
    "合金": [
        "合金", "alloy", "superalloy", "高熵", "HEA",
        "金属间化合物", "intermetallic", "镍基", "钛合金", "铝合金", "镁合金",
    ],
    "半导体": [
        "半导体", "semiconductor", "带隙", "band gap", "能带", "band structure",
        "dos", "态密度", "掺杂", "doping", "载流子", "carrier", "GaN", "SiC",
    ],
    "催化": [
        "催化", "catalys", "吸附", "adsorption", "活性位点", "active site",
        "ORR", "OER", "HER", "电催化", "光催化", "热催化",
    ],
    "能源材料": [
        "电池", "battery", "储能", "储氢", "hydrogen storage",
        "超级电容", "supercapacitor", "扩散", "diffusion",
        "正极", "cathode", "负极", "anode", "电解质", "electrolyte",
    ],
    "生物材料": [
        "生物材料", "biomaterial", "生物相容", "biocompatib",
        "植入", "implant", "生物陶瓷", "bioceramic",
        "羟基磷灰石", "PLA", "壳聚糖", "chitosan",
    ],
    "机械工程": [
        "应力", "stress", "疲劳", "fatigue", "轧制", "rolling",
        "烧结", "sintering", "增材制造", "additive manufacturing",
        "3D打印", "热处理", "heat treatment", "粉末冶金",
        "塑性加工", "锻造", "forging", "磨损", "wear",
    ],
}

# 每个阶段推荐的工具, 给 agent 选工具用
_STAGE_TOOLS: dict[PipelineStage, list[str]] = {
    PipelineStage.STRUCTURE: ["structure_tool", "packing_tool", "convergence_test_tool"],
    PipelineStage.RELAX: ["vasp_tool", "qe_tool", "cp2k_tool", "lammps_tool"],
    PipelineStage.STATIC: ["vasp_tool", "qe_tool", "cp2k_tool"],
    PipelineStage.PROPERTIES: ["vasp_tool", "characterization_tool"],
    PipelineStage.MECHANICAL: ["mechanical_tool", "vasp_tool"],
    PipelineStage.MD: ["lammps_tool", "gromacs_tool"],
    PipelineStage.ANALYSIS: ["compute_msd", "compute_rdf", "characterization_tool"],
}

# 领域特定的注意事项, 帮 agent 避坑
_DOMAIN_NOTES: dict[str, list[str]] = {
    "合金": [
        "力学性质计算需检查弹性常数矩阵正定性",
        "高熵合金建议验证固溶体稳定性 (形成能 > 0 可能分层)",
        "无序合金模型建议用 SQS 处理占据无序",
    ],
    "半导体": [
        "需要检查带隙是直接带隙还是间接带隙",
        "K 点网格需要足够密以收敛带隙值",
        "DFT+U 或杂化泛函常用于修正 PBE 带隙低估问题",
    ],
    "催化": [
        "关注反应物在活性位点的吸附能 (d-band center 理论)",
        "建议计算反应路径和过渡态 (NEB)",
        "表面模型需注意 slab 层数和真空层厚度",
    ],
    "能源材料": [
        "MD 模拟需关注离子扩散系数 (MSD/RDF 分析)",
        "电池材料建议计算迁移势垒 (NEB)",
        "注意电化学窗口和界面稳定性",
    ],
    "生物材料": [
        "MD 模拟需使用适配生物体系的力场 (CHARMM/AMBER)",
        "关注溶剂化效应和 pH 环境",
        "生物相容性需结合实验数据验证",
    ],
    "机械工程": [
        "力学性质计算需关注应力-应变曲线和屈服强度",
        "增材制造模拟需考虑热历史和残余应力",
        "疲劳分析建议用 S-N 曲线方法",
    ],
}


def detect_domain_from_query(query: str) -> str | None:
    """从查询文本中识别材料领域, 返回领域名或 None.

    按 DOMAIN_KEYWORD_HINTS 做关键词匹配, 命中第一个领域就返回.
    """
    if not query:
        return None
    query_lower = query.lower()
    for domain, keywords in DOMAIN_KEYWORD_HINTS.items():
        if any(kw.lower() in query_lower for kw in keywords):
            return domain
    return None


def get_pipeline_for_domain(domain: str) -> list[PipelineStage]:
    """返回该领域的推荐计算阶段列表, 未知领域返回空列表."""
    return DOMAIN_PIPELINE_MAP.get(domain, [])


def get_workflow_guidance(query: str, rag_results: list[dict]) -> dict:
    """综合返回领域识别、推荐管线、工具建议和领域注意事项.

    先从 query 识别领域, 查询没命中再从检索结果文本里找.
    返回 dict 包含 detected_domain / recommended_pipeline / relevant_tools /
    domain_specific_notes 四个字段.
    """
    domain = detect_domain_from_query(query)

    # 查询没命中, 再从检索结果文本里找
    if domain is None and rag_results:
        for r in rag_results:
            text = str(r.get("document") or r.get("text") or "")
            domain = detect_domain_from_query(text)
            if domain is not None:
                break

    pipeline = get_pipeline_for_domain(domain) if domain else []

    # 每个阶段对应的推荐工具
    relevant_tools: dict[str, list[str]] = {}
    for stage in pipeline:
        relevant_tools[stage.value] = _STAGE_TOOLS.get(stage, [])

    return {
        "detected_domain": domain,
        "recommended_pipeline": [s.value for s in pipeline],
        "relevant_tools": relevant_tools,
        "domain_specific_notes": _DOMAIN_NOTES.get(domain, []) if domain else [],
    }


async def workflow_guidance_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: rag_tool 搜索完成后, 附带领域管线建议.

    从 ctx 中获取查询和检索结果, 调用 get_workflow_guidance,
    写入 ctx.metadata["workflow_guidance"], 不 block.
    """
    if ctx.tool_name != "rag_tool":
        return None
    if ctx.error is not None:
        return None
    try:
        result = ctx.result if isinstance(ctx.result, dict) else {}
        if result.get("error"):
            return None
        data = result.get("result", result)
        if not isinstance(data, dict):
            return None
        search_results = data.get("results")
        if not search_results or not isinstance(search_results, list):
            return None

        args = ctx.args
        if hasattr(args, "model_dump"):
            args = args.model_dump()
        args = args if isinstance(args, dict) else {}
        query = args.get("query") or data.get("query") or ""

        guidance = get_workflow_guidance(query, search_results)
        # 只在识别到领域时才写入, 避免无意义 guidance 占用 metadata
        if guidance["detected_domain"] is not None:
            ctx.metadata["workflow_guidance"] = guidance
    except Exception:
        logger.debug("workflow_guidance_hook failed (non-fatal)", exc_info=True)
    return None
