"""RAG 检索结果与 Provenance 联动 — 搜索知识库时附带相关工具的最近产出.

agent 搜知识库时, 如果检索到的文档提到某个工具 (如 VASP), 同时从
ProvenanceRegistry 拉取该工具最近产出的文件和关键属性, 附加到检索结果里.
这样 agent 不用翻对话历史就能知道 "这个工具最近产出了什么文件, 能量多少,
收敛了没有".
"""

from __future__ import annotations

import logging
from typing import Any

from huginn.hooks import HookContext

logger = logging.getLogger(__name__)

# 文档/查询中常出现的工具别名 → ProvenanceRegistry 里的工具名
# key 是用户文档里习惯写法, value 是 registry.register(produced_by=...) 用的名字
_TOOL_ALIASES: dict[str, str] = {
    "vasp": "vasp_tool",
    "lammps": "lammps_tool",
    "gaussian": "gaussian_tool",
    "orca": "orca_tool",
    "qe": "qe_tool",
    "quantum espresso": "qe_tool",
    "cp2k": "cp2k_tool",
    "gromacs": "gromacs_tool",
    "abaqus": "abaqus_tool",
    "comsol": "comsol_tool",
    "elmer": "elmer_tool",
    "fenics": "fenics_tool",
    "openfoam": "openfoam_tool",
}


def _detect_tools(text: str) -> list[str]:
    """从文本中识别提到的工具名, 返回 registry 里的工具名列表.

    按别名长度降序匹配, 避免 "qe" 误命中包含这两个字母的单词.
    """
    text_lower = text.lower()
    found: list[str] = []
    seen: set[str] = set()
    # 长别名优先, "quantum espresso" 要比 "qe" 先匹配
    for alias in sorted(_TOOL_ALIASES, key=len, reverse=True):
        tool_name = _TOOL_ALIASES[alias]
        if alias in text_lower and tool_name not in seen:
            seen.add(tool_name)
            found.append(tool_name)
    return found


def _extract_result_text(result: dict) -> str:
    """从单个检索结果中提取文本内容.

    KB 路径返回 "text" 字段, VectorStore 路径返回 "document" 字段, 都兜底.
    """
    return str(result.get("document") or result.get("text") or "")


def enhance_rag_results(
    query: str, results: list[dict], top_k: int = 3
) -> list[dict]:
    """增强 RAG 检索结果, 附带相关工具的 provenance 上下文.

    从 query 和 results 文本中识别工具名, 从 ProvenanceRegistry 拉取
    这些工具最近的产出 (文件路径、key_properties、溯源链摘要),
    附加到每个 result 的 "provenance_context" 字段.

    不修改原始 results, 返回增强后的浅拷贝列表.
    """
    if not results:
        return []

    # 延迟导入避免循环依赖
    from huginn.provenance.registry import ProvenanceRegistry

    reg = ProvenanceRegistry.shared()

    # 把 query 和所有结果文本拼一起扫工具名
    combined = query + " " + " ".join(_extract_result_text(r) for r in results)
    tools = _detect_tools(combined)

    provenance_context: dict[str, Any] = {}
    if tools:
        provenance_context["detected_tools"] = tools
        recent_outputs: list[dict[str, Any]] = []
        for tool in tools:
            entries = reg.find_by_tool(tool)
            # 按时间倒序取最近 top_k 条
            entries = sorted(entries, key=lambda e: e.produced_at, reverse=True)
            for entry in entries[:top_k]:
                output_info: dict[str, Any] = {
                    "tool": tool,
                    "file_path": entry.file_path,
                    "file_format": entry.file_format,
                    "key_properties": entry.key_properties,
                }
                # 溯源链摘要: 从这个文件往上追几层
                lineage = reg.get_lineage(entry.file_path, depth=3)
                if lineage:
                    output_info["lineage_summary"] = " -> ".join(
                        f"{e.produced_by}:{e.file_format or '?'}"
                        for e in lineage
                    )
                recent_outputs.append(output_info)
        if recent_outputs:
            provenance_context["recent_outputs"] = recent_outputs

    # 给每个 result 附加 provenance_context, 不动原始 dict
    # ARGUS: RAG 检索结果来自外部知识库, 标 source_class=external_content.
    # 下游 PhaseGate / RedTeam 可据此降级或加强审查.
    enhanced: list[dict] = []
    for r in results:
        item = dict(r)
        item["source_class"] = "external_content"
        if provenance_context:
            item["provenance_context"] = provenance_context
        enhanced.append(item)
    return enhanced


async def rag_provenance_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: rag_tool 搜索完成后, 用 provenance 增强结果.

    把增强后的结果写入 ctx.metadata["enhanced_results"], 不 block.
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

        # 从 args 拿 query, 拿不到就从 result 里取
        args = ctx.args
        if hasattr(args, "model_dump"):
            args = args.model_dump()
        args = args if isinstance(args, dict) else {}
        query = args.get("query") or data.get("query") or ""

        enhanced = enhance_rag_results(query, search_results)
        ctx.metadata["enhanced_results"] = enhanced
    except Exception:
        # 增强失败不能影响主流程
        logger.debug("rag_provenance_hook failed (non-fatal)", exc_info=True)
    return None
