"""计算结果自动摄入知识库 + Hook 失败蒸馏联动.

成功路径: 仿真工具通过 hook 检查后, 关键结果转成 DistilledKnowledge 摄入 KB,
后续检索能复用这次经验.
失败路径: science hook block 了调用时, 失败教训蒸馏成知识, 避免重复踩坑.

两个 POST_TOOL_USE hook (calculation_ingest_hook / hook_failure_hook) 都不 block.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from huginn.hooks import HookContext

logger = logging.getLogger(__name__)


# ── 模块级懒加载单例 ─────────────────────────────────────────────
# KB 和 distiller 都在首次访问时才创建, 避免启动时硬依赖 chromadb 等.
# CalculationToKnowledge 如果传了自定义实例就用自定义的, 否则回退到这里.

_kb: Any = None
_distiller: Any = None


def _get_kb() -> Any:
    global _kb
    if _kb is None:
        try:
            from huginn.knowledge.store import get_knowledge_base

            _kb = get_knowledge_base()
        except Exception:
            logger.debug("KnowledgeBase 不可用, 跳过 KB 写入", exc_info=True)
    return _kb


def _get_distiller() -> Any:
    global _distiller
    if _distiller is None:
        try:
            from huginn.evolution.knowledge_distiller import KnowledgeDistiller

            _distiller = KnowledgeDistiller()
        except Exception:
            logger.debug("KnowledgeDistiller 不可用, 跳过蒸馏", exc_info=True)
    return _distiller


# ── 字段提取辅助 ─────────────────────────────────────────────────


def _result_data(tool_output: Any) -> dict:
    """从 tool_output 取 result dict.

    序列化后的结构一般是 {"result": {...}} 或 {"error": "..."},
    也可能是裸 dict, 统一处理.
    """
    if not isinstance(tool_output, dict):
        return {}
    data = tool_output.get("result", tool_output)
    return data if isinstance(data, dict) else {}


def _pick(sources: list[dict], *keys: str) -> Any:
    """从多个 dict 里按 key 顺序找第一个非 None 的值."""
    for d in sources:
        if not isinstance(d, dict):
            continue
        for k in keys:
            v = d.get(k)
            if v is not None:
                return v
    return None


def _fmt(val: Any) -> str:
    """值转简短字符串, None -> 'N/A'."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.6g}"
    return str(val)


def _params_summary(tool_input: dict) -> str:
    """把 tool_input 关键参数拼成短摘要, 给 ERROR LESSON 用."""
    if not isinstance(tool_input, dict) or not tool_input:
        return "default"
    parts = []
    for k in (
        "action", "encut", "kpoints", "functional", "method",
        "basis_set", "timestep", "n_steps", "steps",
    ):
        v = tool_input.get(k)
        if v is not None:
            parts.append(f"{k}={_fmt(v)}")
    return ", ".join(parts) if parts else "default"


def _key_props_summary(data: dict) -> str:
    """把 result 关键属性拼成短摘要, 给通用模板用."""
    if not isinstance(data, dict) or not data:
        return "no properties"
    parts = []
    for k in (
        "energy", "total_energy", "band_gap", "converged",
        "formula", "final_temp", "lattice_constant", "volume",
    ):
        v = data.get(k)
        if v is not None:
            parts.append(f"{k}={_fmt(v)}")
    return ", ".join(parts) if parts else "no key properties"


def _infer_software(tool_name: str) -> str:
    """从工具名推断软件名, 给 distiller 分类用."""
    name = tool_name.lower()
    if "vasp" in name:
        return "vasp"
    if "lammps" in name:
        return "lammps"
    if "gaussian" in name:
        return "gaussian"
    if "orca" in name:
        return "orca"
    if "qe" in name or "quantum" in name:
        return "qe"
    if "cp2k" in name:
        return "cp2k"
    if "gromacs" in name:
        return "gromacs"
    return "general"


def _extract_suggestion(block_reason: str) -> str:
    """从 block_reason 里提取建议.

    science hook 的 block_reason 通常含 'Consider ...' 或 '考虑...',
    直接取那段; 取不到就给通用建议.
    """
    if not block_reason:
        return "Review input parameters and retry."
    lower = block_reason.lower()
    idx = lower.find("consider")
    if idx >= 0:
        return block_reason[idx:].rstrip(".")
    idx = block_reason.find("考虑")
    if idx >= 0:
        return block_reason[idx:]
    return "Review input parameters and retry."


# ── 知识文本生成 ─────────────────────────────────────────────────


def _build_vasp_text(tool_input: dict, data: dict) -> str:
    action = _fmt(_pick([tool_input, data], "action"))
    formula = _fmt(_pick([tool_input, data], "formula", "structure", "poscar_path"))
    encut = _fmt(_pick([tool_input], "encut", "ENCUT"))
    kpoints = _fmt(_pick([tool_input], "kpoints", "kpoints_grid"))
    energy = _fmt(_pick([data], "total_energy", "energy", "E0", "free_energy"))
    converged = _fmt(_pick([data], "converged"))
    band_gap = _fmt(_pick([data], "band_gap"))
    return (
        f"VASP {action} calculation for {formula}: "
        f"encut={encut}, kpoints={kpoints}, energy={energy} eV, "
        f"converged={converged}, band_gap={band_gap} eV"
    )


def _build_lammps_text(tool_input: dict, data: dict) -> str:
    action = _fmt(_pick([tool_input, data], "action"))
    timestep = _fmt(_pick([tool_input], "timestep", "dt"))
    n_steps = _fmt(_pick([tool_input, data], "n_steps", "steps", "num_steps"))
    final_temp = _fmt(_pick([data], "final_temp", "temperature", "temp"))
    final_energy = _fmt(_pick([data], "final_energy", "energy", "total_energy"))
    return (
        f"LAMMPS {action} simulation: timestep={timestep}, "
        f"n_steps={n_steps}, final_temp={final_temp} K, "
        f"final_energy={final_energy} eV"
    )


def _build_qchem_text(software: str, tool_input: dict, data: dict) -> str:
    method = _fmt(_pick([tool_input], "method", "functional"))
    basis = _fmt(_pick([tool_input], "basis_set", "basis"))
    energy = _fmt(_pick([data], "energy", "total_energy"))
    converged = _fmt(_pick([data], "converged"))
    return (
        f"{software} {method} calculation: basis={basis}, "
        f"energy={energy} Hartree, converged={converged}"
    )


def _build_knowledge_text(tool_name: str, tool_input: dict, tool_output: Any) -> str:
    """根据工具类型生成知识文本, 拿不到的字段填 N/A."""
    name_lower = tool_name.lower()
    data = _result_data(tool_output)
    tool_input = tool_input if isinstance(tool_input, dict) else {}

    if "vasp" in name_lower:
        return _build_vasp_text(tool_input, data)
    if "lammps" in name_lower:
        return _build_lammps_text(tool_input, data)
    if "gaussian" in name_lower:
        return _build_qchem_text("Gaussian", tool_input, data)
    if "orca" in name_lower:
        return _build_qchem_text("ORCA", tool_input, data)
    # 通用模板
    return f"{tool_name} produced: {_key_props_summary(data)}"


# ── 核心类 ───────────────────────────────────────────────────────


class CalculationToKnowledge:
    """把工具计算结果转成知识, 摄入 KB 和 distiller."""

    def __init__(self, kb: Any = None, distiller: Any = None) -> None:
        # 传了就用传入的, 没传回退到模块级单例 (懒加载)
        self._kb = kb
        self._distiller = distiller

    def _get_kb(self) -> Any:
        if self._kb is not None:
            return self._kb
        return _get_kb()

    def _get_distiller(self) -> Any:
        if self._distiller is not None:
            return self._distiller
        return _get_distiller()

    def ingest_calculation(
        self,
        tool_name: str,
        tool_input: dict,
        tool_output: Any,
        provenance_entry: Any = None,
    ) -> str | None:
        """从工具调用提取关键信息, 生成知识文本, 摄入 KB.

        返回 doc_id 或 None (没有 KB / 文本为空时返回 None).
        """
        text = _build_knowledge_text(tool_name, tool_input, tool_output)
        if not text.strip():
            return None

        doc_id: str | None = None

        # 1) KB 直接写入, 立即可检索
        kb = self._get_kb()
        if kb is not None:
            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                result = kb.add_text(
                    text,
                    filename=f"calc_{tool_name}_{ts}",
                    metadata={
                        "source": "auto_ingest",
                        "tool_name": tool_name,
                    },
                )
                doc_id = result.get("doc_id") or None
            except Exception:
                logger.debug("kb.add_text 失败 (非致命)", exc_info=True)

        # 2) distiller 也记一条 success_pattern
        distiller = self._get_distiller()
        if distiller is not None:
            try:
                from huginn.evolution.knowledge_distiller import DistilledKnowledge

                kid = f"succ_{tool_name}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                # 已存在就跳过, 避免重复写入
                if not any(k.knowledge_id == kid for k in distiller.knowledge_base):
                    evidence = (
                        [provenance_entry.file_path]
                        if provenance_entry is not None
                        else ["unknown"]
                    )
                    dk = DistilledKnowledge(
                        knowledge_id=kid,
                        content=text,
                        source_type="success_pattern",
                        source_evidence=evidence,
                        confidence=0.7,
                        category=f"calculation_{tool_name}",
                        tags=["success", "pattern", tool_name],
                    )
                    distiller.knowledge_base.append(dk)
                    distiller._save()
            except Exception:
                logger.debug("distiller 记录失败 (非致命)", exc_info=True)

        return doc_id


# ── Hook 失败 → 知识蒸馏 ─────────────────────────────────────────


def distill_hook_failure(
    tool_name: str,
    tool_input: dict,
    block_reason: str,
    metadata: dict,
) -> None:
    """把 hook block 的失败教训蒸馏成知识.

    - 调 distiller.distill_error_lessons 记一条 error_lesson
    - 有 KB 时直接 add_text 一条 ERROR LESSON, 立即可检索
    """
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    params = _params_summary(tool_input)
    software = _infer_software(tool_name)
    calc_type = str(tool_input.get("action", "general"))

    # 1) distiller 蒸馏 error_lesson
    distiller = _get_distiller()
    if distiller is not None:
        try:
            distiller.distill_error_lessons([
                {
                    "tool_name": tool_name,
                    "error_message": block_reason,
                    "software": software,
                    "calculation_type": calc_type,
                    "session_id": metadata.get("thread_id", "unknown"),
                    "tool_input": tool_input,
                }
            ])
        except Exception:
            logger.debug("distill_error_lessons 失败 (非致命)", exc_info=True)

    # 2) KB 直接写入 ERROR LESSON, 立即可检索
    kb = _get_kb()
    if kb is not None:
        suggested_fix = _extract_suggestion(block_reason)
        lesson_text = (
            f"ERROR LESSON: {tool_name} with {params} failed: "
            f"{block_reason}. Suggested fix: {suggested_fix}"
        )
        try:
            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            kb.add_text(
                lesson_text,
                filename=f"error_lesson_{tool_name}_{ts}",
                metadata={
                    "source": "hook_failure",
                    "tool_name": tool_name,
                    "block_reason": block_reason[:200],
                },
            )
        except Exception:
            logger.debug("kb.add_text 写 ERROR LESSON 失败 (非致命)", exc_info=True)


# ── 模块级单例 (钩子用) ──────────────────────────────────────────

_ingester: CalculationToKnowledge | None = None


def _get_ingester() -> CalculationToKnowledge:
    global _ingester
    if _ingester is None:
        _ingester = CalculationToKnowledge()
    return _ingester


# ── Sobko 知识库蒸馏 ─────────────────────────────────────────────
# ponytail: 把 Sobko_MCP_project/normalized/chunks.jsonl 的 11304 chunk
# 灌进 huginn KB. Sobko 自己有 BM25+dense 索引可用, 这层是给 huginn
# 内部 RAG 用的 fallback, 不替代 Sobko MCP server.
# ceiling: 11304 chunk 全量入库会很慢 (~5min), 且 Sobko 的 chunk 文本
# 过 huginn 的 _section_aware_chunk 会二次分块. 升级路径是直接调底层
# chromadb collection.add, 绕过 add_text 的二次分块.


def ingest_sobko_chunks(
    sobko_root: str,
    *,
    kb: Any = None,
    limit: int | None = None,
    batch_log_every: int = 500,
) -> int:
    """把 Sobko normalized/chunks.jsonl 灌进 huginn KB.

    Args:
        sobko_root: Sobko_MCP_project 仓库根目录
        kb: 可选, 不传就用模块级懒加载单例
        limit: 只灌前 N 条 (调试用)
        batch_log_every: 每多少条 log 一次进度

    Returns:
        成功入库的 chunk 数.
    """
    import json as _json
    from pathlib import Path as _Path

    sobko_root_path = _Path(sobko_root)
    chunks_file = sobko_root_path / "normalized" / "chunks.jsonl"
    if not chunks_file.exists():
        logger.warning("Sobko chunks.jsonl 不存在: %s", chunks_file)
        return 0

    if kb is None:
        kb = _get_kb()
    if kb is None:
        logger.warning("KnowledgeBase 不可用, 无法蒸馏 Sobko chunks")
        return 0

    ingested = 0
    with chunks_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                chunk = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            # Sobko chunk 字段: chunk_id / source_id / text / section / page ...
            text = chunk.get("text") or chunk.get("content") or ""
            if not text.strip():
                continue
            source_id = chunk.get("source_id") or chunk.get("source") or f"sobko_{i}"
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or f"chunk_{i}"
            try:
                result = kb.add_text(
                    text,
                    filename=f"sobko_{source_id}_{chunk_id}",
                    metadata={
                        "source": "sobko_kb",
                        "source_id": source_id,
                        "chunk_id": chunk_id,
                        "section": chunk.get("section", ""),
                    },
                )
                if result.get("chunks", 0) > 0:
                    ingested += 1
            except Exception:
                logger.debug("Sobko chunk %s 入库失败 (非致命)", chunk_id, exc_info=True)
            if batch_log_every and (i + 1) % batch_log_every == 0:
                logger.info("Sobko 蒸馏进度: %d chunks 处理, %d 成功", i + 1, ingested)
    return ingested


# ── POST_TOOL_USE hooks ─────────────────────────────────────────


async def calculation_ingest_hook(ctx: HookContext) -> HookContext | None:
    """成功路径: 工具通过 hook 检查后, 把计算结果摄入知识库. 不 block.

    只处理成功的调用: 没被 hook block, 没抛异常, 结果里也没 error.
    """
    try:
        # 被 hook block 的走 hook_failure_hook
        if ctx.metadata.get("blocked_by_hook"):
            return None
        # 工具抛异常或返回 error 的不算成功
        if ctx.error is not None:
            return None
        result = ctx.result if isinstance(ctx.result, dict) else {}
        if result.get("error"):
            return None

        tool_input = ctx.args if isinstance(ctx.args, dict) else {}
        _get_ingester().ingest_calculation(ctx.tool_name, tool_input, ctx.result)

        # G4: 把 visual_primitives 单独 ingest 到 KB, 让视觉经验能被 RAG 召回.
        # visual_hook 给 result 塞 _visual_primitives 字段, 之前只流向 memory/KG,
        # 不进 RAG KB → 这里补上. 非 block, 失败只 debug.
        _vis = result.get("_visual_primitives") if isinstance(result, dict) else None
        if not _vis:
            _res_inner = result.get("result") if isinstance(result, dict) else None
            _vis = _res_inner.get("_visual_primitives") if isinstance(_res_inner, dict) else None
        if _vis and isinstance(_vis, str) and _vis.strip():
            ingest_visual_primitives(ctx.tool_name, _vis, tool_input)
    except Exception:
        logger.debug("calculation_ingest_hook 失败 (非致命)", exc_info=True)
    return None


def ingest_visual_primitives(
    tool_name: str,
    visual_primitives: str,
    tool_input: dict | None = None,
) -> str | None:
    """G4: 把 visual_primitives 作为独立 KB 条目摄入, 让 RAG 能召回视觉经验.

    之前 visual_primitives 只流向 memory/KG/hippocampus, 不进 RAG KB.
    这里补上 — visual_hook 生成的 <point>/<box> 原语文本直接 add_text,
    下次 agent 查 KB 时能按 "band peak" / "EDS coverage" 等关键词召回.

    ponytail: 直接 add_text, 不上 distiller. 升级路径: 调 G8 distill_visual_lessons
    做聚合蒸馏. 当前先把单条视觉经验进 RAG, 能召回就够.
    """
    if not visual_primitives or not visual_primitives.strip():
        return None
    kb = _get_kb()
    if kb is None:
        return None
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 文本加前缀让 RAG 检索时能识别这是视觉经验
    text = f"# Visual Primitives ({tool_name})\n{visual_primitives}"
    try:
        result = kb.add_text(
            text,
            filename=f"visual_{tool_name}_{ts}",
            metadata={
                "source": "visual_primitives",
                "tool_name": tool_name,
                "content_type": "visual_primitives",
            },
        )
        return result.get("doc_id")
    except Exception:
        logger.debug("G4 ingest_visual_primitives 失败 (非致命)", exc_info=True)
        return None


async def hook_failure_hook(ctx: HookContext) -> HookContext | None:
    """失败路径: 被 science hook block 的调用, 蒸馏失败教训. 不 block.

    只处理被 block 的调用, 已经被前面 block 过了, 这里只做知识蒸馏.
    """
    try:
        if not ctx.metadata.get("blocked_by_hook"):
            return None
        block_reason = ctx.metadata.get("block_reason", "unknown reason")
        tool_input = ctx.args if isinstance(ctx.args, dict) else {}
        distill_hook_failure(
            ctx.tool_name, tool_input, block_reason, ctx.metadata
        )
    except Exception:
        logger.debug("hook_failure_hook 失败 (非致命)", exc_info=True)
    return None


# ── 自检 ─────────────────────────────────────────────────────────
# 不依赖 chromadb / sentence-transformers, 只验证文本生成逻辑

if __name__ == "__main__":
    # VASP
    vasp_text = _build_knowledge_text(
        "vasp_tool",
        {"action": "relax", "encut": 520, "kpoints": "4 4 4"},
        {"result": {"total_energy": -12.34, "converged": True, "band_gap": 1.5, "formula": "Si"}},
    )
    assert "VASP relax calculation for Si" in vasp_text
    assert "encut=520" in vasp_text
    assert "energy=-12.34" in vasp_text
    assert "converged=True" in vasp_text
    print("VASP:", vasp_text)

    # LAMMPS
    lammps_text = _build_knowledge_text(
        "lammps_tool",
        {"action": "nvt", "timestep": 0.001, "n_steps": 10000},
        {"result": {"final_temp": 300.0, "final_energy": -456.7}},
    )
    assert "LAMMPS nvt simulation" in lammps_text
    assert "timestep=0.001" in lammps_text
    assert "n_steps=10000" in lammps_text
    assert "final_temp=300" in lammps_text
    print("LAMMPS:", lammps_text)

    # Gaussian
    gauss_text = _build_knowledge_text(
        "gaussian_tool",
        {"method": "B3LYP", "basis_set": "6-31G*"},
        {"result": {"energy": -76.5, "converged": True}},
    )
    assert "Gaussian B3LYP calculation" in gauss_text
    assert "basis=6-31G*" in gauss_text
    assert "energy=-76.5" in gauss_text
    print("Gaussian:", gauss_text)

    # ORCA
    orca_text = _build_knowledge_text(
        "orca_tool",
        {"method": "PBE0", "basis_set": "def2-TZVP"},
        {"result": {"energy": -100.2, "converged": True}},
    )
    assert "ORCA PBE0 calculation" in orca_text
    print("ORCA:", orca_text)

    # 通用
    generic_text = _build_knowledge_text(
        "custom_tool",
        {},
        {"result": {"energy": -1.0, "volume": 50.0}},
    )
    assert "custom_tool produced:" in generic_text
    assert "energy=-1" in generic_text
    print("Generic:", generic_text)

    # 建议提取
    assert "Consider" in _extract_suggestion(
        "VASP did not converge. Consider increasing NSW."
    )
    assert "Review" in _extract_suggestion("something went wrong")

    # 软件推断
    assert _infer_software("vasp_tool") == "vasp"
    assert _infer_software("lammps_tool") == "lammps"
    assert _infer_software("unknown_tool") == "general"

    # Sobko 蒸馏: 不存在的路径应返回 0, 不崩
    n = ingest_sobko_chunks("/nonexistent/sobko_root", limit=10)
    assert n == 0, f"不存在的路径应返回 0, 实际 {n}"
    print("Sobko ingest (missing path):", n)

    print("\nAll self-checks passed.")
