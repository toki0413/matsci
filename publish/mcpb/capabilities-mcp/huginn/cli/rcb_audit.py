"""RCB 审计 / 质量检查 — 依赖 rcb_utils."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# === v17: Effort Budget Allocation — per-item time-sliced budget ===
@dataclass
class _ChecklistItem:
    """checklist.md 解析出的单个 item — 用于 per-item budget 分配."""

    idx: int  # 0-based item 序号
    name: str  # "1.1 Symbolic Deduction Engine"
    label: str = ""  # "[EXACT]" / "[VARIANT]" / "" (未标注)
    expected_output: str = ""  # expected_output 行摘要


def _checklist_item_parser(checklist_text: str) -> list[_ChecklistItem] | None:
    """从 checklist.md 抽 items.

    匹配 `## N.M Name` 标题, 提取 label 和 expected_output.
    返回 None 表示解析失败 (调用方 fallback 到 v16.1 无 per-item budget 行为).
    """
    if not checklist_text:
        return None
    import re as _re

    _item_re = _re.compile(r"^#{2,3}\s+(\d+\.\d+)\s+(.+)$", _re.MULTILINE)
    _label_re = _re.compile(r"\*\*Label\*\*:\s*\[([A-Z]+)\]")
    _output_re = _re.compile(r"\*\*expected_output[^*]*\*\*:\s*(.+)")
    items: list[_ChecklistItem] = []
    matches = list(_item_re.finditer(checklist_text))
    if len(matches) < 2:
        return None
    for i, m in enumerate(matches):
        _id, _name = m.group(1), m.group(2).strip()
        _start = m.end()
        _end = matches[i + 1].start() if i + 1 < len(matches) else len(checklist_text)
        _section = checklist_text[_start:_end]
        _lbl_m = _label_re.search(_section)
        _out_m = _output_re.search(_section)
        items.append(
            _ChecklistItem(
                idx=i,
                name=f"{_id} {_name}",
                label=_lbl_m.group(1) if _lbl_m else "",
                expected_output=(_out_m.group(1).strip()[:120] if _out_m else ""),
            )
        )
    return items


def _time_slot_index(elapsed_s: float, per_item_budget_s: float, n_items: int) -> int:
    """wall_clock 切片 → 当前 item index. ponytail: 简单除法, 不重分配剩余预算."""
    if per_item_budget_s <= 0:
        return 0
    idx = int(elapsed_s / per_item_budget_s)
    return min(idx, n_items - 1)


def _derivation_floor_check(derivation_audit: str) -> tuple[bool, str]:
    """v16: derivation 覆盖率 < 50% 即驳回 (item 只被讨论未被执行)."""
    import re as _re

    _dm = _re.search(r"DERIVATION:\s*(\d+)%\s*\((\d+)/(\d+)", derivation_audit)
    if not _dm:
        return True, ""
    _dev_cov = int(_dm.group(1))
    if _dev_cov >= 50:
        return True, ""
    _disc = ""
    for _line in derivation_audit.split("\n"):
        if _line.lower().startswith("discussed_only:"):
            _disc = _line[len("discussed_only:") :].strip()[:300]
            break
    return False, (
        f"derivation={_dev_cov}% < 50%, agent discussed checklist items "
        f"in report but did NOT execute them (no artifacts in outputs/). "
        f"DISCUSSED_ONLY: {_disc}\n"
        f"→ Produce the missing artifacts in outputs/ before claiming TASK COMPLETE."
    )


def _rcb_effort_floor(
    ws: Path,
    checklist: str,
    *,
    min_cov_pct: int = 70,
    derivation_audit: str | None = None,
) -> tuple[bool, str]:
    """P0.3: RCB 跑分路径的 effort floor 硬下限 — 对齐 AV7 autoloop MinEffortFloor.

    复用 _report_coverage_compass 的 keyword 覆盖度, 不达标 → 驳回 TASK COMPLETE.
    v16: 增加 derivation_audit 参数 — 检查 checklist item 是否被执行.
    """
    if not checklist:
        return True, ""
    _report_path = ws / "report" / "report.md"
    if not _report_path.exists():
        return (
            False,
            "report/report.md does NOT exist — write it BEFORE claiming TASK COMPLETE",
        )
    _compass = _report_coverage_compass(ws, checklist)
    if not _compass:
        return True, ""
    import re as _re

    _m = _re.search(r"\((\d+)%\s*—\s*(\d+)/(\d+)", _compass)
    if not _m:
        return True, ""
    _cov = int(_m.group(1))
    if _cov < min_cov_pct:
        _missing = ""
        for _line in _compass.split("\n"):
            if _line.lower().startswith("missing:"):
                _missing = _line[len("missing:") :].strip()
                break
        return False, f"coverage={_cov}% < {min_cov_pct}%, missing: {_missing}"
    if derivation_audit:
        _ok, _msg = _derivation_floor_check(derivation_audit)
        if not _ok:
            return False, _msg
    return True, ""


def _compass_keywords(checklist: str) -> set:
    """从 checklist 抽 keyword 集合: [EXACT] 行尾词 + 常见 metrics 名."""
    keywords = set()
    for line in checklist.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "[exact]" in line.lower():
            idx = line.lower().find("[exact]")
            tail = line[idx + 7 :].strip().split()
            for w in tail[:4]:
                w = w.strip(".,;:()[]")
                if len(w) >= 3:
                    keywords.add(w.lower())
        for metric in (
            "mae",
            "rmse",
            "r²",
            "r2",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ):
            if metric in line.lower():
                keywords.add(metric)
    return keywords


def _render_coverage_compass(
    report_text: str,
    covered: list,
    missing: list,
    total: int,
    cov_pct: int,
) -> list[str]:
    """渲染 compass 文本: 标题 + 占位警告 + covered/missing 段."""
    lines = [
        f"## Report Coverage Compass ({cov_pct}% — {len(covered)}/{total} checklist keywords found in report.md)",
    ]
    if "expected" in report_text and (
        "placeholder" in report_text
        or "tbd" in report_text
        or "not yet" in report_text
        or "n/a" in report_text
    ):
        lines.append(
            "⚠ WARNING: report.md contains 'Expected'/'placeholder'/'TBD' — "
            "these are NOT results. Replace with actual executed metrics or "
            "declare the gap honestly. Judge scores placeholder content as 0."
        )
    if covered:
        lines.append(f"Covered: {', '.join(sorted(covered))}")
    if missing:
        lines.append(f"Missing: {', '.join(sorted(missing))}")
        lines.append("→ Address a MISSING item next.")
    else:
        lines.append(
            "→ All keywords covered. Verify quality and respond TASK COMPLETE if done."
        )
    return lines


def _report_coverage_compass(ws: Path, checklist: str) -> str:
    """M2Flow compass — 扫 report.md 找已覆盖的 checklist item, 给 agent 显式状态.

    这里不做 NLP 语义匹配, 只做 keyword 命中 — ponytail: 规则版, 升级路径才换 LLM.
    返回 compass 文本, 注入 _iter_prompt. report.md 不存在或 checklist 为空返回 "".
    """
    report_path = ws / "report" / "report.md"
    if not report_path.exists() or not checklist:
        return ""
    try:
        report_text = report_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return ""
    keywords = _compass_keywords(checklist)
    if not keywords:
        return ""
    covered = [kw for kw in keywords if kw in report_text]
    missing = [kw for kw in keywords if kw not in report_text]
    total = len(keywords)
    cov_pct = int(100 * len(covered) / total) if total else 0
    return "\n".join(
        _render_coverage_compass(report_text, covered, missing, total, cov_pct)
    )


# B3: LLM compass 缓存 — report.md 未变不重审.
_LLM_COVERAGE_CACHE: dict[tuple[float, int, int], str] = {}


async def _invoke_audit_model(model: Any, prompt: str) -> str:
    """按 chat → ainvoke → invoke 顺序调用 model, 返回有效文本; 无效返回 ""."""
    if hasattr(model, "chat"):
        resp = await model.chat(prompt)
    elif hasattr(model, "ainvoke"):
        resp = await model.ainvoke(prompt)
    elif hasattr(model, "invoke"):
        resp = model.invoke(prompt)
    else:
        return ""
    resp_text = resp if isinstance(resp, str) else str(getattr(resp, "content", resp))
    if not resp_text or len(resp_text) < 20:
        return ""
    return resp_text


def _store_audit_cache(cache: dict, cache_key: tuple | None, result: str) -> None:
    """缓存审计结果, 超过 100 条淘汰最旧 (按 mtime) 的键."""
    if cache_key is not None:
        cache[cache_key] = result
        if len(cache) > 100:
            oldest = min(cache.keys(), key=lambda k: k[0])
            del cache[oldest]


def _coverage_cache_key(report_path: Path, checklist: str) -> tuple | None:
    """report.md stat + checklist hash 作缓存键; stat 失败返回 None."""
    try:
        stat = report_path.stat()
        checklist_hash = hash(checklist or "")
        return (stat.st_mtime, stat.st_size, checklist_hash)
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return None


def _build_coverage_prompt(rule_compass: str, checklist: str, report_text: str) -> str:
    """LLM 覆盖度审计 prompt."""
    return f"""Audit report.md coverage against checklist. Rule-based compass says:

{rule_compass}

Checklist:
{checklist[:3000]}

Report.md (first 8000 chars):
{report_text}

Task: For each checklist item, judge if it's COVERED / PARTIALLY / MISSING in the report.
Pay attention to synonyms and paraphrases (e.g. "MAE" = "mean absolute error", "CGCNN" = "Crystal Graph Convolutional Network").

Respond in this exact format (no prose):
COVERAGE: X% (M/N items)
COVERED: item1, item2
PARTIAL: item3 (what's missing: ...)
MISSING: item4, item5
NEXT: the single most important missing/partial item to address next"""


async def _llm_coverage_audit(
    model: Any,
    ws: Path,
    checklist: str,
    rule_compass: str,
) -> str:
    """v8: LLM 语义深度审计 report.md 覆盖度. 规则版兜底."""
    report_path = ws / "report" / "report.md"
    if not report_path.exists():
        return ""
    cache_key = _coverage_cache_key(report_path, checklist)
    if cache_key is not None:
        cached = _LLM_COVERAGE_CACHE.get(cache_key)
        if cached is not None:
            return cached
    try:
        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return ""
    if len(report_text) > 8000:
        report_text = report_text[:8000] + "\n... (truncated)"
    prompt = _build_coverage_prompt(rule_compass, checklist, report_text)
    try:
        resp_text = await _invoke_audit_model(model, prompt)
        if not resp_text:
            return ""
        result = f"## LLM Coverage Audit (semantic, cached)\n{resp_text.strip()}"
        _store_audit_cache(_LLM_COVERAGE_CACHE, cache_key, result)
        return result
    except Exception as e:
        logger.debug("LLM coverage audit failed: %s", e)
        return ""


# v16: derivation chain audit 缓存 — outputs/ 变了才重审.
_DERIVATION_AUDIT_CACHE: dict[tuple, str] = {}


def _derivation_cache_key(
    report_path: Path,
    outputs_dir: Path,
    checklist: str,
) -> tuple | None:
    """outputs/ 文件签名 + report.md stat + checklist hash; 失败返回 None."""
    try:
        out_files = sorted(outputs_dir.glob("*"))
        out_sig = hash(
            tuple((f.name, f.stat().st_size) for f in out_files if f.is_file())
        )
        r_stat = report_path.stat()
        ck_hash = hash(checklist or "")
        return (r_stat.st_mtime, r_stat.st_size, out_sig, ck_hash)
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return None


def _outputs_listing(outputs_dir: Path) -> str:
    """outputs/ 清单摘要 (最多 30 项; 空目录返回 "(empty)")."""
    out_listing = []
    for f in sorted(outputs_dir.glob("*")):
        if f.is_file():
            out_listing.append(f"{f.name} ({f.stat().st_size}b)")
        elif f.is_dir():
            try:
                n = len(list(f.glob("*")))
                out_listing.append(f"{f.name}/ ({n} files)")
            except Exception:
                out_listing.append(f"{f.name}/")
    return "\n".join(out_listing[:30]) or "(empty)"


def _derivation_prompt_rules() -> str:
    """derivation audit 的判定规则 + 输出格式段 (纯静态文本)."""
    return (
        "CRITICAL DISTINCTION:\n"
        "- DISCUSSED: report.md mentions the item (cites paper's value, methodology, etc.)\n"
        "- EXECUTED: outputs/ has a corresponding artifact (training log, prediction file,\n"
        "  computed result, plot data) showing the agent ACTUALLY RAN the experiment.\n"
        "A checklist item is FULFILLED only if EXECUTED. Discussion alone = DISCUSSED_ONLY.\n"
        "\n"
        "Examples:\n"
        '- "AlphaGeometry solves 25/30" → EXECUTED requires outputs/imo_results.json with\n'
        '  agent\'s own run results, NOT just report citing "AlphaGeometry achieves 25/30".\n'
        '- "FuXi extends skillful lead to 10.5 days" → EXECUTED requires outputs/skill_scores.*\n'
        "  with computed ACC vs lead time, NOT just report quoting the paper.\n"
        '- "H0 = 73.48" → EXECUTED requires outputs/h0_estimate.* with agent\'s own\n'
        "  least-squares computation, NOT just report citing the value.\n"
        '- "synthetic data scale 100M" → EXECUTED requires outputs/synthetic_data.* showing\n'
        '  agent generated/inspected synthetic samples, NOT just report citing "100M".\n'
        "\n"
        "For each checklist item, classify:\n"
        "- FULFILLED: discussed AND executed (artifact exists in outputs/)\n"
        "- DISCUSSED_ONLY: discussed in report but no corresponding artifact in outputs/\n"
        "- MISSING: not even discussed\n"
        "\n"
        "Respond in this exact format (no prose):\n"
        "DERIVATION: X% (M/N items FULFILLED)\n"
        "FULFILLED: item1, item2\n"
        "DISCUSSED_ONLY: item3 (missing artifact: ...), item4 (missing artifact: ...)\n"
        "MISSING: item5, item6\n"
        "NEXT: the single most important DISCUSSED_ONLY or MISSING item — what artifact\n"
        "should agent produce in outputs/ next"
    )


def _build_derivation_prompt(
    rule_compass: str,
    checklist: str,
    outputs_text: str,
    report_text: str,
) -> str:
    """v16 derivation chain audit prompt."""
    return (
        "Audit DERIVATION CHAIN (not just coverage). Rule-based compass:\n\n"
        f"{rule_compass}\n\n"
        "Checklist (each item is a paper claim requiring experimental reproduction):\n"
        f"{checklist[:3000]}\n\n"
        "Actual artifacts in outputs/:\n"
        f"{outputs_text}\n\n"
        "Report.md (first 6000 chars):\n"
        f"{report_text}\n\n" + _derivation_prompt_rules()
    )


async def _derivation_chain_audit(
    model: Any,
    ws: Path,
    checklist: str,
    rule_compass: str,
) -> str:
    """v16: 检查 checklist item 是否被执行 (outputs/ 有产物), 不只是被提及 (report 讨论)."""
    report_path = ws / "report" / "report.md"
    outputs_dir = ws / "outputs"
    if not report_path.exists() or not outputs_dir.exists() or not checklist:
        return ""
    cache_key = _derivation_cache_key(report_path, outputs_dir, checklist)
    if cache_key is not None:
        cached = _DERIVATION_AUDIT_CACHE.get(cache_key)
        if cached is not None:
            return cached

    outputs_text = _outputs_listing(outputs_dir)

    try:
        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return ""
    if len(report_text) > 6000:
        report_text = report_text[:6000] + "\n... (truncated)"

    prompt = _build_derivation_prompt(
        rule_compass, checklist, outputs_text, report_text
    )

    try:
        resp_text = await _invoke_audit_model(model, prompt)
        if not resp_text:
            return ""
        result = (
            f"## Derivation Chain Audit (v16, semantic, cached)\n{resp_text.strip()}"
        )
        _store_audit_cache(_DERIVATION_AUDIT_CACHE, cache_key, result)
        return result
    except Exception as e:
        logger.debug("v16 derivation audit failed: %s", e)
        return ""
