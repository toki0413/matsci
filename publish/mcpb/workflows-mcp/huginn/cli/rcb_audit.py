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
    idx: int          # 0-based item 序号
    name: str         # "1.1 Symbolic Deduction Engine"
    label: str = ""   # "[EXACT]" / "[VARIANT]" / "" (未标注)
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
        items.append(_ChecklistItem(
            idx=i,
            name=f"{_id} {_name}",
            label=_lbl_m.group(1) if _lbl_m else "",
            expected_output=(_out_m.group(1).strip()[:120] if _out_m else ""),
        ))
    return items


def _time_slot_index(elapsed_s: float, per_item_budget_s: float, n_items: int) -> int:
    """wall_clock 切片 → 当前 item index. ponytail: 简单除法, 不重分配剩余预算."""
    if per_item_budget_s <= 0:
        return 0
    idx = int(elapsed_s / per_item_budget_s)
    return min(idx, n_items - 1)


def _rcb_effort_floor(
    ws: Path, checklist: str, *, min_cov_pct: int = 70,
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
        return False, "report/report.md does NOT exist — write it BEFORE claiming TASK COMPLETE"
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
                _missing = _line[len("missing:"):].strip()
                break
        return False, f"coverage={_cov}% < {min_cov_pct}%, missing: {_missing}"
    if derivation_audit:
        _dm = _re.search(r"DERIVATION:\s*(\d+)%\s*\((\d+)/(\d+)", derivation_audit)
        if _dm:
            _dev_cov = int(_dm.group(1))
            if _dev_cov < 50:
                _disc = ""
                for _line in derivation_audit.split("\n"):
                    if _line.lower().startswith("discussed_only:"):
                        _disc = _line[len("discussed_only:"):].strip()[:300]
                        break
                return False, (
                    f"derivation={_dev_cov}% < 50%, agent discussed checklist items "
                    f"in report but did NOT execute them (no artifacts in outputs/). "
                    f"DISCUSSED_ONLY: {_disc}\n"
                    f"→ Produce the missing artifacts in outputs/ before claiming TASK COMPLETE."
                )
    return True, ""


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
    keywords = set()
    for line in checklist.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "[exact]" in line.lower():
            idx = line.lower().find("[exact]")
            tail = line[idx + 7:].strip().split()
            for w in tail[:4]:
                w = w.strip(".,;:()[]")
                if len(w) >= 3:
                    keywords.add(w.lower())
        for metric in ("mae", "rmse", "r²", "r2", "accuracy", "precision", "recall", "f1"):
            if metric in line.lower():
                keywords.add(metric)
    if not keywords:
        return ""
    covered = [kw for kw in keywords if kw in report_text]
    missing = [kw for kw in keywords if kw not in report_text]
    total = len(keywords)
    cov_pct = int(100 * len(covered) / total) if total else 0
    lines = [
        f"## Report Coverage Compass ({cov_pct}% — {len(covered)}/{total} checklist keywords found in report.md)",
    ]
    if "expected" in report_text and ("placeholder" in report_text or "tbd" in report_text
                                       or "not yet" in report_text or "n/a" in report_text):
        lines.append("⚠ WARNING: report.md contains 'Expected'/'placeholder'/'TBD' — "
                     "these are NOT results. Replace with actual executed metrics or "
                     "declare the gap honestly. Judge scores placeholder content as 0.")
    if covered:
        lines.append(f"Covered: {', '.join(sorted(covered))}")
    if missing:
        lines.append(f"Missing: {', '.join(sorted(missing))}")
        lines.append("→ Address a MISSING item next.")
    else:
        lines.append("→ All keywords covered. Verify quality and respond TASK COMPLETE if done.")
    return "\n".join(lines)


# B3: LLM compass 缓存 — report.md 未变不重审.
_LLM_COVERAGE_CACHE: dict[tuple[float, int, int], str] = {}


async def _llm_coverage_audit(
    model: Any, ws: Path, checklist: str, rule_compass: str,
) -> str:
    """v8: LLM 语义深度审计 report.md 覆盖度. 规则版兜底."""
    report_path = ws / "report" / "report.md"
    if not report_path.exists():
        return ""
    try:
        stat = report_path.stat()
        checklist_hash = hash(checklist or "")
        cache_key = (stat.st_mtime, stat.st_size, checklist_hash)
        if cache_key in _LLM_COVERAGE_CACHE:
            return _LLM_COVERAGE_CACHE[cache_key]
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        cache_key = None
    try:
        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return ""
    if len(report_text) > 8000:
        report_text = report_text[:8000] + "\n... (truncated)"
    prompt = f"""Audit report.md coverage against checklist. Rule-based compass says:

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
    try:
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
        result = f"## LLM Coverage Audit (semantic, cached)\n{resp_text.strip()}"
        if cache_key is not None:
            _LLM_COVERAGE_CACHE[cache_key] = result
            if len(_LLM_COVERAGE_CACHE) > 100:
                oldest = min(_LLM_COVERAGE_CACHE.keys(), key=lambda k: k[0])
                del _LLM_COVERAGE_CACHE[oldest]
        return result
    except Exception as e:
        logger.debug("LLM coverage audit failed: %s", e)
        return ""


# v16: derivation chain audit 缓存 — outputs/ 变了才重审.
_DERIVATION_AUDIT_CACHE: dict[tuple, str] = {}


async def _derivation_chain_audit(
    model: Any, ws: Path, checklist: str, rule_compass: str,
) -> str:
    """v16: 检查 checklist item 是否被执行 (outputs/ 有产物), 不只是被提及 (report 讨论)."""
    report_path = ws / "report" / "report.md"
    outputs_dir = ws / "outputs"
    if not report_path.exists() or not outputs_dir.exists() or not checklist:
        return ""
    try:
        out_files = sorted(outputs_dir.glob("*"))
        out_sig = hash(tuple(
            (f.name, f.stat().st_size) for f in out_files if f.is_file()
        ))
        r_stat = report_path.stat()
        ck_hash = hash(checklist or "")
        cache_key = (r_stat.st_mtime, r_stat.st_size, out_sig, ck_hash)
        if cache_key in _DERIVATION_AUDIT_CACHE:
            return _DERIVATION_AUDIT_CACHE[cache_key]
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        cache_key = None

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
    outputs_text = "\n".join(out_listing[:30]) or "(empty)"

    try:
        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return ""
    if len(report_text) > 6000:
        report_text = report_text[:6000] + "\n... (truncated)"

    prompt = f"""Audit DERIVATION CHAIN (not just coverage). Rule-based compass:

{rule_compass}

Checklist (each item is a paper claim requiring experimental reproduction):
{checklist[:3000]}

Actual artifacts in outputs/:
{outputs_text}

Report.md (first 6000 chars):
{report_text}

CRITICAL DISTINCTION:
- DISCUSSED: report.md mentions the item (cites paper's value, methodology, etc.)
- EXECUTED: outputs/ has a corresponding artifact (training log, prediction file,
  computed result, plot data) showing the agent ACTUALLY RAN the experiment.
A checklist item is FULFILLED only if EXECUTED. Discussion alone = DISCUSSED_ONLY.

Examples:
- "AlphaGeometry solves 25/30" → EXECUTED requires outputs/imo_results.json with
  agent's own run results, NOT just report citing "AlphaGeometry achieves 25/30".
- "FuXi extends skillful lead to 10.5 days" → EXECUTED requires outputs/skill_scores.*
  with computed ACC vs lead time, NOT just report quoting the paper.
- "H0 = 73.48" → EXECUTED requires outputs/h0_estimate.* with agent's own
  least-squares computation, NOT just report citing the value.
- "synthetic data scale 100M" → EXECUTED requires outputs/synthetic_data.* showing
  agent generated/inspected synthetic samples, NOT just report citing "100M".

For each checklist item, classify:
- FULFILLED: discussed AND executed (artifact exists in outputs/)
- DISCUSSED_ONLY: discussed in report but no corresponding artifact in outputs/
- MISSING: not even discussed

Respond in this exact format (no prose):
DERIVATION: X% (M/N items FULFILLED)
FULFILLED: item1, item2
DISCUSSED_ONLY: item3 (missing artifact: ...), item4 (missing artifact: ...)
MISSING: item5, item6
NEXT: the single most important DISCUSSED_ONLY or MISSING item — what artifact
should agent produce in outputs/ next"""

    try:
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
        result = f"## Derivation Chain Audit (v16, semantic, cached)\n{resp_text.strip()}"
        if cache_key is not None:
            _DERIVATION_AUDIT_CACHE[cache_key] = result
            if len(_DERIVATION_AUDIT_CACHE) > 100:
                oldest = min(_DERIVATION_AUDIT_CACHE.keys(), key=lambda k: k[0])
                del _DERIVATION_AUDIT_CACHE[oldest]
        return result
    except Exception as e:
        logger.debug("v16 derivation audit failed: %s", e)
        return ""
