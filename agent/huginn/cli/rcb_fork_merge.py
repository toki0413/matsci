"""rcb_runner 拆分: fork_critique_merge (FCM) + reproduction_gate + judge_fork_reports.

抽自 rcb_runner.py L62-357, 单一职责 = k-path 采样 + 独立评审 + 复现门禁.
原认知原语: LLM 是条件分布不是点值, k 路独立采样把方差从敌人变资源.

ponytail: 不引新依赖, 不改逻辑, 纯 import 抽取.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from huginn.cli.rcb_critique import _strip_code_fences

logger = logging.getLogger(__name__)


# === 认知原语: fork_critique_merge (FCM) — verifier 下沉到 plan 决策点 ===
# G42 第 2 项. 原理: LLM 是条件分布不是点值 — k 路独立采样把方差从敌人
# 变成资源, 独立评审 (无对话历史) 施加选择压力. 治 "单一路径依赖":
# 第一个想到的方案未必最优. 选择压力从 report 级 (太晚) 压到 plan 级 (执行前).
_FCM_PERSPECTIVES = (
    ("fast", "Optimize for speed: prefer simpler implementations, fewer ablations, "
             "shortcuts that don't break [EXACT] components."),
    ("robust", "Optimize for correctness: defensive choices, extra validation steps, "
               "conservative hyperparameters, explicit failure checks."),
    ("exact", "Optimize for fidelity: reproduce the paper as-specified, no deviations, "
              "match every [EXACT] component literally, even if slow."),
)


async def fork_critique_merge(
    checklist: str,
    scan_text: str,
    model: Any,
    k: int = 3,
) -> dict[str, Any]:
    """k 路视角采样执行方案 + 独立评审选优.

    视角差异 (fast/robust/exact) 比 temperature 更可控地保证多样性.
    评审输出 JSON 打分 + winner + 可合并亮点. 全部失败时返回空 winner_plan,
    调用方按 plan-free 跑 (不影响主流程).
    """
    perspectives = _FCM_PERSPECTIVES[: max(1, min(k, len(_FCM_PERSPECTIVES)))]

    from langchain_core.messages import HumanMessage, SystemMessage

    async def _gen(persp_name: str, persp_hint: str) -> tuple[str, str]:
        msgs = [
            SystemMessage(content=(
                f"You are an execution planner with a '{persp_name}' bias. {persp_hint}\n"
                "Output ONLY the plan text, no preamble."
            )),
            HumanMessage(content=(
                "Propose an EXECUTION PLAN for the reproduction task below. "
                "Concrete decisions only — numbers, not adjectives. Max 400 words.\n"
                "Cover: 1) method architecture choices, 2) key hyperparameters, "
                "3) data split/preprocessing, 4) validation order (which checklist items first), "
                "5) top-2 failure modes + fallbacks.\n\n"
                f"## Checklist\n{checklist[:4000]}\n\n"
                f"## Structure scan\n{scan_text[:2000]}"
            )),
        ]
        try:
            resp = await asyncio.to_thread(model.invoke, msgs)
            return persp_name, (resp.content if hasattr(resp, "content") else str(resp)).strip()
        except Exception as e:
            logger.warning("fcm fork(%s) failed: %s", persp_name, e)
            return persp_name, ""

    plans = await asyncio.gather(*(_gen(n, h) for n, h in perspectives))
    candidates = [(n, p) for n, p in plans if p]
    if not candidates:
        return {"winner_plan": "", "winner_perspective": "", "merge_insights": [], "scores": []}
    if len(candidates) == 1:
        return {"winner_plan": candidates[0][1], "winner_perspective": candidates[0][0],
                "merge_insights": [], "scores": []}

    # critique — 独立评审, 新 system prompt, 无对话历史
    numbered = "\n\n".join(f"### Plan {i+1} ({n})\n{p}" for i, (n, p) in enumerate(candidates))
    c_msgs = [
        SystemMessage(content=(
            "You are a SKEPTICAL PLAN REVIEWER. Score plans, find flaws. "
            "Output ONLY a valid JSON object. No markdown fences."
        )),
        HumanMessage(content=(
            f"{numbered}\n\n"
            f"## Checklist (ground truth)\n{checklist[:3000]}\n\n"
            "Score each plan 0-10 on: fidelity to checklist, feasibility, risk coverage.\n"
            "Output JSON: {\"scores\": [float], \"winner\": int (1-based), "
            "\"merge_insights\": [string], \"fatal_flaws\": {\"plan_N\": string}}"
        )),
    ]
    try:
        resp = await asyncio.to_thread(model.invoke, c_msgs)
        text = _strip_code_fences(resp.content if hasattr(resp, "content") else str(resp))
        result = json.loads(text)
        winner_idx = int(result.get("winner", 1)) - 1
        if not (0 <= winner_idx < len(candidates)):
            winner_idx = 0
        scores = [float(s) for s in result.get("scores", [])]
        logger.info("fcm: winner=plan%d(%s) scores=%s",
                    winner_idx + 1, candidates[winner_idx][0], scores)
        return {
            "winner_plan": candidates[winner_idx][1],
            "winner_perspective": candidates[winner_idx][0],
            "merge_insights": [str(x) for x in result.get("merge_insights", [])][:5],
            "scores": scores,
            "fatal_flaws": result.get("fatal_flaws", {}),
        }
    except Exception as e:
        logger.warning("fcm critique failed (%s), fallback to first candidate", e)
        return {"winner_plan": candidates[0][1], "winner_perspective": candidates[0][0],
                "merge_insights": [], "scores": [], "error": str(e)}


# === 认知原语: trajectory fork-merge (TFM) — verifier 下沉到轨迹决策点 ===
# FCM 在 plan 层分叉 (纯文本), TFM 在轨迹层分叉: k 条轨迹各跑一轮真实执行
# (tool calls + fork 报告落盘), verifier 评审 k 份报告选 winner.
#
# 轨迹状态不在 langgraph checkpointer — huginn 每轮 chat() 从 ConversationTree
# active path 重建完整历史传入 graph (stable ID 去重, G34). 所以:
#   fork  = tree.set_active_leaf(branch_point) + 新 thread_id 隔离 graph 内态
#   merge = tree.set_active_leaf(winner_leaf) + 后续迭代换 winner thread_id
# 无 checkpointer 手术.
#
# ponytail: 分叉顺序跑 (共享 workspace, 并行会互踩 report/data 文件);
#   天花板是无文件系统隔离 — 后跑的分叉能看见前序分叉留下的 artifacts
#   (报告名不同, 但 data/outputs 可能已被改写). 升级路径: per-fork workspace
#   copy 或容器隔离 (security/docker_sandbox 已有基建).

def anneal_fork_count(t_hot: float, k_max: int = 3) -> int:
    """认知退火: 探索温度 T_hot 映射到轨迹分叉数.

    >= 0.7 → k_max  (高热, 多轨迹探索)
    >= 0.4 → k_max-1 (中温, 下限 2)
    else   → 1      (低温, 单轨迹利用)
    """
    if k_max <= 1:
        return 1
    if t_hot >= 0.7:
        return k_max
    if t_hot >= 0.4:
        return max(2, k_max - 1)
    return 1


# 复现门禁: 报告的承重数字 (sci-notation, |exp|>=3) 必须能在 fork 的
# 计算产物里复现. 三种写法都抓: 2.6e-20 / 2.6×10^-20 / 2.6×10⁻²⁰.
_SCI_E = re.compile(r"([-+]?\d+(?:\.\d+)?)[eE]([-+]?\d+)")
_SCI_10 = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*[×xX]\s*10\^?\{?([-+]?\d+)\}?")
_SCI_SUP = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*[×xX]\s*10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_SUP_TRANS = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")


def _extract_sci_numbers(text: str, min_exp: int = 3) -> list[float]:
    """报告里的承重数字: sci-notation 且 |exp|>=min_exp (物理结论的量级声明)."""
    out: list[float] = []
    for coef, exp in _SCI_E.findall(text) + _SCI_10.findall(text):
        try:
            if abs(int(exp)) >= min_exp:
                out.append(float(coef) * 10.0 ** int(exp))
        except OverflowError:
            continue
    for coef, exp in _SCI_SUP.findall(text):
        try:
            e = int(exp.translate(_SUP_TRANS))
            if abs(e) >= min_exp:
                out.append(float(coef) * 10.0 ** e)
        except OverflowError:
            continue
    return out


def _collect_artifact_numbers(d: Path | None, cap: int = 200_000) -> list[float]:
    """收集 fork outputs 快照里的全部数值, 作为复现对照集."""
    nums: list[float] = []
    if d is None or not d.is_dir():
        return nums
    try:
        import numpy as np
    except Exception:
        np = None
    for p in sorted(d.rglob("*")):
        if len(nums) >= cap or not p.is_file():
            continue
        try:
            if p.suffix == ".npy" and np is not None:
                a = np.asarray(np.load(p, allow_pickle=False), dtype=float).ravel()
                nums.extend(a[: cap - len(nums)].tolist())
            elif p.suffix == ".npz" and np is not None:
                with np.load(p, allow_pickle=False) as z:
                    for k in z.files:
                        a = np.asarray(z[k], dtype=float).ravel()
                        nums.extend(a[: cap - len(nums)].tolist())
            elif p.suffix in (".json", ".txt", ".csv"):
                nums.extend(
                    float(m.group(0))
                    for m in _SCI_E.finditer(p.read_text(errors="replace"))
                )
        except Exception:
            continue
    return [x for x in nums if math.isfinite(x)]


def _reproduction_gate(
    report: str, artifact_dir: Path | None, rel_tol: float = 0.05
) -> tuple[bool, str]:
    """执行级复现门禁: 承重数字至少 1 个能在 fork 产物里复现 (rel tol 5%).

    规则: 0-1 个承重数字 → unverifiable 放行 (不惩罚不可验证);
    >=2 个且 0 匹配 → fail (编数字直接淘汰, 不看文笔); 否则 pass.

    ponytail: 对照集是 fork outputs 快照的全部数值, 不区分 summary 与原始
    数组 (天花板: 常见量级可能巧合匹配; 只查 |exp|>=3 的极端量级使巧合 ~0).
    升级路径: fork 级 fs 隔离 + deterministic 重跑 pipeline.
    """
    claims = _extract_sci_numbers(report)
    if len(claims) < 2:
        return True, f"unverifiable ({len(claims)} load-bearing claims)"
    artifacts = _collect_artifact_numbers(artifact_dir)
    if not artifacts:
        return True, f"unverifiable (no artifacts, {len(claims)} claims)"
    matched = sum(
        1 for c in claims
        if any(abs(a - c) <= rel_tol * abs(c) for a in artifacts)
    )
    if matched == 0:
        return False, f"FAIL 0/{len(claims)} claims reproduced"
    return True, f"pass {matched}/{len(claims)} claims reproduced"


async def judge_fork_reports(
    reports: dict[str, str],
    checklist: str,
    model: Any,
    artifact_dirs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """verifier 评审 k 份 fork 报告, 选 winner + 从败者里捞可合并亮点.

    reports: {perspective: report_text}. 空报告先过滤 (fork 没写出来不算候选).
    artifact_dirs: {perspective: outputs 快照目录}, 给了就先跑复现门禁 —
    生成/验证分离, 编数字的 fork 不进 LLM 评审. 门禁全灭时不清空候选
    (启发式有误杀率), 但 gate note 进 prompt 让 judge 知道谁可疑.
    LLM 失败 / 编了不存在的 fork 名 → deterministic fallback 第一份非空报告.
    """
    candidates = {name: text for name, text in reports.items() if text.strip()}
    if not candidates:
        return {"winner": None, "scores": {}, "merge_notes": []}

    # 执行级复现门禁 (选择信号从观点换成执行)
    gate_notes: dict[str, str] = {}
    if artifact_dirs:
        survivors: dict[str, str] = {}
        for name, text in candidates.items():
            ok, note = _reproduction_gate(text, artifact_dirs.get(name))
            gate_notes[name] = note
            if ok:
                survivors[name] = text
            else:
                logger.info("tfm gate: fork %s eliminated (%s)", name, note)
        if survivors:
            candidates = survivors

    if len(candidates) == 1:
        return {"winner": next(iter(candidates)), "scores": {}, "merge_notes": [],
                "gate": gate_notes}

    from langchain_core.messages import HumanMessage, SystemMessage

    numbered = "\n\n".join(
        f"### Fork report ({name})\n{text[:4000]}" for name, text in candidates.items()
    )
    gate_text = ""
    if gate_notes:
        gate_text = (
            "\n## Reproduction gate (execution-grounded, distrust FAILed forks)\n"
            + "\n".join(f"- {n}: {t}" for n, t in gate_notes.items()) + "\n"
        )
    msgs = [
        SystemMessage(content=(
            "You are a SKEPTICAL REPRODUCTION REVIEWER comparing parallel execution forks. "
            "Output ONLY a valid JSON object. No markdown fences."
        )),
        HumanMessage(content=(
            f"{numbered}\n{gate_text}\n"
            f"## Checklist (ground truth)\n{checklist[:3000]}\n\n"
            "Score each fork 0-10 on: checklist coverage, [EXACT] fidelity, "
            "metric plausibility (penalize better-than-paper results), completeness.\n"
            "Output JSON: {\"scores\": {\"<name>\": float}, \"winner\": \"<name>\", "
            "\"merge_notes\": [string — salvageable ideas from losing forks]}"
        )),
    ]
    try:
        resp = await asyncio.to_thread(model.invoke, msgs)
        text = _strip_code_fences(resp.content if hasattr(resp, "content") else str(resp))
        result = json.loads(text)
        winner = str(result.get("winner", ""))
        if winner not in candidates:
            winner = next(iter(candidates))
        scores = {str(k): float(v) for k, v in (result.get("scores") or {}).items()}
        logger.info("tfm: winner=%s scores=%s", winner, scores)
        return {
            "winner": winner,
            "scores": scores,
            "merge_notes": [str(x) for x in result.get("merge_notes", [])][:5],
            "gate": gate_notes,
        }
    except Exception as e:
        logger.warning("tfm judge failed (%s), fallback to first non-empty", e)
        return {"winner": next(iter(candidates)), "scores": {}, "merge_notes": [],
                "gate": gate_notes, "error": str(e)}
