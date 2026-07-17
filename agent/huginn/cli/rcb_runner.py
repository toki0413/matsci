"""RCB harness 入口: 读 workspace/INSTRUCTIONS.md, 跑 huginn agent, 输出到 stdout.

ResearchClawBench 的 TaskRunner 通过 subprocess 跑 agent_cmd, 捕获 stdout.
本脚本作为 huginn 的 RCB adapter:
  python huginn/cli/rcb_runner.py --workspace <workspace>

agent 在 workspace 里工作 (cwd=workspace), 用 code_tool/bash_tool 读写文件,
最终产出 report/report.md. RCB 的 INSTRUCTIONS.md 模板已经很详细, system
prompt 只需简短研究导向.

ponytail: 不重复 RCB prompt 已有的内容, 不加交互式渲染, 纯文本输出.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 在 import huginn 之前关掉秒级限流 — RCB 任务的 prompt 长 + 工具多,
# 默认 5000 tokens/s 会在第一轮就超限. RCB 是离线评测, 不需要限流.
os.environ.setdefault("HUGINN_RATE_LIMIT_ENABLED", "0")
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "500000")
# 允许本地沙箱执行 code_tool/bash_tool — RCB subprocess 没有 docker, 用本地 python
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")
# HUGINN_CACHE_DIR 可能被设成空串, 导致 LongTermMemory 用相对路径 memory.db,
# 在 RCB workspace cwd 下 sqlite WAL 创建失败. 强制用绝对路径.
if not os.environ.get("HUGINN_CACHE_DIR"):
    os.environ["HUGINN_CACHE_DIR"] = str(Path.home() / ".huginn")
# RCB 场景用 CSM 子集: 3-step 映射 S1/S4/S6+S7, 不再全 skip (Task 18, R8 减法修正).
# ponytail: S7 自修改仍走 (Task 2), 只跳过 compaction — 见 reflection.py L245.
os.environ.setdefault("HUGINN_RCB_CSM_SUBSET", "1")
# RCB 场景 compaction 保留前 2 条 root (task + Step 1 checklist) — 修同伦断裂 (σ₂)
os.environ.setdefault("HUGINN_KEEP_ROOT_N", "2")
# RCB 场景跳过 Rust sandbox — 它在 RDKit+sklearn GPR 场景静默崩溃返回空 stderr
os.environ.setdefault("HUGINN_NO_RUST_SANDBOX", "1")
# RCB 场景关熔断器 — file_read_tool 误触发 circuit_open 阻止 agent 读文件 (σ₇)
os.environ.setdefault("HUGINN_HEALTH_MONITOR", "0")
# RCB 场景关循环检测 — agent 反复跑 code_tool 是正常行为, 误判为 loop (σ₈)
os.environ.setdefault("HUGINN_SKIP_LOOP_DETECTOR", "1")


# === 认知原语: adversarial_critique ===
# ponytail: 独立 LLM 调用做 skeptical reviewer, 消除 confirmation bias.
# 治 Gap 3 (M_002 MAE=0.032eV 造假) — agent 自己写的 report, 自己 review 会护短.
# 这是 RCB/autoloop/forest 三种循环都能复用的认知原语 (治 F1 起步).
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        text = text[nl + 1:] if nl > 0 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


async def adversarial_critique(
    model: Any = None,
    report: str = "",
    checklist: str = "",
    *,
    mode: str = "object",
    proposal: str = "",
    system_prompt_summary: str = "",
    recent_rejections: list[str] | None = None,
    llm_client: Any = None,
) -> dict[str, Any]:
    """独立 LLM 调用做 skeptical reviewer — 消除 confirmation bias.

    mode="object": critique report (原逻辑不变)
    mode="meta": critique agent 自修改提案 (L2 元元认知层, 哥德尔机 proof
                 verifier 弱化版 — LLM judge 替代形式化 proof)

    不让 agent 自检, 因为 agent 写的 report 它自己不会判造假.
    用独立 LLM 调用 (新 system prompt, 无对话历史) 做 adversarial review.
    返回结构化 JSON, format_critique_for_agent() 把它转成 agent 可读的修复指令.
    """
    if mode == "meta":
        import difflib
        # 早期拒绝查重 — 命中直接返回, 不调 LLM 省 token
        # ponytail: 天花板是 difflib 字符串相似度 (抓不到同义改写这类语义近义,
        #           "always use X" 改写成 "X must be used" 漏判);
        #           升级路径换 embedding 相似度 (sentence-transformers cosine > 0.85)
        for prev in recent_rejections or []:
            ratio = difflib.SequenceMatcher(None, proposal, prev).ratio()
            if ratio > 0.8:
                logger.info("meta_critique early_reject: similarity=%.2f", ratio)
                return {
                    "verdict": "reject",
                    "reason": f"similar to past rejection (similarity={ratio:.2f})",
                    "expected_utility_delta": 0.0,
                    "early_reject": True,
                }
        # 调 LLM 做完整评估 — 复用现有 ainvoke 模式, 同 client 不同 system prompt
        from langchain_core.messages import HumanMessage, SystemMessage
        client = llm_client if llm_client is not None else model
        meta_system = SystemMessage(content=(
            "你是 META-REVIEWER 评估 agent 的自修改提案.\n"
            "你的工作是判断这个提案是否会真正改进 agent 的效用, "
            "还是会污染 gradient 或引入坏习惯.\n\n"
            "评估维度:\n"
            "1. 是否污染 gradient (例如加 CRITICAL: always use X) — 这是 σ₄ lesson\n"
            "2. 是否与最近 rejection 相似 (相似度 > 0.8 直接 reject)\n"
            "3. 是否与现有 stable_principles 冲突\n"
            "4. expected_utility_delta 是否为正\n\n"
            "输出严格 JSON: "
            '{"verdict": "accept"|"reject", "reason": "...", "expected_utility_delta": float}'
        ))
        rejections_block = "\n".join(f"- {r}" for r in (recent_rejections or [])) or "(none)"
        meta_human = HumanMessage(content=(
            f"## Proposal\n{proposal}\n\n"
            f"## Current system prompt summary\n{system_prompt_summary or '(empty)'}\n\n"
            f"## Recent rejections (do not repeat)\n{rejections_block}\n\n"
            "Output ONLY the JSON object."
        ))
        try:
            resp = await client.ainvoke([meta_system, meta_human])
            text = resp.content if hasattr(resp, "content") else str(resp)
            text = _strip_code_fences(text)
            result = json.loads(text)
            result.setdefault("verdict", "reject")
            result.setdefault("reason", "no reason provided")
            result.setdefault("expected_utility_delta", 0.0)
            result["early_reject"] = False
            logger.info("meta_critique: verdict=%s", result["verdict"])
            return result
        except Exception as e:
            logger.warning("meta_critique failed: %s", e)
            return {
                "verdict": "reject",
                "reason": f"meta_critique error: {e}",
                "expected_utility_delta": 0.0,
                "early_reject": False,
                "error": str(e),
            }

    # === object mode (原逻辑不变) ===
    from langchain_core.messages import HumanMessage, SystemMessage
    system = SystemMessage(content=(
        "You are a SKEPTICAL SCIENTIFIC REVIEWER who wants to score this report LOW. "
        "Your job is to find FLAWS. Be adversarial. "
        "Output ONLY a valid JSON object. No markdown fences, no preamble."
    ))
    human = HumanMessage(content=(
        f"## Methodology Checklist (from Step 1)\n{checklist}\n\n"
        f"## Report to Critique\n{report}\n\n"
        "## Your Task (output JSON only)\n"
        "1. \"implausible_metrics\": metrics where report value is BETTER than paper baseline. "
        "Format: [{\"metric\": name, \"paper\": value, \"yours\": value, \"red_flag\": why}]. "
        "Empty list if none.\n"
        "2. \"silent_substitutions\": [EXACT] components silently replaced with simpler alternatives. "
        "Format: [{\"component\": name, \"expected\": what, \"actual\": what}]. Empty list if none.\n"
        "3. \"missing_components\": checklist items absent from report. Empty list if none.\n"
        "4. \"overall_verdict\": \"pass\" | \"fix_needed\" | \"fail\"\n\n"
        "Output ONLY the JSON object."
    ))
    try:
        resp = await model.ainvoke([system, human])
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = _strip_code_fences(text)
        result = json.loads(text)
        result.setdefault("implausible_metrics", [])
        result.setdefault("silent_substitutions", [])
        result.setdefault("missing_components", [])
        result.setdefault("overall_verdict", "fix_needed")
        logger.info("adversarial_critique: verdict=%s", result["overall_verdict"])
        return result
    except Exception as e:
        logger.warning("adversarial_critique failed: %s", e)
        return {
            "implausible_metrics": [],
            "silent_substitutions": [],
            "missing_components": [],
            "overall_verdict": "fix_needed",
            "error": str(e),
        }


# === G42: critique_decision — v4 预留接口第 1 项 ===
# 扩展 adversarial_critique 到 mode/phase/tool 决策层. 不破坏 object/meta 双层.
# ponytail: 用同样的独立 LLM 调用模式, 不引入新组件. 升级路径是接 LeanInterface
# 对决策做形式化验证 (mode/phase 转移是可形式化的有限状态机).
from dataclasses import dataclass, field as _dc_field


@dataclass
class Decision:
    """可 critique 的决策 — mode 切换 / phase 转移 / tool 选择.

    frm/to 用字符串而非 enum, 因为 mode/phase/tool 名字都是有限字符串集合,
    不必为每种决策维护单独 enum. 调用方负责传合法值.
    """
    kind: str  # "mode_switch" | "phase_transition" | "tool_select"
    frm: str
    to: str
    rationale: str = ""
    metadata: dict[str, Any] = _dc_field(default_factory=dict)


@dataclass
class CritiqueResult:
    """critique_decision 的返回 — 含 red_flags + suggestions + 总体 verdict."""
    verdict: str  # "accept" | "reject" | "fix_needed"
    red_flags: list[str] = _dc_field(default_factory=list)
    suggestions: list[str] = _dc_field(default_factory=list)
    reason: str = ""


# 决策类型的合法 from/to 集合 — 模板 critique 用, LLM 路径不依赖
_VALID_MODES = frozenset({"chat", "plan", "research"})
_VALID_PHASES = frozenset({
    "perceive", "hypothesize", "plan", "execute", "validate", "learn", "report",
})


def _template_critique_decision(decision: Decision, context: dict[str, Any]) -> CritiqueResult:
    """模板 critique — 无 LLM 时走规则. 三层检查:
    1. kind 合法
    2. frm/to 在对应合法集合里
    3. rationale 非空 (无理由的决策不可 critique)
    ponytail: 这是廉价兜底, 真正的 critique 走 LLM 路径.
    """
    red_flags: list[str] = []
    suggestions: list[str] = []

    if decision.kind not in ("mode_switch", "phase_transition", "tool_select"):
        return CritiqueResult(
            verdict="reject",
            red_flags=[f"unknown decision kind: {decision.kind}"],
            reason="invalid kind",
        )

    if decision.kind == "mode_switch":
        if decision.frm and decision.frm not in _VALID_MODES:
            red_flags.append(f"frm mode '{decision.frm}' not in {_VALID_MODES}")
        if decision.to not in _VALID_MODES:
            red_flags.append(f"to mode '{decision.to}' not in {_VALID_MODES}")
    elif decision.kind == "phase_transition":
        if decision.frm and decision.frm not in _VALID_PHASES:
            red_flags.append(f"frm phase '{decision.frm}' not in _VALID_PHASES")
        if decision.to not in _VALID_PHASES:
            red_flags.append(f"to phase '{decision.to}' not in _VALID_PHASES")
        # 7-phase pipeline: 只能向前走或回 perceive, 不能跳过 execute → report
        if decision.frm in _VALID_PHASES and decision.to in _VALID_PHASES:
            phases_order = list(_VALID_PHASES)
            # report 不能跳回 execute (除非显式 force)
            if decision.frm == "report" and decision.to == "execute" and not context.get("force"):
                red_flags.append("report → execute backtrack without force=True")
                suggestions.append("if refining after report, pass context['force']=True")

    if not decision.rationale.strip():
        red_flags.append("empty rationale — decision without justification")
        suggestions.append("provide rationale explaining why this decision is needed")

    if red_flags:
        return CritiqueResult(
            verdict="fix_needed",
            red_flags=red_flags,
            suggestions=suggestions,
            reason="template rules failed",
        )
    return CritiqueResult(verdict="accept", reason="template rules passed")


async def critique_decision(
    decision: Decision,
    context: dict[str, Any] | None = None,
    *,
    model: Any = None,
    llm_client: Any = None,
) -> CritiqueResult:
    """L3 decision critique — 扩展 adversarial_critique 到 mode/phase/tool.

    不破坏现有 object/meta 双层 critique. 这是 v4 预留的第 1 项接口实现.
    复用同样独立 LLM 调用模式: 新 system prompt, 无对话历史, 输出结构化 JSON.

    model=None 时走模板规则 (测试用); model 传入时优先调 LLM, 失败降级到模板.
    """
    context = context or {}

    # 先走模板兜底 — 模板都过不了的不必浪费 LLM 调用
    template_result = _template_critique_decision(decision, context)
    if template_result.verdict == "reject":
        return template_result

    client = llm_client if llm_client is not None else model
    if client is None or not hasattr(client, "ainvoke"):
        return template_result

    # LLM 路径 — 让 skeptical reviewer 看决策是否合理
    from langchain_core.messages import HumanMessage, SystemMessage
    system = SystemMessage(content=(
        "你是 DECISION AUDITOR 评估 agent 的 mode/phase/tool 决策.\n"
        "你的工作是找 FLAWS — 不合理的转移、跳阶段、tool 选择错误.\n"
        "输出严格 JSON: "
        '{"verdict": "accept"|"reject"|"fix_needed", '
        '"red_flags": [string], "suggestions": [string], "reason": string}'
    ))
    ctx_str = json.dumps(context, ensure_ascii=False, default=str)[:2000]
    human = HumanMessage(content=(
        f"## Decision\n"
        f"kind: {decision.kind}\n"
        f"from: {decision.frm}\n"
        f"to: {decision.to}\n"
        f"rationale: {decision.rationale}\n\n"
        f"## Context\n{ctx_str}\n\n"
        "Output ONLY the JSON object."
    ))
    try:
        resp = await client.ainvoke([system, human])
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = _strip_code_fences(text)
        data = json.loads(text)
        return CritiqueResult(
            verdict=data.get("verdict", "fix_needed"),
            red_flags=list(data.get("red_flags", [])),
            suggestions=list(data.get("suggestions", [])),
            reason=data.get("reason", ""),
        )
    except Exception as e:
        logger.warning("critique_decision LLM failed: %s — fallback to template", e)
        return template_result


# === G28: 数值重算 — report 中的数值 claim 对 outputs/ 实际数据交叉验证 ===
# ponytail: 治 M_002 MAE=0.032eV 造假 — agent 写 report 时编个数, LLM critique 看
# 文本看不出问题. 这里直接读 outputs/ 里的 metrics.json / predictions.csv 重算,
# 对不上就记 red flag 喂给 LLM critique. 天花板是只认标准 metric 名 (MAE/RMSE/R²/
# accuracy/loss), 不认 AUROC/KL divergence 这类; 升级路径是加 recompute 函数注册表.
import re as _re

_METRIC_CLAIM_PATTERNS: dict[str, str] = {
    "MAE": r"\bMAE\b\s*[:=]\s*([0-9.]+)",
    "RMSE": r"\bRMSE\b\s*[:=]\s*([0-9.]+)",
    "R2": r"\bR[²2]\b\s*[:=]\s*([0-9.]+)",
    "accuracy": r"\baccuracy\b\s*[:=]\s*([0-9.]+)",
    "loss": r"\bloss\b\s*[:=]\s*([0-9.]+)",
}


def _parse_report_metric_claims(report: str) -> dict[str, float]:
    """从 report 文本里抓 MAE=0.05 / R²=0.86 / accuracy=0.56 这类 claim."""
    claims: dict[str, float] = {}
    for name, pat in _METRIC_CLAIM_PATTERNS.items():
        m = _re.search(pat, report, _re.IGNORECASE)
        if m:
            try:
                claims[name] = float(m.group(1))
            except ValueError:
                pass
    return claims


def _recompute_report_metrics(report: str, workspace: Path) -> list[dict[str, Any]]:
    """交叉验证 report 的数值 claim vs outputs/ 实际数据.

    两条路:
      1. JSON metric file — agent 写的 metrics.json/results.json 直接比对
      2. predictions CSV — 有 true/pred 列就从原始数据重算 MAE/RMSE/R²

    返回 red_flag 列表, 每项 {metric, claimed, recomputed, red_flag}.
    空列表 = 无数据可验 / 全部一致.
    """
    red_flags: list[dict[str, Any]] = []
    outputs = workspace / "outputs"
    if not outputs.exists():
        return red_flags

    claims = _parse_report_metric_claims(report)
    if not claims:
        return red_flags

    actual: dict[str, float] = {}

    # Path 1: JSON metric files
    for jf in outputs.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            for k, v in data.items():
                nk = _re.sub(r'[^a-zA-Z0-9]', '', k).lower()
                if isinstance(v, (int, float)) and nk:
                    actual[nk] = float(v)
        except Exception:
            pass

    # Path 2: predictions CSV — recompute MAE/RMSE/R² from raw true/pred
    try:
        import csv as _csv
        for cf in outputs.glob("*.csv"):
            try:
                with cf.open(encoding="utf-8") as f:
                    rows = list(_csv.DictReader(f))
                if not rows:
                    continue
                cols = {c.lower() for c in rows[0].keys()}
                true_col = next((c for c in ("y_true", "true", "actual", "target") if c in cols), None)
                pred_col = next((c for c in ("y_pred", "pred", "predicted", "prediction") if c in cols), None)
                if not (true_col and pred_col):
                    continue
                trues: list[float] = []
                preds: list[float] = []
                for r in rows:
                    try:
                        # case-insensitive match against row keys
                        t = next((r[k] for k in r if k.lower() == true_col), None)
                        p = next((r[k] for k in r if k.lower() == pred_col), None)
                        trues.append(float(t))
                        preds.append(float(p))
                    except (ValueError, TypeError):
                        pass
                if len(trues) < 2:
                    continue
                n = len(trues)
                mae = sum(abs(t - p) for t, p in zip(trues, preds)) / n
                rmse = (sum((t - p) ** 2 for t, p in zip(trues, preds)) / n) ** 0.5
                mean_t = sum(trues) / n
                ss_res = sum((t - p) ** 2 for t, p in zip(trues, preds))
                ss_tot = sum((t - mean_t) ** 2 for t in trues)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                actual["mae"] = mae
                actual["rmse"] = rmse
                actual["r2"] = r2
                break  # first usable CSV is enough
            except Exception:
                pass
    except Exception:
        pass

    # Cross-check: report claim vs actual
    # ponytail: 10% 相对容差 — 小浮点噪声 OK, 大谎报必抓. 极小值场景用 1e-4 绝对底.
    for name, claimed in claims.items():
        nk = _re.sub(r'[^a-zA-Z0-9]', '', name).lower()
        recomputed = actual.get(nk)
        if recomputed is None:
            continue  # 没数据可验, 交给 LLM
        diff = abs(claimed - recomputed)
        scale = max(abs(claimed), abs(recomputed), 1e-4)
        if diff / scale > 0.1:
            red_flags.append({
                "metric": name,
                "claimed": claimed,
                "recomputed": round(recomputed, 6),
                "red_flag": f"report claims {name}={claimed}, recomputed from outputs/ is "
                            f"{recomputed:.4f} ({diff/scale*100:.1f}% mismatch — investigate "
                            f"data leakage, wrong split, or fabrication)",
            })
    return red_flags


def format_critique_for_agent(critique: dict[str, Any]) -> str:
    """把 critique 结果格式化成 agent 可读的修复指令."""
    lines = ["ADVERSARIAL CRITIQUE RESULTS (from independent reviewer):\n"]
    verdict = critique.get("overall_verdict", "fix_needed")
    lines.append(f"Overall verdict: {verdict.upper()}\n")

    implausible = critique.get("implausible_metrics", [])
    if implausible:
        lines.append("## RED FLAG — Implausible Metrics (better than paper)")
        for m in implausible:
            lines.append(
                f"  - {m.get('metric', '?')}: paper={m.get('paper', '?')}, "
                f"yours={m.get('yours', '?')} — {m.get('red_flag', 'investigate')}"
            )
        lines.append("")

    # G28: 数值重算出的 red flags — report 写的数 vs outputs/ 实际数据对不上
    recomputed = critique.get("recomputed_red_flags", [])
    if recomputed:
        lines.append("## RED FLAG — Numeric Claim vs Recomputed Mismatch (fabrication suspect)")
        for m in recomputed:
            lines.append(
                f"  - {m.get('metric', '?')}: report={m.get('claimed', '?')}, "
                f"recomputed={m.get('recomputed', '?')} — {m.get('red_flag', 'investigate')}"
            )
        lines.append("")

    subs = critique.get("silent_substitutions", [])
    if subs:
        lines.append("## RED FLAG — Silent Methodology Substitutions")
        for s in subs:
            lines.append(
                f"  - {s.get('component', '?')}: expected={s.get('expected', '?')}, "
                f"actual={s.get('actual', '?')}"
            )
        lines.append("")

    missing = critique.get("missing_components", [])
    if missing:
        lines.append("## Missing Components")
        for c in missing:
            lines.append(f"  - {c}")
        lines.append("")

    lines.append("## Fix Instructions:")
    lines.append("- RED FLAG metric: investigate cause (data leakage? wrong split? bug?). "
                 "Fix the bug or document honestly.")
    lines.append("- SILENT SUBSTITUTION: implement [EXACT] component as-specified, "
                 "try >=2 approaches before giving up.")
    lines.append("- MISSING COMPONENT: implement now, or add to Limitations with error evidence.")
    lines.append("- OVERWRITE report/report.md with fixes using file_write_tool.")
    return "\n".join(lines)


async def run(workspace: str, extreme: bool = False) -> int:
    ws = Path(workspace).resolve()
    instructions = ws / "INSTRUCTIONS.md"
    if not instructions.exists():
        print(f"ERROR: {instructions} not found", file=sys.stderr)
        return 1

    # RCB subprocess 跑时主 memory.db 可能被 IDE/桌面端锁定 (sqlite WAL),
    # 改用 workspace 下的独立缓存目录. RCB 是无状态离线评测, 不需要跨任务记忆.
    rcb_cache = ws / ".huginn_cache"
    rcb_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HUGINN_CACHE_DIR"] = str(rcb_cache)

    prompt = instructions.read_text(encoding="utf-8")

    from huginn.agent import HuginnAgent
    from huginn.config import HuginnConfig
    from huginn.models.registry import ModelRegistry
    from huginn.tools import register_all_tools

    # snapshot 默认用 ~/.huginn/snapshots, RCB subprocess 跑时该目录可能被
    # IDE/桌面端锁定 (PermissionError). 重定向到 workspace 下的独立目录.
    from huginn.snapshot import file_snapshot as _fs
    _fs._SNAPSHOT_ROOT = rcb_cache / "snapshots"

    cfg = HuginnConfig.from_env()
    # v6 极限模式: 解除一切限制, 性能优先. 更高思考强度 + 更长任务轨迹.
    # ponytail: 不改默认值, 只在 --extreme 时 override. 升级路径是加 profile 系统.
    if extreme:
        os.environ.setdefault("HUGINN_THINKING", "high")
        # v7 长任务: extreme 模式同时放宽 autoloop stop 阈值, 允许 200+ 步轨迹.
        # 对标 Oxelra 206 步. 默认值已放宽 (20/20/10/5), extreme 再翻倍.
        os.environ.setdefault("HUGINN_MAX_CONSECUTIVE_FAILURES", "50")
        os.environ.setdefault("HUGINN_MAX_REFINES", "50")
        os.environ.setdefault("HUGINN_MAX_PIVOTS", "20")
        os.environ.setdefault("HUGINN_DARWIN_STAGNATION_LIMIT", "15")
        cfg = HuginnConfig.from_env()  # 重读 env 拿 thinking
        print("[EXTREME MODE] thinking=high, max_tool_calls=300, context_budget=200K, autoloop thresholds 50/50/20/15", flush=True)

    registry = ModelRegistry.from_config(cfg)
    alias = registry.default_alias()
    if alias:
        model = registry.resolve(alias)
    elif cfg.provider and cfg.provider != "default":
        model = registry.resolve(f"{cfg.provider}/{cfg.model or 'auto'}")
    else:
        print("ERROR: no model configured", file=sys.stderr)
        return 1

    # RCB harness 从 stdout 检测 model 名 (run_task._detect_model)
    model_name = getattr(model, "name", None) or getattr(model, "model_id", None) or str(model)
    print(f'model: {model_name}', flush=True)

    # system prompt: workspace 路径 + 工具操作事实. 让 INSTRUCTIONS.md 做 task gradient.
    # ponytail: 删 CRITICAL override 层 (σ₄) — control loop 不替 LLM 决策"什么重要".
    system_prompt = (
        f"You are an autonomous scientific research agent. "
        f"Your workspace is: {ws}\n"
        f"Current working directory IS the workspace. All relative paths "
        f"(data/, related_work/, code/, outputs/, report/) resolve from here.\n"
        f"Follow INSTRUCTIONS.md as your primary guide — it defines the task.\n"
        f"Prefer real implementations over shortcuts; document failures honestly.\n\n"
        "## Tool facts (sandbox constraints, not priorities)\n"
        "- code_tool: run Python. Sandbox BLOCKS open() and os — CANNOT write files via code_tool.\n"
        "- bash_tool: pip install, run scripts.\n"
        "- file_write_tool: CREATE or OVERWRITE text files (report.md, code/*.py). "
        "Pass FULL content each time.\n"
        "- matplotlib.savefig() WORKS (library code, not AST-scanned) — use it for figures.\n"
        "- code_tool security scanner may false-positive on eval() in torch/numpy — "
        "if so, write script via file_write_tool and run with bash_tool.\n"
        "- file_read_tool/glob/grep: explore data/ and related_work/.\n"
        "- web_search_tool: verify constants, methods, or edge cases.\n\n"
        "## Operating rules\n"
        "- Every response before task completion MUST include a tool call. "
        "Text-only response = task termination.\n"
        "- Push through errors: debug, install missing packages, try alternatives.\n"
        "- Write report/report.md EARLY, then OVERWRITE as you add results.\n"
    )

    # 先注册工具到 ToolRegistry, 再让 agent 从 registry 拉取
    register_all_tools()

    # v6 极限模式: max_tool_calls 300 + context_budget 200K + 每 tool 上限 100
    # 默认 150 / 0 / 50. 极限模式拉满, 让 agent 能跑更长任务轨迹.
    _max_calls = 300 if extreme else 150
    _max_per_tool = 100 if extreme else 50
    _ctx_budget = 200000 if extreme else cfg.context_budget_tokens

    agent = HuginnAgent(
        model=model,
        system_prompt=system_prompt,
        memory_manager=None,
        max_tool_output_tokens=cfg.max_tool_output_tokens,
        context_budget_tokens=_ctx_budget,
        max_tool_calls=_max_calls,
        max_tool_calls_per_tool=_max_per_tool,
        # file_write_tool 写文本文件 (report.md, code/*.py);
        # code_tool 的 sandbox 禁 open(), 只能跑分析/画图 (savefig 库代码不受限).
        # 不给 file_edit_tool: 它要求文件已存在, agent 误用 edit 写新文件会失败.
        tool_filter=[
            "code_tool", "bash_tool",
            "file_read_tool", "file_write_tool",
            "glob", "grep", "web_search_tool",
            "self_observe",
            # G27: 数学工具解除 filter 屏蔽 — repro 数量级错误 (χ=1.0 vs 0.004) 的根因之一
            # 是四个外部适配器 tool_filter 把数学工具整体摘除 (audit 13 F1).
            "symbolic_math_tool", "lean_tool", "validate_tool",
        ],
        # RCB 是无人工 subprocess, 所有工具自动 approve
        auto_approve=True,
    )
    agent.register_tools_from_registry()

    # 3 步认知循环: 论文方法论提取 → 执行 → 自验证
    # ponytail: 不走 autoloop 7 阶段 (太重), 用 3 步循环治 3 个短板:
    #   Step 1 治 "不读论文就动手" — 强制先提取方法核心组件 + baseline 指标
    #   Step 2 治 "方法降级" — checklist 注入, agent 对照方法约束执行
    #   Step 3 治 "不自验证" — 对照 checklist 检查 report 覆盖度, 补缺
    # 用同一 thread_id 保持上下文连续, Step 2 能看到 Step 1 的 checklist.
    from langchain_core.messages import AIMessage

    thread_id = f"rcb_{ws.name}"

    async def _stream_chat(msg: str, step_label: str) -> str:
        """跑一轮 agent.chat, 流式打印 AIMessage, 返回最后的 AI 文本."""
        ai_text = ""
        try:
            async for chunk in agent.chat(msg, thread_id=thread_id):
                msgs = chunk.get("messages", [])
                if not msgs:
                    continue
                last = msgs[-1]
                if not isinstance(last, AIMessage):
                    continue
                content = getattr(last, "content", "")
                if content:
                    print(content, flush=True)
                    ai_text = content
        except Exception as e:
            print(f"ERROR [{step_label}]: {e}", file=sys.stderr)
        return ai_text

    # RCB 3-step 映射 CSM: Step1→S1_DISCOVER, Step2→S4_CONSTRUCT, Step3→S6+S7 (Task 18)
    # ponytail: transition 是 advisory — 不允许就 no-op, 不破坏现有 3-step 流程.
    from huginn.cognitive_engine import TransitionSignal as _RCB_TS

    def _rcb_csm_advance(signal_type: str, ctx: dict) -> None:
        """RCB step 开始时手动推 CSM 状态. advisory: 不允许就 no-op."""
        csm = getattr(agent, "_csm", None)
        if csm is None:
            return
        try:
            csm.transition(_RCB_TS(signal_type, ctx))
        except Exception:
            logger.debug("RCB CSM transition failed", exc_info=True)

    # Step 1: 论文方法论提取
    # agent 读 INSTRUCTIONS.md + related_work/, 输出方法核心组件 + baseline 指标 checklist
    print("\n=== Step 1: Methodology Extraction ===\n", flush=True)
    _rcb_csm_advance("user_goal", {"goal": "understand problem and extract methodology"})
    step1_prompt = (
        f"Read the task instructions below AND explore related_work/ directory for reference papers.\n"
        f"Extract a METHODOLOGY CHECKLIST from the paper:\n"
        f"1. Core method components (model architecture, training protocol, key algorithms).\n"
        f"   For EACH component, label it [EXACT] (must reproduce as-specified) or [VARIANT]\n"
        f"   (justified deviation with reason). Default to [EXACT]. The label forces honesty\n"
        f"   about substitutions — Step 3 will audit them.\n"
        f"2. Key quantitative metrics with the paper's BASELINE VALUES (e.g. 'R²=0.79, MAE=48K').\n"
        f"   These are the targets your results will be compared against in Step 3.\n"
        f"3. Critical implementation details that must be reproduced\n\n"
        f"Output the checklist as a numbered list. Be SPECIFIC (e.g. 'CGCNNConv with gating "
        f"and residual connections', not just 'GNN'). This checklist will guide your implementation "
        f"and will be used in Step 3's substitution audit and sanity check.\n\n"
        f"Task instructions:\n{prompt}"
    )
    checklist = await _stream_chat(step1_prompt, "step1")
    print(f"\n[checklist extracted: {len(checklist)} chars]\n", flush=True)

    # G29: checklist 永驻 system_prompt — 写入 stable_principles (source="checklist"),
    # context.py 的 STABLE_PRINCIPLES 段每轮 build_prompt 重读, 不进 compaction 范围.
    # 修 audit 09: RCB 长任务 compaction 跳过后 checklist 丢失, Step 2/3 看不到方法论约束.
    # ponytail: checklist 是 persona 级输入 (跨 step 不变), 走 stable_principles 通道
    # 比改 prompt_builder 加新段更省代码. 任务结束不清除, 下一任务 init 时会被覆盖语义
    # (新 checklist 会被 store 进来, 旧的仍在文件里但 LLM 会以新为准).
    if checklist and checklist.strip():
        try:
            from huginn.memory import store_stable_principle
            # 截断到 2000 字符防 persona 膨胀, 完整 checklist 在 ws/checklist.md
            store_stable_principle(
                f"[METHODOLOGY CHECKLIST]\n{checklist[:2000]}",
                source="rcb_step1_checklist",
            )
            # 同时写到 ws/checklist.md 让 agent 能 file_read_tool 读完整版
            (ws / "checklist.md").write_text(checklist, encoding="utf-8")
            print(f"[G29: checklist stored as stable_principle + ws/checklist.md]", flush=True)
        except Exception as e:
            print(f"[G29: checklist store skipped: {e}]", flush=True)

    # Step 1.5: Intuitive Gamer + 数学直觉结构识别
    # 两层结合 (arXiv:2510.11503 fast-flat scan + 数学结构识别):
    # - fast flat scan: 不深挖, 快速过一遍 checklist
    # - 数学直觉: 识别每个 item 的数学结构 + invariant, 而非只标难度
    #
    # 平衡点 (advisory + audited, not enforced):
    # - 保守默认: 每个 item 默认 structure=empirical, invariant=none
    # - verifiable_via 枚举 gate: 拿不出验证方法的退回 empirical
    # - 分档处理: hard check (dimensional/exact_formula/conservation_law)
    #            vs soft check (asymptotic/symmetry/topological) vs none
    # - exploratory 合法: 不强制每个 item 都有数学结构
    #
    # ponytail: v7 只做 prompt + 枚举约束, 不写 Lean, 不写 pydantic schema.
    #   v8 升级: 接 cognitive_heat_engine should_imaginate, hard check 失败
    #   触发 imagination; 接 LeanInterface 做形式化验证.
    print("\n=== Step 1.5: Intuitive Gamer + math structure scan ===\n", flush=True)
    scan_prompt = (
        "FAST FLAT SCAN with mathematical structure identification.\n"
        "Goal: identify structure + invariants for each checklist item, not just difficulty.\n\n"
        "For EACH checklist item, output a block:\n"
        "  [item N] structure: <type>\n"
        "    invariant: <one-line statement, or 'none'>\n"
        "    verifiable_via: <method, or 'none'>\n"
        "    anchor: <first-principles reference, or 'exploratory'>\n\n"
        "structure types (enum, pick one):\n"
        "  empirical | symmetry | asymptotic | dimensional | topological | probabilistic | algebraic\n"
        "  - empirical: pure data/observation, no known mathematical structure\n"
        "  - symmetry: invariant under transformation group (rotation, gauge, etc)\n"
        "  - asymptotic: limit behavior (t->inf, x->0) constrains the answer\n"
        "  - dimensional: Buckingham Pi / dimensional homogeneity must hold\n"
        "  - topological: invariant under continuous deformation (winding number, etc)\n"
        "  - probabilistic: distributional constraint (normalization, Bayes consistency)\n"
        "  - algebraic: equation/identity must hold exactly (eigenvalue eq, etc)\n\n"
        "verifiable_via (enum, pick one):\n"
        "  none | dimensional | asymptotic_limit | exact_formula | conservation_law | symmetry_argument | topological_invariant\n"
        "  - 'none' if you cannot specify a concrete verification method (item stays empirical)\n"
        "  - must correspond to the structure type (e.g. structure=dimensional → verifiable_via=dimensional)\n\n"
        "anchor:\n"
        "  - cite first-principles reference (e.g. 'black hole thermodynamics', 'Noether theorem')\n"
        "  - 'exploratory' is valid — accept that structure may be uncertain at this stage\n\n"
        "Constraints:\n"
        "- 1 tool call MAX (file_read or code_tool for quick check). Prefer 0.\n"
        "- Do NOT execute analysis. Do NOT write report.md.\n"
        "- Conservative default: if unsure, structure=empirical, invariant=none, verifiable_via=none.\n"
        "- Do NOT fabricate invariants — unverifiable claims hurt more than they help.\n\n"
        "After all items, output a STRATEGY line:\n"
        "  STRATEGY: <one-line plan — order items by verifiable_via priority:\n"
        "    hard_check (dimensional/exact_formula/conservation_law) first to bank structural wins,\n"
        "    then soft_check (asymptotic/symmetry/topological), then empirical/none last>\n"
        f"\nChecklist:\n{checklist[:4000]}"
    )
    scan_text = await _stream_chat(scan_prompt, "step1.5_structure_scan")
    print(f"\n[structure scan done: {len(scan_text)} chars]\n", flush=True)

    # 写 Meta-Trace entry — role="intuitive_gamer", 带 structure 信息
    try:
        import json as _ig_json
        import time as _ig_time
        _ig_entry = {
            "iteration": 0,
            "ts": _ig_time.time(),
            "role": "intuitive_gamer",
            "attempted": "fast flat scan with mathematical structure identification",
            "found": (scan_text or "")[:500],
            "evidence": [],
            "limitations": [
                "single-sample, no k-sampling (v8 upgrade)",
                "structure labels not schema-validated (v8: pydantic + Lean)",
            ],
            "artifacts": [],
            "next_hint": "execute hard_check items first to bank structural wins",
            "darwin_score": 0.0,
            "supported_ratio": 0.0,
        }
        _ig_trace_path = ws / ".huginn" / "meta_trace.jsonl"
        _ig_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with _ig_trace_path.open("a", encoding="utf-8") as f:
            f.write(_ig_json.dumps(_ig_entry, ensure_ascii=False) + "\n")
        print("[intuitive_gamer + math structure trace entry written]", flush=True)
    except Exception as _e:
        print(f"[intuitive_gamer trace skipped: {_e}]", flush=True)

    # Step 2: 执行任务 (v7 P3: 迭代执行 + Meta-Trace 蒸馏)
    # checklist 已在 thread_id 的对话历史里, agent 能看到. 不需要显式注入.
    #
    # 对标 Oxelra 206 步: 单次 chat() 已能跑 150-300 tool calls (langgraph 内部循环),
    # 但单次 chat() 会因 context 溢出或 agent 主动 emit text-only 提前终止.
    # 迭代执行让 agent 在多次 chat() 间累积进展, 每轮间写 Meta-Trace entry,
    # 下一轮 chat() 的 build_meta_trace_text (P1) 会读回来注入 prompt,
    # 同时 compaction 因 trace 存在会更激进 drop raw messages.
    #
    # ponytail: 不接 AutoloopEngine (它用 CoderRunner/WorkflowEngine, 不写
    #   report/report.md, 会破坏 RCBench 评分). 用 mini-loop + 手写 trace.
    #   升级路径: full AutoloopEngine.run() + 自定义 report writer.
    print("\n=== Step 2: Execution (iterative) ===\n", flush=True)
    _rcb_csm_advance("user_confirmed", {"plan": "execute methodology checklist"})
    # _scan_hint 按 verifiable_via 分档:
    # - hard_check (dimensional/exact_formula/conservation_law): 必须验证, 违反则 debug
    # - soft_check (asymptotic/symmetry/topological): 建议验证, 违反 warn 不 block
    # - none/empirical: 不约束, 按数值精度处理
    # 这呼应物理 precheck "警告 + force_proceed" 偏好 — 结构违反先 warn, 不强制拦截.
    if scan_text and scan_text.strip():
        _scan_hint = (
            f"\n\n## Intuitive Gamer + Math Structure Scan (Step 1.5 result)\n{scan_text}\n\n"
            f"## Execution Guidance\n"
            f"Follow the STRATEGY line above: hard_check items first (bank structural wins),\n"
            f"then soft_check, then empirical/none last.\n\n"
            f"## Invariant Self-Check (per item)\n"
            f"- hard_check (dimensional/exact_formula/conservation_law): result MUST satisfy the invariant.\n"
            f"  Violation → debug and fix, do NOT silent-substitute. This is non-negotiable.\n"
            f"- soft_check (asymptotic/symmetry/topological): result SHOULD satisfy. Violation →\n"
            f"  warn in report.md under 'Limitations' section, continue if fix is expensive.\n"
            f"- none/empirical: no structural constraint, focus on numerical accuracy.\n\n"
            f"## Anti-Fabrication\n"
            f"Do NOT report metrics that violate hard_check invariants. Self-check before writing report.md:\n"
            f"  for each hard_check item, verify result respects invariant. Violations must be fixed, not hidden."
        )
    else:
        _scan_hint = ""
    step2_prompt = (
        "Now execute the task following your methodology checklist. "
        "Implement each [EXACT] component as-specified in the paper. "
        "If a component fails, debug and push through — do NOT silently substitute a simpler model. "
        "Write report/report.md with your results, referencing the checklist items you covered. "
        "Use file_write_tool for report.md, code_tool for analysis/plotting, bash_tool for running scripts."
        + _scan_hint
    )

    import hashlib as _hashlib
    import json as _json
    import time as _time
    _trace_path = ws / ".huginn" / "meta_trace.jsonl"
    _trace_path.parent.mkdir(parents=True, exist_ok=True)
    _max_exec_iters = int(os.environ.get(
        "HUGINN_RCB_EXEC_ITERS",
        "4" if extreme else "2",
    ))
    _prev_report_hash: str | None = None
    _stagnation_count = 0

    for _iter_n in range(_max_exec_iters):
        if _iter_n == 0:
            _iter_prompt = step2_prompt
        else:
            _iter_prompt = (
                f"Continue execution. Iteration {_iter_n + 1}/{_max_exec_iters}.\n"
                f"Review the Research Trace section above for what you've already tried.\n"
                f"Identify the NEXT gap from your checklist (missing component, weak metric, "
                f"untested claim) and address it.\n"
                f"OVERWRITE report/report.md with updated results as you make progress.\n"
                f"If the report is complete and covers ALL checklist items, respond with "
                f"'TASK COMPLETE' followed by a one-paragraph summary. No tool call needed."
            )
        print(f"\n--- Step 2 iter {_iter_n + 1}/{_max_exec_iters} ---\n", flush=True)
        _ai_text = await _stream_chat(_iter_prompt, f"step2_iter{_iter_n + 1}")

        # 写 Meta-Trace entry — P1 的 build_meta_trace_text 下一轮会读到.
        # ponytail: 字段从 self/agent 状态抽, 不调 LLM. RCB mini-loop 不跑 darwin
        #   ratchet, darwin_score/supported_ratio 留 0 (trace 段仍显示 iteration).
        try:
            _report_text = ""
            _report_path_iter = ws / "report" / "report.md"
            if _report_path_iter.exists():
                _report_text = _report_path_iter.read_text(encoding="utf-8")
            _entry = {
                "iteration": _iter_n + 1,
                "ts": _time.time(),
                "role": "rcb_exec",
                "attempted": (_iter_prompt[:200]).replace("\n", " "),
                "found": (_ai_text or "")[:300],
                "evidence": [_report_text[:150]] if _report_text else [],
                "limitations": [],
                "artifacts": ["report/report.md"] if _report_path_iter.exists() else [],
                "next_hint": "continue execution" if _iter_n < _max_exec_iters - 1 else "step3 critique",
                "darwin_score": 0.0,
                "supported_ratio": 0.0,
            }
            with _trace_path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(_entry, ensure_ascii=False) + "\n")
        except Exception as _e:
            print(f"[meta_trace write skipped: {_e}]", flush=True)

        # 停滞检测: report.md 内容 hash 不变 → 可能卡住, 早停
        _curr_hash = (
            _hashlib.md5(_report_text.encode()).hexdigest()
            if _report_text else None
        )
        if _curr_hash == _prev_report_hash and _curr_hash is not None:
            _stagnation_count += 1
            if _stagnation_count >= 2:
                print(
                    f"[stagnation: report.md unchanged for {_stagnation_count} iters, breaking]",
                    flush=True,
                )
                break
        else:
            _stagnation_count = 0
        _prev_report_hash = _curr_hash

        # 早停: agent 明确说完成
        if _ai_text and "TASK COMPLETE" in _ai_text.upper():
            print("[agent signalled TASK COMPLETE, breaking]", flush=True)
            break

    # Step 2.5: report.md 兜底 (σ₆ 修复)
    # 减 CSM (σ₃) 后失去 completion guidance, 加 lightweight gate 补 harmonic.
    # agent 可能在 Step 2 提前终止 (text-only response), 没写 report.md.
    report_path = ws / "report" / "report.md"
    if not report_path.exists():
        print("\n=== Step 2.5: report.md Emergency Write ===\n", flush=True)
        await _stream_chat(
            "CRITICAL: report/report.md does NOT exist. Session scores ZERO without it.\n"
            "Write report/report.md NOW using file_write_tool. Base it on:\n"
            "- Your Step 1 methodology checklist\n"
            "- Your code in code/ and results in outputs/\n"
            "Minimum: # Title, ## Methodology, ## Results (images/*.png), ## Discussion.\n"
            "Be HONEST. A short honest report beats no report. Write it NOW.",
            "step2.5"
        )
    # Deterministic fallback: agent 仍不写就自动生成, 确保有交付物评分
    if not report_path.exists():
        print("[fallback: auto-generating minimal report.md]", flush=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _metrics_parts = []
        for _p in (ws / "outputs").glob("*.json"):
            try:
                _metrics_parts.append(f"### {_p.name}\n```json\n{_p.read_text(encoding='utf-8')}\n```")
            except Exception:
                pass
        _metrics = "\n".join(_metrics_parts) or "None"
        _imgs_dir = ws / "report" / "images"
        _imgs = "\n".join(f"![{p.name}](images/{p.name})" for p in _imgs_dir.glob("*.png")) or "None" if _imgs_dir.exists() else "None"
        _code_dir = ws / "code"
        _code = "\n".join(f"- `{p.name}`" for p in _code_dir.glob("*.py")) or "None" if _code_dir.exists() else "None"
        report_path.write_text(
            f"# Research Report (Auto-generated Fallback)\n\n"
            f"## Methodology\nAgent did not write report.md; auto-generated from artifacts.\n\n"
            f"### Code\n{_code}\n\n### Metrics\n{_metrics}\n\n## Results\n{_imgs}\n",
            encoding="utf-8"
        )

    # Step 3: 对抗式自检 — 不是软自验证, 是 skeptical reviewer 视角
    # ponytail: 治 3 个系统性短板 (跨 4 题评分发现的共性 gap):
    #   A. sanity check — 治 "不可信结果不自检" (M_002 MAE=0.032eV 造假)
    #   B. substitution audit — 治 "沉默方法降级" (4 题全中)
    #   C. hard push — 治 "硬组件轻易放弃" (M_001 无 BO, M_003 无 graph-VAE)
    # 双层 critique (Task 21): object mode (report) + meta mode (directive).
    #   - Layer 1 (object): 独立 LLM 调用读 report.md + checklist → red flags, 直接反馈 agent.
    #   - Layer 2 (meta): reflection._handle_s7_self_modify 在 S7 状态自动调 (Task 2),
    #                     评估 agent 自修改 proposal, accept→stable_principle / reject→rejection log.
    #   object + meta 共享同一 LLM 实例 (model 参数), 不同 system prompt.
    #   合并: object verdict 进 step3_prompt (本轮修复); meta verdict 走 sidecar (下轮 system_prompt).
    print("\n=== Step 3: Adversarial Self-Critique ===\n", flush=True)
    # S6_FEEDBACK: critique 视角找 gap; reflection 检测到实质 gap 时自动进 S7 (Task 2)
    _rcb_csm_advance("tool_failure", {"reason": "adversarial critique — find gaps"})

    # Layer 1 — object mode: 独立 LLM 调用读 report.md + checklist, 输出结构化 red flags.
    # 失败/无 report 时降级为纯 self-critique, 不阻塞 Step 3.
    external_critique_block = ""
    object_verdict = None
    if report_path.exists() and checklist:
        try:
            report_text = report_path.read_text(encoding="utf-8")
            print(f"[adversarial_critique: reading {len(report_text)} chars of report.md]", flush=True)
            object_verdict = await adversarial_critique(
                model, report_text, checklist, mode="object",
            )
            # G28: 数值重算 — report 写的 MAE=0.05 vs outputs/ 实际 MAE 对不上就记 red flag
            # ponytail: 不调 validate_tool/numerical_tool (agent-side 工具), 直接 Python 重算.
            # 天花板同 _recompute_report_metrics 注释. 不破坏 object/meta 双层 critique:
            # 走 object verdict 的额外字段 recomputed_red_flags, format_critique_for_agent 已处理.
            try:
                recomputed = _recompute_report_metrics(report_text, ws)
                if recomputed:
                    object_verdict.setdefault("recomputed_red_flags", []).extend(recomputed)
                    # 有 fabrication 嫌疑就强制 fail, 不让 LLM critique 放过
                    if object_verdict.get("overall_verdict") == "pass":
                        object_verdict["overall_verdict"] = "fix_needed"
                    print(f"[G28: {len(recomputed)} metric claim(s) mismatch recomputed values]", flush=True)
                    external_critique_block = format_critique_for_agent(object_verdict)
            except Exception as e:
                print(f"[G28: recompute skipped: {e}]", flush=True)
            external_critique_block = format_critique_for_agent(object_verdict)
            print(f"[adversarial_critique: verdict={object_verdict.get('overall_verdict', '?')}]", flush=True)
        except Exception as e:
            print(f"[adversarial_critique: skipped due to error: {e}]", flush=True)
    else:
        print("[adversarial_critique: skipped — report.md or checklist missing]", flush=True)

    # Layer 2 — meta mode: 触发 CSM 进 S6_FEEDBACK → S7_SELF_MODIFY,
    # reflection._handle_s7_self_modify 自动调 adversarial_critique(mode="meta").
    # ponytail: Task 18 改用 RCB_CSM_SUBSET (不再全 skip CSM), reflection loop
    #           现在正常运行, S7 handler 会被 reflection 自动调. 这里显式 trigger 是
    #           belt-and-suspenders — 确保 object_verdict 非 pass 时一定进 S7.
    try:
        from huginn.cognitive_engine import TransitionSignal, CognitiveState
        csm = getattr(agent, "_csm", None)
        if csm is not None and object_verdict is not None:
            verdict_flag = object_verdict.get("overall_verdict", "fix_needed")
            # 非 pass 视作 gap 信号, 触发 S6_FEEDBACK (Task 18 显式触发点)
            sig = "tool_failure" if verdict_flag != "pass" else "tool_success"
            new_state = csm.transition(TransitionSignal(sig, {
                "objective": "step3_critique",
                "result_summary": f"object_verdict={verdict_flag}",
            }))
            # S6 + 实质 gap → S7_SELF_MODIFY, reflection 自动调 meta mode (Task 2)
            if new_state == CognitiveState.S6_FEEDBACK and verdict_flag != "pass":
                csm.transition(TransitionSignal("gap_found", {
                    "gap": external_critique_block[:200] or "step3 object critique red flags",
                }))
    except Exception:
        logger.debug("Step 3 CSM S6/S7 trigger failed", exc_info=True)

    step3_prompt = (
        "ADVERSARIAL SELF-CRITIQUE. You are now a SKEPTICAL REVIEWER who wants to score this report LOW. "
        "Do NOT be lenient with yourself.\n\n"
        "## A. Sanity Check (do this FIRST — catches fabricated/impossible results)\n"
        "Read your report/report.md. Extract EVERY quantitative claim (MAE, R², accuracy, loss, etc.).\n"
        "Compare each to the paper's baseline value from your Step 1 checklist.\n"
        "Build a table: | Metric | Paper Value | Your Value | Better? |\n"
        "If ANY of your metrics is BETTER than the paper's — that is a RED FLAG.\n"
        "Investigate why: data leakage? wrong train/test split? simplified geometry? fabricated?\n"
        "Fix the bug, or honestly document the discrepancy. "
        "Implausibly good results get ZERO from reviewers.\n\n"
        "## B. Substitution Audit (catches silent methodology downgrade)\n"
        "List every [EXACT] component from your Step 1 checklist.\n"
        "For each, answer honestly: did I implement it AS-SPECIFIED, or did I substitute a simpler alternative?\n"
        "  - Substituted WITHOUT trying the real implementation → FAILURE. Implement it now.\n"
        "  - Substituted AFTER ≥2 genuine failed attempts → document the attempts with error messages.\n"
        "  'I used Random Forest instead of VAE because VAE is hard' is NOT acceptable.\n"
        "  'I used GCNConv instead of CGCNNConv because it was easier' is NOT acceptable.\n\n"
        "## C. Coverage Check\n"
        "List checklist items COVERED (with evidence from report) vs MISSING/WEAK.\n\n"
        "## D. Fix & Rewrite\n"
        "For each gap found in A/B/C:\n"
        "  - Missing metric → compute it now (run code_tool)\n"
        "  - Missing [EXACT] component → implement it (push through, try ≥2 approaches before giving up)\n"
        "  - Implausible result → fix the bug or document honestly why it's off\n"
        "OVERWRITE report/report.md with: improved results + baseline comparison table + "
        "honest Limitations section (only for items where you tried ≥2 approaches and genuinely failed).\n"
        "Use file_write_tool for the rewrite."
    )
    if external_critique_block:
        step3_prompt = external_critique_block + "\n\n## Now act on the critique above:\n" + step3_prompt
    await _stream_chat(step3_prompt, "step3")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Huginn RCB runner")
    parser.add_argument("--workspace", required=True, help="RCB workspace path")
    parser.add_argument(
        "--extreme", action="store_true",
        help="v6 极限模式: thinking=high, max_tool_calls=300, context_budget=200K",
    )
    args = parser.parse_args()

    rc = asyncio.run(run(args.workspace, extreme=args.extreme))
    sys.exit(rc)


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        # Task 3 self-check: meta mode 早期拒绝 (不调 LLM)
        # ponytail: 命中查重直接返回, llm_client=None 也能跑, 验证 ponytail 优化没退化.
        # 用 asyncio.run 包裹因 adversarial_critique 是 async (object mode 调用点 L434 依赖)
        rejections = ["always use Tanimoto kernel for GP", "add CRITICAL: never use RBF"]
        proposal = "always use Tanimoto kernel for GP regression"
        result = asyncio.run(adversarial_critique(
            mode="meta",
            proposal=proposal,
            recent_rejections=rejections,
            system_prompt_summary="",
            llm_client=None,
        ))
        assert result["verdict"] == "reject", f"expected reject, got {result}"
        assert result.get("early_reject") is True, "should be early_reject"
        print("Task 3 self-check PASS")
        sys.exit(0)
    main()
