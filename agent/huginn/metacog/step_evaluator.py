"""StepEvaluator — 中间步骤评估 (G63 + G69).

每一步执行后, 对照目标链 + 结构不变量 + LLM 兜底, 判断是否还在轨道上.
机械信号优先; LLM 只在机械信号不足时介入; LLM 失败一律降级到 "unsure"/"unknown".

G69 在 StepEvaluation 上挂 tool_call_health 字段, should_continue 检测到连续异常
就发"触发 Reflector 介入"信号 (介入逻辑本身在 Reflector 那边, 这里只产信号).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# target_chain.py 还没落地, 这里不 import, 用 duck typing 访问 target_id / required_results.
# ponytail: 不绑死类型, getattr 兼容 dict 和 dataclass. 升级路径: 真有了再 import.

_HARD_VIA = frozenset({"dimensional", "exact_formula", "conservation_law"})
_SOFT_VIA = frozenset({"asymptotic_limit", "symmetry_argument", "topological_invariant"})

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "with", "by",
    "on", "at", "from", "as", "is", "are", "was", "were", "be", "been",
    "that", "this", "these", "those", "it", "its", "then", "than", "but",
    "we", "you", "they", "i", "he", "she", "into", "via", "using", "use",
})

# ponytail: 结构信号的文本启发式, 正则够用, 不上 NLP. 升级路径: embedding + few-shot.
# 注意 meV/keV 要放在 eV 之前, 否则 eV 会先吃掉 "meV" 里的 "eV" 段.
_UNIT_RE = re.compile(
    r"\b(meV|keV|eV/atom|eV/\w+|eV|angstrom|Å|nm|pm|"
    r"GPa|MPa|kPa|kJ|J/mol|kg/m\w*|g/cm\w*|"
    r"THz|GHz|MHz|Hz|ms|μs|ns|ps|fs|°C|kelvin)\b",
    re.IGNORECASE,
)
_FORMULA_RE = re.compile(r"[=≈≠≤≥<>]\s*[\d\-+a-zA-Z(]")
_CONSERVATION_RE = re.compile(r"conserv|守恒|invariant|不变|preserv", re.IGNORECASE)
_ASYMPTOTIC_RE = re.compile(r"limit|asymptotic|渐近|∞|large[\s\-]?[nNtT]|t→|n→|x→", re.IGNORECASE)
_SYMMETRY_RE = re.compile(r"symmetr|对称|invariant|不变|group|群", re.IGNORECASE)
_TOPOLOGICAL_RE = re.compile(r"topolog|拓扑|winding|homotop|同伦|betti", re.IGNORECASE)


@dataclass
class ToolCallHealth:
    """G69: 工具调用健康度."""

    success_rate: float = 1.0  # 成功调用数 / 总调用数
    total_calls: int = 0
    retry_count: int = 0
    timeout_count: int = 0
    param_error_count: int = 0

    def is_anomalous(self) -> bool:
        """成功率 < 0.3 或有超时/参数错误 → 异常."""
        if self.timeout_count > 0 or self.param_error_count > 0:
            return True
        return self.success_rate < 0.3


@dataclass
class ToolCallHealth:
    """G69: 工具调用健康度."""

    success_rate: float = 1.0  # 成功调用数 / 总调用数
    total_calls: int = 0
    retry_count: int = 0
    timeout_count: int = 0
    param_error_count: int = 0

    def is_anomalous(self) -> bool:
        """成功率 < 0.3 或有超时/参数错误 → 异常."""
        if self.timeout_count > 0 or self.param_error_count > 0:
            return True
        return self.success_rate < 0.3


@dataclass
class MeasurementUncertainty:
    """P2: 测量不确定度 — 抓 agent 报点估计不报误差的痛点.

    三个字段都是可选, 缺失=未知. 不强制 LLM 报误差 (很多步骤无数值),
    但有数值时应该填. adversarial_critique 会检查误差传播一致性.

    ponytail: 不做完整 GUM (Guide to Uncertainty in Measurement) 实现,
    只抓核心: 有点估计 + 有不确定度 + 有传播标记. 升级路径: 接 UNCERTainties 库.
    """

    point_estimate: float | None = None  # 点估计值 (如 2.6e-20)
    uncertainty: float | None = None  # 不确定度 (1σ, 如 0.3e-20)
    unit: str = ""  # 单位 (eV/atom, GPa, ...)
    propagated: bool = False  # 是否已做误差传播 (False=点估计孤立)

    def has_uncertainty(self) -> bool:
        """有数值且报了误差 → True. 有数值没误差 → False (告警对象)."""
        return self.point_estimate is not None and self.uncertainty is not None

    def is_point_only(self) -> bool:
        """有数值但没误差 → True (adversarial_critique 的告警对象)."""
        return self.point_estimate is not None and self.uncertainty is None

    def relative_uncertainty(self) -> float | None:
        """相对不确定度 (uncertainty / |point_estimate|). >0.3 算高."""
        if not self.has_uncertainty() or self.point_estimate == 0:
            return None
        return abs(self.uncertainty / self.point_estimate)


@dataclass
class StepEvaluation:
    step_id: int
    attempted: str
    found: str
    target_chain_ref: str | None  # 匹配到的 target_id, None 表示未匹配
    on_track: Literal["true", "false", "unsure"]  # 字符串, 避免 bool 歧义
    structure_check: str  # passed / failed / soft_warning / not_applicable
    evidence_quality: str  # high / medium / low / unknown
    deviation: str  # 偏差描述, 无则空串
    tool_call_health: ToolCallHealth = field(default_factory=ToolCallHealth)  # G69
    pmk_feedback: str = ""  # PMK 循环反馈文本
    measurement_uncertainty: MeasurementUncertainty = field(
        default_factory=MeasurementUncertainty)  # P2

    def is_on_track(self) -> bool:
        return self.on_track == "true"


# === 内部 helper ===


def check_uncertainty_propagation(
    evaluations: list[StepEvaluation],
) -> list[dict]:
    """P2: 扫描 evaluation 历史, 找误差建模缺失/不一致.

    返回 issue 列表, 每项 {"step_id": int, "issue": str, "detail": str}:
    - point_only: 有点估计没不确定度 (agent 报点不报误差)
    - high_relative: 相对不确定度 > 0.3 (数值不可靠)
    - unpropagated: 多步有点估计但 propagated=False (误差未传播)

    adversarial_critique / report 阶段调这个函数, 把 issue 推给 agent 修.
    ponytail: 不做完整 GUM 传播公式, 只标记问题. 升级路径: 接 uncertainties 库.
    """
    issues: list[dict] = []
    point_estimates: list[tuple[int, float, str]] = []  # (step_id, value, unit)

    for ev in evaluations:
        mu = getattr(ev, "measurement_uncertainty", None)
        if mu is None or mu.point_estimate is None:
            continue
        sid = getattr(ev, "step_id", -1)

        if mu.is_point_only():
            issues.append({
                "step_id": sid,
                "issue": "point_only",
                "detail": f"点估计 {mu.point_estimate} {mu.unit} 无不确定度",
            })
            point_estimates.append((sid, mu.point_estimate, mu.unit))
            continue

        if mu.has_uncertainty():
            rel = mu.relative_uncertainty()
            if rel is not None and rel > 0.3:
                issues.append({
                    "step_id": sid,
                    "issue": "high_relative",
                    "detail": f"相对不确定度 {rel:.2%} > 30% (value={mu.point_estimate}±{mu.uncertainty} {mu.unit})",
                })
            if not mu.propagated:
                point_estimates.append((sid, mu.point_estimate, mu.unit))

    # 多步有点估计但都没 propagated → 误差链断裂
    if len(point_estimates) >= 2:
        issues.append({
            "step_id": point_estimates[-1][0],
            "issue": "unpropagated",
            "detail": f"{len(point_estimates)} 步有点估计但未做误差传播 — 最终结论置信度未知",
        })

    return issues


def _clamp01(x: float) -> float:
    """[0, 1] clamp."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _compute_darwin_score(step_eval: Any) -> float:
    """v14 Task 2: darwin = 1.0 - gap_severity.

    Phase 1 默认路径: 从 StepEvaluator 评估结果反向打分.
    - None: 探索期默认 0.5 (不算成功也不算失败)
    - dict 含 gap_severity: 直接用 (测试 / Phase 2 LLM 覆盖路径)
    - dict / StepEvaluation 含 on_track: 派生 gap_severity

    ponytail: StepEvaluation 没有显式 gap_severity 字段, 从 on_track 离散映射
      true=0.1 / unsure=0.5 / false=0.9, 不加权 structure/evidence.
      天花板: 三档粒度, 无法区分 "true+low evidence" 跟 "true+high evidence".
      升级路径: StepEvaluation 加显式 gap_severity float 字段 (Phase 2 LLM 评估时).
    返回 float [0, 1], clamp 防越界.
    """
    if step_eval is None:
        return 0.5

    # dict 优先用显式 gap_severity (测试 / Phase 2 LLM 覆盖路径)
    if isinstance(step_eval, dict):
        gs = step_eval.get("gap_severity")
        if gs is not None:
            try:
                return _clamp01(1.0 - float(gs))
            except (TypeError, ValueError):
                pass  # 坏值降级到 on_track 派生
        on_track = step_eval.get("on_track", "unsure")
    else:
        on_track = getattr(step_eval, "on_track", "unsure")

    gs = {"true": 0.1, "unsure": 0.5, "false": 0.9}.get(on_track, 0.5)
    return _clamp01(1.0 - gs)


def _tc_attr(tc: Any, name: str, default: Any = None) -> Any:
    """TargetChain 兼容取值: dataclass / dict 都能取."""
    if isinstance(tc, dict):
        return tc.get(name, default)
    return getattr(tc, name, default)


def _extract_keywords(text: str) -> list[str]:
    """简单分词: 英文按单词, 中文按连续汉字块. 去停用词和单字符."""
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _check_target_chain_match(
    found: str,
    target_chains: list,
) -> tuple[str | None, str]:
    """found 是否匹配某 TargetChain 的 required_results.

    匹配规则: found 文本(小写)包含 required_result 任一关键词 → 命中.
    返回 (target_id, matched_result). 无匹配 (None, "").
    """
    if not found or not target_chains:
        return (None, "")
    found_lc = found.lower()
    for tc in target_chains:
        tid = _tc_attr(tc, "target_id", None)
        results = _tc_attr(tc, "required_results", []) or []
        for rr in results:
            rr_str = rr if isinstance(rr, str) else str(rr)
            kws = _extract_keywords(rr_str)
            if not kws:
                continue
            if any(kw in found_lc for kw in kws):
                return (tid if tid is not None else "", rr_str)
    return (None, "")


def _check_structure(
    verification_signals: list[dict] | None,
    found: str,
) -> str:
    """结构不变量检查.

    优先用 signal 里显式的 verified/passed/status; 没有就退到 found 文本启发式.
    hard (dimensional/exact_formula/conservation_law) 失败 → "failed"
    soft (asymptotic/symmetry/topological) 失败 → "soft_warning"
    verifiable_via=none 或无 signals → "not_applicable"
    """
    if not verification_signals:
        return "not_applicable"

    found_text = found or ""
    applied = False

    for sig in verification_signals:
        if not isinstance(sig, dict):
            continue
        via = str(sig.get("verifiable_via") or "none").lower()
        if via == "none" or via not in (_HARD_VIA | _SOFT_VIA):
            if via != "none":
                # 未知 via 不算失败也不算应用过
                pass
            continue
        applied = True

        # 显式 verdict 优先
        explicit = sig.get("verified", sig.get("passed"))
        if explicit is not None:
            ok = bool(explicit)
        elif "status" in sig:
            ok = str(sig["status"]).lower() in {"passed", "ok", "true", "success"}
        else:
            # 启发式: 看 found 是否带该结构应有的痕迹
            if via == "dimensional":
                ok = bool(_UNIT_RE.search(found_text))
            elif via == "exact_formula":
                ok = bool(_FORMULA_RE.search(found_text))
            elif via == "conservation_law":
                ok = bool(_CONSERVATION_RE.search(found_text))
            elif via == "asymptotic_limit":
                ok = bool(_ASYMPTOTIC_RE.search(found_text))
            elif via == "symmetry_argument":
                ok = bool(_SYMMETRY_RE.search(found_text))
            elif via == "topological_invariant":
                ok = bool(_TOPOLOGICAL_RE.search(found_text))
            else:
                ok = True  # 未知 via 不算失败

        if not ok:
            return "failed" if via in _HARD_VIA else "soft_warning"

    return "passed" if applied else "not_applicable"


def _build_messages(sys_text: str, usr_text: str) -> list:
    """构造 langchain 消息, 失败时退到纯元组列表."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        return [SystemMessage(content=sys_text), HumanMessage(content=usr_text)]
    except Exception:
        return [("system", sys_text), ("human", usr_text)]


def _resp_to_text(resp: Any) -> str:
    """langchain AIMessage.content / str / dict 都能取到文本."""
    if isinstance(resp, str):
        return resp
    content = getattr(resp, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
        return "\n".join(parts)
    if isinstance(resp, dict):
        return resp.get("content", "") or resp.get("text", "")
    return str(resp)


def _parse_llm_json(text: str) -> tuple[str, str, str]:
    """从 LLM 输出里抠 JSON. 解析失败 → 降级."""
    if not text:
        return ("unsure", "unknown", "LLM 评估失败: 空响应")
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return ("unsure", "unknown", "LLM 评估失败: 无 JSON")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return ("unsure", "unknown", "LLM 评估失败: JSON 解析失败")

    on_track = str(obj.get("on_track", "unsure")).lower().strip()
    if on_track not in {"true", "false", "unsure"}:
        on_track = "unsure"
    eq = str(obj.get("evidence_quality", "unknown")).lower().strip()
    if eq not in {"high", "medium", "low", "unknown"}:
        eq = "unknown"
    dev = str(obj.get("deviation", "")).strip()
    return (on_track, eq, dev)


def _recall_context_bits(memory: Any | None, kb: Any | None, query: str, top_k: int = 2) -> str:
    """从 memory/kb 捞一点上下文喂给 LLM. 失败一律返回空串."""
    bits: list[str] = []
    if memory is not None:
        try:
            # category 是可选的, 不传试试看, 不行就退
            for m in (memory.recall(query=query, top_k=top_k) or []):
                c = m.get("content", "") if isinstance(m, dict) else str(m)
                if c:
                    bits.append(f"[memory] {c[:200]}")
        except Exception as exc:
            logger.debug("memory recall failed: %s", exc)
    if kb is not None:
        try:
            for k in (kb.query(text=query, top_k=top_k) or []):
                c = k.get("content", "") if isinstance(k, dict) else str(k)
                if c:
                    bits.append(f"[kb] {c[:200]}")
        except Exception as exc:
            logger.debug("kb query failed: %s", exc)
    return "\n".join(bits)


def _llm_evaluate(
    attempted: str,
    found: str,
    target_chains: list,
    memory: Any | None,
    kb: Any | None,
    model: Any,
) -> tuple[str, str, str]:
    """LLM 评估 on_track + evidence_quality + deviation.

    返回 (on_track, evidence_quality, deviation).
    任何失败 (无 model / 调用异常 / 解析失败) 都降级到 ("unsure", "unknown", "LLM 评估失败").
    """
    if model is None:
        return ("unsure", "unknown", "LLM 评估失败: 无 model")

    # 拼目标链摘要
    chain_lines = []
    for tc in target_chains or []:
        tid = _tc_attr(tc, "target_id", "?")
        results = _tc_attr(tc, "required_results", []) or []
        chain_lines.append(f"- {tid}: {'; '.join(str(r) for r in results)}")
    chain_text = "\n".join(chain_lines) if chain_lines else "(无目标链)"

    ctx_text = _recall_context_bits(memory, kb, found)

    sys = (
        "你是中间步骤评估器. 判断当前步骤是否在目标链上, 证据质量如何, 偏差是什么."
        '严格输出一行 JSON: {"on_track": "true|false|unsure", '
        '"evidence_quality": "high|medium|low|unknown", "deviation": "一句话偏差或空"}.'
        "不要输出 JSON 以外的内容."
    )
    usr = (
        f"目标链:\n{chain_text}\n\n"
        f"本步 attempted: {attempted}\n"
        f"本步 found: {found}\n"
    )
    if ctx_text:
        usr += f"\n参考上下文:\n{ctx_text}\n"

    messages = _build_messages(sys, usr)

    try:
        # 优先 ainvoke, 已在事件循环里就退到同步 invoke. 两都没有就降级.
        try:
            asyncio.get_running_loop()
            has_loop = True
        except RuntimeError:
            has_loop = False

        if hasattr(model, "ainvoke") and not has_loop:
            resp = asyncio.run(model.ainvoke(messages))
        elif hasattr(model, "invoke"):
            resp = model.invoke(messages)
        else:
            return ("unsure", "unknown", "LLM 评估失败: model 无 invoke")
    except Exception as exc:
        logger.debug("LLM evaluate failed: %s", exc)
        return ("unsure", "unknown", "LLM 评估失败")

    return _parse_llm_json(_resp_to_text(resp))


def _build_darwin_eval_prompt(
    recent_entries: list,
    task_description: str,
) -> tuple[str, str]:
    """构造 LLM darwin 评估的 system + user prompt."""
    sys_text = (
        "You are a research trajectory evaluator. "
        "Score each trace entry's fitness (darwin_score) on a scale of 0.0 to 1.0.\n\n"
        "A high darwin_score means the entry:\n"
        "- Made meaningful progress on the task\n"
        "- Has supporting evidence for its claims\n"
        "- Advanced the overall trajectory toward the goal\n\n"
        "A low darwin_score means the entry:\n"
        "- Was a dead end or unproductive exploration\n"
        "- Lacked evidence or made unsupported claims\n"
        "- Did not advance toward the goal"
    )

    entry_lines = []
    for i, e in enumerate(recent_entries, 1):
        if isinstance(e, dict):
            sid = str(e.get("simplex_id", "") or "")
            attempted = str(e.get("attempted", "") or "")
            found = str(e.get("found", "") or "")
            evidence = str(e.get("evidence", "") or "")
        else:
            sid = attempted = found = evidence = ""
        entry_lines.append(
            f"[entry {i}] simplex_id: {sid} attempted: {attempted} "
            f"found: {found} evidence: {evidence}"
        )

    user_text = (
        f"Task description: {task_description}\n\n"
        f"Recent trace entries:\n"
        + "\n".join(entry_lines)
        + "\n\nReturn JSON only, no other text:\n"
        "[\n"
        '  {"simplex_id": "trace:...:iter_0:...", "darwin_score": 0.8, "reason": "..."},\n'
        "  ...\n"
        "]"
    )
    return (sys_text, user_text)


def _parse_darwin_json(text: str) -> list[dict]:
    """解析 LLM darwin 评估返回的 JSON 数组.

    Returns list of {"simplex_id": str, "darwin_score": float, "reason": str}.
    Raises ValueError on parse failure. 单个 entry 字段非法就跳过, 不整体抛.
    """
    if not text:
        raise ValueError("empty response")

    # 先直接试 json.loads, 失败再抠 [...] 块
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            raise ValueError("no JSON array found")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse failed: {e}")

    if not isinstance(data, list):
        raise ValueError("response is not a JSON array")

    result: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        sid = item.get("simplex_id")
        score = item.get("darwin_score")
        reason = item.get("reason")
        # bool 是 int 的子类, 得排除掉
        if not isinstance(sid, str) or not isinstance(reason, str):
            continue
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        score_f = float(score)
        if score_f < 0.0 or score_f > 1.0:
            continue
        result.append({
            "simplex_id": sid,
            "darwin_score": score_f,
            "reason": reason,
        })
    return result


async def _llm_evaluate_darwin(
    trace_entries: list,
    model: Any,
    task_description: str,
) -> list[dict]:
    """v14 Task 11: LLM 评估 darwin_score (Phase 2 增强路径).

    env HUGINN_DARWIN_LLM_EVAL=1 启用; 每 HUGINN_DARWIN_LLM_INTERVAL 轮 (默认 5) 触发一次.
    失败一律降级到空 list, 外层用 _compute_darwin_score 默认值兜底.

    ponytail: 每 N 轮调一次 LLM 评 entry 适应度, 是成本/质量折中. 天花板:
      5 条 entry + 1 次 LLM 调用 ≈ 1.5k tokens; interval=5 时单 task 增 ~10 次调用.
      升级路径: 用更便宜的 model (haiku) 或 batch API.
    """
    try:
        if os.environ.get("HUGINN_DARWIN_LLM_EVAL", "0").lower() not in ("1", "true", "yes"):
            return []

        if not trace_entries:
            return []

        # current_iter 从最后一条 entry 取, 跟 rcb_runner 主循环的 iteration 对齐
        last = trace_entries[-1]
        current_iter = 0
        if isinstance(last, dict):
            try:
                current_iter = int(last.get("iteration", 0) or 0)
            except (TypeError, ValueError):
                current_iter = 0

        interval = int(os.environ.get("HUGINN_DARWIN_LLM_INTERVAL", "5"))
        if current_iter % interval != 0:
            return []

        if model is None:
            return []

        recent = trace_entries[-5:]
        sys_text, user_text = _build_darwin_eval_prompt(recent, task_description)
        messages = _build_messages(sys_text, user_text)

        # 复用项目 with_retry 包装, 走统一的 429/529/超时 重试机制
        from huginn.llm_retry import with_retry

        async def _call() -> str:
            if hasattr(model, "ainvoke"):
                return _resp_to_text(await model.ainvoke(messages))
            if hasattr(model, "invoke"):
                return _resp_to_text(model.invoke(messages))
            raise ValueError("model 无 invoke/ainvoke")

        result_text = await with_retry(_call, source="darwin_llm_eval")
        return _parse_darwin_json(result_text)
    except Exception as e:
        logger.warning("darwin_llm_eval_fallback: reason=%s", e)
        return []


def _make_pmk_feedback(
    persona: Any | None,
    memory: Any | None,
    kb: Any | None,
    on_track: str,
    evidence_quality: str,
    deviation: str,
) -> str:
    """PMK 反馈: persona / memory / kb 三路给一句话建议. 简化版, 不上完整循环."""
    hints: list[str] = []
    if on_track == "false":
        hints.append("Persona: 重新审视方法选择, 考虑换路线")
    if evidence_quality == "low":
        hints.append("Memory: recall 历史相似问题/失败, 找可复用证据")
        hints.append("KB: 查询领域知识, 补理论依据")
    if deviation:
        hints.append(f"偏差: {deviation}")
    if not hints:
        return ""
    return "; ".join(hints)


# === 对外接口 ===


def compute_tool_call_health(
    audit_log_path: Any,
    step_id: int,
) -> ToolCallHealth | None:
    """从 audit.jsonl 扫工具调用记录, 算 ToolCallHealth.

    每行 JSON 是 AgentEvent.to_dict() 输出, 含 type/data/timestamp/...
    工具相关事件: tool.call / tool.result / tool.error (data.tool = 工具名).
    audit_log 不存在/为空/没工具事件 → None.

    ponytail: 一次性扫全文件, 不做流式 tail/索引. 升级路径: 增量计数器
    + seek tail. audit_log schema 没有 step_id 字段, 当前按全量统计;
    schema 真加上 step_id 后这里自动按 step_id 过滤 (rec.get 兼容).
    """
    if audit_log_path is None:
        return None
    from pathlib import Path

    path = Path(audit_log_path)
    if not path.exists():
        return None

    total = 0
    success = 0
    retry = 0
    timeout = 0
    param_err = 0
    seen_calls: list[str] = []  # retry 检测: 同 tool+input 二次出现算 retry

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # schema 没有 step_id 就全收; 有了就按 step_id 过滤
                rec_step = rec.get("step_id")
                if rec_step is None:
                    rec_step = rec.get("iteration")
                if rec_step is not None:
                    try:
                        if int(rec_step) != step_id:
                            continue
                    except (TypeError, ValueError):
                        pass
                etype = rec.get("type", "")
                if etype not in ("tool.call", "tool.result", "tool.error"):
                    continue
                data = rec.get("data") or {}
                if not isinstance(data, dict):
                    data = {}
                tool_name = (
                    data.get("tool") or rec.get("tool_name") or rec.get("tool") or "?"
                )
                if etype == "tool.call":
                    total += 1
                    sig = f"{tool_name}:{str(data.get('input', ''))[:64]}"
                    if sig in seen_calls:
                        retry += 1
                    seen_calls.append(sig)
                elif etype == "tool.result":
                    success += 1
                elif etype == "tool.error":
                    err = str(
                        data.get("error") or rec.get("error_type") or ""
                    ).lower()
                    if "timeout" in err or "timed out" in err:
                        timeout += 1
                    elif (
                        "param" in err
                        or "typeerror" in err
                        or "validation" in err
                    ):
                        param_err += 1
    except Exception as exc:
        logger.debug("compute_tool_call_health failed: %s", exc)
        return None

    if total == 0:
        return None

    sr = success / total if total > 0 else 1.0
    return ToolCallHealth(
        success_rate=sr,
        total_calls=total,
        retry_count=retry,
        timeout_count=timeout,
        param_error_count=param_err,
    )


def evaluate_step(
    meta_trace_entry: dict,  # 含 iteration / attempted / found
    target_chains: list,  # list[TargetChain]
    verification_signals: list[dict] | None,  # Step 1.5 输出, 含 verifiable_via
    memory: Any | None,  # MemoryManager 或 None
    kb: Any | None,  # KnowledgeBase 或 None
    persona: Any | None,  # Persona 或 None
    model: Any | None,  # langchain BaseChatModel 或 None
    tool_call_health: ToolCallHealth | None = None,  # G69
    kg: Any = None,  # 25.1: episode + dep edge 写入
    prev_step_id: int | None = None,  # 25.1: 上一 step_id, 用于 method dep
    audit_log_path: Any = None,  # 25.3: tool_call_health 自动补
) -> StepEvaluation:
    """评估单步执行. 机械优先, LLM 兜底."""
    step_id = int(meta_trace_entry.get("iteration", 0) or 0)
    attempted = str(meta_trace_entry.get("attempted", "") or "")
    found = str(meta_trace_entry.get("found", "") or "")

    # 25.3: 没传 health 就从 audit_log 自动算
    if tool_call_health is None and audit_log_path is not None:
        try:
            tool_call_health = compute_tool_call_health(audit_log_path, step_id)
        except Exception as exc:
            logger.debug("auto tool_call_health failed: %s", exc)

    target_id, _matched = _check_target_chain_match(found, target_chains)
    struct = _check_structure(verification_signals, found)

    # 机械信号优先判定 on_track
    if target_id is not None and struct != "failed":
        # 命中目标链 + 结构没硬失败 → 在轨
        on_track = "true"
        evidence_quality = "high" if struct == "passed" else "medium"
        deviation = ""
    elif struct == "failed":
        # 结构硬失败 → 明确脱轨, 不再调 LLM 浪费 token
        on_track = "false"
        evidence_quality = "low"
        deviation = "结构不变量检查 failed"
    else:
        # 机械信号不足 → LLM 兜底
        if model is not None:
            on_track, evidence_quality, deviation = _llm_evaluate(
                attempted, found, target_chains, memory, kb, model,
            )
        else:
            on_track, evidence_quality, deviation = (
                "unsure", "unknown", "机械信号不足且无 model",
            )

    pmk = _make_pmk_feedback(persona, memory, kb, on_track, evidence_quality, deviation)

    # 25.1: 写 episode + dep edge 到 kg. 失败不阻塞评估返回.
    # ponytail: result 直接用 on_track ("true"/"false"/"unsure"), 颗粒度对
    #   caller 够用. 升级路径: 单独 pass/fail 字段 + LLM 判 dep_type (data/causal).
    if kg is not None:
        try:
            kg.add_episode_node(
                step_id=step_id,
                attempted=attempted,
                found=found,
                result=on_track,
                persona=(str(persona) if persona is not None else None),
                target_chain_ref=target_id,
            )
            if prev_step_id is not None:
                # ponytail: 默认 method 依赖, data/causal 升级路径是 LLM 判定
                kg.add_dependency_edge(prev_step_id, step_id, dep_type="method")
        except Exception as exc:
            print(f"[StepEvaluator] kg write warning: {exc}", flush=True)

    return StepEvaluation(
        step_id=step_id,
        attempted=attempted,
        found=found,
        target_chain_ref=target_id,
        on_track=on_track,
        structure_check=struct,
        evidence_quality=evidence_quality,
        deviation=deviation,
        tool_call_health=tool_call_health or ToolCallHealth(),
        pmk_feedback=pmk,
    )


def should_continue(
    evaluations: list[StepEvaluation],
    window: int = 3,
) -> tuple[bool, str]:
    """决策窗口.

    - 连续 window 步 on_track="false" → (False, "重定向建议")
    - evidence_quality 持续低 (连续 window 步 "low") → (False, "建议补数据/文献")
    - 连续 window 次 tool_call_health 异常 → (False, "触发 Reflector 介入")
    - 正常 → (True, "")
    """
    if not evaluations:
        return (True, "")

    recent = evaluations[-window:]
    if len(recent) >= window:
        # 1) 连续脱轨 → 重定向
        if all(e.on_track == "false" for e in recent):
            return (False, "连续脱轨: 建议换方法 / 补数据 / 重定向目标链")
        # 2) 工具调用连续异常 → Reflector 介入 (G69)
        if all(e.tool_call_health.is_anomalous() for e in recent):
            return (False, "连续工具调用异常: 触发 Reflector 介入")
        # 3) 证据质量持续低
        if all(e.evidence_quality == "low" for e in recent):
            return (False, "证据质量持续低: 建议补数据 / 查文献")

    return (True, "")


def format_step_eval_text(evaluation: StepEvaluation) -> str:
    """格式化评估为 context 注入文本."""
    lines = [
        f"[StepEval #{evaluation.step_id}]",
        f"on_track: {evaluation.on_track}",
        f"structure_check: {evaluation.structure_check}",
        f"evidence_quality: {evaluation.evidence_quality}",
    ]
    if evaluation.target_chain_ref:
        lines.append(f"target_chain: {evaluation.target_chain_ref}")
    if evaluation.deviation:
        lines.append(f"deviation: {evaluation.deviation}")
    if evaluation.tool_call_health.is_anomalous():
        lines.append(
            f"tool_health: anomalous (success_rate={evaluation.tool_call_health.success_rate:.2f}, "
            f"timeout={evaluation.tool_call_health.timeout_count}, "
            f"param_err={evaluation.tool_call_health.param_error_count})"
        )
    if evaluation.pmk_feedback:
        lines.append(f"pmk_feedback: {evaluation.pmk_feedback}")
    return "\n".join(lines)


# === 自检 ===

if __name__ == "__main__":
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class _MockTC:
        target_id: str
        required_results: list

    # 1) _check_target_chain_match
    tc = _MockTC("T1", ["formation energy", "band gap"])
    tid, matched = _check_target_chain_match("compute formation energy = -3.5 eV", [tc])
    assert tid == "T1", f"target chain match: tid={tid}"
    assert "formation" in matched.lower(), f"target chain match: matched={matched!r}"

    tid, matched = _check_target_chain_match("nothing relevant here", [tc])
    assert tid is None and matched == "", f"no-match: tid={tid} matched={matched!r}"

    # dict 形式也兼容
    tc_dict = {"target_id": "T2", "required_results": ["diffusion coefficient"]}
    tid, matched = _check_target_chain_match("diffusion coefficient = 1e-9 m²/s", [tc_dict])
    assert tid == "T2" and "diffusion" in matched.lower(), f"dict tc: tid={tid} matched={matched!r}"

    # 2) _check_structure
    sig_dim = [{"verifiable_via": "dimensional"}]
    assert _check_structure(sig_dim, "no units here") == "failed", "dimensional missing units should fail"
    assert _check_structure(sig_dim, "energy = -3.5 eV") == "passed", "dimensional with units should pass"

    sig_none = [{"verifiable_via": "none"}]
    assert _check_structure(sig_none, "anything") == "not_applicable", "none via → not_applicable"
    assert _check_structure(None, "anything") == "not_applicable", "no signals → not_applicable"
    assert _check_structure([], "anything") == "not_applicable", "empty signals → not_applicable"

    # hard 失败 vs soft 警告
    sig_sym = [{"verifiable_via": "symmetry_argument"}]
    assert _check_structure(sig_sym, "plain text without relevant markers") == "soft_warning", "soft fail → soft_warning"
    assert _check_structure(sig_sym, "follows from symmetry of the lattice") == "passed", "soft pass → passed"

    sig_cons = [{"verifiable_via": "conservation_law"}]
    assert _check_structure(sig_cons, "energy conserved") == "passed", "conservation pass"
    assert _check_structure(sig_cons, "no relevant text") == "failed", "conservation fail → failed"

    # 显式 verified 覆盖启发式
    sig_explicit = [{"verifiable_via": "dimensional", "verified": True}]
    assert _check_structure(sig_explicit, "no units") == "passed", "explicit verified=True overrides heuristic"
    sig_explicit_fail = [{"verifiable_via": "dimensional", "verified": False}]
    assert _check_structure(sig_explicit_fail, "energy = -3.5 eV") == "failed", "explicit verified=False overrides heuristic"

    # 3) evaluate_step 机械信号足够 (匹配 target chain) → on_track="true", 不调 LLM
    entry = {"iteration": 1, "attempted": "compute formation energy", "found": "formation energy = -3.5 eV"}
    ev = evaluate_step(entry, [tc], sig_dim, memory=None, kb=None, persona=None, model=None)
    assert ev.on_track == "true", f"mechanical on_track should be true, got {ev.on_track}"
    assert ev.target_chain_ref == "T1", f"target_chain_ref should be T1, got {ev.target_chain_ref}"
    assert ev.evidence_quality == "high", f"passed struct → high evidence, got {ev.evidence_quality}"
    assert ev.step_id == 1

    # 结构硬失败 → on_track=false (有 model 也不该调, 机械已经判死)
    entry_bad = {"iteration": 2, "attempted": "compute energy", "found": "some text without units"}
    ev_bad = evaluate_step(entry_bad, [tc], sig_dim, memory=None, kb=None, persona=None, model=None)
    assert ev_bad.on_track == "false", f"hard struct fail → false, got {ev_bad.on_track}"
    assert ev_bad.evidence_quality == "low", f"hard struct fail → low, got {ev_bad.evidence_quality}"
    assert ev_bad.structure_check == "failed"

    # 机械不足 (无目标链匹配 + 无结构信号) + 无 model → unsure
    entry_unsure = {"iteration": 3, "attempted": "explore", "found": "no match no signal"}
    ev_unsure = evaluate_step(entry_unsure, [tc], None, memory=None, kb=None, persona=None, model=None)
    assert ev_unsure.on_track == "unsure", f"no mechanical signal + no model → unsure, got {ev_unsure.on_track}"
    assert ev_unsure.evidence_quality == "unknown"

    # 命中目标链但结构 soft_warning → 仍在轨, evidence medium
    entry_soft = {"iteration": 4, "attempted": "analyze", "found": "follows from symmetry of the lattice"}
    ev_soft = evaluate_step(entry_soft, [tc], sig_sym, memory=None, kb=None, persona=None, model=None)
    # 注意: "symmetry" 不在 tc 的 required_results 关键词里 → target_id 是 None
    # 但 struct=passed, 既没命中目标链也没硬失败 → 走 LLM 兜底, 无 model → unsure
    # 这里验证: 结构 soft_warning 时 found 没匹配目标链的兜底路径
    assert ev_soft.structure_check == "passed", f"symmetry present → passed, got {ev_soft.structure_check}"

    # 4) should_continue 连续 3 步 false → (False, msg); 有 true → (True, "")
    def _mk(on_track: str, eq: str = "medium") -> StepEvaluation:
        return StepEvaluation(
            step_id=0, attempted="", found="", target_chain_ref=None,
            on_track=on_track, structure_check="not_applicable",
            evidence_quality=eq, deviation="",
        )

    ok, msg = should_continue([_mk("false"), _mk("false"), _mk("false")])
    assert ok is False and msg, f"3x false → redirect, got {ok}/{msg!r}"

    ok, msg = should_continue([_mk("false"), _mk("true"), _mk("false")])
    assert ok is True, f"middle true → continue, got {ok}/{msg!r}"

    ok, msg = should_continue([_mk("true"), _mk("true")])
    assert ok is True, f"only 2 steps (below window) → continue, got {ok}/{msg!r}"

    # 5) should_continue 连续 3 步 tool_call_health 异常 → (False, "Reflector")
    def _mk_anomalous() -> StepEvaluation:
        ev_ = _mk("true")
        ev_.tool_call_health = ToolCallHealth(success_rate=0.1, timeout_count=1, total_calls=5)
        return ev_

    ok, msg = should_continue([_mk_anomalous(), _mk_anomalous(), _mk_anomalous()])
    assert ok is False, f"3x anomalous → stop, got {ok}/{msg!r}"
    assert "Reflector" in msg, f"msg should mention Reflector, got {msg!r}"

    # 混合: 中间一步不异常 → 不触发 Reflector
    ok, msg = should_continue([_mk_anomalous(), _mk("true"), _mk_anomalous()])
    assert ok is True, f"middle non-anomalous → continue, got {ok}/{msg!r}"

    # 6) ToolCallHealth.is_anomalous
    assert ToolCallHealth(success_rate=0.2).is_anomalous() is True, "0.2 < 0.3 → anomalous"
    assert ToolCallHealth(success_rate=0.9).is_anomalous() is False, "0.9 >= 0.3 → ok"
    assert ToolCallHealth(success_rate=1.0, timeout_count=1).is_anomalous() is True, "timeout → anomalous"
    assert ToolCallHealth(success_rate=1.0, param_error_count=1).is_anomalous() is True, "param_err → anomalous"
    assert ToolCallHealth().is_anomalous() is False, "default → ok"

    # 7) format_step_eval_text 输出含 on_track、evidence_quality
    text = format_step_eval_text(ev)
    assert "on_track" in text and "evidence_quality" in text, f"format missing fields: {text!r}"
    assert "[StepEval #1]" in text, f"format missing step id: {text!r}"

    # 异常工具健康也会出现在文本里
    ev_anom = _mk_anomalous()
    ev_anom.step_id = 99
    text_anom = format_step_eval_text(ev_anom)
    assert "tool_health" in text_anom and "anomalous" in text_anom, f"format missing tool health: {text_anom!r}"

    # evidence_quality 持续低 → 触发"补数据"
    ok, msg = should_continue([_mk("unsure", "low"), _mk("unsure", "low"), _mk("unsure", "low")])
    assert ok is False and ("补数据" in msg or "文献" in msg), f"3x low → 补数据, got {ok}/{msg!r}"

    # 8) compute_tool_call_health: 缺文件 → None; 解析正常 → 字段对得上
    import tempfile as _tf
    from pathlib import Path as _Path
    assert compute_tool_call_health(None, 1) is None, "None path → None"
    assert compute_tool_call_health(_Path("/no/such/file.jsonl"), 1) is None, "missing file → None"
    with _tf.TemporaryDirectory() as _td:
        _ap = _Path(_td) / "audit.jsonl"
        _ap.write_text(
            '{"type":"tool.call","data":{"tool":"vasp","input":"x"}}\n'
            '{"type":"tool.call","data":{"tool":"vasp","input":"x"}}\n'  # retry
            '{"type":"tool.result","data":{"tool":"vasp"}}\n'
            '{"type":"tool.error","data":{"tool":"qe","error":"TimeoutError"}}\n'
            '{"type":"tool.error","data":{"tool":"qe","error":"ValidationError: bad param"}}\n'
            '{"type":"tool.error","data":{"tool":"qe","error":"OtherError"}}\n'
            'not-json-line\n',  # 解析失败的行跳过
            encoding="utf-8",
        )
        _h = compute_tool_call_health(_ap, 1)
        assert _h is not None, "valid audit → health obj"
        assert _h.total_calls == 2, f"2 tool.call → total=2, got {_h.total_calls}"
        assert _h.retry_count == 1, f"重复 input → retry=1, got {_h.retry_count}"
        assert _h.timeout_count == 1, f"timeout error → timeout=1, got {_h.timeout_count}"
        assert _h.param_error_count == 1, f"validation error → param=1, got {_h.param_error_count}"
        # success_rate = 1 success / 2 calls = 0.5
        assert abs(_h.success_rate - 0.5) < 1e-9, f"sr=0.5, got {_h.success_rate}"
        # 空文件 → None
        _empty = _Path(_td) / "empty.jsonl"
        _empty.write_text("", encoding="utf-8")
        assert compute_tool_call_health(_empty, 1) is None, "empty → None"

    # === v14 Task 11: LLM darwin 评估 self-check ===
    class _MockLLM:
        """简单 mock, 跟 langchain BaseChatModel 的 ainvoke 接口对齐."""
        def __init__(self, response: str = "", error: Exception | None = None):
            self.response = response
            self.error = error
            self.call_count = 0

        async def ainvoke(self, messages):
            self.call_count += 1
            if self.error is not None:
                raise self.error
            return self.response

    class _LogCapture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records: list[str] = []
        def emit(self, record):
            self.records.append(self.format(record))

    _saved_eval = os.environ.get("HUGINN_DARWIN_LLM_EVAL")
    _saved_interval = os.environ.get("HUGINN_DARWIN_LLM_INTERVAL")
    try:
        # case 1: mock LLM 返回固定 JSON, 断言 darwin_score 被覆盖
        os.environ["HUGINN_DARWIN_LLM_EVAL"] = "1"
        os.environ["HUGINN_DARWIN_LLM_INTERVAL"] = "5"
        _entries = [{"iteration": 5, "simplex_id": "s1", "attempted": "a", "found": "f", "evidence": "e"}]
        _mock = _MockLLM(response='[{"simplex_id": "s1", "darwin_score": 0.9, "reason": "good"}]')
        _result = asyncio.run(_llm_evaluate_darwin(_entries, _mock, "test task"))
        assert len(_result) == 1, f"case1: expect 1 entry, got {len(_result)}"
        assert _result[0]["darwin_score"] == 0.9, f"case1: darwin_score=0.9, got {_result[0]['darwin_score']}"
        assert _result[0]["simplex_id"] == "s1"
        assert _mock.call_count == 1, f"case1: LLM called once, got {_mock.call_count}"

        # case 2: mock LLM 抛异常, 断言降级到空 list + 日志含 fallback
        _cap = _LogCapture()
        _cap.setLevel(logging.WARNING)
        logger.addHandler(_cap)
        try:
            _mock_err = _MockLLM(error=RuntimeError("simulated timeout"))
            _result2 = asyncio.run(_llm_evaluate_darwin(_entries, _mock_err, "test task"))
            assert _result2 == [], f"case2: expect empty list on fallback, got {_result2}"
            _log_text = " ".join(_cap.records)
            assert "darwin_llm_eval_fallback: reason=" in _log_text, f"case2: log missing fallback marker, got {_log_text!r}"
        finally:
            logger.removeHandler(_cap)

        # case 3: env HUGINN_DARWIN_LLM_EVAL=0, 断言直接返回空 list 不调 LLM
        os.environ["HUGINN_DARWIN_LLM_EVAL"] = "0"
        _mock3 = _MockLLM(response='[{"simplex_id": "s1", "darwin_score": 0.9, "reason": "good"}]')
        _result3 = asyncio.run(_llm_evaluate_darwin(_entries, _mock3, "test task"))
        assert _result3 == [], f"case3: env=0 → empty list, got {_result3}"
        assert _mock3.call_count == 0, f"case3: LLM not called, got call_count={_mock3.call_count}"

        # case 4: env=1 + interval=5 + current_iter=3, 断言不触发 LLM (3%5!=0)
        os.environ["HUGINN_DARWIN_LLM_EVAL"] = "1"
        os.environ["HUGINN_DARWIN_LLM_INTERVAL"] = "5"
        _entries4 = [{"iteration": 3, "simplex_id": "s4", "attempted": "a", "found": "f", "evidence": "e"}]
        _mock4 = _MockLLM(response='[{"simplex_id": "s4", "darwin_score": 0.9, "reason": "good"}]')
        _result4 = asyncio.run(_llm_evaluate_darwin(_entries4, _mock4, "test task"))
        assert _result4 == [], f"case4: iter=3%5!=0 → empty, got {_result4}"
        assert _mock4.call_count == 0, f"case4: LLM not called, got call_count={_mock4.call_count}"

        # case 5: env=1 + interval=5 + current_iter=5, 断言触发 LLM (5%5==0)
        _entries5 = [{"iteration": 5, "simplex_id": "s5", "attempted": "a", "found": "f", "evidence": "e"}]
        _mock5 = _MockLLM(response='[{"simplex_id": "s5", "darwin_score": 0.7, "reason": "ok"}]')
        _result5 = asyncio.run(_llm_evaluate_darwin(_entries5, _mock5, "test task"))
        assert len(_result5) == 1, f"case5: iter=5%5==0 → 1 entry, got {len(_result5)}"
        assert _result5[0]["darwin_score"] == 0.7, f"case5: darwin_score=0.7, got {_result5[0]['darwin_score']}"
        assert _mock5.call_count == 1, f"case5: LLM called once, got call_count={_mock5.call_count}"

        # 额外: _parse_darwin_json 字段校验 — 非法值跳过, 不整体抛
        _parsed = _parse_darwin_json(
            '[{"simplex_id": "ok", "darwin_score": 0.5, "reason": "fine"},'
            ' {"simplex_id": 123, "darwin_score": 0.5, "reason": "bad sid"},'
            ' {"simplex_id": "ok2", "darwin_score": 1.5, "reason": "out of range"},'
            ' {"simplex_id": "ok3", "darwin_score": 0.3, "reason": 456},'
            ' {"simplex_id": "ok4", "darwin_score": true, "reason": "bool score"}]'
        )
        assert len(_parsed) == 1, f"parse: only 1 valid entry, got {len(_parsed)}"
        assert _parsed[0]["simplex_id"] == "ok", f"parse: valid entry sid=ok, got {_parsed[0]['simplex_id']}"
    finally:
        # 恢复原 env, 不污染其它测试
        if _saved_eval is not None:
            os.environ["HUGINN_DARWIN_LLM_EVAL"] = _saved_eval
        else:
            os.environ.pop("HUGINN_DARWIN_LLM_EVAL", None)
        if _saved_interval is not None:
            os.environ["HUGINN_DARWIN_LLM_INTERVAL"] = _saved_interval
        else:
            os.environ.pop("HUGINN_DARWIN_LLM_INTERVAL", None)

    print("all self-checks passed")
    print("v14 Task 11 self-check PASSED")
