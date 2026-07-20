"""PlanCheck — 反向规划校验函数模块.

从 engine.py 抽出的 16 个方法. KRCL 闭环: 反向校验 plan, 失败反馈 LLM 重生成,
超限不阻塞. phase-aware tier + 自适应 max_refines + 失败模式跨 run 持久化.

通过 engine 实例访问状态字段 (_plan_check_patterns/_plan_check_history/
_plan_check_warnings/_plan_check_last_result/_scene_tag_extra_keywords/
_iteration/workspace/_speculator_hint) 和 engine 方法 (_maybe_clarify/
_llm_chat/_parse_plan/_override_plan_mode).

调用点: engine._prepare_run 调 load_plan_check_patterns();
engine._plan 调 plan_check_and_refine().
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


async def plan_check_and_refine(
    engine: Any,
    plan: dict[str, Any],
    hypothesis: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """KRCL 闭环: 反向校验 plan, 失败反馈 LLM 重生成, 超限不阻塞.

    phase-aware: iteration tier (open/medium/light) + plan 复杂度综合判定.
    open 或 skip 跳过校验, medium 只校验不 refine, light 完整闭环.
    自适应: 按 scene_tag 分桶的最近 5 次 success rate 微调 max_refines.
    失败模式记忆: 失败记到 _plan_check_patterns, 跨 run JSON 持久化.
    连续失败澄清: 同 scene 连续 3 次失败 + scene != "other" -> 问用户.
    失败不拦截 (physical_precheck 同款), warning 留痕给 _validate.
    """
    desc = plan.get("description", "")
    if len(desc) < 20:
        return plan
    tier = plan_check_tier(engine, plan)
    if tier in ("open", "skip"):
        logger.debug("plan_check skipped (tier=%s, iter=%d)", tier, getattr(engine, "_iteration", 0))
        return plan
    scene = plan_check_scene_tag(engine, plan)
    max_refines = plan_check_max_refines(engine, tier, scene)
    for attempt in range(max_refines + 1):
        try:
            check = await plan_check(engine, plan, hypothesis, context)
        except Exception as e:
            logger.debug("plan_check LLM call failed: %s", e)
            return plan
        check["scene_tag"] = scene
        if check.get("is_valid", True):
            check["plan_snapshot"] = {
                "mode": plan.get("mode", ""),
                "description": plan.get("description", "")[:200],
            }
        engine._plan_check_last_result = check
        engine._plan_check_history.append(check)
        if len(engine._plan_check_history) > 20:
            del engine._plan_check_history[: len(engine._plan_check_history) - 20]
        if check.get("is_valid", True):
            confidence = float(check.get("confidence", 0.8))
            if confidence >= 0.5 or attempt >= max_refines or max_refines == 0:
                logger.info(
                    "plan_check passed (attempt %d, tier=%s, scene=%s, conf=%.2f)",
                    attempt, tier, scene, confidence,
                )
                if len(engine._plan_check_history) % 5 == 0:
                    discover_scene_tags(engine)
                return plan
            logger.info("plan_check passed but low confidence (conf=%.2f), refining", confidence)
        else:
            record_plan_check_failure(engine, plan, check, scene)
            confidence = float(check.get("confidence", 0.8))
            if confidence < 0.3:
                reason = check.get("reason", "unknown")
                engine._plan_check_warnings.append(f"[{scene}] {reason} (low_conf={confidence:.2f})")
                logger.warning(
                    "plan_check failed low-conf (tier=%s, scene=%s, conf=%.2f): %s",
                    tier, scene, confidence, reason,
                )
                await maybe_trigger_plan_check_clarify(engine, scene, reason, plan)
                return plan
            if attempt >= max_refines:
                reason = check.get("reason", "unknown")
                engine._plan_check_warnings.append(f"[{scene}] {reason}")
                logger.warning(
                    "plan_check failed (tier=%s, scene=%s, max_refines=%d): %s",
                    tier, scene, max_refines, reason,
                )
                await maybe_trigger_plan_check_clarify(engine, scene, reason, plan)
                return plan
        logger.info(
            "plan_check refining (attempt %d, tier=%s, scene=%s, conf=%.2f): %s",
            attempt, tier, scene, float(check.get("confidence", 0.8)), check.get("reason"),
        )
        plan = await refine_plan(engine, plan, check, hypothesis, context)
    return plan


async def maybe_trigger_plan_check_clarify(
    engine: Any, scene: str, reason: str, plan: dict[str, Any]
) -> None:
    """连续 N 次同场景失败 + 场景已知 -> 问用户方向.

    ponytail: 阈值 3 写死, 跟 validation_fail 同款; 不阻塞, 异常吞掉.
    """
    if scene == "other":
        return
    recent_fails = 0
    for c in reversed(engine._plan_check_history):
        if c.get("scene_tag") == scene and not c.get("is_valid", True):
            recent_fails += 1
        else:
            break
    if recent_fails < 3:
        return
    try:
        await engine._maybe_clarify(
            "plan_check_fail",
            {"scene": scene, "reason": reason, "consecutive_fails": recent_fails, "plan": plan},
        )
    except Exception as e:
        logger.debug("plan_check clarify failed: %s", e)


def plan_check_tier(engine: Any, plan: dict[str, Any] | None = None) -> str:
    """phase-aware tier: iteration + plan 复杂度综合判定.

    iteration baseline: open (1-10) / medium (11-30) / light (31+).
    plan 复杂度修正: 复杂 plan 升级到 medium, 简单 plan 降级到 skip.
    阈值分场景校准: DFT/MD/workflow 各有自己的 success rate.
    """
    n = getattr(engine, "_iteration", 0)
    if n <= 10:
        base = "open"
    elif n <= 30:
        base = "medium"
    else:
        base = "light"
    if plan is None:
        return base
    complexity = plan_check_complexity(engine, plan)
    scene = plan_check_scene_tag(engine, plan)
    upgrade_t, downgrade_t = plan_check_complexity_thresholds(engine, scene)
    if complexity >= upgrade_t and base == "open":
        return "medium"
    if complexity < downgrade_t and base == "light":
        return "skip"
    return base


def plan_check_complexity_thresholds(engine: Any, scene: str = "") -> tuple[float, float]:
    """用历史 success rate 自动校准复杂度阈值, 分场景.

    默认: upgrade=0.7, downgrade=0.25.
    >=0.8 (一直成功) -> (0.8, 0.15); <=0.2 (一直失败) -> (0.6, 0.35).
    样本 <5 走默认, 早期不误判.
    """
    history = getattr(engine, "_plan_check_history", [])
    if scene:
        bucket = [c for c in history if c.get("scene_tag") == scene]
    else:
        bucket = history
    if len(bucket) < 5:
        if scene and len(history) >= 5:
            bucket = history
        else:
            return (0.7, 0.25)
    recent = bucket[-10:]
    success_rate = sum(1 for c in recent if c.get("is_valid", True)) / len(recent)
    if success_rate >= 0.8:
        return (0.8, 0.15)
    if success_rate <= 0.2:
        return (0.6, 0.35)
    return (0.7, 0.25)


def plan_check_scene_tag(engine: Any, plan: dict[str, Any]) -> str:
    """从 plan 抽场景标签. 写死关键词表 + 自动发现关键词互补.

    ponytail: 关键词匹配, 不上 embedding.
    """
    desc = (plan.get("description", "") + " " + plan.get("mode", "")).lower()
    if any(kw in desc for kw in ["vasp", "scf", "band", "dos", "dft", "qe", "cp2k", "gaussian", "orca"]):
        return "dft"
    if any(kw in desc for kw in ["lammps", "molecular dynamics", "minimize", "nvt", "npt", "md ", "gromacs", "openmm"]):
        return "md"
    if any(kw in desc for kw in ["workflow", "pipeline", "orchestrat"]):
        return "workflow"
    if plan.get("mode") == "skill":
        return "skill"
    if any(kw in desc for kw in ["fenics", "abaqus", "comsol", "openfoam", "fem", "elmer"]):
        return "fem"
    for label, keywords in getattr(engine, "_scene_tag_extra_keywords", {}).items():
        if any(kw in desc for kw in keywords):
            return label
    return "other"


def discover_scene_tags(engine: Any) -> None:
    """从 _plan_check_history 里 scene='other' 的 plans 做关键词统计,
    发现高频词 (>=3 次) 自动加到 _scene_tag_extra_keywords.

    双重识别: unigram (>=4 chars) + bigram (两词短语).
    ponytail: 简单词频统计, 不上 TF-IDF/embedding.
    """
    other_descs: list[str] = []
    for c in getattr(engine, "_plan_check_history", []):
        snapshot = c.get("plan_snapshot") or {}
        if c.get("scene_tag") == "other" and snapshot.get("description"):
            other_descs.append(snapshot["description"].lower())
    if len(other_descs) < 3:
        return
    stop = {"the", "and", "for", "with", "that", "this", "from", "run", "then",
            "calc", "calculate", "using", "use", "plan", "step"}
    word_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    for desc in other_descs:
        words = [w for w in re.findall(r"[a-z][a-z0-9_]{3,}", desc) if w not in stop]
        for word in words:
            word_counts[word] += 1
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigram_counts[bigram] += 1
    for word, count in word_counts.most_common(10):
        if count >= 3:
            label = f"auto_{word}"
            engine._scene_tag_extra_keywords.setdefault(label, set()).add(word)
    for bigram, count in bigram_counts.most_common(5):
        if count >= 3:
            label = f"auto_{bigram.replace(' ', '_')}"
            engine._scene_tag_extra_keywords.setdefault(label, set()).add(bigram)


def plan_check_complexity(engine: Any, plan: dict[str, Any]) -> float:
    """plan 复杂度评分 [0, 1].

    维度: description 长度 (0.3) + mode 复杂度 (0.4) + prediction (0.15) +
    同场景历史失败数 (0.15, 踩过坑的要复查).
    ponytail: 启发式打分, 不上结构化解析.
    """
    score = 0.0
    desc = plan.get("description", "")
    score += min(len(desc), 50) / 50 * 0.3
    mode = plan.get("mode", "coder")
    score += {"workflow": 0.4, "skill": 0.3, "coder": 0.2, "explore": 0.1}.get(mode, 0.2)
    if plan.get("expected_prediction"):
        score += 0.15
    scene = plan_check_scene_tag(engine, plan)
    similar_fails = sum(
        1 for p in getattr(engine, "_plan_check_patterns", []) if p.get("scene_tag") == scene
    )
    score += min(similar_fails, 3) / 3 * 0.15
    return min(score, 1.0)


def plan_check_max_refines(engine: Any, tier: str, scene: str = "") -> int:
    """自适应: 按场景分桶的 EWMA success rate 微调 max_refines.

    baseline: medium=0, light=1. >=80% 放宽 (-1), <=20% 收紧 (+1).
    alpha 自适应: 桶 3-4 条用 0.3, 桶 5 条用 0.4. 样本 <3 走 baseline.
    ponytail: EWMA 简单指数加权; alpha 分两档.
    """
    baseline = {"medium": 0, "light": 1}.get(tier, 1)
    history = getattr(engine, "_plan_check_history", [])
    bucket = [c for c in history if c.get("scene_tag") == scene] if scene else history
    if len(bucket) < 3:
        return baseline
    recent = bucket[-5:]
    alpha = 0.3 if len(recent) < 5 else 0.4
    weights = [alpha * (1 - alpha) ** (len(recent) - 1 - i) for i in range(len(recent))]
    total_w = sum(weights)
    if total_w == 0:
        return baseline
    ewma_success = (
        sum(w * (1.0 if c.get("is_valid", True) else 0.0) for w, c in zip(weights, recent))
        / total_w
    )
    if ewma_success >= 0.8:
        return max(0, baseline - 1)
    if ewma_success <= 0.2:
        return min(2, baseline + 1)
    return baseline


async def plan_check(
    engine: Any,
    plan: dict[str, Any],
    hypothesis: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """单次反向校验: 让 LLM 判断 plan 执行后能否达成 hypothesis.

    用 task='verification' 让 model_router 路由到独立验证模型.
    v6 G57: L1 LLM plan_check + L2 dimensional pre-check + L3 physical_precheck.
    """
    dim_warnings = dimensional_pre_check(plan, hypothesis)
    if dim_warnings:
        context = dict(context)
        context["dimensional_warnings"] = "\n".join(dim_warnings)

    prompt = build_plan_check_prompt(engine, plan, hypothesis, context)
    response = await engine._llm_chat(prompt, persona_name="default", task="verification")
    result = parse_plan_check(response)
    if dim_warnings:
        result["dimensional_warnings"] = dim_warnings
        existing = result.get("risks") or []
        existing.extend(dim_warnings)
        result["risks"] = existing
        engine._plan_check_warnings.extend(dim_warnings)
    return result


def dimensional_pre_check(plan: dict[str, Any], hypothesis: str) -> list[str]:
    """L2 dimensional pre-check — 扫 plan + hypothesis 里的等式, 验量纲.

    纯函数 (无 engine 依赖). ponytail: regex 抓 "<number> <unit>" 量 + "=" 等式.
    只在能解析出两侧都带量纲的等式时跑; 否则跳过 (不误报).
    """
    warnings: list[str] = []
    try:
        from huginn.validation.dimensional import DimensionalValidator
    except Exception:
        return warnings

    text_parts = [hypothesis or ""]
    for k in ("description", "expected_prediction", "prediction"):
        v = plan.get(k) if isinstance(plan, dict) else None
        if isinstance(v, str) and v:
            text_parts.append(v)
    text = "\n".join(text_parts)
    if "=" not in text:
        return warnings

    validator = DimensionalValidator()
    qty_re = re.compile(
        r"([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+([A-Za-z][A-Za-z0-9/\^\-\*\.\(\)]+)"
    )
    for line in text.splitlines():
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        lhs_qs = [f"{m[0]} {m[1]}" for m in qty_re.findall(lhs)]
        rhs_qs = [f"{m[0]} {m[1]}" for m in qty_re.findall(rhs)]
        if not lhs_qs or not rhs_qs:
            continue
        try:
            result = validator.check_equation(lhs_qs, rhs_qs, equation_name=line.strip()[:80])
            if not result.consistent:
                warnings.append(
                    f"dimensional inconsistency: '{line.strip()[:80]}' "
                    f"LHS={result.lhs_dimensions} RHS={result.rhs_dimensions}"
                )
        except Exception:
            continue
    return warnings


def build_plan_check_prompt(
    engine: Any,
    plan: dict[str, Any],
    hypothesis: str,
    context: dict[str, Any],
) -> str:
    """反向规划识别器 prompt: 判断 plan 能否达成 hypothesis."""
    failure_modes = context.get("failure_modes", "")
    if not failure_modes and getattr(engine, "_speculator_hint", ""):
        failure_modes = engine._speculator_hint[-500:]
    scene = plan_check_scene_tag(engine, plan)
    similar = [
        p for p in getattr(engine, "_plan_check_patterns", []) if p.get("scene_tag") == scene
    ][-3:]
    if similar:
        similar_text = "\n".join(
            f"- {p['reason']} (缺: {', '.join(p.get('missing_steps', [])) or 'N/A'})"
            for p in similar
        )
    else:
        similar_text = "N/A"
    dim_warnings_text = context.get("dimensional_warnings", "") or "N/A"
    return f"""你是反向规划识别器 (KRCL 启发). 判断以下 plan 执行后能否达成 hypothesis.

# 目标 (hypothesis)
{hypothesis}

# 当前 plan
MODE: {plan.get('mode', 'coder')}
DESCRIPTION: {plan.get('description', '')}
PREDICTION: {plan.get('expected_prediction', 'N/A')}

# 已知失败模式 (避免重蹈覆辙)
{failure_modes or 'N/A'}

# 同场景历史失败 (scene={scene}, 跨 run 积累)
{similar_text}

# 量纲预检查警告 (L2 dimensional pre-check, v6 G57)
{dim_warnings_text}

# 任务
判断这个 plan 执行后能否达成 hypothesis. 严格检查:
- MODE 是否匹配任务类型 (coder 写代码 / workflow 跑流程 / explore 探索 / skill 复合技能)
- DESCRIPTION 是否覆盖 hypothesis 的关键要求
- PREDICTION 是否可验证 (能跑出数值/结构/代码对比)
- 是否遗漏必要前置步骤 (如 band 前需 SCF / MD 前需 minimize / elastic 前需 relax)
- 是否重复了"同场景历史失败"里列出的坑
- 量纲预检查有警告时, 把它列入 risks

输出 JSON (不要其他文本):
{{
  "is_valid": true 或 false,
  "confidence": 0.0 到 1.0 (对判断的置信度, 1.0=非常确定, 0.5=模棱两可, 0.0=完全没把握),
  "reason": "为什么 valid / invalid",
  "missing_steps": ["如果 invalid, 缺少哪些步骤"],
  "risks": ["潜在风险"]
}}"""


def record_plan_check_failure(
    engine: Any, plan: dict[str, Any], check: dict[str, Any], scene: str
) -> None:
    """失败模式记到 patterns, 跨 run 持久化给下次注入 prompt.

    ponytail: 内存 append + 同步 dump JSON, 量小 (<=50 条) 写快.
    """
    engine._plan_check_patterns.append({
        "scene_tag": scene,
        "reason": check.get("reason", "unknown"),
        "missing_steps": check.get("missing_steps", []),
        "mode": plan.get("mode", ""),
        "description": plan.get("description", "")[:200],
    })
    if len(engine._plan_check_patterns) > 50:
        del engine._plan_check_patterns[: len(engine._plan_check_patterns) - 50]
    save_plan_check_patterns(engine)


def load_plan_check_patterns(engine: Any) -> None:
    """跨 run 加载历史失败模式.

    ponytail: JSON 文件, 不上 DB; 只在 _prepare_run 调一次.
    """
    path = engine.workspace / ".huginn" / "plan_check_patterns.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            engine._plan_check_patterns = data[-50:]
            logger.info("loaded %d plan_check patterns from %s", len(engine._plan_check_patterns), path)
    except Exception as e:
        logger.debug("load plan_check_patterns failed: %s", e)


def save_plan_check_patterns(engine: Any) -> None:
    """dump 失败模式到 workspace, 跨 run 积累.

    ponytail: 同步写, 量小 (<=50 条).
    """
    path = engine.workspace / ".huginn" / "plan_check_patterns.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(engine._plan_check_patterns[-50:], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug("save plan_check_patterns failed: %s", e)


def parse_plan_check(response: str) -> dict[str, Any]:
    """解析反向校验 JSON — 括号配平法.

    纯函数 (无 engine 依赖). 解析失败返回 is_valid=True (跳过校验, 不阻塞).
    """
    start = response.find("{")
    if start < 0:
        return {"is_valid": True, "reason": "no json, skip"}
    depth = 0
    for i, ch in enumerate(response[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(response[start : i + 1])
                    obj.setdefault("is_valid", True)
                    obj.setdefault("confidence", 0.8)
                    obj.setdefault("reason", "")
                    obj.setdefault("missing_steps", [])
                    obj.setdefault("risks", [])
                    return obj
                except json.JSONDecodeError:
                    return {"is_valid": True, "reason": "json parse failed, skip"}
    return {"is_valid": True, "reason": "no closing brace, skip"}


async def refine_plan(
    engine: Any,
    plan: dict[str, Any],
    check: dict[str, Any],
    hypothesis: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """根据反向校验反馈, 让 LLM 重新生成 plan (保留 plan_id).

    few-shot: 从 _plan_check_history 抽同场景最近 1 条成功 plan 塞进 prompt.
    """
    scene = plan_check_scene_tag(engine, plan)
    success_example = None
    for c in reversed(getattr(engine, "_plan_check_history", [])):
        if c.get("is_valid") and c.get("scene_tag") == scene and c.get("plan_snapshot"):
            success_example = c["plan_snapshot"]
            break
    few_shot_block = "N/A"
    if success_example:
        few_shot_block = (
            f"MODE: {success_example.get('mode', 'coder')}\n"
            f"DESCRIPTION: {success_example.get('description', '')[:200]}"
        )
    prompt = f"""之前的 plan 未通过反向校验. 根据反馈重新生成.

# 目标
{hypothesis}

# 之前的 plan
MODE: {plan.get('mode', 'coder')}
DESCRIPTION: {plan.get('description', '')}

# 校验反馈
reason: {check.get('reason', '')}
missing_steps: {check.get('missing_steps', [])}
risks: {check.get('risks', [])}

# 同场景成功示例 (scene={scene}, 跨 iteration 积累, 仅供参考结构)
{few_shot_block}

# 任务
根据反馈重新生成 plan. 参考成功示例的结构 (不要照抄内容). 严格按格式输出:
MODE: <coder|workflow|explore|skill|visual_inspect>
DESCRIPTION: <brief description>
SKILL: <composite skill name, only if MODE is skill>
PREDICTION: <预期结果, 用于后续 validate 对比>"""
    try:
        response = await engine._llm_chat(prompt, persona_name="default", task="planning")
        new_plan = engine._parse_plan(response)
        new_plan = engine._override_plan_mode(new_plan)
        if "plan_id" in plan:
            new_plan["plan_id"] = plan["plan_id"]
        return new_plan
    except Exception as e:
        logger.debug("plan refine failed: %s", e)
        return plan


# === 自检 ===

if __name__ == "__main__":
    import asyncio
    from types import SimpleNamespace

    # 1) parse_plan_check: 正常 JSON
    r = parse_plan_check('{"is_valid": false, "confidence": 0.9, "reason": "missing SCF"}')
    assert r["is_valid"] is False
    assert r["confidence"] == 0.9
    assert r["reason"] == "missing SCF"

    # 2) parse_plan_check: 字段补全
    r = parse_plan_check('{"is_valid": true}')
    assert r["confidence"] == 0.8
    assert r["missing_steps"] == []

    # 3) parse_plan_check: 无 JSON → is_valid=True (跳过)
    r = parse_plan_check("no json here")
    assert r["is_valid"] is True

    # 4) parse_plan_check: 嵌套 JSON
    r = parse_plan_check('{"is_valid": true, "risks": ["a", "b", {"c": 1}]}')
    assert r["is_valid"] is True
    assert len(r["risks"]) == 3

    # 5) dimensional_pre_check: 无 "=" → 空 list
    w = dimensional_pre_check({"description": "no equation"}, "hypothesis text")
    assert w == []

    # 6) dimensional_pre_check: 有 "=" 但无量纲 → 空 list
    w = dimensional_pre_check({"description": "x = y"}, "")
    assert w == []

    # 7) plan_check_scene_tag: DFT 关键词
    eng = SimpleNamespace(_scene_tag_extra_keywords={})
    assert plan_check_scene_tag(eng, {"description": "run vasp scf", "mode": "workflow"}) == "dft"
    assert plan_check_scene_tag(eng, {"description": "lammps nvt", "mode": "workflow"}) == "md"
    assert plan_check_scene_tag(eng, {"description": "defect detection", "mode": "coder"}) == "other"

    # 8) plan_check_scene_tag: 自动发现关键词 (不加 mode=workflow, 否则先命中写死的 workflow 规则)
    eng2 = SimpleNamespace(_scene_tag_extra_keywords={"auto_neb": {"neb chain"}})
    assert plan_check_scene_tag(eng2, {"description": "neb chain calculation", "mode": "coder"}) == "auto_neb"

    # 9) plan_check_tier: iteration baseline
    eng3 = SimpleNamespace(_iteration=5, _plan_check_history=[], _plan_check_patterns=[], _scene_tag_extra_keywords={})
    assert plan_check_tier(eng3) == "open"
    eng3._iteration = 20
    assert plan_check_tier(eng3) == "medium"
    eng3._iteration = 50
    assert plan_check_tier(eng3) == "light"

    # 10) plan_check_tier: plan 复杂度修正 (open + 复杂 → medium)
    eng4 = SimpleNamespace(_iteration=5, _plan_check_history=[], _plan_check_patterns=[], _scene_tag_extra_keywords={})
    # 长描述 + workflow → 高复杂度 → 升级
    complex_plan = {"description": "x" * 60, "mode": "workflow", "expected_prediction": "yes"}
    assert plan_check_tier(eng4, complex_plan) == "medium"

    # 11) plan_check_complexity_thresholds: 默认值
    eng5 = SimpleNamespace(_plan_check_history=[])
    assert plan_check_complexity_thresholds(eng5, "dft") == (0.7, 0.25)

    # 12) plan_check_complexity_thresholds: 高 success rate
    eng6 = SimpleNamespace(_plan_check_history=[
        {"scene_tag": "dft", "is_valid": True}] * 8)
    assert plan_check_complexity_thresholds(eng6, "dft") == (0.8, 0.15)

    # 13) plan_check_complexity_thresholds: 低 success rate
    eng7 = SimpleNamespace(_plan_check_history=[
        {"scene_tag": "dft", "is_valid": False}] * 8)
    assert plan_check_complexity_thresholds(eng7, "dft") == (0.6, 0.35)

    # 14) plan_check_max_refines: 样本不足 → baseline
    eng8 = SimpleNamespace(_plan_check_history=[])
    assert plan_check_max_refines(eng8, "medium", "dft") == 0
    assert plan_check_max_refines(eng8, "light", "dft") == 1

    # 15) plan_check_max_refines: 高 success → 放宽
    eng9 = SimpleNamespace(_plan_check_history=[
        {"scene_tag": "dft", "is_valid": True}] * 5)
    assert plan_check_max_refines(eng9, "light", "dft") == 0  # baseline 1 - 1

    # 16) plan_check_max_refines: 低 success → 收紧
    eng10 = SimpleNamespace(_plan_check_history=[
        {"scene_tag": "dft", "is_valid": False}] * 5)
    assert plan_check_max_refines(eng10, "medium", "dft") == 1  # baseline 0 + 1

    # 17) record_plan_check_failure + save/load (用 tmpdir)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        eng11 = SimpleNamespace(
            workspace=Path(td),
            _plan_check_patterns=[],
        )
        record_plan_check_failure(
            eng11, {"mode": "coder", "description": "test"}, {"reason": "bad"}, "dft"
        )
        assert len(eng11._plan_check_patterns) == 1
        assert eng11._plan_check_patterns[0]["scene_tag"] == "dft"
        # 文件应已写入
        assert (Path(td) / ".huginn" / "plan_check_patterns.json").exists()
        # load 回来
        eng12 = SimpleNamespace(workspace=Path(td), _plan_check_patterns=[])
        load_plan_check_patterns(eng12)
        assert len(eng12._plan_check_patterns) == 1

    # 18) discover_scene_tags: 高频词发现
    eng13 = SimpleNamespace(
        _plan_check_history=[
            {"scene_tag": "other", "plan_snapshot": {"description": "phase diagram calc"}, "is_valid": False},
            {"scene_tag": "other", "plan_snapshot": {"description": "phase diagram run"}, "is_valid": False},
            {"scene_tag": "other", "plan_snapshot": {"description": "phase diagram plot"}, "is_valid": False},
        ],
        _scene_tag_extra_keywords={},
    )
    discover_scene_tags(eng13)
    # "phase" 和 "diagram" 都应被识别, "phase diagram" bigram 也应被识别
    labels = list(eng13._scene_tag_extra_keywords.keys())
    assert any("phase" in l for l in labels)
    assert any("diagram" in l for l in labels)

    # 19) plan_check_and_refine: trivial plan (描述 <20 chars) 直接返回
    async def _test_trivial():
        eng = SimpleNamespace(_iteration=10, _plan_check_history=[], _plan_check_patterns=[],
                              _scene_tag_extra_keywords={}, _plan_check_warnings=[],
                              _plan_check_last_result=None)
        r = await plan_check_and_refine(eng, {"description": "short"}, "h", {})
        assert r == {"description": "short"}
    asyncio.run(_test_trivial())

    # 20) plan_check_and_refine: open tier → 跳过
    async def _test_open_tier():
        eng = SimpleNamespace(_iteration=5, _plan_check_history=[], _plan_check_patterns=[],
                              _scene_tag_extra_keywords={}, _plan_check_warnings=[],
                              _plan_check_last_result=None)
        r = await plan_check_and_refine(eng, {"description": "long enough description"}, "h", {})
        assert r == {"description": "long enough description"}
    asyncio.run(_test_open_tier())

    print("all self-checks passed")
