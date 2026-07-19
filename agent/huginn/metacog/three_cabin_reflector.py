"""ThreeCabinReflector — 单步三舱反射控制器.

autoloop reflect_fn 现在用 SimpleNamespace 包 _validation_to_step_eval_fields,
字段残缺 (tool_call_health=None / target_chain_ref=None / pmk_feedback="" /
structure_check 二元 / 无 LLM 兜底), pmk_cycle_count/tool_call_health_avg 不增.

三舱封装, 复用现有积木:
- 舱 1 Step: cog + validation → meta_trace_entry
- 舱 2 Evaluate: step_evaluator.evaluate_step → 真 StepEvaluation
- 舱 3 Reflect: should_continue 检 Reflector 信号 → reflect → 文本

暂传项 (升级路径清晰, 不预先抽象):
- target_chains=[]: 真值在 Goal.required_results / plan dict, 暂不解析
- verification_signals=None: 真值在 _validate 输出, 暂不抽
- audit_log_path 硬拼 workspace/.huginn/audit.jsonl

flag: HUGINN_USE_THREE_CABIN=1 开启, 默认 off. 失败返 None 让调用方 fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _build_meta_trace_entry(
    cog: dict[str, Any],
    step_id: int,
) -> dict[str, Any]:
    """舱 1: 从 cog 拼 meta_trace_entry.

    attempted: 这步想做什么 (plan.description 优先, 退化到 action 文本)
    found: 这步发现了什么 (validation.summary 优先, 退化到 execution_result)
    """
    _val = cog.get("validation") or {}
    _exec = cog.get("execution_result") or {}
    _plan = cog.get("plan") or {}

    attempted = (
        str(_plan.get("description") or _plan.get("mode") or "")
        or str(_val.get("attempted") or "")
        or f"step {step_id}"
    )
    found = (
        str(_val.get("summary") or _val.get("result_data") or "")
        or str(_exec.get("summary") or "")
        or str(_val.get("tests_passed") or "")
    )

    return {"iteration": step_id, "attempted": attempted, "found": found}


def run_three_cabin(
    cog: dict[str, Any],
    evals_history: list,
    step_id: int,
    workspace: Path,
    model: Any | None = None,
    persona: Any | None = None,
    kb: Any | None = None,
    memory: Any | None = None,
) -> tuple[Any, str]:
    """三舱主入口. 返回 (step_eval, reflector_hint_text). 失败返 (None, "").

    step_eval 已 append 到 evals_history, 调用方别再 append.
    reflector_hint_text 非空时, 调用方注入到 _speculator_hint.
    """
    try:
        from huginn.metacog.step_evaluator import (
            evaluate_step, should_continue,
        )
        from huginn.metacog.reflector import reflect, format_reflector_text
    except Exception:
        logger.debug("three_cabin imports failed", exc_info=True)
        return (None, "")

    audit_log_path = workspace / ".huginn" / "audit.jsonl"

    # 舱 1: Step
    try:
        mte = _build_meta_trace_entry(cog, step_id)
    except Exception:
        logger.debug("cabin 1 (Step) failed", exc_info=True)
        return (None, "")

    # 舱 2: Evaluate — 真 StepEvaluation, 自动从 audit_log 算 tool_call_health
    try:
        step_eval = evaluate_step(
            meta_trace_entry=mte,
            target_chains=[],
            verification_signals=None,
            memory=memory,
            kb=kb,
            persona=persona,
            model=model,
            audit_log_path=audit_log_path,
        )
        evals_history.append(step_eval)
    except Exception:
        logger.debug("cabin 2 (Evaluate) failed", exc_info=True)
        return (None, "")

    # 舱 3: Reflect — should_continue 抓 Reflector 信号, reflect 产 hint
    hint_text = ""
    try:
        _cont, _reason = should_continue(evals_history, window=3)
        if not _cont and "Reflector" in _reason:
            _actions = reflect(
                step_eval.tool_call_health,
                last_step_evaluations=evals_history[-3:],
                audit_log_path=audit_log_path,
            )
            hint_text = format_reflector_text(_actions)
        if not _cont and not hint_text and _reason:
            # 其他 should_continue=False 信号也透传, 不只 Reflector
            hint_text = f"[StepEval] {_reason}"
    except Exception:
        logger.debug("cabin 3 (Reflect) failed", exc_info=True)

    return (step_eval, hint_text)


# === 自检 ===

if __name__ == "__main__":
    import tempfile

    # 1) _build_meta_trace_entry: 完整 cog
    cog = {
        "validation": {"summary": "tests passed", "tests_passed": True},
        "execution_result": {"summary": "ran 5 tests"},
        "plan": {"description": "run tests", "mode": "execute"},
    }
    mte = _build_meta_trace_entry(cog, step_id=1)
    assert mte["iteration"] == 1
    assert "run tests" in mte["attempted"]
    assert "tests passed" in mte["found"]

    # 2) _build_meta_trace_entry: 空 cog 兜底
    mte2 = _build_meta_trace_entry({}, step_id=2)
    assert mte2["iteration"] == 2
    assert "step 2" in mte2["attempted"]

    # 3) run_three_cabin: 基本流程 (无 model/persona/kb, 走机械信号路径)
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        evals = []
        step_eval, hint = run_three_cabin(
            cog=cog, evals_history=evals, step_id=1, workspace=ws,
        )
        # 没 target_chains / verification_signals / model → on_track="unsure"
        assert step_eval is not None, "step_eval should not be None"
        assert step_eval.on_track == "unsure"
        assert step_eval.attempted == mte["attempted"]
        assert step_eval.found == mte["found"]
        assert len(evals) == 1
        # tool_call_health 字段在 (audit_log 不存在 → None → 兜底 ToolCallHealth())
        assert step_eval.tool_call_health is not None
        # target_chain_ref 暂为 None (target_chains=[])
        assert step_eval.target_chain_ref is None

    # 4) run_three_cabin: 多步累积, evals_history 增长
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        evals = []
        for i in range(3):
            se, _ = run_three_cabin(
                cog=cog, evals_history=evals, step_id=i, workspace=ws,
            )
            assert se is not None
        # 3 步 on_track="unsure" 不触发 should_continue=False
        # (should_continue 只抓 false / low / anomalous)
        assert len(evals) == 3

    # 5) run_three_cabin: 验证 StepEvaluation 是真 dataclass 不是 SimpleNamespace
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        evals = []
        se, _ = run_three_cabin(
            cog=cog, evals_history=evals, step_id=0, workspace=ws,
        )
        # 真 StepEvaluation 有 is_on_track 方法, SimpleNamespace 没有
        assert hasattr(se, "is_on_track"), "should be real StepEvaluation"
        assert callable(se.is_on_track)
        # 字段完整性 (vs SimpleNamespace 缺字段)
        assert hasattr(se, "pmk_feedback")
        assert hasattr(se, "tool_call_health")
        assert hasattr(se, "structure_check")
        assert hasattr(se, "evidence_quality")

    print("all self-checks passed")
