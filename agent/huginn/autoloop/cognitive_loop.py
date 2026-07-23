"""CognitiveLoop — rcb_runner 和 autoloop 的共享控制流内核.

核心问题: rcb_runner (6-step mini-loop) 和 autoloop (7-phase heavy loop) 是两条
独立代码线, 每次改 metacog 都要手动接两遍. Step 0 抽共享内核, 让优化改一处
影响两条路径.

设计原则 (ponytail):
- 不强行统一 6-step 和 7-phase — 它们任务粒度不同, 强行统一会引入抽象税
- 只统辖 4 个钩子: observe / decide / execute_action / reflect
- 两条路径各自实现钩子, CognitiveLoop 只负责编排
- 不绑定输出格式 — 通过 output_writer 接口注入 (RCBReportWriter / ProvenanceWriter)

向上兼容:
- rcb_runner: 6-step 逻辑作为 execute_action 的实现, observe/decide/reflect 退化为
  轻量钩子 (首轮 observe 读 INSTRUCTIONS.md, decide 固定 "execute", reflect 跑
  StepEvaluator + should_continue + detect_drift)
- autoloop: 7-phase 逻辑作为 execute_action 的实现, observe=perceive, decide=LLM
  选 next phase, reflect=metacog check

不做的事 (反模式警示):
- 不新建 7 个独立 phase 文件 — phase 方法保持原位
- 不做完整 state machine — decide() 输出字符串 action
- 不引入新依赖 — 用现有 LLM client
- 不一次全改 — 先抽骨架, 两条路径逐步迁移
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# 700 万步极限场景的 action_history 滑动窗口. 所有调用方只用尾部 (decider prompt
# 取 [-10:], cycle_detect O(n²) 在 1000 可接受, count/reversed/len 都 O(n)).
# ponytail: 环境变量覆盖, 极限模式可调大. 升级路径: deque + 专用 tail_n().
_MAX_ACTION_HIST = int(os.environ.get("HUGINN_ACTION_HIST_MAX", "1000"))


def _extract_tests_passed(validation: Any) -> bool:
    """从 validation 结果里抽 tests_passed 布尔, 给 validate→learn 门用.

    validation 形状不固定 (dict / str / None), 抽不出明确失败就默认 True,
    避免门控把现有 happy path 误阻断. 只有明确说 fail / passed=False 才拦.
    """
    if isinstance(validation, dict):
        for key in ("tests_passed", "passed", "success", "ok"):
            if key in validation:
                return bool(validation[key])
        return True
    if isinstance(validation, str):
        low = validation.lower()
        if "fail" in low:
            return False
        return True
    # None 或其它: 没有明确失败信号, 放行
    return True


@dataclass
class LoopState:
    """CognitiveLoop 的可观测状态 — observe 填, decide 读, reflect 更新.

    不含 phase 内部状态 (那些留在 rcb_runner/autoloop 的 self 上).
    只含跨钩子共享的控制流状态.
    """
    iteration: int = 0
    max_iterations: int = 20
    should_stop: bool = False
    should_redirect: bool = False  # reflect 设 True, decide 看到 → 换 action
    last_action: str = ""  # 上次执行的 action (observe/decide/execute/learn/report/...)
    last_action_result: Any = None
    redirect_reason: str = ""  # reflect 写, decide 读
    # 死循环防护: action 历史 + 重复检测
    action_history: list[str] = field(default_factory=list)
    # 基模自主性: decide 返回的 rationale (供 reflect 评估)
    last_rationale: str = ""


@dataclass
class ActionDecision:
    """decide() 的返回 — LLM 或规则选的下一个 action."""
    action: str  # observe/hypothesize/plan/execute/validate/learn/pivot/skip/stop
    rationale: str = ""
    expected_outcome: str = ""
    force: bool = False  # True = 跳过 reflect 的 redirect 建议

# action 合法集合 — decide() 只能返回这些, 否则 reflect 标 redirect
# D3: report 不在 VALID_ACTIONS — _finalize_run 自动跑, LLM 选 report 等于
# 浪费一轮 (execute_fn 是 no-op). 升级路径: 如果要 LLM 主动触发 report,
# 改成 action="stop" + rationale="report ready".
VALID_ACTIONS = frozenset({
    "observe", "hypothesize", "plan", "execute", "validate",
    "learn", "pivot", "skip", "stop",
})


@dataclass
class ReflectionResult:
    """reflect() 的返回 — 评估上轮 action, 决定是否 redirect/stop."""
    should_continue: bool = True
    should_redirect: bool = False
    should_stop: bool = False
    redirect_reason: str = ""
    # 死循环检测: 连续重复 action 数
    repeated_action_count: int = 0
    # 评估意见 (供下轮 decide 看到)
    advice: str = ""


class CognitiveLoop:
    """共享控制流内核 — 编排 observe/decide/execute_action/reflect 四个钩子.

    使用方式:
        loop = CognitiveLoop(
            observe_fn=my_observe,
            decide_fn=my_decide,
            execute_fn=my_execute,
            reflect_fn=my_reflect,
            output_writer=my_writer,
            max_iterations=20,
        )
        result = await loop.run(initial_state)

    两条路径各自实现 4 个钩子, CognitiveLoop 不关心具体实现.
    """

    def __init__(
        self,
        observe_fn: Callable[[LoopState], Awaitable[dict[str, Any]]],
        decide_fn: Callable[[LoopState, dict[str, Any]], Awaitable[ActionDecision]],
        execute_fn: Callable[[LoopState, ActionDecision], Awaitable[Any]],
        reflect_fn: Callable[[LoopState, ActionDecision, Any], Awaitable[ReflectionResult]],
        output_writer: Any | None = None,
        max_iterations: int = 20,
        max_repeated_actions: int = 3,  # 死循环防护: 连续重复 action 超过此数 → stop
    ) -> None:
        self._observe = observe_fn
        self._decide = decide_fn
        self._execute = execute_fn
        self._reflect = reflect_fn
        self._output_writer = output_writer
        self._max_iterations = max_iterations
        self._max_repeated_actions = max_repeated_actions

    async def run(self, initial_state: LoopState | None = None) -> LoopState:
        """主循环: observe → decide → execute → reflect, 直到 should_stop 或 max_iter.

        不抛异常 — 钩子内部失败由钩子自己处理, CognitiveLoop 只负责编排.
        返回最终 LoopState, 调用方从 state.last_action_result 取最终产物.
        """
        state = initial_state or LoopState(max_iterations=self._max_iterations)
        state.max_iterations = self._max_iterations

        while state.iteration < state.max_iterations and not state.should_stop:
            state.iteration += 1
            logger.info("CognitiveLoop iter %d/%d", state.iteration, state.max_iterations)

            # 1. observe — 轻量环境扫描, 不强制 git diff
            try:
                observation = await self._observe(state)
            except Exception as e:
                logger.warning("observe failed: %s", e)
                observation = {}

            # 2. decide — 选下一个 action (LLM 自主或规则)
            try:
                decision = await self._decide(state, observation)
                if decision.action not in VALID_ACTIONS:
                    logger.warning(
                        "decide returned invalid action %r, defaulting to 'skip'",
                        decision.action,
                    )
                    decision.action = "skip"
            except Exception as e:
                logger.warning("decide failed: %s, defaulting to 'stop'", e)
                state.should_stop = True
                break

            # 死循环防护: 连续重复 action 超上限 → 强制 redirect, 超过 2 倍上限 → stop
            state.action_history.append(decision.action)
            # 700 万步极限场景防内存爆炸: 截断到滑动窗口. 所有调用方只用尾部
            # ([-10:], count, reversed, len), 截断前缀不影响语义. ponytail: 保留
            # list 不改 deque, 避免切片兼容性. 升级路径: deque + 专用 tail_n().
            if len(state.action_history) > _MAX_ACTION_HIST:
                del state.action_history[: -_MAX_ACTION_HIST]
            state.last_action = decision.action
            state.last_rationale = decision.rationale
            repeated = self._count_repeated_tail(state.action_history)
            if repeated >= self._max_repeated_actions and not decision.force:
                logger.warning(
                    "CognitiveLoop: %d consecutive '%s' actions, forcing redirect",
                    repeated, decision.action,
                )
                state.should_redirect = True
                state.redirect_reason = (
                    f"死循环防护: 连续 {repeated} 次 '{decision.action}', 换方向"
                )
                # 超过 2 倍上限仍未换方向 → 真死循环, 强制 stop
                if repeated >= self._max_repeated_actions * 2:
                    logger.error(
                        "CognitiveLoop: %d consecutive '%s' (2x limit), forcing stop",
                        repeated, decision.action,
                    )
                    state.should_stop = True
                    break

            # 3. execute_action — 跑 LLM 选的 action
            try:
                result = await self._execute(state, decision)
                state.last_action_result = result
            except Exception as e:
                logger.warning("execute_action '%s' failed: %s", decision.action, e)
                state.last_action_result = None
                # 失败不阻塞, 让 reflect 评估

            # 4. reflect — 评估上轮, 决定是否继续/换方向/停
            try:
                reflection = await self._reflect(state, decision, state.last_action_result)
                reflection.repeated_action_count = repeated
            except Exception as e:
                logger.warning("reflect failed: %s, defaulting to continue", e)
                reflection = ReflectionResult()

            state.should_stop = state.should_stop or reflection.should_stop
            state.should_redirect = reflection.should_redirect
            if reflection.should_redirect and reflection.redirect_reason:
                state.redirect_reason = reflection.redirect_reason

            # output_writer 钩子 — 每轮末尾写产物 (RCBench 写 report.md, autoloop 写 provenance)
            if self._output_writer is not None:
                try:
                    self._output_writer.write_step(
                        iteration=state.iteration,
                        action=decision.action,
                        result=state.last_action_result,
                        reflection=reflection,
                    )
                except Exception as e:
                    logger.debug("output_writer.write_step failed: %s", e)

            if decision.action == "stop":
                state.should_stop = True
                break

        return state

    @staticmethod
    def _count_repeated_tail(history: list[str]) -> int:
        """数 history 末尾连续重复的 action 数."""
        if not history:
            return 0
        last = history[-1]
        count = 0
        for a in reversed(history):
            if a == last:
                count += 1
            else:
                break
        return count


# === output_writer 接口 (供两条路径各自实现) ===

class OutputWriter:
    """output_writer 接口 — 两条路径各自实现 write_step.

    RCBench: RCBReportWriter 把 step 结果累积到 report/report.md
    Autoloop: ProvenanceWriter 把 step 结果写 provenance JSONL
    """

    def write_step(
        self,
        iteration: int,
        action: str,
        result: Any,
        reflection: ReflectionResult,
    ) -> None:
        raise NotImplementedError


# === AV4 路径 B: 共享元认知护航原语 ===
# rcb_runner mini-loop 和 autoloop run_cognitive 各自接了一遍 heat_engine /
# detect_drift / TaskMetrics / should_pause_for_decision, 代码独立. 这里抽共享
# 函数, 让两边调同一份, 避免 AV2/AV6/AV7/AV8 类型的偏移再发生.
# ponytail: 只抽无状态纯函数, 不引入 CognitiveLoop 子类, 不绑定 LLM.

def update_heat_engine_after_step(
    heat_engine: Any,
    step_eval: Any,
    prompt_len: int,
    idea_count: int,
    *,
    stable_principles_count: int = 1,
) -> None:
    """AV4: heat_engine 闭环共享函数 — StepEvaluation → T_hot/T_cold/kinematics.

    rcb_runner (AV8) 和 autoloop run_cognitive 都调这个, 避免 4 档映射逻辑两边各写一份.
    T_hot: evidence_quality 熵代理 (low=0.8 / medium=0.5 / high=0.2 / unknown=0.5).
    T_cold: on_track 秩序代理 (true=0.7 / false=0.2 / unsure=0.4).
    ponytail: 4 档离散映射, 天花板: 跳变不连续; 升级路径接连续 evidence_score.
    """
    if heat_engine is None:
        return
    try:
        _eq = (getattr(step_eval, "evidence_quality", "unknown") or "unknown").lower().strip()
        _t_hot_proxy = {"low": 0.8, "medium": 0.5, "high": 0.2}.get(_eq, 0.5)
        _ot = (getattr(step_eval, "on_track", "unsure") or "unsure").lower().strip()
        _t_cold_proxy = {"true": 0.7, "false": 0.2}.get(_ot, 0.4)
        heat_engine.update_T_hot(_t_hot_proxy)
        heat_engine.update_T_cold(_t_cold_proxy, darwin_score=0.0)
        heat_engine.update_kinematics(
            idea_count=idea_count,
            stable_principles_count=stable_principles_count,
            system_prompt_len=prompt_len,
        )
    except Exception as exc:
        logger.debug("update_heat_engine_after_step failed: %s", exc)


def update_drift_and_metrics(
    evals_history: list,
    step_eval: Any,
    task_metrics: Any,
    task_state: Any,
    workspace: Any,
    run_id: str,
    max_iterations: int,
) -> tuple[tuple | None, Any]:
    """AV4: detect_drift + TaskMetrics 滚动更新共享函数.

    返回 (drift_info, task_metrics). 失败时 drift_info=None, task_metrics 不变.
    rcb_runner (G62+G70) 和 autoloop reflect_fn (AV2) 都调这个.
    ponytail: duck typing — step_eval 可以是 StepEvaluation 或 SimpleNamespace.
    """
    drift_info: tuple | None = None
    try:
        from huginn.metacog.target_chain import detect_drift as _detect_drift
        drift_info = _detect_drift(evals_history, window=3)
    except Exception as exc:
        logger.debug("AV4 detect_drift failed: %s", exc)

    try:
        from huginn.runtime.task_metrics import (
            TaskMetrics as _TM, load_metrics as _lm,
            update_metrics as _um, save_metrics as _sm,
        )
        if task_metrics is None:
            import time
            task_metrics = _lm(run_id, workspace) or _TM(
                task_id=run_id, total_steps=max_iterations * 6,
            )
            if task_state is None:
                from types import SimpleNamespace as _NS
                task_state = _NS(created_at=time.time())
        task_metrics = _um(task_metrics, step_eval, task_state=task_state)
        _sm(task_metrics, workspace)
    except Exception as exc:
        logger.debug("AV4 TaskMetrics update failed: %s", exc)

    return drift_info, task_metrics


def build_pmk_state(
    persona: Any,
    last_step_eval: Any,
    kb: Any,
    *,
    top_k: int = 2,
) -> dict[str, str] | None:
    """AV4: PMK 三路立场状态构建共享函数.

    persona: PersonaManager 返回的对象或 dict, 取 description.
    last_step_eval: 上一步 StepEvaluation, 取 pmk_feedback 中 Memory 段.
    kb: 知识库, 用 last_step_eval.attempted 查 top_k hits 取前 200 字.
    返回 {"persona","memory","kb"} 或 None (全空时).
    rcb_runner (P0-A) 和 autoloop reflect_fn (AV2) 都调这个.
    ponytail: 三段文本拼接, 不上 LLM 抽取; 升级路径接 LLM 立场抽取.
    """
    try:
        _persona_text = ""
        if persona is not None:
            _persona_text = str(
                getattr(persona, "description", None)
                or (persona.get("description") if isinstance(persona, dict) else "")
                or ""
            )
        _mem_text = ""
        if last_step_eval is not None:
            _pmk_fb = getattr(last_step_eval, "pmk_feedback", "") or ""
            for _seg in _pmk_fb.split(";"):
                _seg = _seg.strip()
                if _seg.lower().startswith("memory:"):
                    _mem_text = _seg[len("memory:"):].strip()
                    break
        _kb_text = ""
        if kb is not None and last_step_eval is not None:
            try:
                _kb_hits = kb.query(
                    getattr(last_step_eval, "attempted", "") or "", top_k=top_k)
                if _kb_hits:
                    _kb_text = " ".join(
                        str(h.get("content", "") if isinstance(h, dict) else h)
                        for h in _kb_hits[:top_k]
                    )[:200]
            except Exception:
                pass
        if _persona_text or _mem_text or _kb_text:
            return {
                "persona": _persona_text,
                "memory": _mem_text,
                "kb": _kb_text,
            }
    except Exception as exc:
        logger.debug("AV4 build_pmk_state failed: %s", exc)
    return None


def check_pause_decision(
    evals_history: list,
    target_chains: list,
    kb: Any,
    fired_intentions: list | None,
    pmk_state: dict[str, str] | None,
    grill_state: dict | None = None,
) -> tuple[bool, str, list]:
    """AV4: should_pause_for_decision 共享包装.

    返回 (pause, reason, opts). 失败时 (False, "", []).
    rcb_runner (G71) 和 autoloop reflect_fn (AV2) 都调这个.
    ponytail: 只调判定, 不做 lifecycle/resume — 那些动作两条路径不同, 留在 caller.

    grill_state (P0 grill-me) 由 caller 在 plan_check 阶段构造:
    {"has_grilled": bool, "ambiguity_score": float, "tier": str,
     "scene_tag": str, "plan_is_empty": bool}.
    """
    try:
        from huginn.runtime.task_lifecycle import (
            should_pause_for_decision as _spd,
        )
        _pause, _reason, _opts = _spd(
            evals_history, target_chains,
            kb_recall_empty=(kb is None),
            fired_intentions=fired_intentions or [],
            pmk_state=pmk_state,
            grill_state=grill_state,
        )
        return bool(_pause), str(_reason or ""), list(_opts or [])
    except Exception as exc:
        logger.debug("AV4 check_pause_decision failed: %s", exc)
        return False, "", []


# 已知 _validate dict 字段 (autoloop engine.py _validate 方法返回):
#   tests_passed: bool
#   constraints_satisfied: bool
#   benchmarks: dict (key: benchmark_name, val: {metric: value})
#   r_phys / physics_validation / physics_validation_error
#   thinking_collapse / thinking_collapse_error
#   test_output / math_validation / math_validation_error / math_evidence_error
#   generative_verify / generative_verify_error
#   reviewer_critique / reviewer_critique_error
#   grader_scores / grader_reward / grader_error
#   eval_summary / prediction_error
#   effort_floor_passed / effort_floor_deficits / failure_kind
#   emergent_complexity / literature_comparison / ...
_VALIDATION_ERROR_KEYS: tuple[str, ...] = (
    "thinking_collapse",
    "physics_validation_error",
    "math_validation_error",
    "math_evidence_error",
    "generative_verify_error",
    "reviewer_critique_error",
    "grader_error",
    "prediction_error",
    "test_output",
    "effort_floor_deficits",
)


def _validation_to_step_eval_fields(
    validation: dict,
    tests_ok: bool,
    execution_result: Any,
    *,
    step_id: int,
) -> dict:
    """P0.2: _validate dict → StepEvaluation 兼容字段映射.

    之前 reflect_fn 硬取 summary/result/errors 三个不存在的字段, 导致
    _step_eval.attempted/found/deviation 全空, PMK/drift/metrics 吃空数据.
    这里集中映射, 测试能直接验.

    attempted: execution_result 的 description (validation 里没有).
    found: tests_passed + benchmarks 关键指标摘要.
    deviation: 失败时收集所有 *_error / thinking_collapse / effort_floor_deficits.
    evidence_quality: tests_ok=high, 否则 low; 有 *_error 降级.
    structure_check: tests_ok=passed, 否则 failed.

    ponytail: dict 字段映射, 不上 schema 库. 升级路径: pydantic model.
    """
    # attempted: validation 里没有 attempted, 从 execution_result 拼
    # G6: PMK 的 K 查询看不到 visual_primitives → 加 visual 摘要让 KB 能召回视觉经验.
    _attempted = ""
    if isinstance(execution_result, dict):
        _attempted = str(
            execution_result.get("description")
            or execution_result.get("summary")
            or execution_result.get("result_type")
            or ""
        )[:200]
    # G6: 附 visual_primitives 摘要 (前 150 字), 让 PMK 的 K 查询能看到视觉内容.
    # ponytail: 直接拼字符串, 不上 schema. 升级路径: StepEvaluation 加 visual_context 字段.
    _vis = validation.get("visual_primitives") if isinstance(validation, dict) else None
    if _vis and isinstance(_vis, str):
        _attempted = (f"{_attempted} | Visual: {_vis[:150]}").strip(" |")

    # found: tests_passed 状态 + benchmarks 关键指标
    _found_parts: list[str] = []
    if tests_ok:
        _found_parts.append("tests_passed=True")
    else:
        _found_parts.append("tests_passed=False")
    _benches = validation.get("benchmarks") or {}
    if isinstance(_benches, dict):
        for _bn, _bv in list(_benches.items())[:3]:
            if isinstance(_bv, dict):
                _metric = _bv.get("metric") or _bv.get("value") or ""
                if _metric:
                    _found_parts.append(f"{_bn}={_metric}")
            else:
                _found_parts.append(f"{_bn}={_bv}")
    _found = "; ".join(_found_parts)[:200]

    # deviation: 失败时收集错误信号
    _dev_parts: list[str] = []
    if not tests_ok:
        for _k in _VALIDATION_ERROR_KEYS:
            _v = validation.get(_k)
            if _v:
                _dev_parts.append(f"{_k}: {str(_v)[:100]}")
    _deviation = "; ".join(_dev_parts)[:300]

    # evidence_quality: tests_ok=high, 否则 low
    _eq = "high" if tests_ok else "low"
    if _dev_parts:
        _eq = "low"

    return {
        "step_id": step_id,
        "attempted": _attempted,
        "found": _found,
        "on_track": "true" if tests_ok else "false",
        "evidence_quality": _eq,
        "deviation": _deviation,
        "structure_check": "passed" if tests_ok else "failed",
        "pmk_feedback": "",
        "tool_call_health": None,
        "target_chain_ref": None,
    }




class CognitiveLoopMixin:
    """cognitive loop 主循环方法族, 从 engine.py 下沉 (P3 slim-down). 通过 self 访问 engine 状态."""

    pass  # methods migrated from engine.py via P3 slim-down

# === 自检 ===

    def _run_phase(self, name: str, fn, *args) -> LoopPhase:
        """Run a synchronous phase function."""
        phase = LoopPhase(name=name)
        phase.start_time = time.time()
        phase.status = "running"
        # 同步路径: 如果当前在 event loop 里, fire-and-forget 发开始事件.
        # 不 await, 因为 _run_phase 本身是同步的.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._dispatch_stage_event(EventType.ON_WORKFLOW_STAGE_START, name)
            )
        except RuntimeError:
            logger.warning(
                "error in _run_phase: stage-start event dispatch skipped (no running loop)",
                exc_info=True,
            )
        # 包 telemetry span: 把 phase 级决策也记进轨迹, 回放时不止看 tool_call
        from huginn.telemetry import get_telemetry_collector

        span_cm = get_telemetry_collector().span(f"phase:{name}")
        try:
            with span_cm as phase_span:
                phase.result = fn(*args)
                phase.status = "completed"
                phase_span.metadata["status"] = "completed"
        except Exception as e:
            phase.status = "failed"
            phase.error = str(e)
            try:
                phase_span.metadata["status"] = "failed"
                phase_span.metadata["error"] = str(e)
            except Exception:
                logger.warning(
                    "error in _run_phase: span metadata update failed", exc_info=True
                )
        phase.end_time = time.time()
        # fire-and-forget 发结束/失败事件
        try:
            loop = asyncio.get_running_loop()
            done_type = (
                EventType.ON_WORKFLOW_STAGE_DONE
                if phase.status == "completed"
                else EventType.ON_WORKFLOW_FAILED
            )
            loop.create_task(
                self._dispatch_stage_event(
                    done_type,
                    name,
                    duration_sec=phase.end_time - (phase.start_time or 0),
                    error=phase.error,
                )
            )
        except RuntimeError:
            logger.warning(
                "error in _run_phase: stage-done event dispatch skipped (no running loop)",
                exc_info=True,
            )
        return phase

    async def _run_phase_async(self, name: str, fn, *args) -> LoopPhase:
        """Run an async phase function."""
        phase = LoopPhase(name=name)
        phase.start_time = time.time()
        phase.status = "running"
        # 记下当前 phase, 让 _llm_chat 能注入 phase-aware thinking effort 指令.
        # ponytail: 隐式状态, 但 run() 是 single-threaded async, 无竞态.
        self._current_phase = name
        # C2: 追踪本 run 的 phase 序列, 供 trajectory_match 召回用.
        if not hasattr(self, "_current_run_phases"):
            self._current_run_phases = []
        self._current_run_phases.append(name)
        # 防止无界增长 (1000+ iter × 7 phase = 7000+), 只保留最近 50 个
        if len(self._current_run_phases) > 50:
            self._current_run_phases = self._current_run_phases[-50:]
        await self._dispatch_stage_event(EventType.ON_WORKFLOW_STAGE_START, name)
        from huginn.telemetry import get_telemetry_collector

        span_cm = get_telemetry_collector().span(f"phase:{name}")
        try:
            with span_cm as phase_span:
                phase.result = await fn(*args)
                phase.status = "completed"
                phase_span.metadata["status"] = "completed"
        except Exception as e:
            phase.status = "failed"
            phase.error = str(e)
            try:
                phase_span.metadata["status"] = "failed"
                phase_span.metadata["error"] = str(e)
            except Exception:
                logger.warning(
                    "error in _run_phase_async: span metadata update failed",
                    exc_info=True,
                )
        phase.end_time = time.time()
        if phase.status == "completed":
            await self._dispatch_stage_event(
                EventType.ON_WORKFLOW_STAGE_DONE,
                name,
                duration_sec=phase.end_time - (phase.start_time or 0),
            )
        else:
            await self._dispatch_stage_event(
                EventType.ON_WORKFLOW_FAILED,
                name,
                duration_sec=phase.end_time - (phase.start_time or 0),
                error=phase.error,
            )
        return phase

    def _render_report(self, data: dict[str, Any]) -> str:
        """Render a markdown report."""
        lines = [
            "# Huginn Autoloop Report",
            "",
            f"**Objective:** {data['objective']}",
            f"**Run ID:** {data['run_id']}",
            f"**Total Time:** {data['total_time_seconds']:.1f}s",
            "",
            "## Phases",
            "",
            "| Phase | Status | Duration (s) | Error |",
            "|-------|--------|--------------|-------|",
        ]
        for p in data["phases"]:
            lines.append(
                f"| {p['name']} | {p['status']} | {p['duration']:.1f} | {p['error'] or ''} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("Generated by Huginn Autoloop Engine")
        return "\n".join(lines)
    def _git_commit_after_execute(self, plan: dict, iteration: int) -> None:
        """execute 后 git commit — 让下轮 perceive 看到 diff (从 run() 抽出)."""
        try:
            import subprocess as _sp
            import time as _time
            _sp.run(["git", "add", "-A"], cwd=self.workspace,
                    capture_output=True, timeout=10)
            _msg = f"[iter {iteration}] {plan.get('mode','?')}: {plan.get('description','')[:80]}"
            for _attempt in range(3):
                _r = _sp.run(["git", "commit", "-m", _msg], cwd=self.workspace,
                             capture_output=True, timeout=10)
                if _r.returncode == 0:
                    break
                if _attempt < 2:
                    _time.sleep(1 * (_attempt + 1))
        except Exception:
            pass  # no git repo or git unavailable — not our problem

    def _darwin_ratchet_check(self) -> None:
        """Darwin ratchet: 算假设质量分, 只保留改进, 连续低增益 → early stop.

        评分 (0-10, 对齐 darwin-skill 原版 0-10 分制):
        - supported_ratio * 10: 证据强度 (supported 节点占比)
        - testable_ratio * 10: 可证伪性 (有 testable_prediction 的节点占比)
        - graph_diversity * 10: 假设多样性 (unique statements 占比)
        - topology_richness * 10: 假设网络结构丰富度 (β₁/n, 独立环数占比)
        四项平均 → 0-10 分

        β₁ 解释: 假设图的独立环数. β₁=0 → 树状 (无交叉支持);
        β₁>0 → 有交叉支持/反驳链 (假设间相互关联). 标准化到 [0,1] 避免大图偏向.
        ponytail: 不区分"良性交叉支持"和"恶性循环论证" — 留给 red_team._topology_scan 判.
        这里只测结构丰富度, 作为 4 维代理之一. 升级: LLM 9 维评分 (darwin-skill 原版).

        棘轮逻辑:
        - score > best_score → 更新 best, stagnation=0
        - score <= best_score → stagnation++, 回退 (不更新 preferred_hypothesis)
        - 连续 2 轮 Δ<0.5 (0-10 分制下, 增量 <0.5) → early stop

        ponytail: 4 维代理是粗启发式. 升级: LLM 9 维评分 (darwin-skill 原版).
        ponytail: 回退只标记, 不真删假设 (保留在图里供 cross-pollination).
        """
        graph = self.hypothesis_graph
        all_nodes = graph.all_nodes()
        if not all_nodes:
            return

        supported = graph.supported()
        n = len(all_nodes)
        supported_ratio = len(supported) / n

        testable = sum(
            1 for nd in all_nodes if getattr(nd, "testable_prediction", None)
        )
        testable_ratio = testable / n

        statements = [nd.statement for nd in all_nodes if nd.statement]
        unique = len(set(statements))
        graph_diversity = unique / len(statements) if statements else 0.0

        # 第 4 维: 拓扑丰富度 — 用 hodge_signature 的 β₁ 算独立环数占比
        # ponytail: 失败时降级为 0 (不影响其他 3 维). 升级: gudhi 算真实 Betti.
        topology_richness = 0.0
        try:
            from huginn.metacog.topology_lens import hodge_signature

            node_ids = [nd.id for nd in all_nodes]
            edge_pairs = []
            for e in graph.edges():
                if e.from_id in node_ids and e.to_id in node_ids:
                    edge_pairs.append((e.from_id, e.to_id))
            sig = hodge_signature(node_ids, edge_pairs)
            # β₁/n 标准化到 [0,1]: 树状图 β₁=0 → 0 分; 完全交叉 → 趋近 1
            topology_richness = min(sig.beta1_approx / max(n, 1), 1.0)
        except Exception:
            logger.debug("topology_richness calc failed (non-fatal)", exc_info=True)

        # 0-10 分制, 对齐 darwin-skill 原版
        score = (
            (supported_ratio + testable_ratio + graph_diversity + topology_richness)
            / 4.0
            * 10.0
        )

        delta = score - self._darwin_last_score
        if delta < 0.5:
            self._darwin_stagnation += 1
        else:
            self._darwin_stagnation = 0

        if score > self._darwin_best_score:
            self._darwin_best_score = score
            # ponytail: 只在改进时更新 preferred, 退化时保留上次最佳
        # else: 保留 best_score, preferred_hypothesis 不更新 (棘轮)

        self._darwin_last_score = score

        # P2-6 belief: Gaussian 后验更新. 单值 score 当观测, obs_sigma2=1.0
        # (0-10 分制下单次观测噪声约 1 分). σ² 减小 = belief 收敛.
        # toggle: HUGINN_BELIEF_DARWIN (默认 on). off 时只走原 delta<0.5 逻辑.
        if os.environ.get("HUGINN_BELIEF_DARWIN", "1") != "0":
            try:
                from huginn.tools.subagent_tool import _gaussian_update
                self._darwin_belief_mu, self._darwin_belief_sigma2 = _gaussian_update(
                    self._darwin_belief_mu, self._darwin_belief_sigma2,
                    float(score), 1.0,
                )
                try:
                    from huginn.routes.metrics import track_belief_update
                    track_belief_update("gaussian")
                except Exception:
                    pass
            except Exception:
                pass  # 循环 import 或其他故障 → 回退原逻辑

        # v6 G54: 把 darwin 分数 / supported_ratio 暴露给 _plan / _validate
        # ponytail: evidence_strength 用 supported_ratio 做代理, 已在算分时拿到,
        # 不重复调 RAG. 升级路径: 真 RAG recall 命中数 / provenance 引用数.
        self._last_hypothesis_confidence = score / 10.0
        self._last_hypothesis_evidence_strength = float(supported_ratio)

        # v7 G59: 更新认知热机 T_cold (paradigm 秩序代理)
        # supported_ratio 高 = validation 提取有序能力强 = 冷源温度低
        try:
            from huginn.metacog.cognitive_heat_engine import get_heat_engine
            eng = get_heat_engine()
            eng.update_T_cold(float(supported_ratio), float(score))

            # 推送 health 到 EventBus + SSE. 每轮 darwin 后推一次, 让前端实时看
            # Re_cog / η_cog / status. _should_imaginate 已 update_kinematics,
            # 但若本轮没触发 imaginate, 这里强制 update 保证 health 反映当前状态.
            n_ideas = len(all_nodes)
            n_principles = 0
            try:
                sp = getattr(self, "stable_principles", None)
                n_principles = len(sp) if sp else 0
            except Exception:
                pass
            sys_prompt_len = 0
            try:
                sys_prompt_len = len(getattr(self, "system_prompt", "") or "")
            except Exception:
                pass
            eng.update_kinematics(n_ideas, n_principles + 1, sys_prompt_len)

            health = eng.health_check()
            self._emit_campaign("heat_engine.health", health)
        except Exception:
            logger.debug("heat_engine.update_T_cold failed (non-fatal)", exc_info=True)

        # v7 长任务: stagnation 阈值 2→5. Oxelra 206 步允许长期低增益,
        # 2 轮就 early stop 太激进, 真正突破常在 10+ 轮停滞之后.
        _stag_limit = int(os.environ.get("HUGINN_DARWIN_STAGNATION_LIMIT", "5"))
        if self._darwin_stagnation >= _stag_limit and self._iteration > 2:
            # P2: stagnation 触发前先分类 (chaoxu 启发).
            # method_failure → pivot 换方法继续, 不 stop
            # evidence_against → counterexample hunt, 不 stop
            # unclassifiable / 已试过 → 真 stop
            _stall_action = self._classify_stall()
            if _stall_action == "pivot":
                logger.info(
                    "darwin ratchet: stagnation %d → method_failure, pivot (不 stop)",
                    self._darwin_stagnation,
                )
                self._darwin_stagnation = 0  # 给 pivot 后的新假设重新累积
            elif _stall_action == "counterexample":
                logger.info(
                    "darwin ratchet: stagnation %d → evidence_against, counterexample hunt (不 stop)",
                    self._darwin_stagnation,
                )
                self._darwin_stagnation = 0
                self._trigger_counterexample_hunt()
            else:
                # P5 (chaoxu 启发): persistent goal mode — stagnation 分类为 stop
                # 时, 如果开了 HUGINN_PERSISTENT_GOAL_MODE 且有 active goal 且
                # 挂钟预算未耗尽, 不 early stop, 重置 stagnation 继续.
                # 无 active goal 或挂钟耗尽才真 stop.
                _persistent = (
                    os.environ.get("HUGINN_PERSISTENT_GOAL_MODE", "0") == "1"
                )
                _wall_expired = False
                _has_active_goal = False
                if _persistent:
                    try:
                        from huginn.autoloop.goal_store import get_goal_store
                        _gs = get_goal_store()
                        _ag = _gs.get_active()
                        if _ag is not None:
                            _has_active_goal = True
                            _wall_expired = _gs.wall_clock_expired(_ag.id)
                    except Exception:
                        logger.debug("P5 wall_clock check failed", exc_info=True)
                if _persistent and _has_active_goal and not _wall_expired:
                    logger.info(
                        "darwin ratchet: stagnation %d → stop, but persistent goal "
                        "mode on + wall_clock not expired, reset & continue",
                        self._darwin_stagnation,
                    )
                    self._darwin_stagnation = 0
                else:
                    logger.info(
                        "darwin ratchet: stagnation %d rounds (Δ<0.5), best=%.2f, early stop",
                        self._darwin_stagnation,
                        self._darwin_best_score,
                    )
                    self._should_stop = True

        # P2-6 belief: σ² 收敛也作为 stop 信号. σ² < 0.1 = belief 不确定性低,
        # 后续观测不会显著改变 μ, 边际信息收益递减. 跟 stagnation 互补:
        # stagnation 测"score 不增", σ² 测" belief 不再变". 两者任一触发即 stop.
        if (
            os.environ.get("HUGINN_BELIEF_DARWIN", "1") != "0"
            and self._darwin_belief_sigma2 < 0.1
            and self._iteration > 2
        ):
            logger.info(
                "darwin ratchet: belief converged σ²=%.4f μ=%.2f, early stop",
                self._darwin_belief_sigma2, self._darwin_belief_mu,
            )
            self._should_stop = True

        # v7 Meta-Trace: 每轮蒸馏成结构化科研要点, 对标 Oxelra Meta-Trace.
        # 目标: 长任务不靠完整 transcript, 用结构化要点保持 context 密度.
        # ponytail: 从已有 self.* 字段抽, 不调 LLM (省 token). ceiling 是 LLM 蒸馏.
        try:
            self._distill_meta_trace(score, supported_ratio)
        except Exception:
            logger.debug("meta_trace distill failed (non-fatal)", exc_info=True)

    def _classify_stall(self) -> str:
        """P2: stagnation 触发时归因 (chaoxu 启发).

        分两类:
        - method_failure: 当前方法/工具用尽, 换方法能救 → 返回 "pivot"
        - evidence_against: 证据指向假设本身错 → 返回 "counterexample"
        - unclassifiable / 已试过太多次 → 返回 "stop"

        ponytail: 用 _last_failure_mode + _consecutive_failures + pivot_count
        做规则归因, 不调 LLM (省 token). 升级: LLM judge 归因.
        ceiling: 只用已有信号, 不引入新 sensor.
        """
        _fail_mode = getattr(self, "_last_failure_mode", "") or ""
        _consec = getattr(self, "_consecutive_failures", 0)
        _pivots = getattr(self, "_pivot_count", 0)
        _max_pivots = getattr(self, "_max_pivots", 10)
        # evidence_against 信号: failure_mode 含 hypothesis_error / refuted /
        # counterexample, 或最近 validation 明确反驳
        _evidence_signals = (
            "hypothesis_error" in _fail_mode
            or "refuted" in _fail_mode.lower()
            or "counterexample" in _fail_mode.lower()
            or "contradicts" in _fail_mode.lower()
        )
        # method_failure 信号: failure_mode 含 tool_error / param_error /
        # timeout / convergence, 或纯工具失败
        _method_signals = (
            "tool_error" in _fail_mode
            or "param_error" in _fail_mode
            or "timeout" in _fail_mode
            or "convergence" in _fail_mode
            or "data_noise" in _fail_mode
        )
        # 已 pivot 太多次 → 不再 pivot, 考虑 stop
        if _pivots >= _max_pivots:
            return "stop"
        if _evidence_signals and not _method_signals:
            return "counterexample"
        if _method_signals and not _evidence_signals:
            return "pivot"
        # 混合信号或无信号: 看 consecutive_failures
        # 高失败率 → 假设方向可能错 → counterexample
        # 中低失败率 → 方法问题 → pivot
        if _consec >= 10:
            return "counterexample"
        if _consec >= 3:
            return "pivot"
        return "stop"

    def _trigger_counterexample_hunt(self) -> None:
        """P2: 触发反例搜索 (chaoxu 启发).

        两种路径:
        1. SMT 离散反例 (已有 _discrete_counterexample_scan, 需 evidence 带
           discrete_hypothesis 字段)
        2. LLM 主动构造反例 scenario — 让 imagination block 强制开, 下轮
           hypothesize 时 LLM 被要求考虑反事实

        ponytail: 不新起 subagent (贵), 只设 flag 让下轮 _should_imaginate
        返 True + 注入 counterexample hint. 升级: LLM 把 hypothesis 翻译成
        z3 表达式跑 SMT.
        """
        # P0 Task 3: per-hyp budget 检查 — ce 轮数上限, toggle off 时不检查
        if os.environ.get("HUGINN_PER_HYP_BUDGET", "0") == "1":
            try:
                _ce_hyp = getattr(self, "_current_hyp_id_for_plan", None)
                if _ce_hyp:
                    _ce_node = self.hypothesis_graph._nodes.get(_ce_hyp)
                    if _ce_node is not None:
                        _ce_vb = _ce_node.evidence.get("verification_budget")
                        if _ce_vb is not None:
                            _ce_used = _ce_node.evidence.get("ce_rounds_used", 0)
                            if _ce_used >= _ce_vb.get("ce_rounds", 0):
                                _ce_node.evidence["budget_exhausted"] = True
                                logger.debug(
                                    "counterexample hunt skipped: budget_exhausted (hyp=%s)",
                                    _ce_hyp,
                                )
                                return
                            _ce_node.evidence["ce_rounds_used"] = _ce_used + 1
            except Exception:
                logger.debug("per-hyp ce budget check failed", exc_info=True)
        # 强制开 imagination (override _should_imaginate 的判断)
        self._force_imaginate = True
        # 注入 counterexample hint 给下轮 hypothesize
        _cur_hyp = getattr(self, "_current_hyp_id_for_plan", None)
        _stmt = ""
        if _cur_hyp:
            try:
                _node = self.hypothesis_graph._nodes.get(_cur_hyp)
                if _node:
                    _stmt = _node.statement[:200]
            except Exception:
                pass
        _hint = (
            f"Stagnation classified as evidence_against. "
            f"Current hypothesis may be wrong. Hunt for a counterexample.\n"
            f"Hypothesis: {_stmt}\n"
            f"Construct a specific scenario / parameter set where this hypothesis "
            f"would fail. If found, refute and pivot to a corrected hypothesis."
        )
        # _speculator_hint 会被 _build_hypothesis_prompt 读取注入
        self._speculator_hint = (
            (getattr(self, "_speculator_hint", "") or "") + "\n" + _hint
        )
        # Task 4: 拉历史 failure trace exemplar 给 LLM 参考.
        # 复用 recall_failed_directions; 只挑 reason 含 [FAILURE TRACE]/[BREAK POINT]
        # 标记的 (Task 3 反推产物), 旧数据 (简短 error 串) 跳过 — 没推理链, 当 exemplar
        # 没价值. ponytail: recall 接口本身不接受 query, 只返最近 N 条; 用 persona 过滤.
        try:
            _hist_traces: list[str] = []
            _mem = getattr(self, "memory", None)
            if _mem is not None and hasattr(_mem, "recall_failed_directions"):
                _failed = _mem.recall_failed_directions(
                    limit=5,
                    persona_id=getattr(self, "_last_persona", None),
                )
                for _h, _reason, _mc in _failed:
                    if not _reason:
                        continue
                    if "[FAILURE TRACE]" not in _reason and "[BREAK POINT]" not in _reason:
                        continue  # 旧数据: 简短 error 串, 当 exemplar 没价值
                    _hist_traces.append(_reason[:500])
                    if len(_hist_traces) >= 2:
                        break
            if _hist_traces:
                _block = "[HISTORICAL FAILURE TRACES]"
                for _i, _t in enumerate(_hist_traces, 1):
                    _block += f"\n--- {_i} ---\n{_t}"
                self._speculator_hint = (
                    (getattr(self, "_speculator_hint", "") or "") + "\n" + _block
                )
        except Exception:
            logger.debug("recall failed directions for exemplar failed", exc_info=True)
        # P1 Task 8: inject [VERIFIER WEAKNESS] from past blind mismatches
        try:
            _mm = self.memory.recall_typed("verification_mismatch", limit=3)
            if _mm:
                _wl: list[str] = []
                for _m in _mm:
                    _mc = _m.get("content", "") if isinstance(_m, dict) else str(_m)
                    try:
                        _md = json.loads(_mc)
                        _wl.append(
                            f"- hyp: {str(_md.get('hypothesis', '?'))[:100]} "
                            f"(blind={_md.get('blind_holds')}, orig={_md.get('orig_holds')})"
                        )
                    except Exception:
                        _wl.append(f"- {_mc[:100]}")
                if _wl:
                    _wb = (
                        "\n[VERIFIER WEAKNESS] Past cases where blind "
                        "reconstruction disagreed with original reasoning:\n"
                        + "\n".join(_wl[:3]) + "\n"
                    )
                    self._speculator_hint = (
                        (getattr(self, "_speculator_hint", "") or "") + _wb
                    )
        except Exception:
            logger.debug("verifier weakness hint failed", exc_info=True)
        logger.info("P2 counterexample hunt triggered, hint injected")
    def _emit_campaign(self, event_type: str, data: dict) -> None:
        """发布 campaign.* 事件到 EventBus + SSE 流, fire-and-forget.

        双通道: EventBus 给 audit/telemetry, SSE 给前端 IterationTimeline.
        之前只发 EventBus, 前端只能正则刮消息文本, retry/suspect/refine
        根本到不了前端.
        """
        try:
            from huginn.events.integration import _publish
            from huginn.utils.concurrency import track_task

            asyncio.get_running_loop()  # 检测在 event loop 里
            track_task(
                _publish(event_type, data, source="autoloop"), name="campaign-emit"
            )
        except Exception:
            logger.debug("campaign EventBus emit failed", exc_info=True)
        # SSE 推送到 /tasks/stream 的 'campaign' event, 前端结构化消费
        try:
            from huginn.interaction.progress import get_progress_tracker

            get_progress_tracker().emit_campaign_event(
                getattr(self, "_progress_task_id", ""), event_type, data
            )
        except Exception:
            logger.debug("campaign SSE emit failed", exc_info=True)

    def _prepare_run(
        self, objective: str, progressive_budget: bool, goal: Goal | None
    ) -> tuple[str, Any, Any]:
        """Set up run state: provenance, telemetry, budget, speculator."""
        run_id = f"loop_{uuid.uuid4().hex[:8]}"
        self._run_start_time = time.time()
        self._objective = objective

        from huginn.provenance import ProvenanceLogger, ProvenanceRecord

        provenance_logger = ProvenanceLogger(
            self.workspace / ".huginn" / "provenance.jsonl"
        )
        provenance_record = ProvenanceRecord(
            run_id=run_id,
            objective=objective,
            timestamps={"start": datetime.now().isoformat()},
        )
        self._provenance_record = provenance_record
        self._provenance_logger = provenance_logger

        from huginn.telemetry import TelemetryCollector, set_telemetry_collector

        run_collector = TelemetryCollector()
        set_telemetry_collector(run_collector)

        self._iteration = 0
        self._should_stop = False
        self._consecutive_failures = 0
        # F-borrow: 分类计数器随 run 重置 (跨 run 失败模式记忆没意义, 误导自适应).
        self._consecutive_failures_by_type = {}
        # 700 万步场景: 滑动窗口随 run 重置 (跨 run 失败率无意义).
        self._validate_window = []
        # 700 万步场景: 加载上 run 失败模式快照, 让 decider 知道历史卡点.
        # 不恢复计数器 (跨 run 计数无意义), 只注入 prompt 作为参考.
        self._last_run_failure_pattern: str = ""
        try:
            self._last_run_failure_pattern = self._load_failure_pattern()
        except Exception:
            logger.debug("load failure pattern failed", exc_info=True)

        if goal is not None and goal.status == "pending":
            goal.status = "active"
            if self._goal_scheduler is not None:
                self._goal_scheduler.update_goal(goal.id, status="active")

        self._budget = ProgressiveBudget.default() if progressive_budget else None
        self._budget_rejects: dict[str, int] = {}
        self._budget_degraded = False
        # plan_check 状态随 run 重置 — 跨 run 的历史成功率没意义, 会误导自适应.
        # patterns 例外: 跨 run 保留 (失败模式记忆), 加载 workspace 里的历史.
        # extra_keywords 也跨 run 保留 (自动发现的 scene_tag 关键词).
        self._plan_check_history = []
        self._plan_check_last_result = None
        self._plan_check_warnings = []
        self._plan_check_patterns = []
        self._scene_tag_extra_keywords = {}
        self._load_plan_check_patterns()

        self._speculator_hint = ""
        self._last_visual_context = ""  # reset per run, stale data shapes mislead
        self._current_prediction = ""  # reset JEPA prediction buffer
        self._last_surprise = 0.0
        self._last_raw_hypothesis = ""  # 完整 LLM 输出, 含 LUCID review
        # G2: 加载历史 trajectory action 序列, 给 _check_stuck 当 VF2 匹配历史.
        # 失败/空都不影响 run, 只是少了 cross-run 匹配能力.
        try:
            self._traj_history = self._load_trajectory_action_history(limit=20)
        except Exception:
            self._traj_history = []
            logger.debug("G2 traj history load failed (non-fatal)", exc_info=True)
        try:
            from huginn.agents.speculator import on_turn_start

            spec_result = on_turn_start(objective)
            self._speculator_hint = spec_result.get("hint", "")
            if spec_result.get("predictions"):
                logger.info("autoloop speculator: %s", self._speculator_hint)
        except Exception:
            logger.warning("autoloop speculator skipped", exc_info=True)

        return run_id, provenance_record, run_collector

    def _persist_failure_pattern(self, run_id: str) -> None:
        """run 结束时把 by_type + window 快照存 longterm. 供下 run 加载.

        700 万步场景: 单 run 可能只跑几十万步, 跨 run 失败模式记忆让 decider
        知道"上次主要卡在 tool_error" → 这次优先换工具/换 backend.
        ponytail: 复用 longterm.store, JSON 序列化. 升级路径: 独立 failure_pattern 表.
        """
        by_type = getattr(self, "_consecutive_failures_by_type", {}) or {}
        vwin = getattr(self, "_validate_window", None) or []
        # 只在有失败数据时存 — 全 pass 的 run 存了也没参考价值.
        if not by_type and not vwin:
            return
        wsize = getattr(self, "_validate_window_size", 100)
        fail_rate = 1.0 - (sum(vwin) / len(vwin)) if vwin else 0.0
        snapshot = {
            "run_id": run_id,
            "by_type": by_type,
            "window_size": len(vwin),
            "window_fail_rate": round(fail_rate, 3),
            "total_consecutive": getattr(self, "_consecutive_failures", 0),
            "objective": (getattr(self, "_objective", "") or "")[:200],
        }
        try:
            content = json.dumps(snapshot, ensure_ascii=False)
            self.memory.remember(
                content=content,
                category="failure_pattern",
                tags=["failure_pattern", run_id],
                importance=0.6,
                tier="mid",
            )
        except Exception:
            logger.debug("failure_pattern store failed", exc_info=True)

    def _load_failure_pattern(self) -> str:
        """加载最近一条 failure_pattern, 返回人类可读摘要供 decider prompt 注入.

        返回空串表示无历史或加载失败. ponytail: 只取最近 1 条, 不做聚合.
        用空 query + category 过滤 — content 是 JSON, FTS5 语义匹配不到.
        """
        try:
            results = self.memory.recall(
                query="",
                category="failure_pattern",
                top_k=1,
            )
        except Exception:
            return ""
        if not results:
            return ""
        entry = results[0] if isinstance(results, list) else results
        content = entry.get("content", "") if isinstance(entry, dict) else str(entry)
        if not content:
            return ""
        try:
            snap = json.loads(content)
        except (ValueError, TypeError):
            return ""
        by_type = snap.get("by_type", {}) or {}
        if not by_type:
            return ""
        parts = [f"{k}={v}" for k, v in sorted(by_type.items()) if v > 0]
        if not parts:
            return ""
        rate = snap.get("window_fail_rate", 0.0)
        return (
            f"last run: {', '.join(parts)}, "
            f"window fail rate={rate:.2f} "
            f"(n={snap.get('window_size', 0)})"
        )
    async def _decide_next_action_llm(
        self, state: LoopState, cog: dict, obs: dict,
    ) -> ActionDecision | None:
        """LLM 自主选 next action. 失败/非法返回 None, 调用方 fallback 到规则版.

        上下文: iteration / last_action / hypothesis/plan/execution/validation 状态.
        输出: JSON {"action", "rationale", "expected_outcome"}.
        合法性: 没hyp不能plan, 没plan不能execute, etc. 不合法 → None.
        """
        prompt = self._build_decider_prompt(state, cog, obs)
        try:
            raw = await self._llm_chat(prompt, persona_name="reviewer", task="reasoning")
        except Exception as e:
            logger.debug("decider LLM call failed: %s", e)
            return None
        if not raw:
            return None
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(raw[start:end + 1])
        except Exception:
            return None
        action = str(data.get("action", "")).strip().lower()
        if action not in VALID_ACTIONS:
            logger.debug("decider returned invalid action %r", action)
            return None
        if not self._is_action_legal(action, cog):
            logger.info("decider illegal action %s (preconditions not met), fallback", action)
            return None
        return ActionDecision(
            action=action,
            rationale=str(data.get("rationale", ""))[:200],
            expected_outcome=str(data.get("expected_outcome", ""))[:200],
        )

    def _build_decider_prompt(
        self, state: LoopState, cog: dict, obs: dict,
    ) -> str:
        """简短 prompt — 给 LLM 控制流状态, 不重复 phase 内部细节.

        D1: 扩字段 — validation details / consecutive_failures / pivot_count /
        refine_count / action_history / speculator_hint / last_learn_summary.
        之前 LLM 只看到 5 个粗投影, 现在给 belief state 多几维.
        ponytail: 不重组结构, 只在 State block 后追加 Detail block. 升级路径:
        把这些字段做成 PhaseRegistry extra, 跟 H4 phase 分批一致.
        """
        hyp = cog.get("hypothesis") or "NONE"
        if isinstance(hyp, str) and len(hyp) > 120:
            hyp = hyp[:120]
        plan = cog.get("plan")
        plan_mode = plan.get("mode", "NONE") if isinstance(plan, dict) else "NONE"
        exec_done = cog.get("execution_result") is not None
        val = cog.get("validation") or {}
        val_status = "PASSED" if _extract_tests_passed(val) else "FAILED" if val else "NONE"

        # D1: validation 具体字段 — 让 LLM 看到为什么 PASSED/FAILED
        val_detail = ""
        if isinstance(val, dict) and val:
            _vd_keys = (
                "thinking_collapse", "physics_validation_error",
                "reviewer_critique", "dimensional_consistent",
                "pde_classification", "constraint_check",
            )
            _vd_parts = []
            for _k in _vd_keys:
                _v = val.get(_k)
                if _v:
                    _vd_parts.append(f"{_k}={str(_v)[:80]}")
            if _vd_parts:
                val_detail = "; ".join(_vd_parts)[:300]

        # D1: 控制流统计 — 让 LLM 感知 stuck 程度
        # getattr 防 __new__ 测试场景 (selfcheck 绕过 __init__).
        _max_fail = getattr(self, "_max_consecutive_failures", 20)
        _max_pivot = 10  # ponytail: 硬编码上限, 升级路径走 PhaseRegistry extra
        action_hist_str = ", ".join(state.action_history[-10:]) or "none"
        spec_hint = (getattr(self, "_speculator_hint", "") or "")[:300]
        last_learn = (cog.get("last_learn_summary") or "none")[:200]
        # F-borrow: 分类失败计数 — 让 LLM 看到是 tool_error 多还是 hypothesis_error 多,
        # 不同失败类型语义不同 (技术故障 vs 方向错). 空时不显示, 避免噪声.
        _by_type = getattr(self, "_consecutive_failures_by_type", {}) or {}
        _type_max = getattr(self, "_max_failures_by_type", {}) or {}
        _type_parts = [
            f"{k}={v}/{_type_max.get(k, '?')}"
            for k, v in sorted(_by_type.items())
            if v > 0
        ]
        _type_str = ", ".join(_type_parts) if _type_parts else "none"
        # 700 万步场景: 滑动窗口失败率. 让 LLM 看到整体趋势, 不只是 consecutive.
        # 窗口未满时不显示 (数据不足). ponytail: 简单 fail rate, 不做加权.
        _vwin = getattr(self, "_validate_window", None) or []
        _wsize = getattr(self, "_validate_window_size", 100)
        if len(_vwin) >= _wsize:
            _wfail_rate = 1.0 - (sum(_vwin) / len(_vwin))
            _window_str = f"{_wfail_rate:.2f} (last {_wsize})"
        else:
            _window_str = f"building ({len(_vwin)}/{_wsize})"
        # 700 万步场景: 跨 run 失败模式. 让 LLM 知道上 run 主要卡在哪类失败.
        _last_pattern = (getattr(self, "_last_run_failure_pattern", "") or "").strip()

        return f"""You are the cognitive controller of a research agent. Choose the next action.

Iteration: {state.iteration}/{state.max_iterations}
Last action: {state.last_action or 'NONE'}
Last rationale: {state.last_rationale or 'none'}

State:
- Hypothesis: {hyp}
- Plan mode: {plan_mode}
- Execution: {'DONE' if exec_done else 'NONE'}
- Validation: {val_status}
- Consecutive failures: {getattr(self, "_consecutive_failures", 0)}/{_max_fail}
- Failures by type: {_type_str}
- Window fail rate: {_window_str}
- Last run pattern: {_last_pattern or 'none'}
- Pivot count: {getattr(self, "_pivot_count", 0)}/{_max_pivot}
- Refine count: {getattr(self, "_refine_count", 0)}
- Action history (last 10): {action_hist_str}
- Last learn summary: {last_learn}

Validation details: {val_detail or 'none'}
Speculator hints: {spec_hint or 'none'}
Last reflection advice: {state.redirect_reason or 'none'}

Actions:
- observe: re-perceive environment (context stale / need fresh data)
- hypothesize: generate new hypothesis (no hypothesis / after pivot)
- plan: design execution plan for current hypothesis
- execute: run the plan
- validate: check execution results
- learn: update memory/KG with results
- pivot: switch to new hypothesis (current path stuck)
- skip: do nothing this iteration
- stop: end the loop

Note: report runs automatically when loop ends — do not pick "report".

Respond JSON only:
{{"action": "one of above", "rationale": "1 sentence why", "expected_outcome": "1 sentence what you expect"}}"""

    def _is_action_legal(self, action: str, cog: dict) -> bool:
        """LLM 选择的 action 是否合法 (前置条件满足)."""
        if action in ("observe", "hypothesize", "skip", "stop"):
            return True
        # D3: report 不让 LLM 选 — _finalize_run 自动跑. LLM 选 report 等于
        # 浪费一轮 (execute_fn 是 no-op). 升级路径: 如果要 LLM 主动触发,
        # 改成 action="stop" + rationale="report ready".
        if action == "report":
            return False
        if action == "plan":
            return bool(cog.get("hypothesis"))
        if action == "execute":
            return bool(cog.get("plan"))
        if action == "validate":
            return cog.get("execution_result") is not None
        if action == "learn":
            return all(cog.get(k) for k in ("hypothesis", "plan", "validation"))
        if action == "pivot":
            return bool(cog.get("current_hyp_id") or cog.get("hypothesis"))
        return False
    async def _finalize_run(
        self,
        objective: str,
        phases: list[LoopPhase],
        run_id: str,
        provenance_record: Any,
        run_collector: Any,
        tracker: Any,
        progress_task_id: str,
        completed_steps: int,
    ) -> AutoloopResult:
        """Report, save trajectory, judge goal, write provenance + FAIR metadata."""
        # 700 万步场景: 失败模式跨 run 持久化. run 结束时存 by_type + window 快照,
        # 下个 run 开始时加载, 让 LLM 知道"上次主要卡在 tool_error 还是 hypothesis_error".
        # ponytail: 复用 longterm.store, JSON 序列化. 升级路径: 独立 failure_pattern 表.
        try:
            self._persist_failure_pattern(run_id)
        except Exception:
            logger.debug("persist failure pattern failed", exc_info=True)
        total_time = time.time() - getattr(self, "_run_start_time", time.time())
        report_phase = await self._run_phase_async(
            "report", self._report, objective, phases, total_time
        )
        phases.append(report_phase)
        completed_steps += 1
        tracker.update(
            progress_task_id,
            current_step=completed_steps,
            current_label=f"report ({report_phase.status})",
        )

        if report_phase.status == "completed":
            tracker.complete(
                progress_task_id, result={"report_path": report_phase.result}
            )
        else:
            tracker.fail(progress_task_id, f"report phase failed: {report_phase.error}")

        # session summary → long-term memory
        try:
            self.memory.promote_session_summary(tier="long")
        except Exception:
            logger.debug("session summary promotion failed", exc_info=True)

        # trajectory
        trajectory_path = None
        trajectory_data = None
        try:
            from huginn.telemetry import load_trajectory, save_trajectory

            traj_dir = self.workspace / ".huginn" / "trajectories"
            trajectory_path = traj_dir / f"{run_id}.json"
            save_trajectory(
                run_collector,
                trajectory_path,
                metadata={
                    "run_id": run_id,
                    "objective": objective[:200],
                    "phases": [p.name for p in phases],
                    "total_time": total_time,
                },
            )
            trajectory_data = load_trajectory(trajectory_path)
            # G31: trajectory tool_calls 断言 — 修 audit 06 F1
            # (18/18 轨迹 0 工具调用, perceive+report 两阶段空转).
            # 非查询 objective + 0 工具调用 = autoloop 装置未激活, 记 warning.
            _tc = trajectory_data.get("tool_calls", []) if trajectory_data else []
            _obj_lower = (objective or "").strip().lower()
            _is_query = _obj_lower.startswith("query") or _obj_lower.startswith("read")
            if not _tc and objective and not _is_query:
                logger.warning(
                    "G31: trajectory has 0 tool_calls for non-query objective "
                    "'%s' — autoloop may be空转 (audit 06 F1)",
                    objective[:100],
                )
        except Exception:
            trajectory_path = None

        # goal judgment
        goal_achieved = None
        goal_judgment = None
        try:
            from huginn.evaluation.goal_judge import GoalJudge

            final_output = str(report_phase.result or "")
            judge = GoalJudge(llm=self.verification_model or self.model)
            goal_judgment = judge.judge(
                objective=objective,
                trajectory=trajectory_data,
                final_output=final_output,
            )
            goal_achieved = goal_judgment.get("achieved")
        except Exception:
            logger.warning("autoloop goal judge skipped", exc_info=True)

        # provenance
        provenance_path = None
        try:
            provenance_record.timestamps["end"] = datetime.now().isoformat()
            self._provenance_logger.log(provenance_record)
            provenance_path = str(self._provenance_logger.path)
        except Exception:
            provenance_path = None

        # FAIR metadata
        try:
            from huginn.export.fair_metadata import (
                generate_dataset_metadata,
                write_fair_jsonld,
            )

            run_results: dict[str, Any] = {}
            for ph in phases:
                if ph.result and isinstance(ph.result, dict):
                    run_results.update(ph.result)
            fair_metadata = generate_dataset_metadata(
                run_id=run_id,
                objective=objective,
                results=run_results,
                provenance={
                    "report_path": (
                        str(report_phase.result) if report_phase.result else None
                    ),
                    "trajectory_path": (
                        str(trajectory_path) if trajectory_path else None
                    ),
                    "provenance_path": provenance_path,
                    "start_time": provenance_record.timestamps.get("start"),
                    "end_time": provenance_record.timestamps.get("end"),
                },
            )
            jsonld_path = self.workspace / f"{run_id}_dataset.jsonld"
            write_fair_jsonld(fair_metadata, jsonld_path)
            logger.info("FAIR JSON-LD written to %s", jsonld_path)
        except Exception:
            logger.debug("FAIR metadata generation failed", exc_info=True)

        # P2: trajectory success pattern 抽取 — 复用 KB + auto_ingest 路径
        # 不新建 skill_library 组件. 仅在 goal_achieved=True 时调一次 LLM 抽
        # 可复用 pattern, 写入 KB. 下次任务开始时 RAG 自然召回.
        # 默认关, HUGINN_TRAJECTORY_PATTERN=1 开启 (跟 PRT Level 1 / PRM verifier
        # 同款策略: 有 LLM 成本).
        if goal_achieved and os.environ.get("HUGINN_TRAJECTORY_PATTERN", "0") == "1":
            try:
                from huginn.knowledge.trajectory_pattern import (
                    extract_and_store_pattern,
                )

                async def _pattern_chat(prompt: str) -> str:
                    # 复用 verification_model (默认 fallback 到 self.model)
                    from langchain_core.messages import HumanMessage
                    resp = await self.verification_model.ainvoke(
                        [HumanMessage(content=prompt)]
                    )
                    return getattr(resp, "content", str(resp))

                pattern_doc_id = await extract_and_store_pattern(
                    objective=objective,
                    trajectory=trajectory_data,
                    final_output=str(report_phase.result or ""),
                    llm_chat_fn=_pattern_chat,
                    run_id=run_id,
                )
                if pattern_doc_id:
                    logger.info(
                        "trajectory pattern stored: doc_id=%s (run %s)",
                        pattern_doc_id, run_id,
                    )
            except Exception:
                logger.debug(
                    "trajectory pattern extraction failed (non-fatal)",
                    exc_info=True,
                )

        return AutoloopResult(
            run_id=run_id,
            objective=objective,
            phases=phases,
            success=all(p.status == "completed" for p in phases[-7:]),
            report_path=report_phase.result,
            total_time_seconds=total_time,
            trajectory_path=str(trajectory_path) if trajectory_path else None,
            goal_achieved=goal_achieved,
            goal_judgment=goal_judgment,
            provenance_path=provenance_path,
            merged_graph=self._merged_graph,
            speculator_hint=self._speculator_hint,
        )
    async def run_cognitive(
        self,
        objective: str,
        max_iterations: int = 50,
        progressive_budget: bool = True,
        goal: Goal | None = None,
        max_refines: int = 8,
        timeout_seconds: float | None = None,
    ) -> AutoloopResult:
        """CognitiveLoop 入口 — 用 4 钩子编排 7-phase.

        返回 AutoloopResult, 与 run() 接口一致, 调用方无需感知差异.
        """
        from huginn.autoloop.cognitive_loop import (
            CognitiveLoop, LoopState, ActionDecision, ReflectionResult,
        )

        self._max_refines = max_refines
        self._refine_count = 0
        self._max_iterations = max_iterations
        # AV2: 每次新 run 重置元认知护航状态 (避免跨 run 串味)
        self._evals_history = []
        self._task_metrics = None
        self._task_state_for_metrics = None
        self._drift_info = None
        get_shared_phase_gate_state().reset_runtime()
        run_id, provenance_record, run_collector = self._prepare_run(
            objective, progressive_budget, goal
        )
        self._run_id = run_id
        self._parent_run_id = None
        tracker = get_progress_tracker()
        total_steps = max_iterations * 6 + 1
        progress_task_id = f"autoloop:{run_id}"
        tracker.start_task(
            task_id=progress_task_id,
            description=f"autoloop: {objective[:80]}",
            total_steps=total_steps,
            stage_labels=list(AUTOLOOP_PHASES),
            engine_kind="autoloop",
            metadata={"run_id": run_id, "objective": objective[:200]},
            timeout_seconds=timeout_seconds,
        )
        self._progress_task_id = progress_task_id

        # P0-2: bridge progress_cb → _emit_campaign. autoloop 路径不走 _stream_agent_response,
        # progress_cb 默认 None, 导致 subagent_tool._on_state 早 return (cb is None).
        # 这里 set 一个桥: subagent_event / autoloop_thinking 等事件 → campaign SSE.
        # 外层已 set (WS 路径嵌入 autoloop) 则不覆盖. ponytail: 复用 contextvar, 不新开通道.
        # 不显式 reset — run_cognitive 通常被 asyncio.create_task 包, contextvar 跟 task 同生命周期.
        from huginn.types import progress_cb as _progress_cb
        if _progress_cb.get(None) is None:
            _autoloop_engine = self

            async def _progress_bridge(msg: dict) -> None:
                _etype = msg.get("type", "progress")
                _data = {k: v for k, v in msg.items() if k != "type"}
                _data.setdefault("run_id", run_id)
                _autoloop_engine._emit_campaign(f"campaign.{_etype}", _data)

            _progress_cb.set(_progress_bridge)

        # phase 间传递的中间结果 — 不放 LoopState (那是控制流状态)
        cog: dict[str, Any] = {
            "context": {},
            "hypothesis": None,
            "plan": None,
            "execution_result": None,
            "validation": None,
            "current_hyp_id": None,
            "phases": [],
            "completed_steps": 0,
        }

        async def observe_fn(state: LoopState) -> dict[str, Any]:
            # v10: 外部 stop() 设 self._should_stop, 同步到 state.should_stop
            # 让 CognitiveLoop while guard 能感知. 否则 stop() 对 run_cognitive 无效.
            if getattr(self, "_should_stop", False):
                state.should_stop = True
                return {
                    "context_summary": "",
                    "redirect_reason": state.redirect_reason,
                    "iteration": state.iteration,
                    "last_action": state.last_action,
                    "external_stop": True,
                }
            # P1.4: 每轮开头发 campaign.iteration — 对齐 run() L1305.
            # 前端 IterationTimeline 依赖这个事件渲染轮次进度.
            self._emit_campaign(
                "campaign.iteration",
                {
                    "iteration": state.iteration,
                    "max": max_iterations,
                    "objective": objective[:200],
                },
            )
            # v10-F1+F7: goal 持久化 + budget 硬停 — 对齐 run() L1272-1304.
            # 每轮 increment_iteration, is_budget_exhausted → fail_goal + should_stop.
            # ponytail: spec 把 F1 放 reflect / F7 放 observe, 但 increment 和 budget
            #   check 必须原子 (不原子会读 stale iteration), 这里合并到 observe 开头.
            #   spec F7 阶段 2 只剩 build_continuation_prompt / drain_side.
            try:
                from huginn.autoloop.goal_store import get_goal_store

                _gs = get_goal_store()
                _active_goal = _gs.get_active()
                if _active_goal:
                    _gs.increment_iteration(_active_goal.id)
                    if GoalScheduler.is_budget_exhausted(_active_goal):
                        logger.info(
                            "v10 goal budget exhausted: iter=%d max=%d, failing %s",
                            _active_goal.iteration, _active_goal.max_iterations, _active_goal.id,
                        )
                        try:
                            _gs.fail_goal(
                                _active_goal.id,
                                reason=f"budget exhausted: {_active_goal.iteration}/{_active_goal.max_iterations}",
                            )
                        except Exception:
                            logger.debug("fail_goal failed (non-fatal)", exc_info=True)
                        self._emit_campaign(
                            "campaign.budget_exhausted",
                            {
                                "iteration": state.iteration,
                                "goal_id": _active_goal.id,
                                "budget": _active_goal.max_iterations,
                                "used": _active_goal.iteration,
                            },
                        )
                        state.should_stop = True
                        return {
                            "context_summary": "",
                            "redirect_reason": state.redirect_reason,
                            "iteration": state.iteration,
                            "last_action": state.last_action,
                            "budget_exhausted": True,
                        }
            except Exception:
                logger.debug("v10 goal increment/budget failed (non-fatal)", exc_info=True)

            # v10-F6: build_continuation_prompt — 对齐 run() L1321-1331.
            # goal 非空且 iteration > 1 时拼续跑提示到 speculator_hint, 让 LLM
            # 看到自己在续跑而非从头开始. 第 1 轮不拼 (算首次).
            if goal is not None and state.iteration > 1:
                try:
                    _cont = GoalScheduler.build_continuation_prompt(goal)
                    if _cont:
                        self._speculator_hint = (
                            (self._speculator_hint + "\n" + _cont).strip()
                            if self._speculator_hint else _cont
                        )
                except Exception:
                    logger.debug("v10 F6 build_continuation_prompt failed (non-fatal)", exc_info=True)

            # _perceive 是 sync (跑 git subprocess + rglob), 丢线程池不阻塞
            # v10: 记本轮 perceive 是否返回空, F8 用这个 flag 而非 cog["context"]
            # (cog["context"] 跨轮持久, 上轮 forced/residual 会掩盖本轮空感知).
            _perceived_empty = True
            try:
                if self._next_phase_hint not in ("plan", "execute"):
                    ctx = await asyncio.to_thread(self._perceive)
                    if ctx:
                        cog["context"] = ctx
                        _perceived_empty = False
                else:
                    # hint=plan/execute 时跳过 perceive, 不算空 ( intentional skip)
                    _perceived_empty = False
            except Exception as e:
                logger.warning("cognitive observe failed: %s", e)

            # v10-F15: G31 bypass — 对齐 run() L1369-1382.
            # perceive 返回空 + 首轮 + objective 存在 → 强制注入 minimal context,
            # 避免首轮轮空导致 18/18 轨迹 perceive+report 两阶段 0 工具调用.
            # ponytail: 仅首轮 bypass, 后续轮走 F8 drain_side.
            if _perceived_empty and state.iteration == 1 and objective:
                logger.info(
                    "v10 G31: perceive empty on iter 1 with objective, forcing hypothesize"
                )
                cog["context"] = {
                    "forced": True,
                    "objective": objective,
                    "note": "perceive returned empty, G31 forces hypothesize",
                }

            # v10-F16: timeout 硬停 (observe 阶段) — 对齐 run() L1248.
            # reflect_fn 已有 timeout 检查, 这里加 observe 阶段检查让 timeout 更早触发,
            # 不必跑完当前轮的 decide/execute/reflect.
            # ponytail: tracker.is_expired 是 O(1) 字典查, 不阻塞.
            if tracker.is_expired(progress_task_id):
                logger.info("v10 F16 timeout expired in observe, stopping")
                state.should_stop = True
                return {
                    "context_summary": "",
                    "redirect_reason": state.redirect_reason,
                    "iteration": state.iteration,
                    "last_action": state.last_action,
                    "timeout_expired": True,
                }

            # v10-F8: drain_side_questions — 对齐 run() L1384-1387.
            # run() 在 perceive 返回空 (轮空) 时调 _drain_side_questions + continue.
            # run_cognitive 不能 continue (CognitiveLoop 每轮要走完 4 钩子), 改为
            # perceive 返回空时顺手答 pending 侧边问题, 不跳过本轮.
            # ponytail: 用 _perceived_empty 而非 not cog["context"], 避免上轮
            # forced/residual context 掩盖本轮空感知.
            if _perceived_empty:
                try:
                    _n_drained = await self._drain_side_questions()
                    if _n_drained:
                        logger.info("v10 F8 drained %d side questions", _n_drained)
                except Exception:
                    logger.debug("v10 F8 drain_side_questions failed (non-fatal)", exc_info=True)

            # v10-F5: blind_spot_pass — 对齐 run() L1391-1402.
            # spec F5 描述 "强制 stop" 不准, run() 实际行为是注入 context + 写 GoalStore.unknowns.
            # ponytail: 每隔 5 轮做一次, 避免 token 浪费. 异步 LLM 调用.
            if state.iteration == 1 or state.iteration % 5 == 0:
                try:
                    _bs = await self._blind_spot_pass(cog["context"] or {}, self._objective)
                    if _bs:
                        cog["context"]["blind_spots"] = _bs
                        logger.info("v10 blind spot pass: %d unknowns", len(_bs))
                except Exception:
                    logger.debug("v10 blind_spot_pass failed (non-fatal)", exc_info=True)

            return {
                "context_summary": (cog["context"] or {}).get("summary", ""),
                "redirect_reason": state.redirect_reason,
                "iteration": state.iteration,
                "last_action": state.last_action,
            }

        async def decide_fn(state: LoopState, obs: dict[str, Any]) -> ActionDecision:
            # Step C: LLM 自主选 action (优先) → 规则版兜底
            # redirect / hint 仍走规则版 — 死循环防护和上轮 reflect 的明确建议不交给 LLM.
            # 首轮 (last in ("", "skip")) 不调 LLM, 直接走规则版 hypothesize.
            if state.should_redirect:
                state.should_redirect = False
                # 没 hyp 可以 pivot → 直接停, 避免 pivot 空转死循环
                if not cog.get("current_hyp_id") and not cog.get("hypothesis"):
                    return ActionDecision(action="stop", rationale="no hyp to pivot from")
                return ActionDecision(action="pivot", rationale=f"redirect: {state.redirect_reason}")
            hint = self._next_phase_hint
            if hint == "execute" and self._refined_hypothesis:
                cog["hypothesis"] = self._refined_hypothesis
                return ActionDecision(action="execute", rationale="refine reuse")
            if hint == "plan":
                return ActionDecision(action="plan", rationale="hint=plan")
            if hint == "perceive":
                return ActionDecision(action="observe", rationale="hint=perceive")
            # LLM 自主决策 (开启时, 且非首轮). 失败/非法 → fallback 到规则版
            if self._use_llm_decider and state.last_action not in ("", "skip"):
                try:
                    llm_decision = await self._decide_next_action_llm(state, cog, obs)
                    if llm_decision is not None:
                        return llm_decision
                except Exception as e:
                    logger.debug("LLM decider failed: %s, fallback to rule", e)
            # 规则版兜底: 默认 7-phase 顺序
            last = state.last_action
            if last in ("", "observe", "pivot", "skip"):
                return ActionDecision(action="hypothesize", rationale="seq→hyp")
            if last == "hypothesize":
                if not cog["hypothesis"]:
                    return ActionDecision(action="observe", rationale="no hyp, re-observe")
                return ActionDecision(action="plan", rationale="seq→plan")
            if last == "plan":
                if not cog["plan"]:
                    return ActionDecision(action="hypothesize", rationale="no plan, re-hyp")
                return ActionDecision(action="execute", rationale="seq→exec")
            if last == "execute":
                if cog["execution_result"] is None:
                    return ActionDecision(action="plan", rationale="exec None, re-plan")
                return ActionDecision(action="validate", rationale="seq→validate")
            if last == "validate":
                # v10: 规则版总是推进到 learn, 让 validate→learn gate 决定放行/阻断.
                # 对齐 run() 顺序执行语义 (run() 不因 tests_passed=False 跳过 learn,
                # 而是让 gate 评估 evidence). LLM decider 可智能选 re-execute.
                # ponytail: 规则版是 fallback, 不做智能判断.
                return ActionDecision(action="learn", rationale="seq→learn")
            if last == "learn":
                # v10: cycle 回 hypothesize 而非 stop — 对齐 run() while 循环自然
                # 进入下一 iter 的行为. stop 由 max_iter / should_stop / F3 darwin
                # / F4 surprise / F2 completion / F17 GoalJudge 触发, 不靠规则版.
                # ponytail: cycling 是 rule-based fallback 的语义, LLM decider 不受此影响.
                return ActionDecision(action="hypothesize", rationale="cycle→hyp")
            return ActionDecision(action="stop", rationale=f"unknown last {last}")

        async def execute_fn(state: LoopState, decision: ActionDecision) -> Any:
            action = decision.action
            ctx = cog["context"]
            self._iteration = state.iteration
            try:
                if action == "observe":
                    phase = await self._run_phase_async(
                        "perceive", lambda: asyncio.to_thread(self._perceive)
                    )
                    cog["phases"].append(phase)
                    cog["completed_steps"] += 1
                    if phase.result:
                        cog["context"] = phase.result
                    return phase.result
                if action == "hypothesize":
                    # v11: FDE 对齐轮 — hypothesize 前问用户方向 (首轮/有 blind_spots).
                    # 不阻塞, 60s timeout, 用户回答 append 到 _speculator_hint.
                    # ponytail: 复用 _maybe_clarify 管道, 不新增 phase.
                    try:
                        await self._maybe_clarify(
                            "hypothesize_align", ctx, thread_id="autoloop",
                        )
                    except Exception:
                        logger.debug("v11 FDE hypothesize_align failed (non-fatal)", exc_info=True)
                    phase = await self._run_phase_async(
                        "hypothesize", self._hypothesize, ctx
                    )
                    cog["phases"].append(phase)
                    cog["completed_steps"] += 1
                    cog["hypothesis"] = phase.result
                    if phase.result:
                        try:
                            cog["current_hyp_id"] = self.hypothesis_graph.add_hypothesis(
                                statement=phase.result,
                                rationale=ctx.get("summary", ""),
                            )
                            self._current_hyp_id_for_plan = cog["current_hyp_id"]
                        except Exception:
                            logger.debug("hypothesis_graph add failed", exc_info=True)
                    # P0 Task 3: per-hyp 验证预算 — 创建时评估 informativeness + 分配 budget
                    # toggle off 时跳过 (向后兼容, 不消耗 LLM 调用)
                    if (
                        os.environ.get("HUGINN_PER_HYP_BUDGET", "0") == "1"
                        and cog.get("current_hyp_id")
                    ):
                        try:
                            _info = await self._evaluate_informativeness(
                                cog["current_hyp_id"]
                            )
                            self._compute_verification_budget(
                                cog["current_hyp_id"],
                                _info["expected_informativeness"],
                            )
                        except Exception:
                            logger.debug("per-hyp budget eval failed", exc_info=True)
                    # P1.4: campaign SSE 对齐 run() L1435
                    self._emit_campaign(
                        "campaign.hypothesis",
                        {
                            "iteration": state.iteration,
                            "hypothesis": str(phase.result or "")[:300],
                        },
                    )
                    return phase.result
                if action == "plan":
                    if not cog["hypothesis"]:
                        return None
                    phase = await self._run_phase_async(
                        "plan", self._plan, cog["hypothesis"], ctx
                    )
                    cog["phases"].append(phase)
                    cog["completed_steps"] += 1
                    cog["plan"] = phase.result
                    return phase.result
                if action == "execute":
                    if not cog["plan"]:
                        return None
                    self._current_prediction = cog["plan"].get("expected_prediction", "")
                    # v10: 下沉 run() L1493+L1497 budget + gate 检查到 execute_fn.
                    # spec 漏列, 但没有这俩 check, budget tier / phase gate 在
                    # run_cognitive 路径完全失效. ponytail: check 失败不抛,
                    # 写 hint 让下轮 decide 看到, 当前 return None 跳过 execute.
                    _plan = cog["plan"]
                    if not self._check_budget(state.iteration, _plan):
                        # budget 拒: hint 已被 _check_budget 写, 这里不重复
                        return None
                    if not self._check_gate(
                        "plan", "execute",
                        {"mode": _plan.get("mode"), "description": _plan.get("description")},
                    ):
                        # gate 阻断: 写 hint 让 LLM 下轮改 plan
                        self._speculator_hint += (
                            "\n[gate: plan→execute blocked] "
                            + str(self.phase_gate_hook.evaluate(
                                "plan", "execute",
                                {"mode": _plan.get("mode"), "description": _plan.get("description")},
                            ).feedback or "")
                            + "\n"
                        )
                        await self._wait_if_checkpoint_pending("plan", "execute")
                        return None
                    phase = await self._run_phase_async(
                        "execute", self._execute, cog["plan"], ctx
                    )
                    cog["phases"].append(phase)
                    cog["completed_steps"] += 1
                    cog["execution_result"] = phase.result
                    # v10: 下沉 run() L1567-1577 plan 完成标记.
                    _plan_id = cog["plan"].get("plan_id") if isinstance(cog["plan"], dict) else None
                    if _plan_id:
                        try:
                            _store = self._get_plan_store()
                            if _store is not None:
                                _store.complete_plan(_plan_id)
                        except Exception:
                            logger.warning("v10 complete_plan failed (non-fatal)", exc_info=True)
                    # git commit after execute (同 run(): 让下轮 perceive 看到 diff)
                    await asyncio.to_thread(self._git_commit_after_execute,
                                            cog["plan"], state.iteration)
                    # P1.4: execute 失败 (无 phase.result) → campaign.retry 对齐 run() L1651
                    if phase.result is None:
                        self._emit_campaign(
                            "campaign.retry",
                            {
                                "iteration": state.iteration,
                                "reason": "execute returned None",
                            },
                        )
                    return phase.result
                if action == "validate":
                    if cog["execution_result"] is None:
                        return None
                    phase = await self._run_phase_async(
                        "validate", self._validate, cog["execution_result"]
                    )
                    cog["phases"].append(phase)
                    cog["completed_steps"] += 1
                    cog["validation"] = phase.result
                    # P1.4: validate 失败 → campaign.suspect 对齐 run() L1668
                    _val = phase.result or {}
                    if not _extract_tests_passed(_val):
                        self._emit_campaign(
                            "campaign.suspect",
                            {
                                "iteration": state.iteration,
                                "reason": str(_val.get("thinking_collapse")
                                              or _val.get("physics_validation_error")
                                              or "tests_failed")[:200],
                            },
                        )
                    return phase.result
                if action == "learn":
                    if not all(cog.get(k) for k in ("hypothesis", "plan", "validation")):
                        return None
                    # v10: 下沉 run() L1859 validate→learn gate 检查.
                    _val = cog["validation"] or {}
                    _exec = cog.get("execution_result") if isinstance(cog.get("execution_result"), dict) else {}
                    _gate_evidence = {k: _val[k] for k in (
                        "tests_passed", "reviewer_critique", "thinking_collapse",
                        "physics_validation_error", "dimensional_consistent",
                        "pde_classification", "sobol_top_features",
                        "constraint_check", "literature_claims",
                    ) if k in _val}
                    if isinstance(_exec.get("physics_audit"), dict):
                        _gate_evidence["physics_audit"] = _exec["physics_audit"]
                    if not self._check_gate("validate", "learn", _gate_evidence):
                        await self._wait_if_checkpoint_pending("validate", "learn")
                        return None
                    phase = await self._run_phase_async(
                        "learn", self._learn,
                        cog["hypothesis"], cog["plan"], cog["validation"],
                    )
                    cog["phases"].append(phase)
                    cog["completed_steps"] += 1
                    # D2: learn 写 cog, 让下轮 decider 看到正反馈. 之前 learn
                    # 是哑 action (不更新 cog), LLM 选了 learn 没反馈, 下轮
                    # 容易重复 learn 或乱选. ponytail: 只塞 1 行摘要, 不暴露
                    # _learn 完整内部状态. 升级路径: 结构化 summary 走 cog dict.
                    _learned = phase.result if isinstance(phase.result, dict) else {}
                    if _learned:
                        cog["last_learn_summary"] = (
                            f"learned: persona={_learned.get('persona','?')} "
                            f"r_phys={_learned.get('r_phys','?')} "
                            f"tests_passed={_learned.get('tests_passed','?')} "
                            f"principles_added={_learned.get('principles_added',0)}"
                        )
                    else:
                        cog["last_learn_summary"] = "learn ran (no summary)"
                    return phase.result
                if action == "pivot":
                    _obj = self._objective if hasattr(self, "_objective") else ""
                    _cur = cog.get("current_hyp_id")
                    if _cur:
                        try:
                            new_hyp = self.hypothesis_graph.pivot(
                                _cur,
                                evidence={"reason": "cognitive pivot"},
                                model=self._get_refine_model(),
                                objective=_obj,
                            )
                            self._refine_count = 0
                            self._pivot_count += 1
                            self._next_phase_hint = "perceive"
                            logger.info("CognitiveLoop pivot: %s → %s", _cur, new_hyp)
                            # P1.4: pivot → campaign.refine 对齐 run() L1729
                            self._emit_campaign(
                                "campaign.refine",
                                {
                                    "iteration": state.iteration,
                                    "old_hyp_id": _cur,
                                    "new_hyp_id": new_hyp,
                                    "reason": "cognitive pivot",
                                },
                            )
                            # P15: pivot 是关键事件, 立刻 save (force=True)
                            self._maybe_save_engine_state(force=True, reason="pivot")
                        except Exception:
                            logger.warning("cognitive pivot failed", exc_info=True)
                    # 清中间状态, 下轮重新 observe
                    for k in ("hypothesis", "plan", "execution_result", "validation", "current_hyp_id"):
                        cog[k] = None
                    return "pivoted"
                if action in ("skip", "stop", "report"):
                    # report 由 _finalize_run 跑; stop/skip 是控制信号
                    return action
            except Exception as e:
                logger.warning("cognitive execute '%s' failed: %s", action, e)
                return None
            return None

        async def reflect_fn(
            state: LoopState, decision: ActionDecision, result: Any
        ) -> ReflectionResult:
            action = decision.action
            advice = ""
            redirect = False

            # 失败检测 — 各 action 的"无产出"判为 failed → redirect
            if action == "hypothesize" and not cog["hypothesis"]:
                redirect = True
                advice = "hypothesize 无产出, 下轮重新 observe"
            elif action == "plan" and not cog["plan"]:
                redirect = True
                advice = "plan 无产出, 下轮重新 hypothesize"
            elif action == "execute" and cog["execution_result"] is None:
                redirect = True
                advice = "execute None, 下轮重新 plan"

            # gate 检查 — 把 evidence 传给 _check_gate
            if action == "plan" and cog["plan"] and not redirect:
                if not self._check_gate(
                    "plan", "execute",
                    {"mode": cog["plan"].get("mode"),
                     "description": cog["plan"].get("description")},
                ):
                    redirect = True
                    advice = "gate plan→execute blocked"
            if action == "execute" and cog["plan"] and not redirect:
                if not self._check_gate(
                    "execute", "validate",
                    {"mode": cog["plan"].get("mode")},
                ):
                    redirect = True
                    advice = "gate execute→validate blocked"

            # consecutive_failures — 只在 validate 后算 (同 run())
            if action == "validate":
                validation = cog["validation"] or {}
                tests_ok = _extract_tests_passed(validation)
                # 700 万步极限场景: 滑动窗口失败率. 推入当前结果, 超窗口截断.
                # consecutive 在长轨迹里太窄 (20 次 tool timeout 就停), windowed rate
                # 允许局部失败 — 最近 100 次 validate 失败率 > 0.8 才算真死路.
                _vwin = getattr(self, "_validate_window", None)
                if _vwin is not None:
                    _vwin.append(bool(tests_ok))
                    _wsize = getattr(self, "_validate_window_size", 100)
                    if len(_vwin) > _wsize:
                        del _vwin[: -_wsize]
                if tests_ok:
                    self._consecutive_failures = 0
                    self._consecutive_failures_by_type = {}
                else:
                    self._consecutive_failures += 1
                    # F-borrow: 按 failure_type 分类计数 (forge 双预算思路).
                    # _classify_failure 已存在但之前没在 reflect 路径用 — 闭合断层.
                    # 失败分类后按类阈值 stop, 避免 tool_error 跟 hypothesis_error 混算.
                    try:
                        _redteam = self._redteam_findings()
                        ftype = AutoloopEngine._classify_failure(validation, _redteam)
                    except Exception:
                        ftype = "hypothesis_error"
                    by_type = getattr(self, "_consecutive_failures_by_type", {}) or {}
                    by_type[ftype] = by_type.get(ftype, 0) + 1
                    self._consecutive_failures_by_type = by_type
                    _type_max = getattr(self, "_max_failures_by_type", {}).get(
                        ftype, self._max_consecutive_failures
                    )
                    if by_type[ftype] >= _type_max:
                        logger.warning(
                            "cognitive stop: %d consecutive %s failures",
                            by_type[ftype], ftype,
                        )
                        return ReflectionResult(
                            should_stop=True,
                            advice=f"{by_type[ftype]} consecutive {ftype} failures",
                        )
                    if self._consecutive_failures >= self._max_consecutive_failures:
                        # 700 万步兜底: consecutive 触顶时, 检查滑动窗口失败率.
                        # 如果窗口内失败率低于阈值, 说明只是局部连续失败, 整体仍在进展 —
                        # 不停, 只清 consecutive 让它重新计数. 避免长轨迹被短期波动截停.
                        _win = getattr(self, "_validate_window", None)
                        _wsize = getattr(self, "_validate_window_size", 100)
                        _wthresh = getattr(self, "_validate_window_fail_threshold", 0.8)
                        if _win and len(_win) >= _wsize:
                            _fail_rate = 1.0 - (sum(_win) / len(_win))
                            if _fail_rate < _wthresh:
                                logger.info(
                                    "consecutive=%d 但窗口失败率 %.2f < %.2f, 不停, 清计数",
                                    self._consecutive_failures, _fail_rate, _wthresh,
                                )
                                self._consecutive_failures = 0
                                self._consecutive_failures_by_type = {}
                            else:
                                logger.warning(
                                    "cognitive stop: consecutive=%d 且窗口失败率 %.2f >= %.2f",
                                    self._consecutive_failures, _fail_rate, _wthresh,
                                )
                                return ReflectionResult(
                                    should_stop=True,
                                    advice=f"{self._consecutive_failures} consecutive failures (window fail rate {_fail_rate:.2f})",
                                )
                        else:
                            logger.warning(
                                "cognitive stop: %d consecutive failures (total cap)",
                                self._consecutive_failures,
                            )
                            return ReflectionResult(
                                should_stop=True,
                                advice=f"{self._consecutive_failures} consecutive failures",
                            )

            # G2: 周期检测 + 历史轨迹匹配 (M3 cycle_detect + M2 trajectory_match).
            # 不 should_stop — 给建议, 让 LLM decider / 规则版自己决定是否 pivot.
            # cycle 信号强 → 强制 redirect; match 信号弱 → 只注入 hint.
            try:
                stuck = self._check_stuck(state.action_history)
                if stuck:
                    if stuck["type"] == "cycle":
                        redirect = True
                        advice = (advice + " | G2 cycle: " + stuck["advice"]).strip(" |")
                        logger.warning("G2 stuck: %s", stuck["advice"])
                    elif stuck["type"] == "match":
                        # match 不是 stuck, 只是建议下一步. 注入 _speculator_hint
                        # 让下轮 hypothesize 能看到. 不 redirect.
                        self._speculator_hint = (
                            (self._speculator_hint + " | " + stuck["advice"])
                            .strip(" |")[:2000]
                        )
                        advice = (advice + " | G2 match: " + stuck["advice"]).strip(" |")
                        logger.info("G2 trajectory match: %s", stuck["advice"])
            except Exception:
                logger.debug("G2 _check_stuck failed (non-fatal)", exc_info=True)

            # timeout / pivot 预算 (硬停)
            if tracker.is_expired(progress_task_id):
                return ReflectionResult(should_stop=True, advice="timeout")
            if self._pivot_count >= self._max_pivots:
                return ReflectionResult(should_stop=True, advice="pivot budget exhausted")
            # 死循环防护: pivot 后还反复 fail → 别再 pivot, 直接停.
            # CognitiveLoop 自带的 repeated-action 检测抓不到 pivot/hyp 交替的情况.
            # ponytail: 用 action_history 数 pivot 次数, 不引入新状态字段.
            if action == "pivot" and state.action_history.count("pivot") >= 3:
                return ReflectionResult(
                    should_stop=True,
                    advice="3+ pivots without progress, stop",
                )

            # hint 用完清空 (同 run() 末尾)
            self._next_phase_hint = None
            self._refined_hypothesis = None
            # speculator hint 截断 (同 run())
            if len(self._speculator_hint) > 2000:
                self._speculator_hint = self._speculator_hint[-2000:]

            # AV2+AV4: PMK + TaskMetrics + detect_drift + heat_engine 接入 (reflect 末尾).
            # ponytail: 只在 validate 后跑 — perceive/hypothesize/plan 没产出
            # StepEvaluation 等价物. autoloop validation dict 字段不全 (无
            # evidence_quality/pmk_feedback/tool_call_health), 用 SimpleNamespace
            # 兜底, duck typing 够 update_metrics/detect_drift/should_pause_for_decision 用.
            # 天花板: pmk_cycle_count/tool_call_health_avg 在 autoloop 路径不增;
            # 升级路径: 在 _validate 里跑 StepEvaluator 填全字段.
            # autoloop 无人在环, pause 退化为日志 + hint, 不真停 (不设 should_stop).
            # AV4: detect_drift + TaskMetrics 抽到 update_drift_and_metrics 共享;
            #   heat_engine 抽到 update_heat_engine_after_step 共享 (对齐 rcb_runner AV8).
            if action == "validate" and cog.get("validation") is not None:
                try:
                    from types import SimpleNamespace as _NS
                    _val = cog["validation"] or {}
                    _tests_ok = _extract_tests_passed(_val)
                    # P0.2: _validate 真实字段是 tests_passed/benchmarks/
                    # thinking_collapse/*_error/effort_floor_deficits 等, 不是
                    # summary/result/errors. 之前硬取 summary/result/errors 全是
                    # 空串, 导致 PMK/drift/metrics 全在吃空数据.
                    _se_fields = _validation_to_step_eval_fields(
                        _val, _tests_ok, cog.get("execution_result"),
                        step_id=len(self._evals_history),
                    )
                    _step_eval = _NS(**_se_fields)
                    self._evals_history.append(_step_eval)

                    # AV4: detect_drift + TaskMetrics — 调共享函数
                    from huginn.autoloop.cognitive_loop import (
                        update_drift_and_metrics,
                        update_heat_engine_after_step,
                    )
                    self._drift_info, self._task_metrics = update_drift_and_metrics(
                        self._evals_history, _step_eval,
                        self._task_metrics, self._task_state_for_metrics,
                        self.workspace, self._run_id, self._max_iterations,
                    )
                    if self._drift_info and self._drift_info[0]:
                        advice = (advice + " | drift: " + self._drift_info[1]).strip(" |")
                        logger.warning("autoloop drift: %s", self._drift_info[1])

                    # AV4: heat_engine 闭环 — 对齐 rcb_runner AV8
                    try:
                        from huginn.metacog.cognitive_heat_engine import get_heat_engine
                        _he = get_heat_engine()
                        update_heat_engine_after_step(
                            _he, _step_eval,
                            prompt_len=len(getattr(self, "_last_hypothesis", "") or ""),
                            idea_count=self.hypothesis_graph.component_count() if hasattr(self, "hypothesis_graph") else 1,
                        )
                    except Exception:
                        logger.debug("AV4 heat_engine update in autoloop failed", exc_info=True)
                except Exception:
                    logger.debug("AV2 metrics/drift update failed", exc_info=True)

                # PMK 一致性 + should_pause_for_decision — autoloop 无人在环,
                # pause 退化为 hint 注入. 升级路径: 接 routes SSE 决策流.
                try:
                    from huginn.autoloop.cognitive_loop import (
                        build_pmk_state, check_pause_decision,
                    )
                    _persona_obj = None
                    try:
                        _persona_obj = self._get_persona_manager().get_persona("default")
                    except Exception:
                        pass
                    _pmk_state = build_pmk_state(
                        _persona_obj, _step_eval, self._get_kb() if hasattr(self, "_get_kb") else None,
                    )
                    _pause, _reason, _opts = check_pause_decision(
                        self._evals_history, [],
                        self._get_kb() if hasattr(self, "_get_kb") else None,
                        None, _pmk_state,
                    )
                    if _pause:
                        logger.warning("autoloop pause signal (no human): %s", _reason)
                        self._speculator_hint = (
                            (self._speculator_hint + f"\n[PAUSE] {_reason}\n").strip()
                        )
                        # H4: GRILL pause → 进入 grill 模式, 下次 _llm_chat 注入
                        # GRILL_SYSTEM_PROMPT_CN. 之前 pause 后只 auto-resume,
                        # LLM 看不到 grill 约束, "一次一问" 形同虚设.
                        if "GRILL" in _reason and not self._grill_active:
                            self._grill_active = True
                            self._grill_turns = 0
                            logger.info("GRILL mode activated: %s", _reason)
                except Exception:
                    logger.debug("AV2 should_pause_for_decision failed", exc_info=True)

                # v10-F2: completion audit — 对齐 run() L1878-1897.
                # goal 达标 + metacog 不阻断 → goal.status=completed + should_stop.
                # ponytail: check_completion 在 goal 无 criteria 时返回 False, 不影响.
                if goal is not None and not state.should_stop:
                    try:
                        _val_for_goal = cog["validation"] or {}
                        if GoalScheduler.check_completion(goal, _val_for_goal):
                            _blk, _why = self._metacog_check_completion()
                            if _blk:
                                logger.info("v10 completion audit blocked: %s", _why)
                                self._speculator_hint = (
                                    (self._speculator_hint + f"\n[completion audit] {_why}").strip()
                                )
                            else:
                                logger.info("v10 goal completed: %s", goal.objective)
                                goal.status = "completed"
                                if self._goal_scheduler is not None:
                                    try:
                                        self._goal_scheduler.complete_goal(goal.id)
                                    except Exception:
                                        logger.debug("complete_goal failed (non-fatal)", exc_info=True)
                                state.should_stop = True
                    except Exception:
                        logger.debug("v10 F2 completion audit failed (non-fatal)", exc_info=True)

                # v10-F17: GoalJudge — 对齐 run() L1899-1945.
                # 每 3 轮或最后一轮调 GoalJudge.judge 判 goal_achieved.
                # achieved + metacog 不阻断 → should_stop; gaps → 注入 hint.
                # ponytail: GoalJudge(llm=None) 走规则版, LLM judge 留 exit 阶段.
                if goal is not None and not state.should_stop:
                    if state.iteration % 3 == 2 or state.iteration >= max_iterations - 1:
                        try:
                            from huginn.evaluation.goal_judge import GoalJudge

                            _judge = GoalJudge(llm=None)
                            _final_text = str(
                                (cog["validation"] or {}).get("summary")
                                or (cog["validation"] or {}).get("result_data")
                                or (cog.get("execution_result") or {}).get("summary", "")
                            )
                            _gj = _judge.judge(goal.objective, None, _final_text)
                            if _gj.get("achieved"):
                                _blk, _why = self._metacog_check_completion()
                                if _blk:
                                    logger.info("v10 GoalJudge audit blocked: %s", _why)
                                    self._speculator_hint = (
                                        (self._speculator_hint + f"\n[completion audit] {_why}").strip()
                                    )
                                else:
                                    logger.info("v10 GoalJudge achieved (score=%s)", _gj.get("score"))
                                    state.should_stop = True
                            elif _gj.get("gaps"):
                                _gap_hint = "; ".join(_gj["gaps"][:3])
                                self._speculator_hint = (
                                    (self._speculator_hint + "\n" + _gap_hint).strip()
                                    if self._speculator_hint else _gap_hint
                                )
                                logger.info("v10 GoalJudge gaps: %s", _gap_hint)
                        except Exception:
                            logger.debug("v10 F17 GoalJudge failed (non-fatal)", exc_info=True)

                # v10-F4: surprise 早停 — 对齐 run() L1967-1999.
                # 连续 3 轮低 surprise + audit 不阻断 → should_stop.
                # 阈值自适应: noise 大时严格 (0.08), noise 小时宽松 (0.20).
                if not state.should_stop and len(self._surprise_history) >= 3:
                    try:
                        _recent = self._surprise_history[-3:]
                        _worsts = [w for w, _ in _recent]
                        _avg_noise = sum(s for _, s in _recent) / len(_recent)
                        _thr = max(0.08, 0.20 - 0.4 * _avg_noise)
                        if all(w < _thr for w in _worsts):
                            _blk, _why = self._metacog_check_completion()
                            if _blk:
                                logger.info("v10 surprise audit blocked: %s", _why)
                                self._speculator_hint = (
                                    (self._speculator_hint + f"\n[completion audit] {_why}").strip()
                                )
                            else:
                                logger.info(
                                    "v10 surprise converged < %.2f (noise=%.2f), stop",
                                    _thr, _avg_noise,
                                )
                                state.should_stop = True
                    except Exception:
                        logger.debug("v10 F4 surprise early-stop failed (non-fatal)", exc_info=True)

                # v10-F3: darwin_ratchet — 对齐 run() L2003-2004.
                # 内部判 stagnation >= 5 设 self._should_stop; 这里同步到 state.
                # ponytail: _darwin_ratchet_check 也更新 heat_engine T_cold + health,
                #   不只是 stop 判定. run() 用 self._should_stop, run_cognitive 用 state.should_stop.
                if not state.should_stop:
                    try:
                        self._darwin_ratchet_check()
                        if getattr(self, "_should_stop", False):
                            state.should_stop = True
                    except Exception:
                        logger.debug("v10 F3 darwin_ratchet failed (non-fatal)", exc_info=True)

            # P15: 周期 save — flag off 时 no-op, iteration % save_every == 0 才真写.
            # refute 在 _learn 内发生, reflect 末尾的周期 save 会在 ≤save_every 步内捕获.
            self._maybe_save_engine_state(reason="periodic")

            return ReflectionResult(
                should_continue=True,
                should_redirect=redirect,
                redirect_reason=advice if redirect else "",
                advice=advice,
                should_stop=state.should_stop,
            )

        loop = CognitiveLoop(
            observe_fn=observe_fn,
            decide_fn=decide_fn,
            execute_fn=execute_fn,
            reflect_fn=reflect_fn,
            output_writer=None,  # provenance 走 _run_phase_async 内的 _record_provenance
            max_iterations=max_iterations,
            max_repeated_actions=3,
        )
        state = await loop.run(LoopState(max_iterations=max_iterations))

        # finalize — 复用 run() 的收尾 (含 _report)
        return await self._finalize_run(
            objective,
            cog["phases"],
            run_id,
            provenance_record,
            run_collector,
            tracker,
            progress_task_id,
            cog["completed_steps"],
        )






def _selfcheck() -> None:
    """CognitiveLoop 骨架自检 — 不依赖 LLM, 用 mock 钩子验证控制流."""
    import asyncio

    async def mock_observe(state: LoopState) -> dict[str, Any]:
        return {"step": state.iteration}

    async def mock_decide(state: LoopState, obs: dict[str, Any]) -> ActionDecision:
        # 第 3 轮后选 stop, 验证主循环能退出
        if state.iteration >= 3:
            return ActionDecision(action="stop", rationale="done")
        return ActionDecision(action="execute", rationale="working")

    async def mock_execute(state: LoopState, decision: ActionDecision) -> Any:
        return f"result_iter_{state.iteration}"

    async def mock_reflect(
        state: LoopState, decision: ActionDecision, result: Any
    ) -> ReflectionResult:
        return ReflectionResult(should_continue=True)

    class MockWriter(OutputWriter):
        def __init__(self):
            self.steps = []
        def write_step(self, iteration, action, result, reflection):
            self.steps.append((iteration, action))

    async def run_test():
        writer = MockWriter()
        loop = CognitiveLoop(
            observe_fn=mock_observe,
            decide_fn=mock_decide,
            execute_fn=mock_execute,
            reflect_fn=mock_reflect,
            output_writer=writer,
            max_iterations=10,
        )
        state = await loop.run()
        assert state.iteration == 3, f"expected 3 iters, got {state.iteration}"
        assert state.should_stop, "should_stop should be True after 'stop' action"
        assert len(writer.steps) == 3, f"writer should have 3 steps, got {len(writer.steps)}"
        assert writer.steps[-1] == (3, "stop"), f"last step wrong: {writer.steps[-1]}"

        # 死循环防护测试: decide 总返回相同 action
        async def stuck_decide(state, obs):
            return ActionDecision(action="execute", rationale="stuck")
        writer2 = MockWriter()
        loop2 = CognitiveLoop(
            observe_fn=mock_observe,
            decide_fn=stuck_decide,
            execute_fn=mock_execute,
            reflect_fn=mock_reflect,
            output_writer=writer2,
            max_iterations=10,
            max_repeated_actions=3,
        )
        state2 = await loop2.run()
        # 死循环防护: 6 次 (2x limit) 后强制 stop, should_redirect 在过程中被设过
        assert state2.should_stop, "stuck loop should force stop at 2x limit"
        assert state2.iteration == 6, f"stuck loop should stop at 6 iters (2x limit=3), got {state2.iteration}"
        assert state2.should_redirect, "should_redirect should be set before force stop"
        assert "死循环防护" in state2.redirect_reason, f"redirect_reason wrong: {state2.redirect_reason}"

        # invalid action 测试
        async def bad_decide(state, obs):
            return ActionDecision(action="invalid_action", rationale="bad")
        loop3 = CognitiveLoop(
            observe_fn=mock_observe,
            decide_fn=bad_decide,
            execute_fn=mock_execute,
            reflect_fn=mock_reflect,
            max_iterations=2,
        )
        state3 = await loop3.run()
        # invalid action 被替换为 'skip', 不崩溃
        assert state3.iteration >= 1, "should run at least 1 iter"

        print("CognitiveLoop selfcheck OK (3/3: basic flow + stuck protection + invalid action)")

    asyncio.run(run_test())


if __name__ == "__main__":
    _selfcheck()
