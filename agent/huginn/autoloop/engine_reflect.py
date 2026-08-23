"""EngineReflectMixin — AutoloopEngine 的 validate / learn / report 阶段方法族.

从 engine.py 拆出 (P3 slim-down 续). 包含:
- _validate (结果校验: pytest/benchmark/literature comparison/surprise/generative verify)
- _learn (反思学习: skill 抽象/self-goal 合成/Feynman 学习/blind-spot pass)
- _report (科学报告生成)
- 各类校验辅助 (blind reconstruct/derivation consistency/failure trace inversion 等)

通过 self 访问 engine 状态. 方法体原样搬迁, 不改逻辑.

设计原则 (ponytail):
- 对 engine.py 模块级符号用方法内 lazy import, 避免 circular
- Mixin 不持有自己的状态, 全部走 self
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import uuid
from typing import Any

# 反思阶段方法引用的 engine.py 模块级 import (均为叶子模块, 无 circular 风险)
from huginn.autoloop.types import LoopPhase
from huginn.core_types import ToolContext
from huginn.utils.runtime import HUGINN_DIR_NAME

logger = logging.getLogger(__name__)


class EngineReflectMixin:
    """validate / learn / report 阶段方法族. 通过 self 访问 engine 状态."""

    _FEYNMAN_PROMPT = """You are studying your own research iteration using the Feynman Learning Method.The core principle: if you can't explain it in simple terms, you don't truly understand it.## Iteration Context- Hypothesis: {hypothesis}- Plan mode: {mode}- R_phys (physical reward): {r_phys}- Surprise: {surprise} (how much the actual result differed from prediction)- Validation summary: {validation}- Deviations from plan: {deviations}## Your TaskWrite TWO sections:### Simple ExplanationExplain what happened in this iteration as if teaching a newcomer who has basicmaterials science knowledge but no experience with computational tools.Focus on: What was the physical question? What did the calculation reveal?Why does the result make sense (or not)? Use analogies where helpful.If there were deviations from the plan, explain WHY the path changed.### Knowledge GapsList specific things you CANNOT confidently explain. Be honest — admittinggaps is the point of this exercise. Mark each gap:- [KU] for "known unknown" — you know you don't understand this- [UU] for "unknown unknown" — you didn't even think about this until nowExamples:- "[KU] I don't understand why the band gap changed non-monotonically with doping"- "[UU] I never considered that GaN has two polymorphs until the result came back"Output format (Markdown, no code blocks):## Simple Explanation...## Knowledge Gaps- [KU] gap 1- [UU] gap 2..."""

    _BLIND_SPOT_PROMPT = """You are about to start a research task. Before diving in, do a blindspot pass.## TaskObjective: {objective}## Current Context{context_summary}## Your JobIdentify potential UNKNOWN UNKNOWNS — things that might go wrong, assumptions that mightbe invalid, or aspects of the problem that haven't been considered yet.Think about:1. Physical assumptions: Are there structural/phase/electronic considerations being missed?2. Computational pitfalls: Convergence, basis set, pseudopotential, k-grid issues?3. Data gaps: Is there reference data missing? Are there known experimental values to compare against?4. Methodology blind spots: Could the chosen method give qualitatively wrong results for this system?5. Edge cases: Temperature, pressure, doping level boundaries?Output up to 5 potential blind spots, one per line, prefixed with "BS:".For each, also note the type: [structural], [computational], [data], [method], [edge_case].Format: BS: [type] descriptionIf you genuinely can't find any blind spots (unlikely), output: NONE"""

    _NEXT_STEP_ADVISOR_PROMPT = """你刚结束一段研究任务, 要给用户推荐下一步. 你不是裁判, 是科研伴侣.姿态原则:- 不说"你应该"、"建议你"、"最优选择"- 说"我注意到 X"、"这里可能值得 Y"、"你对 Z 有感觉吗"- 捕捉用户初发的模糊直觉, 不强迫收敛- 给 2-3 个具体选项 + 1 个"我自己有想法"的出口输入上下文:- 本轮假设: {hypothesis}- 本轮结果: {outcome}- iteration_history 反常点: {anomalies}- _physical_timeseries 反常: {ts_anomaly}- prev_run_context 尾巴: {prev_tail}输出格式 (markdown, 简洁, 总长 < 200 字):## 本轮看到- 一句话 outcome- 一句话反常点 (没有就省略)## 可能的下一步**A. 方向名**  基于: 信号来源  代价: cost_tier 或 walltime 估计  价值: 一句话  风险: 一句话**B. 方向名**  ...**C. 方向名**  ...**D. 我自己有想法**  你对某个方向有直觉吗? 哪怕还没成形也行.约束:- A/B/C 不能来自同一信号源- 至少一个"深化"方向 + 一个"横向"方向- 若有 _physical_timeseries 反常, 必须有一条推荐是"验证这个反常"- 若 prev_run_context.outcome = inconclusive, 必须有一条是"接着上轮尾巴"- D 永远是用户自由出口"""

    async def _validate(self, execution_result: Any) -> dict[str, Any]:
        """Validate execution results using benchmarks and constraints."""
        # H4: reviewer 阈值 + MatWorldBench 白名单 + needs_retry 阈值从 PhaseRegistry extra 取
        from huginn.harness.phase_spec import get_phase_extra
        _reviewer_threshold = get_phase_extra("_validate", "reviewer_threshold", 0.5)
        _mwb_categories = get_phase_extra("_validate", "matworldbench_categories", [
            "structure", "thermo", "electronic",
        ])
        _needs_retry_threshold = get_phase_extra("_validate", "needs_retry_threshold", 0.5)
        results = {
            "tests_passed": False,
            "constraints_satisfied": False,
            "benchmarks": {},
        }

        if isinstance(execution_result, dict):
            # Extract visual primitives from tool output — the deictic pointers
            # that let the next iteration's hypothesis/plan reason about data
            # shape without needing image input (Mirage + Visual Primitives).
            visual_hint = execution_result.get("_visual_hint")
            if visual_hint:
                results["visual_primitives"] = visual_hint
                self._last_visual_context = visual_hint

            # 比较性视觉原语: 把本轮结果和上轮做差分, 突出变化.
            # 峰值位移/新异常/趋势反转 — 这些是 agent 最关心的信号.
            prev_exec = getattr(self, "_last_execution_result", None)
            if prev_exec and isinstance(prev_exec.get("result"), dict):
                try:
                    from huginn.tools.visual_hook import extract_comparative_primitives

                    comp = extract_comparative_primitives(
                        prev_exec.get("result", {}), execution_result
                    )
                    if comp:
                        results["comparative_primitives"] = comp
                        # 也拼进 visual_context, 下轮 hypothesis 能看到
                        self._last_visual_context = (
                            f"{self._last_visual_context}\n{comp}".strip()
                            if self._last_visual_context
                            else comp
                        )
                except Exception:
                    logger.debug("comparative primitives extraction skipped", exc_info=True)

            r_phys = execution_result.get("r_phys")
            if r_phys is None:
                result_type = execution_result.get("result_type")
                result_data = execution_result.get("result_data")
                if result_type and result_data:
                    try:
                        from huginn.tools.validate_tool import (
                            ValidateTool,
                            ValidateToolInput,
                        )

                        validator = ValidateTool()
                        tool_ctx = ToolContext(
                            session_id=f"validate_{uuid.uuid4().hex[:8]}",
                            workspace=str(self.workspace),
                            config=self.settings,
                        )
                        vr = await validator.call(
                            ValidateToolInput(
                                result_type=result_type,
                                result_data=result_data,
                            ),
                            tool_ctx,
                        )
                        if vr.success and vr.data:
                            r_phys = vr.data.get("r_phys")
                            results["physics_validation"] = vr.data
                    except Exception as e:
                        results["physics_validation_error"] = str(e)
            if r_phys is not None:
                results["r_phys"] = r_phys

        try:
            collapse = self._detect_thinking_collapse(execution_result)
            if collapse:
                results["thinking_collapse"] = collapse
        except Exception as e:
            results["thinking_collapse_error"] = str(e)

        # pytest, benchmark, math validation — sync, offload to thread
        py_test, bench_report, math_val = await asyncio.gather(
            self._run_pytest(),
            self._run_benchmark(),
            self._run_math_validation(execution_result),
            return_exceptions=True,
        )

        if isinstance(py_test, dict):
            results.update(py_test)
        elif isinstance(py_test, Exception):
            results["test_output"] = f"Test execution error: {py_test}"

        if isinstance(bench_report, dict):
            results["benchmarks"] = bench_report
        elif isinstance(bench_report, Exception):
            logger.warning("BenchmarkRunner failed", exc_info=True)

        if isinstance(math_val, dict):
            results["math_validation"] = math_val
        elif isinstance(math_val, Exception):
            results["math_validation_error"] = str(math_val)

        try:
            math_ev = await self._collect_math_evidence(
                execution_result, results.get("math_validation", {})
            )
            for _k, _v in math_ev.items():
                results[_k] = _v
        except Exception as e:
            results["math_evidence_error"] = str(e)

        # Conditional verification: run cheap generative_verify first,
        # only call expensive reviewer critique when score < 0.5.
        # ponytail: _generative_verify 依赖 results (被 math_evidence 修改过),
        # 不能和 _collect_math_evidence 并行. 但可以和 emergent_complexity/
        # literature_comparison 并行 — 它们在下面已经 gather 了.
        gen_verify = None
        try:
            gen_verify = await self._generative_verify(execution_result, results)
            if gen_verify:
                results["generative_verify"] = gen_verify
        except Exception as e:
            results["generative_verify_error"] = str(e)

        needs_review = gen_verify is None or gen_verify.get("score", 0.5) < _reviewer_threshold
        if needs_review:
            try:
                reviewer_kb = self._build_kb_text(
                    query=self._summarize_for_kb(execution_result, results)
                )
                critique = await self._llm_chat(
                    self._build_reviewer_prompt(execution_result, results, reviewer_kb),
                    persona_name="reviewer",
                    model=self.verification_model,
                )
                if critique and critique.strip():
                    results["reviewer_critique"] = critique.strip()
            except Exception as e:
                results["reviewer_critique_error"] = str(e)

        # emergent complexity + literature + grader + eval — independent
        ec_task = asyncio.create_task(
            self._safe_emergent_complexity(execution_result, results)
        )
        lit_task = asyncio.create_task(
            self._safe_literature_comparison(execution_result, results)
        )
        await asyncio.gather(ec_task, lit_task)

        try:
            from huginn.validation.grader import default_registry

            # ValidityJudge 需要 model + 对话日志/代码做 post-hoc 审查
            # NatureBench judge.py 启发: r_phys 高不代表真算, 可能 gaming grader
            reg = default_registry(
                model=getattr(self, "verification_model", None) or self.model
            )
            merged: dict[str, Any] = {}
            if isinstance(execution_result, dict):
                merged.update(execution_result)
            merged.update(results)
            # 喂给 ValidityJudge: 从 memory 取最近对话 + execution_result 里的 code
            try:
                recent = self.memory.get_recent_messages(n=20)
                conv_snippets = []
                for m in recent:
                    role = getattr(m, "role", "?")
                    content = getattr(m, "content", "")
                    if isinstance(content, (dict, list)):
                        content = str(content)[:500]
                    conv_snippets.append(f"[{role}] {str(content)[:500]}")
                merged["conversation_log"] = "\n".join(conv_snippets)
            except Exception:
                logger.debug("conversation_log extract for judge failed", exc_info=True)
            # agent_code: execution_result 里可能带 code/parsed/script
            if isinstance(execution_result, dict):
                for k in ("code", "script", "generated_code", "final_answer"):
                    v = execution_result.get(k)
                    if v and isinstance(v, str) and len(v) > 50:
                        merged["agent_code"] = v
                        break
                else:
                    # 退而求其次: tool_input 里的 description 可能含代码片段
                    ti = execution_result.get("_tool_input") or {}
                    if isinstance(ti, dict):
                        merged["agent_code"] = str(ti.get("description", ""))[:5000]
            grader_list = reg.evaluate_all(merged)
            results["grader_scores"] = {
                gr.name: {
                    "score": gr.score,
                    "passed": gr.passed,
                    "message": gr.message,
                }
                for gr in grader_list
            }
            if grader_list:
                avg_score = sum(gr.score for gr in grader_list) / len(grader_list)
                results["grader_reward"] = round(avg_score, 4)
                from huginn.events.unified_bus import publish_event

                publish_event(
                    "quality.check",
                    {
                        "iteration": self._iteration,
                        "graders": results["grader_scores"],
                        "reward": results.get("grader_reward", 0),
                    },
                    source="autoloop",
                )
        except Exception as e:
            results["grader_error"] = str(e)

        try:
            from huginn.evaluation.matworld_bench import MatWorldBench

            bench = MatWorldBench()
            exec_data = execution_result if isinstance(execution_result, dict) else {}
            eval_scores: list[dict] = []
            for task in bench.tasks:
                if task.category in tuple(_mwb_categories):
                    try:
                        br = bench.evaluate(task.id, exec_data)
                        eval_scores.append(
                            {
                                "task_id": task.id,
                                "category": task.category,
                                "passed": br.passed,
                                "score": br.score,
                            }
                        )
                    except Exception:
                        logger.debug("benchmark result collect skipped", exc_info=True)
            if eval_scores:
                passed = sum(1 for e in eval_scores if e["passed"])
                results["eval_summary"] = {
                    "bench_passed": passed,
                    "bench_total": len(eval_scores),
                    "bench_pass_rate": round(passed / len(eval_scores), 4),
                    "details": eval_scores,
                }
        except Exception as e:
            logger.debug(f"[validate] eval bench failed: {e}")

        # JEPA 式预测误差: 对比 plan 阶段的预测 vs 实际结果.
        # 预测误差高 = surprise = 值得探索的方向 (intrinsic motivation).
        # 低误差 = agent 对这类任务已有良好心智模型.
        prediction = getattr(self, "_current_prediction", "")
        if prediction:
            actual_text = self._extract_text(execution_result)[:500]
            robust = self._compute_surprise_robust(prediction, actual_text)
            surprise = robust["worst"]
            results["prediction_error"] = {
                "predicted": prediction[:200],
                "actual": actual_text[:200],
                "surprise": round(surprise, 3),
                "surprise_mean": round(robust["mean"], 3),
                "surprise_worst": round(robust["worst"], 3),
                "surprise_std": round(robust["std"], 3),
            }
            self._last_surprise = surprise
            self._surprise_history.append((surprise, robust["std"]))

        # AV7: 最小努力下限硬阻断. _metacog_check_completion 已封装
        # families/live_components/UNEXPLORED 自白收集, 这里复用.
        # 不达标时: 强制 tests_passed=False → run loop L1616-1647 走失败分支;
        # 设 failure_kind=effort_floor_retry → _classify_failure 归 tool_error
        # (不 refute, 下轮重试同一假设扩方法族, 避免污染 hypothesis_graph).
        # ponytail: 复用现成 refine/retry 控制流, 不新写迭代触发逻辑.
        try:
            _eff_blk, _eff_why = self._metacog_check_completion()
            results["effort_floor_passed"] = not _eff_blk
            if _eff_blk:
                results["effort_floor_deficits"] = _eff_why
                results["failure_kind"] = "effort_floor_retry"
                results["tests_passed"] = False
                results["constraints_satisfied"] = False
                _hint = (
                    f"[effort floor] 探索未达硬下限, 不算通过: {_eff_why}. "
                    "下轮必须扩方法族或保留更多假设, 不要再收敛."
                )
                self._speculator_hint = (
                    (self._speculator_hint + "\n" + _hint).strip()
                    if self._speculator_hint else _hint
                )
                if len(self._speculator_hint) > 2000:
                    self._speculator_hint = self._speculator_hint[-2000:]
        except Exception:
            logger.debug("AV7 effort floor check in _validate failed", exc_info=True)

        # H2: bandit 记录 variant outcome (r_phys + efficiency + novelty 都算出后)
        # 只对 dynamic_workflow bandit 路径生效 (execution_result 带 _variant_id)
        if isinstance(execution_result, dict) and execution_result.get("_variant_id"):
            try:
                from huginn.autoloop.bandit import (
                    VariantArchive,
                    WorkflowBandit,
                )
                _vid = execution_result["_variant_id"]
                _obj_hash = execution_result.get("_objective_hash", "")
                _obj = execution_result.get("_objective", "")
                _novelty = float(execution_result.get("_novelty", 0.0))
                _eff = float(execution_result.get("_efficiency", 0.0))
                _r_phys = float(results.get("r_phys", 0.0) or 0.0)
                _success = bool(results.get("tests_passed", False))
                _script_dict = execution_result.get("_script_dict", {})
                bandit = WorkflowBandit.get_instance()
                bandit.record_variant_outcome(
                    _vid, _obj_hash, _success,
                    r_phys=_r_phys, efficiency=_eff, novelty=_novelty,
                )
                # archive: fitness = [r_phys, efficiency, novelty]
                archive = VariantArchive.get_instance()
                b = bandit.get_belief(_vid, _obj_hash)
                archive.add_variant(
                    _obj_hash, _obj, _vid, _script_dict,
                    fitness=[_r_phys, _eff, _novelty],
                    alpha=b.successes if b else 1,
                    beta=b.failures if b else 1,
                )
            except Exception:
                logger.debug("H2 bandit record in _validate failed", exc_info=True)

        self._last_validation = json.dumps(results, ensure_ascii=False, default=str)[
            :1000
        ]
        # Store failure_mode for next hypothesis loop (Dream Layer: crash = discovery)
        _gv = results.get("generative_verify", {})
        if isinstance(_gv, dict):
            self._last_failure_mode = _gv.get("failure_mode", "")
        # P1: 盲重建 verification + support/refute 闭环.
        # 之前 _validate 算出分数但不调 hypothesis_graph.support/refute, 图全 untested.
        # 现在开 toggle 时: (1) fresh subagent 从 statement 独立推导 (2) 比对盲重建
        # vs execution_result (3) mismatch→refute / match→support, 写 FAILED/PROVED.md.
        # ponytail: 默认 off (贵, 多一次 subagent dispatch). 升级: 只在割点/关键假设上开.
        if os.environ.get("HUGINN_BLIND_RECONSTRUCTION", "0") == "1":
            try:
                await self._blind_reconstruct_verify(execution_result, results)
            except Exception:
                logger.debug("P1 blind reconstruct failed", exc_info=True)

        # Epistemic gate (IOED 兜底): 强断言 + 证据缺失 → 暴露知识缺口.
        # 纯启发式, 零 LLM 成本; 只报告不改 tests_passed, 永不抛异常.
        # 与 prompts.py Epistemic Honesty 指令配合, 模型失灵时兜底暴露缺口.
        try:
            from huginn.validation.epistemic import detect_epistemic_gap

            _gap = detect_epistemic_gap(execution_result, results)
            if _gap:
                results["epistemic_gap"] = _gap
                logger.warning(
                    "epistemic_gap: %s", _gap.get("advice", "")[:120]
                )
        except Exception:
            logger.debug("epistemic gate check failed (non-fatal)", exc_info=True)

        return results



    async def _blind_reconstruct_verify(
        self, execution_result: Any, results: dict[str, Any],
    ) -> None:
        """P1: 盲重建 + support/refute 闭环 (Task 2 升级为三档交叉验证).

        1. 拿当前 hypothesis statement (不传 proof/evidence)
        2. SubagentDispatch("blind_reconstructor") 独立推导, 返 holds + derivation
        3. 三档判定:
           - holds match + derivation 一致 → strong support (写 PROVED.md)
           - holds match + derivation 冲突 → weak (不调 support, further_verification)
           - holds mismatch → refute (写 FAILED.md, evidence 含 blind derivation)
           - 缺 derivation / 缺 orig reasoning → 走原两档 (legacy)
        """
        _hyp_id = getattr(self, "_current_hyp_id_for_plan", None)
        if not _hyp_id:
            return
        try:
            _node = self.hypothesis_graph._nodes.get(_hyp_id)
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return
        if _node is None or _node.status != "untested":
            return
        _statement = _node.statement
        if not _statement or len(_statement) < 10:
            return
        # P0 Task 3: per-hyp budget 检查 — toggle off 时不检查 (向后兼容)
        if os.environ.get("HUGINN_PER_HYP_BUDGET", "0") == "1":
            try:
                _vb = _node.evidence.get("verification_budget")
                if _vb is not None:
                    _used = _node.evidence.get("blind_rounds_used", 0)
                    if _used >= _vb.get("blind_rounds", 0):
                        _node.evidence["budget_exhausted"] = True
                        results["blind_reconstruction"] = {
                            "match": None, "skipped": "budget_exhausted",
                        }
                        logger.debug(
                            "blind reconstruct skipped: budget_exhausted (hyp=%s)",
                            _hyp_id,
                        )
                        return
                    _node.evidence["blind_rounds_used"] = _used + 1
            except Exception:
                logger.debug("per-hyp blind budget check failed", exc_info=True)
        if self._agent_factory is None:
            logger.debug("P1 blind reconstruct: no agent_factory, skip")
            return
        from huginn.agents.subagent import SubagentDispatch
        _dispatch = SubagentDispatch()
        _task = (
            f"Independently derive whether this statement holds, from first principles. "
            f"Do NOT assume any prior proof. Output JSON.\n\n"
            f"Statement: {_statement}"
        )
        _ctx = {"agent_factory": self._agent_factory}
        try:
            _res = await _dispatch.dispatch("blind_reconstructor", _task, context=_ctx)
        except Exception:
            logger.debug("P1 blind reconstruct dispatch failed", exc_info=True)
            return
        if not _res.success or not _res.summary:
            return
        # 解析盲重建结果 (JSON summary)
        import json as _json
        try:
            _blind = _json.loads(_res.summary)
        except Exception:
            # LLM 没输出合法 JSON, 从 summary 文本推断
            _blind = {"holds": "true" in _res.summary.lower(), "confidence": 0.5}
        _blind_holds = bool(_blind.get("holds", False))
        # 原 execution_result 是否支持 hypothesis (tests_passed / grader_reward)
        _orig_holds = bool(
            results.get("tests_passed")
            or results.get("grader_reward", 0) > 0.5
            or results.get("generative_verify", {}).get("score", 0) > 0.5
        )
        # derivation 字段 (可能缺, 向后兼容)
        _blind_derivation = str(_blind.get("derivation", "") or "").strip()
        _has_derivation = bool(_blind_derivation)
        # 取原始推理路径 (session.reasoning_trace 最近几条), 用于跟 blind derivation 比对
        _orig_reasoning = ""
        try:
            _rt = getattr(self.memory.session, "reasoning_trace", None) or []
            if _rt:
                _orig_reasoning = "\n".join(str(_x) for _x in _rt[-5:])[:2000]
        except Exception:
            logger.debug("orig reasoning_trace extract failed", exc_info=True)
        # derivation 语义一致性: 仅当 blind+orig 都有内容时才判
        # ponytail: LLM zero-shot 判语义一致, 失败降级 token Jaccard heuristic
        # ceiling: heuristic 抓不到同义改写; 升级换 embedding cosine > 0.85
        _derivation_consistent: bool | None = None
        if _has_derivation and _orig_reasoning:
            try:
                _derivation_consistent = await self._judge_derivation_consistency(
                    _blind_derivation, _orig_reasoning,
                )
            except Exception:
                logger.debug("derivation consistency judge crashed", exc_info=True)
                _derivation_consistent = None
        _evidence = {
            "modality": "blind_reconstruction",
            "data_source": f"subagent:{_res.spec_name}",
            "blind_holds": _blind_holds,
            "blind_confidence": float(_blind.get("confidence", 0.5)),
            "blind_derivation": _blind_derivation[:500],
            "orig_holds": _orig_holds,
            "orig_reasoning_excerpt": _orig_reasoning[:500],
            "derivation_consistent": _derivation_consistent,
            "tests_passed": results.get("tests_passed"),
            "grader_reward": results.get("grader_reward"),
        }
        if _blind_holds != _orig_holds:
            # holds mismatch → refute (走原逻辑, evidence 含 blind derivation)
            _evidence["errors"] = (
                f"blind_holds={_blind_holds} vs orig_holds={_orig_holds} "
                f"mismatch — blind reconstruction disagrees with execution result"
            )
            _evidence["verification_level"] = "refute"
            self.hypothesis_graph.refute(_hyp_id, _evidence)
            results["blind_reconstruction"] = {"match": False, **_evidence}
            # P1 Task 8: record verification_mismatch for counterexample hint
            try:
                self.memory.remember_typed(
                    content=json.dumps({
                        "hypothesis": _statement[:300],
                        "blind_holds": _blind_holds,
                        "orig_holds": _orig_holds,
                        "blind_derivation": _blind_derivation[:500],
                        "hyp_id": _hyp_id,
                    }, ensure_ascii=False),
                    memory_type="verification_mismatch",
                    tags=["verification_mismatch", f"hyp:{_hyp_id}"],
                    source="blind_reconstruction",
                    importance=0.7,
                    tier="mid",
                )
            except Exception:
                logger.debug("verification_mismatch record failed", exc_info=True)
            logger.info("P1 blind reconstruct: mismatch → refute %s", _hyp_id)
            return
        # holds match — 按 derivation 一致性分档
        if _derivation_consistent is None:
            # 缺 derivation 或缺 orig reasoning → 走原两档 (legacy, 向后兼容)
            _evidence["verification_level"] = "legacy"
            self.hypothesis_graph.support(_hyp_id, _evidence)
            results["blind_reconstruction"] = {"match": True, **_evidence}
            logger.info("P1 blind reconstruct: legacy match → support %s", _hyp_id)
        elif _derivation_consistent:
            # holds match + derivation 一致 → strong support
            _evidence["verification"] = "blind_strong"
            _evidence["verification_level"] = "strong"
            self.hypothesis_graph.support(_hyp_id, _evidence)
            results["blind_reconstruction"] = {"match": True, **_evidence}
            logger.info("P1 blind reconstruct: strong match → support %s", _hyp_id)
        else:
            # holds match + derivation 冲突 → weak, 不调 support, 不写 PROVED.md
            _evidence["verification_level"] = "weak"
            _evidence["further_verification_needed"] = True
            results["blind_reconstruction"] = {"match": True, **_evidence}
            logger.info("P1 blind reconstruct: weak (derivation 冲突) %s", _hyp_id)



    async def _judge_derivation_consistency(
        self, blind_derivation: str, orig_reasoning: str,
    ) -> bool | None:
        """LLM zero-shot 比对 blind derivation 与原始推理路径是否语义一致.

        Returns True=一致 / False=冲突 / None=无法判 (heuristic 也判不出时).

        ponytail: 复用 verification_model.ainvoke 路径 (跟 trajectory_pattern 同款),
                  不新增模型注册. LLM 失败降级到 token Jaccard > 0.3
                  (ceiling: 抓不到同义改写; 升级换 embedding cosine > 0.85).
        """
        if not blind_derivation or not orig_reasoning:
            return None
        # 1. LLM 判
        try:
            import json as _json_jdg

            from langchain_core.messages import HumanMessage, SystemMessage
            _sys = SystemMessage(content=(
                "You are a semantic consistency judge. Compare two reasoning paths "
                "and decide if they are semantically consistent (same conclusion via "
                "compatible reasoning, even if worded differently) or in conflict "
                "(same conclusion but contradictory reasoning steps/assumptions).\n\n"
                "Output ONLY a JSON object: "
                '{"consistent": true|false, "reason": "..."}'
            ))
            _human = HumanMessage(content=(
                f"## Reasoning path A (blind reconstruction)\n"
                f"{blind_derivation[:1500]}\n\n"
                f"## Reasoning path B (original)\n{orig_reasoning[:1500]}\n\n"
                "Are A and B semantically consistent? Output ONLY the JSON."
            ))
            _model = self.verification_model
            _resp = await _model.ainvoke([_sys, _human])
            _text = getattr(_resp, "content", str(_resp))
            _text = _text.strip()
            if _text.startswith("```"):
                _nl = _text.find("\n")
                _text = _text[_nl + 1:] if _nl > 0 else _text[3:]
                if _text.endswith("```"):
                    _text = _text[:-3]
                _text = _text.strip()
            _verdict = _json_jdg.loads(_text)
            return bool(_verdict.get("consistent", False))
        except Exception:
            logger.debug(
                "LLM derivation consistency judge failed, fallback heuristic",
                exc_info=True,
            )
        # 2. 降级: token Jaccard > 0.3 视为一致
        # ponytail: heuristic ceiling 是抓不到同义改写 (例如 "use ENCUT=520"
        # vs "ENCUT should be 520"); 升级路径换 sentence-transformers cosine > 0.85
        try:
            import re as _re_jdg
            _tok_a = set(_re_jdg.findall(r"\w+", blind_derivation.lower()))
            _tok_b = set(_re_jdg.findall(r"\w+", orig_reasoning.lower()))
            if not _tok_a or not _tok_b:
                return None
            _jac = len(_tok_a & _tok_b) / len(_tok_a | _tok_b)
            return _jac > 0.3
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return None



    async def _invert_failure_trace(
        self, input_params: str, failed_result: str, failure_mode: str = "",
    ) -> str:
        """Task 3: 失败时反推完整 failure reasoning trace.

        复用 blind_reconstructor 的 dispatch 模式, 调 failure_inverter subagent
        从 (input, failed_result) 反推"为什么这个 input 会导致这个 failed result".
        返回结构化文本含 [FAILURE TRACE]/[BREAK POINT]/[COUNTERFACTUAL] 三段.
        任何失败返回空串 — 失败反推本身是 enhancement, 不能阻塞主流程.

        ponytail: 不引入新 dispatch 路径, 复用 SubagentDispatch("failure_inverter").
        升级路径: 多次 sample 投票降单次偏差; 当前一次 dispatch 够用.
        """
        if self._agent_factory is None:
            logger.debug("Task 3 failure inversion: no agent_factory, skip")
            return ""
        from huginn.agents.subagent import SubagentDispatch
        _dispatch = SubagentDispatch()
        _task = (
            f"Invert the failure reasoning: why does this input lead to this "
            f"failed result? At which step does it break? What counterfactual "
            f"change could make it work? Output JSON.\n\n"
            f"Input parameters:\n{input_params}\n\n"
            f"Failed result:\n{failed_result}\n\n"
            f"Failure mode: {failure_mode}"
        )
        _ctx = {"agent_factory": self._agent_factory}
        try:
            _res = await _dispatch.dispatch("failure_inverter", _task, context=_ctx)
        except Exception:
            logger.debug("failure inversion dispatch failed", exc_info=True)
            return ""
        if not _res.success or not _res.summary:
            return ""
        import json as _json_inv
        try:
            _inv = _json_inv.loads(_res.summary)
        except Exception:
            logger.debug("failure inversion summary not JSON, skip")
            return ""
        _reasoning = str(_inv.get("failure_reasoning", "") or "").strip()
        _break = str(_inv.get("failure_point", "") or "").strip()
        _counter = str(_inv.get("counterfactual", "") or "").strip()
        if not _reasoning:
            return ""
        return (
            f"[FAILURE TRACE]\n{_reasoning}\n\n"
            f"[BREAK POINT]\n{_break}\n\n"
            f"[COUNTERFACTUAL]\n{_counter}"
        )



    async def _abstract_skill_if_ready(self) -> None:
        """P0 Task 1: 扫 longterm trace clusters, ≥3 条 + 无 skill 时归纳 skill.

        Voyager-style skill library: 同簇 N 条 reasoning trace 喂 skill_abstractor
        subagent, 归纳成参数化 skill (function_name/params/precondition/reasoning_template),
        存 typed memory (memory_type='skill') 给后续 retrieve 命中直接注入模板.

        toggle HUGINN_SKILL_ABSTRACTION 默认 off — 不开不消耗 dispatch 配额.
        失败 try/except 全包, skill 抽象是 enhancement 不阻塞主流程.

        ponytail: 复用 _invert_failure_trace 的 dispatch 模式 + typed memory 存储,
        不引新 dispatch 路径. 升级路径: 多次 sample 投票降单次偏差 + trace 写入时
        带 dimension tag 让 _cluster_traces_by_dimension 真能按 dimension 分簇.
        """
        if os.environ.get("HUGINN_SKILL_ABSTRACTION", "0") != "1":
            return
        if self._agent_factory is None:
            return
        try:
            clusters = self.memory.longterm._cluster_traces_by_dimension()
            for cluster_key, traces in clusters.items():
                if len(traces) < 3:
                    continue
                if self.memory.longterm._get_skill_for_cluster(cluster_key) is not None:
                    continue
                # 前 5 条 trace 的 reasoning 内容喂 abstractor, 截到 1500 字控 prompt
                sample = traces[:5]
                blocks: list[str] = []
                trace_ids: list[str] = []
                for i, tr in enumerate(sample, 1):
                    tid = tr.get("id", f"trace_{i}")
                    trace_ids.append(tid)
                    content = tr.get("content", "") or ""
                    blocks.append(f"--- Trace {i} (id={tid}) ---\n{content[:1500]}")
                _task = (
                    f"Abstract a reusable parameterized skill from the following "
                    f"{len(sample)} reasoning traces (cluster={cluster_key}). "
                    f"Output JSON.\n\n" + "\n\n".join(blocks)
                )
                from huginn.agents.subagent import SubagentDispatch
                _dispatch = SubagentDispatch()
                _ctx = {"agent_factory": self._agent_factory}
                try:
                    _res = await _dispatch.dispatch(
                        "skill_abstractor", _task, context=_ctx
                    )
                except Exception:
                    logger.debug("skill_abstractor dispatch failed", exc_info=True)
                    continue
                if not _res.success or not _res.summary:
                    continue
                try:
                    _skill = json.loads(_res.summary)
                except Exception:
                    logger.debug("skill_abstractor summary not JSON, skip")
                    continue
                # function_name 为空 = traces 太散, abstractor 自己放弃
                if not str(_skill.get("function_name", "")).strip():
                    continue
                _skill["cluster_key"] = cluster_key
                _skill["source_traces"] = trace_ids
                _tags = [
                    "skill",
                    f"cluster_key:{cluster_key}",
                    f"applicable_dimension:{_skill.get('applicable_dimension', 'unknown')}",
                ]
                try:
                    self.memory.remember_typed(
                        content=json.dumps(_skill, ensure_ascii=False),
                        memory_type="skill",
                        tags=_tags,
                        source="skill_abstraction",
                        importance=0.7,
                        tier="long",
                    )
                    logger.info(
                        "skill abstracted for cluster=%s function=%s",
                        cluster_key, _skill.get("function_name"),
                    )
                except Exception:
                    logger.debug("skill store failed", exc_info=True)
        except Exception:
            logger.debug("skill abstraction failed", exc_info=True)



    async def _synthesize_self_goal_if_ready(self) -> None:
        """P1 Task 7: scan self_model weak clusters, synthesize self-goal.

        rate<0.3 + n>=5 -> pending_confirmation goal (origin=self).
        toggle HUGINN_SELF_GOAL_SYNTHESIS off. dedup by cluster_key.
        ponytail: reuse self_model (T4) + GoalStore, no new stat path.
        """
        if os.environ.get("HUGINN_SELF_GOAL_SYNTHESIS", "0") != "1":
            return
        _mem = getattr(self, "memory", None)
        if _mem is None or not hasattr(_mem, "longterm"):
            return
        try:
            _sm = _mem.longterm.get_self_model()
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return
        if not _sm:
            return
        try:
            from huginn.autoloop.goal_store import get_goal_store
            _gs = get_goal_store()
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return
        _existing: set[str] = set()
        try:
            for _g in _gs.list_goals():
                if getattr(_g, "origin", "user") != "self":
                    continue
                if _g.status in ("pending_confirmation", "active"):
                    _ck = (_g.metadata or {}).get("cluster_key")
                    if _ck:
                        _existing.add(_ck)
        except Exception:
            logger.debug("cluster_key dedup skipped", exc_info=True)
        _synth = 0
        for _key, _v in _sm.items():
            _rate = _v.get("rate")
            _n = _v.get("success", 0) + _v.get("failure", 0)
            if not isinstance(_rate, (int, float)):
                continue
            if _rate >= 0.3 or _n < 5:
                continue
            _dim = _v.get("dimension", "unknown")
            _ht = _v.get("hyp_type", "unknown")
            _ck = f"{_dim}/{_ht}"
            if _ck in _existing:
                continue
            _text = f"master {_dim} {_ht} verification (rate={_rate:.2f}, n={_n})"
            try:
                _gs.create_goal(_text, origin="self", status="pending_confirmation",
                    metadata={"cluster_key": _ck, "rate": _rate, "sample_size": _n})
                _synth += 1
                _existing.add(_ck)
                logger.info("self-goal synthesized: %s r=%.2f n=%d", _ck, _rate, _n)
            except Exception:
                logger.debug("self-goal create failed", exc_info=True)
        if _synth:
            logger.info("self-goal synthesis: %d pending_confirmation", _synth)



    def _compute_verification_budget(
        self, hypothesis_id: str, informativeness: float,
    ) -> None:
        """P0 Task 3: 映射 informativeness 到 verification_budget, 写进 node.evidence.

        基础预算 = informativeness * MAX (blind=5, ce=3).
        查 self_model: 该体系历史成功率低 → *1.5, 高 → *0.7, 无统计 → 基础.
        wall_clock 全局预算耗尽 → 强制 0.
        写 evidence["verification_budget"] = {blind_rounds, ce_rounds, rationale}.

        ponytail: 复用 get_self_model 接口, 不引新统计路径.
        ceiling: self_model 按 (dimension, hyp_type) 聚合, 关键词命中粗.
        """
        _MAX_BLIND = 5
        _MAX_CE = 3
        try:
            _node = self.hypothesis_graph._nodes.get(hypothesis_id)
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return
        if _node is None:
            return
        _info = max(0.0, min(1.0, float(informativeness)))
        _blind = _info * _MAX_BLIND
        _ce = _info * _MAX_CE
        _rationale = "base_informativeness"
        # 查 self_model 调整预算
        try:
            _dim = _node.dimension or ""
            _htype = "other"
            _mem = getattr(self, "memory", None)
            if _mem is not None and hasattr(_mem, "longterm"):
                with contextlib.suppress(Exception):
                    _htype = _mem.longterm._infer_hyp_type(_node.statement)
                _sm = _mem.longterm.get_self_model(
                    dimension=_dim or None, hyp_type=_htype,
                )
                if _sm:
                    # 多 key 取平均 rate (过滤后通常 1 个 key, 防御性聚合)
                    _tot = 0.0
                    _cnt = 0
                    for _v in _sm.values():
                        _r = _v.get("rate")
                        if isinstance(_r, (int, float)):
                            _tot += float(_r)
                            _cnt += 1
                    if _cnt > 0:
                        _rate = _tot / _cnt
                        if _rate < 0.4:
                            _blind *= 1.5
                            _ce *= 1.5
                            _rationale = "low_self_efficacy"
                        elif _rate > 0.7:
                            _blind *= 0.7
                            _ce *= 0.7
                            _rationale = "high_self_efficacy"
                        else:
                            _rationale = "normal_self_efficacy"
                else:
                    _rationale = "no_self_model"
        except Exception:
            logger.debug("self_model lookup for budget failed", exc_info=True)
            _rationale = "no_self_model"
        # wall_clock 全局上限: 耗尽则预算强制 0
        try:
            if os.environ.get("HUGINN_PERSISTENT_GOAL_MODE", "0") == "1":
                from huginn.autoloop.goal_store import get_goal_store
                _gs = get_goal_store()
                _ag = _gs.get_active()
                if _ag is not None and _gs.wall_clock_expired(_ag.id):
                    _blind = 0.0
                    _ce = 0.0
                    _rationale = "wall_clock_exhausted"
        except Exception:
            logger.debug("wall_clock budget check failed", exc_info=True)
        # 轮数取整 (正数 floor), 存 node evidence
        _node.evidence["verification_budget"] = {
            "blind_rounds": max(0, int(_blind)),
            "ce_rounds": max(0, int(_ce)),
            "rationale": _rationale,
            "informativeness": _info,
        }
        _node.evidence.setdefault("blind_rounds_used", 0)
        _node.evidence.setdefault("ce_rounds_used", 0)



    async def _run_pytest(self) -> dict[str, Any]:
        """Run pytest in workspace, return results dict."""
        import subprocess

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["python", "-m", "pytest", "-x", "-q", "--tb=line"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                "tests_passed": result.returncode == 0,
                "test_output": result.stdout + result.stderr,
            }
        except Exception as e:
            return {"test_output": f"Test execution error: {e}"}



    async def _run_benchmark(self) -> dict[str, Any]:
        """Run BenchmarkRunner, return results dict."""
        try:
            from huginn.bench.runner import BenchmarkRunner

            runner = BenchmarkRunner()
            report = await asyncio.to_thread(runner.run, categories=["math", "coding"])
            return {
                "passed": report.passed,
                "failed": report.failed,
                "skipped": report.skipped,
            }
        except Exception:
            logger.warning("BenchmarkRunner failed", exc_info=True)
            return {}



    async def _safe_emergent_complexity(
        self, execution_result: Any, results: dict[str, Any]
    ) -> None:
        """Compute emergent complexity, mutate results in place."""
        try:
            from huginn.validation.emergent_complexity import compute_ec

            results["emergent_complexity"] = compute_ec(execution_result, results)
            ec_score = results["emergent_complexity"].get("ec_score", 0)
            if ec_score < 0.2 and self._iteration > 0:
                ec_hint = f"EC={ec_score:.2f}: low emergent complexity, try diverse tools or cross-domain reasoning"
                self._speculator_hint = (
                    (self._speculator_hint + "\n" + ec_hint).strip()
                    if self._speculator_hint
                    else ec_hint
                )
        except Exception as e:
            results["emergent_complexity_error"] = str(e)



    async def _safe_literature_comparison(
        self, execution_result: Any, results: dict[str, Any]
    ) -> None:
        """Run literature comparison, mutate results in place."""
        try:
            lit_comp = await self._literature_comparison(execution_result)
            if lit_comp:
                results["literature_comparison"] = lit_comp
                # 平铺 high_confidence_claims 给 red_team._literature_consensus_check
                # 消费. evidence["literature_claims"] 是 red_team 约定的 key.
                high_claims = lit_comp.get("high_confidence_claims") or []
                if high_claims:
                    results["literature_claims"] = high_claims
        except Exception as e:
            results["literature_comparison_error"] = str(e)



    async def _literature_comparison(self, execution_result: Any) -> dict[str, Any]:
        """Extract numeric results, look up literature benchmarks, run innovation
        signal detection. Best-effort — any failure just skips that property.

        Returns {property_key: InnovationSignal} for properties where literature
        data was found. Empty dict if nothing to compare.
        """

        from huginn.autoloop.engine import _LIT_PROPERTY_MAP
        if not isinstance(execution_result, dict):
            return {}

        result_data = (
            execution_result.get("result_data") or execution_result.get("parsed") or {}
        )
        if not isinstance(result_data, dict):
            return {}

        # system / formula: benchmark_lookup 必须知道查什么材料
        system = None
        for key in ("formula", "system", "material", "compound"):
            val = result_data.get(key) or execution_result.get(key)
            if val:
                system = str(val)
                break
        if not system:
            return {}

        # 抽数值: 扁平 key + 嵌套 lattice_params
        numerics: dict[str, float] = {}
        for key in _LIT_PROPERTY_MAP:
            val = result_data.get(key)
            if val is not None:
                try:
                    numerics[key] = float(val)
                except (TypeError, ValueError):
                    logger.warning(
                        "error in _literature_comparison: numeric property cast failed",
                        exc_info=True,
                    )
        lattice = result_data.get("lattice_params") or {}
        if isinstance(lattice, dict):
            for param in ("a", "b", "c"):
                val = lattice.get(param)
                if val is not None:
                    try:
                        numerics[f"lattice_{param}"] = float(val)
                    except (TypeError, ValueError):
                        logger.warning(
                            "error in _literature_comparison: lattice param cast failed",
                            exc_info=True,
                        )

        if not numerics:
            return {}

        try:
            from huginn.tools.literature import LiteratureInput, LiteratureTool
            from huginn.validation.innovation_signal import InnovationSignalDetector
        except ImportError:
            logger.debug("best-effort op failed", exc_info=True)
            return {}

        tool = LiteratureTool()
        tool_ctx = ToolContext(
            session_id=f"litcmp_{uuid.uuid4().hex[:8]}",
            workspace=str(self.workspace),
            config=self.settings,
        )
        detector = InnovationSignalDetector()

        comparison: dict[str, Any] = {}
        for prop_key, agent_value in numerics.items():
            prop_name = _LIT_PROPERTY_MAP.get(prop_key, prop_key)
            try:
                res = await tool.call(
                    LiteratureInput(
                        action="benchmark_lookup",
                        system=system,
                        property=prop_name,
                        max_results=10,
                    ),
                    tool_ctx,
                )
                if not res.success or not res.data:
                    continue
                reported = res.data.get("reported_values") or []
                lit_values = [r["value"] for r in reported if "value" in r]
                if not lit_values:
                    continue
                signal = detector.detect(prop_key, agent_value, lit_values)
                comparison[prop_key] = signal
            except Exception:
                logger.debug("best-effort op failed", exc_info=True)
                continue

        # 知识注入层: multi_review 产 high_confidence_claims, 平铺到 comparison
        # 给 red_team._literature_consensus_check 消费 (evidence["literature_claims"])
        # ponytail: 2 透镜 + max_results=5 控成本. 失败无所谓, comparison 已有 benchmark 部分
        try:
            mr_res = await tool.call(
                LiteratureInput(
                    action="multi_review",
                    query=system,
                    max_results=5,
                    lenses=["methodology", "limitations"],
                    verify_claims=True,
                ),
                tool_ctx,
            )
            if mr_res.success and mr_res.data:
                high_claims = mr_res.data.get("high_confidence_claims") or []
                if high_claims:
                    comparison["high_confidence_claims"] = high_claims
        except Exception:
            logger.debug(
                "multi_review in _literature_comparison failed (non-fatal)",
                exc_info=True,
            )

        return comparison



    @staticmethod
    def _summarize_for_kb(execution_result: Any, results: dict[str, Any]) -> str:
        """把 execution_result + validation results 拍扁成短串当 KB query.
        给 reviewer 检索已知 first-principles 结论用, 失败无所谓."""
        try:
            parts: list[str] = []
            if isinstance(execution_result, dict):
                for k in ("result_type", "equations", "lagrangian", "summary"):
                    v = execution_result.get(k)
                    if v:
                        parts.append(str(v)[:120])
            for k in ("tests_passed", "constraints_satisfied"):
                v = results.get(k)
                if v is not None:
                    parts.append(f"{k}={v}")
            return " ".join(parts)[:400]
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return ""



    def _detect_thinking_collapse(self, execution_result: Any) -> dict[str, Any] | None:
        """检查 LLM 输出是否陷入重复推理 / 发散 / 工具调用循环.

        三条规则, 纯文本分析不需要 LLM:
          1. 相同短语 (5 词 n-gram) 出现 3+ 次 → 重复推理路径
          2. 输出 > 200 词但 unique word ratio < 0.3 → 发散但不前进
          3. 相同工具 + 相同参数出现 2+ 次 → 工具循环

        检测到任一信号就返回 dict, 否则 None.
        """
        from collections import Counter

        text = self._extract_text(execution_result)
        if not text or len(text.strip()) < 20:
            return None

        signals: dict[str, Any] = {}

        # Rule 1: 重复短语 — 5 词 n-gram 出现 3+ 次
        words = text.lower().split()
        if len(words) >= 10:
            ngrams = [" ".join(words[i : i + 5]) for i in range(len(words) - 4)]
            counts = Counter(ngrams)
            repeated = [(p, c) for p, c in counts.items() if c >= 3]
            if repeated:
                repeated.sort(key=lambda x: -x[1])
                signals["repeated_phrases"] = repeated[:5]

        # Rule 2: 发散推理 — 长文本但词汇丰富度低
        if len(words) > 200:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                signals["divergent_reasoning"] = {
                    "word_count": len(words),
                    "unique_ratio": round(unique_ratio, 3),
                }

        # Rule 3: 工具调用循环 — 相同工具 + 相同参数 2+ 次
        if isinstance(execution_result, dict):
            loops = self._find_tool_call_loops(execution_result)
            if loops:
                signals["tool_call_loops"] = loops

        if not signals:
            return None

        # 严重度: 有重复短语或工具循环 = high, 只有发散 = medium
        has_loop = bool(signals.get("tool_call_loops"))
        has_repeat = bool(signals.get("repeated_phrases"))
        signals["severity"] = "high" if (has_loop or has_repeat) else "medium"
        return signals



    @staticmethod
    def _find_tool_call_loops(execution_result: dict) -> list[dict[str, Any]]:
        """从 execution_result 里找重复的工具调用 (同工具 + 同参数 2+ 次)."""

        calls = (
            execution_result.get("tool_calls")
            or execution_result.get("steps")
            or execution_result.get("actions")
            or []
        )
        if not isinstance(calls, list):
            return []

        seen: dict[str, int] = {}
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("tool") or call.get("name") or call.get("action") or ""
            params = call.get("input") or call.get("params") or call.get("args") or {}
            try:
                payload = name + json.dumps(params, sort_keys=True, default=str)
            except Exception:
                payload = name + str(params)
            key = hashlib.sha256(payload.encode()).hexdigest()[:12]
            seen[key] = seen.get(key, 0) + 1

        return [{"call_hash": k, "count": c} for k, c in seen.items() if c >= 2]



    def _load_trajectory_action_history(self, limit: int = 20) -> list[list[str]]:
        """从 workspace/.huginn/trajectories/ 加载历史 run 的 action 序列.

        每个 trajectory.json 的 spans 里 phase 名就是 action. 抽出来给
        trajectory_match 当 history 用 (VF2 子图同构 prefix 匹配).

        额外填充 self._traj_run_ids (平行数组), 供 _learn 用 run_id 反查 KB
        做 ±ε 反馈 (C3 闭环).

        ponytail: 只读最近 limit 个文件, 不做全量索引. 升级路径: KB 索引 + 元数据过滤.
        """
        traj_dir = self.workspace / HUGINN_DIR_NAME / "trajectories"
        if not traj_dir.exists():
            self._traj_run_ids = []
            return []
        history: list[list[str]] = []
        run_ids: list[str] = []
        try:
            files = sorted(
                traj_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:limit]
        except Exception:
            self._traj_run_ids = []
            return []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                logger.debug("best-effort op failed", exc_info=True)
                continue
            spans = data.get("spans") or []
            actions = [
                s.get("phase") or s.get("name") or ""
                for s in spans
                if isinstance(s, dict)
            ]
            actions = [a for a in actions if a]
            if len(actions) >= 2:
                history.append(actions)
                # 文件名 stem = run_id (cognitive_loop save_trajectory 用 run_id 命名)
                run_ids.append(f.stem)
        self._traj_run_ids = run_ids
        return history



    def _check_stuck(self, action_history: list[str]) -> dict[str, Any] | None:
        """G2 主入口: 周期检测 + 历史轨迹 prefix 匹配.

        - cycle_detect.is_stuck: 当前 action 序列是否陷入周期 (M3, O(n²) 暴力).
        - trajectory_match: 当前序列是否是某历史成功轨迹的 prefix (M2, VF2 子图同构).
          匹配到 → 取下一步作为建议, 注入 _speculator_hint.

        返回 None (无信号) 或 dict:
          {"type": "cycle", "period": lam, "advice": "..."}
          {"type": "match", "history_id": i, "similarity": s, "next_step": X, "advice": "..."}

        启用条件: HUGINN_EXTREME_DISPATCH=1, 或长程任务 (max_iterations >= 20).
        短程任务默认关, 省 cycle/trajectory 计算.
        """
        # 长程任务 (max_iterations >= 20) 默认开 cycle/trajectory 检测,
        # 短程任务仍需 HUGINN_EXTREME_DISPATCH=1 才开 (省计算).
        _max_iter = getattr(self, "_max_iterations", 10)
        from huginn.feature_flags import FeatureFlags
        _extreme = FeatureFlags.shared().is_enabled("extreme_dispatch")
        if not (_extreme or _max_iter >= 20):
            return None
        if len(action_history) < 4:
            return None  # 太短不检
        try:
            from huginn.knowledge.trajectory_pattern import trajectory_match
            from huginn.runtime.cycle_detect import detect_cycle, is_stuck
        except ImportError:
            logger.debug("best-effort op failed", exc_info=True)
            return None

        # M3: 周期检测 (在当前 run 内)
        try:
            if is_stuck(action_history, min_cycle_len=2, min_repeats=2):
                cycle = detect_cycle(action_history, min_cycle_len=2, min_repeats=2)
                lam = cycle[1] if cycle else 0
                return {
                    "type": "cycle",
                    "period": lam,
                    "advice": (
                        f"action 序列陷入周期 (period={lam}), 强制 pivot. "
                        f"最近 {len(action_history)} 步: {action_history[-8:]}"
                    ),
                }
        except Exception:
            logger.debug("G2 cycle_detect failed (non-fatal)", exc_info=True)

        # M2: 历史轨迹 prefix 匹配 (跨 run)
        try:
            history = getattr(self, "_traj_history", None) or []
            if history:
                match = trajectory_match(
                    action_history, history, min_similarity=0.4,
                )
                if match and match.get("next_step"):
                    return {
                        "type": "match",
                        "history_id": match["history_id"],
                        "similarity": match["similarity"],
                        "next_step": match["next_step"],
                        "advice": (
                            f"匹配历史成功轨迹 (sim={match['similarity']:.2f}), "
                            f"考虑下一步: {match['next_step']}"
                        ),
                    }
        except Exception:
            logger.debug("G2 trajectory_match failed (non-fatal)", exc_info=True)

        return None



    @staticmethod
    def _extract_text(execution_result: Any) -> str:
        """从 execution_result 里抽文本, 给坍塌检测做分析用."""
        if execution_result is None:
            return ""
        if isinstance(execution_result, str):
            return execution_result
        if not isinstance(execution_result, dict):
            return str(execution_result)

        parts: list[str] = []
        for key in (
            "summary",
            "description",
            "result_data",
            "output",
            "error",
            "reasoning",
            "plan",
            "hypothesis",
        ):
            v = execution_result.get(key)
            if v:
                parts.append(str(v))
        # 嵌套的 steps / tool_calls 里的文本也抽出来
        for key in ("steps", "tool_calls", "actions"):
            items = execution_result.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        for sk in ("description", "output", "result", "error"):
                            sv = item.get(sk)
                            if sv:
                                parts.append(str(sv))
        return " ".join(parts)



    def _compute_surprise(self, prediction: str, actual: str) -> float:
        """JEPA 式预测误差: 预测文本 vs 实际文本的语义距离.

        ponytail: 用关键词 Jaccard 距离代替真正的嵌入余弦距离.
        纯文本操作, 零依赖, 零 LLM 调用. 对于"预测说了 energy, 实际也出了
        energy"这种常见场景已经够用. 升级路径: 用 sentence-transformers
        算 cosine distance, 或训练专门的 JEPA 编码器.
        """
        s = self._compute_surprise_robust(prediction, actual)
        return s["worst"]



    def _compute_surprise_robust(
        self, prediction: str, actual: str
    ) -> dict[str, float]:
        """分布鲁棒 surprise 估计.

        对 keyword 提取做多种扰动 (不同 stopword 集 / n-gram / 阈值),
        取 worst-case 作为决策依据. 这避免单一扰动下 surprise 被低估.

        返回 {mean, worst, std, point}:
        - point: 原始 Jaccard 距离 (兼容旧逻辑)
        - worst: 多扰动下的最大值 (决策用)
        - mean: 多扰动平均值 (趋势分析用)
        - std: 多扰动标准差 (置信度信号)
        """
        if not prediction or not actual:
            return {"mean": 0.0, "worst": 0.0, "std": 0.0, "point": 0.0}

        import statistics

        # 扰动 1: 标准停用词集
        stop1 = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "and",
            "or",
            "not",
            "this",
            "that",
            "it",
            "with",
            "from",
            "by",
            "as",
            "will",
            "can",
            "may",
        }
        # 扰动 2: 更激进的停用词集 (去掉更多常见词)
        stop2 = stop1 | {
            "energy",
            "result",
            "value",
            "system",
            "model",
            "data",
            "using",
            "shown",
            "show",
            "also",
            "which",
            "has",
            "have",
            "had",
            "been",
            "were",
            "more",
            "than",
        }
        # 扰动 3: 只保留长关键词 (>=5 chars)
        # 扰动 4: bigram Jaccard

        def keywords(text: str, stop: set[str], min_len: int = 3) -> set[str]:
            words = __import__("re").findall(r"[a-zA-Z_]\w{2,}", text.lower())
            return {w for w in words if w not in stop and len(w) >= min_len}

        def bigrams(text: str) -> set[str]:
            words = __import__("re").findall(r"[a-zA-Z_]\w{2,}", text.lower())
            return {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)}

        def jaccard(a: set, b: set) -> float:
            if not a and not b:
                return 0.0
            union = a | b
            if not union:
                return 0.0
            return 1.0 - len(a & b) / len(union)

        pred_kw1 = keywords(prediction, stop1)
        actual_kw1 = keywords(actual, stop1)
        pred_kw2 = keywords(prediction, stop2)
        actual_kw2 = keywords(actual, stop2)
        pred_kw3 = keywords(prediction, stop1, min_len=5)
        actual_kw3 = keywords(actual, stop1, min_len=5)
        pred_bg = bigrams(prediction)
        actual_bg = bigrams(actual)

        estimates = [
            jaccard(pred_kw1, actual_kw1),
            jaccard(pred_kw2, actual_kw2),
            jaccard(pred_kw3, actual_kw3),
            jaccard(pred_bg, actual_bg) if pred_bg or actual_bg else 0.0,
        ]

        return {
            "point": estimates[0],
            "mean": statistics.mean(estimates),
            "worst": max(estimates),
            "std": statistics.stdev(estimates) if len(estimates) > 1 else 0.0,
        }



    async def _generative_verify(
        self, execution_result: Any, results: dict[str, Any]
    ) -> dict[str, Any] | None:
        """用 verification_model 给 agent 输出打 0-1 质量分.

        分数 < 0.5 标记 needs_retry. verification_model 不可用时返回
        None, 让上层降级到规则检查 (thinking_collapse 等).
        """
        if self.verification_model is None:
            return None

        from huginn.harness.phase_spec import get_phase_extra
        _needs_retry_threshold = get_phase_extra("_validate", "needs_retry_threshold", 0.5)

        text = self._extract_text(execution_result)
        if not text or len(text.strip()) < 10:
            return None

        # 截断防止 prompt 爆炸
        snippet = text[:2000]
        collapse = results.get("thinking_collapse", {})
        collapse_hint = ""
        if collapse:
            collapse_hint = f"\nNote: automated checks detected: {json.dumps(collapse, default=str)[:300]}"

        # 注入历史记忆, 检查本次结果是否与历史迭代结果矛盾
        memory_hint = ""
        try:
            _mem_text = self._build_memory_text(query=snippet[:200])
            if _mem_text:
                memory_hint = (
                    f"\nPast iterations memory:\n{_mem_text}\n"
                    "Cross-check: does the current result contradict any historical finding above?\n"
                    "If yes, note the contradiction in 'reason'.\n"
                )
        except Exception:
            logger.debug(
                "_build_memory_text failed — validate prompt missing cross-check",
                exc_info=True,
            )

        # P0: 迭代历史栈注入 — 让 validate 看到前几轮的 hypothesis/val_status,
        # 判断当前结果是否在重复已失败的路径. ponytail: 从 self 镜像读, 最近 5 轮.
        _iter_hist = getattr(self, "_iteration_history", None) or []
        _prev_block = ""
        if _iter_hist:
            _lines = []
            for _h in _iter_hist[-5:]:
                _lines.append(
                    f"  iter{_h.get('iter', '?')}: {_h.get('action', '?')} "
                    f"({_h.get('val_status', 'none')})"
                    + (f" [{_h.get('plan_mode')}]" if _h.get('plan_mode') else "")
                    + (f" → {_h.get('advice', '')[:80]}" if _h.get('redirect') else "")
                )
            _prev_block = (
                "\nPrevious iterations (avoid re-validating already-failed paths):\n"
                + "\n".join(_lines) + "\n"
            )

        prompt = (
            "You are a verification model. Score the quality of this agent output "
            "from 0.0 to 1.0.\n"
            "1.0 = well-reasoned, complete, correct.\n"
            "0.5 = acceptable but has issues.\n"
            "0.0 = poor, incorrect, or incomplete.\n"
            "Also check evidence chain (RCBench failure mode: evidence mismatch):\n"
            "- Does the conclusion have specific data/numbers supporting it?\n"
            "- Are claims grounded in the execution results, not assumed?\n"
            "- Is there a clear data→inference→conclusion link?\n"
            "Also describe failure mode (Dream Layer insight: how it crashes = new discovery):\n"
            "- If this hypothesis is WRONG, in what specific way would it fail?\n"
            "- What would the system look like if the opposite were true?\n"
            f"{collapse_hint}{memory_hint}{_prev_block}\n\n"
            f"Agent output:\n{snippet}\n\n"
            "Respond with ONLY a JSON object: "
            '{"score": <float>, "evidence_score": <float 0-1>, '
            '"reason": "<brief>", "evidence_gap": "<what data is missing>", '
            '"failure_mode": "<how it would crash if wrong>"}'
        )

        resp = await self._llm_chat(prompt, model=self.verification_model)
        score, reason, evidence_score, evidence_gap, failure_mode = (
            self._parse_verify_score(resp)
        )

        return {
            "score": score,
            "reason": reason,
            "needs_retry": score < _needs_retry_threshold,
            "evidence_score": evidence_score,
            "evidence_gap": evidence_gap,
            "failure_mode": failure_mode,
        }



    @staticmethod
    def _parse_verify_score(resp: str) -> tuple[float, str, float, str, str]:
        """Parse score, reason, evidence_score, evidence_gap, failure_mode from LLM response."""

        if not resp:
            return 0.5, "empty response", 0.5, "", ""

        # try JSON first
        try:
            data = json.loads(resp.strip())
            score = float(data.get("score", 0.5))
            reason = str(data.get("reason", ""))
            ev_score = float(data.get("evidence_score", 0.5))
            ev_gap = str(data.get("evidence_gap", ""))
            fail_mode = str(data.get("failure_mode", ""))
            return score, reason, ev_score, ev_gap, fail_mode
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "error in _parse_verify_score: JSON parse failed, falling back to regex",
                exc_info=True,
            )

        # fallback: regex for first float
        m = re.search(r"([01]\.\d+|[01])\b", resp)
        if m:
            return float(m.group(1)), resp[:200], 0.5, "", ""

        return 0.5, resp[:200], 0.5, "", ""



    def _query_kb_reference(self, equations: str, lagrangian: str) -> list[dict]:
        """查 KB 拿 first-principles 参考块. 把 equations + lagrangian 拼成
        query 串, 命中返回 [{text, source}], 失败/空都返回 []."""
        query = " ".join(filter(None, [equations, lagrangian])).strip()
        if not query:
            return []
        kb = self._get_kb()
        if kb is None:
            return []
        try:
            if kb.count() == 0:
                return []
            chunks = kb.query(f"conservation law variational {query}", top_k=2)
            return [
                {"text": (c.get("text") or "")[:300], "source": c.get("source", "")}
                for c in chunks
                if c.get("text")
            ]
        except Exception:
            return []



    @staticmethod
    def _build_reviewer_prompt(
        execution_result: Any,
        results: dict[str, Any],
        kb_text: str = "",
    ) -> str:
        """构造让 reviewer persona 点评执行结果的 prompt."""
        try:
            exec_blob = json.dumps(execution_result, ensure_ascii=False, default=str)[
                :1500
            ]
        except Exception:
            exec_blob = str(execution_result)[:1500]
        try:
            res_blob = json.dumps(results, ensure_ascii=False, default=str)[:1500]
        except Exception:
            res_blob = str(results)[:1500]
        kb_section = f"\n{kb_text}\n" if kb_text else ""
        return (
            "Below is the execution result and validation summary from an "
            "autonomous materials-science research loop iteration.\n\n"
            f"Execution result:\n{exec_blob}\n\n"
            f"Validation summary:\n{res_blob}\n"
            f"{kb_section}"
            "As a critical peer reviewer, point out:\n"
            "1. Any methodological weakness or missing convergence check.\n"
            "2. Whether the result is reproducible and benchmarked.\n"
            "3. Whether the result aligns with the domain knowledge context above "
            "(if any), or contradicts known first-principles.\n"
            "4. Concrete next-step improvements.\n"
            "Be concise and direct."
        )



    async def _learn(
        self, hypothesis: str, plan: dict[str, Any], validation: dict[str, Any]
    ) -> dict[str, Any]:
        """Learn from iteration results — update memory, knowledge graph, evolution rules.

        D2: 返回 summary dict, 让 caller (execute_fn 的 learn 分支) 写入
        cog["last_learn_summary"], 下轮 decider 看到正反馈. 之前返回 None,
        LLM 选 learn 没反馈, 下轮容易重复 learn.
        ponytail: 只返 4 个标量字段, 不暴露内部状态. 升级路径: 结构化
        summary 走专门 cog slot (cog["last_learn_detail"]).
        """
        # H4: importance 公式常量从 PhaseRegistry extra 取
        from huginn.harness.phase_spec import get_phase_extra
        _imp_default = get_phase_extra("_learn", "importance_default", 0.6)
        _imp_max = get_phase_extra("_learn", "importance_max", 0.9)
        r_phys = validation.get("r_phys") if isinstance(validation, dict) else None

        # Log to memory
        self.memory.add_message(
            "system",
            {
                "iteration": self._iteration,
                "hypothesis": hypothesis,
                "plan": plan,
                "validation": validation,
                "r_phys": r_phys,
            },
        )

        # Long-term memory: 把关键迭代写入 long-term, 下次 RAG 能检索到
        # 包含 visual primitives 和 surprise 分数, 跨会话完整恢复上下文
        try:
            persona_name = getattr(self, "_last_persona", "unknown")
            mem_content = f"iter {self._iteration}: {hypothesis[:120]}"
            # Visual primitives 入 memory, 下次 recall_for_prompt 能检索到数据形状
            visual_ctx = (
                validation.get("visual_primitives")
                if isinstance(validation, dict)
                else None
            )
            if visual_ctx:
                mem_content += f"\nVisual: {visual_ctx[:200]}"
            # Surprise 入 memory, 下次能检索到"这类任务预测准不准"
            pred_err = (
                validation.get("prediction_error", {})
                if isinstance(validation, dict)
                else {}
            )
            if pred_err:
                mem_content += f"\nSurprise: {pred_err.get('surprise', 0)} (worst: {pred_err.get('surprise_worst', pred_err.get('surprise', 0))}, std: {pred_err.get('surprise_std', 0)}) (predicted: {pred_err.get('predicted', '')[:80]})"
            # Persona 入 memory, 下次 _pick_hypothesis_persona 能查到历史效果
            mem_content += f"\nPersona: {persona_name}, r_phys: {r_phys}"
            # 结构化 tags: 供后续按 persona/r_phys/surprise 过滤检索
            _tags = [
                "autoloop",
                f"persona:{persona_name}",
                f"r_phys:{r_phys}" if r_phys is not None else "r_phys:none",
                (
                    f"surprise:{pred_err.get('surprise', 0):.2f}"
                    if pred_err
                    else "surprise:0"
                ),
            ]
            # C4: typed memory 默认 on, 走 remember_typed (含 iteration_result
            # + persona_id + status). 旧行 NULL 通过 lazy migrate 自动反推.
            # typed 写入失败时 fallback 到 legacy remember.
            try:
                # status: 用 validation 结果映射 (supported/refuted)
                _tests_ok = (
                    validation.get("tests_passed")
                    if isinstance(validation, dict)
                    else False
                )
                _typed_status = "supported" if _tests_ok else "refuted"
                self.memory.remember_typed(
                    content=mem_content,
                    memory_type="iteration_result",
                    run_id=getattr(self, "_run_id", None),
                    persona_id=persona_name,
                    status=_typed_status,
                    importance=_imp_default if r_phys is None else min(_imp_max, float(r_phys)),
                    tier="mid",
                    tags=_tags,
                )
                # C5: 额外写一条 persona_history (给 _pick_hypothesis_persona 查).
                # 不双写完整 content, 只记 persona + r_phys 摘要, 避免冗余.
                _ph_content = (
                    f"Persona: {persona_name}, r_phys: {r_phys}"
                    f", iter: {self._iteration}"
                )
                self.memory.remember_typed(
                    content=_ph_content,
                    memory_type="persona_history",
                    run_id=getattr(self, "_run_id", None),
                    persona_id=persona_name,
                    status=_typed_status,
                    importance=_imp_default if r_phys is None else min(_imp_max, float(r_phys)),
                    tier="mid",
                    tags=_tags,
                )
            except Exception:
                logger.debug(
                    "typed remember_typed failed, fallback to legacy remember",
                    exc_info=True,
                )
                self.memory.remember(
                    content=mem_content,
                    category="autoloop_iteration",
                    importance=_imp_default if r_phys is None else min(_imp_max, float(r_phys)),
                    tier="mid",
                    tags=_tags,
                )
        except Exception:
            logger.warning(
                "error in _learn: memory.remember iteration failed", exc_info=True
            )

        # 桥 A: surprise → hypothesis 触发. 高 surprise 说明结构预测跟实际对不上,
        # 喂回 _hypothesize 生成解释差异的新假设 (接通 trigger_alignment_surprise_hypothesis).
        # flag HUGINN_ALIGNMENT_SURPRISE_TRIGGER 默认 off, off 时行为不变. 失败非致命.
        # ponytail: 复用 _last_surprise (validate 已算好), 不重算. 阈值 2.0 跟 spec 对齐.
        if os.environ.get("HUGINN_ALIGNMENT_SURPRISE_TRIGGER", "0").lower() in ("1", "true"):
            _surprise = getattr(self, "_last_surprise", 0.0)
            if _surprise > 2.0:
                try:
                    await self.trigger_alignment_surprise_hypothesis(
                        [(hypothesis[:80], _surprise)]
                    )
                except Exception:
                    logger.debug(
                        "surprise → hypothesis trigger failed (non-fatal)",
                        exc_info=True,
                    )

        # 奖励回流: 把 R_phys 喂给 evolution engine, 驱动基于奖励的进化
        # 这是阶段4 单轨的核心闭环——物理校验分数真正影响 agent 后续行为
        if r_phys is not None:
            try:
                evolution = self._get_evolution()
                # 记录本次迭代的 reward, 供 evolve_from_rewards 消费
                evolution.logger.log_tool_call(
                    session_id=f"loop_{self._iteration}",
                    tool_name=plan.get("mode", "unknown"),
                    tool_input={"hypothesis": hypothesis, "plan": plan},
                    result=validation,
                    reward=r_phys,
                )
                reward_result = evolution.evolve_from_rewards()
                n_skills = len(reward_result["high_reward_skills"])
                n_patches = len(reward_result["low_reward_patches"])
                if n_skills or n_patches:
                    logger.info(
                        "reward evolution: +%d skills, +%d patches (R_phys=%.2f)",
                        n_skills,
                        n_patches,
                        r_phys,
                    )
            except Exception as e:
                logger.warning("reward evolution failed: %s", e)

        # H1: 给本轮 apply 过的 prompt patch 更新 Beta 信念. tests_passed 决定
        # success/fail. _apply_block_patches 在 _build_hypothesis_prompt /
        # _build_plan_prompt 里记录了 _last_applied_patches = (phase, [ids]).
        # ponytail: _last_applied_patches 只记最近一次, 多 phase 同轮 apply 时
        # 后者覆盖前者. 升级路径: 改成 dict[phase, ids] 完整追踪.
        try:
            applied = getattr(self, "_last_applied_patches", None)
            if applied:
                from huginn.harness.prompt_patch import PromptPatchStore
                _phase, _ids = applied
                _tests_passed = (
                    validation.get("tests_passed", False)
                    if isinstance(validation, dict)
                    else False
                )
                store = PromptPatchStore.get_instance()
                for _pid in _ids:
                    store.update_alpha_beta(_pid, success=bool(_tests_passed))
        except Exception:
            logger.debug("H1 patch Beta update failed", exc_info=True)

        # H3: 记录 (block_subset, workflow_params) 组合的 outcome 给 JointBandit.
        # block_subset 从 _last_hypothesis_blocks / _last_plan_blocks 拿 block 名;
        # workflow_params 留空 dict (reasoning-only 没 workflow stage 参数).
        # 分格存档: problem_domain 用本轮 hypothesis 摘要, 不同假设分格存储,
        # 避免单一 UCB 偏好已验证组合而饿死结构不同的激进方案.
        # ponytail: 不追踪 select 返回值, 从 _last_*_blocks 反推. 升级路径:
        # select_block_subset_for_phase 返回 (blocks, selected_names) 避免 重算.
        try:
            from huginn.harness.joint_optimizer import JointBandit
            from huginn.harness.joint_optimizer import _harness_enabled as _h3_on
            if _h3_on("harness_joint_optimizer"):
                _h3_phase = "hypothesize"
                _h3_blocks = getattr(self, "_last_hypothesis_blocks", None) or []
                if not _h3_blocks:
                    _h3_blocks = getattr(self, "_last_plan_blocks", None) or []
                    _h3_phase = "plan"
                if _h3_blocks:
                    _h3_subset = [n for n, _ in _h3_blocks]
                    _h3_success = bool(
                        validation.get("tests_passed", False)
                        if isinstance(validation, dict) else False
                    )
                    JointBandit.get_instance().record_joint_outcome(
                        _h3_phase, _h3_subset, {}, _h3_success,
                        problem_domain=str(hypothesis)[:64],
                    )
        except Exception:
            logger.debug("H3 joint record failed", exc_info=True)

        # Forest 回流: 如果是森林模式运行, 把 merged_graph 合并到本地假设图
        # 并写入 memory, 供后续迭代接续探索多树共识的结论.
        if self._merged_graph is not None:
            try:
                # 合并到本地 hypothesis_graph
                for node_id in self._merged_graph.nodes:
                    node = self._merged_graph.nodes.get(node_id)
                    if node and hasattr(node, "statement") and not any(
                        existing.statement == node.statement
                        for existing in self.hypothesis_graph.nodes.values()
                    ):
                            nid = self.hypothesis_graph.add_hypothesis(
                                statement=node.statement,
                                rationale=getattr(node, "rationale", ""),
                                testable_prediction=getattr(
                                    node, "testable_prediction", ""
                                ),
                            )
                            if nid is not None:
                                if getattr(node, "status", "") == "supported":
                                    self.hypothesis_graph.support(
                                        nid, getattr(node, "evidence", {})
                                    )
                                elif getattr(node, "status", "") == "refuted":
                                    self.hypothesis_graph.refute(
                                        nid, getattr(node, "evidence", {})
                                    )
                # 写入 memory
                graph_summary = f"Forest merged: {len(self._merged_graph.nodes)} nodes"
                self.memory.add_message(
                    "system",
                    {
                        "iteration": self._iteration,
                        "type": "forest_merge",
                        "graph_summary": graph_summary,
                    },
                )
                logger.info(
                    "Forest merged %d nodes into hypothesis_graph",
                    len(self._merged_graph.nodes),
                )
            except Exception:
                logger.warning("Forest merge failed", exc_info=True)

        # KB 回写: 把本次实验结论存入知识库, 下次同类问题能从 KB 召回.
        # 不存原始数据 (太大), 只存 hypothesis + validation 摘要.
        # JEPA: 预测误差也写入, 下次同类任务能从 KB 检索到"这类任务
        # agent 的预测准不准", 帮助判断是否需要更多探索.
        try:
            kb = self._get_kb()
            if kb:
                pred_err = validation.get("prediction_error", {})
                surprise_line = ""
                if pred_err:
                    surprise_line = f"\nPrediction surprise: {pred_err.get('surprise', 0)}\nPredicted: {pred_err.get('predicted', '')[:100]}\nActual: {pred_err.get('actual', '')[:100]}"
                summary_text = (
                    f"Iteration {self._iteration}: {hypothesis[:200]}\n"
                    f"Mode: {plan.get('mode', 'unknown')}\n"
                    f"R_phys: {r_phys}\n"
                    f"Validation: {json.dumps(validation, default=str)[:500]}"
                    f"{surprise_line}"
                )
                kb.add_document(
                    filename=f"autoloop_iter_{self._iteration}.txt",
                    content=summary_text.encode("utf-8"),
                )
        except Exception:
            logger.warning("error in _learn: KB writeback failed", exc_info=True)

        # KB 自动清理: 每 10 轮迭代清理一次旧文档, 防止 KB 无限增长.
        # autoloop_iter_ 文档只保留最近 50 轮, 总文档上限 200.
        # 这解决了"每轮写入但永不删除"的内存泄漏问题.
        if self._iteration > 0 and self._iteration % 10 == 0:
            try:
                kb = self._get_kb()
                if kb and hasattr(kb, "cleanup_old_documents"):
                    deleted = kb.cleanup_old_documents(max_docs=200)
                    if deleted:
                        logger.info("KB cleanup: removed %d old documents", deleted)
            except Exception:
                logger.debug("kb cleanup skipped", exc_info=True)

        # KG 回写: 把 hypothesis 作为 experiment 实体加入知识图,
        # 让 ProjectKnowledgeGraph 随实验增长而非只读展示.
        # 视觉基元 + surprise 都写入实体属性, 下次 KG 查询能检索到.
        try:
            kg_attrs: dict[str, Any] = {
                "iteration": self._iteration,
                "r_phys": r_phys,
            }
            visual_ctx = (
                validation.get("visual_primitives")
                if isinstance(validation, dict)
                else None
            )
            if visual_ctx:
                kg_attrs["visual_primitives"] = visual_ctx[:500]
            # JEPA: surprise 分数存入 KG, 下次查同类实验能看到"这类任务
            # agent 预测准不准", 帮助判断是否值得继续探索.
            pred_err = (
                validation.get("prediction_error", {})
                if isinstance(validation, dict)
                else {}
            )
            if pred_err:
                kg_attrs["surprise"] = pred_err.get("surprise", 0)
                kg_attrs["predicted"] = pred_err.get("predicted", "")[:200]
            # Persona 入 KG: 以后可以查 "reviewer persona 的 experiments 平均 r_phys 是多少"
            kg_attrs["persona"] = getattr(self, "_last_persona", "unknown")
            exp_id = self.kg.add_entity(
                label=hypothesis[:80],
                entity_type="experiment",
                source="autoloop",
                confidence=float(r_phys) if r_phys is not None else 0.5,
                **kg_attrs,
            )
            # KG confidence 衰减: validation 失败时降低实验实体置信度.
            # 之前 confidence 只增不减, 被refute的假设在 KG 里永远高置信.
            tests_passed = (
                validation.get("tests_passed")
                if isinstance(validation, dict)
                else False
            )
            if not tests_passed and exp_id and hasattr(self.kg, "_graph"):
                try:
                    if exp_id in self.kg._graph:
                        old_conf = self.kg._graph.nodes[exp_id].get("confidence", 0.5)
                        self.kg._graph.nodes[exp_id]["confidence"] = old_conf * 0.7
                except Exception:
                    logger.debug("kg confidence decay skipped", exc_info=True)
            # Hyperedge: 把 hypothesis → plan_mode → validation 结果
            # 连成 n-ary 关系. 之前 add_hyperedge 是死代码, 现在接上.
            plan_id = self.kg.add_entity(
                label=f"plan_{plan.get('mode', 'unknown')}_iter{self._iteration}",
                entity_type="Method",
                source="autoloop",
            )
            result_label = (
                "pass"
                if (
                    validation.get("tests_passed")
                    if isinstance(validation, dict)
                    else False
                )
                else "fail"
            )
            result_id = self.kg.add_entity(
                label=f"{result_label}_iter{self._iteration}",
                entity_type="Fact",
                source="autoloop",
                surprise=pred_err.get("surprise", 0) if pred_err else 0,
            )
            if exp_id and plan_id and result_id:
                self.kg.add_hyperedge(
                    [exp_id, plan_id, result_id],
                    relation="experiment_pipeline",
                    source="autoloop",
                    iteration=self._iteration,
                )
            # C5: persona_use entity — KG 层 persona 选择历史.
            # _pick_hypothesis_persona 遍历 persona_use 节点按 r_phys 均值召回.
            # ponytail: 不引入 embedding 相似度, 直接遍历 _graph.nodes 按 type 过滤.
            # 升级路径: context_hash 距离 (Hamming) 或 embedding 相似度召回.
            try:
                from huginn.utils.common import hash_text
                _ctx = getattr(self, "_last_context", {}) or {}
                _ctx_hash = hash_text(
                    json.dumps(_ctx, ensure_ascii=False, sort_keys=True, default=str)
                )
                self.kg.add_entity(
                    label=f"{persona_name}_iter{self._iteration}",
                    entity_type="persona_use",
                    source="autoloop",
                    confidence=float(r_phys) if r_phys is not None else 0.5,
                    persona=persona_name,
                    context_hash=_ctx_hash,
                    r_phys=r_phys,
                    iteration=self._iteration,
                )
            except Exception:
                logger.debug("persona_use entity write skipped", exc_info=True)
            self.kg.save()
        except Exception:
            logger.warning("error in _learn: KG add_entity failed", exc_info=True)

        # Benchmark 失败回写: 把验证失败写入 memory, 下次 _plan 能读到.
        if isinstance(validation, dict) and not validation.get("tests_passed", True):
            try:
                self.memory.remember(
                    content=(
                        f"Validation failure iter {self._iteration}: "
                        f"{json.dumps(validation, default=str)[:400]}"
                    ),
                    category="benchmark_failure",
                    tags=["autoloop", "validation"],
                    importance=0.7,
                    tier="mid",
                )
            except Exception:
                logger.warning(
                    "error in _learn: benchmark_failure memory writeback failed",
                    exc_info=True,
                )
            # C4: typed memory 默认 on, 同步写 failed_direction,
            # 让 _recent_failed_hypotheses 跨 session 能恢复 (不靠 hypothesis_graph).
            # ponytail: 不动 HypothesisNode setter, 在 _learn 里集中写, 最小改动.
            # EvolutionManager 默认 ON 时跳过旧路径, 避免跟 record_outcome 双写
            # FailedDirectionStore (record_outcome 内部已写). flag off 回退旧路径.
            try:
                _fail_reason = (
                    validation.get("error")
                    or validation.get("reason")
                    or json.dumps(validation, default=str)[:200]
                )
                # Task 3: toggle on 时反推 failure reasoning trace, 替换简短 error 串.
                # 默认 off — 旧行为完全不变. 失败静默回退到原 _fail_reason.
                if os.environ.get("HUGINN_FAILURE_INVERSION", "0") == "1":
                    try:
                        _inverted = await self._invert_failure_trace(
                            input_params=str(hypothesis)[:500],
                            failed_result=json.dumps(validation, default=str)[:1000],
                            failure_mode=str(_fail_reason),
                        )
                        if _inverted:
                            _fail_reason = _inverted
                    except Exception:
                        logger.debug(
                            "failure inversion failed, fallback to original reason",
                            exc_info=True,
                        )
                if os.environ.get("HUGINN_USE_EVOLUTION_MANAGER", "1") != "1":
                    self.memory.record_failed_direction(
                        hypothesis_text=hypothesis[:200],
                        reason=str(_fail_reason),
                        run_id=getattr(self, "_run_id", "") or "",
                        persona_id=getattr(self, "_last_persona", None),
                        math_concept="",
                    )
            except Exception:
                logger.debug(
                    "record_failed_direction failed, fallback to legacy path",
                    exc_info=True,
                )

        # Feynman learning: 高 surprise 或高奖励时, 让 agent 用通俗语言重新解释本轮发现.
        # 解释不出来的部分就是知识缺口, 写入 GoalStore 作为下轮子目标.
        # 触发条件: surprise > 0.5 (预测错误大) 或 r_phys > 0.7 (值得总结的成功)
        _should_feynman = False
        try:
            _surprise_val = 0.0
            if isinstance(validation, dict):
                _pe = validation.get("prediction_error", {})
                _surprise_val = _pe.get("surprise", 0) if isinstance(_pe, dict) else 0
            if _surprise_val > 0.5 or (r_phys is not None and r_phys > 0.7):
                _should_feynman = True
        except Exception:
            logger.debug(
                "surprise detection failed — _feynman_learn trigger may silently skip",
                exc_info=True,
            )

        if _should_feynman:
            try:
                await self._feynman_learn(
                    hypothesis, plan, validation, r_phys,
                    getattr(self, "_last_context", {}) or {},
                )
            except Exception:
                logger.warning(
                    "error in _learn: feynman note generation failed", exc_info=True
                )

        # 把 plan 进度存进 long-term memory, 下次会话能接续
        _plan_id = plan.get("plan_id") if isinstance(plan, dict) else None
        if _plan_id:
            try:
                store = self._get_plan_store()
                if store is not None:
                    persisted = store.get_plan(_plan_id)
                    if persisted is not None:
                        self.memory.store_plan_progress(
                            plan_id=persisted.id,
                            objective=persisted.objective,
                            step_index=len(
                                [s for s in persisted.steps if s.status == "done"]
                            ),
                            status=persisted.status,
                            l1_coordinates=f"autoloop: {persisted.objective[:100]}",
                        )
            except Exception:
                logger.warning(
                    "error in _learn: store_plan_progress writeback failed",
                    exc_info=True,
                )

        # RSI 入门: 让 agent 反思本轮, 给下一轮的自己写一条指令.
        # 借鉴 Inkling self-finetune loop: agent 改的不是自己的权重, 是自己下一轮的 prompt.
        # directive 写进 memory (不走 prompt 注入), 下轮 _build_hypothesis_prompt
        # 和 _build_plan_prompt 的 _build_memory_text 自然检索到 — 复用现有 memory loop.
        # maker/checker split: learn 写 directive, 下轮 validate 校验效果.
        # ponytail: 复用 memory tier 机制做衰减, 不引入新字段. 升级: 结构化 directive
        try:
            await self._generate_next_loop_directive(
                hypothesis, plan, validation, r_phys
            )
        except Exception:
            logger.debug(
                "RSI directive generation failed — loop continues without directive",
                exc_info=True,
            )

        # 收尾: 把本轮结果落一条 autoloop_summary, chat agent recall 时能拉到.
        # ponytail: 只落 summary 不共享 SessionContext; 升级路径是共享 SessionContext
        # 或加 autoloop_result 专用 category (目前先复用 memory.remember 通用通道).
        try:
            _tests_passed = (
                validation.get("tests_passed", True)
                if isinstance(validation, dict)
                else True
            )
            _summary = (
                f"iteration_count={self._iteration}; "
                f"refined_hypotheses={len(self.hypothesis_graph.nodes)}; "
                f"speculator_hints={self._speculator_hint[:200]!r}; "
                f"benchmark_failures={'no' if _tests_passed else 'yes'}; "
                f"r_phys={r_phys}; hypothesis={hypothesis[:120]!r}"
            )
            self.memory.remember(
                content=_summary,
                category="autoloop_summary",
                importance=0.9,
                tier="long",
                tags=["autoloop", "summary", f"iter:{self._iteration}"],
            )
        except Exception:
            # memory 失败不阻断 _learn, 上一轮的迭代已经入账
            logger.debug(
                "autoloop_summary writeback failed — loop continues",
                exc_info=True,
            )

        # P14: EvolutionManager.record_outcome — 默认 ON, 把 outcome 统一记到
        # FailedDirectionStore + SkillEvolutionLayer. flag off 回退分散路径.
        if os.environ.get("HUGINN_USE_EVOLUTION_MANAGER", "1") == "1":
            try:
                from huginn.evolution.manager import EvolutionManager

                em = EvolutionManager.shared(self.memory)
                em.record_outcome(
                    hypothesis=hypothesis,
                    plan=plan if isinstance(plan, dict) else None,
                    validation=validation if isinstance(validation, dict) else None,
                    persona_id=getattr(self, "_last_persona", None),
                    run_id=getattr(self, "_run_id", "") or "",
                    math_concept="",
                    # 归因由上游 agent 推理显式判定 (validation["gap_type"]),
                    # 缺省 unknown, 不做规则推断.
                    gap_type=(
                        validation.get("gap_type", "unknown")
                        if isinstance(validation, dict)
                        else "unknown"
                    ),
                )
            except Exception:
                logger.warning(
                    "EvolutionManager.record_outcome failed", exc_info=True
                )

        # H0: 触发 episodic → procedural 蒸馏 (修死代码, 让 stable_principles
        # 真有产出). 触发条件由 distill_episodic_to_procedural 内部判断
        # (连续 3 次同 skill 成功), 不降低阈值. ponytail: 失败不阻塞主循环.
        # D2: 捕获返回值, 用于 last_learn_summary 的 principles_added 字段.
        _principles_added = 0
        try:
            _pid = self.memory.distill_episodic_to_procedural(
                self._evals_history, self.workspace
            )
            if _pid:
                _principles_added = 1
        except Exception:
            logger.debug(
                "distill_episodic_to_procedural failed", exc_info=True
            )

        # P0 Task 4: 周期更新 self_model 缓存. 每 10 次实验 (成功+失败都算)
        # 触发 _compute_self_model + _store_self_model_cached. toggle
        # HUGINN_SELF_MODEL 默认 off, off 时不触发 (向后兼容). 失败 try/except
        # 包, 不阻塞主流程. ponytail: 复用 longterm typed memory 存储.
        try:
            self._experiment_count_since_self_model_update += 1
            if (
                os.environ.get("HUGINN_SELF_MODEL", "0") == "1"
                and self._experiment_count_since_self_model_update >= 10
            ):
                _sm = self.memory.longterm._compute_self_model()
                self.memory.longterm._store_self_model_cached(_sm)
                self._experiment_count_since_self_model_update = 0
                logger.info(
                    "self_model updated: %d groups, sample_size=%d",
                    len(_sm),
                    sum(s["success"] + s["failure"] for s in _sm.values()),
                )
        except Exception:
            logger.debug("self_model periodic update failed", exc_info=True)

        # P0 Task 1: 实验成功后扫 trace clusters, ≥3 条同簇 + 无 skill 时
        # 触发 skill 抽象 (Voyager-style). toggle HUGINN_SKILL_ABSTRACTION 默认 off,
        # 失败不阻塞主流程. ponytail: 接在 _learn 末尾是最低侵入的接入点 —
        # 每轮实验成功都查一次, cluster_key 命中已有 skill 直接跳过.
        try:
            await self._abstract_skill_if_ready()
        except Exception:
            logger.debug("skill abstraction hook failed", exc_info=True)

        # P1 Task 7: scan self_model weak clusters, synthesize self-goal.
        try:
            await self._synthesize_self_goal_if_ready()
        except Exception:
            logger.debug("self-goal synthesis hook failed", exc_info=True)

        # C3 闭环: 本轮如果命中过 trajectory_match, 按 validation 结果做 ±ε.
        # spec (layered_memory_spec.md:124) 设计的闭环, 之前只开了一半
        # (extract_and_store_pattern 在 goal_achieved 时 +ε), 缺 -ε 反馈.
        # 用 run_id 反查 KB doc_id (metadata.run_id), 调 update_pattern_confidence.
        _traj_run_id = getattr(self, "_last_traj_match_run_id", None)
        if _traj_run_id:
            try:
                kb = self._get_kb()
                if kb and hasattr(kb, "collection"):
                    data = kb.collection.get(
                        where={"source": "trajectory_pattern"},
                        include=["metadatas"],
                    )
                    doc_id = None
                    for meta in (data.get("metadatas") or []):
                        if meta.get("run_id") == _traj_run_id:
                            doc_id = meta.get("doc_id")
                            break
                    if doc_id:
                        from huginn.knowledge.trajectory_pattern import (
                            update_pattern_confidence,
                        )
                        _tests_ok = bool(
                            validation.get("tests_passed")
                            if isinstance(validation, dict) else False
                        )
                        update_pattern_confidence(kb, doc_id, success=_tests_ok)
            except Exception:
                logger.debug("C3 trajectory confidence update failed", exc_info=True)
            # 清掉本轮 match 标记, 下轮重新记
            self._last_traj_match_run_id = None
            self._last_traj_match_doc_id = None

        # D2: 返回 summary, 让 caller 写 cog["last_learn_summary"]
        return {
            "persona": getattr(self, "_last_persona", "unknown"),
            "r_phys": r_phys,
            "tests_passed": bool(
                validation.get("tests_passed")
                if isinstance(validation, dict) else False
            ),
            "principles_added": _principles_added,
        }



    async def _generate_next_loop_directive(
        self,
        hypothesis: str,
        plan: dict[str, Any],
        validation: dict[str, Any],
        r_phys: Any,
    ) -> None:
        """生成下一轮的自我指令 — RSI 的最小工程实现.

        Agent 反思本轮, 输出一条 directive 写入 memory (category=self_directive).
        下轮 _build_hypothesis_prompt / _build_plan_prompt 通过 _build_memory_text
        自然检索到, 不需要显式注入. memory tier 机制负责衰减, 老指令自动淡出.

        失败静默 — 这是 enhancement 不是 critical path.
        """
        tests_passed = (
            validation.get("tests_passed", False)
            if isinstance(validation, dict)
            else False
        )
        pred_err = (
            validation.get("prediction_error", {})
            if isinstance(validation, dict)
            else {}
        )
        surprise = pred_err.get("surprise", 0) if isinstance(pred_err, dict) else 0

        prompt = (
            "You just finished an autoloop iteration. Reflect on it and write "
            "a single concise directive to your future self for the NEXT iteration.\n\n"
            f"Hypothesis (this iter): {hypothesis[:200]}\n"
            f"Mode: {plan.get('mode', 'unknown') if isinstance(plan, dict) else 'unknown'}\n"
            f"Tests passed: {tests_passed}\n"
            f"R_phys: {r_phys}\n"
            f"Surprise: {surprise:.2f}\n\n"
            "Based on this, output ONE directive (max 2 sentences, no preamble):\n"
            "- If failed: what to AVOID next time (which method/path didn't work)\n"
            "- If high surprise: what to INVESTIGATE deeper\n"
            "- If high r_phys: what method to REUSE\n"
            "- If mundane: what to SKIP to save tokens\n\n"
            "Output only the directive, no markdown headers."
        )

        try:
            response = await self._llm_chat(prompt, task="summarize")
        except Exception:
            # LLM 挂了不阻断 — directive 是 enhancement, 不是 critical path
            logger.debug("RSI directive LLM call failed", exc_info=True)
            return

        if not (response and response.strip()):
            return

        directive = response.strip()[:300]
        # 写入 memory: 用 self_directive category + rsi tag, 让 recall 能定向检索.
        # tier=mid: 几轮后衰减, 不会永久占据 context. importance 跟 surprise 挂钩 —
        # 高 surprise 的 directive 更重要, 衰减更慢.
        importance = 0.5 + min(0.4, surprise * 0.4)
        try:
            self.memory.remember(
                content=f"[self-directive iter {self._iteration}] {directive}",
                category="self_directive",
                tags=["rsi", "autoloop"],
                importance=importance,
                tier="mid",
            )
            logger.info("RSI directive stored in memory: %s", directive[:120])
        except Exception:
            logger.debug("RSI directive memory write failed", exc_info=True)

        # H1: 看 r_phys + directive + 当前 hypothesis/plan blocks, LLM 生成
        # prompt patch 写入 patch store. 下轮 _build_*_prompt 的 _apply_block_patches
        # 自动 apply (Beta mean > 0.5 才生效, 新 patch alpha=beta=1 不会立即应用).
        # 失败静默 — generate_patch 内部已 catch. ponytail: 不接 plan blocks,
        # 只接 hypothesis — 单 phase 试点够验证, 多 phase 升级路径明确.
        try:
            from huginn.harness.prompt_patch import generate_patch
            # 用最近一次 _build_hypothesis_prompt 的 blocks 做 context (无则跳过)
            _hyp_blocks = getattr(self, "_last_hypothesis_blocks", None)
            if _hyp_blocks:
                await generate_patch(
                    phase="hypothesize",
                    blocks=_hyp_blocks,
                    r_phys=float(r_phys) if r_phys is not None else None,
                    directive=directive,
                    llm_chat_fn=self._llm_chat,
                )
        except Exception:
            logger.debug("H1 generate_patch failed", exc_info=True)



    async def _report(
        self, objective: str, phases: list[LoopPhase], total_time: float
    ) -> str | None:
        """Generate a structured scientific research report.

        RCBench expects y=(π, o, r) where r is a research report with
        Introduction/Methods/Results/Discussion. We assemble execution data
        from self and let the LLM write a proper report instead of a loop summary.
        """
        report_data = {
            "objective": objective,
            "run_id": f"loop_{uuid.uuid4().hex[:8]}",
            "total_time_seconds": total_time,
            "phases": [
                {
                    "name": p.name,
                    "status": p.status,
                    "duration": (
                        (p.end_time or 0) - (p.start_time or 0)
                        if p.start_time and p.end_time
                        else 0
                    ),
                    "error": p.error,
                }
                for p in phases
            ],
        }

        # Collect scientific evidence from the engine instance for the report.
        # This is the (π, o) data RCBench expects: what ran, what came out.
        last_exec = getattr(self, "_last_execution_result", None)
        exec_summary = ""
        if last_exec and isinstance(last_exec, dict):
            _tool = last_exec.get("_tool_name", "unknown")
            _res = last_exec.get("result", last_exec)
            exec_summary = json.dumps(_res, ensure_ascii=False, default=str)[:1500]
            exec_summary = f"Tool: {_tool}\nResult: {exec_summary}"

        visual_ctx = getattr(self, "_last_visual_context", "")
        last_validation = getattr(self, "_last_validation", "")
        last_surprise = getattr(self, "_last_surprise", 0.0)
        last_hypothesis = getattr(self, "_last_hypothesis", "")

        kb_text = self._build_kb_text(query=objective)
        # H4: persona 从 PhaseRegistry 取, toggle off 回退 "tutor"
        from huginn.harness.phase_spec import get_phase_persona
        _report_persona = get_phase_persona("_report") or "tutor"
        report_narrative = ""
        try:
            report_narrative = await self._llm_chat(
                self._build_science_report_prompt(
                    report_data,
                    kb_text,
                    exec_summary,
                    visual_ctx,
                    last_validation,
                    last_hypothesis,
                    last_surprise,
                ),
                persona_name=_report_persona,
                task="summarize",
            )
            report_narrative = (report_narrative or "").strip()
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            report_narrative = ""

        report_path = (
            self.workspace / f"huginn_autoloop_report_{report_data['run_id']}.md"
        )
        report_content = self._render_report(report_data)
        # P0-5: 无数据硬门 — exec_summary 为空 = 本循环未执行任何计算.
        # 禁止虚构 Results/Discussion, 强制声明零执行 (audit 06: 零执行仍虚构 Bader 电荷).
        if not exec_summary.strip():
            report_content += (
                "\n\n## Execution Status\n\n"
                "**本循环未执行任何计算。** 上述 Results/Discussion 无产物支撑, "
                "不应作为结论引用。需要重新执行工具调用获取真实数据。\n"
            )
        else:
            if kb_text:
                report_content += "\n\n## Domain Knowledge References\n\n" + kb_text + "\n"
            if report_narrative:
                report_content += "\n\n## Research Report\n\n" + report_narrative + "\n"
        report_path.write_text(report_content, encoding="utf-8")

        return str(report_path)



    async def _feynman_learn(
        self,
        hypothesis: str,
        plan: dict[str, Any],
        validation: dict[str, Any],
        r_phys: Any,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Feynman 学习法: 让 agent 用通俗语言解释本轮发现, 暴露知识缺口.

        生成的教学笔记存入蒸馏知识库 (feynman_note 类型, KB 检索优先).
        知识缺口写入 GoalStore 作为下轮子目标.
        """
        pred_err = (
            validation.get("prediction_error", {})
            if isinstance(validation, dict)
            else {}
        )
        surprise_val = pred_err.get("surprise", 0) if isinstance(pred_err, dict) else 0

        # 收集 deviation log, 让 Feynman 解释也覆盖 "为什么偏移了计划"
        deviation_text = ""
        if context and context.get("_deviation_log"):
            deviations = context["_deviation_log"]
            deviation_text = "\n".join(
                f"- [{d.get('type', '?')}] {d.get('deviation', '')}"
                for d in deviations[-3:]  # 最近 3 条
            )

        prompt = self._FEYNMAN_PROMPT.format(
            hypothesis=hypothesis[:300],
            mode=plan.get("mode", "unknown") if isinstance(plan, dict) else "unknown",
            r_phys=r_phys,
            surprise=f"{surprise_val:.2f}",
            validation=json.dumps(validation, default=str)[:600],
            deviations=deviation_text or "(none)",
        )

        # 用 summarize task 路由到便宜模型 — Feynman note 不需要强推理
        response = await self._llm_chat(prompt, task="summarize")
        if not response or not response.strip():
            return

        # 解析 explanation 和 gaps
        text = response.strip()
        explanation = ""
        gaps: list[str] = []

        # 简单解析: ## Simple Explanation 和 ## Knowledge Gaps 两段
        parts = text.split("## Knowledge Gaps")
        explanation_part = parts[0].replace("## Simple Explanation", "", 1).strip()
        # gaps 带 [KU]/[UU] 分类, 传给 GoalStore
        gaps: list[tuple[str, str]] = []  # (text, unknown_type)
        if len(parts) > 1:
            for line in parts[1].strip().split("\n"):
                line = line.strip().lstrip("-").strip()
                if not line or len(line) <= 5:
                    continue
                # 解析 [KU] / [UU] 标记
                if line.startswith("[UU]"):
                    gaps.append((line[4:].strip(), "unknown_unknown"))
                elif line.startswith("[KU]"):
                    gaps.append((line[4:].strip(), "known_unknown"))
                else:
                    # 无标记时用启发式分类
                    gap_lower = line.lower()
                    is_uu = any(
                        kw in gap_lower
                        for kw in [
                            "didn't",
                            "never",
                            "wasn't aware",
                            "didn't think",
                            "hadn't",
                            "overlooked",
                            "完全没",
                            "之前没",
                            "没想到",
                        ]
                    )
                    gaps.append((line, "unknown_unknown" if is_uu else "known_unknown"))

        if not explanation_part:
            explanation_part = text[:500]

        explanation = explanation_part

        # 存入蒸馏知识库
        _feynman_conf = min(0.9, 0.5 + (r_phys or 0) * 0.3)
        try:
            from huginn.evolution.knowledge_distiller import KnowledgeDistiller

            distiller = KnowledgeDistiller()
            tags = ["feynman", "autoloop", f"iter_{self._iteration}"]
            if surprise_val > 0.5:
                tags.append("high_surprise")
            # gaps 现在是 list[tuple[str, str]], 转回 list[str] 给 distiller
            gap_texts = [g[0] for g in gaps]
            distiller.store_feynman_note(
                explanation=explanation,
                gaps=gap_texts,
                iteration=self._iteration,
                hypothesis=hypothesis,
                tags=tags,
                confidence=_feynman_conf,
            )
        except Exception:
            logger.warning("feynman note storage failed", exc_info=True)

        # 缺口写入 GoalStore, 分类为 known_unknown / unknown_unknown
        # known_unknown: "我知道我不懂X" → 直接当子目标, 下轮解决
        # unknown_unknown: "我之前完全没想到X" → 标记为需要更深的探索
        # 借鉴 "Finding Your Unknowns" 四象限框架
        if gaps:
            try:
                from huginn.autoloop.goal_store import get_goal_store

                _gs = get_goal_store()
                _active = _gs.get_active()
                if _active:
                    for gap_text, gap_type in gaps[:3]:  # 最多 3 个, 避免子目标爆炸
                        _gs.add_sub_goal(_active.id, f"[Feynman {gap_type}] {gap_text}")
                        _gs.add_unknown(_active.id, gap_text, unknown_type=gap_type)
            except Exception:
                logger.debug("feynman gap subgoal skipped", exc_info=True)

        # 同时把 feynman note 写入 KB, 下次检索能命中
        try:
            kb = self._get_kb()
            if kb:
                note_text = f"# Feynman Note (iter {self._iteration})\n\n{explanation}\n\n## Gaps\n"
                for g_text, g_type in gaps:
                    note_text += f"- [{g_type}] {g_text}\n"
                kb.add_text(
                    text=note_text,
                    filename=f"feynman_iter_{self._iteration}.txt",
                    metadata={"confidence": str(_feynman_conf)},
                )
        except Exception:
            logger.debug("feynman note save failed", exc_info=True)



    async def _blind_spot_pass(
        self, context: dict[str, Any], objective: str
    ) -> list[dict[str, str]]:
        """Pre-implementation blind spot scan.

        借鉴 "Finding Your Unknowns" 的 Blind Spot Pass 技术:
        在开始工作前主动问 "我可能没想到什么?"
        发现的盲区写入 GoalStore.unknowns 供后续消解追踪.
        """
        # 压缩 context 到摘要, 避免太长
        ctx_parts: list[str] = []
        for k, v in context.items():
            if isinstance(v, str):
                ctx_parts.append(f"- {k}: {v[:150]}")
            elif isinstance(v, list) and v:
                ctx_parts.append(f"- {k}: {len(v)} items")
            elif isinstance(v, dict):
                ctx_parts.append(f"- {k}: {len(v)} keys")
        ctx_summary = "\n".join(ctx_parts[:10]) or "(minimal context)"

        prompt = self._BLIND_SPOT_PROMPT.format(
            objective=objective[:300],
            context_summary=ctx_summary,
        )

        response = await self._llm_chat(prompt, task="summarize")
        if not response or not response.strip():
            return []

        results: list[dict[str, str]] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line.startswith("BS:"):
                continue
            content = line[3:].strip()
            # 解析 [type] 前缀
            btype = "general"
            if content.startswith("["):
                end = content.find("]")
                if end > 0:
                    btype = content[1:end].strip()
                    content = content[end + 1 :].strip()
            if content and content != "NONE":
                results.append({"type": btype, "text": content})
                # 写入 GoalStore
                try:
                    from huginn.autoloop.goal_store import get_goal_store

                    _gs = get_goal_store()
                    _active = _gs.get_active()
                    if _active:
                        _gs.add_unknown(
                            _active.id,
                            content,
                            unknown_type="blind_spot",
                        )
                except Exception:
                    logger.debug("blind_spot unknown add skipped", exc_info=True)

        return results



    def _has_post_task_signal(
        self, state: Any, prev_outcome: str = "",
    ) -> tuple[bool, str]:
        """检查是否值得给用户推 next-step 推荐. 返回 (has_signal, reason).

        ponytail: 三个触发条件, 任一满足即推. 升级路径: 加 LLM 判定信号质量.
        """
        # 1. iteration_history 最近 5 轮里 failed 同一方向 >=2 次
        _hist = getattr(state, "iteration_history", []) or []
        if _hist:
            _recent = _hist[-5:]
            _failed = [h for h in _recent if h.get("val_status") == "failed"]
            if len(_failed) >= 2:
                _modes = [h.get("plan_mode") or h.get("mode") or "" for h in _failed]
                _mode_counts: dict[str, int] = {}
                for m in _modes:
                    _mode_counts[m] = _mode_counts.get(m, 0) + 1
                _max_count = max(_mode_counts.values()) if _mode_counts else 0
                if _max_count >= 2:
                    return True, f"iteration_history: {_max_count}x failed in same mode"

        # 2. _physical_timeseries 反常 (peak_drift 或 v_decay 绝对值 > 0.01)
        _ts_list = getattr(self, "_physical_timeseries", []) or []
        for _ts in _ts_list[-5:]:
            if not _ts.get("spatial"):
                continue
            _data = _ts.get("data") or []
            if len(_data) < 2:
                continue
            # 三元组 (t, r, v) — 算首末帧 peak v 差
            _frames: dict = {}
            for entry in _data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    _frames.setdefault(entry[0], []).append(entry[2])
            if len(_frames) >= 2:
                _ts_keys = sorted(_frames.keys())
                _first_peak = max(_frames[_ts_keys[0]]) if _frames[_ts_keys[0]] else 0
                _last_peak = max(_frames[_ts_keys[-1]]) if _frames[_ts_keys[-1]] else 0
                if abs(_last_peak - _first_peak) > 0.01:
                    return True, f"timeseries: peak_v decay {_last_peak - _first_peak:+.3f}"

        # 3. prev_run_context.outcome == inconclusive
        if prev_outcome == "inconclusive":
            return True, "prev_run_context.outcome=inconclusive"

        return False, ""



    async def _advisor_post_task_recommend(
        self, run_id: str, objective: str, cog: dict, state: Any,
        prev_outcome: str = "",
    ) -> None:
        """任务完成后给用户推 next-step 推荐.

        轻量: 默认只写 memory (category=next_step_hint), HUMAN_PAUSE=1 才走
        pause_for_decision 让用户选.

        ponytail: 不每次都推, 触发条件见 _has_post_task_signal. 升级路径:
        让 LLM 判断信号质量, 避免低质推荐打扰用户.
        """
        try:
            _has_signal, _reason = self._has_post_task_signal(state, prev_outcome)
            if not _has_signal:
                return

            # 构造 prompt 上下文
            _hypothesis = (cog.get("hypothesis") or "")[:300]
            _outcome = prev_outcome or (
                "completed" if (cog.get("validation") or {}).get("tests_passed")
                else "inconclusive"
            )

            # iteration_history 反常点 (最近 5 轮 failed 方向)
            _hist = getattr(state, "iteration_history", []) or []
            _anomalies = ""
            if _hist:
                _recent_failed = [
                    f"iter{h.get('iter')}: {h.get('advice', '')[:80]}"
                    for h in _hist[-5:]
                    if h.get("val_status") == "failed"
                ]
                if _recent_failed:
                    _anomalies = "; ".join(_recent_failed[:3])

            # _physical_timeseries 反常摘要
            _ts_anomaly = ""
            _ts_list = getattr(self, "_physical_timeseries", []) or []
            if _ts_list:
                _ts_ctx = self._format_timeseries_context()
                if _ts_ctx:
                    _ts_anomaly = _ts_ctx[:300]

            # prev_run_context 尾巴
            _prev_tail = (getattr(self, "_prev_run_context", "") or "")[:200]

            prompt = self._NEXT_STEP_ADVISOR_PROMPT.format(
                hypothesis=_hypothesis,
                outcome=_outcome,
                anomalies=_anomalies or "(none)",
                ts_anomaly=_ts_anomaly or "(none)",
                prev_tail=_prev_tail or "(none)",
            )

            response = await self._llm_chat(prompt, persona_name="reviewer", task="reasoning")
            if not response or not response.strip():
                return

            # 写入 memory (category=next_step_hint) — 用户问"接下来呢"时召回
            _mem = getattr(self, "memory", None)
            if _mem is not None:
                try:
                    _mem.remember(
                        content=response.strip(),
                        category="next_step_hint",
                        memory_type="next_step_hint",
                        metadata={
                            "run_id": run_id,
                            "objective": objective[:200],
                            "outcome": _outcome,
                            "trigger_reason": _reason,
                        },
                    )
                except Exception:
                    logger.debug("next_step_hint memory write failed (non-fatal)", exc_info=True)

            # HUMAN_PAUSE=1 时走 pause_for_decision 让用户选
            _human_pause = os.environ.get("HUGINN_AUTOLOOP_HUMAN_PAUSE", "0") == "1"
            if _human_pause:
                _options = [
                    {"id": "A", "label": "选 A 方向 (见推荐)", "pros": "深化本轮", "cons": "可能 local minimum"},
                    {"id": "B", "label": "选 B 方向 (见推荐)", "pros": "横向探索", "cons": "换方向成本"},
                    {"id": "C", "label": "选 C 方向 (见推荐)", "pros": "验证反常", "cons": "可能 false alarm"},
                    {"id": "D", "label": "我自己有想法", "pros": "用户直觉", "cons": "需要用户输入"},
                ]
                _step_id = getattr(self, "_iteration", 0) or 0
                try:
                    await self._await_human_decision_via_inbox(
                        f"本轮结束, 推荐下一步 (触发: {_reason}):\n\n{response[:500]}",
                        _options, _step_id,
                    )
                except Exception:
                    logger.debug("pause_for_decision failed (non-fatal)", exc_info=True)

        except Exception:
            logger.debug("_advisor_post_task_recommend failed (non-fatal)", exc_info=True)



    @staticmethod
    def _build_science_report_prompt(
        report_data: dict[str, Any],
        kb_text: str = "",
        exec_summary: str = "",
        visual_ctx: str = "",
        validation_summary: str = "",
        hypothesis: str = "",
        surprise: float = 0.0,
    ) -> str:
        """Build a prompt for generating a structured scientific research report.

        RCBench evaluates y=(π, o, r) where r must contain scientific findings,
        not just a loop status table. This prompt produces Introduction /
        Methods / Results / Discussion structure from the actual execution data.
        """
        try:
            phases_blob = json.dumps(report_data["phases"], ensure_ascii=False)[:800]
        except Exception:
            phases_blob = str(report_data.get("phases", ""))[:800]
        kb_section = f"\n## Domain Knowledge\n{kb_text}\n" if kb_text else ""
        exec_section = f"\n## Execution Data\n{exec_summary}\n" if exec_summary else ""
        visual_section = f"\n## Visual Primitives\n{visual_ctx}\n" if visual_ctx else ""
        val_section = (
            f"\n## Validation\n{validation_summary}\n" if validation_summary else ""
        )
        hyp_section = f"\n## Hypothesis Tested\n{hypothesis}\n" if hypothesis else ""

        return (
            "You are writing a structured scientific research report based on an "
            "autonomous research loop's execution data. This is NOT a loop summary — "
            "it must read like a research paper section.\n\n"
            f"Objective: {report_data['objective']}\n"
            f"Phases:\n{phases_blob}\n"
            f"Surprise score: {surprise:.2f} (0=predicted, 1=unexpected)"
            f"{hyp_section}{exec_section}{visual_section}{val_section}{kb_section}"
            "\nWrite the report with these sections (Markdown):\n"
            "## Introduction\n"
            "State the scientific question and why it matters. Reference domain knowledge above.\n\n"
            "## Methods\n"
            "Describe the computational approach: what tools were used, what parameters, "
            "what workflow. Be specific enough for reproducibility.\n\n"
            "## Results\n"
            "Report the key findings with specific numbers. If visual primitives are "
            "available, describe the trends/peaks/anomalies they indicate.\n\n"
            "## Discussion\n"
            "Interpret the results: Do they support the hypothesis? What was surprising "
            "(reference surprise score)? What are the limitations? "
            "What should the next experiment be?\n"
        )



