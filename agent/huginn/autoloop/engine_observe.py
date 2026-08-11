"""EngineObserveMixin — AutoloopEngine 的 prompt 拼装 + 元认知方法族.

从 engine.py 拆出 (P3 slim-down 续). 包含:
- prompt block 构造 (compress/trim/budget/patch)
- _build_hypothesis_prompt (核心 prompt 拼装)
- 各类 *_block 辅助 (curiosity/world_model/skill_context/episodic_replay/pmk)
- 元认知层 (metacog auditors/registries, completion/topology check)

通过 self 访问 engine 状态. 方法体原样搬迁, 不改逻辑.

设计原则 (ponytail):
- 对 engine.py 模块级符号 (constants / helpers) 用方法内 lazy import, 避免 circular
- Mixin 不持有自己的状态, 全部走 self
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


class EngineObserveMixin:
    """prompt 拼装 + 元认知层方法族. 通过 self 访问 engine 状态."""

    # 上下文预算: 防止 prompt block 累积超过 token 上限.
    # 优先级: body > math > kg > visual > kb > mem > pm > hint > skill > composite > pipeline
    # 超预算时不是直接丢弃, 而是分层压缩: 先截断 → 再摘要 → 最后才删.
    # 视觉语言比文字语言更能压缩信息 — 一行 "[energies] peak=idx3, trend=↑"
    # 传达的信息等于 200 chars 的 JSON. 用压缩替代丢弃, 保留信息密度.
    _PROMPT_BUDGET = 12000  # chars, 约 3K tokens (fallback default)
    # C-budget: 分层 budget — hypothesis/plan 主战场留满, 其他 phase 不走 prompt builder.
    # ponytail: 只 dict + getter, 不引 BudgetPolicy 抽象. env 覆盖留调参口子.
    # 升级路径: 加 learn/pivot 的 prompt builder 后再扩 phase 覆盖.
    _PROMPT_BUDGET_BY_PHASE: dict[str, int] = {
        "hypothesize": 12000,
        "plan": 12000,
    }

    # 数学深度提示块: 在 hypothesis / plan prompt 里持续提醒 agent 用
    # 符号数学工具把"现象"翻译成"PDE / 变分 / 几何 / 灵敏度"语言.
    # 用户偏好: 物理、化学本质上是数学的一部分 — 这里把那条原则落进 prompt.
    _MATH_DEPTH_PROMPT_BLOCK = """
Math depth guidance (treat physics/chemistry as mathematics):
- Identify the governing PDE: use symbolic_math_tool action=pde_classify
  (A;B;C discriminant) to classify elliptic/parabolic/hyperbolic, then
  pde_separation or pde_characteristics for analytic structure.
- If the phenomenon extremizes a functional, derive Euler-Lagrange:
  symbolic_math_tool action=euler_lagrange or action=derive (alias).
  Check symmetries with action=noether to predict conserved currents.
- For curved manifolds (defects, interfaces, crystal plasticity), compute
  Christoffel/Ricci via action=diffgeo_metric or diffgeo_curvature.
- Before fitting data, run symbolic_regression_tool action=sobol_indices
  to rank feature importance, then discover expressions with
  action=discover and validate candidates with action=constraint_check
  (positivity / monotonicity / finiteness priors).
"""

    _IMAGINATION_PROMPT_BLOCK = """
Imagination directive (speculative mode activated):
- Your prediction was significantly off, or your hypotheses keep getting refuted.
- Consider a counterfactual: what if the governing structure is different from what you assumed?
- Try shifting between mathematical structure families: PDE ↔ variational, continuum ↔ discrete, deterministic ↔ stochastic, linear ↔ nonlinear.
- This is NOT random guessing — the shift must be between mathematically valid structure families, grounded in the domain context.
- The conjecture hint above uses forget-then-generate: known failed approaches have been deliberately suppressed.

LUCID review (mandatory after generating hypothesis):
- You are allowed an absurd premise, but the reasoning must be rigorous.
- State ONE necessary condition: without it, your hypothesis definitely fails.
- State ONE hidden assumption from the source domain that may not hold here.
- State ONE falsifiable test: if result is X, hypothesis is refuted.
- If you cannot state these, the hypothesis is dream-only and must be discarded.
"""

    @staticmethod
    def _compress_block(name: str, text: str, level: int) -> str:
        """分层压缩: level 0=原样, 1=截断, 2=一行摘要, 3=删除."""
        if not text or level <= 0:
            return text if level <= 0 else ""
        if level == 1:
            # 截断到 300 字符, 保留开头
            if len(text) <= 300:
                return text
            return text[:300] + "..."
        if level == 2:
            # 压缩成一行摘要: 取关键信息
            lines = text.strip().split("\n")
            # KB/KG/mem: 只保留前 2 行 + "..."
            if len(lines) <= 2:
                return lines[0][:100] if lines else ""
            return lines[0][:100] + " | " + lines[1][:100] + " | ..."
        return ""

    def _scan_block_conflicts(self, blocks: list[tuple[str, str]]) -> str:
        """Lightweight cross-source conflict detection: same property, different values.

        Scans block text for 'property = value unit' patterns. When the same
        property appears in multiple blocks with different numeric values,
        returns a warning string. Uses regex only, no LLM calls.
        """
        # ponytail: 属性名前缀不一致 (如 "band gap" vs "the band gap") 会导致漏检,
        # 但对 <10 blocks 的 prompt 场景足够; 如需精确匹配可改用 NER 提取属性名
        from huginn.autoloop.engine import _PROP_RE
        prop_values: dict[str, dict[str, str]] = {}
        for name, text in blocks:
            if not text:
                continue
            for m in _PROP_RE.finditer(text):
                prop = m.group(1).strip().lower()
                val = m.group(2)
                unit = m.group(3).strip()
                key = f"{prop} ({unit})"
                prop_values.setdefault(key, {})
                if val not in prop_values[key]:
                    prop_values[key][val] = name
        conflicts = []
        for key, vals in prop_values.items():
            if len(vals) > 1:
                sources = ", ".join(f"{v} in [{s}]" for v, s in vals.items())
                conflicts.append(f"{key}: {sources}")
        if not conflicts:
            return ""
        return (
            "Cross-source conflicts detected (same property, different values):\n"
            + "\n".join(f"  - {c}" for c in conflicts[:5])
            + "\nVerify which value is correct before proceeding.\n"
        )

    def _get_prompt_budget(self, phase: str | None) -> int:
        """按 phase 取 prompt budget. env 覆盖优先, dict 次之, fallback _PROMPT_BUDGET."""
        import os
        if phase:
            env_key = f"HUGINN_PROMPT_BUDGET_{phase.upper()}"
            env_val = os.environ.get(env_key)
            if env_val:
                try:
                    return int(env_val)
                except ValueError:
                    logger.debug("best-effort op failed", exc_info=True)
            if phase in self._PROMPT_BUDGET_BY_PHASE:
                return self._PROMPT_BUDGET_BY_PHASE[phase]
        return self._PROMPT_BUDGET

    def _apply_block_patches(
        self,
        blocks: list[tuple[str, str]],
        phase: str,
    ) -> list[tuple[str, str]]:
        """H1: 在 _trim_to_budget 前应用 prompt patch.

        apply_patches 内部按 Beta mean > 0.5 过滤 + 同名 block 取最高 Beta mean.
        这里重算一遍 by_block 拿到实际应用的 patch ids, 存到
        _last_applied_patches 供 _learn 更新 Beta. toggle off 或没 patch 时
        直接返回原 blocks (apply_patches 内部处理, 这里零开销).

        ponytail: 重算 by_block 跟 apply_patches 内部逻辑重复, 但避免改
        apply_patches 返回签名. 升级路径: apply_patches 返回 (blocks, ids).
        """
        from huginn.harness.prompt_patch import PromptPatchStore, apply_patches
        new_blocks = apply_patches(blocks, phase)
        # 记录原始 blocks 供 _generate_next_loop_directive 调 generate_patch 用
        # (generate_patch 需要看 block 名字 + 内容才能生成合理 patch)
        if phase == "hypothesize":
            self._last_hypothesis_blocks = blocks
        elif phase == "plan":
            self._last_plan_blocks = blocks
        if new_blocks is blocks:
            return new_blocks
        try:
            store = PromptPatchStore.get_instance()
            patches = [
                p for p in store.list_patches(phase=phase)
                if p.alpha / max(1, p.alpha + p.beta) > 0.5
            ]
            by_block: dict[str, Any] = {}
            for p in patches:
                cur = by_block.get(p.block_name)
                if cur is None or (
                    p.alpha / max(1, p.alpha + p.beta)
                    > cur.alpha / max(1, cur.alpha + cur.beta)
                ):
                    by_block[p.block_name] = p
            applied_ids = [p.id for p in by_block.values()]
            if applied_ids:
                self._last_applied_patches = (phase, applied_ids)
        except Exception:
            logger.debug("_apply_block_patches: track applied fail", exc_info=True)
        return new_blocks

    def _trim_to_budget(
        self,
        blocks: list[tuple[str, str]],
        *,
        phase: str | None = None,
    ) -> str:
        """按优先级拼接 blocks, 超预算时分层压缩: 截断→摘要→删除.

        phase 不传时走 _PROMPT_BUDGET (fallback), 传 "hypothesize"/"plan"
        走 _PROMPT_BUDGET_BY_PHASE 分层 budget. env HUGINN_PROMPT_BUDGET_<PHASE>
        覆盖一切.
        """
        budget = self._get_prompt_budget(phase)
        # 跨源冲突检测: 扫描各 block 中的 property=value 对, 标注矛盾
        conflict_warn = self._scan_block_conflicts(blocks)
        if conflict_warn:
            blocks = [("conflict", conflict_warn)] + blocks

        kept = [(n, v) for n, v in blocks]
        total = sum(len(v) for _, v in kept)
        if total <= budget:
            return "".join(v for _, v in kept)

        # Pass 1: 截断低优先级 block 到 300 字符
        for i in range(len(kept) - 1, -1, -1):
            if total <= budget:
                break
            name, text = kept[i]
            if name == "body":  # body 永远不压缩
                continue
            compressed = self._compress_block(name, text, 1)
            total -= len(text) - len(compressed)
            kept[i] = (name, compressed)

        if total <= budget:
            return "".join(v for _, v in kept)

        # Pass 2: 压缩成一行摘要
        for i in range(len(kept) - 1, -1, -1):
            if total <= budget:
                break
            name, text = kept[i]
            if name == "body":
                continue
            compressed = self._compress_block(name, text, 2)
            total -= len(text) - len(compressed)
            kept[i] = (name, compressed)

        if total <= budget:
            return "".join(v for _, v in kept)

        # Pass 3: 从最低优先级开始删除
        # skill/composite 受保护 — skills 引用保留系统: 可截断可摘要, 但不可清空
        for i in range(len(kept) - 1, -1, -1):
            if total <= budget:
                break
            name, text = kept[i]
            if name in ("body", "skill", "composite"):
                continue
            total -= len(text)
            kept[i] = (name, "")

        return "".join(v for _, v in kept)

    def _persona_system_prompt(self, persona_name: str | None) -> str:
        """取 persona 的 system prompt, 按层组装.

        层级 (SillyTavern 角色卡分层启发):
        1. permanent_core (或 system_prompt 向后兼容) — 身份/角色/安全约束
        2. adaptive_layer — 会话级风格/偏好 (由 StyleLearner/TasteProfile 填充)
        """
        if not persona_name:
            return ""
        try:
            persona = self._get_persona_manager().get(persona_name)
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return ""
        # 优先用 permanent_core, 没设就退回 system_prompt (老 persona)
        core = persona.permanent_core or persona.system_prompt or ""
        adaptive = persona.adaptive_layer or ""
        if adaptive:
            return f"{core}\n\n--- Adaptive ---\n{adaptive}"
        return core

    # _phase_persona removed — per-call persona_name= in each phase method
    # is the active injection path. _PHASE_PERSONAS stays as documentation.

    def _build_curiosity_block(self) -> str:
        """P1 Task 6: 检索 self_model 里成功率低的簇, 拼 [CURIOSITY] block.

        把"哪类问题我长期搞不定"喂给 hypothesize prompt, 让 agent 主动
        seek 而非被动 escape. 复用 Task 4 self_model (rate<0.4 且样本≥3).
        toggle HUGINN_CURIOSITY_HINT 默认 off — 不开不消耗 self_model 查询.
        ceiling: self_model 按 (dimension, hyp_type) 聚合, 关键词命中粗.
        """
        if os.environ.get("HUGINN_CURIOSITY_HINT", "0") != "1":
            return ""
        _mem = getattr(self, "memory", None)
        if _mem is None or not hasattr(_mem, "longterm"):
            return ""
        try:
            _sm = _mem.longterm.get_self_model()
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return ""
        if not _sm:
            return ""
        # 过滤: rate<0.4 且样本量≥3 (统计有意义)
        weak: list[str] = []
        for _key, _v in _sm.items():
            _rate = _v.get("rate")
            _succ = _v.get("success", 0)
            _fail = _v.get("failure", 0)
            _n = _succ + _fail
            if isinstance(_rate, (int, float)) and _rate < 0.4 and _n >= 3:
                _dim = _v.get("dimension", "?")
                _htype = _v.get("hyp_type", "?")
                weak.append(f"- {_dim}/{_htype}: rate={_rate:.2f} (n={_n})")
        if not weak:
            return ""
        return (
            "\n[CURIOSITY] 历史预测不准的簇 (主动探索方向, 而非被动 escape):\n"
            + "\n".join(weak[:5])
            + "\n"
        )

    def _build_world_model_block(self, hypothesis: str) -> str:
        """P1 Task 5: retrieval world model (懒路) — predict_via_analogy.

        把 hypothesis 当 query, 调 longterm.predict_via_analogy 检索相似
        历史实验, 注入 [WORLD MODEL] block 给 plan prompt. LLM 参考历史
        结果做 plan, 不是硬约束. toggle HUGINN_WORLD_MODEL 默认 off.
        ceiling: 无外推能力, 仅返回历史最相似实验. 升级: P2 surrogate.
        """
        if os.environ.get("HUGINN_WORLD_MODEL", "0") != "1":
            return ""
        _mem = getattr(self, "memory", None)
        if _mem is None or not hasattr(_mem, "longterm"):
            return ""
        try:
            _pred = _mem.longterm.predict_via_analogy(
                {"hypothesis": hypothesis[:200]},
                top_k=3,
                similarity_threshold=0.6,
            )
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return ""
        if _pred.get("prediction_type") != "analogy":
            return ""
        _analogy = _pred.get("analogy", [])
        if not _analogy:
            return ""
        _lines: list[str] = []
        for _a in _analogy[:3]:
            _content = str(_a.get("content", ""))[:120]
            _score = _a.get("score", 0)
            _lines.append(f"  - (sim={_score:.2f}) {_content}")
        return (
            "\n[WORLD MODEL] 历史相似实验结果 (懒路预测, 仅供参考, 无外推):\n"
            + "\n".join(_lines) + "\n"
        )

    def _build_skill_context_block(self) -> str:
        """SkillEvolutionLayer 信念注入 — 接通 get_skill_context 死代码.

        把 skill evolution 从过往 trajectory 学到的参数信念 (成功率/置信度)
        注入 hypothesis prompt, 让 agent 参考"之前哪些参数组合 work 过".
        toggle HUGINN_SKILL_CONTEXT 默认 off, off 时返空串.
        ceiling: 信念空间是 Beta 后验, 不直接是规则. 升级路径: Beta→rule 映射.
        """
        if os.environ.get("HUGINN_SKILL_CONTEXT", "0") != "1":
            return ""
        try:
            from huginn.skills.evolution import SkillEvolutionLayer
            return SkillEvolutionLayer.shared().get_skill_context()
        except Exception:
            logger.debug("skill context injection failed", exc_info=True)
            return ""

    @property
    def _episodic_replay(self):
        """情景重放: 懒加载 + 只在 flag 打开时构建. 失败非致命.

        先建 EpisodicIndexStore (读 shard + 编码), 再包一层 EpisodicReplay.
        build 返 0 (没历史) 时直接放弃, 不留空壳.
        """
        if not hasattr(self, "_episodic_replay_obj"):
            self._episodic_replay_obj = None
            if os.environ.get("HUGINN_EPISODIC_REPLAY", "0").lower() in ("1", "true"):
                try:
                    from huginn.metacog.episodic_index import EpisodicIndexStore
                    from huginn.metacog.episodic_replay import EpisodicReplay
                    run_id = getattr(self, "_run_id", None) or "default"
                    store = EpisodicIndexStore(self.workspace, str(run_id))
                    if store.build() == 0:
                        self._episodic_replay_obj = None
                    else:
                        self._episodic_replay_obj = EpisodicReplay(store)
                except Exception:
                    logger.debug(
                        "episodic replay init failed (non-fatal)", exc_info=True
                    )
                    self._episodic_replay_obj = None
        return self._episodic_replay_obj

    def _build_episodic_replay_block(self, context: dict[str, Any]) -> str:
        """cue-based 情景重放: 从索引采样最相似情境, 注入其 advice 给 decider.

        flag off 或索引空时返回空串, 不影响现有流程.
        """
        replay = self._episodic_replay
        if replay is None:
            return ""
        try:
            cue = {
                "iter": context.get("iter", 0),
                "mode": context.get("mode", ""),
                "phase": context.get("phase", ""),
                "val_status": context.get("val_status", ""),
                "structure_desc": context.get("structure_desc"),
                # 桥 J: cue 带 surprise, 让 replay 能按 surprise 回溯高发现情境
                "surprise": getattr(self, "_last_surprise", 0.0),
            }
            replays = replay.replay(cue, top_k=3)
            if not replays:
                return ""
            lines = [
                f"[{r['replay_type']}] iter={r['iter']} val={r['val_status']} "
                f"dist={r.get('distance', '?')}: {r['advice']}"
                for r in replays
            ]
            return "\n".join(lines) + "\n"
        except Exception:
            logger.debug("episodic replay failed (non-fatal)", exc_info=True)
            return ""

    def _build_pmk_block(self, context: dict[str, Any]) -> str:
        """PMK 三路立场注入决策上下文. flag off 或全空时返回空串.

        接通死代码: build_pmk_state + build_pmk_text + _check_pmk_consistency.
        INCONSISTENT 时追加 abductive hint + 写回 episodic shard (Step 2+3).
        ponytail: hypothesize 和 plan 同一 iteration 都调本方法, side-effect
        (hint append + episodic write) 只在首次调用跑一次, 避免重复污染.
        """
        if os.environ.get("HUGINN_PMK_INJECT", "0").lower() not in ("1", "true"):
            return ""

        try:
            from huginn.autoloop.cognitive_loop import build_pmk_state
            from huginn.runtime.task_lifecycle import _check_pmk_consistency

            # persona: 用 .get (PersonaManager 真实 API), get_persona 不存在.
            persona_obj = None
            with contextlib.suppress(Exception):
                persona_obj = self._get_persona_manager().get("default")

            last_eval = self._evals_history[-1] if self._evals_history else None
            kb = self._get_kb()
            since = getattr(self, "_run_start_iso", None)

            pmk_state = build_pmk_state(
                persona_obj, last_eval, kb,
                since=since,
                mem_mgr=getattr(self, "memory", None),
                timeseries_ctx=self._format_timeseries_context() or None,
            )
            if not pmk_state:
                return ""

            # 一致性检查 (Step 2 入口) — 含 timeseries 路, 比 build_pmk_text 内部
            # 的 3 路检查更全. 这里的判定驱动 hint + shard side-effect.
            is_inconsistent, reason = _check_pmk_consistency(pmk_state)

            # side-effect 每 iteration 只跑一次: hypothesize + plan 都调本方法,
            # 不 guard 会重复 append hint + 重复写 episodic entry.
            cur_iter = getattr(self, "_iteration", 0)
            if cur_iter != getattr(self, "_pmk_side_effect_iter", -1):
                if is_inconsistent:
                    conflict_hint = f"[PMK CONFLICT] {reason[:200]}"
                    self._speculator_hint = (
                        (self._speculator_hint + f"\n{conflict_hint}\n").strip()
                        if self._speculator_hint else f"{conflict_hint}\n"
                    )
                    logger.info("PMK conflict detected: %s", reason[:100])
                    self._write_pmk_conflict_to_episodic(pmk_state, reason)
                    # 桥 B: PMK 冲突 → evolution 触发. 把冲突作为一种"认知失败"
                    # 喂给 evolution logger, 让 get_failure_patterns 能提取 PMK
                    # 冲突模式 (≥2 次同类冲突就生成 heuristic 规则). 失败非致命,
                    # 不破坏现有 pause 路径. ponytail: 复用 _get_evolution 懒加载.
                    try:
                        _evo = self._get_evolution()
                        _evo.logger.log_tool_call(
                            session_id=f"pmk_{cur_iter}",
                            tool_name="pmk_conflict",
                            tool_input={
                                "reason": reason[:200],
                                "pmk_state": {k: v[:100] for k, v in pmk_state.items()},
                            },
                            result=None,
                            error=reason[:300],
                        )
                        _evo.evolve_from_failures()
                    except Exception:
                        logger.debug(
                            "PMK → evolution bridge failed (non-fatal)",
                            exc_info=True,
                        )
                self._pmk_side_effect_iter = cur_iter

            # 格式化给 LLM 看 — 接通 build_pmk_text (死代码变活).
            # build_pmk_text 不读 self 实例状态, 用 __new__ 绕过 __init__ (跟
            # selfcheck 同模式). ponytail: build_pmk_text 内部再查一次一致性
            # (只 3 路, 不含 timeseries), 标签可能跟上面的 is_inconsistent 略有
            # 出入 — 接受, 不重写 build_pmk_text. 升级路径: 给它加 timeseries 参数.
            try:
                from huginn.context_builder import ContextBuilder
                ctx = ContextBuilder.__new__(ContextBuilder)
                pmk_text = ctx.build_pmk_text(
                    persona=persona_obj,
                    last_step_evaluation=last_eval,
                    kb=kb,
                )
                if not pmk_text:
                    pmk_text = self._format_pmk_fallback(pmk_state, is_inconsistent)
            except Exception:
                pmk_text = self._format_pmk_fallback(pmk_state, is_inconsistent)

            return pmk_text

        except Exception:
            logger.debug("PMK block failed (non-fatal)", exc_info=True)
            return ""

    def _format_pmk_fallback(self, pmk_state: dict, inconsistent: bool) -> str:
        """build_pmk_text 不可用时的降级格式化."""
        label = "INCONSISTENT" if inconsistent else "consistent"
        lines = [f"### PMK ({label})"]
        for src in ("persona", "memory", "kb", "timeseries"):
            if pmk_state.get(src):
                lines.append(f"  {src}: {pmk_state[src][:150]}")
        return "\n".join(lines) + "\n"

    def _write_pmk_conflict_to_episodic(self, pmk_state: dict, reason: str):
        """PMK 冲突写入 episodic shard, 下一轮 Memory 路可见. 失败非致命.

        复用 reflect 阶段那套 _episodic_writer — 没有就提前建好缓存到 self,
        reflect 检查到已存在会直接复用. 不调 flush_and_archive, 避免归档
        live writer 正在写的 shard (会破坏其 _fh 状态, 丢后续 entry).
        """
        try:
            iter_n = getattr(self, "_iteration", 0)
            entry = {
                "iter": iter_n,
                "action": "pmk_conflict",
                "pmk_conflict": reason[:300],
                "pmk_state": {k: v[:200] for k, v in pmk_state.items()},
                "val_status": "failed",
                "mode": getattr(self, "_last_failure_mode", "") or "",
                "phase": getattr(self, "_current_phase", "") or "",
                # 桥 J: surprise 进 episodic, replay 能按 surprise 回溯 PMK 冲突
                "surprise": float(getattr(self, "_last_surprise", 0.0)),
            }
            writer = getattr(self, "_episodic_writer", None)
            if writer is None:
                from huginn.memory.episodic_shard import EpisodicShardWriter
                writer = EpisodicShardWriter(
                    workspace=self.workspace,
                    task_id=str(getattr(self, "_run_id", None) or "default"),
                )
                self._episodic_writer = writer
            writer.append(iter_n, entry)
            logger.debug("PMK conflict written to episodic shard")
        except Exception:
            logger.debug("PMK conflict write to episodic failed (non-fatal)", exc_info=True)

    def _ensure_hypo_manifold(self, context: dict[str, Any]) -> Any:
        """E2-1: 通用假设流形懒加载 + seed, 让 core 主循环可用.

        从 context 抽数值 target 构造 3 个 generic hypothesis
        (h_strong / h_partial / h_null), 复用 hint_coordinator 的公共抽取.
        失败降级 None, 后续 hint 走空路径, 不阻塞主循环.
        """
        if self._hypo_manifold is not None:
            return self._hypo_manifold
        try:
            from huginn.agent.hint_coordinator import extract_numeric_targets
            from huginn.metacog.hypothesis_manifold import (
                Hypothesis,
                HypothesisManifold,
            )

            text_pool = " ".join(
                str(v) for v in (context or {}).values() if isinstance(v, (str, int, float))
            )
            targets = extract_numeric_targets(text_pool)
            manifold = HypothesisManifold()
            for hid, desc, scale, n in (
                ("h_strong", "Result reproduces the target signal", 1.0, 2),
                ("h_partial", "Result partially reproduces the signal", 0.5, 3),
                ("h_null", "No signal / null baseline", 0.0, 1),
            ):
                with contextlib.suppress(ValueError):
                    manifold.add(Hypothesis(
                        h_id=hid,
                        description=desc,
                        predictions={k: v * scale for k, v in targets.items()},
                        n_params=n,
                    ))
            self._hypo_manifold = manifold
        except Exception:
            logger.debug("hypo manifold init failed (non-fatal)", exc_info=True)
            self._hypo_manifold = None
        return self._hypo_manifold

    def _build_hypothesis_prompt(self, context: dict[str, Any]) -> str:
        # 投机执行 hint: 基于历史预测的下一步意图, 注入给 LLM 参考
        # 预测只是 hint, LLM 可以无视, 不强制. 截断到 500 字符防止无界增长
        # — _speculator_hint 有 5 处 append, 不截断 20 轮后可能数 KB.
        hint_block = ""
        if self._speculator_hint:
            hint_block = (
                f"\nSpeculator hint (advisory, may be ignored): {self._speculator_hint[:500]}\n"
                "想返回时必须输出 UNEXPLORED: 块, 列出至少 3 个未探索的方向 "
                "(方法族/等价性陷阱/连通分量/缺口).\n"
            )
        # episodic replay: 回溯最相似的成功/失败情境, 复用方法或避免同样的失败
        replay_block = self._build_episodic_replay_block(context)
        if replay_block:
            hint_block = (hint_block + f"\n### Episodic Replay\n{replay_block}") if hint_block else f"\n### Episodic Replay\n{replay_block}"
        # PMK 三路立场 + 一致性标签 (HUGINN_PMK_INJECT off 或全空时返空串, 不影响 prompt)
        pmk_block = self._build_pmk_block(context)
        if pmk_block:
            hint_block = (hint_block + f"\n### PMK\n{pmk_block}") if hint_block else f"\n### PMK\n{pmk_block}"
        # P1 Task 6: curiosity bonus — self_model 里预测不准的簇, 喂给 hypothesize
        # 让 agent 主动 seek "哪类问题我长期搞不定", 而非等 stagnation 才 escape.
        # toggle HUGINN_CURIOSITY_HINT 默认 off, off 时 _build_curiosity_block 返空.
        curiosity_block = self._build_curiosity_block()
        if curiosity_block:
            hint_block = (hint_block + curiosity_block) if hint_block else curiosity_block
        # 三路检索共用一个 query — 从 context 提取有意义的检索词,
        # 不用 json.dumps (JSON 语法噪声会淹没 embedding 语义锚点)
        ctx_query = self._extract_search_query(context)
        # 领域知识库检索: 命中 first-principles 参考块就拼进 prompt
        kb_block = self._build_kb_text(query=ctx_query)
        if kb_block:
            kb_block = f"\n{kb_block}\n"
        # 知识图谱检索: 把之前 run 发现的实体和关系拉回来, 避免
        # 重复发现已有结论, 也让假设能建立在已有发现上
        kg_block = self._build_kg_text(query=ctx_query)
        if kg_block:
            kg_block = f"\n{kg_block}\n"
        # 长期记忆检索: 跨会话的失败教训和发现. 之前只写不读,
        # 现在闭合 — _learn 写入的迭代记录下次能检索到.
        mem_block = self._build_memory_text(query=ctx_query)
        if mem_block:
            mem_block = f"\n{mem_block}\n"
        # C2: PM 层 trajectory_match 召回 (极限模式才开)
        pm_block = self._build_pm_text()
        if pm_block:
            pm_block = f"\n{pm_block}\n"
        # C2: metacog 信号注入 — target_chain (objective 在目标分解树的位置) +
        # prospective (待执行前瞻意图). 跟 rcb_runner 同源, 之前 autoloop 零接.
        metacog_block = self._build_metacog_block(include_prospective=True)
        if metacog_block:
            metacog_block = f"\n{metacog_block}\n"
        # E2-1: 通用后验引导 hint — 所有 benchmark 都能用假设流形, 不只 RCB.
        # 观测从 context 抽, 空/无字段自动不命中; manifold 失败降级, 不阻塞.
        # MCMC 步进是 advisory, 暂不接入 core (沿用 rcb 的触发), 见路线图 E2-1.
        try:
            _manifold = self._ensure_hypo_manifold(context)
            text_pool = " ".join(
                str(v) for v in context.values() if isinstance(v, (str, int, float))
            )
            from huginn.agent.hint_coordinator import (
                _build_posterior_guided_hint,
                extract_observations,
            )
            _obs = extract_observations(text_pool)
            if _manifold is not None and _obs:
                # E2-1: 在通用主循环推进 MCMC 链 (原本只在 rcb_runner 跑).
                # 每 HUGINN_MCMC_INTERVAL 轮推进一次 mcmc_step, 首次用
                # abductive_inference 初始化 current. 推进是 advisory only,
                # 失败静默降级到 None (hint 走空路径, 不阻塞主循环).
                try:
                    _iter_n = getattr(self, "_iteration", 0)
                    _mcmc_interval = int(
                        os.environ.get("HUGINN_MCMC_INTERVAL", "5"))
                    if _iter_n % _mcmc_interval == 0:
                        _cur = getattr(self, "_mcmc_current", None)
                        if _cur is None or _cur not in _manifold._hyp:
                            # 初始化: 用 posterior 最高的假设 (abductive_inference)
                            _abd = _manifold.abductive_inference(_obs)
                            _cur = _abd.h_id if _abd else None
                            if _cur is None:
                                _h_ids = list(_manifold._hyp)
                                _cur = _h_ids[0] if _h_ids else None
                        if _cur is not None:
                            _prev = _cur
                            _rng = getattr(self, "_mcmc_rng", None)
                            _next_h, _next_logp = _manifold.mcmc_step(
                                _obs, _cur, rng=_rng,
                                cached_log_p_current=getattr(
                                    self, "_mcmc_cached_log_p", None),
                                global_proposal_prob=0.3,
                            )
                            self._mcmc_current = _next_h
                            self._mcmc_cached_log_p = _next_logp
                            self._mcmc_step_count = getattr(
                                self, "_mcmc_step_count", 0) + 1
                            if _next_h != _prev:
                                self._mcmc_accept_count = getattr(
                                    self, "_mcmc_accept_count", 0) + 1
                except Exception:
                    logger.debug(
                        "MCMC step advance skipped (non-fatal)", exc_info=True)
                # MCMC 动态采样路径接入: 把采样链当前驻留的假设作为 hint 注入,
                # 补充 abductive_inference (argmax) 的贪婪盲区. 无 MCMC 状态时
                # 传 None, _build_posterior_guided_hint 内部跳过, 行为不变.
                _mcmc_cur = getattr(self, "_mcmc_current", None)
                _pg = _build_posterior_guided_hint(
                    _manifold, _obs, mcmc_current=_mcmc_cur)
                if _pg:
                    hint_block = (
                        hint_block + f"\n### Posterior-guided\n{_pg}"
                        if hint_block else f"\n### Posterior-guided\n{_pg}"
                    )
        except Exception:
            logger.debug("posterior hint injection skipped (non-fatal)", exc_info=True)
        # H0: stable_principles 注入 (修 P3 断链 — 之前只进 chat agent system prompt,
        # autoloop 完全跳过 PM 层). 取 top-5 避免塞爆 prompt.
        try:
            from huginn.memory.longterm import load_stable_principles
            _principles = load_stable_principles()[:5]
        except Exception:
            _principles = []
        principles_block = (
            "\n".join(f"- {p}" for p in _principles) if _principles else ""
        )
        if principles_block:
            principles_block = (
                f"\n### Stable Principles (procedural memory)\n{principles_block}\n"
            )
        # 视觉基元: 上一轮 tool 输出的数值指针 (峰值/趋势/异常),
        # 给 LLM 具体坐标锚定推理 — Thinking with Visual Primitives 的
        # "point while it reasons" 原则, Mirage 效应的文本路径
        visual_block = getattr(self, "_last_visual_context", "")
        if visual_block:
            visual_block = (
                f"\n### Visual Primitives (from last tool output)\n{visual_block}\n"
            )
        ctx_blob = json.dumps(context, ensure_ascii=False).lower()
        from huginn.autoloop.engine import _MATH_SIGNALS
        math_block = (
            self._MATH_DEPTH_PROMPT_BLOCK
            if any(s in ctx_blob for s in _MATH_SIGNALS)
            else ""
        )
        # MatterChat 启发: 把上轮 execution 结果摘要注入 hypothesis prompt,
        # 让假设建立在"上轮实际发生了什么"之上, 不只看 workspace 变化.
        # _last_execution_result 在 _execute 里写入, 之前只 _build_plan_prompt 用.
        exec_block = ""
        last_exec = getattr(self, "_last_execution_result", None)
        if last_exec and isinstance(last_exec, dict):
            _tool = last_exec.get("_tool_name", "unknown")
            _res = last_exec.get("result", last_exec)
            _summary = json.dumps(_res, ensure_ascii=False, default=str)[:500]
            exec_block = f"\n### Last Execution Result ({_tool})\n{_summary}\n"
        # H2: frontier_ranked 注入 — Ising 能量最低 K-子集未测试假设.
        # 之前 frontier()/frontier_ranked() 0 生产调用, Ising 排序整套死代码.
        # 现在作为 hint 注入, LLM 可选优先测这些 (refute 过的 parent 的子假设优先,
        # 同 sibling_group 互斥已避开). ponytail: hint 不强制, LLM 自己决定.
        frontier_block = ""
        try:
            _frontier = self.hypothesis_graph.frontier_ranked(
                top_k=3,
                # settings 可能未初始化 (__new__ 建的空引擎), 容错取默认 0
                phys_gain=getattr(getattr(self, "settings", None), "phys_steering_gain", 0.0),
            )
            if _frontier:
                _lines = []
                for nd in _frontier:
                    _stmt = (nd.statement or "")[:100]
                    _lines.append(f"- [{nd.id}] {_stmt}")
                frontier_block = (
                    "\n### Untested Hypotheses (Ising-ranked, energy-low)\n"
                    + "\n".join(_lines) + "\n"
                    "Consider testing one of these before generating a new hypothesis.\n"
                )
        except Exception:
            logger.debug("frontier_ranked injection failed", exc_info=True)
        # P0: FAILED.md / PROVED.md durable state 注入 (chaoxu 启发).
        # context 压缩后 agent 重读这两个文件, 不重试死路, 不重新证明已过的.
        # ponytail: 读文件首 N 行避免膨胀, full text 留给 _extract_compact_attachments.
        failed_block = ""
        proved_block = ""
        try:
            from huginn.autoloop.hypothesis_loop import HypothesisGraph
            _ws = str(self.workspace) if hasattr(self, "workspace") else None
            _failed_txt = HypothesisGraph.load_failed(_ws)
            if _failed_txt:
                _failed_lines = _failed_txt.strip().split("\n")[:40]
                failed_block = (
                    "\n### Dead Routes (FAILED.md)\n"
                    + "\n".join(_failed_lines) + "\n"
                    "Do NOT re-attempt these unless the Reopen-if condition is met.\n"
                )
            _proved_txt = HypothesisGraph.load_proved(_ws)
            if _proved_txt:
                _proved_lines = _proved_txt.strip().split("\n")[:40]
                proved_block = (
                    "\n### Verified Results (PROVED.md)\n"
                    + "\n".join(_proved_lines) + "\n"
                    "These are already established — build on them.\n"
                )
        except Exception:
            logger.debug("FAILED/PROVED injection failed", exc_info=True)
        # 想象力引导: 高 surprise 或连续 refine 时, 要求 LLM 跳出分析思维,
        # 考虑反事实假设. 基于 MToM P4 (hybrid ST+TT): 心智模型预测错误时
        # 切到仿真理论重新建模. 结构切换在数学结构族之间, 不是随机猜.
        imagination_block = ""
        if self._should_imaginate():
            imagination_block = self._IMAGINATION_PROMPT_BLOCK
        # Failure mode feedback (Dream Layer): 上轮 validate 描述的"如何崩溃"
        # 注入 hypothesis prompt, 让 agent 从崩溃模式中找新发现.
        # _last_failure_mode 在 _validate 里写入, 空字符串表示无上轮或未解析出.
        fail_block = ""
        _last_fail = getattr(self, "_last_failure_mode", "")
        if _last_fail:
            fail_block = f"\n### Previous Failure Mode\nIf the previous hypothesis is wrong, it would fail in this way:\n{_last_fail}\nConsider whether this failure mode points to a new hypothesis.\n"
        # Git log: EurekAgent artifact engineering — 让 agent 看到前几轮
        # 做了什么, 避免重复尝试已失败的方案. 只取 oneline 前 10 条.
        git_log_block = ""
        try:
            import subprocess as _sp

            _r = _sp.run(
                ["git", "log", "--oneline", "-10"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if _r.returncode == 0 and _r.stdout.strip():
                git_log_block = (
                    f"\n### Recent Experiments (git log)\n{_r.stdout.strip()}\n"
                )
        except Exception:
            logger.debug("git log block build skipped", exc_info=True)

        # 分量代表制: 多条独立探索路线时, 给 LLM 看各路线的代表假设,
        # 防止单分量靠节点数主导综合判断. 只在 >1 分量时注入, advisory.
        # v11: 升级为 cluster_block — 优先用 cluster_by_dimension 展示维度分布,
        # 退化为 topology_block (分量代表) 当 dimension 全空.
        cluster_block = ""
        try:
            _clusters = self.hypothesis_graph.cluster_by_dimension()
            _known_dims = {k: v for k, v in _clusters.items() if k != "unknown"}
            if _known_dims and len(_clusters) > 1:
                lines = []
                for dim, nodes in list(_known_dims.items())[:5]:
                    _stmt = nodes[0].statement[:120] if nodes else ""
                    lines.append(f"  - {dim} ({len(nodes)} 个假设): {_stmt}")
                cluster_block = (
                    "\n### Cluster (advisory)\n"
                    "当前假设按 dimension 分布:\n"
                    + "\n".join(lines)
                    + "\n新假设应优先补未覆盖的 dimension, 避免在已饱和维度堆叠.\n"
                )
            else:
                # 退化路径: dimension 全空时用 topology_block (分量代表)
                reps = self._metacog_component_representatives()
                if len(reps) > 1:
                    lines = []
                    for rid in reps[:5]:
                        try:
                            stmt = self.hypothesis_graph.get(rid).statement
                        except Exception:
                            logger.debug("best-effort op failed", exc_info=True)
                            stmt = ""
                        lines.append(f"  - {rid}: {stmt[:120]}")
                    cluster_block = (
                        f"\n### Topology (advisory)\n"
                        f"当前有 {len(reps)} 条独立探索路线, 代表假设分别是:\n"
                        + "\n".join(lines)
                        + "\n"
                        "综合判断时不要让某条路线靠节点数主导, 注意挑战和重定向.\n"
                    )
        except Exception:
            logger.debug("cluster block build skipped", exc_info=True)

        # 按优先级拼接, 超预算自动裁剪低优先级 block
        blocks = self._apply_block_patches(
            [
                (
                    "body",
                    f"""You are an autonomous material science research agent.

Perceived context:
{json.dumps(context, indent=2, ensure_ascii=False)[:2000]}

Generate 3 divergent candidate hypotheses. Each MUST be grounded in a
DIFFERENT assumption dimension. Pick dimensions from this list (or propose
a new one tagged [NEW]):
- composition (Ca/Si/Al/O ratio, doping, alloy)
- temperature (thermal dependence, phase transition)
- defect (vacancy, dislocation, interface)
- structure (crystal symmetry, lattice parameter)
- transport (diffusion, conductivity, mobility)

Format each candidate as:
[DIM: <dimension>] <statement> | pro: ... | con: ...

After listing 3, select the most testable+novel one after "SELECTED:".
The 3 candidates must NOT be variations of each other — if two share the
same dimension, the second is invalid and must be replaced.
Ground it in the domain knowledge context above when relevant.
Prefer hypotheses that can be expressed as governing PDEs, variational
principles, or conservation laws; identify the mathematical structure
before proposing numerical experiments.

Hypothesis:""",
                ),
                ("git_log", git_log_block),
                ("fail", fail_block),
                ("imagination", imagination_block),
                ("exec", exec_block),
                ("frontier", frontier_block),
                ("failed", failed_block),
                ("proved", proved_block),
                ("math", math_block),
                ("kg", kg_block),
                ("visual", visual_block),
                ("kb", kb_block),
                ("principles", principles_block),
                ("mem", mem_block),
                ("pm", pm_block),
                ("metacog", metacog_block),
                ("cluster", cluster_block),
                ("skill", self._build_skill_context_block()),
                ("hint", hint_block),
            ],
            "hypothesize",
        )
        return self._trim_to_budget(blocks, phase="hypothesize")

    # ── 元认知层辅助 ────────────────────────────────────────────────
    # 懒加载 metacog 组件, 避免循环 import 和测试 mock 复杂度.
    # _hypothesize 拿到假设后调 _metacog_audit_hypothesis 做等价性审计 +
    # 方法族归类. 审计是 advisory (不阻断), 对齐用户 "math 结构 advisory" 偏好.

    def _get_metacog_auditor(self):
        if self._metacog_auditor is None:
            from huginn.metacog.equivalence_auditor import EquivalenceAuditor

            self._metacog_auditor = EquivalenceAuditor(model=self.model)
        return self._metacog_auditor

    def _get_metacog_block_registry(self):
        if self._metacog_block_registry is None:
            from huginn.metacog.block_registry import BlockRegistry

            self._metacog_block_registry = BlockRegistry(
                auditor=self._get_metacog_auditor()
            )
        return self._metacog_block_registry

    def _get_metacog_method_registry(self):
        if self._metacog_method_registry is None:
            from huginn.metacog.method_registry import MethodRegistry

            self._metacog_method_registry = MethodRegistry()
        return self._metacog_method_registry

    def _get_metacog_convergence_detector(self):
        if self._metacog_convergence_detector is None:
            from huginn.metacog.depth_search import PrematureConvergenceDetector

            self._metacog_convergence_detector = PrematureConvergenceDetector()
        return self._metacog_convergence_detector

    def _get_metacog_completion_auditor(self):
        if self._metacog_completion_auditor is None:
            from huginn.metacog.completion_auditor import CompletionAuditor

            self._metacog_completion_auditor = CompletionAuditor(
                convergence_detector=self._get_metacog_convergence_detector(),
                equivalence_auditor=self._get_metacog_auditor(),
            )
        return self._metacog_completion_auditor

    # ── 触觉层: 同构不同性 → 触发新 hypothesis ──────────────────
    # detect_isomorphic_anomaly 发现 "结构同 + 力学不同" (石墨 vs 金刚石类) 时,
    # 调 _hypothesize 生成解释差异的新 hypothesis. ponytail: hypothesis_generator
    # 不可用 (无 model / _hypothesize 失败) 只 log warn 不 fatal.
    async def trigger_isomorphic_anomaly_hypothesis(
        self, anomaly_pairs: list[tuple[str, str]],
    ) -> list[str]:
        """同构不同性异常 → 触发新 hypothesis 生成.

        Args:
            anomaly_pairs: detect_isomorphic_anomaly 的返回值 [(h_a, h_b), ...].

        Returns:
            生成的 hypothesis 文本列表 (失败/不可用时为空).
        """
        if not anomaly_pairs:
            return []
        generated: list[str] = []
        for h_a, h_b in anomaly_pairs:
            logger.warning(
                "isomorphic anomaly: structure matches but haptic differs "
                "(%s vs %s) — triggering new hypothesis", h_a, h_b,
            )
            ctx = {
                "summary": (
                    f"Isomorphic anomaly detected between {h_a} and {h_b}: "
                    "structures match (RMSD<1Å) but mechanical properties differ "
                    "(haptic_distance>=0.5). Generate a hypothesis explaining the "
                    "structure-property mismatch (e.g. graphite vs diamond)."
                ),
                "anomaly_pair": (h_a, h_b),
            }
            try:
                new_hyp = await self._hypothesize(ctx)
                if new_hyp:
                    generated.append(new_hyp)
                    self._last_hypothesis = new_hyp
                    self._last_raw_hypothesis = new_hyp
            except Exception:
                logger.warning(
                    "hypothesis_generator unavailable for anomaly %s/%s, "
                    "log only (non-fatal)", h_a, h_b, exc_info=True,
                )
        return generated

    # ── 对齐层: surprise 驱动发现 → 触发新 hypothesis ──────────────
    # check_surprise 发现 "结构预测的力学跟实际力学对不上" 时, 调 _hypothesize
    # 生成解释差异的新 hypothesis. ponytail: hypothesis_generator 不可用
    # (无 model / _hypothesize 失败) 只 log warn 不 fatal.
    async def trigger_alignment_surprise_hypothesis(
        self, surprise_findings: list[tuple[str, float]],
    ) -> list[str]:
        """对齐 surprise → 触发新 hypothesis 生成.

        Args:
            surprise_findings: [(h_id, surprise_score), ...] — score > threshold.

        Returns:
            生成的 hypothesis 文本列表 (失败/不可用时为空).
        """
        if not surprise_findings:
            return []
        generated: list[str] = []
        for h_id, score in surprise_findings:
            logger.warning(
                "surprise detected: score=%.2f on %s — "
                "structure-haptic alignment violated, triggering new hypothesis",
                score, h_id,
            )
            ctx = {
                "summary": (
                    f"Alignment surprise detected on hypothesis {h_id} "
                    f"(surprise score={score:.2f}): the mechanical properties "
                    f"deviate significantly from what the learned structure-haptic "
                    f"alignment predicts. Generate a hypothesis explaining the mismatch "
                    f"(e.g. structural phase transition, Kohn anomaly, or novel mechanism)."
                ),
                "h_id": h_id,
                "surprise_score": score,
            }
            try:
                new_hyp = await self._hypothesize(ctx)
                if new_hyp:
                    generated.append(new_hyp)
                    self._last_hypothesis = new_hyp
                    self._last_raw_hypothesis = new_hyp
            except Exception:
                logger.warning(
                    "hypothesis_generator unavailable for surprise %s (score=%.2f), "
                    "log only (non-fatal)", h_id, score, exc_info=True,
                )
        return generated

    def _metacog_check_effort_floor(self) -> tuple[bool, str]:
        """[deprecated] 旧的最小努力下限检查, 保留向后兼容.

        新代码用 _metacog_check_completion — 它在 effort floor 之上加了
        等价性陷阱检查和显式不完整性自白. 这个方法只是 thin wrapper,
        保留是因为可能有外部调用方直接调它.
        """
        return self._metacog_check_completion()

    def _metacog_check_completion(self) -> tuple[bool, str]:
        """反完成审计: 综合检查是否过早收敛.

        替代旧的纯 effort floor, 调 CompletionAuditor 做四层检查:
        - 最小努力下限 (迭代/方法族/连通分量)
        - 等价性陷阱 (内部调 EquivalenceAuditor)
        - 显式不完整性自白 (从 _last_raw_hypothesis 提取 UNEXPLORED 块)
        - 对抗否决 (本 engine 暂不接 red_team, 留口子)

        advisory: 出错放行, 不阻断.
        """
        try:
            auditor = self._get_metacog_completion_auditor()
            families_explored = len(
                [
                    f
                    for f in self._get_metacog_method_registry().all()
                    if f.member_agent_ids
                ]
            )
            live_components = self.hypothesis_graph.component_count()

            # 从最近一次 LLM 原始输出提取 UNEXPLORED: 块
            # ponytail: 字符串切片, 不上正则. 升级路径: 结构化 schema.
            unexplored = ""
            raw = getattr(self, "_last_raw_hypothesis", "") or ""
            if "UNEXPLORED:" in raw:
                unexplored = raw.split("UNEXPLORED:", 1)[1].strip()
                # 截到下一个大写标记或结尾, 避免把后续块都吞进来
                for marker in ["\n\nHYPOTHESIS", "\n\nSELECTED", "\n\nRATIONALE"]:
                    if marker in unexplored:
                        unexplored = unexplored.split(marker)[0].strip()
                        break

            checklist = auditor.audit(
                iteration=self._iteration,
                families_explored=families_explored,
                live_components=live_components,
                total_iterations=(
                    self._max_iterations if hasattr(self, "_max_iterations") else 10
                ),
                candidate_finding=getattr(self, "_last_hypothesis", "") or "",
                original_problem=str(getattr(self, "_objective", "") or ""),
                unexplored_declaration=unexplored,
            )

            if not checklist.is_complete:
                return True, checklist.block_reason()
            return False, ""
        except Exception:
            logger.debug("metacog completion check failed", exc_info=True)
            return False, ""  # 出错不阻断, advisory

    def _metacog_check_topology_collapse(self) -> None:
        """坍缩检测: 连通分量数过低时, 强制从冷门族启动新探索.

        对应 prompt: "不要让一种方法占据主导...并发起新一轮".
        is_collapsed 时把重定向建议拼进 _speculator_hint, 下轮 hypothesize
        会看到. advisory: 不阻断, 出错放行.
        """
        try:
            from huginn.metacog.depth_search import DynamicComponentFloor

            # 动态下限: 早期 4, 中期 2, 后期 1. 实例化成本可忽略.
            floor = DynamicComponentFloor().current_floor(
                self._iteration, self._max_iterations
            )
            if self.hypothesis_graph.is_collapsed(min_components=floor):
                redirect = self._get_metacog_method_registry().suggest_redirect()
                hint = (
                    f"[topology] 搜索空间坍缩! "
                    f"连通分量 {self.hypothesis_graph.component_count()} < 下限 {floor}. "
                )
                if redirect:
                    hint += f"强制重定向到 {redirect.target_family}: {redirect.reason}"
                else:
                    hint += "建议启动新方法族探索"
                    # 接入 BlockRegistry: 坍缩且无重定向目标 → 当前主导方法族
                    # 标记为 incubating (缺口尚未具体化).
                    # v23 Round 8: 已实际调用 _get_metacog_block_registry(),
                    # 阻塞-新机制重启协议已接入主循环.
                    try:
                        _block_reg = self._get_metacog_block_registry()
                        _dominant = self._metacog_dominant_family()
                        if _dominant:
                            _block_reg.block(
                                method_family=_dominant,
                                block_reason=(
                                    f"搜索空间坍缩: 连通分量 {self.hypothesis_graph.component_count()} "
                                    f"< 下限 {floor}, 且无重定向目标"
                                ),
                            )
                    except Exception:
                        logger.debug("block_registry register skipped (non-fatal)", exc_info=True)
                if self._speculator_hint:
                    self._speculator_hint = f"{self._speculator_hint}\n{hint}"
                else:
                    self._speculator_hint = hint
                logger.info("metacog: %s", hint)
        except Exception:
            logger.debug("metacog topology check failed", exc_info=True)

    def _metacog_component_representatives(self) -> list[str]:
        """每个连通分量的代表假设 id.

        代表参与根 agent 综合判断, 防止单分量靠节点数主导.
        对应 prompt: "根智能体应反复综合、挑战、重定向".
        出错返回空列表, 调用方按 advisory 处理.
        """
        try:
            components = self.hypothesis_graph.connected_components()
            reps = []
            for comp in components:
                rep = self.hypothesis_graph.component_representative(comp)
                if rep:
                    reps.append(rep)
            return reps
        except Exception:
            return []

    def _metacog_dominant_family(self) -> str:
        """当前占主导地位的方法族名 (节点数最多的分量代表).

        BlockRegistry 标记阻塞路线时用 — 坍缩场景下谁占主导就标记谁.
        出错返回空串, 调用方按 advisory 处理.
        """
        try:
            components = self.hypothesis_graph.connected_components()
            if not components:
                return ""
            # 取最大分量的代表
            largest = max(components, key=len)
            rep = self.hypothesis_graph.component_representative(largest)
            return rep or ""
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            return ""

    @staticmethod
    def _extract_lucid_prereqs(raw: str) -> dict[str, str]:
        """从 LLM 输出里解析 LUCID review 的三项必要条件.

        prompt 要求 LLM 在 SELECTED 后输出:
        - necessary condition: ...
        - hidden assumption: ...
        - falsifiable test: ...

        返回 {"necessary": ..., "hidden": ..., "falsifiable": ...}, 缺项为空串.
        LLM 格式不固定时降级到关键词模糊匹配."""
        if not raw:
            return {"necessary": "", "hidden": "", "falsifiable": ""}
        text = raw.lower()
        result = {"necessary": "", "hidden": "", "falsifiable": ""}
        # 关键词 + 后续行内容, 容忍中英文标点
        patterns = {
            "necessary": r"necessary[^:：]*[:：]\s*(.+)",
            "hidden": r"hidden\s*assumption[^:：]*[:：]\s*(.+)",
            "falsifiable": r"falsifiable\s*test[^:：]*[:：]\s*(.+)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text)
            if m:
                # 取到行尾或句号
                val = m.group(1).split("\n")[0].strip().rstrip(".。")
                result[key] = val[:300]  # 长度限制, 防异常输入
        return result
