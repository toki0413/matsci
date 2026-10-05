"""BranchIncubator — 探索分支隔离孵化器.

核心思想: 每个方法族起一个独立 Subagent, 互不可见, 对抗 LLM 快速收敛偏差.
复用三个闲置积木:
- context_isolation: ContextBundle + isolate — 探索 agent 上下文裁剪
- method_registry: MethodRegistry — 族管理 + 收敛度监控
- depth_search: PrematureConvergenceDetector — 反完成审计

复用 SubagentDispatch 起真并发 agent (explore spec, 只读不写).

接入点: engine._hypothesize — 加 use_branch_incubator flag, 默认 off.
  flag on 时替代 main+hot_model 2 路采样, 改用 N 路隔离采样.

ponytail: 单文件, 不引入新组件. Subagent 失败降级到 family.essence 模板.
升级路径: 接入 agents/swarm.py 做跨进程分布式孵化.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from huginn.agents.subagent import SubagentDispatch, SubagentResult
from huginn.metacog.context_isolation import ContextBundle, isolate
from huginn.metacog.depth_search import PrematureConvergenceDetector
from huginn.metacog.method_registry import MethodRegistry

logger = logging.getLogger(__name__)


@dataclass
class BranchResult:
    """单条探索分支的产出."""

    family_id: str
    agent_id: str
    hypothesis: str  # Subagent summary 或模板降级
    full_output: str = ""
    success: bool = True
    error: str | None = None
    tokens_used: int = 0
    round_idx: int = 0
    # PTD tree-shape: layer2 sub-branch 的父 agent_id. layer1 为空.
    parent_agent_id: str = ""
    # P4 (chaoxu 启发): 分配的 model family (如 "openai"/"anthropic"/"deepseek").
    # 空串 = 未指定 (向后兼容). 调用方可据此选不同 profile, 实现跨模型多样性.
    model_family: str = ""
    # rollout value (MCTS 树的价值) — 由调用方经 value_fn 注入, 典型来源是
    # step_verifier.aggregate_step_scores (轨迹 step 分 → 路径价值, 越高越好).
    # None = 未评分 → prune 回退旧价值 (min tokens_used), 行为不变.
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "agent_id": self.agent_id,
            "hypothesis": self.hypothesis,
            "success": self.success,
            "error": self.error,
            "tokens_used": self.tokens_used,
            "round_idx": self.round_idx,
            "parent_agent_id": self.parent_agent_id,
            "model_family": self.model_family,
            "value": self.value,
        }


class BranchIncubator:
    """探索分支隔离孵化器.

    每轮 run_round:
    1. MethodRegistry.suggest_redirect 给 N 个 branch 分族 (冷门优先)
    2. ContextBundle + isolate 裁剪上下文, 每个 Subagent 看不到其他族
    3. asyncio.gather 并发起 N 个 SubagentDispatch.dispatch("explore", ...)
    4. 失败的 branch 降级到 family.essence 模板
    5. PrematureConvergenceDetector 检查, 过热族 mark_blocked
    """

    DEFAULT_N_BRANCHES = 3

    def __init__(
        self,
        registry: MethodRegistry | None = None,
        detector: PrematureConvergenceDetector | None = None,
        dispatch: SubagentDispatch | None = None,
    ) -> None:
        self._registry = registry or MethodRegistry()
        self._detector = detector or PrematureConvergenceDetector()
        self._dispatch = dispatch or SubagentDispatch()

    @property
    def registry(self) -> MethodRegistry:
        return self._registry

    async def run_round(
        self,
        task: str,
        agent_factory: Any,
        *,
        n_branches: int = DEFAULT_N_BRANCHES,
        math_background: str = "",
        researcher_intuition: str = "",
        round_idx: int = 0,
        total_rounds: int = 10,
        depth: int = 1,
        width: int = 2,
        wave_mode: str = "push",
        model_families: list[str] | None = None,
        value_fn: Callable[[BranchResult], Any] | None = None,
        budget_remaining_fn: Callable[[], float | None] | None = None,
        trace_fn: Callable[[str, str, str], None] | None = None,
        slice_min_s: float | None = None,
    ) -> list[BranchResult]:
        """跑一轮隔离探索.

        agent_factory 必传, 跟 SubagentDispatch.dispatch 一致.
        返回 N 个 BranchResult, 失败的也有记录 (success=False).

        depth=1: flat asyncio.gather N (Part 1 兼容).
        depth=2: PTD tree-shape — layer1 N flat → 每 layer1 成功 branch 派
          width 个 sub-branch → 每 parent prune top-1 (tokens_used 最小 + success).
          layer2 全失败的 parent 回退到 layer1. 返回长度 = n_branches.

        P4 (chaoxu 启发):
        - wave_mode: "push" 生成假设 / "verify" 验证上轮假设. 轮替使用避免
          "自己生成自己验证"的确认偏差. verify 模式 enhanced_task 加验证引导.
        - model_families: 传入可用 model family 列表时, 给每个 branch 分配一个
          model_family (尽量分散), 记录在 BranchResult.model_family 供调用方调度.
          None 时不分配 (向后兼容).
        - value_fn: rollout value 注入器 (MCTS select/prune 用). 每个 BranchResult
          过一遍它 (可同步或 async, 典型来自 step_verifier 的 PRM 打分),
          结果写进 .value (越高越好). layer2 prune 优先取 max value; value 缺失
          (None) 时回退旧价值 min(tokens_used). None 时所有 .value 保持 None →
          行为 100% 不变.

        D-slice (细粒度预算切片, 不设硬时长):
        把一轮拆成**有序、各自独立有价值**的 slice —
          S1 layer1 扇出   (产出 N 条可用假设, 核心)
          S2 layer1 PRM 打分 (可选)
          S3 layer2 精修扇出 (逐 parent, 可选)
          S4 layer2 PRM 打分 (可选)
          S5 prune          (廉价, 总是执行)
        每个可选 slice **启动前**查一次剩余挂钟预算 `budget_remaining_fn()`; 剩余
        不足 `slice_min_s` 就跳过该 slice, 用"已产出的最优结果"收尾 —— 单阶段超支
        被限制在**一个 slice** 以内 (而非整棵树 2 层 × N 路). 这是"少给硬时长、多
        切细粒度"的实现: 不 kill 正在跑的动作, 只拒绝启动注定越预算的下一个 slice.

        切片成本**自校准**: 下一片 (layer2) 的成本用上一片 (layer1) 的**实测挂钟**
        估计, 门槛取 `max(slice_min_s, 上一片实测)`. 于是"刚烧了 300s 只剩 120s"
        时不会再启动一个同样要 ~300s 的 layer2 —— 而不是拿一个拍脑袋的固定阈值.
        `budget_remaining_fn=None` (非长程 / 无 goal) 或返回 None → 全部 slice 照跑,
        行为 100% 不变. `slice_min_s` 缺省读 HUGINN_BRANCH_SLICE_MIN_S (默认 60s).
        """
        # slice 最低成本 (秒): 剩余预算 < 此值就不启动下一个可选 slice.
        _min_s = (
            slice_min_s if slice_min_s is not None
            else float(os.environ.get("HUGINN_BRANCH_SLICE_MIN_S", "60"))
        )
        bundle = ContextBundle(
            global_math_background=math_background,
            task_definition=task,
            current_preferred_hypothesis=None,  # exploratory → 自动放宽隔离
            researcher_intuition=researcher_intuition,
            method_family_registry=self._registry.to_dict(),
        )

        family_assignments = self._assign_families(n_branches)
        # P4: model_family 分配 (尽量分散, None 时不分配)
        model_assignments = (
            self._assign_model_families(n_branches, model_families)
            if model_families else [""] * n_branches
        )
        # Layer 1: flat asyncio.gather (depth=1 和 depth=2 都跑这步)
        # D-slice S1: 核心 slice, 不可跳过 (调用方启动前已查 _budget_exhausted).
        _t_layer1 = time.monotonic()
        coros = [
            self._run_single_branch(
                fam, bundle, task, agent_factory, round_idx,
                model_family=mf, wave_mode=wave_mode,
                budget_remaining_fn=budget_remaining_fn,
            )
            for fam, mf in zip(family_assignments, model_assignments)
        ]
        raw_results = await asyncio.gather(*coros, return_exceptions=True)
        # D-slice 自校准: layer1 实测挂钟 = 下一片 (同形状的 layer2) 的成本估计.
        _layer1_cost = time.monotonic() - _t_layer1

        layer1: list[BranchResult] = []
        for family_id, raw in zip(family_assignments, raw_results):
            if isinstance(raw, Exception):
                # gather 抓到的异常 (不应发生, _run_single_branch 内部已 catch)
                layer1.append(BranchResult(
                    family_id=family_id,
                    agent_id="",
                    hypothesis=self._fallback_hypothesis(family_id),
                    success=False,
                    error=f"unexpected: {raw}",
                    round_idx=round_idx,
                ))
                continue
            layer1.append(raw)

        # 反完成审计: 过热族 mark_blocked, 下一轮强制 redirect
        # (layer1 后调, 跟 Part 1 一致; layer2 是 refinement, 不改 family 分布)
        self._check_convergence(round_idx, total_rounds, layer1)
        # D-slice 自校准门槛: 后续可选 slice 至少要有"跑完一个同形状切片"的预算.
        # 固定 _min_s 只作下限; 上界用 layer1 实测 (run72: layer1 烧 300s 只剩
        # 120s → 门槛 300s → 拒绝启动同样要 ~300s 的 layer2).
        _need = max(_min_s, _layer1_cost)
        # D-slice S2: layer1 PRM 打分 (可选). 预算不足则跳过 → value 保持 None,
        # prune 回退 tokens_used, 结果仍可用.
        if self._slice_affordable(budget_remaining_fn, _need):
            await self._assign_values(layer1, value_fn)
        else:
            self._slice_skip(
                trace_fn, "slice=layer1_value", _need,
            )

        if depth < 2:
            return layer1  # Part 1 flat 行为

        # D-slice S3 门: layer2 精修是可选 slice. 剩余预算不足 → 直接用 layer1
        # 收尾, 不启动注定越预算的深搜. 单阶段超支由此 ≤ 一个 slice (layer1).
        if not self._slice_affordable(budget_remaining_fn, _need):
            self._slice_skip(
                trace_fn, "slice=layer2 (return layer1)", _need,
            )
            logger.info(
                "branch_incubator: skip layer2 (budget<%.0fs, layer1 cost %.0fs), "
                "return %d layer1 branches",
                _need, _layer1_cost, len(layer1),
            )
            return layer1

        # Layer 2: PTD tree-shape — 每 layer1 成功 branch 派 width 个 sub-branch
        layer2_by_parent = await self._run_tree_layer(
            layer1, bundle, task, agent_factory, round_idx, width,
            wave_mode=wave_mode, value_fn=value_fn,
            budget_remaining_fn=budget_remaining_fn,
            slice_min_s=_need, trace_fn=trace_fn,
        )

        # Prune + Fallback: 每 parent 保留 top-1 (value 最大 + success;
        # value 缺失时回退 tokens_used 最小), layer1 失败 / layer2 全失败 → 回退 layer1
        final: list[BranchResult] = []
        for l1 in layer1:
            if not l1.success or not l1.hypothesis:
                final.append(l1)  # layer1 失败, 不派 layer2
                continue
            subs = layer2_by_parent.get(l1.agent_id, [])
            successful_subs = [s for s in subs if s.success and s.hypothesis]
            if not successful_subs:
                final.append(l1)  # layer2 全失败, fallback layer1
                continue
            final.append(self._pick_winner(successful_subs))
        return final

    @staticmethod
    def _slice_affordable(
        budget_remaining_fn: Callable[[], float | None] | None,
        need_s: float,
    ) -> bool:
        """下一个可选 slice 是否负担得起 (剩余挂钟预算 ≥ need_s).

        D-slice 的单一预算判据. `budget_remaining_fn=None` (非长程) 或返回 None
        (无 goal) → 视为不受预算约束, 返回 True (行为 100% 不变). 查询异常
        fail-open (不误判耗尽而跳过真该跑的 slice).
        """
        if budget_remaining_fn is None:
            return True
        try:
            rem = budget_remaining_fn()
        except Exception:  # 防御: 预算查询失败 fail-open
            logger.debug("branch budget_remaining_fn failed", exc_info=True)
            return True
        if rem is None:
            return True
        return rem >= need_s

    @staticmethod
    def _slice_skip(
        trace_fn: Callable[[str, str, str], None] | None,
        what: str,
        need_s: float,
    ) -> None:
        """记录一次"因预算不足跳过可选 slice" — 控制面可观测, 不静默."""
        logger.info("branch_incubator: skip %s (budget<%.0fs)", what, need_s)
        if trace_fn is None:
            return
        try:
            trace_fn(
                "branch_slice_skip",
                f"{what}: budget<{need_s:.0f}s",
                "skip",
            )
        except Exception:  # 防御: 观测失败不影响主流程
            logger.debug("branch trace_fn failed", exc_info=True)

    @staticmethod
    async def _assign_values(
        results: list[BranchResult],
        value_fn: Callable[[BranchResult], Any] | None,
        budget_remaining_fn: Callable[[], float | None] | None = None,
        need_s: float = 0.0,
    ) -> None:
        """用 value_fn 给每个 branch 注入 rollout value (原地改 .value).

        value_fn 可为同步或 async (返回协程时自动 await, 如 step_verifier 的
        PRM 打分). None → 全部保持 None (旧行为). 单个失败不影响其他 branch.

        D-slice S2/S4: PRM 打分逐 branch 是顺序 LLM 调用 (累计耗时). 每条前查一次
        剩余预算, 不足 need_s 就停在此条, 后面 branch 的 value 保持 None (回退
        token 剪枝) —— 把"打分"也切成可中断的细粒度 slice.
        """
        if value_fn is None:
            return
        for r in results:
            if not BranchIncubator._slice_affordable(budget_remaining_fn, need_s):
                logger.info(
                    "branch_incubator: stop value scoring at %s (budget<%.0fs)",
                    r.agent_id, need_s,
                )
                return
            try:
                v = value_fn(r)
                if inspect.isawaitable(v):
                    v = await v
                r.value = v
            except Exception:
                logger.debug("branch value_fn failed (non-fatal)", exc_info=True)
                r.value = None

    @staticmethod
    def _pick_winner(subs: list[BranchResult]) -> BranchResult:
        """从同一 parent 的 sub-branch 里选 winner.

        有 value 的优先按 value 最大 (tie-break 更省 tokens); 全无 value →
        回退旧价值 min(tokens_used). 保证 value_fn=None 时行为 100% 不变.
        """
        valued = [s for s in subs if s.value is not None]
        if valued:
            return max(valued, key=lambda r: (r.value, -r.tokens_used))
        return min(subs, key=lambda r: r.tokens_used)

    async def _run_tree_layer(
        self,
        layer_branches: list[BranchResult],
        bundle: ContextBundle,
        task: str,
        agent_factory: Any,
        round_idx: int,
        width: int,
        wave_mode: str = "push",
        value_fn: Callable[[BranchResult], Any] | None = None,
        budget_remaining_fn: Callable[[], float | None] | None = None,
        slice_min_s: float = 0.0,
        trace_fn: Callable[[str, str, str], None] | None = None,
    ) -> dict[str, list[BranchResult]]:
        """对 layer_branches 里成功的 branch 派 width 个 sub-branch 做 refinement.

        PTD tree-shape 的 layer2 — sub-branch 复用父 family_id (不重派),
        看到父 hypothesis (祖先 π(v)). 返回 {parent_agent_id: [sub-branch results]}.

        D-slice S3: 逐 parent 是一个细 slice. 派某 parent 的 sub-branch **前**查一次
        剩余预算, 不足 slice_min_s 就停止派新 parent (已派的照常收, 不 kill) ——
        避免"已经没预算还继续摊大扇出".
        """
        eligible = [b for b in layer_branches if b.success and b.hypothesis]

        coros = []
        parent_map: list[str] = []  # 每 coro 对应的 parent_agent_id
        for parent in eligible:
            # D-slice S3 逐 parent 预算门: 剩余不足就不再派新 parent.
            if not self._slice_affordable(budget_remaining_fn, slice_min_s):
                self._slice_skip(
                    trace_fn, f"layer2 parent {parent.agent_id}", slice_min_s,
                )
                break
            for _ in range(width):
                coros.append(self._run_single_branch(
                    family_id=parent.family_id,
                    bundle=bundle,
                    task=task,
                    agent_factory=agent_factory,
                    round_idx=round_idx,
                    parent_hypothesis=parent.hypothesis,
                    parent_agent_id=parent.agent_id,
                    model_family=parent.model_family,
                    wave_mode=wave_mode,
                    budget_remaining_fn=budget_remaining_fn,
                ))
                parent_map.append(parent.agent_id)

        if not coros:
            return {}

        _t_l2 = time.monotonic()
        raw_results = await asyncio.gather(*coros, return_exceptions=True)
        # D-slice 自校准: layer2 实测挂钟 = S4 打分 slice 的成本估计 (下限 slice_min_s).
        _need_l2 = max(slice_min_s, time.monotonic() - _t_l2)

        layer2: dict[str, list[BranchResult]] = {}
        for parent_id, raw in zip(parent_map, raw_results):
            if isinstance(raw, Exception):
                # 不应发生, _run_single_branch 内部已 catch. 跳过, 不入 layer2
                continue
            layer2.setdefault(parent_id, []).append(raw)
        # prune 前先注入 rollout value (value_fn=None 时全 None, 行为不变);
        # D-slice S4: 逐 sub-branch 预算门在 _assign_values 内.
        for _subs in layer2.values():
            await self._assign_values(
                _subs, value_fn, budget_remaining_fn, _need_l2,
            )
        return layer2

    def _assign_families(self, n: int) -> list[str]:
        """给 n 个 branch 分配族. 优先冷门族, 避开已阻塞族, 尽量分散.

        ponytail: 简单贪心 + 本地计数去重, 不预注册避免失败时清理.
        同一轮内尽量分到不同族, 否则 3 个 branch 全挤 dft-direct 失去隔离意义.
        升级路径: 加权采样, 让热族也有少量 agent 做对照.
        """
        assignments: list[str] = []
        chosen: dict[str, int] = {}  # 本轮已分配计数, 让后续选择考虑本轮分布
        for _ in range(n):
            sug = self._registry.suggest_redirect()
            if sug is not None and chosen.get(sug.target_family, 0) == 0:
                target = sug.target_family
            else:
                active = [f for f in self._registry.all() if not f.is_blocked]
                if not active:
                    target = "dft-direct"  # 全阻塞时的兜底
                else:
                    total = max(self._registry.total_agents(), 1)
                    # 综合: 本轮已分配 + 全局 member 比例, 取最小
                    target = min(active, key=lambda f: (
                        chosen.get(f.id, 0) + f.member_count(total)
                    )).id
            assignments.append(target)
            chosen[target] = chosen.get(target, 0) + 1
        return assignments

    def _assign_model_families(
        self, n: int, available: list[str] | None,
    ) -> list[str]:
        """P4: 给 n 个 branch 分配 model family, 尽量分散.

        跟 _assign_families 同范式 (贪心 + 去重), 但管 model family 而非 method family.
        available 不足 n 时循环复用 (保证每个 branch 有分配).
        None / 空列表时返 n 个空串 (向后兼容).

        ponytail: 不接 ModelRouter (避免循环 import), 调用方传可用列表.
        ceiling: 纯轮询分配, 不考虑 model 能力差异. 升级: 按 task 类型选 model.
        """
        if not available:
            return [""] * n
        n_avail = len(available)
        return [available[i % n_avail] for i in range(n)]

    async def _run_single_branch(
        self,
        family_id: str,
        bundle: ContextBundle,
        task: str,
        agent_factory: Any,
        round_idx: int,
        parent_hypothesis: str = "",
        parent_agent_id: str = "",
        model_family: str = "",
        wave_mode: str = "push",
        budget_remaining_fn: Callable[[], float | None] | None = None,
    ) -> BranchResult:
        """起单个 Subagent, 注入隔离后的 context + family 引导.

        parent_hypothesis 非空时 (layer2 sub-branch), 加进 enhanced_task
        让 sub-branch 看到祖先 π(v) — PTD tree 的偏序祖先链机制.

        P4: wave_mode="verify" 时 enhanced_task 加验证引导 (而非生成引导).
        model_family 非 "" 时记录到 BranchResult 供调用方调度.

        D-slice: 起 Subagent **前**把"此刻"的剩余挂钟预算刷进 `remaining_budget_s`
        contextvar (随 task 传播给子智能体的 streaming) — 子智能体据此自限降级空闲
        阈值, 而不是被父级硬砍. budget_remaining_fn=None 时不写, 保持旧行为.
        """
        # D-slice: 刷新预算 contextvar (尽力, 失败不影响 branch).
        if budget_remaining_fn is not None:
            try:
                from huginn.agent.streaming import remaining_budget_s as _rb_s

                _rb_s.set(budget_remaining_fn())
            except Exception:  # 防御: streaming 不可用 / 查询失败 → 不刷
                logger.debug("branch budget contextvar refresh failed", exc_info=True)

        ctx = isolate(bundle, role="exploration")
        family = self._registry.by_id(family_id)
        family_essence = family.essence if family else ""

        # enhanced_task 把 family essence + 可见族 id 列表加进去, 引导 Subagent
        enhanced_task = (
            f"{task}\n\n"
            f"[Method family assigned: {family_id}]\n"
            f"[Family essence: {family_essence}]\n"
        )
        # PTD tree: layer2 sub-branch 看到父 hypothesis (祖先 π(v))
        if parent_hypothesis:
            enhanced_task += f"[Parent hypothesis: {parent_hypothesis}]\n"
        # P4: wave_mode="verify" 时改引导为验证 (而非生成), 避免"自己生成自己验证"
        if wave_mode == "verify":
            enhanced_task += (
                "[Wave mode: VERIFY — do NOT generate new hypotheses. "
                "Instead, critically evaluate the hypotheses above. "
                "Look for counterexamples, logical gaps, or unsupported claims. "
                "Report whether each holds or fails, with evidence.]\n"
            )
        enhanced_task += (
            f"[Visible method families for self-categorization: "
            f"{ctx.get('_method_family_ids', [])}]\n"
            f"Approach this problem from the {family_id} perspective. "
            f"Do not assume other families' progress is visible to you."
        )

        agent_id = f"branch_{family_id}_{uuid.uuid4().hex[:8]}"
        try:
            result: SubagentResult = await self._dispatch.dispatch(
                "explore",
                enhanced_task,
                context={"agent_factory": agent_factory},
            )
        except Exception as exc:
            logger.debug("branch %s dispatch failed: %s", family_id, exc)
            return BranchResult(
                family_id=family_id,
                agent_id=agent_id,
                hypothesis=self._fallback_hypothesis(family_id),
                success=False,
                error=f"dispatch error: {exc}",
                round_idx=round_idx,
                parent_agent_id=parent_agent_id,
                model_family=model_family,
            )

        if not result.success:
            # Subagent 自身失败 (unknown spec / factory 缺失等) — 降级模板
            return BranchResult(
                family_id=family_id,
                agent_id=agent_id,
                hypothesis=self._fallback_hypothesis(family_id),
                success=False,
                error=result.error,
                round_idx=round_idx,
                parent_agent_id=parent_agent_id,
                model_family=model_family,
            )

        # 成功 — 注册到 registry, 让后续轮的 suggest_redirect 看到分布
        self._registry.register_agent(family_id, agent_id)
        return BranchResult(
            family_id=family_id,
            agent_id=agent_id,
            hypothesis=result.summary,
            full_output=result.full_output,
            success=True,
            tokens_used=result.tokens_used,
            round_idx=round_idx,
            parent_agent_id=parent_agent_id,
            model_family=model_family,
        )

    def _fallback_hypothesis(self, family_id: str) -> str:
        """Subagent 失败时降级到 family.essence 模板."""
        fam = self._registry.by_id(family_id)
        if fam is None:
            return f"[fallback] {family_id} branch failed, no hypothesis"
        return f"[fallback] apply {family_id}: {fam.essence}"

    def _check_convergence(
        self,
        round_idx: int,
        total_rounds: int,
        results: list[BranchResult],
    ) -> None:
        """反完成审计: 未达标时标记过热族, 下一轮强制 redirect.

        ponytail: 只标过热族, 不阻断 run_round 本身 (调用方决定是否继续).
        升级路径: 返回 EffortStatus 让调用方做决策.
        """
        successful = [r for r in results if r.success and r.hypothesis]
        if not successful:
            return  # 全失败, 没东西可审计

        families_explored = len({r.family_id for r in successful})
        # live_components: 不同 hypothesis 的数量 (粗估存活连通分量)
        live_components = len({r.hypothesis for r in successful})

        status = self._detector.check(
            iteration=round_idx,
            families_explored=families_explored,
            live_components=live_components,
            total_iterations=total_rounds,
        )
        blocked, reason = self._detector.should_block_return(status)
        if not blocked:
            return

        # 找当前最热的活跃族标 blocked, 让下一轮 suggest_redirect 绕开它
        total_agents = self._registry.total_agents()
        if total_agents == 0:
            return
        active = [f for f in self._registry.all() if not f.is_blocked]
        if not active:
            return
        hottest = max(active, key=lambda f: f.member_count(total_agents))
        self._registry.mark_blocked(
            hottest.id,
            f"convergence pressure at round {round_idx}: {reason}",
        )
        logger.info(
            "branch_incubator: marked %s blocked (pressure %.0f%%)",
            hottest.id, hottest.member_count(total_agents) * 100,
        )


# ── 自检 ─────────────────────────────────────────────────────────
# ponytail: 非平凡逻辑留 runnable check. Subagent 用 mock dispatch, 不调真 LLM.


class _MockSubagentDispatch:
    """测试用 mock — 不真起 Subagent, 返回固定 summary."""

    def __init__(
        self,
        summary_template: str = "hypothesis from {family}",
        fail_families: set[str] | None = None,
        fail_when_parent_hypothesis: bool = False,
    ) -> None:
        self._summary_template = summary_template
        self._fail_families = fail_families or set()
        # layer2 sub-branch 的 task 含 [Parent hypothesis: ...], 用这个开关
        # 让 layer2 全失败 (场景 10: fallback layer1)
        self._fail_when_parent_hypothesis = fail_when_parent_hypothesis
        # 记录调用, 让测试断言 task 内容含 family 引导
        self.calls: list[tuple[str, str, dict]] = []

    async def dispatch(
        self,
        spec_name: str,
        task: str,
        context: dict | None = None,
        on_state: Any = None,
    ) -> SubagentResult:
        self.calls.append((spec_name, task, context or {}))
        # 从 task 里解析 family_id (因为 _run_single_branch 把它塞进 enhanced_task)
        import re
        m = re.search(r"\[Method family assigned: ([\w-]+)\]", task)
        family_id = m.group(1) if m else "unknown"
        if family_id in self._fail_families:
            return SubagentResult(
                summary="", full_output="",
                success=False,
                error=f"mock failure for {family_id}",
                spec_name=spec_name,
            )
        if self._fail_when_parent_hypothesis and "[Parent hypothesis:" in task:
            return SubagentResult(
                summary="", full_output="",
                success=False,
                error="mock layer2 failure",
                spec_name=spec_name,
            )
        return SubagentResult(
            summary=self._summary_template.format(family=family_id),
            full_output=f"full output for {family_id}",
            success=True,
            spec_name=spec_name,
        )


def _selfcheck() -> None:
    import asyncio

    # 1. _assign_families: 冷门优先, 避开阻塞族
    inc = BranchIncubator()
    assignments = inc._assign_families(3)
    assert len(assignments) == 3, f"应分 3 个族, got {assignments}"
    assert all(isinstance(a, str) for a in assignments)
    # 第一个应来自 suggest_redirect 的默认 (dft-direct, 无 agent 时)
    assert assignments[0] == "dft-direct", assignments

    # 2. 阻塞族不参与分配
    inc._registry.mark_blocked("dft-direct", "test block")
    assignments2 = inc._assign_families(3)
    assert "dft-direct" not in assignments2, "阻塞族不应被分配"

    # 3. run_round: mock dispatch, 3 个 branch 都成功
    # 保留本地 mock 引用, 避免访问 inc._dispatch 私有属性 (消除 type: ignore)
    mock = _MockSubagentDispatch()
    inc2 = BranchIncubator(dispatch=mock)
    results = asyncio.run(inc2.run_round(
        task="test task",
        agent_factory=object(),  # mock dispatch 不用真 factory
        n_branches=3,
        round_idx=0,
        total_rounds=10,
    ))
    assert len(results) == 3, f"应 3 个结果, got {len(results)}"
    assert all(r.success for r in results), [r.error for r in results]
    # 每个 result 应有 hypothesis
    assert all(r.hypothesis for r in results)
    # agent_id 应含 family_id
    assert all(r.family_id in r.agent_id for r in results)
    # registry 应登记了 3 个 agent
    assert inc2._registry.total_agents() == 3

    # 4. mock dispatch 收到的 task 含 family 引导 + 可见族列表
    assert len(mock.calls) == 3
    spec_name, task_content, _ = mock.calls[0]
    assert spec_name == "explore"
    assert "[Method family assigned:" in task_content
    assert "[Family essence:" in task_content
    assert "_method_family_ids" not in task_content  # 这是 ctx 的 key, 不应直接进 task
    # task 含可见族列表 (exploratory 放宽后)
    assert "Visible method families" in task_content

    # 5. 部分失败: 1 个 family 失败, 降级到 fallback hypothesis
    inc3 = BranchIncubator(dispatch=_MockSubagentDispatch(
        fail_families={"gaussian-process"},
    ))
    # 强制分配到 gaussian-process
    inc3._registry.mark_blocked("dft-direct", "force redirect")
    results3 = asyncio.run(inc3.run_round(
        task="test",
        agent_factory=object(),
        n_branches=3,
        round_idx=0,
    ))
    # 至少有一个失败 + 降级
    failed = [r for r in results3 if not r.success]
    assert failed, "应有失败的 branch"
    assert all(r.hypothesis.startswith("[fallback]") for r in failed), (
        "失败应降级到 fallback hypothesis"
    )

    # 6. 反完成审计: round_idx=0 + families_explored 少 → 标过热族
    # 第 0 轮 3 个 branch, 如果都分到不同族 families_explored=3 ≥ min=3 不触发
    # 强制触发: n_branches=1, round_idx=0, total_rounds=10
    inc4 = BranchIncubator(dispatch=_MockSubagentDispatch())
    # 预先注册一堆 agent 到 dft-direct 让它过热
    for i in range(5):
        inc4._registry.register_agent("dft-direct", f"pre-{i}")
    asyncio.run(inc4.run_round(
        task="test",
        agent_factory=object(),
        n_branches=1,
        round_idx=0,
        total_rounds=10,
    ))
    # round_idx=0 < min_iterations=3 → 触发 _check_convergence 标过热
    # 但只有 1 个 active 族时不标 (无替代), 检查没崩即可
    # 强制构造过热场景: 多个活跃族, 1 个独大
    inc5 = BranchIncubator(dispatch=_MockSubagentDispatch())
    inc5._registry.mark_blocked("calphad-thermo", "skip")
    inc5._registry.mark_blocked("phase-field", "skip")
    inc5._registry.mark_blocked("bourbaki-structure", "skip")
    inc5._registry.mark_blocked("extreme-argument", "skip")
    inc5._registry.mark_blocked("computational-check", "skip")
    # dft-direct 占绝对多数, ml-potential/symbolic-regression/gaussian-process 冷门
    for i in range(6):
        inc5._registry.register_agent("dft-direct", f"hot-{i}")
    asyncio.run(inc5.run_round(
        task="test",
        agent_factory=object(),
        n_branches=1,
        round_idx=0,  # 早期, min_live_components=4
        total_rounds=10,
    ))
    # dft-direct 应被 mark_blocked (过热 + 触发反完成)
    dft = inc5._registry.by_id("dft-direct")
    assert dft is not None and dft.is_blocked, (
        f"dft-direct 应被标 blocked (过热), got blocked={dft.is_blocked if dft else None}"
    )

    # 7. BranchResult.to_dict 序列化 (含 parent_agent_id)
    r = BranchResult(
        family_id="x", agent_id="y", hypothesis="z", round_idx=5,
        parent_agent_id="parent_y",
    )
    d = r.to_dict()
    assert d["family_id"] == "x"
    assert d["round_idx"] == 5
    assert d["parent_agent_id"] == "parent_y"
    assert d["value"] is None, "默认 value 应为 None"
    # 默认 parent_agent_id 为空
    r2 = BranchResult(family_id="a", agent_id="b", hypothesis="c")
    assert r2.parent_agent_id == ""
    assert r2.to_dict()["parent_agent_id"] == ""
    assert BranchResult(
        family_id="a", agent_id="b", hypothesis="c", value=0.7,
    ).to_dict()["value"] == 0.7

    # 8. depth=2 正常路径: 3 layer1 + 2×3 layer2 = 9 dispatch 调用
    mock_tree = _MockSubagentDispatch()
    inc_tree = BranchIncubator(dispatch=mock_tree)
    results_tree = asyncio.run(inc_tree.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        depth=2, width=2,
    ))
    assert len(results_tree) == 3, f"返回长度应 = n_branches, got {len(results_tree)}"
    # layer1 全成功 → 每 parent 派 2 sub-branch → 总 dispatch = 3 + 6 = 9
    assert len(mock_tree.calls) == 9, (
        f"应 9 dispatch (3 layer1 + 6 layer2), got {len(mock_tree.calls)}"
    )
    # 返回的应是 layer2 winner (success + parent_agent_id 非空)
    assert all(r.success for r in results_tree), [r.error for r in results_tree]
    assert all(r.parent_agent_id for r in results_tree), (
        "depth=2 正常路径返回应是 layer2 winner, parent_agent_id 非空"
    )
    # layer1 的 task 不含 [Parent hypothesis], layer2 的含
    layer1_calls = [c for c in mock_tree.calls if "[Parent hypothesis:" not in c[1]]
    layer2_calls = [c for c in mock_tree.calls if "[Parent hypothesis:" in c[1]]
    assert len(layer1_calls) == 3
    assert len(layer2_calls) == 6

    # 9. layer1 部分失败: dft-direct fail → layer2 只对 2 个成功 parent 派
    # 总 dispatch = 3 layer1 + 2×2 layer2 = 7
    mock_partial = _MockSubagentDispatch(
        fail_families={"dft-direct"},
    )
    inc_partial = BranchIncubator(dispatch=mock_partial)
    results_partial = asyncio.run(inc_partial.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        depth=2, width=2,
    ))
    assert len(results_partial) == 3
    # layer1: 3 (dft-direct fail, ml-potential ok, symbolic-regression ok)
    # layer2: 2 sub-branch × 2 success parent = 4
    # 总 = 3 + 4 = 7
    assert len(mock_partial.calls) == 7, (
        f"应 7 dispatch (3 layer1 + 4 layer2), got {len(mock_partial.calls)}"
    )
    # dft-direct 的返回应是 layer1 (fail, parent_agent_id 空)
    dft_result = [r for r in results_partial if r.family_id == "dft-direct"]
    assert dft_result and not dft_result[0].success, "dft-direct 应失败"
    assert dft_result[0].parent_agent_id == "", "layer1 失败 branch parent_agent_id 应空"
    # ml-potential / symbolic-regression 的返回应是 layer2 winner (success, parent_agent_id 非空)
    success_results = [r for r in results_partial if r.success]
    assert len(success_results) == 2
    assert all(r.parent_agent_id for r in success_results)

    # 10. layer2 全失败 fallback layer1
    # fail_when_parent_hypothesis=True: layer1 全成功, layer2 全失败
    mock_l2fail = _MockSubagentDispatch(
        fail_when_parent_hypothesis=True,
    )
    inc_l2fail = BranchIncubator(dispatch=mock_l2fail)
    results_l2fail = asyncio.run(inc_l2fail.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        depth=2, width=2,
    ))
    assert len(results_l2fail) == 3
    # 全 fallback 到 layer1 (success, parent_agent_id 空)
    assert all(r.success for r in results_l2fail), "fallback layer1 应成功"
    assert all(r.parent_agent_id == "" for r in results_l2fail), (
        "fallback layer1 应 parent_agent_id 空"
    )
    # 3 layer1 + 6 layer2 (layer2 全失败但仍 dispatch)
    assert len(mock_l2fail.calls) == 9

    # 11. parent_agent_id: layer1 为空, layer2 = layer1 父 agent_id
    mock_pa = _MockSubagentDispatch()
    inc_pa = BranchIncubator(dispatch=mock_pa)
    results_pa = asyncio.run(inc_pa.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        depth=2, width=2,
    ))
    # 所有返回的都是 layer2 winner, parent_agent_id 非空
    [c for c in mock_pa.calls if "[Parent hypothesis:" not in c[1]]
    # layer1 dispatch 的 context 没法直接拿 agent_id, 但 winner.parent_agent_id
    # 应是 layer1 branch 的 agent_id (格式 branch_<family>_<8hex>)
    for winner in results_pa:
        assert winner.parent_agent_id.startswith("branch_"), (
            f"layer2 winner parent_agent_id 应是 layer1 agent_id, got {winner.parent_agent_id}"
        )
        # layer2 winner 自己的 agent_id 也是 branch_<family>_<8hex>
        assert winner.agent_id.startswith("branch_")
        # layer2 winner 的 family_id 应跟 parent 一致 (layer2 复用父 family)
        # 找 parent: layer1 dispatch 里 family 跟 winner.family_id 相同的
        # (这里只验证格式, 具体匹配在场景 12 验)

    # 12. enhanced_task 含 [Parent hypothesis: ...] 当 parent_hypothesis 非空
    mock_ph = _MockSubagentDispatch()
    inc_ph = BranchIncubator(dispatch=mock_ph)
    asyncio.run(inc_ph.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        depth=2, width=2,
    ))
    layer1_calls_ph = [c for c in mock_ph.calls if "[Parent hypothesis:" not in c[1]]
    layer2_calls_ph = [c for c in mock_ph.calls if "[Parent hypothesis:" in c[1]]
    assert len(layer1_calls_ph) == 3, "layer1 task 不应含 [Parent hypothesis]"
    assert len(layer2_calls_ph) == 6, "应有 6 个 layer2 task 含 [Parent hypothesis]"
    # layer2 task 含 family + essence + Parent hypothesis + Visible
    for _spec_name, task_content, _ in layer2_calls_ph:
        assert spec_name == "explore"
        assert "[Method family assigned:" in task_content
        assert "[Family essence:" in task_content
        assert "[Parent hypothesis:" in task_content
        assert "Visible method families" in task_content
    # layer1 task 不含 [Parent hypothesis]
    for _spec_name, task_content, _ in layer1_calls_ph:
        assert "[Parent hypothesis:" not in task_content

    # 13. P4: wave_mode="verify" 时 enhanced_task 含 VERIFY 引导
    mock_wv = _MockSubagentDispatch()
    inc_wv = BranchIncubator(dispatch=mock_wv)
    asyncio.run(inc_wv.run_round(
        task="test hypothesis to verify", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        wave_mode="verify",
    ))
    assert len(mock_wv.calls) == 3
    for _spec_name, task_content, _ in mock_wv.calls:
        assert "[Wave mode: VERIFY" in task_content, \
            "verify 模式 enhanced_task 应含 VERIFY 引导"
    print("13. P4 wave_mode=verify 注入 VERIFY 引导 OK")

    # 14. P4: wave_mode="push" (默认) 不含 VERIFY 引导 (向后兼容)
    mock_wp = _MockSubagentDispatch()
    inc_wp = BranchIncubator(dispatch=mock_wp)
    asyncio.run(inc_wp.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
    ))
    for _spec_name, task_content, _ in mock_wp.calls:
        assert "[Wave mode: VERIFY" not in task_content, \
            "push 模式不应含 VERIFY 引导"
    print("14. P4 wave_mode=push (默认) 不含 VERIFY 引导 OK")

    # 15. P4: model_families 分配 — 3 个 branch 分到不同 model_family
    inc_mf = BranchIncubator(dispatch=_MockSubagentDispatch())
    results_mf = asyncio.run(inc_mf.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        model_families=["openai", "anthropic", "deepseek"],
    ))
    assert len(results_mf) == 3
    _mfs = [r.model_family for r in results_mf]
    assert _mfs == ["openai", "anthropic", "deepseek"], \
        f"3 个 branch 应分到 3 个不同 model_family, got {_mfs}"
    print("15. P4 model_families 分配 (3 branch → 3 不同 model_family) OK")

    # 16. P4: model_families=None (默认) → model_family 空串 (向后兼容)
    inc_nf = BranchIncubator(dispatch=_MockSubagentDispatch())
    results_nf = asyncio.run(inc_nf.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
    ))
    assert all(r.model_family == "" for r in results_nf), \
        "model_families=None 时 model_family 应为空串"
    print("16. P4 model_families=None 向后兼容 (model_family 空串) OK")

    # 17. P4: model_families 不足时循环复用 (2 family → 3 branch, 有重复)
    inc_lr = BranchIncubator(dispatch=_MockSubagentDispatch())
    results_lr = asyncio.run(inc_lr.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        model_families=["openai", "anthropic"],
    ))
    _mfs_lr = [r.model_family for r in results_lr]
    assert _mfs_lr == ["openai", "anthropic", "openai"], \
        f"2 family → 3 branch 应循环复用, got {_mfs_lr}"
    print("17. P4 model_families 不足循环复用 OK")

    # 18. P4: wave_mode + model_families 组合 (verify wave + model 多样性)
    mock_combo = _MockSubagentDispatch()
    inc_combo = BranchIncubator(dispatch=mock_combo)
    results_combo = asyncio.run(inc_combo.run_round(
        task="verify this", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        wave_mode="verify",
        model_families=["openai", "anthropic", "deepseek"],
    ))
    # 双重检查: VERIFY 引导 + model_family 分配
    for _spec_name, task_content, _ in mock_combo.calls:
        assert "[Wave mode: VERIFY" in task_content
    assert [r.model_family for r in results_combo] == ["openai", "anthropic", "deepseek"]
    print("18. P4 wave_mode=verify + model_families 组合 OK")

    # 19. _pick_winner: 有 value 按 value 最大 (忽略 tokens); 全无 value 回退 min tokens
    _s_low = BranchResult(family_id="f", agent_id="a1", hypothesis="h",
                          tokens_used=1, value=0.2)
    _s_high = BranchResult(family_id="f", agent_id="a2", hypothesis="h",
                           tokens_used=9, value=0.9)
    assert BranchIncubator._pick_winner([_s_low, _s_high]) is _s_high, \
        "有 value 时应按 value 最大选 (不看 tokens_used)"
    _s_a = BranchResult(family_id="f", agent_id="a1", hypothesis="h", tokens_used=5)
    _s_b = BranchResult(family_id="f", agent_id="a2", hypothesis="h", tokens_used=2)
    assert BranchIncubator._pick_winner([_s_a, _s_b]) is _s_b, \
        "无 value 时应回退 min(tokens_used)"
    print("19. _pick_winner value-priority + token fallback OK")

    # 20. run_round value_fn: 注入 rollout value, prune 按 value 选 (不同于 token 选)
    class _ValMock(_MockSubagentDispatch):
        """tokens_used 递增, 便于区分 value 驱动 vs token 驱动选主."""

        def __init__(self) -> None:
            super().__init__()
            self._n = 0

        async def dispatch(self, spec_name, task, context=None, on_state=None):
            r = await super().dispatch(spec_name, task, context, on_state)
            self._n += 1
            r.tokens_used = self._n
            return r

    _res_val = asyncio.run(BranchIncubator(dispatch=_ValMock()).run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10, depth=2, width=2,
        value_fn=lambda r: float(r.tokens_used),  # 越多 token value 越高
    ))
    assert len(_res_val) == 3
    assert all(r.value is not None for r in _res_val), "value_fn 应写入 value"

    _res_null = asyncio.run(BranchIncubator(dispatch=_ValMock()).run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10, depth=2, width=2,
    ))
    assert all(r.value is None for r in _res_null), "无 value_fn 应保持 value=None"

    _toks_val = sorted(r.tokens_used for r in _res_val)
    _toks_null = sorted(r.tokens_used for r in _res_null)
    assert _toks_val != _toks_null, (
        f"value 驱动与 token 驱动应选不同 winner: {_toks_val} vs {_toks_null}"
    )
    assert all(v > n for v, n in zip(_toks_val, _toks_null)), (
        f"max-value winner 应比 min-token winner 用更多 token: {_toks_val} vs {_toks_null}"
    )
    print("20. run_round value_fn rollout-value pruning OK")

    # 21. async value_fn: 协程自动 await (如 step_verifier 的 PRM 打分)
    async def _async_val(r: BranchResult) -> float | None:
        await asyncio.sleep(0)
        return 1.0 if r.tokens_used % 2 == 0 else 0.0

    _res_async = asyncio.run(BranchIncubator(dispatch=_ValMock()).run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10, depth=2, width=2,
        value_fn=_async_val,
    ))
    assert all(r.value in (0.0, 1.0) for r in _res_async), (
        f"async value_fn 应被 await 并写入 value: "
        f"{[r.value for r in _res_async]}"
    )
    print("21. run_round async value_fn (PRM coroutine) OK")

    # 22. D-slice _slice_affordable: None / None-return / 高 / 低 / 异常 五态
    assert BranchIncubator._slice_affordable(None, 60.0) is True, \
        "budget_remaining_fn=None (非长程) 应视为不受约束"
    assert BranchIncubator._slice_affordable(lambda: None, 60.0) is True, \
        "无 goal (返回 None) 应视为不受约束"
    assert BranchIncubator._slice_affordable(lambda: 100.0, 60.0) is True, \
        "剩余 100s ≥ 60s 应负担得起"
    assert BranchIncubator._slice_affordable(lambda: 10.0, 60.0) is False, \
        "剩余 10s < 60s 应负担不起"

    def _boom() -> float:
        raise RuntimeError("budget query failed")

    assert BranchIncubator._slice_affordable(_boom, 60.0) is True, \
        "预算查询异常应 fail-open (不误判耗尽)"
    print("22. D-slice _slice_affordable 五态 OK")

    # 23. D-slice 预算不足: depth=2 也跳过 layer2, 只跑 layer1 (3 dispatch)
    mock_low = _MockSubagentDispatch()
    results_low = asyncio.run(BranchIncubator(dispatch=mock_low).run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10, depth=2, width=2,
        budget_remaining_fn=lambda: 5.0, slice_min_s=60.0,
    ))
    assert len(results_low) == 3, f"应返回 3 条 layer1, got {len(results_low)}"
    assert len(mock_low.calls) == 3, (
        f"预算不足应跳过 layer2 (只 3 dispatch), got {len(mock_low.calls)}"
    )
    assert all(r.parent_agent_id == "" for r in results_low), \
        "预算不足返回的应是 layer1 (parent_agent_id 空)"
    assert all(r.hypothesis for r in results_low), "layer1 结果仍应可用"
    print("23. D-slice 预算不足跳过 layer2 (3 dispatch) OK")

    # 24. D-slice 预算充足 / 无预算约束: 行为与旧版一致 (9 dispatch)
    mock_rich = _MockSubagentDispatch()
    _res_rich = asyncio.run(BranchIncubator(dispatch=mock_rich).run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10, depth=2, width=2,
        budget_remaining_fn=lambda: 1000.0, slice_min_s=60.0,
    ))
    assert len(mock_rich.calls) == 9, (
        f"预算充足应跑满 layer1+layer2 (9 dispatch), got {len(mock_rich.calls)}"
    )
    assert all(r.parent_agent_id for r in _res_rich), "充足预算应返回 layer2 winner"
    # budget_remaining_fn=None → 不受约束, 同样 9 dispatch (向后兼容)
    mock_none = _MockSubagentDispatch()
    asyncio.run(BranchIncubator(dispatch=mock_none).run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10, depth=2, width=2,
    ))
    assert len(mock_none.calls) == 9, "无 budget_remaining_fn 行为应不变 (9 dispatch)"
    print("24. D-slice 预算充足/无约束行为不变 (9 dispatch) OK")

    # 25. D-slice _assign_values 逐 branch 预算门: 中途预算掉线 → 后续保持 None
    _rs = [BranchResult(family_id="f", agent_id=f"a{i}", hypothesis="h")
           for i in range(4)]
    _seq = iter([100.0, 100.0, 1.0, 100.0])
    asyncio.run(BranchIncubator._assign_values(
        _rs, lambda r: 0.7, lambda: next(_seq), 60.0,
    ))
    assert _rs[0].value == 0.7 and _rs[1].value == 0.7, "预算足时应打分"
    assert _rs[2].value is None and _rs[3].value is None, (
        "预算掉线后应停止打分, 后续 value 保持 None"
    )
    print("25. D-slice _assign_values 逐 branch 预算门 OK")

    # 26. D-slice trace_fn: 跳过 slice 时上报 branch_slice_skip (控制面可观测)
    _traces: list[tuple[str, str, str]] = []

    def _tf(name: str, evidence: str, action: str) -> None:
        _traces.append((name, evidence, action))

    asyncio.run(BranchIncubator(dispatch=_MockSubagentDispatch()).run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10, depth=2, width=2,
        budget_remaining_fn=lambda: 5.0, slice_min_s=60.0, trace_fn=_tf,
    ))
    assert any(n == "branch_slice_skip" for n, _, _ in _traces), (
        f"应上报 branch_slice_skip trace, got {_traces}"
    )
    print("26. D-slice trace_fn branch_slice_skip 上报 OK")

    # 27. D-slice 自校准: layer1 实测成本 > 剩余预算 → 跳过同形状的 layer2.
    #     场景复刻 run72: layer1 烧掉大部分预算, 只剩一点, 不足以再跑一个 layer2.
    #     用"模拟花费时钟"记账 (不依赖真 LLM 计时, 也不依赖 streaming 是否可导入).
    _clock = {"t": 0.0}
    _total = 0.25  # 总预算 0.25s; layer1 花 0.2s → 只剩 0.05s, 不够再跑 0.2s 的 layer2

    def _spend_budget() -> float:
        return _total - _clock["t"]

    class _SlowL1Mock(_MockSubagentDispatch):
        """layer1 (无 Parent hypothesis) 慢且记账, layer2 快 — 制造"上一片昂贵"."""

        def __init__(self, l1_cost: float) -> None:
            super().__init__()
            self._l1_cost = l1_cost

        async def dispatch(self, spec_name, task, context=None, on_state=None):
            if "[Parent hypothesis:" not in task:
                await asyncio.sleep(self._l1_cost)  # 真挂钟 → _layer1_cost 实测
                _clock["t"] = self._l1_cost         # 记账 (并发分支同值, 幂等)
            return await super().dispatch(spec_name, task, context, on_state)

    _slow = _SlowL1Mock(l1_cost=0.2)
    _res_cal = asyncio.run(BranchIncubator(dispatch=_slow).run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10, depth=2, width=2,
        budget_remaining_fn=_spend_budget, slice_min_s=0.01,
    ))
    assert len(_slow.calls) == 3, (
        f"自校准应据 layer1 实测 (~0.2s) 跳过 layer2 (剩 ~0.05s), "
        f"got {len(_slow.calls)} dispatch"
    )
    assert len(_res_cal) == 3 and all(r.parent_agent_id == "" for r in _res_cal)
    # 对照: layer1 飞快 (实测≈0, 不记账) → 门槛回落到固定下限 0.01s → 跑满 9
    _clock["t"] = 0.0
    _fast = _MockSubagentDispatch()
    asyncio.run(BranchIncubator(dispatch=_fast).run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10, depth=2, width=2,
        budget_remaining_fn=_spend_budget, slice_min_s=0.01,
    ))
    assert len(_fast.calls) == 9, (
        f"layer1 便宜时同一预算应够跑 layer2 (9 dispatch), got {len(_fast.calls)}"
    )
    print("27. D-slice 自校准门槛 (layer1 实测 → 决定 layer2 是否负担得起) OK")

    print("branch_incubator selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
