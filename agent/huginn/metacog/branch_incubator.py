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
import logging
import uuid
from dataclasses import dataclass, field
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
    ) -> list[BranchResult]:
        """跑一轮隔离探索.

        agent_factory 必传, 跟 SubagentDispatch.dispatch 一致.
        返回 N 个 BranchResult, 失败的也有记录 (success=False).

        depth=1: flat asyncio.gather N (Part 1 兼容).
        depth=2: PTD tree-shape — layer1 N flat → 每 layer1 成功 branch 派
          width 个 sub-branch → 每 parent prune top-1 (tokens_used 最小 + success).
          layer2 全失败的 parent 回退到 layer1. 返回长度 = n_branches.
        """
        bundle = ContextBundle(
            global_math_background=math_background,
            task_definition=task,
            current_preferred_hypothesis=None,  # exploratory → 自动放宽隔离
            researcher_intuition=researcher_intuition,
            method_family_registry=self._registry.to_dict(),
        )

        family_assignments = self._assign_families(n_branches)
        # Layer 1: flat asyncio.gather (depth=1 和 depth=2 都跑这步)
        coros = [
            self._run_single_branch(fam, bundle, task, agent_factory, round_idx)
            for fam in family_assignments
        ]
        raw_results = await asyncio.gather(*coros, return_exceptions=True)

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

        if depth < 2:
            return layer1  # Part 1 flat 行为

        # Layer 2: PTD tree-shape — 每 layer1 成功 branch 派 width 个 sub-branch
        layer2_by_parent = await self._run_tree_layer(
            layer1, bundle, task, agent_factory, round_idx, width,
        )

        # Prune + Fallback: 每 parent 保留 top-1 (tokens_used 最小 + success),
        # layer1 失败 / layer2 全失败 → 回退 layer1
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
            best = min(successful_subs, key=lambda r: r.tokens_used)
            final.append(best)
        return final

    async def _run_tree_layer(
        self,
        layer_branches: list[BranchResult],
        bundle: ContextBundle,
        task: str,
        agent_factory: Any,
        round_idx: int,
        width: int,
    ) -> dict[str, list[BranchResult]]:
        """对 layer_branches 里成功的 branch 派 width 个 sub-branch 做 refinement.

        PTD tree-shape 的 layer2 — sub-branch 复用父 family_id (不重派),
        看到父 hypothesis (祖先 π(v)). 返回 {parent_agent_id: [sub-branch results]}.
        """
        eligible = [b for b in layer_branches if b.success and b.hypothesis]

        coros = []
        parent_map: list[str] = []  # 每 coro 对应的 parent_agent_id
        for parent in eligible:
            for _ in range(width):
                coros.append(self._run_single_branch(
                    family_id=parent.family_id,
                    bundle=bundle,
                    task=task,
                    agent_factory=agent_factory,
                    round_idx=round_idx,
                    parent_hypothesis=parent.hypothesis,
                    parent_agent_id=parent.agent_id,
                ))
                parent_map.append(parent.agent_id)

        if not coros:
            return {}

        raw_results = await asyncio.gather(*coros, return_exceptions=True)

        layer2: dict[str, list[BranchResult]] = {}
        for parent_id, raw in zip(parent_map, raw_results):
            if isinstance(raw, Exception):
                # 不应发生, _run_single_branch 内部已 catch. 跳过, 不入 layer2
                continue
            layer2.setdefault(parent_id, []).append(raw)
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

    async def _run_single_branch(
        self,
        family_id: str,
        bundle: ContextBundle,
        task: str,
        agent_factory: Any,
        round_idx: int,
        parent_hypothesis: str = "",
        parent_agent_id: str = "",
    ) -> BranchResult:
        """起单个 Subagent, 注入隔离后的 context + family 引导.

        parent_hypothesis 非空时 (layer2 sub-branch), 加进 enhanced_task
        让 sub-branch 看到祖先 π(v) — PTD tree 的偏序祖先链机制.
        """
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
    inc2 = BranchIncubator(dispatch=_MockSubagentDispatch())
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
    mock = inc2._dispatch  # type: ignore
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
    # 默认 parent_agent_id 为空
    r2 = BranchResult(family_id="a", agent_id="b", hypothesis="c")
    assert r2.parent_agent_id == ""
    assert r2.to_dict()["parent_agent_id"] == ""

    # 8. depth=2 正常路径: 3 layer1 + 2×3 layer2 = 9 dispatch 调用
    inc_tree = BranchIncubator(dispatch=_MockSubagentDispatch())
    results_tree = asyncio.run(inc_tree.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        depth=2, width=2,
    ))
    assert len(results_tree) == 3, f"返回长度应 = n_branches, got {len(results_tree)}"
    # layer1 全成功 → 每 parent 派 2 sub-branch → 总 dispatch = 3 + 6 = 9
    mock_tree = inc_tree._dispatch  # type: ignore
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
    inc_partial = BranchIncubator(dispatch=_MockSubagentDispatch(
        fail_families={"dft-direct"},
    ))
    results_partial = asyncio.run(inc_partial.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        depth=2, width=2,
    ))
    assert len(results_partial) == 3
    mock_partial = inc_partial._dispatch  # type: ignore
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
    inc_l2fail = BranchIncubator(dispatch=_MockSubagentDispatch(
        fail_when_parent_hypothesis=True,
    ))
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
    mock_l2fail = inc_l2fail._dispatch  # type: ignore
    # 3 layer1 + 6 layer2 (layer2 全失败但仍 dispatch)
    assert len(mock_l2fail.calls) == 9

    # 11. parent_agent_id: layer1 为空, layer2 = layer1 父 agent_id
    inc_pa = BranchIncubator(dispatch=_MockSubagentDispatch())
    results_pa = asyncio.run(inc_pa.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        depth=2, width=2,
    ))
    # 所有返回的都是 layer2 winner, parent_agent_id 非空
    layer1_results_in_call = [c for c in inc_pa._dispatch.calls if "[Parent hypothesis:" not in c[1]]  # type: ignore
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
    inc_ph = BranchIncubator(dispatch=_MockSubagentDispatch())
    asyncio.run(inc_ph.run_round(
        task="test", agent_factory=object(),
        n_branches=3, round_idx=0, total_rounds=10,
        depth=2, width=2,
    ))
    mock_ph = inc_ph._dispatch  # type: ignore
    layer1_calls_ph = [c for c in mock_ph.calls if "[Parent hypothesis:" not in c[1]]
    layer2_calls_ph = [c for c in mock_ph.calls if "[Parent hypothesis:" in c[1]]
    assert len(layer1_calls_ph) == 3, "layer1 task 不应含 [Parent hypothesis]"
    assert len(layer2_calls_ph) == 6, "应有 6 个 layer2 task 含 [Parent hypothesis]"
    # layer2 task 含 family + essence + Parent hypothesis + Visible
    for spec_name, task_content, _ in layer2_calls_ph:
        assert spec_name == "explore"
        assert "[Method family assigned:" in task_content
        assert "[Family essence:" in task_content
        assert "[Parent hypothesis:" in task_content
        assert "Visible method families" in task_content
    # layer1 task 不含 [Parent hypothesis]
    for spec_name, task_content, _ in layer1_calls_ph:
        assert "[Parent hypothesis:" not in task_content

    print("branch_incubator selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
