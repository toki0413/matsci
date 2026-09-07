"""SupervisorStrategy — 把人机交互反馈与想法评审(debate/tournament)接进探索循环.

审计缺口补齐之二/之三:
  - HITL 反馈闭环: 此前 ask-user / 审批门是孤立通道, 没有"专家反馈回流进下一轮
    想法/实验生成"。本策略在指定迭代间隔 (``review_every``) 把"当前 Pareto 前沿 +
    拟议动作"打包成决策卡交给 ``human_review(card)`` 回调, 把返回的
    ``{drop:[...], keep:[...], guidance:"..."}`` 落地为剪枝 Action / 注入下一批
    子代假说 —— 从而专家见解直接改变想法演化, 形成闭环。
  - 想法 tournament (多智能体筛选): 通过 ``debate_evaluator(ideas) -> survivors``
    对候选存活想法做交叉评审, 未存活者被剪枝淘汰 (确定性 scorer 或 LLM critic
    均可注入)。生产管线再叠加 ``research/program`` 的 LLM 评审即为完整辩论。

设计: 装饰器 —— 包裹任意基础策略 (默认 ParetoPruningStrategy), ``evaluate()``
先算基础 Action, 再叠加 tournament 淘汰 + 人工反馈, 复用现有
``ExplorationOrchestrator`` 的落子机制, 不侵入引擎。
"""
from __future__ import annotations

import random
from typing import Any, Callable

from huginn.exploration.core import BranchStatus, ExplorationSpace
from huginn.exploration.strategies import Action, ExplorationStrategy, ParetoPruningStrategy


class SupervisorStrategy(ExplorationStrategy):
    """HITL 反馈 + 想法评审装饰器."""

    def __init__(
        self,
        wrapped: ExplorationStrategy | None = None,
        *,
        human_review: Callable[[dict], dict] | None = None,
        review_every: int = 0,
        debate_evaluator: Callable[[list[dict]], list[dict]] | None = None,
        rng: random.Random | None = None,
    ):
        self.wrapped = wrapped or ParetoPruningStrategy()
        self.human_review = human_review          # callable(card) -> {drop, keep, guidance}
        self.review_every = review_every          # 每 N 轮给一次人工评审; 0=关闭
        self.debate_evaluator = debate_evaluator  # callable(ideas) -> survivor ideas
        self.rng = rng or random.Random(0)
        self._iterations = 0
        self.review_log: list[dict[str, Any]] = []

    def name(self) -> str:
        return f"supervised({self.wrapped.name()})"

    def _front_ideas(self, space: ExplorationSpace) -> list[dict[str, Any]]:
        front = list(space.update_pareto_front() or [])
        return [
            {
                "id": bid,
                "name": space.branches[bid].name,
                "hypothesis": space.branches[bid].hypothesis,
                "objectives": dict(space.branches[bid].objectives),
            }
            for bid in front
        ]

    def evaluate(self, space: ExplorationSpace) -> list[Action]:
        self._iterations += 1
        actions = self.wrapped.evaluate(space)

        # ── 1) 想法 tournament: 交叉评审存活者, 淘汰弱者 ─────────────
        ideas = self._front_ideas(space)
        if self.debate_evaluator and ideas:
            survivors = self.debate_evaluator(ideas)
            keep = {s["name"] for s in survivors}
            for idea in ideas:
                b = space.branches[idea["id"]]
                if b.name not in keep and b.status == BranchStatus.COMPLETED:
                    actions.append(Action(
                        action_type="prune", target_branch=idea["id"],
                        reason="eliminated in idea tournament"))

        # ── 2) HITL 评审 checkpoint ─────────────────────────────────
        if (
            self.human_review
            and self.review_every > 0
            and self._iterations % self.review_every == 0
        ):
            pending = [a for a in actions if a.new_branches]
            card = {
                "iteration": self._iterations,
                "objective": space.objective,
                "pareto_front": ideas,
                "proposed_actions": len(pending),
            }
            fb = self.human_review(card) or {}
            self.review_log.append(fb)

            drop = set(fb.get("drop") or [])
            for bid in list(space.branches):
                b = space.branches[bid]
                if b.name in drop and b.status == BranchStatus.COMPLETED:
                    actions.append(Action(
                        action_type="prune", target_branch=bid,
                        reason="human feedback: drop"))

            guidance = str(fb.get("guidance") or "").strip()
            if guidance:
                # 专家见解回流: 附加到本轮拟生成的子代假说正文
                for a in actions:
                    if not a.new_branches:
                        continue
                    for nb in a.new_branches:
                        nb["hypothesis"] = (
                            f"[human guidance] {guidance}\n{nb.get('hypothesis', '')}")
        return actions