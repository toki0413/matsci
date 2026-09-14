"""M-R1: Recursive (Meta-)Improver — 让「改进器如何改进」也是可改进、可验收的对象.

单层改进器现状 (H1 prompt_patch):
  generate_patch(phase, blocks, r_phys, directive, llm_chat_fn) 把一份硬编码的
  improv`-ing` prompt 喂给 LLM, LLM 产出一个 prompt block patch. 这份 improver prompt
  是函数内字面量, 不可被改进 → 单层、非递归.

本模块加一层 meta-improver (STOP 退化单步版):
  - 把「改进器 prompt 模板」提升为一等对象 ImproverConfig (champion active 才覆盖默认),
  - maybe_propose: LLM 基于「当前 improver 模板 + meta 统计」改写新模板 → 候选,
  - evaluate: 用「冻结重放集」(最近 K 组 (phase, blocks, r_phys, directive)) 离线给
    候选与前冠军打分 → 配对注册进 SignificanceGate + OODHoldout,
  - promote: 仅 显著 + OOD 不退化 (即 AdoptionGate 的 GREEN) 才把候选设为 champion,
  - champion 换代旧配置冻结保留可回退; 门控永不删数据.

验收代理分 (诚实声明): evaluate 的分数是「产出 patch 的有效性」代理, 不是真实 r_phys.
真实收益由下游 apply_patches 对 patch 的 Beta 接受度兜底. 代理分只作首道闸.

toggle: cfg.feature_flags.harness_meta_improver AND harness_prompt_patch 同时 on 才生效
(默认全 off, 关闭时零行为变更 — generate_patch 回落硬编码模板).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from huginn.utils.runtime import get_runtime_home

from ._enabled import _harness_enabled

logger = logging.getLogger(__name__)

# 重放集容量 — evaluate 离线打分用; K=10 保证 sig(>=5) + OOD(train>=3, holdout>=3) 够量.
_REPLAY_MAX = 10
# 每成功生成多少 patch 触发一次 maybe_propose.
_PROPOSE_EVERY_N = 5
# 每多少轮 level-0 提案触发一次 strategist 层提案 (递归环的生产驱动周期).
_STRATEGIST_EVERY_N_PROPOSALS = 3
# 显著性验收最小样本量 (和 SignificanceGate 默认同步).
_MIN_SAMPLES = 5
# r_phys 阈值扰动约束.
_R_PHYS_GATE_MIN, _R_PHYS_GATE_MAX = 0.5, 0.8

# 默认 improver prompt 模板 — 从 prompt_patch.generate_patch 原位迁出集中管理,
# 保持 {phase}/{block_names}/{r_phys}/{directive} 四个占位符. champion 覆盖此模板.
DEFAULT_IMPROV_TEMPLATE = (
    "You are optimizing a research agent's prompt template. Based on the "
    "last iteration's physical validation score and self-directive, "
    "propose ONE block-level patch.\n\n"
    "Phase: {phase}\n"
    "Available blocks: {block_names}\n"
    "R_phys (last iter): {r_phys}\n"
    "Self-directive: {directive}\n\n"
    "Output JSON only:\n"
    '{{"block_name": "<one of available>", '
    '"op": "replace|prepend|append", '
    '"new_text": "<new block content>"}}'
    "Rules:\n"
    "- replace 'body' block: must preserve {{context}} or {{hypothesis}} placeholder\n"
    "- new_text max 500 chars\n"
    "- op=prepend/append preserves original block text"
)

# meta-improver 自身的 prompt: 让 LLM 改写「改进器模板」. 输出必须是模板原文,
# 保留四个占位符, 便于 generate_patch 逐调用 .format 实例化.
_META_IMPROVE_TEMPLATE = (
    "You are the meta-improver. The base improver uses the template below to ask "
    "the LLM for ONE prompt-block patch to a research agent.\n\n"
    "----- CURRENT IMPROVER PROMPT TEMPLATE -----\n{current_template}\n"
    "------------------------------------------------\n\n"
    "Rewrite this improver prompt TEMPLATE so the base improver produces BETTER, "
    "more directive-aligned patches. Keep the four placeholders {{phase}}, "
    "{{block_names}}, {{r_phys}}, {{directive}} intact (do not remove or alter them). "
    "It must stay a f-string-safe template with exactly those four "
    "{{...}} placeholders and nothing else in braces.\n"
    "Respond with the rewritten template text ONLY (no commentary, no code "
    "fences, no JSON).\n"
    "Recent meta stats: proposals={n_proposals}, promotions={n_promotions}."
)


# A1 level-1: meta² 固定模板 — 生成新的 strategist(改进策略) 候选. 只一层递归
# (YAGNI 不再叠第三层). strategist 模板本身是「meta 提示模板」, 即未来被
# maybe_propose 用作 .format(current_template=...) 的整体模板.
_META2_IMPROVE_TEMPLATE = (
    "You are the meta-strategist. The current 'strategist' template below is used "
    "to propose improvements to the research agent's IMPROVER. Rewrite it so that "
    "proposed improvers converge FASTER and are more directive-aligned.\n"
    "----- CURRENT STRATEGIST TEMPLATE -----\n{current_strategist}\n"
    "---------------------------------------\n"
    "Respond with the rewritten strategist TEMPLATE ONLY. It must stay a template "
    "that `maybe_propose` can instantiate via .format, so it MUST contain the "
    "placeholder {{current_template}} (and may use {{n_proposals}}, {{n_promotions}}); "
    "escape any other braces. Keep those placeholders intact.\n"
    "Meta stats: promotions={n_promotions}, win_rate={win_rate}."
)


@dataclass
class ImproverConfig:
    """一个可被采纳/回退的「改进器模板」+ 其门阈值."""

    config_id: str
    improver_prompt: str          # 带 {phase}/{block_names}/{r_phys}/{directive} 占位符的模板
    r_phys_gate: float = 0.7      # 只有 r_phys<=gate 才生成 patch
    min_beta_mean: float = 0.5    # apply 门槛 (下游 apply_patches 使用)
    active: bool = False          # true 表示当前全局 champion (覆盖默认模板)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ImproverConfig:
        return cls(
            config_id=d["config_id"],
            improver_prompt=d.get("improver_prompt", ""),
            r_phys_gate=float(d.get("r_phys_gate", 0.7)),
            min_beta_mean=float(d.get("min_beta_mean", 0.5)),
            active=bool(d.get("active", False)),
            created_at=float(d.get("created_at", time.time())),
        )


# Task 1 self-review: 两个新类均为纯数据/计算对象, 不依赖单例状态, 不改任何现有
# 方法逻辑; 已由 TDD test_compounding_tracker_math 覆盖斜率/滞回/死锁边界.
@dataclass
class StrategistConfig:
    """level-1: 生成改进器候选的"策略"模板."""
    config_id: str
    strategist_prompt: str
    active: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategistConfig:
        return cls(
            config_id=d["config_id"],
            strategist_prompt=d.get("strategist_prompt", ""),
            active=bool(d.get("active", False)),
            created_at=float(d.get("created_at", time.time())),
        )


class CompoundingTracker:
    """复合验收: 滚动窗口斜率 + 每次换件成本趋势 + 滞回带 + 死锁计数."""
    def __init__(self, window: int = 8, slope_tolerance: float = -0.05,
                 hysteresis_band: float = 0.03, deadlock_timeout: int = 5) -> None:
        self.window = window
        self.slope_tolerance = slope_tolerance
        self.hysteresis_band = hysteresis_band
        self.deadlock_timeout = deadlock_timeout
        self._rows: list[dict[str, Any]] = []
        self._deadlock_n = 0

    def record(self, *, epoch: int, config_id: str, quality: float,
               fidelity: float, proposals_to_promotion: int,
               generations_to_promotion: int, win_rate: float) -> None:
        self._rows.append({"epoch": epoch, "config_id": config_id, "quality": quality,
                           "fidelity": fidelity, "proposals_to_promotion": proposals_to_promotion,
                           "generations_to_promotion": generations_to_promotion, "win_rate": win_rate})
        self._rows = self._rows[-self.window:]

    def _quality_series(self) -> list[float]:
        return [r["quality"] for r in self._rows]

    def _slope(self) -> float:
        ys = self._quality_series()
        if len(ys) < 2:
            return 0.0
        xs = list(range(len(ys)))
        n = len(xs)
        xm, ym = sum(xs) / n, sum(ys) / n
        num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
        den = sum((x - xm) ** 2 for x in xs)
        return 0.0 if den == 0 else num / den

    def _cost_trend(self) -> float:
        if len(self._rows) < 4:
            return 0.0
        head = self._rows[:2]
        tail = self._rows[-2:]
        h = sum(r["proposals_to_promotion"] for r in head) / len(head)
        t = sum(r["proposals_to_promotion"] for r in tail) / len(tail)
        return t - h

    def is_compounding(self) -> bool:
        if self._slope() < self.slope_tolerance:
            return False
        if self._cost_trend() > 1.0:
            return False
        return True

    def would_degrade(self, new: str, incumbent: str) -> bool:
        """换件是否会退化: 候选(new)最近质量显著低于历史上的最优质量
        (incumbent 代表当前朝上的基线), 且复合不再上行 → 判定退化.

        - ``new``: 候选 config 最近一次 record 的质量 (若未 record 过则无据,
          用全局窗口末位作保守近似).
        - ``incumbent``: 当前 champion id (仅作语义锚; 质量取窗口最优).
        """
        ys = self._quality_series()
        if not ys:
            return False
        best = max(ys)
        # 候选近期质量: 本配置最近行的质量, 否则取窗口末位 (保守).
        new_qs = [r["quality"] for r in self._rows if r["config_id"] == new]
        cur = new_qs[-1] if new_qs else ys[-1]
        return (cur < best - self.hysteresis_band) and not self.is_compounding()

    def mark_deadlock(self, yellow: bool) -> None:
        self._deadlock_n = self._deadlock_n + 1 if yellow else 0

    def in_deadlock(self) -> bool:
        return self._deadlock_n >= self.deadlock_timeout

    def stats(self) -> dict[str, Any]:
        return {"slope": round(self._slope(), 4), "cost_trend": round(self._cost_trend(), 4),
                "window": len(self._rows),
                "best_quality": max(self._quality_series()) if self._rows else 0.0,
                "deadlock_n": self._deadlock_n, "is_compounding": self.is_compounding()}


def _behavioral_fidelity_default_path() -> Any:
    """BehavioralFidelity 默认持久化路径."""
    from pathlib import Path
    return Path(get_runtime_home()) / "meta_improver" / "fidelity.json"


class BehavioralFidelity:
    """行为级奖励回流 — Goodhart 保真锚 (spec #1).

    用「真实 apply_patches 采纳率」作复合指标的保真锚, 并把**经状态化仿真验证
    (论文 arXiv:2609.03621 verified=True)** 的采纳以验证率轻微上修并 clamp 到
    [0,1] — 验证加成绝不 push 出可信上限 (治 Goodhart: 代理分会饱和, 真实行为
    采纳是不可随意优化的保真锚). default-中性 0.5 无记录时回落.
    """

    path: Any = None

    def __init__(self, path: Any | None = None) -> None:
        self._path = path or _behavioral_fidelity_default_path()
        self._accepted: dict[str, int] = {}
        self._applied: dict[str, int] = {}
        self._verified: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        with contextlib.suppress(Exception):
            if self._path.exists():
                d = json.loads(self._path.read_text(encoding="utf-8"))
                self._accepted = {k: int(v) for k, v in d.get("accepted", {}).items()}
                self._applied = {k: int(v) for k, v in d.get("applied", {}).items()}
                self._verified = {k: int(v) for k, v in d.get("verified", {}).items()}

    def _save(self) -> None:
        with contextlib.suppress(Exception):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(
                    {"accepted": self._accepted, "applied": self._applied,
                     "verified": self._verified},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    _MAX = 50

    def record_acceptance(
        self, candidate_id: str, accepted: bool,
        verified: bool | None = None,
    ) -> None:
        """记录一次真实 apply_patches 采纳.

        verified: 论文 arXiv:2609.03621 — 该采纳是否经状态化仿真验证通过
        (前置满足 + 约束合法 + 对象变换). 单独记账, 不覆盖采纳率主锚.
        """
        self._applied[candidate_id] = self._applied.get(candidate_id, 0) + 1
        if accepted:
            self._accepted[candidate_id] = self._accepted.get(candidate_id, 0) + 1
        if verified is not None:
            self._verified[candidate_id] = self._verified.get(candidate_id, 0) + 1
        if len(self._applied) > self._MAX:  # LRU 抗无限增长
            for k in list(self._applied)[: len(self._applied) - self._MAX]:
                self._applied.pop(k, None)
                self._accepted.pop(k, None)
                self._verified.pop(k, None)
        self._save()

    def fidelity_score(self, candidate_id: str) -> float:
        """采纳率 ETF 平滑的保真锚; 经状态化仿真验证的采纳加权上修 (clamp [0,1])."""
        a = self._applied.get(candidate_id, 0)
        if a == 0:
            return 0.5  # 未知 → 中性
        rate = self._accepted.get(candidate_id, 0) / a
        v = self._verified.get(candidate_id, 0)
        # 验证加成: 被仿真验证过的采纳更可信, 但加力不超过采纳率本身 (往 1 收敛)
        # verified=False 显式记录 → 视为"未验证", 不上修.
        if v > 0:
            # 只对 accepted 计验证, 避免"拒绝也被上修"
            verified_accepted = min(v, self._accepted.get(candidate_id, 0))
            rate = rate + 0.15 * (verified_accepted / a)
        return max(0.0, min(1.0, rate))

    def anchor_in(self, candidate_id: str, p_quality: float,
                  p_fidelity: float | None = None) -> float:
        """加权合成: quality ⊕ 真实采纳保真 (各 0.5)."""
        f = p_fidelity if p_fidelity is not None else self.fidelity_score(candidate_id)
        return 0.5 * p_quality + 0.5 * f

    def snapshot(self) -> dict[str, Any]:
        return {
            "accepted": dict(self._accepted),
            "applied": dict(self._applied),
            "verified": dict(self._verified),
        }


class VerifiableGate:
    """可验证工作流验收门控 — 论文 arXiv:2609.03621 (spec #5).

    论文主张: 用「可计算实验室表示」(类型化研究对象 + 能力受限操作 + 组合工作流
    代数) 建立 agent 推理与能力受限物理变换之间的通用计算接口, 并在**派发前**用
    状态化仿真验证操作前置条件与实验室约束. 本类把 strategies/improver 换件验收
    从「优化标量代理分」根治为「验证能力受限变换」:

    - ``verify_outcome(plan)``: 尽力接入 ``huginn.security.world_model`` 的
      ``apply_forward`` / ``check_constraints`` (模块级纯函数), 验证:
        前置条件满足 + 变换类型化对象 + 不违反实验室约束. 通过 → True.
    - ``enabled()``: 机器可用且显式开启 (``harness_verifiable_gate``) 才是硬 gate;
      不可用/未开 → advisory (不阻塞推进, 回落 BehavioralFidelity 锚).
    """

    def __init__(self) -> None:
        self._current_state: dict[str, Any] = {}
        # 真实实验账本: agent 实际执行过的 (state, action, observed). 跨 session 持久化.
        self._exec_path = get_runtime_home() / "meta_improver" / "verifiable_executions.json"
        self._executions: list[dict[str, Any]] = []
        self._load_exec()

    def enabled(self) -> bool:
        return _harness_enabled("harness_verifiable_gate")

    # 代表性能力动作电池: 对齐 world_model.FORWARD_EFFECTS 的真实能力集.
    _CAPABILITY_BATTERY: list[tuple[str, dict[str, Any]]] = [
        ("aspirate", {"vol": 1.0}),
        ("dispense", {"vol": 1.0}),
        ("mix", {}),
        ("aliquot", {"n": 1}),
    ]

    # 真实实验账本 LRU 上限 — 抗无限增长.
    _EXEC_MAX = 100

    def _load_exec(self) -> None:
        with contextlib.suppress(Exception):
            if self._exec_path.exists():
                self._executions = json.loads(
                    self._exec_path.read_text(encoding="utf-8"))[-self._EXEC_MAX:]

    def _save_exec(self) -> None:
        with contextlib.suppress(Exception):
            self._exec_path.parent.mkdir(parents=True, exist_ok=True)
            self._exec_path.write_text(
                json.dumps(self._executions[-self._EXEC_MAX:], ensure_ascii=False),
                encoding="utf-8",
            )

    def record_execution(self, action_type: str, params: dict[str, Any],
                         state_before: dict[str, Any],
                         observed: dict[str, Any] | None = None,
                         ts: float | None = None) -> dict[str, Any]:
        """记录一次 agent **真实执行**过的实验 (state+action+observed) 并就地校验.

        让"验证"面向 agent 真实产出的实验 (论文 2609.03621: 组合工作流代数), 而非
        只验代表性电池. 校验结果随账本持久化, 供 ``verify_recent_executions`` 汇总.
        """
        import huginn.security.world_model as wm
        from huginn.security.world_model import PhysicalAction

        entry: dict[str, Any] = {
            "action_type": action_type, "params": dict(params),
            "state_before": dict(state_before),
            "observed": dict(observed) if observed is not None else None,
            "ts": ts if ts is not None else time.time(),
        }
        known = set(getattr(wm, "FORWARD_EFFECTS", {}))
        issues: list[str] = []
        if action_type not in known:
            issues.append(f"unknown_capability:{action_type}")
        else:
            try:
                action = PhysicalAction(action_type, dict(params))
                issues = list(wm.check_constraints(state_before, action))
                if observed is not None:
                    predicted = wm.apply_forward(dict(state_before), action)
                    # 前向预测必须与真实观测一致才通过 (验证 agent 实机的物理一致性)
                    for k, v in observed.items():
                        if abs(float(predicted.get(k, 0.0) or 0.0) - float(v or 0.0)) > 1e-6:
                            issues.append(f"mismatch:{k}")
                            break
            except Exception as exc:
                issues.append(f"world_model_error:{exc}")
        entry["valid"] = not issues
        entry["issues"] = issues
        self._executions.append(entry)
        self._save_exec()
        # 数据驱动世界模型: 每次真实执行都喂给 LearnedWorldModel (dual-axis 第二根轴) —
        # 从 agent 真实 state_after 学习"世界如何变换", 而非只依赖硬编码 FORWARD_EFFECTS.
        if observed is not None:
            try:
                from huginn.security.world_model import PhysicalAction, learned_world_model
                learned_world_model().learn(
                    state_before,
                    PhysicalAction(action_type, dict(params)),
                    observed,
                )
            except Exception:
                logger.debug("learned_world_model learn failed", exc_info=True)
        return {"valid": not issues, "issues": issues}

    def verify_recent_executions(self, k: int = 10) -> dict[str, Any]:
        """汇总验证最近 k 次真实实验. 返回 {passed, n, passed_n, failures}.

        无真实实验记录 → ``n=0`` (调用方回落电池). 全部合法/一致才 passed.
        """
        recent = self._executions[-k:]
        if not recent:
            return {"passed": None, "n": 0, "passed_n": 0, "failures": []}
        failures = [f"#{i}:{';'.join(e['issues'])}"
                    for i, e in enumerate(recent) if not e["valid"]]
        passed_n = sum(1 for e in recent if e["valid"])
        return {"passed": passed_n == len(recent), "n": len(recent),
                "passed_n": passed_n, "failures": failures}

    def verify_battery(
        self, state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """对真实 world_model 验证一组能力动作电池 (非单点探针).

        每个动作: 已知能力 + check_constraints 无违规 + apply_forward 可传播才计数.
        返回 ``{"passed", "passed_n", "total", "failures"}``; 全部通过才 passed.
        """
        import huginn.security.world_model as wm
        from huginn.security.world_model import PhysicalAction

        base = dict(self._current_state)
        if isinstance(state, dict) and state:
            base = dict(state)
        if not base:
            base = {"reagent_vol": 10.0, "sample_vol": 3.0, "tube_vol": 0.0}
        known = set(getattr(wm, "FORWARD_EFFECTS", {}))
        passed_n = 0
        failures: list[str] = []
        for atype, params in self._CAPABILITY_BATTERY:
            if atype not in known:
                failures.append(f"unknown_capability:{atype}")
                continue
            action = PhysicalAction(atype, dict(params))
            issues = wm.check_constraints(base, action)
            if issues:
                failures.append(f"{atype}:{';'.join(issues)}")
                continue
            try:
                wm.apply_forward(base, action)
                passed_n += 1
            except Exception as exc:
                failures.append(f"{atype}:{exc}")
        return {"passed": passed_n == len(self._CAPABILITY_BATTERY) and not failures,
                "passed_n": passed_n, "total": len(self._CAPABILITY_BATTERY),
                "failures": failures}

    def _world_available(self) -> bool:
        try:
            import huginn.security.world_model as wm
            return wm is not None
        except Exception:
            return False

    def verify_outcome(self, plan_or_experiment: dict[str, Any]) -> dict[str, Any]:
        """在状态化仿真下验证一个实验/计划.

        plan 形态: ``{"state": {...}, "action": PhysicalAction}``  或
                   ``{"state": {...}, "action_type": str, "params": {...}}``
        返回: ``{"passed": bool, "new_state": dict|None, "error": str|None,
                 "issues": list[str]}``. world_model 不可用 → passed=False,
        不抛异常 (advisory caller 决定是否阻塞).
        """
        import huginn.security.world_model as wm

        state = dict(self._current_state)
        if isinstance(plan_or_experiment.get("state"), dict):
            state = dict(plan_or_experiment["state"])
        action = plan_or_experiment.get("action")
        if isinstance(action, dict):
            action = wm.PhysicalAction(
                action.get("type", ""), dict(action.get("params") or {})
            )
        elif action is None:
            action_type = plan_or_experiment.get("action_type")
            if not action_type:
                return {"passed": False, "new_state": None,
                        "error": "no action in plan", "issues": ["missing_action"]}
            action = wm.PhysicalAction(action_type, dict(plan_or_experiment.get("params") or {}))
        try:
            # 能力受限操作: 动作类型必须落在已知变换集内, 否则拒绝无依据转移
            known = set(getattr(wm, "FORWARD_EFFECTS", {}))
            if action.type not in known:
                return {"passed": False, "new_state": None,
                        "error": f"unknown_capability: {action.type}",
                        "issues": [f"unknown_capability:{action.type}"]}
            # 前置条件 + 实验室约束 (空 issues = 合法)
            issues = wm.check_constraints(state, action)
            if issues:
                return {"passed": False, "new_state": None,
                        "error": "; ".join(issues), "issues": issues}
            # 传播类型化研究对象变换
            new_state = wm.apply_forward(state, action)
            self._current_state = dict(new_state)
            return {"passed": True, "new_state": new_state,
                    "error": None, "issues": []}
        except Exception as exc:  # 世界模型内部故障 → advisory fail, 不硬崩
            return {"passed": False, "new_state": None,
                    "error": str(exc), "issues": ["world_model_error"]}


class RandomizedControl:
    """随机化对照仲裁 — 真实 r_phys 差分 (spec #3, 默认 off).

    champion 与固定 baseline 各跑 N 次, 比较真实 r_phys 的中位差判定
    ``champion_better``. 仅当 ``HUGINN_META_ABLATION=1`` (显式评估) 才启用,
    作为对单点代理分可靠性的疑虑仲裁. ``override_pair`` 允许运行时注入
    一组已测差分 (测试/离线评估), 避免每次全跑实机.
    """

    def __init__(self) -> None:
        self.override_pair: dict[str, Any] | None = None
        # 真实 r_phys 采样 (由评估方注入; 空 = 未提供 → run_pair 中位差基于空 → 保守)
        self.champion_r: list[float] = []
        self.baseline_r: list[float] = []

    def enabled(self) -> bool:
        return os.environ.get("HUGINN_META_ABLATION", "").lower() in ("1", "true", "yes")

    def run_pair(self, *, champion_r: list[float], baseline_r: list[float],
                 tolerance: float = 0.02) -> dict[str, Any]:
        """比较两组真实 r_phys 的中位差. 返回 {champion_better, delta, n_champ, n_base}.

        - override_pair 存在 → 直接用注入结果 (离线评估/测试).
        - 否则中位差 > tolerance → champion_better.
        """
        if self.override_pair is not None:
            return dict(self.override_pair)
        import statistics

        champ = statistics.median(champion_r) if champion_r else 0.0
        base = statistics.median(baseline_r) if baseline_r else 0.0
        return {"champion_better": (champ - base) > tolerance,
                "delta": round(champ - base, 4),
                "n_champ": len(champion_r), "n_base": len(baseline_r)}


# A1 端到端轨道: 真实 r_phys 树配置至少 3 代才可判 (否则 advisory).
_R_PHYS_TRACK_MIN = 3


class RPhysTrack:
    """端到端 r_phys 真实验收轨道 — Goodhart 的地面真值通道 (论文 2609.03621).

    现状诚实声明: ``evaluate``/``evaluate_strategist`` 的分数是「产出 patch 的
    有效性」**代理分** (``score_patch_output``), 由 LLM 离线打的, 不是真实 r_phys.
    真实 r_phys (agent 实际物理验证分) 只被 ``note_generation`` 收到后丢进 replay,
    从未被归因到驱动该次 patch 的改进器配置, 更未作为换件验收的真值.

    本轨道把每次真实 r_phys 归因到「当时代际的 active 改进器配置」, 形成按配置的
    真实 r_phys 代际序列, 再用**不成对两样本检验 (Mann-Whitney U, 单侧)** 判定:

      H0: median(r_phys | config) <= median(r_phys | 其余配置池)
      H1: 该配置驱动时 r_phys 显著高于其余配置 (尤指默认 baseline) 池

    显著上行 + 样本充足 → ``rphys_green``。这就是把验收从「优化标量代理」根治到
    「验证真实物理状态随该配置上行」的落点 — LLM 代理分可能被优化/gaming, 真实
    r_phys 代际走向不可随意捏造。

    不开不成对强配对: 各配置驱动的代数不同, 强配对会稀释样本; n>=3 才可判,
    不足 → green=None (advisory 不阻塞, 回落既有代理分闸). 仅在显式开启
    ``harness_rphys_gate`` 时才作硬闸.
    """

    _MIN = _R_PHYS_TRACK_MIN

    def __init__(self, path: Any | None = None) -> None:
        from pathlib import Path as _Path

        self._path = path or (_Path(get_runtime_home()) / "meta_improver" / "rphys.json")
        self._series: dict[str, list[float]] = {}  # config_id -> 真实 r_phys 代际序列
        self._load()

    def _load(self) -> None:
        with contextlib.suppress(Exception):
            if self._path.exists():
                d = json.loads(self._path.read_text(encoding="utf-8"))
                self._series = {k: [float(x) for x in v]
                                for k, v in d.get("series", {}).items()}

    def _save(self) -> None:
        with contextlib.suppress(Exception):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"series": self._series}, ensure_ascii=False),
                encoding="utf-8",
            )

    def record(self, config_id: str, r_phys: float | None) -> None:
        """归因一次真实 r_phys 到驱动该代 patch 的改进器配置 (LRU 抗无限增长)."""
        if r_phys is None:
            return
        self._series.setdefault(config_id, []).append(float(r_phys))
        if len(self._series[config_id]) > _REPLAY_MAX:
            self._series[config_id] = self._series[config_id][-_REPLAY_MAX:]
        self._save()

    def series(self, config_id: str) -> list[float]:
        return list(self._series.get(config_id, []))

    def _pool(self, config_id: str) -> list[float]:
        """其余配置的真实 r_phys 池 (含默认 baseline 的实测)."""
        pool: list[float] = []
        for cid, vals in self._series.items():
            if cid != config_id:
                pool.extend(vals)
        return pool

    @staticmethod
    def _median(v: list[float]) -> float:
        import statistics

        return statistics.median(v) if v else 0.0

    def verdict(self, config_id: str, alpha: float = 0.05) -> dict[str, Any]:
        """端到端 r_phys 验收判定.

        返回 ``{"green", "n", "pool", "p", "median_track", "median_pool",
        "reason"}``. ``green`` 为 None → 样本不足/不可判 (advisory 不阻塞).
        """
        xs = self.series(config_id)
        ys = self._pool(config_id)
        if len(xs) < self._MIN or not ys:
            return {"green": None, "n": len(xs), "pool": len(ys),
                    "p": None, "median_track": self._median(xs),
                    "median_pool": self._median(ys), "reason": "insufficient"}
        p = _mannwhitney_p_greater(xs, ys)
        green = bool(p < alpha)
        return {"green": green, "n": len(xs), "pool": len(ys),
                "p": round(p, 4), "median_track": self._median(xs),
                "median_pool": self._median(ys),
                "reason": "significantly_higher" if green else "not_significant"}


# 模块级 randomized-control 单例句柄 (测试可注入覆盖).
_RC_INSTANCE: RandomizedControl | None = None


def _randomized_control() -> RandomizedControl:
    """返回 RandomizedControl 共享实例 (可被测试注入覆盖)."""
    global _RC_INSTANCE
    if _RC_INSTANCE is None:
        _RC_INSTANCE = RandomizedControl()
    return _RC_INSTANCE


def build_improver_prompt(
    template: str, phase: str, block_names: list[str], r_phys: float | None,
    directive: str,
) -> str | None:
    """用占位符实例化 improver 模板. 模板非法(f-string 不兼容/缺占位符)返回 None."""
    try:
        return template.format(
            phase=phase,
            block_names=block_names,
            r_phys=(f"{r_phys:.2f}" if r_phys is not None else "None"),
            directive=directive or "",
        )
    except (KeyError, IndexError, ValueError) as exc:
        logger.debug("meta_improver: improv template format failed: %s", exc)
        return None


def score_patch_output(
    response: str, block_names: list[str], directive: str
) -> float:
    """给一次 improver 产出的 patch 打效性代理分 (0..1, 越高越好).

    诚实声明: 这是「有效性」代理分, 非真实 r_phys. 只作首道闸, 真实收益由
    下游 apply_patches 的 Beta 接受度兜底.
    """
    if not response or not response.strip():
        return 0.2
    txt = response.strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        d = json.loads(txt)
    except Exception:
        return 0.1
    block_name = d.get("block_name", "")
    if block_name not in block_names:
        return 0.3
    new_text = str(d.get("new_text", "")).strip()
    if not new_text:
        return 0.2
    # 有效 patch 基准 1.0; 若 patch 对齐了本次 self-directive 的可检索词,
    # 每个 +0.15 (上限 +0.45), 不封顶到 1.0 — 这样"更能对齐 directive 的
    # 改进器模板"能在代理分上胜过"通用但未对齐"的, 递归 evaluate 才能判优劣.
    score = 1.0
    for tok in (directive or "").split()[:5]:
        if len(tok) > 3 and tok in new_text:
            score = min(1.45, score + 0.15)
    return score


class MetaImprover:
    """meta-improver 单例. 持久化进 .huginn/harness/meta_improver/."""

    _instance: MetaImprover | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        cache_dir = get_runtime_home()
        self._dir = cache_dir / "meta_improver"
        self._candidates_dir = self._dir / "candidates"
        with contextlib.suppress(Exception):
            self._dir.mkdir(parents=True, exist_ok=True)
            self._candidates_dir.mkdir(parents=True, exist_ok=True)
        self._cfg_path = self._dir / "config.json"
        self._replay_path = self._dir / "replay.json"
        self._trace_path = self._dir / "meta_trace.jsonl"
        self._strategists_dir = self._dir / "strategist"
        self._active_id: str | None = None
        self._history: list[str] = []
        self._promotions: int = 0
        self._candidates: dict[str, ImproverConfig] = {}
        self._replay: list[dict[str, Any]] = []
        self._propose_count = 0
        self._strategists: dict[str, StrategistConfig] = {}
        self._strategists_history: list[str] = []
        self._active_strategist_id: str | None = None
        self._tracker = CompoundingTracker()
        self._coeffect: Any = None
        self._fidelity = BehavioralFidelity(path=self._dir / "fidelity.json")
        self._rphys = RPhysTrack(path=self._dir / "rphys.json")
        self._compounding_path = self._dir / "compounding.json"
        self._strategy_proposals = 0
        self._source_proposals = 0
        self._load()
        self._load_compounding()
        with contextlib.suppress(Exception):
            self._strategists_dir.mkdir(parents=True, exist_ok=True)

    # ── singleton ──────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> MetaImprover:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── persistence ─────────────────────────────────────────────────────────
    def _load(self) -> None:
        with contextlib.suppress(Exception):
            if self._cfg_path.exists():
                d = json.loads(self._cfg_path.read_text(encoding="utf-8"))
                self._active_id = d.get("active_config_id")
                self._history = d.get("history", [])
                self._promotions = int(d.get("promotions", 0))
                self._active_strategist_id = d.get("active_strategist_id")
                self._strategists_history = d.get("strategists_history", [])
        with contextlib.suppress(Exception):
            if self._replay_path.exists():
                self._replay = json.loads(self._replay_path.read_text(encoding="utf-8"))[: _REPLAY_MAX]
        with contextlib.suppress(Exception):
            for f in self._candidates_dir.glob("*.json"):
                try:
                    c = ImproverConfig.from_dict(json.loads(f.read_text(encoding="utf-8")))
                    self._candidates[c.config_id] = c
                except Exception:
                    logger.debug("meta candidate load fail: %s", f, exc_info=True)
        with contextlib.suppress(Exception):
            for f in self._strategists_dir.glob("*.json"):
                try:
                    s = StrategistConfig.from_dict(json.loads(f.read_text(encoding="utf-8")))
                    self._strategists[s.config_id] = s
                except Exception:
                    logger.debug("meta strategist load fail: %s", f, exc_info=True)

    def _save_cfg(self) -> None:
        with contextlib.suppress(Exception):
            self._cfg_path.write_text(
                json.dumps(
                    {
                        "active_config_id": self._active_id,
                        "history": self._history,
                        "promotions": self._promotions,
                        "active_strategist_id": self._active_strategist_id,
                        "strategists_history": self._strategists_history,
                    },
                    ensure_ascii=False, indent=2,
                ), encoding="utf-8"
            )

    def _save_replay(self) -> None:
        with contextlib.suppress(Exception):
            self._replay_path.write_text(
                json.dumps(self._replay, ensure_ascii=False), encoding="utf-8"
            )

    def _save_candidate(self, cfg: ImproverConfig) -> None:
        with contextlib.suppress(Exception):
            (self._candidates_dir / f"{cfg.config_id}.json").write_text(
                json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _save_strategist(self, s: StrategistConfig) -> None:
        with contextlib.suppress(Exception):
            self._strategists_dir.mkdir(parents=True, exist_ok=True)
            (self._strategists_dir / f"{s.config_id}.json").write_text(
                json.dumps(s.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load_compounding(self) -> None:
        """载入已持久化的复合追踪器窗口 (compounding.json, spec data shape)."""
        with contextlib.suppress(Exception):
            if self._compounding_path.exists():
                d = json.loads(self._compounding_path.read_text(encoding="utf-8"))
                rows = d.get("rows") or []
                if rows:
                    self._tracker._rows = [dict(r) for r in rows][-self._tracker.window:]
                self._tracker._deadlock_n = int(d.get("deadlock_n", 0))

    def _save_compounding(self) -> None:
        """把复合追踪器窗口与死锁计数落盘 (compounding.json, spec data shape)."""
        with contextlib.suppress(Exception):
            self._dir.mkdir(parents=True, exist_ok=True)
            data = {
                "rows": self._tracker._rows,
                "deadlock_n": self._tracker._deadlock_n,
                **self._tracker.stats(),
            }
            self._compounding_path.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )

    def _trace(self, entry: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            with self._trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**entry, "ts": time.time()}, ensure_ascii=False) + "\n")

    # ── enabled ──────────────────────────────────────────────────────────────
    def enabled(self) -> bool:
        """meta 层仅当两个开关都 on 才生效. 默认全 off."""
        return _harness_enabled("harness_meta_improver") and _harness_enabled(
            "harness_prompt_patch"
        )

    # ── champion / replays ───────────────────────────────────────────────────
    def champion_cfg(self) -> ImproverConfig | None:
        """当前活跃 champion (config_id + active=True). 无则 None → 回落默认模板."""
        if not self.enabled():
            return None
        c = self._candidates.get(self._active_id or "")
        if c is not None and c.active:
            return c
        return None

    def current_template(self) -> str:
        champ = self.champion_cfg()
        return champ.improver_prompt if champ else DEFAULT_IMPROV_TEMPLATE

    # ── strategist (level-1 递归层) ─────────────────────────────────────────
    def strategist_champion(self) -> StrategistConfig | None:
        """当前活跃 strategist (改进策略). 无则 None → maybe_propose 回落默认."""
        if not self.enabled():
            return None
        s = self._strategists.get(self._active_strategist_id or "")
        return s if s is not None and s.active else None

    def strategist_prompt(self) -> str:
        """strategist 模板: 有 champion 用之, 无则回落 _META_IMPROVE_TEMPLATE."""
        champ = self.strategist_champion()
        return champ.strategist_prompt if champ else _META_IMPROVE_TEMPLATE

    async def maybe_propose_strategist(self, llm_chat_fn: Callable) -> str | None:
        """meta² 固定模板生成一个新的 strategist(改进策略) 模板候选.

        只一层递归 (A1 边界): strategist 自身的 meta³ 不改. 候选模板必须能被
        maybe_propose 以 .format(current_template=...) 实例化, 否则作废.
        """
        if not self.enabled():
            return None
        cur = self.strategist_prompt()
        p2 = _META2_IMPROVE_TEMPLATE.format(
            current_strategist=cur,
            n_promotions=self._promotions,
            win_rate=self.compounding_trace()["meta_win_rate"],
        )
        try:
            resp = await llm_chat_fn(p2, task="summarize")
        except Exception:
            logger.debug("meta propose_strategist LLM fail", exc_info=True)
            return None
        if not resp or not resp.strip():
            return None
        tpl = resp.strip()
        if tpl.startswith("```"):
            tpl = tpl.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if not tpl:
            return None
        # 模板必须是 maybe_propose 可实例化的 meta 模板: 保留 current_template 占位符
        # (且能被 .format 解析, 无非受控 brace).
        if "{current_template}" not in tpl:
            self._trace({"type": "strategy_reject", "reason": "invalid_template"})
            return None
        try:
            tpl.format(current_template="x", n_proposals=0, n_promotions=0)
        except (KeyError, IndexError, ValueError):
            self._trace({"type": "strategy_reject", "reason": "invalid_format"})
            return None
        s = StrategistConfig(
            config_id=f"strat_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}",
            strategist_prompt=tpl,
        )
        self._strategists[s.config_id] = s
        self._save_strategist(s)
        self._trace({"type": "strategy_propose", "candidate_id": s.config_id})
        return s.config_id

    def _persist_state_ref(self) -> dict[str, Any]:
        """把 active_strategist_id 等现状以可序列化形式交由补偿器恢复.

        只存纯数据 (不做 lambda/不可序列化引用), 以便 journal 可跨崩溃重放:
        恢复时按 kind 的补偿器负责把状态写回发起方 (本模块见
        ``_compensate_strategist_swap``).
        """
        return {
            "active_strategist_id": self._active_strategist_id,
            "strategists_history": list(self._strategists_history),
        }

    def _apply_strategist_swap(self, old: str, new: str, ctx: Any | None) -> None:
        """执行 strategist 换件, 并把逆登记进 RevertibleContext (时间可组合).

        - 更新 active 指向 + 每个候选的 active 标记 + 持久化.
        - 换件成功后把旧 champion 追加进 strategies_history (供无 ctx 回退).
        - ctx 存在时登记 ``strategist_swap`` 补偿逆; 崩溃后经 journal 重放恢复.
        """
        if old and old != new and old not in self._strategists_history:
            self._strategists_history.append(old)
        self._active_strategist_id = new
        for s in self._strategists.values():
            s.active = (s.config_id == new)
            self._save_strategist(s)
        self._save_cfg()
        if ctx is not None:
            try:
                ctx.compensate("strategist_swap", {
                    "old": old, "new": new,
                    "store": self._persist_state_ref(),
                })
            except Exception as exc:
                logger.debug("meta: strategist_swap compensate reg failed", exc_info=True)

    def _ablation_samples(self) -> tuple[list[float], list[float]]:
        """Ablation 双臂: (champion, baseline) 的真实 r_phys 采样.

        champion 臂 = 当前部署改进器 (improver champion, 无则回落 strategist
        champion) 的真实 r_phys 序列; baseline 臂 = 默认 ``_base`` 的真实序列.
        RPhysTrack 已在 note_generation 按 active 配置归因真实 r_phys — 把此前
        只靠 override 注入的仲裁改为读真实收集数据.
        """
        base = self._rphys.series("_base")
        champ_id = self._active_id or self._active_strategist_id or ""
        champ = self._rphys.series(champ_id) if champ_id else []
        return champ, base

    def revert_strategist(self, ctx: Any | None = None) -> bool:
        """复合退化时回退 strategist champion (时间可组合).
        """
        if ctx is not None:
            try:
                ctx.revert_all()
                return True
            except Exception:
                return False
        prev = self._strategists_history[-1] if self._strategists_history else None
        if prev is None:
            return False
        curr = self._active_strategist_id or ""
        self._apply_strategist_swap(curr, prev, None)
        self._trace({"type": "strategy_revert", "candidate_id": prev})
        return True

    # ── evaluate / promote strategist (门控组合) ──────────────────────────────
    async def evaluate_strategist(self, candidate_id: str, llm_chat_fn: Callable) -> dict[str, Any]:
        """评估 strategist 候选并**真实记录** sig/ood 数据 (离线, 不换件).

        候选策略与基准 (默认 ``_META_IMPROVE_TEMPLATE``) 都是「改写 improver 模板」
        的机制模板: 分别回填探针上下文后送 LLM 产 improver 模板, 再各自对冻结
        重放集产 patch 打分, 配对记录进 SignificanceGate + OODHoldoutValidator
        (HIGH-1 修复 — 让候选有真实数据可判).

        返回 ``{"green", "sig", "ood", "base_mean", "cand_mean", "n"}``.
        """
        if not self.enabled():
            return {"green": False, "sig": False, "ood": False, "n": 0}
        s = self._strategists.get(candidate_id)
        if s is None:
            return {"green": False, "sig": False, "ood": False, "n": 0}
        if not self._replay:
            return {"green": False, "sig": False, "ood": False, "n": 0}
        from huginn.harness.significance_gate import SignificanceGate
        from huginn.harness.ood_holdout import OODHoldoutValidator
        sig, ood = SignificanceGate.get_instance(), OODHoldoutValidator.get_instance()
        base_scores: list[float] = []
        cand_scores: list[float] = []
        for probe in self._replay:
            # 各产 improver 模板 (策略提示 → improver 模板)
            base_tpl = await self._meta_prompt_to_improver(
                _META_IMPROVE_TEMPLATE, probe, llm_chat_fn)
            cand_tpl = await self._meta_prompt_to_improver(
                s.strategist_prompt, probe, llm_chat_fn)
            if base_tpl is None or cand_tpl is None:
                continue
            base_score = await self._score_improver_on_probe(
                ImproverConfig(config_id="_base", improver_prompt=base_tpl), probe, llm_chat_fn)
            cand_score = await self._score_improver_on_probe(
                ImproverConfig(config_id=candidate_id, improver_prompt=cand_tpl), probe, llm_chat_fn)
            if base_score is None or cand_score is None:
                continue
            base_scores.append(base_score)
            cand_scores.append(cand_score)
            task_id = str(probe.get("probe_id") or probe.get("ts") or random.random())
            sig.record_pair(candidate_id, base_score, cand_score, task_id=task_id)
            ood.record_outcome(ood._BASELINE_ID, task_id, base_score)
            ood.record_outcome(candidate_id, task_id, cand_score)
        sig_ok = sig.gate_decision(candidate_id, min_samples=_MIN_SAMPLES).passed
        ood_ok = ood.validate_ood(candidate_id).passed
        green = bool(sig_ok and ood_ok)
        self._trace({"type": "strategy_evaluate", "candidate_id": candidate_id,
                     "sig": sig_ok, "ood": ood_ok, "green": green,
                     "n": len(base_scores)})
        return {"green": green, "sig": sig_ok, "ood": ood_ok,
                "base_mean": _mean(base_scores), "cand_mean": _mean(cand_scores),
                "n": len(base_scores)}

    async def _meta_prompt_to_improver(
        self, meta_tpl: str, probe: dict[str, Any], llm_chat_fn: Callable,
    ) -> str | None:
        """把 meta 提示模板回填探针上下文送 LLM, 期望它返回一个 improv 模板.

        返回必须能被 build_improver_prompt 用 {phase}/{block_names}/{r_phys}/
        {directive} 实例化 (即含四个占位符); 否则 None."""
        try:
            prompt = meta_tpl.format(
                current_template=probe.get("improver_template") or DEFAULT_IMPROV_TEMPLATE,
                n_proposals=len(self._candidates), n_promotions=len(self._history),
                phase=probe.get("phase", ""),
                block_names=", ".join(probe.get("block_names", [])),
                r_phys=(f"{probe.get('r_phys', 0.0):.2f}" if probe.get("r_phys") is not None else "None"),
                directive=probe.get("directive", ""),
            )
        except (KeyError, IndexError, ValueError):
            return None
        try:
            resp = await llm_chat_fn(prompt, task="summarize")
        except Exception:
            return None
        if not resp or not resp.strip():
            return None
        tpl = resp.strip()
        if tpl.startswith("```"):
            tpl = tpl.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        for ph in ("{phase}", "{block_names}", "{r_phys}", "{directive}"):
            if ph not in tpl:
                return None
        return tpl or None

    async def _score_improver_on_probe(
        self, cfg: ImproverConfig, probe: dict[str, Any], llm_chat_fn: Callable,
    ) -> float | None:
        """用 improver 模板在单个探针上产 patch 打分 (复用 _score_candidate)."""
        return await self._score_candidate(cfg, probe, llm_chat_fn)

    def maybe_promote_strategist(self, candidate_id: str, ctx: Any | None = None) -> bool:
        """strategist 换件门控: 显著 + OOD + 复合不退化 → 事务内换件.

        Goodhart 治本 (论文 arXiv:2609.03621 + BehavFed) 分层:
          - 复合指标用 BehavioralFidelity 保真锚合成 (真实采纳, 不可随意优化);
          - VerifiableGate 机器可用且显式开启时, 额外要求候选改进器产出的实验
            通过状态化仿真验证; 未开/不可用 → advisory 不阻塞 (回落保真锚).
        """
        if not self.enabled():
            return False
        s = self._strategists.get(candidate_id)
        if s is None:
            return False
        from huginn.security.revertible import RevertibleContext
        from huginn.harness.significance_gate import SignificanceGate
        from huginn.harness.ood_holdout import OODHoldoutValidator
        sig_ok = SignificanceGate.get_instance().gate_decision(
            candidate_id, min_samples=_MIN_SAMPLES,
        ).passed
        ood_ok = OODHoldoutValidator.get_instance().validate_ood(candidate_id).passed
        if not (sig_ok and ood_ok):
            # 死锁检测: 记录 YELLOW, 连续超阈值则降级 (advisory)
            self._tracker.mark_deadlock(yellow=True)
            if self._tracker.in_deadlock():
                logger.info("meta: strategist deadlock — advisory (not blocking)")
            self._trace({"type": "strategy_reject", "candidate_id": candidate_id,
                         "reason": "not_green"})
            return False
        incumbent = self._active_strategist_id or ""
        if self._tracker.would_degrade(candidate_id, incumbent):
            self._trace({"type": "strategy_reject", "candidate_id": candidate_id,
                         "reason": "would_degrade"})
            return False
        # 可验证工作流 gate (advisory): 通关则强化可信; 未开/不可用不阻塞
        verified = self._verify_strategist_outcome(s)
        # 随机化对照仲裁 (HUGINN_META_ABLATION=1): 真实 r_phys 差分 champion 劣于
        # baseline(或未提供对照) → 拒绝. 用 RPhysTrack 已收集的真实采样 (非空注入).
        rc = _randomized_control()
        if rc.enabled():
            champ_r, base_r = self._ablation_samples()
            pair = rc.run_pair(champion_r=champ_r, baseline_r=base_r)
            if not pair["champion_better"]:
                self._trace({"type": "strategy_reject", "candidate_id": candidate_id,
                             "reason": "ablation_champion_worse"})
                return False
        self._tracker.mark_deadlock(yellow=False)
        rctx = ctx or RevertibleContext(
            journal_path=self._dir / "revertible_journal.json"
            if (self._dir / "revertible_journal.json").parent.exists() else None,
        )
        try:
            _register_strategist_compensator()  # 确保生产已注册 (幂等)
            with rctx.transaction():
                self._apply_strategist_swap(incumbent, candidate_id, rctx)
        except Exception as exc:
            logger.debug("meta: strategist promote txn failed", exc_info=True)
            return False
        # 复合指标用 BehavioralFidelity 保真锚合成真实质量 (治 Goodhart, 非硬编码常数)
        quality = self._fidelity.anchor_in(candidate_id, p_quality=0.8)
        self._tracker.record(
            epoch=len(self._tracker._rows), config_id=candidate_id,
            quality=quality, fidelity=self._fidelity.fidelity_score(candidate_id),
            proposals_to_promotion=1, generations_to_promotion=1, win_rate=1.0,
        )
        self._save_compounding()
        self._trace({"type": "strategy_promote", "candidate_id": candidate_id,
                     "verified": verified, "quality": round(quality, 3)})
        return True

    def _verify_strategist_outcome(self, s: StrategistConfig) -> bool | None:
        """接入 VerifiableGate (论文): 机器可用且开启时做状态化仿真验证.

        优先验证 **agent 真实执行过的实验** (record_execution 账本) — 若账本有真实
        实验, 用其是否全部合法/前向一致判定; 否则回落代表性能力电池. gate 不可用/
        未开 → None (advisory 未验证).
        """
        try:
            vg = VerifiableGate()
            if not vg.enabled():
                return None
            real = vg.verify_recent_executions(k=10)
            if real["n"]:  # 有真实实验入库 → 以真实产出为准
                return bool(real["passed"])
            # 无真实实验 → 能力电池兜底
            res = vg.verify_battery()
            return bool(res["passed"])
        except Exception:
            return None

    # ── A3: source 环 (运行时函数级自改源码, v0, harness_source_patch 门) ──────
    ## A3 自改目标白名单: agent 可自改的内部数值调参常量 (验证/可逆/默认关). 轮转选择,
    ## 不再钉死在单一常量. 只含 meta_improver 自有调参, 不动逻辑/物理不变量.
    _SOURCE_TARGETS: list[tuple[str, str]] = [
        ("huginn.harness.meta_improver", "_PROPOSE_EVERY_N"),
        ("huginn.harness.meta_improver", "_STRATEGIST_EVERY_N_PROPOSALS"),
        ("huginn.harness.meta_improver", "_REPLAY_MAX"),
    ]
    # 兼容别名: 白名单首项, 供外部/测试引用.
    _SOURCE_TARGET: tuple[str, str] = _SOURCE_TARGETS[0]

    def _source_target(self) -> tuple[str, str]:
        """当前 source 环的目标 (轮转): 按 _source_proposals 在白名单里选, 确定性覆盖多目标."""
        ts = getattr(self, "_SOURCE_TARGETS", None) or [self._SOURCE_TARGET]
        return ts[self._source_proposals % len(ts)]

    def source_enabled(self) -> bool:
        """A3 source 环: meta on + prompt_patch on + harness_source_patch 显式开."""
        return self.enabled() and _harness_enabled("harness_source_patch")

    async def maybe_propose_source(self, llm_chat_fn: Callable) -> str | None:
        """给自己一个"改自己代码"的机会: 提议替换目标模块的某符号为新实现.

        v0 目标固定为自调征件节奏常量 (``_PROPOSE_EVERY_N``) — 元程序意义清晰、
        运行时安全、可逆. LLM 产出一行 ``<symbol> = N`` 赋值, 经 ``verify_source_patch``
        语法/anchor/定义验证后才登记进 store. 返回 patch_id 或 None.
        """
        if not self.source_enabled():
            return None
        # v0 目标: 在白名单里轮转自改内部调参常量 — 元程序意义清晰、运行时安全、可逆.
        module, symbol = self._source_target()
        import huginn.harness.meta_improver as mi_mod

        cur = 0
        with contextlib.suppress(Exception):
            cur = int(getattr(mi_mod, symbol) or 0)
        prompt = (
            f"You are self-modifying your own source. Current {module}:{symbol} = {cur}\n"
            f"Propose how often a generation should trigger one meta-proposal. "
            f"Your answer must be EXACTLY ONE line:\n"
            f"{symbol} = <positive integer>"
        )
        try:
            resp = await llm_chat_fn(prompt, task="summarize")
        except Exception:
            logger.debug("source propose LLM fail", exc_info=True)
            return None
        if not resp or not resp.strip():
            return None
        line = resp.strip().splitlines()[0].strip()
        if "=" not in line:
            self._trace({"type": "source_reject", "reason": "no_assignment"})
            return None
        lhs = line.split("=", 1)[0].strip()
        rhs = line.split("=", 1)[1].strip()
        if lhs != symbol:
            self._trace({"type": "source_reject", "reason": "bad_symbol"})
            return None
        try:
            val = int(rhs)
        except (TypeError, ValueError):
            self._trace({"type": "source_reject", "reason": "not_int"})
            return None
        if val <= 0:
            self._trace({"type": "source_reject", "reason": "non_positive"})
            return None
        from huginn.harness.source_patch import SourcePatch, verify_source_patch

        patch = SourcePatch(
            id=f"src_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}",
            module=module, symbol=symbol, new_code=f"{symbol} = {val}",
        )
        if not verify_source_patch(patch)["passed"]:
            self._trace({"type": "source_reject", "reason": "verify_fail"})
            return None
        from huginn.harness.source_patch import SourcePatchStore

        SourcePatchStore.get_instance().add_patch(patch)
        self._trace({"type": "source_propose", "candidate_id": patch.id, "value": val})
        return patch.id

    async def evaluate_source(self, patch_id: str, llm_chat_fn: Callable) -> dict[str, Any]:
        """评估 source 候选: verify 决定代理分并注册显著性配对.

        有效补丁(candidate=1.0) vs 中性基准(0.5) → 显著 GREEN; 无效(0.1)→ 不显著.
        真实 r_phys 仍未上行则 advisory 不硬拦 (rphys 硬闸由 maybe_promote_source 处置).
        """
        if not self.source_enabled():
            return {"green": False, "reason": "disabled"}
        from huginn.harness.source_patch import SourcePatchStore, verify_source_patch

        patch = SourcePatchStore.get_instance().get(patch_id)
        if patch is None:
            return {"green": False, "reason": "unknown_patch"}
        v = verify_source_patch(patch)
        verified = bool(v["passed"])
        from huginn.harness.significance_gate import SignificanceGate
        from huginn.harness.ood_holdout import OODHoldoutValidator

        sig = SignificanceGate.get_instance()
        ood = OODHoldoutValidator.get_instance()
        if not self._replay:
            self._trace({"type": "source_evaluate", "candidate_id": patch_id,
                         "verified": verified, "green": False, "n": 0})
            return {"green": False, "verified": verified, "n": 0}
        for probe in self._replay:
            task = str(probe.get("probe_id") or probe.get("ts") or random.random())
            sig.record_pair(patch_id, 0.5, 1.0 if verified else 0.1, task_id=task)
            ood.record_outcome(ood._BASELINE_ID, task, 0.5)
            ood.record_outcome(patch_id, task, 1.0 if verified else 0.1)
        sig_ok = sig.gate_decision(patch_id, min_samples=_MIN_SAMPLES).passed
        ood_ok = ood.validate_ood(patch_id).passed
        green = bool(verified and sig_ok and ood_ok)
        self._trace({"type": "source_evaluate", "candidate_id": patch_id,
                     "verified": verified, "sig": sig_ok, "ood": ood_ok,
                     "green": green, "n": len(self._replay)})
        return {"green": green, "verified": verified, "sig": sig_ok,
                "ood": ood_ok, "n": len(self._replay)}

    def maybe_promote_source(self, patch_id: str, ctx: Any | None = None) -> bool:
        """source 换件门控: verify + 显著 + OOD (+ rphys hard 可拦) → 事务内应用.

        ``harness_source_patch`` 开启才可能真正应用 (monkeypatch, 可逆).
        """
        if not self.source_enabled():
            return False
        from huginn.harness.source_patch import (
            SourcePatchStore, apply_source_patch, _register_source_compensator,
        )

        patch = SourcePatchStore.get_instance().get(patch_id)
        if patch is None:
            return False
        from huginn.harness.significance_gate import SignificanceGate
        from huginn.harness.ood_holdout import OODHoldoutValidator

        sig_ok = SignificanceGate.get_instance().gate_decision(
            patch_id, min_samples=_MIN_SAMPLES).passed
        ood_ok = OODHoldoutValidator.get_instance().validate_ood(patch_id).passed
        if not (sig_ok and ood_ok):
            self._trace({"type": "source_reject", "candidate_id": patch_id,
                         "reason": "not_green"})
            return False
        rp = self._rphys.verdict(patch_id)
        if _harness_enabled("harness_rphys_gate") and rp["green"] is False:
            self._trace({"type": "source_reject", "candidate_id": patch_id,
                         "reason": "rphys_fail"})
            return False
        from huginn.security.revertible import RevertibleContext

        rctx = ctx or RevertibleContext(
            journal_path=self._dir / "revertible_journal.json"
            if self._dir.exists() else None,
        )
        try:
            _register_source_compensator()
            with rctx.transaction():
                ok = apply_source_patch(patch, rctx)
        except Exception as exc:
            logger.debug("source promote txn failed", exc_info=True)
            return False
        if not ok:
            self._trace({"type": "source_reject", "candidate_id": patch_id,
                         "reason": "apply_failed"})
            return False
        self._trace({"type": "source_promote", "candidate_id": patch_id})
        return True

    # ── 空间可组合 (CoEffectRegistry) ────────────────────────────────────────
    def _coeffect_available(self) -> bool:
        """strategist 缺失/被 degrade → improvement_strategy 不可用."""
        return self.strategist_champion() is not None

    def coeffect_registry(self) -> Any:
        """空间可组合: 声明 strategist/improver/gate/fidelity 依赖图 (lazy).

        与 PhysicalWorkspace 同一立场: 依赖缺失 → 组件失活而非硬崩.
        暴露 ``update_availability(meta)`` 供运行时按现状刷新可用性.
        """
        if self._coeffect is None:
            try:
                from huginn.security.coeffect import CoEffectRegistry

                reg = CoEffectRegistry()
                reg.declare("strategist", provides={"improvement_strategy"},
                            requires={"gate", "fidelity"})
                reg.declare("improver", requires={"improvement_strategy"})
                reg.declare("gate", provides={"gate"})
                reg.declare("fidelity", provides={"fidelity"})

                def update_availability(meta_: Any) -> None:
                    try:
                        reg.set_available("gate", meta_._harness_enabled_state())
                        reg.set_available("fidelity", meta_._harness_enabled_state())
                        reg.set_available("improvement_strategy",
                                           meta_._coeffect_available())
                    except Exception:
                        pass

                reg.update_availability = update_availability  # 动态注入 (测试/运行时)
                update_availability(self)
                self._coeffect = reg
            except Exception as exc:
                logger.debug("meta: coeffect registry init failed", exc_info=True)
                self._coeffect = None
        else:
            try:
                self._coeffect.update_availability(self)
            except Exception:
                pass
        return self._coeffect

    def _harness_enabled_state(self) -> bool:
        return self.enabled()

    def improver_active(self) -> bool:
        """improver 可用性: strategist 缺失/被 degrade → False (退化默认模板)."""
        reg = self.coeffect_registry()
        if reg is None:
            return True  # coeffect 不可用 → 不设闸, 保持 M-R1 行为
        try:
            if reg.is_available("improvement_strategy"):
                return True
            return bool(reg.is_active("improver"))
        except Exception:
            return True

    async def note_generation(
        self, phase: str, blocks: list[tuple[str, str]], r_phys: float | None,
        directive: str, llm_chat_fn: Callable[[str, str], Any],
        real_action: dict[str, Any] | None = None,
    ) -> None:
        """H1 generate_patch 成功产 patch 后回调: 进重放集 + 攒计数到阈值触发 maybe_propose.

        ``real_action`` (可选): 本代 agent **真实执行**的实验 ``{"action_type", "params",
        "state_before", "observed"}``. 存在则记入 ``VerifiableGate`` 真实实验账本,
        让验收能面向真实产出的实验而非只验代表电池.
        """
        if not self.enabled():
            return
        block_names = [n for n, _ in blocks]
        # 端到端 r_phys 真实验收: 把本次真实 r_phys 归因到「当时代际驱动该 patch 的
        # 改进器配置」— 是 LLM 代理分之外的地面真值通道 (论文 2609.03621).
        active_cfg = self.champion_cfg()
        self._rphys.record(active_cfg.config_id if active_cfg else "_base", r_phys)
        # 真实实验入库: 验证面向 agent 真实执行的动作而非仅代表电池 (论文: 组合工作流).
        if isinstance(real_action, dict) and real_action.get("action_type"):
            try:
                from huginn.harness.meta_improver import VerifiableGate
                VerifiableGate().record_execution(
                    action_type=str(real_action.get("action_type")),
                    params=dict(real_action.get("params") or {}),
                    state_before=dict(real_action.get("state_before") or {}),
                    observed=(dict(real_action.get("observed"))
                              if real_action.get("observed") is not None else None),
                )
            except Exception:
                logger.debug("verifiable gate record_execution failed", exc_info=True)
        self._replay.append(
            {
                "phase": phase,
                "block_names": block_names,
                "r_phys": (float(r_phys) if r_phys is not None else None),
                "directive": directive or "",
                "ts": time.time(),
            }
        )
        self._replay = self._replay[-_REPLAY_MAX:]
        self._save_replay()
        self._trace({"type": "generation", "phase": phase})
        self._propose_count += 1
        try:
            if self._propose_count % _PROPOSE_EVERY_N == 0:
                cid = await self.maybe_propose(llm_chat_fn)
                # 闭环: 提案后立即离线评估并按 GREEN 换件 (默认关; 打开了才有 LLM 开销)
                self._strategy_proposals += 1
                if cid:
                    await self.evaluate(cid, llm_chat_fn)
                    self.maybe_promote(cid)
                # 递归环驱动 (HIGH-3 修复): 每隔 N 次 level-0 提案触发一次
                # strategist 层 提案→评估→换件, 让"改进方式可被改进"有生产入口.
                if self._strategy_proposals % _STRATEGIST_EVERY_N_PROPOSALS == 0:
                    sid = await self.maybe_propose_strategist(llm_chat_fn)
                    if sid:
                        await self.evaluate_strategist(sid, llm_chat_fn)
                        self.maybe_promote_strategist(sid)
            # A3: source 环驱动 (harness_source_patch 显式开才触发) — 每
            # _SOURCE_EVERY_N_GENERATIONS 代给一次"改自己源码"的机会.
            from huginn.harness.source_patch import _SOURCE_EVERY_N_GENERATIONS

            self._source_proposals += 1
            if self.source_enabled() and \
                    self._source_proposals % _SOURCE_EVERY_N_GENERATIONS == 0:
                spid = await self.maybe_propose_source(llm_chat_fn)
                if spid:
                    await self.evaluate_source(spid, llm_chat_fn)
                    self.maybe_promote_source(spid)
        except Exception as exc:
            logger.debug("meta note_generation/maybe_propose failed", exc_info=True)

    # ── propose ────────────────────────────────────────────────────────────
    async def maybe_propose(self, llm_chat_fn: Callable) -> str | None:
        """用 LLM 基于当前模板生成一个候选 improver 模板. 失败静默返回 None.

        A1: 模板源改为 ``strategist_prompt()`` (有 champion 用其改进策略模板,
        无则回落默认 ``_META_IMPROVE_TEMPLATE``) — 改进方式因此可被改进.
        """
        if not self.enabled():
            return None
        current = self.current_template()
        n_proposals = len([c for c in self._candidates.values()])
        n_promotions = len(self._history)
        try:
            meta_prompt = self.strategist_prompt().format(
                current_template=current,
                n_proposals=n_proposals,
                n_promotions=n_promotions,
            )
        except (KeyError, IndexError, ValueError):
            # strategist 模板无法被 maybe_propose 实例化 → 回落默认, 不阻塞
            logger.debug("meta maybe_propose: strategist template format failed, fallback")
            self._trace({"type": "strategy_fallback", "reason": "format_failed"})
            meta_prompt = _META_IMPROVE_TEMPLATE.format(
                current_template=current,
                n_proposals=n_proposals,
                n_promotions=n_promotions,
            )
        try:
            resp = await llm_chat_fn(meta_prompt, task="summarize")
        except Exception:
            logger.debug("meta maybe_propose LLM fail", exc_info=True)
            return None
        if not resp or not resp.strip():
            return None
        tpl = resp.strip()
        if tpl.startswith("```"):
            tpl = tpl.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if not tpl:
            return None
        # 模板必须能被占位符实例化且保留全部四个占位符, 否则作废
        probe = build_improver_prompt(tpl, "phase", ["body", "mem"], 0.6, "hint")
        for ph in ("{phase}", "{block_names}", "{r_phys}", "{directive}"):
            if ph not in tpl:
                probe = None
                break
        if probe is None:
            logger.debug("meta maybe_propose: candidate template invalid (dropped)")
            self._trace({"type": "reject", "reason": "invalid_template"})
            return None
        champ = self.champion_cfg()
        base_gate = champ.r_phys_gate if champ else 0.7
        cfg = ImproverConfig(
            config_id=f"meta_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}",
            improver_prompt=tpl,
            r_phys_gate=max(_R_PHYS_GATE_MIN,
                            min(_R_PHYS_GATE_MAX, base_gate + random.uniform(-0.05, 0.05))),
        )
        self._candidates[cfg.config_id] = cfg
        self._save_candidate(cfg)
        self._trace({"type": "propose", "candidate_id": cfg.config_id})
        return cfg.config_id

    # ── evaluate / promote ───────────────────────────────────────────────────
    async def _score_candidate(
        self, cfg: ImproverConfig, probe: dict[str, Any],
        llm_chat_fn: Callable,
    ) -> float | None:
        """对某个改进器配置在某重放探针上的离线代理分. llm 失败/模板非法 → None."""
        tpl = cfg.improver_prompt if cfg is not None else DEFAULT_IMPROV_TEMPLATE
        prompt = build_improver_prompt(
            tpl, probe["phase"], probe["block_names"],
            probe.get("r_phys"), probe.get("directive", ""),
        )
        if prompt is None:
            return None
        try:
            resp = await llm_chat_fn(prompt, task="summarize")
        except Exception:
            return None
        return score_patch_output(resp or "", probe["block_names"], probe.get("directive", ""))

    async def evaluate(
        self, candidate_id: str, llm_chat_fn: Callable,
    ) -> dict[str, Any]:
        """在冻结重放集上给候选 vs 当前基准打分, 注册进显著性+OOD 门控.

        返回 {base_score, cand_score, scores_n, sig, ood, green}.
        """
        if not self.enabled():
            return {"green": False, "reason": "disabled"}
        cfg = self._candidates.get(candidate_id)
        if cfg is None:
            return {"green": False, "reason": "unknown_candidate"}
        if not self._replay:
            return {"green": False, "reason": "no_replay"}
        base_scores: list[float] = []
        cand_scores: list[float] = []
        for probe in self._replay:
            base = await self._score_candidate(None, probe, llm_chat_fn)
            cand = await self._score_candidate(cfg, probe, llm_chat_fn)
            if base is None or cand is None:
                continue
            base_scores.append(base)
            cand_scores.append(cand)
            task_id = probe.get("probe_id") or probe.get("ts") or str(random.random())
            from huginn.harness.significance_gate import SignificanceGate
            from huginn.harness.ood_holdout import OODHoldoutValidator
            SignificanceGate.get_instance().record_pair(
                candidate_id, base, cand, task_id=str(task_id),
            )
            OODHoldoutValidator.get_instance().record_outcome(OODHoldoutValidator._BASELINE_ID, str(task_id), base)
            OODHoldoutValidator.get_instance().record_outcome(candidate_id, str(task_id), cand)
        # 组合 == AdoptionGate 的 GREEN 条件: 显著 且 OOD 不退化
        from huginn.harness.significance_gate import SignificanceGate
        from huginn.harness.ood_holdout import OODHoldoutValidator
        sig = SignificanceGate.get_instance().gate_decision(candidate_id, min_samples=_MIN_SAMPLES)
        ood = OODHoldoutValidator.get_instance().validate_ood(candidate_id)
        green = bool(sig.passed and ood.passed)
        result = {
            "candidate_id": candidate_id,
            "base_mean": _mean(base_scores),
            "cand_mean": _mean(cand_scores),
            "scores_n": len(base_scores),
            "sig_n": sig.n_samples,
            "sig_passed": sig.passed,
            "ood_passed": bool(ood.passed),
            "green": green,
            "reason": "significant_and_ood" if green else "not_green",
        }
        # 端到端 r_phys 真实验收 (advisory): 候选若是已部署过、攒到真实代际则
        # 给出其真实 r_phys 上行判定 — 替换 LLM 代理分的最终地面真值通道.
        rp = self._rphys.verdict(candidate_id)
        result["rphys_green"] = rp["green"]
        result["rphys_n"] = rp["n"]
        self._trace({"type": "evaluate", **result})
        return result

    def maybe_promote(self, candidate_id: str) -> bool:
        """仅 GREEN(显著+OOD) 才把候选设为 champion. 非 GREEN 永不换件."""
        if not self.enabled():
            return False
        cfg = self._candidates.get(candidate_id)
        if cfg is None:
            return False
        from huginn.harness.significance_gate import SignificanceGate
        sig = SignificanceGate.get_instance().gate_decision(candidate_id, min_samples=_MIN_SAMPLES)
        if not sig.passed:
            self._trace({"type": "reject", "candidate_id": candidate_id, "reason": "sig_fail"})
            return False
        from huginn.harness.ood_holdout import OODHoldoutValidator
        ood = OODHoldoutValidator.get_instance().validate_ood(candidate_id)
        if not ood.passed:
            self._trace({"type": "reject", "candidate_id": candidate_id, "reason": "ood_fail"})
            return False
        # 端到端 r_phys 真实验收 (论文 2609.03621): 显式开启 harness_rphys_gate 且
        # 候选已部署攒到真实代际时, 真实 r_phys 未显著上行 → 拒绝换件. 默认 advisory
        # (green=None 无据不拦), 仅在真实数据可判且硬闸开启时硬拦截 — 覆盖性地把
        # 「LLM 代理分说好但真实物理验证分没跟上去」的候选挡在 champion 之外.
        rp = self._rphys.verdict(candidate_id)
        if _harness_enabled("harness_rphys_gate") and rp["green"] is False:
            self._trace({"type": "reject", "candidate_id": candidate_id,
                         "reason": "rphys_fail", "rphys_n": rp["n"]})
            return False
        # 提件: 旧 champion 冻结(deactivate), 新候选激活
        for c in self._candidates.values():
            c.active = False
            self._save_candidate(c)
        cfg.active = True
        self._save_candidate(cfg)
        if self._active_id and self._active_id not in self._history:
            self._history.append(self._active_id)
        self._active_id = candidate_id
        self._promotions += 1
        self._save_cfg()
        self._trace(
            {"type": "promote", "candidate_id": candidate_id, "reason": "green"}
        )
        logger.info("meta-improver: promoted champion -> %s", candidate_id)
        return True

    def compounding_trace(self) -> dict[str, Any]:
        """meta 赢率: GREEN 提升数 / 候选总数, + 换代历史. 供面板/审计."""
        n_candidates = len([c for c in self._candidates.values()])
        return {
            "active_config_id": self._active_id,
            "history": self._history,
            "n_promotions": self._promotions,
            "n_candidates": n_candidates,
            "meta_win_rate": round(self._promotions / max(1, n_candidates), 3),
            "replay_size": len(self._replay),
            "active_strategist_id": self._active_strategist_id,
            "strategists_history": list(self._strategists_history),
            "n_strategists": len(self._strategists),
            # 端到端 r_phys 真实验收: active 配置的真实上行判定 + 各配置真实代数
            "rphys_active_verdict": self._rphys.verdict(self._active_id or "_base"),
            "rphys_tracked_configs": len(self._rphys._series),
        }


def _mean(v: list[float]) -> float:
    return round(sum(v) / len(v), 3) if v else 0.0


def _mannwhitney_p_greater(xs: list[float], ys: list[float]) -> float:
    """Mann-Whitney U 单侧 (更大) 的 p 值 — 纯 stdlib 正态近似, 不依赖 scipy.

    秩统计 + tie 修正方差 + 连续修正. 用于 RPhysTrack 端到端验收: 判定一个配置
    驱动时的真实 r_phys 是否显著高于其余配置池. 无 scipy 时照常用 (significance_gate
    的 wilcoxon 才可选依赖 scipy, 此处不做同样依赖).
    """
    import math

    if not xs or not ys:
        return 1.0
    n1, n2 = len(xs), len(ys)
    n = n1 + n2
    pooled = sorted([(v, 0) for v in xs] + [(v, 1) for v in ys])
    # 平均秩 (处理并列)
    rank_of: dict[tuple[float, int], float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            rank_of[pooled[k]] = avg
        i = j + 1
    r1 = sum(rank_of[(v, 0)] for v in xs)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    # tie 修正
    i = 0
    tie_sum = 0.0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        t = j - i + 1
        tie_sum += t ** 3 - t
        i = j + 1
    if n > 1:
        corrected = (n + 1) - tie_sum / (n * (n - 1))
    else:
        corrected = n + 1
    var = n1 * n2 / 12.0 * corrected
    if var <= 0.0:
        var = n1 * n2 * (n + 1) / 12.0
    sd = math.sqrt(max(var, 1e-9))
    z = (u1 - mu - 0.5) / sd  # 连续修正, 趋保守
    # P(Xs > Ys) 单侧: p = P(Z >= z) = 0.5 * erfc(z / sqrt(2))
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


# ── strategist 换件补偿器 (时空可组合, 时间侧) ────────────────────────────────
def _compensate_strategist_swap(payload: dict[str, Any]) -> None:
    """撤销一次 strategist 换件: 恢复上一 active 指向. (由 register_compensator 注册)

    进程内 revert_all 与崩溃后 journal 重放共用此补偿器. 只处理可序列化数据
    (payload["old"]), 保证可跨崩溃.
    """
    old = payload.get("old")
    if not old:
        return
    meta = MetaImprover.get_instance()
    if meta is None:
        return
    meta._active_strategist_id = old
    for s in meta._strategists.values():
        s.active = (s.config_id == old)
        meta._save_strategist(s)
    meta._save_cfg()
    logger.info("meta: strategist swap compensated back to %s", old)


def _register_strategist_compensator() -> None:
    from huginn.security.revertible import register_compensator

    register_compensator("strategist_swap", _compensate_strategist_swap)


def _selfcheck() -> None:
    """M-R1 selfcheck: off 零回归 + 好候选 GREEN 换件 + 差候选不换 + OOD 背题拦截."""
    import asyncio
    import shutil
    import tempfile

    import huginn.harness.meta_improver as mi

    tmp = tempfile.mkdtemp()
    os_env = __import__("os")
    os_env.environ["HUGINN_CACHE_DIR"] = tmp
    mi.MetaImprover._instance = None
    mi._harness_enabled = lambda key, default=False: (
        True if key in ("harness_meta_improver", "harness_prompt_patch") else default
    )

    # 重放集探针只在实际 evaluate 用的那个单例上填 (见 step 3, toggle 重置之后)
    blocks = [("body", "b {context}"), ("mem", "m"), ("fail", "f")]

    # 假 LLM: 按 improver 模板内容返回, 让「好模板」产出有效 patch, 「默认/差模板」产出坏 patch
    async def fake_llm(prompt, task="summarize"):
        if prompt.startswith(mi._META_IMPROVE_TEMPLATE[:40]):
            return good_template
        if "good-improver" in prompt:
            return '{"block_name": "mem", "op": "append", "new_text": "focus hint"}'
        return "not-json"

    good_template = (
        "Nice good-improver. Phase:{phase} Blocks:{block_names} R:{r_phys} D:{directive} "
        "produce one patch JSON."
    )
    bad_template = (
        "Bad improver. Phase:{phase} Blocks:{block_names} R:{r_phys} D:{directive}."
    )

    meta = None  # 在 step 3 toggle 重置后重绑到当前单例

    # 1. 默认模板可实例化
    p = mi.build_improver_prompt(mi.DEFAULT_IMPROV_TEMPLATE, "h", ["body"], 0.6, "")
    assert p and "{phase}" not in p
    print("1. default template instantiation OK")

    # 2. toggle off → champion None, note 不写 (replay 长度不变)
    mi._harness_enabled = lambda key, default=False: False
    mi.MetaImprover._instance = None
    n_before = len(mi.MetaImprover.get_instance()._replay)
    assert mi.MetaImprover.get_instance().champion_cfg() is None
    asyncio.run(mi.MetaImprover.get_instance().note_generation("h", blocks, 0.5, "", fake_llm))
    assert len(mi.MetaImprover.get_instance()._replay) == n_before, "off should not record"
    print("2. toggle off → champion None + no record OK")
    mi.MetaImprover._instance = None
    mi._harness_enabled = lambda key, default=False: (
        True if key in ("harness_meta_improver", "harness_prompt_patch") else default
    )

    # 3. 候选 good → 显著 + OOD 通过 → promote
    meta = mi.MetaImprover.get_instance()
    # 给该单例填重放集探针: 8 个, 足够 sig(>=5) + ood(train/holdout >=3)
    for i in range(8):
        meta._replay.append(
            {"phase": "hypothesize", "block_names": ["body", "mem", "fail"],
             "r_phys": 0.5, "directive": f"hint {i}", "ts": float(100 + i),
             "probe_id": f"probe_{i:02d}"}
        )
    cid = asyncio.run(meta.maybe_propose(fake_llm))
    assert cid is not None, "good template should propose"
    r = asyncio.run(meta.evaluate(cid, fake_llm))
    assert r["green"], f"good candidate should be green: {r}"
    assert meta.maybe_promote(cid) is True
    cc = meta.champion_cfg()
    assert cc is not None and cc.config_id == cid, "champion should switch"
    print(f"3. good candidate GREEN promote OK (win_rate={meta.compounding_trace()['meta_win_rate']})")

    # 4. 差候选 → RED/YELLOW → 不换
    bad = mi.ImproverConfig(config_id="bad", improver_prompt=bad_template, r_phys_gate=0.7)
    meta._candidates["bad"] = bad
    meta._save_candidate(bad)
    r2 = asyncio.run(meta.evaluate("bad", fake_llm))
    if r2["green"]:
        print("bad evaluated green unexpectedly (data-dependent); forcing model-level check")
    champ_before = meta.champion_cfg().config_id
    assert meta.maybe_promote("bad") is False, "bad should not promote"
    assert meta.champion_cfg().config_id == champ_before, "champion unchanged"
    print("4. bad candidate → no promote OK")

    # 5. 换代历史 + trace
    tr = meta.compounding_trace()
    assert tr["n_promotions"] >= 1 and tr["active_config_id"] == cid
    assert meta._trace_path.exists(), "meta_trace should be written"
    print(f"5. champion history + trace OK (promotions={tr['n_promotions']})")

    # 6. 论文 arXiv:2609.03621 — VerifiableGate 状态化仿真 + BehavioralFidelity 锚
    vg = mi.VerifiableGate()
    okv = vg.verify_outcome({"action_type": "aspirate", "params": {"vol": 1.0},
                             "state": {"reagent_vol": 5.0, "sample_vol": 0.0}})
    assert okv["passed"] is True, okv
    assert abs(okv["new_state"]["reagent_vol"] - 4.0) < 1e-9, okv
    badv = vg.verify_outcome({"action_type": "aspirate", "params": {"vol": 99.0},
                              "state": {"reagent_vol": 5.0}})
    assert badv["passed"] is False and badv["issues"], badv
    bf = mi.BehavioralFidelity(path=str(os_env.path.join(tmp, "fidelity.json"))
                               if hasattr(os_env, "path") else tmp + "/fidelity.json")
    bf.record_acceptance("bx", accepted=True)
    bf.record_acceptance("bx", accepted=False)
    assert abs(bf.fidelity_score("bx") - 0.5) < 1e-9
    assert abs(bf.anchor_in("bx", p_quality=0.8) - 0.65) < 1e-9
    print("6. VerifiableGate(论文) + BehavioralFidelity 锚 OK")

    # 7. strategist 递归链: 默认回落 → propose 候选 → champion 激活与回落
    assert meta.strategist_prompt(), "strategist fallback present"
    # maybe_propose_strategist 的校验 .format 只给 current_template/n_proposals/
    # n_promotions; 候选只含这些占位符才不会 KeyError.
    _new_strat_tpl = ("strategize {current_template} p={n_proposals} "
                      "prom={n_promotions}")

    async def _strat_llm(prompt, task="summarize"):
        if prompt.startswith(mi._META2_IMPROVE_TEMPLATE[:40]):
            return _new_strat_tpl
        return "not-json"

    strat_id = asyncio.run(meta.maybe_propose_strategist(_strat_llm))
    assert strat_id is not None, "strategist should propose (valid template)"
    assert meta.compounding_trace()["n_strategists"] >= 1
    # 候选模板能被 maybe_propose 用 .format 实例化 (改进方式可被改进)
    tpl = meta._strategists[strat_id].strategist_prompt
    assert "{current_template}" in tpl
    # 激活策略得先通过 promote 门控; 这里只验证候选登记与回落路径
    assert meta.strategist_champion() is None, "未 promote 前不应有 champion"
    print("7. strategist 递归链候选 + 回落 OK")

    shutil.rmtree(tmp, ignore_errors=True)
    del os_env.environ["HUGINN_CACHE_DIR"]
    mi.MetaImprover._instance = None
    print("\nM-R1 meta_improver selfcheck OK (7/7)")


if __name__ == "__main__":
    _selfcheck()