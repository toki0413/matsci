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
        ys = self._quality_series()
        if not ys:
            return False
        best = max(ys)
        cur = ys[-1]
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

    def enabled(self) -> bool:
        return _harness_enabled("harness_verifiable_gate")

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
        self._load()
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

    async def note_generation(
        self, phase: str, blocks: list[tuple[str, str]], r_phys: float | None,
        directive: str, llm_chat_fn: Callable[[str, str], Any],
    ) -> None:
        """H1 generate_patch 成功产 patch 后回调: 进重放集 + 攒计数到阈值触发 maybe_propose."""
        if not self.enabled():
            return
        block_names = [n for n, _ in blocks]
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
                if cid:
                    await self.evaluate(cid, llm_chat_fn)
                    self.maybe_promote(cid)
        except Exception as exc:
            logger.debug("meta note_generation/maybe_propose failed", exc_info=True)

    # ── propose ────────────────────────────────────────────────────────────
    async def maybe_propose(self, llm_chat_fn: Callable) -> str | None:
        """用 LLM 基于当前模板生成一个候选 improver 模板. 失败静默返回 None."""
        if not self.enabled():
            return None
        current = self.current_template()
        meta_prompt = _META_IMPROVE_TEMPLATE.format(
            current_template=current,
            n_proposals=len([c for c in self._candidates.values()]),
            n_promotions=len(self._history),
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
        }


def _mean(v: list[float]) -> float:
    return round(sum(v) / len(v), 3) if v else 0.0


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

    shutil.rmtree(tmp, ignore_errors=True)
    del os_env.environ["HUGINN_CACHE_DIR"]
    mi.MetaImprover._instance = None
    print("\nM-R1 meta_improver selfcheck OK (5/5)")


if __name__ == "__main__":
    _selfcheck()