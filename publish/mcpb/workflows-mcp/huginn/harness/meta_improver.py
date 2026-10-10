"""M-R1: Recursive (Meta-)Improver — 让「改进器如何改进」也是可改进、可验收的对象.

单层改进器现状 (H1 prompt_patch):
  generate_patch(phase, blocks, r_phys, directive, llm_chat_fn) 把一份 improver
  prompt 喂给 LLM, LLM 产出一个 prompt block patch. 这份 improver prompt 若写死在
  函数里就不可被改进 → 单层、非递归.

本模块加一层 meta-improver (STOP 退化单步版):
  - 把「改进器 prompt 模板」提升为一等对象 ImproverConfig (champion active 才覆盖默认),
  - maybe_propose: LLM 基于「当前 improver 模板 + meta 统计」改写新模板 → 候选,
  - select_generation_arm: 每轮产 patch 时选臂 — champion 在位走 champion (exploit);
    无 champion 时以 _CANARY_P 概率让候选接管 (canary 探索臂), 否则 baseline,
  - record_real_outcome: 真实迭代跑完把真实 r_phys 按「生成该 patch 的臂」回填进 ledger,
  - evaluate: 用回填的真实 r_phys 注册进 SignificanceGate + OODHoldout (不再用离线
    格式代理分),
  - promote: 仅 显著 + OOD 不退化 (即 AdoptionGate 的 GREEN) 才把候选设为 champion,
  - champion 换代旧配置冻结保留可回退; 门控永不删数据.

验收信号 (真实 r_phys, 诚实声明): evaluate 的分数是真实迭代的 r_phys, 按生成该 patch
的臂 (champion / canary 候选 / baseline) 归因. 真实 A/B 无法在同一 task 上同时跑两臂,
因此用确定性分桶把「同类任务」配对 (分桶思路与 OODHoldout 一致). 样本不足时不判定、
不换件 — 宁可保持默认也不靠噪声换件.

canary 的 explore 影响: 候选模板以 _CANARY_P 概率接管 patch 生成, 但其产出的 patch 仍需
过 Beta 门 (α>=β, 即"无证据时可试一次") 才真正 apply, 因此对 explore 的扰动被 Beta 闸兜住.

toggle: cfg.feature_flags.harness_meta_improver AND harness_prompt_patch 同时 on 才生效
(默认全 off, 关闭时零行为变更 — generate_patch 回落默认模板).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import random
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from huginn.utils.runtime import get_runtime_home

from ._enabled import _harness_enabled

logger = logging.getLogger(__name__)

# 每成功生成多少 patch 触发一次 maybe_propose.
_PROPOSE_EVERY_N = 5
# 显著性验收最小样本量 (和 SignificanceGate 默认同步).
_MIN_SAMPLES = 5
# r_phys 阈值扰动约束.
_R_PHYS_GATE_MIN, _R_PHYS_GATE_MAX = 0.5, 0.8
# baseline 臂: 未覆盖默认模板时生成 patch 的臂 id.
_BASELINE_ID = "__baseline__"
# canary 探索概率: 无 champion 时让候选接管 patch 生成的比例.
_CANARY_P = 0.3
# 真实 r_phys ledger 容量 (环形, 超了丢最老).
_REAL_OUTCOME_MAX = 500
# 真实 r_phys 配对分桶数. 桶 = "同类任务"; OOD 的 train/holdout 切分在同一组桶键上做,
# 16 桶下 holdout ~5 个, 满足 OOD min_per_split=3 且 sig 需要 >=5 对.
_PAIR_BUCKETS = 16

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


def _bucket_of(task_id: str, n_buckets: int = _PAIR_BUCKETS) -> str:
    """确定性任务分桶: 同一 task_id 永远落同一桶 (与 OODHoldout 分桶同思路).

    真实 A/B 无法在同一 task 上跑两臂, 用桶把"同类任务"配对 — 桶键即配对键.
    """
    h = int(
        hashlib.md5(task_id.encode("utf-8"), usedforsecurity=False).hexdigest()[:8],
        16,
    )
    return f"bucket_{h % n_buckets:02d}"


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
        self._arm_ledger_path = self._dir / "arm_ledger.json"
        self._real_outcomes_path = self._dir / "real_outcomes.json"
        self._trace_path = self._dir / "meta_trace.jsonl"
        self._active_id: str | None = None
        self._history: list[str] = []
        self._promotions: int = 0
        self._candidates: dict[str, ImproverConfig] = {}
        # patch_id → 生成该 patch 的臂 (champion config_id / 候选 id / _BASELINE_ID)
        self._arm_by_patch: dict[str, str] = {}
        # 真实迭代的 (arm, task, r_phys) 观测, 环形
        self._real_outcomes: list[dict[str, Any]] = []
        self._propose_count = 0
        self._load()

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
        with contextlib.suppress(Exception):
            if self._arm_ledger_path.exists():
                self._arm_by_patch = json.loads(
                    self._arm_ledger_path.read_text(encoding="utf-8")
                )
        with contextlib.suppress(Exception):
            if self._real_outcomes_path.exists():
                self._real_outcomes = json.loads(
                    self._real_outcomes_path.read_text(encoding="utf-8")
                )[-_REAL_OUTCOME_MAX:]
        with contextlib.suppress(Exception):
            for f in self._candidates_dir.glob("*.json"):
                try:
                    c = ImproverConfig.from_dict(json.loads(f.read_text(encoding="utf-8")))
                    self._candidates[c.config_id] = c
                except Exception:
                    logger.debug("meta candidate load fail: %s", f, exc_info=True)

    def _save_cfg(self) -> None:
        with contextlib.suppress(Exception):
            self._cfg_path.write_text(
                json.dumps(
                    {
                        "active_config_id": self._active_id,
                        "history": self._history,
                        "promotions": self._promotions,
                    },
                    ensure_ascii=False, indent=2,
                ), encoding="utf-8"
            )

    def _save_arm_ledger(self) -> None:
        with contextlib.suppress(Exception):
            self._arm_ledger_path.write_text(
                json.dumps(self._arm_by_patch, ensure_ascii=False), encoding="utf-8"
            )

    def _save_real_outcomes(self) -> None:
        with contextlib.suppress(Exception):
            self._real_outcomes_path.write_text(
                json.dumps(self._real_outcomes, ensure_ascii=False), encoding="utf-8"
            )

    def _save_candidate(self, cfg: ImproverConfig) -> None:
        with contextlib.suppress(Exception):
            (self._candidates_dir / f"{cfg.config_id}.json").write_text(
                json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _trace(self, entry: dict[str, Any]) -> None:
        with contextlib.suppress(Exception), self._trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**entry, "ts": time.time()}, ensure_ascii=False) + "\n")

    # ── enabled ──────────────────────────────────────────────────────────────
    def enabled(self) -> bool:
        """meta 层仅当两个开关都 on 才生效. 默认全 off."""
        return _harness_enabled("harness_meta_improver") and _harness_enabled(
            "harness_prompt_patch"
        )

    # ── champion / arm selection ─────────────────────────────────────────────
    def champion_cfg(self) -> ImproverConfig | None:
        """当前活跃 champion (config_id + active=True). 无则 None → 回落默认模板."""
        if not self.enabled():
            return None
        c = self._candidates.get(self._active_id or "")
        if c is not None and c.active:
            return c
        return None

    def candidate_ids(self) -> list[str]:
        """全部候选 config_id. 供回填后逐个 evaluate (不含 champion 特判)."""
        return list(self._candidates.keys())

    def current_template(self) -> str:
        champ = self.champion_cfg()
        return champ.improver_prompt if champ else DEFAULT_IMPROV_TEMPLATE

    def select_generation_arm(self) -> tuple[str, str]:
        """选本轮生成 patch 的臂, 返回 (arm_id, template).

        - champion 在位 → champion 臂 (exploit).
        - 无 champion 且有候选 → 以 _CANARY_P 概率让最近提出的候选接管 (canary 探索臂),
          否则 baseline 臂. 这样候选能在真实迭代里产 patch, 拿到真实 r_phys 归因.
        - 未开启/无候选 → baseline 臂 + 默认模板 (零行为变更).
        """
        if not self.enabled():
            return _BASELINE_ID, DEFAULT_IMPROV_TEMPLATE
        champ = self.champion_cfg()
        if champ is not None:
            return champ.config_id, champ.improver_prompt
        challengers = [c for c in self._candidates.values() if not c.active]
        if challengers and random.random() < _CANARY_P:
            cand = max(challengers, key=lambda c: c.created_at)
            return cand.config_id, cand.improver_prompt
        return _BASELINE_ID, DEFAULT_IMPROV_TEMPLATE

    # ── real r_phys ledger ───────────────────────────────────────────────────
    def record_patch_arm(self, patch_id: str, arm_id: str) -> None:
        """记 patch → 生成臂. 供回填时把真实 r_phys 归因到臂."""
        self._arm_by_patch[patch_id] = arm_id
        if len(self._arm_by_patch) > _REAL_OUTCOME_MAX:
            for k in list(self._arm_by_patch)[: len(self._arm_by_patch) - _REAL_OUTCOME_MAX]:
                self._arm_by_patch.pop(k, None)
        self._save_arm_ledger()

    def arm_for_patch(self, patch_id: str) -> str | None:
        return self._arm_by_patch.get(patch_id)

    def record_real_outcome(self, arm_id: str, task_id: str, r_phys: float) -> None:
        """把一次真实迭代的 r_phys 回填到某个臂. 未开启时 no-op."""
        if not self.enabled():
            return
        self._real_outcomes.append(
            {
                "arm_id": arm_id,
                "task_id": str(task_id),
                "r_phys": float(r_phys),
                "ts": time.time(),
            }
        )
        self._real_outcomes = self._real_outcomes[-_REAL_OUTCOME_MAX:]
        self._save_real_outcomes()
        self._trace(
            {"type": "real_outcome", "arm_id": arm_id, "r_phys": float(r_phys)}
        )

    def _real_observations(
        self, candidate_id: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """取 baseline 臂与指定候选臂的真实观测 (candidate 臂即候选 config_id)."""
        base = [o for o in self._real_outcomes if o.get("arm_id") == _BASELINE_ID]
        cand = [o for o in self._real_outcomes if o.get("arm_id") == candidate_id]
        return base, cand

    def _bucket_pairs(
        self, base: list[dict[str, Any]], cand: list[dict[str, Any]]
    ) -> list[tuple[str, float, float]]:
        """按任务桶配对: 同桶内 baseline 与候选的均值成对 (桶键即配对键)."""
        base_by: dict[str, list[float]] = {}
        cand_by: dict[str, list[float]] = {}
        for o in base:
            base_by.setdefault(_bucket_of(o["task_id"]), []).append(float(o["r_phys"]))
        for o in cand:
            cand_by.setdefault(_bucket_of(o["task_id"]), []).append(float(o["r_phys"]))
        return [
            (b, _mean(base_by[b]), _mean(cand_by[b]))
            for b in sorted(set(base_by) & set(cand_by))
        ]

    async def note_generation(
        self, phase: str, blocks: list[tuple[str, str]], r_phys: float | None,
        directive: str, llm_chat_fn: Callable[[str, str], Any],
        patch_id: str | None = None, arm_id: str | None = None,
    ) -> None:
        """H1 generate_patch 成功产 patch 后回调: 记 patch→臂 + 攒计数触发 maybe_propose."""
        if not self.enabled():
            return
        if patch_id is not None and arm_id is not None:
            self.record_patch_arm(patch_id, arm_id)
        self._trace({"type": "generation", "phase": phase, "arm_id": arm_id})
        self._propose_count += 1
        try:
            if self._propose_count % _PROPOSE_EVERY_N == 0:
                await self.maybe_propose(llm_chat_fn)
                # 不在此处 evaluate/promote: 换件要等真实 r_phys 回填攒够样本
                # (engine_reflect 每轮回填后调 evaluate + maybe_promote).
        except Exception:
            logger.debug("meta note_generation/maybe_propose failed", exc_info=True)

    # ── propose ────────────────────────────────────────────────────────────
    async def maybe_propose(self, llm_chat_fn: Callable) -> str | None:
        """用 LLM 基于当前模板生成一个候选 improver 模板. 失败静默返回 None."""
        if not self.enabled():
            return None
        current = self.current_template()
        meta_prompt = _META_IMPROVE_TEMPLATE.format(
            current_template=current,
            n_proposals=len(list(self._candidates.values())),
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
    async def evaluate(self, candidate_id: str) -> dict[str, Any]:
        """用回填的真实 r_phys 给候选 vs baseline 打分, 注册进显著性+OOD 门控.

        真实 A/B 无法在同一 task 上同时跑两臂, 故按任务桶配对 (见 _bucket_pairs).
        样本不足 → 不判定、不换件 (宁可保持默认).

        返回 {score_source, pairs_n, base_mean, cand_mean, sig_passed, ood_passed, green}.
        """
        if not self.enabled():
            return {"green": False, "reason": "disabled", "score_source": "real_r_phys"}
        cfg = self._candidates.get(candidate_id)
        if cfg is None:
            return {
                "green": False,
                "reason": "unknown_candidate",
                "score_source": "real_r_phys",
            }
        base_obs, cand_obs = self._real_observations(candidate_id)
        pairs = self._bucket_pairs(base_obs, cand_obs)
        if len(pairs) < _MIN_SAMPLES:
            result = {
                "candidate_id": candidate_id,
                "score_source": "real_r_phys",
                "pairs_n": len(pairs),
                "green": False,
                "reason": f"insufficient_real_outcomes: {len(pairs)}/{_MIN_SAMPLES}",
            }
            self._trace({"type": "evaluate", **result})
            return result
        from huginn.harness.ood_holdout import OODHoldoutValidator
        from huginn.harness.significance_gate import SignificanceGate
        sig_gate = SignificanceGate.get_instance()
        ood_gate = OODHoldoutValidator.get_instance()
        # ledger 是唯一真源: 每次 evaluate 幂等重建 derived 视图 (清候选自身 + baseline),
        # 否则每轮重复 record 会膨胀样本量, 让 Wilcoxon 假性显著 / OOD 记录无限增长.
        sig_gate.clear(candidate_id)
        ood_gate.clear(candidate_id)
        ood_gate.clear(ood_gate._BASELINE_ID)
        # baseline OOD 用 ledger 全部 baseline 观测 (与候选无关) → 重建幂等.
        # 桶键与候选一致, 保证 train/holdout 切分在同一任务集上做.
        for o in base_obs:
            ood_gate.record_outcome(
                ood_gate._BASELINE_ID, _bucket_of(o["task_id"]), float(o["r_phys"])
            )
        base_scores: list[float] = []
        cand_scores: list[float] = []
        for key, b, c in pairs:
            sig_gate.record_pair(candidate_id, b, c, task_id=key)
            ood_gate.record_outcome(candidate_id, key, c)
            base_scores.append(b)
            cand_scores.append(c)
        sig = sig_gate.gate_decision(candidate_id, min_samples=_MIN_SAMPLES)
        ood = ood_gate.validate_ood(candidate_id)
        green = bool(sig.passed and ood.passed)
        result = {
            "candidate_id": candidate_id,
            "score_source": "real_r_phys",
            "pairs_n": len(pairs),
            "base_mean": _mean(base_scores),
            "cand_mean": _mean(cand_scores),
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
        n_candidates = len(list(self._candidates.values()))
        return {
            "active_config_id": self._active_id,
            "history": self._history,
            "n_promotions": self._promotions,
            "n_candidates": n_candidates,
            "meta_win_rate": round(self._promotions / max(1, n_candidates), 3),
            "real_outcomes_n": len(self._real_outcomes),
        }


def _mean(v: list[float]) -> float:
    return round(sum(v) / len(v), 3) if v else 0.0


def _selfcheck() -> None:
    """M-R1 selfcheck: off 零回归 + canary 选臂 + 真实 r_phys 换件 + 差候选不换."""
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

    blocks = [("body", "b {context}"), ("mem", "m"), ("fail", "f")]

    async def fake_llm(prompt, task="summarize"):
        return "You are the improver. Phase:{phase} Blocks:{block_names} R:{r_phys} D:{directive}"

    good_template = (
        "Nice good-improver. Phase:{phase} Blocks:{block_names} R:{r_phys} D:{directive} "
        "produce one patch JSON."
    )
    bad_template = (
        "Bad improver. Phase:{phase} Blocks:{block_names} R:{r_phys} D:{directive}."
    )

    def seed_real(meta: mi.MetaImprover, candidate_id: str, base: float, cand: float) -> None:
        """造真实回填数据: 16 桶都有两臂 → sig 配对 >=5 + OOD train/holdout 够量."""
        for i in range(_PAIR_BUCKETS):
            task = f"task_{i:03d}"
            meta.record_real_outcome(mi._BASELINE_ID, task, base)
            meta.record_real_outcome(candidate_id, task, cand)

    # 1. 默认模板可实例化 + bucket 确定性
    p = mi.build_improver_prompt(
        mi.DEFAULT_IMPROV_TEMPLATE, "h", [n for n, _ in blocks], 0.6, ""
    )
    assert p and "{phase}" not in p
    assert mi._bucket_of("t1") == mi._bucket_of("t1")
    print("1. default template + bucket determinism OK")

    # 2. toggle off → champion None + select_generation_arm 回落 baseline + 不记 outcome
    mi._harness_enabled = lambda key, default=False: False
    mi.MetaImprover._instance = None
    m_off = mi.MetaImprover.get_instance()
    assert m_off.champion_cfg() is None
    arm, tpl = m_off.select_generation_arm()
    assert arm == mi._BASELINE_ID and tpl == mi.DEFAULT_IMPROV_TEMPLATE, (arm, tpl)
    m_off.record_real_outcome(mi._BASELINE_ID, "t", 0.9)
    assert len(m_off._real_outcomes) == 0, "off should not record real outcomes"
    print("2. toggle off → baseline arm + no record OK")
    mi.MetaImprover._instance = None
    mi._harness_enabled = lambda key, default=False: (
        True if key in ("harness_meta_improver", "harness_prompt_patch") else default
    )

    # 3. canary 选臂: 有候选时以 _CANARY_P 概率接管, 否则 baseline
    meta = mi.MetaImprover.get_instance()
    meta._candidates["cand_a"] = mi.ImproverConfig(
        config_id="cand_a", improver_prompt=good_template, r_phys_gate=0.7
    )
    _orig_random = mi.random.random
    mi.random.random = lambda: 0.0  # < _CANARY_P → 命中 canary
    arm_c, tpl_c = meta.select_generation_arm()
    mi.random.random = lambda: 0.99  # > _CANARY_P → baseline
    arm_b, _ = meta.select_generation_arm()
    mi.random.random = _orig_random
    assert arm_c == "cand_a" and "good-improver" in tpl_c, (arm_c, tpl_c)
    assert arm_b == mi._BASELINE_ID, arm_b
    print("3. canary arm selection OK")

    # 4. patch → 臂记账 + 真实 r_phys 回填 → 显著 + OOD → promote
    cid = asyncio.run(meta.maybe_propose(fake_llm))
    assert cid is not None, "template should propose"
    meta.record_patch_arm("patch_x", cid)
    assert meta.arm_for_patch("patch_x") == cid
    r0 = asyncio.run(meta.evaluate(cid))
    assert not r0["green"] and r0["reason"].startswith("insufficient_real_outcomes"), r0
    seed_real(meta, cid, base=0.3, cand=0.9)
    r = asyncio.run(meta.evaluate(cid))
    assert r["score_source"] == "real_r_phys" and r["green"], r
    assert meta.maybe_promote(cid) is True
    cc = meta.champion_cfg()
    assert cc is not None and cc.config_id == cid, "champion should switch"
    print(f"4. real r_phys GREEN promote OK (pairs={r['pairs_n']})")

    # 5. 差候选 → 不换 + 换代历史/trace
    bad = mi.ImproverConfig(config_id="bad", improver_prompt=bad_template, r_phys_gate=0.7)
    meta._candidates["bad"] = bad
    meta._save_candidate(bad)
    seed_real(meta, "bad", base=0.9, cand=0.2)  # 候选显著更差
    asyncio.run(meta.evaluate("bad"))
    champ_before = meta.champion_cfg().config_id
    assert meta.maybe_promote("bad") is False, "bad should not promote"
    assert meta.champion_cfg().config_id == champ_before, "champion unchanged"
    tr = meta.compounding_trace()
    assert tr["n_promotions"] >= 1 and tr["active_config_id"] == cid
    assert tr["real_outcomes_n"] > 0
    assert meta._trace_path.exists(), "meta_trace should be written"
    print(f"5. bad no-promote + history OK (promotions={tr['n_promotions']})")

    # 6. evaluate 幂等: 重复跑不膨胀 sig 样本 (否则 Wilcoxon 假性显著)
    from huginn.harness.significance_gate import SignificanceGate

    _n1 = len(SignificanceGate.get_instance().get_pairs(cid))
    asyncio.run(meta.evaluate(cid))
    asyncio.run(meta.evaluate(cid))
    _n2 = len(SignificanceGate.get_instance().get_pairs(cid))
    assert _n1 == _n2 >= _MIN_SAMPLES, f"evaluate not idempotent: {_n1} -> {_n2}"
    print(f"6. evaluate idempotent OK (pairs stay {_n2})")

    shutil.rmtree(tmp, ignore_errors=True)
    del os_env.environ["HUGINN_CACHE_DIR"]
    mi.MetaImprover._instance = None
    print("\nM-R1 meta_improver selfcheck OK (6/6)")


if __name__ == "__main__":
    _selfcheck()
