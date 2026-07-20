"""EvolutionManager — 统一 evolution 接口.

P14 spec: 把 4 处失败记录合并到统一 record_outcome / distill / recommend API.
借鉴 EvoScientist EMA (Evolution Manager Agent).

3 个核心 API:
- record_outcome(hypothesis, plan, validation, persona_id, run_id): 单步 outcome
- distill(run_id=None): 蒸馏 STABLE_PRINCIPLE
- recommend(hypothesis_context) -> Recommendation: 推荐

HUGINN_USE_EVOLUTION_MANAGER=1 启用. 默认 off, 走原分散路径.

ponytail: 单例 (跟 SkillEvolutionLayer.shared() 风格一致), 不新建引擎.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_FLAG = "HUGINN_USE_EVOLUTION_MANAGER"


def _use_evolution_manager() -> bool:
    return os.environ.get(_FLAG, "0") == "1"


@dataclass
class Recommendation:
    """recommend() 返回值."""

    avoid_directions: list[str] = field(default_factory=list)
    prefer_strategies: list[str] = field(default_factory=list)
    rationale: str = ""


class EvolutionManager:
    """统一 evolution 接口. 单例."""

    _shared: "EvolutionManager | None" = None

    def __init__(self, memory_manager=None, skill_evolution=None) -> None:
        self._mm = memory_manager
        self._se = skill_evolution
        self._failed_store = None
        if memory_manager is not None:
            from huginn.evolution.failed_direction_store import FailedDirectionStore

            self._failed_store = FailedDirectionStore(memory_manager)

    @classmethod
    def shared(cls, memory_manager=None) -> "EvolutionManager":
        """单例. 第一次调时传 memory_manager 初始化.

        后续调用不传 mm 也能拿到已初始化的实例; 若首次没传 mm, 后续调用
        可以补传, 内部把 _failed_store 补上.
        """
        if cls._shared is None:
            cls._shared = cls(memory_manager=memory_manager)
        elif memory_manager is not None and cls._shared._mm is None:
            cls._shared._mm = memory_manager
            from huginn.evolution.failed_direction_store import FailedDirectionStore

            cls._shared._failed_store = FailedDirectionStore(memory_manager)
        return cls._shared

    @classmethod
    def _reset_for_test(cls) -> None:
        """测试用: 重置单例. 生产代码别调."""
        cls._shared = None

    def record_outcome(
        self,
        hypothesis: str,
        plan: dict | None,
        validation: dict | None,
        persona_id: str | None,
        run_id: str,
        math_concept: str = "",
    ) -> None:
        """统一记录单步 outcome. 内部分流到 FailedDirectionStore + SkillEvolutionLayer.

        flag off 时 no-op (走原 _learn 路径).
        """
        if not _use_evolution_manager():
            return
        if self._failed_store is None:
            logger.warning("EvolutionManager: no memory_manager, skip record_outcome")
            return

        val_status = ""
        val_reason = ""
        if isinstance(validation, dict):
            # 多种 key 兼容: status / tests_passed → 推断; reason / error / summary
            val_status = validation.get("status", "")
            if not val_status:
                tests_passed = validation.get("tests_passed")
                if tests_passed is False:
                    val_status = "refuted"
                elif tests_passed is True:
                    val_status = "supported"
            val_reason = (
                validation.get("reason")
                or validation.get("error")
                or validation.get("summary")
                or ""
            )

        # refuted/superseded/failed 时记 failed_direction (hypothesis 级)
        if val_status in ("refuted", "superseded", "failed"):
            try:
                self._failed_store.record(
                    hypothesis_text=hypothesis,
                    reason=val_reason,
                    run_id=run_id,
                    persona_id=persona_id,
                    math_concept=math_concept,
                    status=val_status,
                )
            except Exception:
                logger.warning("record failed_direction failed", exc_info=True)

        # strategy 级失败: 抽 plan.mode/strategy, 单独记一条 strategy_failed
        strategy_name = ""
        if isinstance(plan, dict):
            strategy_name = plan.get("mode") or plan.get("strategy") or ""

        if strategy_name and val_status in ("refuted", "superseded", "failed"):
            try:
                self._failed_store.record(
                    hypothesis_text=hypothesis,
                    reason=f"strategy {strategy_name} failed: {val_reason}",
                    run_id=run_id,
                    persona_id=persona_id,
                    math_concept=math_concept,
                    strategy_name=strategy_name,
                    status="strategy_failed",
                )
            except Exception:
                logger.warning("record strategy_failed failed", exc_info=True)

        # 同步到 SkillEvolutionLayer (hypothesis / strategy 级, 不只是工具参数)
        if self._se is not None and val_status in ("refuted", "superseded", "failed"):
            try:
                self._se.record_hypothesis_failure(
                    hypothesis_text=hypothesis,
                    reason=val_reason,
                    math_concept=math_concept,
                )
                if strategy_name:
                    self._se.record_strategy_failure(
                        strategy_name=strategy_name,
                        reason=val_reason,
                    )
            except Exception:
                logger.warning("SkillEvolutionLayer record failed", exc_info=True)

    def distill(self, run_id: str | None = None) -> list[str]:
        """蒸馏 STABLE_PRINCIPLE. 返回写入的 principle 文本列表.

        触发条件 (spec):
        - 连续 3 次同 persona + 同 math_concept 失败 → 写 "avoid persona X for math concept Y"
        - 连续 3 次同 strategy 成功 → 写 "prefer strategy Z"

        ponytail: 失败方向查 typed memory 即可; 成功方向目前 engine 没把
        iteration_result 写 strategy_name, "prefer" 蒸馏暂留为 TODO, 不伪造数据.
        升级路径: 给 iteration_result 加 strategy_name 字段后补全.
        """
        if not _use_evolution_manager():
            return []
        if self._mm is None or not hasattr(self._mm, "remember_typed"):
            return []
        if self._failed_store is None:
            return []

        written: list[str] = []
        try:
            records = self._failed_store.query(limit=20)
            # 按 (persona_id, math_concept) 分组, 找 ≥3 次失败的
            groups: dict[tuple[str, str], list] = defaultdict(list)
            for r in records:
                key = (r.persona_id or "", r.math_concept or "")
                groups[key].append(r)
            for (pid, mc), recs in groups.items():
                if len(recs) < 3:
                    continue
                # ponytail: 不做"连续"严格判定 (没时间序), 用 count≥3 近似.
                # 升级路径: 按 created_at 排序后验连续 3 次.
                principle = (
                    f"avoid persona {pid or 'default'} for math concept "
                    f"{mc or 'unknown'} (3+ failures)"
                )
                try:
                    from huginn.memory.typing import remember_typed

                    remember_typed(
                        self._mm,
                        content=principle,
                        memory_type="stable_principle",
                        persona_id=pid or None,
                        status="avoid",
                        importance=0.8,
                        tier="long",
                        source="evolution:distill",
                    )
                    written.append(principle)
                except Exception:
                    logger.warning("distill stable_principle write failed", exc_info=True)
        except Exception:
            logger.warning("distill failed", exc_info=True)
        return written

    def recommend(self, hypothesis_context: dict | None = None) -> Recommendation:
        """给 _hypothesize 推荐避开的失败方向 + 优先策略.

        avoid_directions: 最近 5 条 failed_direction (跨 session 可恢复)
        prefer_strategies: 从 STABLE_PRINCIPLE status="prefer" 取 (当前 distill
                          只写 avoid, 这里预留接口, 没数据时返空)
        """
        if not _use_evolution_manager() or self._failed_store is None:
            return Recommendation(rationale="flag off or no memory_manager")

        avoid: list[str] = []
        try:
            records = self._failed_store.query(limit=5)
            avoid = [r.hypothesis_text for r in records if r.hypothesis_text]
        except Exception:
            logger.warning("recommend avoid query failed", exc_info=True)

        prefer: list[str] = []
        try:
            if hasattr(self._mm, "recall_typed"):
                from huginn.memory.typing import recall_typed

                principles = recall_typed(
                    self._mm,
                    memory_type="stable_principle",
                    status="prefer",
                    limit=3,
                )
                prefer = [
                    p.get("content", "") for p in principles if p.get("content")
                ]
        except Exception:
            logger.warning("recommend prefer query failed", exc_info=True)

        rationale = (
            f"avoid {len(avoid)} failed directions, "
            f"prefer {len(prefer)} strategies"
        )
        return Recommendation(
            avoid_directions=avoid, prefer_strategies=prefer, rationale=rationale
        )


# ── selfcheck ─────────────────────────────────────────────────────────────
# 3 场景: flag off no-op / record_outcome + recommend 闭环 / distill 3+ 触发.

def _selfcheck() -> None:
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    _agent_root = Path(__file__).resolve().parents[2]
    if str(_agent_root) not in sys.path:
        sys.path.insert(0, str(_agent_root))

    from huginn.memory.longterm import LongTermMemory
    from huginn.memory.manager import MemoryManager

    # 场景 1: flag off → record_outcome / distill / recommend 全 no-op
    os.environ.pop(_FLAG, None)
    EvolutionManager._reset_for_test()
    em = EvolutionManager.shared()
    # 没传 mm 也不抛, 全 no-op
    em.record_outcome("hyp", None, None, None, "r1")
    assert em.distill() == []
    rec = em.recommend()
    assert rec.avoid_directions == [] and rec.prefer_strategies == []
    assert "flag off" in rec.rationale, rec.rationale
    print("1. flag off no-op OK")

    # 场景 2: flag on → record_outcome 写 failed_direction, recommend 查到
    tmpdir = tempfile.mkdtemp(prefix="huginn_em_selfcheck_")
    db_path = Path(tmpdir) / "memory.db"
    ltm = LongTermMemory(db_path=str(db_path), enable_semantic=False)
    mm = MemoryManager(longterm=ltm)
    os.environ[_FLAG] = "1"
    EvolutionManager._reset_for_test()
    em = EvolutionManager.shared(memory_manager=mm)

    em.record_outcome(
        hypothesis="GaN gap > 4 eV with LDA",
        plan={"mode": "lda_direct"},
        validation={"status": "refuted", "reason": "LDA underestimates gap"},
        persona_id="dft_expert",
        run_id="run_001",
        math_concept="DFT-PZ LDA gap underestimation",
    )
    rec = em.recommend()
    assert rec.avoid_directions, f"expected avoid_directions, got {rec.avoid_directions}"
    assert any("GaN gap" in a for a in rec.avoid_directions), rec.avoid_directions
    print("2. record_outcome + recommend round-trip OK")

    # 场景 3: distill — 连续 3 次同 persona+math_concept 失败 → STABLE_PRINCIPLE
    for i in range(2):
        em.record_outcome(
            hypothesis=f"hyp attempt {i}",
            plan={"mode": "lda_direct"},
            validation={"status": "refuted", "reason": f"fail {i}"},
            persona_id="dft_expert",
            run_id=f"run_{i+2}",
            math_concept="DFT-PZ LDA gap underestimation",
        )
    principles = em.distill()
    assert principles, f"expected ≥1 principle, got {principles}"
    assert any("avoid persona dft_expert" in p for p in principles), principles
    print("3. distill 3+ failures → STABLE_PRINCIPLE OK")

    # 清理
    os.environ.pop(_FLAG, None)
    EvolutionManager._reset_for_test()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("evolution_manager selfcheck OK (3 scenarios)")


if __name__ == "__main__":
    _selfcheck()
