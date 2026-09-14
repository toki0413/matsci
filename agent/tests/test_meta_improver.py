"""M-R1 Recursive (Meta-)Improver pytest 验收.

Locks the recursive-improver behavior:
  1. toggle（meta off）→ generate_patch 仍用默认 improver 模板（零回归，无 champion 覆盖）。
  2. champion 模板被 activate 后 → generate_patch 用 champion 模板覆盖默认。
  3. evaluate 在冻结重放集上给候选 vs 基准打分并注册显著性配对（cand_mean > base_mean）。
  4. 好候选（有 GREEN 数据）→ maybe_promote 切换 champion。
  5. 差候选/背题候选 → 不提升（显著性不过 / OOD 退化拦截）。
  6. meta_trace + 换代历史 + 赢率被持久化，reload 保留。

全部 LLM 交互用确定性 async fake，未用 unittest.mock。
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

import huginn.harness.prompt_patch as pp
import huginn.harness.meta_improver as mi

from huginn.harness.meta_improver import (
    ImproverConfig,
    MetaImprover,
)
from huginn.harness.prompt_patch import generate_patch
from huginn.harness.significance_gate import SignificanceGate
from huginn.harness.ood_holdout import OODHoldoutValidator

GOOD_TPL = (
    "You are a careful improver. Phase:{phase} Blocks:{block_names} "
    "R:{r_phys} D:{directive} Emit one patch JSON."
)
BAD_TPL = (
    "You are a sloppy improver. Phase:{phase} Blocks:{block_names} "
    "R:{r_phys} D:{directive} Say nonsense, no JSON."
)

# 合法 strategist 候选模板: 是给 maybe_propose 的 .format 用的 meta 提示模板,
# 需含 {current_template}/{n_proposals}/{n_promotions} 占位符.
_META2_TPL = (
    "You are the meta-strategist. Current improver template:\n"
    "{current_template}\nPropose a better improver. n_proposals={n_proposals} "
    "promotions={n_promotions}. Respond template ONLY."
)


HTuple = tuple["MetaImprover", Path]


def _reseed(tmp_path: Path) -> None:
    os.environ["HUGINN_CACHE_DIR"] = str(tmp_path)
    MetaImprover._instance = None
    SignificanceGate._instance = None
    OODHoldoutValidator._instance = None


def _meta_on(tmp_path: Path, on_meta: bool = True, on_prompt_patch: bool = True) -> MetaImprover:
    """meta 层开关: 只影响 meta 模块 (champion/enabled), 不碰 prompt_patch 模块."""
    _reseed(tmp_path)
    mi._harness_enabled = lambda key, default=False: (
        (on_meta and key == "harness_meta_improver")
        or (on_prompt_patch and key == "harness_prompt_patch")
        or default
    )
    return mi.MetaImprover.get_instance()


def _pp_on(on: bool, meta_on: bool) -> None:
    """patch prompt_patch 模块自己的 _harness_enabled (generate_patch 读它)."""
    pp._harness_enabled = lambda key, default=False: (
        (on and key == "harness_prompt_patch")
        or (meta_on and key == "harness_meta_improver")
        or default
    )


def _seed_green(candidate_id: str, base: float = 0.4, cand: float = 0.9) -> None:
    """确定性 GREEN 数据: 显著性全正 + OOD 候选优于基线(12 任务够 train/holdout)."""
    sig = SignificanceGate.get_instance()
    for i in range(8):
        sig.record_pair(candidate_id, 0.3, 0.9, task_id=f"sig_{i}")
    val = OODHoldoutValidator.get_instance()
    for i in range(24):
        t = f"seed_{i:02d}"
        val.record_outcome(val._BASELINE_ID, t, base)
        val.record_outcome(candidate_id, t, cand)


async def _fake_good_default_bad(prompt: str, task: str = "summarize") -> str:
    """default/sloppy 模板 -> 坏产出; careful 模板 -> 有效 patch."""
    if "careful improver" in prompt:
        return '{"block_name": "mem", "op": "append", "new_text": "focus-hint"}'
    return "not-json-at-all"


def _add_replay(meta: MetaImprover, n: int = 8) -> None:
    for i in range(n):
        meta._replay.append(
            {
                "phase": "hypothesize",
                "block_names": ["body", "mem", "fail"],
                "r_phys": 0.5,
                "directive": f"hint {i}",
                "ts": float(100 + i),
                "probe_id": f"probe_{i:02d}",
            }
        )


# ── 1/2 零回归: meta off 不覆盖默认模板 ─────────────────────────────────────
def test_meta_off_uses_default_template(tmp_path: Path) -> None:
    """meta off + prompt_patch on → generate_patch 用默认 improver 模板."""
    _reseed(tmp_path)
    _pp_on(on=True, meta_on=False)

    seen: list[str] = []

    async def fake(prompt: str, task: str = "summarize"):
        seen.append(prompt)
        return '{"block_name": "mem", "op": "append", "new_text": "x"}'

    pulled = asyncio.run(
        generate_patch("hypothesize", [("body", "b"), ("mem", "m")], 0.5, "hint", fake)
    )
    assert pulled is not None
    # 行的就是默认模板 → 改进器行为与改动前一致 (零回归)
    assert "You are optimizing a research agent's prompt template" in seen[0]


def test_champion_overrides_default_template(tmp_path: Path) -> None:
    """meta on + champion active → generate_patch 用 champion 模板."""
    meta = _meta_on(tmp_path, on_meta=True, on_prompt_patch=True)
    _pp_on(on=True, meta_on=True)

    champ = ImproverConfig(config_id="champ1", improver_prompt=GOOD_TPL, r_phys_gate=0.7)
    champ.active = True
    meta._candidates["champ1"] = champ
    meta._active_id = "champ1"
    meta._save_candidate(champ)
    meta._save_cfg()

    seen: list[str] = []

    async def fake(prompt: str, task: str = "summarize"):
        seen.append(prompt)
        return '{"block_name": "mem", "op": "append", "new_text": "x"}'

    pulled = asyncio.run(
        generate_patch("hypothesize", [("body", "b"), ("mem", "m")], 0.5, "hint", fake)
    )
    assert pulled is not None
    assert "careful improver" in seen[0], "champion template should be used"


# ── 3 evaluate 打分 ──────────────────────────────────────────────────────────
def test_evaluate_scores_candidate_over_default(tmp_path: Path) -> None:
    """evaluate 在 replay 上给候选打分并注册显著性配对 (cand > base)."""
    meta = _meta_on(tmp_path)
    _add_replay(meta)

    cfg = ImproverConfig(config_id="good", improver_prompt=GOOD_TPL, r_phys_gate=0.7)
    meta._candidates["good"] = cfg

    res = asyncio.run(meta.evaluate("good", _fake_good_default_bad))
    assert res["scores_n"] == 8, res
    assert res["cand_mean"] > res["base_mean"], res

    sig = SignificanceGate.get_instance()
    pairs = sig.get_pairs("good")
    assert len(pairs) >= 8, "sig pairs should be registered"


# ── 4 好候选 GREEN 提升 ──────────────────────────────────────────────────────
def test_good_candidate_promotes_on_green(tmp_path: Path) -> None:
    """候选有 GREEN 数据 → maybe_promote 切换 champion."""
    meta = _meta_on(tmp_path)
    _seed_green("good")
    meta._candidates["good"] = ImproverConfig(
        config_id="good", improver_prompt=GOOD_TPL, r_phys_gate=0.7
    )

    assert meta.maybe_promote("good") is True
    assert meta.champion_cfg().config_id == "good"
    assert meta.compounding_trace()["n_promotions"] >= 1


# ── 5 差候选 / 背题候选 不提升 ────────────────────────────────────────────────
def test_bad_candidate_does_not_promote(tmp_path: Path) -> None:
    """候选无有效数据(不显著) → 不换件, champion 不变."""
    meta = _meta_on(tmp_path)
    champ = ImproverConfig(config_id="champ", improver_prompt=GOOD_TPL, r_phys_gate=0.7)
    champ.active = True
    meta._candidates["champ"] = champ
    meta._active_id = "champ"

    meta._candidates["bad"] = ImproverConfig(
        config_id="bad", improver_prompt=BAD_TPL, r_phys_gate=0.7
    )
    before = meta.champion_cfg().config_id
    assert meta.maybe_promote("bad") is False, "bad must not promote"
    assert meta.champion_cfg().config_id == before


def test_ood_overfit_not_promoted(tmp_path: Path) -> None:
    """背题候选: 显著性过但 OOD holdout 退化 → 不提升."""
    meta = _meta_on(tmp_path)
    val = OODHoldoutValidator.get_instance()
    sig = SignificanceGate.get_instance()

    # 显著性: 全正 → 过
    for i in range(6):
        sig.record_pair("overfit", 0.3, 0.9, task_id=f"f{i}")

    # baseline 在 20 个任务上稳定 0.5; candidate train 好 holdout 差
    from huginn.harness.ood_holdout import _is_holdout

    for i in range(24):
        t = f"oft_{i:03d}"
        val.record_outcome(val._BASELINE_ID, t, 0.5)
        val.record_outcome("overfit", t, 0.9 if not _is_holdout(t) else 0.2)

    ood = val.validate_ood("overfit")
    assert not ood.passed, f"overfit should fail OOD: {ood}"
    assert sig.gate_decision("overfit").passed, "sig should pass (isolate OOD)"
    meta._candidates["overfit"] = ImproverConfig(
        config_id="overfit", improver_prompt=BAD_TPL
    )
    assert meta.maybe_promote("overfit") is False, "overfit must be OOD-blocked"


# ── 6 trace / 持久化 ─────────────────────────────────────────────────────────
def test_trace_persistence_and_win_rate(tmp_path: Path) -> None:
    """meta_trace 落盘 + champion 换代 + 持久化 reload 保留."""
    meta = _meta_on(tmp_path)
    _seed_green("promo")
    meta._candidates["promo"] = ImproverConfig(
        config_id="promo", improver_prompt=GOOD_TPL, r_phys_gate=0.7
    )
    assert meta.maybe_promote("promo") is True

    tr = meta.compounding_trace()
    assert tr["active_config_id"] == "promo"
    assert tr["n_promotions"] >= 1
    assert tr["meta_win_rate"] > 0

    assert meta._trace_path.exists()
    lines = meta._trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    assert any("promote" in json.loads(l)["type"] for l in lines)

    # reload 保留
    MetaImprover._instance = None
    meta2 = MetaImprover.get_instance()
    assert meta2.compounding_trace()["active_config_id"] == "promo"


def test_compounding_tracker_math():
    from huginn.harness.meta_improver import CompoundingTracker
    tr = CompoundingTracker(window=4)
    for i in range(4):
        tr.record(epoch=i, config_id=f"c{i}", quality=0.5 + 0.1 * i,
                  fidelity=0.6, proposals_to_promotion=2,
                  generations_to_promotion=5, win_rate=0.5)
    assert tr.is_compounding() is True
    assert tr.stats()["slope"] > 0
    dec = CompoundingTracker(window=4)
    for i in range(4):
        dec.record(epoch=i, config_id=f"d{i}", quality=0.9 - 0.2 * i,
                   fidelity=0.6, proposals_to_promotion=2,
                   generations_to_promotion=5, win_rate=0.5)
    assert dec.would_degrade(new="n", incumbent="i") is True


# ── 论文 arXiv:2609.03621: BehavioralFidelity 保真锚 + VerifiableGate 验证 ────
def test_behavioral_fidelity_anchor_with_verified(tmp_path: Path) -> None:
    """采纳保真锚: 真实采纳率作复合指标锚; 经状态化仿真验证的采纳上修保真."""
    from huginn.harness.meta_improver import BehavioralFidelity

    bf = BehavioralFidelity(path=tmp_path / "fidelity.json")
    bf.record_acceptance("c2", accepted=True)
    bf.record_acceptance("c2", accepted=False)
    assert abs(bf.fidelity_score("c2") - 0.5) < 1e-9
    hi = bf.anchor_in("c2", p_quality=0.8, p_fidelity=None)
    assert abs(hi - 0.65) < 1e-9  # 0.5*0.8 + 0.5*0.5

    # 论文: 经状态化仿真验证(verified=True)的采纳 → 保真上修
    bf2 = BehavioralFidelity(path=tmp_path / "fidelity2.json")
    bf2.record_acceptance("v1", accepted=True, verified=True)
    bf2.record_acceptance("v1", accepted=True, verified=True)
    assert bf2.fidelity_score("v1") == 1.0  # 全部采纳 + 全部验证 → 封顶可信
    # 验证加分不越界: 全采纳但未验证 = 1.0; verified 不把分数推过采纳率主锚
    bf3 = BehavioralFidelity(path=tmp_path / "fidelity3.json")
    bf3.record_acceptance("p1", accepted=True, verified=False)
    assert bf3.fidelity_score("p1") == 1.0
    # 持久化
    assert bf2._path.exists()
    data = json.loads(bf2._path.read_text(encoding="utf-8"))
    assert data["accepted"]["v1"] == 2


def test_verifiable_gate_preconditions_and_constraints(tmp_path: Path) -> None:
    """VerifiableGate(论文 2609.03621): 状态化仿真在派发前验证前置/约束.

    - 合法操作(源量充足 + 无约束违例) → passed, 且前向传播对象变换.
    - 前置不满足(源量不足) → 拦截, 返回违例列表.
    - 未知/非法动作类型 → 不做无依据转移.
    """
    from huginn.harness.meta_improver import VerifiableGate

    vg = VerifiableGate()
    ok = vg.verify_outcome({"action_type": "aspirate", "params": {"vol": 3.0},
                            "state": {"reagent_vol": 10.0, "sample_vol": 0.0}})
    assert ok["passed"] is True, ok
    assert ok["issues"] == []
    assert abs(ok["new_state"]["reagent_vol"] - 7.0) < 1e-9, ok
    assert abs(ok["new_state"]["sample_vol"] - 3.0) < 1e-9, ok

    # 前置不满足 → 拦截
    bad = vg.verify_outcome({"action_type": "aspirate", "params": {"vol": 99.0},
                             "state": {"reagent_vol": 10.0, "sample_vol": 0.0}})
    assert bad["passed"] is False, bad
    assert bad["issues"], bad

    # 未知动作 → 无依据转移, error 标记 (避免硬崩溃)
    unknown = vg.verify_outcome({"action_type": "teleport", "params": {},
                                 "state": {"reagent_vol": 10.0}})
    assert unknown["passed"] is False, unknown


# ── Task3: strategist 递归化 ─────────────────────────────────────────────────
def test_strategist_fallback_and_propose(tmp_path: Path) -> None:
    """无 strategist champion → strategist_prompt 回落默认; propose 出合法候选."""
    from huginn.harness.meta_improver import (
        MetaImprover, _META_IMPROVE_TEMPLATE,
    )
    meta = _meta_on(tmp_path)
    assert meta.strategist_prompt() == _META_IMPROVE_TEMPLATE
    assert meta.strategist_champion() is None

    # meta² 固定模板生成新 strategist
    async def fake(prompt: str, task: str = "summarize"):
        return _META2_TPL

    cid = asyncio.run(meta.maybe_propose_strategist(fake))
    assert cid is not None
    s = meta._strategists[cid]
    assert s.strategist_prompt == _META2_TPL
    assert not s.active
    # 候选持久化
    assert (meta._dir / "strategist" / f"{cid}.json").exists()
    # compounding_trace 暴露 strategist 状态
    tr = meta.compounding_trace()
    assert tr["active_strategist_id"] is None
    assert tr["n_strategists"] >= 1


def test_strategist_invalid_template_rejected(tmp_path: Path) -> None:
    """strategist 候选缺 current_template 占位符 → 拒绝."""
    meta = _meta_on(tmp_path)

    async def fake(prompt: str, task: str = "summarize"):
        return "no placeholder at all"

    cid = asyncio.run(meta.maybe_propose_strategist(fake))
    assert cid is None
    assert meta.compounding_trace()["n_strategists"] == 0


# ── Task4: strategist 换件可逆补偿 (时空可组合) ─────────────────────────────
def test_strategist_swap_revertible(tmp_path: Path) -> None:
    """换件在 RevertibleContext 内; 复合护栏触发 revert_all → 恢复上一 champion."""
    from huginn.harness.meta_improver import (
        MetaImprover, StrategistConfig, _register_strategist_compensator,
    )
    from huginn.security.revertible import RevertibleContext

    meta = _meta_on(tmp_path)
    _register_strategist_compensator()

    old = "s_old"
    meta._strategists[old] = StrategistConfig(config_id=old, strategist_prompt="old X", active=True)
    meta._active_strategist_id = old
    new = "s_new"
    meta._strategists[new] = StrategistConfig(config_id=new, strategist_prompt="new Y")

    # 换件在事务内, 正常提交保留
    ctx = RevertibleContext()
    with ctx.transaction():
        meta._apply_strategist_swap(old, new, ctx)
    assert meta.strategist_champion().config_id == new

    # 复合护栏触发 → revert_all 恢复上一 champion
    ctx2 = RevertibleContext()
    with ctx2.transaction():
        meta._apply_strategist_swap(old, new, ctx2)
    ctx2.revert_all()
    assert meta.strategist_champion().config_id == old, "revert 应恢复上一 champion"

def test_maybe_propose_uses_strategist(tmp_path: Path) -> None:
    """maybe_propose 改用 strategist champion 的模板 (改进方式可被改进)."""
    from huginn.harness.meta_improver import (
        MetaImprover, StrategistConfig,
    )
    meta = _meta_on(tmp_path)

    # 无 strategist → maybe_propose 用默认 meta 模板生成 improver 候选
    improver_template = "Optimize agent phase:{phase} blocks:{block_names} r:{r_phys} d:{directive} patch JSON"
    async def fake(prompt: str, task: str = "summarize"):
        return improver_template
    cid = asyncio.run(meta.maybe_propose(fake))
    assert cid is not None, "应能走默认 meta 模板 propose improver"
    assert meta._candidates[cid].improver_prompt == improver_template
    assert mi._META_IMPROVE_TEMPLATE in meta.strategist_prompt()  # 无 champion 回落默认

    # 有 strategist champion → strategist_prompt 用其模板
    s_champ = StrategistConfig(config_id="strat_c", strategist_prompt=_META2_TPL, active=True)
    meta._strategists["strat_c"] = s_champ
    meta._active_strategist_id = "strat_c"
    assert meta.strategist_prompt() == _META2_TPL


# ── Task5: evaluate/promote_strategist 门控组合 (sig+OOD+复合不退化+死锁) ───
def test_promote_strategist_green_and_reject(tmp_path: Path) -> None:
    """好 strategist (显著+OOD+复合不退化) → promote; 差 strategist → 拒绝."""
    from huginn.harness.meta_improver import (
        MetaImprover, StrategistConfig, _register_strategist_compensator,
    )
    meta = _meta_on(tmp_path)
    _register_strategist_compensator()
    sg = SignificanceGate.get_instance()
    ood = OODHoldoutValidator.get_instance()

    # 好候选: 显著全正 + OOD 候选优于基线
    for i in range(8):
        sg.record_pair("strat_good", 0.3, 0.9, task_id=f"g{i}")
    for i in range(24):
        t = f"sg{i:02d}"
        ood.record_outcome(ood._BASELINE_ID, t, 0.4)
        ood.record_outcome("strat_good", t, 0.9)
    meta._strategists["strat_good"] = StrategistConfig(config_id="strat_good", strategist_prompt="G")
    # 复合: 上升 → 不退化
    for i in range(4):
        meta._tracker.record(epoch=i, config_id="x", quality=0.5 + 0.1 * i, fidelity=0.6,
                             proposals_to_promotion=1, generations_to_promotion=3, win_rate=0.9)

    ok = meta.maybe_promote_strategist("strat_good")
    assert ok is True
    assert meta.strategist_champion() is not None
    assert meta.strategist_champion().config_id == "strat_good"

    # 差候选: 无数据(不显著) → 拒绝, champion 不变
    incumbent = meta.strategist_champion().config_id
    meta._strategists["strat_bad"] = StrategistConfig(config_id="strat_bad", strategist_prompt="B")
    assert meta.maybe_promote_strategist("strat_bad") is False
    assert meta.strategist_champion().config_id == incumbent
    # 死锁计数被标记 (连续拒 → in_deadlock)
    assert meta._tracker._deadlock_n >= 1


def test_strategist_would_degrade_blocks_promotion(tmp_path: Path) -> None:
    """复合已退化 (当前质量低于含滞回带的基线) → 拒绝换件."""
    from huginn.harness.meta_improver import (
        MetaImprover, StrategistConfig, _register_strategist_compensator,
    )
    meta = _meta_on(tmp_path)
    _register_strategist_compensator()
    sg = SignificanceGate.get_instance()
    ood = OODHoldoutValidator.get_instance()
    for i in range(8):
        sg.record_pair("cand", 0.3, 0.9, task_id=f"w{i}")
    for i in range(24):
        t = f"wd{i:02d}"
        ood.record_outcome(ood._BASELINE_ID, t, 0.4)
        ood.record_outcome("cand", t, 0.9)
    meta._strategists["cand"] = StrategistConfig(config_id="cand", strategist_prompt="C")
    # 复合: 下滑 → would_degrade True → 换件会被拒
    for i in range(4):
        meta._tracker.record(epoch=i, config_id="x", quality=0.9 - 0.2 * i, fidelity=0.6,
                             proposals_to_promotion=1, generations_to_promotion=3, win_rate=0.9)
    assert meta.maybe_promote_strategist("cand") is False
    assert meta.strategist_champion() is None, "复合退化时不应 promote"


def test_evaluate_strategist_trace(tmp_path: Path) -> None:
    """evaluate_strategist 返回 sig+ood verdict 并写 trace."""
    from huginn.harness.meta_improver import (
        MetaImprover, StrategistConfig,
    )
    meta = _meta_on(tmp_path)
    meta._strategists["s1"] = StrategistConfig(config_id="s1", strategist_prompt="S")
    # 无数据 → 不 GREEN
    r = asyncio.run(meta.evaluate_strategist("s1", None))
    assert r["sig"] is False and r["ood"] is False
    assert r["green"] is False


# ── Task6: CoEffectRegistry 空间可组合 (strategist 缺失 → improver 退化) ────
def test_coeffect_degrade(tmp_path: Path) -> None:
    """无 strategist champion → improver 空间依赖缺失 → improver_active False."""
    from huginn.harness.meta_improver import MetaImprover
    meta = _meta_on(tmp_path)
    reg = meta.coeffect_registry()
    assert reg is not None
    reg.update_availability(meta)
    # 无 gate(未开) + 无 strategist champion → improver 依赖缺失 → 失活
    assert reg.is_active("improver") is False
    assert reg.is_available("improvement_strategy") is False


def test_coeffect_strategist_active_enables_improver(tmp_path: Path) -> None:
    """有 strategist champion + gate enabled → improver 依赖满足 → active."""
    from huginn.harness.meta_improver import (
        MetaImprover, StrategistConfig, _register_strategist_compensator,
    )
    meta = _meta_on(tmp_path)
    _register_strategist_compensator()
    # 激活一个 strategist champion
    meta._strategists["strat_act"] = StrategistConfig(
        config_id="strat_act", strategist_prompt=_META2_TPL, active=True)
    meta._active_strategist_id = "strat_act"
    reg = meta.coeffect_registry()
    reg.update_availability(meta)
    assert reg.is_available("improvement_strategy") is True
    assert reg.is_active("improver") is True


# ── Task7: RandomizedControl 随机化对照仲裁 ─────────────────────────────────
def test_randomized_control(tmp_path: Path) -> None:
    """真实 r_phys 差分: champion 优于 baseline → champion_better."""
    from huginn.harness.meta_improver import RandomizedControl
    rc = RandomizedControl()
    outcome = rc.run_pair(champion_r=[0.8, 0.85, 0.9], baseline_r=[0.6, 0.62, 0.58])
    assert outcome["champion_better"] is True
    assert outcome["delta"] > 0
    assert outcome["n_champ"] == 3 and outcome["n_base"] == 3


def test_randomized_control_champion_worse(tmp_path: Path) -> None:
    """champion 劣于 baseline → champion_better False."""
    from huginn.harness.meta_improver import RandomizedControl
    rc = RandomizedControl()
    outcome = rc.run_pair(champion_r=[0.4, 0.5], baseline_r=[0.7, 0.8],
                          tolerance=0.02)
    assert outcome["champion_better"] is False


def test_maybe_promote_gates_on_ablation(tmp_path: Path) -> None:
    """HUGINN_META_ABLATION=1 且 champion 劣于 baseline → 拒绝换件."""
    from huginn.harness.meta_improver import (
        MetaImprover, StrategistConfig, RandomizedControl,
        _register_strategist_compensator,
    )
    os.environ["HUGINN_META_ABLATION"] = "1"
    try:
        meta = _meta_on(tmp_path)
        _register_strategist_compensator()
        sg = SignificanceGate.get_instance()
        ood = OODHoldoutValidator.get_instance()
        for i in range(8):
            sg.record_pair("cand", 0.3, 0.9, task_id=f"ab{i}")
        for i in range(24):
            t = f"ab{i:02d}"
            ood.record_outcome(ood._BASELINE_ID, t, 0.4)
            ood.record_outcome("cand", t, 0.9)
        meta._strategists["cand"] = StrategistConfig(config_id="cand", strategist_prompt="C")
        for i in range(4):
            meta._tracker.record(epoch=i, config_id="x", quality=0.5 + 0.1 * i, fidelity=0.6,
                                 proposals_to_promotion=1, generations_to_promotion=3, win_rate=0.9)
        # 注入 ablation: 争取真实 r_phys 差分 (champion 差于 baseline)
        rc = RandomizedControl()
        rc.override_pair = {"champion_better": False, "delta": -0.2,
                            "n_champ": 3, "n_base": 3}
        from huginn.harness import meta_improver as mi2
        mi2._RC_INSTANCE = rc
        ok = meta.maybe_promote_strategist("cand")
        # maybe_promote 内部随机化对照仅在 ablation 且 GREEN 时触发;
        # 由于覆盖 pair champion_better=False, 应拒绝.
        assert ok is False
        assert meta.strategist_champion() is None
    finally:
        os.environ.pop("HUGINN_META_ABLATION", None)
        mi2._RC_INSTANCE = None


# ── HIGH-1/3 修复验收: strategist 层端到端真实数据流 ─────────────────────────
def test_strategist_ring_drives_production(tmp_path: Path) -> None:
    """note_generation 驱动 strategist 递归环 (HIGH-3), evaluate_strategist 产
    真实 sig/ood 数据 (HIGH-1) → 好策略 GREEN 换件."""

    async def good_llm(prompt: str, task: str = "summarize") -> str:
        # 区分三类提示:
        # 1) "产出 improver 模板" 的 meta 提示 — 返回一个不同类型模板,
        #    让候选策略产生的模板比默认的更能对齐 directive (打分更高).
        # 2) "用 improver 模板产 patch" — 返回有效 JSON patch; 若当前 improver
        #    模板指向 good(therefore cand)则 patch 含 directive 词→高分, 否则低分.
        if prompt.startswith(mi._META2_IMPROVE_TEMPLATE[:40]):
            # level-1 提案: 返回候选策略模板 (含 GOOD-STRATEGIST 标记 + 4 占位)
            return "GOOD-STRATEGIST current_template={current_template} p={n_proposals} q={n_promotions}"
        if prompt.startswith(mi._META_IMPROVE_TEMPLATE[:40]):
            # 默认 meta 模板 → 差 improver 模板 (不含 phase 对齐词)
            return "sloppy improver Phase:{phase} Blocks:{block_names} R:{r_phys} D:{directive} JSON only."
        if "GOOD-STRATEGIST" in prompt:
            # 候选策略模板 → 好 improver 模板 (对齐 direct)
            return "careful improver Phase:{phase} Blocks:{block_names} R:{r_phys} D:{directive} focus-hint JSON only."
        # 产 patch: 若 prompt 里含 directive 对齐词 focus-hint 说明是好 improver
        return ('{"block_name": "mem", "op": "append", "new_text": "focus-hint"}'
                if "focus-hint" in prompt else "not-json")

    meta = _meta_on(tmp_path)
    # 填重放集, 让 evaluate 有样本
    for i in range(40):
        meta._replay.append(
            {"phase": "hypothesize", "block_names": ["body", "mem", "fail"],
             "r_phys": 0.5, "directive": f"plain-directive {i}",
             "ts": float(200 + i), "probe_id": f"ring_{i:02d}"}
        )
    meta._save_replay()

    # 直接驱动一环: 提案→评估→换件
    sid = asyncio.run(meta.maybe_propose_strategist(good_llm))
    assert sid is not None, "strategist should propose"
    r = asyncio.run(meta.evaluate_strategist(sid, good_llm))
    # 记录的数据应让它进入 sig/ood 判定
    assert r["sig"] is True and r["ood"] is True, f"应 GREEEN: {r}"
    assert meta.maybe_promote_strategist(sid) is True
    assert meta.strategist_champion() is not None, "换件后应有 champion"
    assert meta.compounding_trace()["n_strategists"] >= 1


def test_note_generation_triggers_strategist_ring(tmp_path: Path) -> None:
    """note_generation 在多次 level-0 提案后触发 strategist 环 (有生产入口)."""
    meta = _meta_on(tmp_path)

    async def ring_llm(prompt: str, task: str = "summarize") -> str:
        if prompt.startswith(mi._META2_IMPROVE_TEMPLATE[:40]):
            return "my-strategist current_template={current_template} p={n_proposals} q={n_promotions}"
        return '{"block_name": "mem", "op": "append", "new_text": "x"}'

    # 每 _PROPOSE_EVERY_N=5 次 generation 触发一次 level-0; _STRATEGIST_EVERY_N=3
    # 次 level-0 提案触发一次 strategist. 跑足 5*3=15 次 generation
    n_before = meta.compounding_trace()["n_strategists"]
    for i in range(16):
        asyncio.run(meta.note_generation(
            "hypothesize", [("body", "b"), ("mem", "m")], 0.5, f"hint {i}", ring_llm,
        ))
    assert meta.compounding_trace()["n_strategists"] > n_before, "strategist 环应被生产驱动"


def test_compounding_persistence(tmp_path: Path) -> None:
    """promote 后 compounding.json 落盘, reload 保留窗口 (spec data shape)."""
    from huginn.harness.meta_improver import (
        MetaImprover, StrategistConfig, _register_strategist_compensator,
    )
    meta = _meta_on(tmp_path)
    _register_strategist_compensator()
    sg = SignificanceGate.get_instance()
    ood = OODHoldoutValidator.get_instance()
    for i in range(8):
        sg.record_pair("s_persist", 0.3, 0.9, task_id=f"p{i}")
    for i in range(24):
        t = f"ps{i:02d}"
        ood.record_outcome(ood._BASELINE_ID, t, 0.4)
        ood.record_outcome("s_persist", t, 0.9)
    meta._strategists["s_persist"] = StrategistConfig(config_id="s_persist", strategist_prompt="P")
    for i in range(4):
        meta._tracker.record(epoch=i, config_id="x", quality=0.5 + 0.1 * i, fidelity=0.6,
                             proposals_to_promotion=1, generations_to_promotion=3, win_rate=0.9)
    assert meta.maybe_promote_strategist("s_persist") is True
    assert meta._compounding_path.exists(), "compounding.json 应落盘"
    data = json.loads(meta._compounding_path.read_text(encoding="utf-8"))
    assert data["rows"], "compounding.json 应含窗口 rows"
    # reload: 窗口保留
    MetaImprover._instance = None
    meta2 = MetaImprover.get_instance()
    assert len(meta2._tracker._rows) >= 1, "reload 后复合窗口保留"


# ── A1 端到端轨道: 真实 r_phys 真实验收 (论文 2609.03621 地面真值通道) ────────
def test_rphys_track_verdict_up() -> None:
    """配置驱动时真实 r_phys 显著高于池 (其余配置) → rphys_green."""
    from huginn.harness.meta_improver import RPhysTrack
    tr = RPhysTrack(path=None)
    # 目标配置 + 一个低频池配置 + 默认 baseline: 让池足够且明显更低
    for v in (0.2, 0.2, 0.2):
        tr.record("other", v)
    for v in (0.8, 0.8, 0.8):
        tr.record("good", v)
    res = tr.verdict("good")
    assert res["green"] is True, res
    assert res["n"] == 3 and res["pool"] == 3
    assert res["median_track"] > res["median_pool"]


def test_rphys_track_verdict_down() -> None:
    """配置驱动时真实 r_phys 不高于池 → 非 green (显著上行失败)."""
    from huginn.harness.meta_improver import RPhysTrack
    tr = RPhysTrack(path=None)
    for v in (0.8, 0.8, 0.8):
        tr.record("good_base", v)
    for v in (0.2, 0.2, 0.2):
        tr.record("bad", v)
    res = tr.verdict("bad")
    assert res["green"] is False, res


def test_rphys_track_insufficient_advisory() -> None:
    """样本不足 (<3 或 无池) → green=None (advisory 不阻塞, 回落代理分闸)."""
    from huginn.harness.meta_improver import RPhysTrack
    tr = RPhysTrack(path=None)
    tr.record("solo", 0.8)
    tr.record("solo", 0.8)
    assert tr.verdict("solo")["green"] is None  # 无池
    tr.record("pool", 0.5)
    assert tr.verdict("solo")["green"] is None  # 自身 <3
    assert tr.verdict("solo")["reason"] == "insufficient"


def test_rphys_track_persistence(tmp_path: Path) -> None:
    """rphys.json 落盘 + reload 保留真实代际序列."""
    from huginn.harness.meta_improver import RPhysTrack
    path = tmp_path / "rphys.json"
    tr = RPhysTrack(path=path)
    tr.record("good", 0.9)
    tr.record("good", 0.95)
    assert path.exists(), "rphys.json 应落盘"
    tr2 = RPhysTrack(path=path)
    assert tr2.series("good") == [0.9, 0.95]


def test_note_generation_attributes_rphys(tmp_path: Path) -> None:
    """note_generation 把真实 r_phys 归因到当时 active 改进器配置;
    无 champion → 归因 _base. 供 evaluate/compounding_trace 地面真值通道使用."""
    from huginn.harness.meta_improver import ImproverConfig, MetaImprover
    meta = _meta_on(tmp_path)

    async def fake(prompt: str, task: str = "summarize") -> str:
        return '{"block_name": "mem", "op": "append", "new_text": "x"}'

    # 无 champion → 归因 _base
    asyncio.run(meta.note_generation("h", [("body", "b"), ("mem", "m")],
                                     0.55, "hint", fake))
    assert meta._rphys.series("_base") == [0.55], "无 champion 应归因 _base"
    assert meta._rphys._path.exists(), "rphys.json 应写入"

    # 有 active champion → 归因该 champion
    champ = ImproverConfig(config_id="c1", improver_prompt="P", r_phys_gate=0.7)
    champ.active = True
    meta._candidates["c1"] = champ
    meta._active_id = "c1"
    asyncio.run(meta.note_generation("h", [("body", "b"), ("mem", "m")],
                                     0.7, "hint", fake))
    assert meta._rphys.series("c1") == [0.7], "应归因 active champion"
    # 冷启动时 champion 可能未部署, 此时 verdict 为 advisory (None)
    assert meta.compounding_trace()["rphys_active_verdict"]["green"] is None


def test_maybe_promote_rphys_hard_gate(tmp_path: Path) -> None:
    """harness_rphys_gate 开启 + 候选真实 r_phys 未显著上行 → 拒绝换件.

    代理分 (sig+ood) 说好, 但真实物理验证分没跟上去 — 端到端地面真值拦下, 防
    LLM 代理分 gaming (论文 2609.03621). 默认 off 时该闸不拦 (advisory).
    """
    from huginn.harness.meta_improver import ImproverConfig
    meta = _meta_on(tmp_path)
    _seed_green("rg")  # 代理分 GREEN (sig + OOD)
    meta._candidates["rg"] = ImproverConfig(config_id="rg", improver_prompt=GOOD_TPL)
    # 真实 r_phys: 候选显著低于池 → rphys 不可过
    for v in (0.2, 0.2, 0.2):
        meta._rphys.record("rg", v)
    for v in (0.8, 0.8, 0.8):
        meta._rphys.record("other", v)
    geo = meta._rphys.verdict("rg")
    assert geo["green"] is False, "候选真实 r_phys 应判定非上行"

    # 默认 off → 硬闸不拦, 照常 promote (遵循 M-R1 代理分门控)
    assert meta.maybe_promote("rg") is True, "advisory 默认不拦"

    # 开启 harness_rphys_gate 后重新判定: 需先让该候选处于待 promote 态
    _reseed(tmp_path)
    meta2 = _meta_on(tmp_path)
    mi._harness_enabled = lambda key, default=False: (
        (key in ("harness_meta_improver", "harness_prompt_patch", "harness_rphys_gate"))
        or default
    )
    _seed_green("rg2")
    meta2._candidates["rg2"] = ImproverConfig(config_id="rg2", improver_prompt=GOOD_TPL)
    for v in (0.2, 0.2, 0.2):
        meta2._rphys.record("rg2", v)
    for v in (0.8, 0.8, 0.8):
        meta2._rphys.record("other", v)
    assert meta2.maybe_promote("rg2") is False, "硬闸开启 + 真实 r_phys 未上行 → 拒绝"
    assert meta2.champion_cfg() is None or meta2.champion_cfg().config_id != "rg2", "rg2 不应换件"