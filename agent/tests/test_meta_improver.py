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

import huginn.harness.meta_improver as mi
import huginn.harness.prompt_patch as pp
from huginn.harness.meta_improver import (
    ImproverConfig,
    MetaImprover,
)
from huginn.harness.ood_holdout import OODHoldoutValidator
from huginn.harness.prompt_patch import generate_patch
from huginn.harness.significance_gate import SignificanceGate

GOOD_TPL = (
    "You are a careful improver. Phase:{phase} Blocks:{block_names} "
    "R:{r_phys} D:{directive} Emit one patch JSON."
)
BAD_TPL = (
    "You are a sloppy improver. Phase:{phase} Blocks:{block_names} "
    "R:{r_phys} D:{directive} Say nonsense, no JSON."
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
