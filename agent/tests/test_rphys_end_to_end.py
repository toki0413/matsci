"""端到端生产路径冒烟 — `generate_patch`（引擎 RSI 流同款入口）驱动 meta 全链开闸.

A1 各门控单元级已由 test_meta_improver.py 覆盖；本测试走**真实生产入口**
`huginn.harness.prompt_patch.generate_patch`（engine_reflect._rsi_self_directive
第 3037-3055 行调用它），把 harness_prompt_patch + meta_improver + rphys_gate
三个开关全开，用确定性 fake LLM 推演一条完整 champion 生命周期，验证：

  1. generate_patch 成功产 patch → note_generation 收真实 r_phys → 写入 replay；
  2. 真实 r_phys 归因到当时代际 active 的改进器 champion（RPhysTrack）;
  3. 上行 champion 的 RPhysTrack.verdict 判 green（真实物理验证分随其驱动上行）;
  4. 代理分 GREEN 但真实 r_phys 未上行的挑战者，在 harness_rphys_gate 硬闸下被拒。

不 mock 任何真实对象。全部交互为确定性 async fake LLM。
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

import huginn.harness.meta_improver as mi
import huginn.harness.prompt_patch as pp

from huginn.harness.meta_improver import ImproverConfig, MetaImprover
from huginn.harness.prompt_patch import generate_patch
from huginn.harness.significance_gate import SignificanceGate
from huginn.harness.ood_holdout import OODHoldoutValidator

GOOD_TPL = (
    "careful-improver Phase:{phase} Blocks:{block_names} R:{r_phys} "
    "D:{directive} Emit one patch JSON."
)


def _all_on(tmp_path: Path) -> MetaImprover:
    """全开三维开关并重置单例, 返回 meta 层单例."""
    os.environ["HUGINN_CACHE_DIR"] = str(tmp_path)
    for cls in (MetaImprover, SignificanceGate, OODHoldoutValidator):
        cls._instance = None
    pp.PromptPatchStore._instance = None  # 隔离跨测试泄漏的 patch store
    mi._harness_enabled = lambda key, default=False: key in (
        "harness_prompt_patch", "harness_meta_improver", "harness_rphys_gate",
    )
    pp._harness_enabled = lambda key, default=False: key == "harness_prompt_patch"
    return MetaImprover.get_instance()


def _seed_proxy_green(candidate_id: str) -> None:
    """给候选择显著+OOD 的 LLM 代理分 (proxy GREEN)."""
    sig = SignificanceGate.get_instance()
    for i in range(8):
        sig.record_pair(candidate_id, 0.3, 0.9, task_id=f"sig_{i}")
    val = OODHoldoutValidator.get_instance()
    for i in range(24):
        t = f"seed_{i:02d}"
        val.record_outcome(val._BASELINE_ID, t, 0.4)
        val.record_outcome(candidate_id, t, 0.9)


async def fake_llm(prompt: str, task: str = "summarize") -> str:
    """确定性 fake:
    - maybe_propose(默认 meta 模板开头) → 无效 improver 模板 → 提案被拒 (保持 champion 稳定)
    - maybe_propose_strategist(meta² 开头) → 无效 strategist 模板 → 被拒
    - 其余 (generate_patch 的 improver 模板) → 有效 patch JSON
    """
    if prompt.startswith(mi._META_IMPROVE_TEMPLATE[:40]):
        return "bad {phase}"  # 缺 {block_names}/{r_phys}/{directive} → 拒绝
    if prompt.startswith(mi._META2_IMPROVE_TEMPLATE[:40]):
        return "bad strategist no current_template placeholder"  # 拒绝
    return '{"block_name": "mem", "op": "append", "new_text": "classical-focus-hint"}'


def test_rphys_end_to_end_generate_patch_chain(tmp_path: Path) -> None:
    """真实引擎入口 generate_patch 跑通: 归因→上行 green→硬闸拒未上行挑战者."""
    meta = _all_on(tmp_path)

    # 收敛 champion: 一个好改进器模板 + 一个低频 baseline 池, 让真实 r_phys 的上行可判
    champ = ImproverConfig(config_id="good", improver_prompt=GOOD_TPL, r_phys_gate=0.7)
    champ.active = True
    meta._candidates["good"] = champ
    meta._active_id = "good"
    for v in (0.40, 0.42, 0.44):
        meta._rphys.record("_base", v)  # 低频池 (默认)

    blocks = [("mem", "original mem block")]
    # 用 generate_patch 驱动 N 代, 喂递增的真实 r_phys (<=0.7 才会 generate)
    trajectory = [0.50, 0.55, 0.60, 0.65, 0.70]
    replay_before = len(meta._replay)
    for r in trajectory:
        pulled = asyncio.run(
            generate_patch("hypothesize", blocks, r, "classical-focus hint", fake_llm)
        )
        assert pulled is not None, f"generate_patch should succeed at r_phys={r}"
    assert len(meta._replay) == replay_before + len(trajectory), "note_generation 应写 replay"

    # 1. champion 模板被 generate_patch 实际采用 (真实入口覆盖, 非仅单元)
    store = pp.PromptPatchStore.get_instance()
    for p in store.list_patches(phase="hypothesize"):
        assert p.directive_in.startswith("classical-focus"), "应产自 champion 模板的 directive"

    # 2. 真实 r_phys 归因到 active champion (RPhysTrack)
    champ_series = meta._rphys.series("good")
    assert champ_series, "champion 应有真实 r_phys 代际"
    assert champ_series[-1] == trajectory[-1], "最近一代真实 r_phys 应归因到 good"

    # 3. 上行 champion → 真实 r_phys 验收 green
    verdict = meta._rphys.verdict("good")
    assert verdict["green"] is True, f"champion 真实 r_phys 应判上行: {verdict}"
    assert meta.compounding_trace()["rphys_active_verdict"]["green"] is True

    # 4. 代理分 GREEN 但真实 r_phys 未上行的挑战者, 在 harness_rphys_gate 硬闸下被拒
    _seed_proxy_green("challenger")
    meta._candidates["challenger"] = ImproverConfig(
        config_id="challenger", improver_prompt=GOOD_TPL, r_phys_gate=0.7,
    )
    # 挑战者真实 r_phys 低 (相对池) → verdict 非 green → 硬闸拦截
    for v in (0.20, 0.22, 0.24):
        meta._rphys.record("challenger", v)
    assert meta._rphys.verdict("challenger")["green"] is False
    assert meta.maybe_promote("challenger") is False, "rphys 硬闸应拒代理分green但真实未上行"
    assert meta.champion_cfg().config_id == "good", "champion 应保持不变"

    # 5. 端到端落了盘 (trace/rphys.json)
    assert meta._rphys._path.exists(), "rphys.json 应落盘"
    assert meta._trace_path.exists()
    lines = meta._trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("generation" in json.loads(l)["type"] for l in lines)