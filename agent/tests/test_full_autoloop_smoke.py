"""引擎级全开冒烟: 经真实 `EngineReflect._generate_next_loop_directive` 入口驱动全部三环.

不构造完整 AutoloopEngine（本沙箱缺 model/依赖, 构造即炸）; 而是用一个满足该方法
所需字段的最小 stub engine + 真实 `EngineReflect` 协作对象, 确定性 fake LLM:

  - 对 directive 提示返回真实 directive → 走 memory.remember;
  - 对 improver 提示返回有效 patch → generate_patch → note_generation 驱动
    meta(level-0) / strategist / source 三环 + RPhysTrack 真实 r_phys 归因.

验证点:
  1. directive 被写入 memory (真实入口的 RSI 自指令);
  2. generate_patch 经真实入口调用, 真实 r_phys 归因到 MetaImprover 的 RPhysTrack;
  3. meta/strategist/source 三环全开下被驱动, 不 crash.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

import huginn.harness.meta_improver as mi
import huginn.harness.prompt_patch as pp
import huginn.harness.source_patch as sp

from huginn.autoloop.engine_reflect import EngineReflect
from huginn.harness.meta_improver import MetaImprover


class _StubEngine:
    """满足 _generate_next_loop_directive 字段的最小真实引擎替身 (非协作对象)."""

    def __init__(self) -> None:
        self.memory = None
        self._llm_chat = None
        self._iteration = 1
        self._last_hypothesis_blocks = [("mem", "m"), ("body", "b {context}")]


class _StubMemory:
    """真实入口 memory.remember 的替身: 记录收到的 directive."""

    def __init__(self) -> None:
        self.remembered: list[str] = []

    def remember(self, content: str, category: str, tags: list[str], importance: float,
                 tier: str) -> None:
        self.remembered.append(content)


def test_full_autoloop_engine_entry_smoke(tmp_path: Path) -> None:
    # 隔离运行时: 所有 meta/source store 进 tmp; 清空单例 + 全开三环
    os.environ["HUGINN_CACHE_DIR"] = str(tmp_path)
    for cls in (MetaImprover, sp.SourcePatchStore):
        cls._instance = None

    def flags(key: str, default: bool = False) -> bool:
        return key in (
            "harness_meta_improver", "harness_prompt_patch", "harness_rphys_gate",
            "harness_verifiable_gate", "harness_source_patch",
        )

    mi._harness_enabled = flags
    pp._harness_enabled = lambda key, default=False: key == "harness_prompt_patch"

    async def fake_llm(prompt: str, task: str = "summarize") -> str:
        # directive 提示 → 真实 directive (让流程走到底)
        if "single concise directive" in prompt:
            return "focus on classical overlap methods"
        # source 环提示 (harness_source_patch 开) → 一行符号赋值 (多目标轮转, 动态取目标)
        if "self-modifying" in prompt:
            current = MetaImprover.get_instance()._source_target()[1]
            return f"{current} = 6"
        # improver 提示 (generate_patch) → 有效 patch JSON
        return '{"block_name": "mem", "op": "append", "new_text": "classical-focus-hint"}'

    stub = _StubEngine()
    stub.memory = _StubMemory()
    stub._llm_chat = fake_llm
    reflect = EngineReflect(stub)

    meta = MetaImprover.get_instance()

    # pass 1: directive 写 memory + generate_patch 经真实入口跑, 真实 r_phys 归因
    asyncio.run(reflect._generate_next_loop_directive(
        hypothesis="test hypothesis", plan={"mode": "explore"},
        validation={"tests_passed": True}, r_phys=0.50,
    ))
    assert stub.memory.remembered, "directive 应写进 memory (真实入口 RSI 自指令)"
    assert stub.memory.remembered[0].startswith("[self-directive iter"), stub.memory.remembered
    # 真实 r_phys 归因: 无 champion → 归因 _base
    assert meta._rphys.series("_base") == [0.5], meta._rphys.series("_base")

    # pass 2~8: 连续驱动直到 source 环阈值 (_SOURCE_EVERY_N_GENERATIONS=7) 命中,
    # 验证 meta/strategist/source 三环在阈值触发时不 crash + 真实 r_phys 持续归因
    import huginn.harness.meta_improver as _mi_mod

    orig_every = _mi_mod._PROPOSE_EVERY_N
    try:
        for i in range(7):
            asyncio.run(reflect._generate_next_loop_directive(
                hypothesis=f"hyp {i}", plan={"mode": "explore"},
                validation={"tests_passed": True}, r_phys=0.51 + i * 0.02,
            ))
        # source 环被驱动 (harness_source_patch 开) → 至少登记一条 source patch
        assert sp.SourcePatchStore.get_instance().list_patches(), "source 环应被驱动"
    finally:
        # 恢复可能被 source 环自改的全部白名单目标常量 (多目标轮转)
        _mi_mod._PROPOSE_EVERY_N = 5
        _mi_mod._STRATEGIST_EVERY_N_PROPOSALS = 3
        _mi_mod._REPLAY_MAX = 10
        sp._ORIGINALS.clear()
        sp.SourcePatchStore._instance = None

    assert len(meta._rphys.series("_base")) >= 8, "真实 r_phys 序列应持续累积"