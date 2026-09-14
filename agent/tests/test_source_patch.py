"""A3 Direct self-source modification — 运行时函数级自改源码 验收.

Locks the safe runtime self-mod behavior:
  1. verify 门: 非法语法 / 缺 anchor / def-time 错误 → 拒, 绝不 apply.
  2. apply: 验证通过后 monkeypatch 目标符号, 行为改变, 落盘 active.
  3. 可逆: RevertibleContext.revert_all() 恢复原实现 (会话级).
  4. meta source 环: source_enabled 默认关 → 不提议不改代码 (零回归).
  5. 好 source 候选 (verify + sig GREEN) → maybe_promote_source 应用; 环有生产入口.

不 mock 任何真实对象; 全部交互确定性 async fake LLM.
- 机制层目标: source_patch 自有常量 ``_SOURCE_EVERY_N_GENERATIONS`` (anchor 稳定存在).
- 环层目标: ``MetaImprover._SOURCE_TARGET`` = meta_improver 的 ``_PROPOSE_EVERY_N``.
每次应用后都恢复原值, 避免跨用例污染真实模块.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

import huginn.harness.meta_improver as mi
import huginn.harness.source_patch as sp

from huginn.harness.meta_improver import MetaImprover, _PROPOSE_EVERY_N
from huginn.harness.significance_gate import SignificanceGate
from huginn.harness.ood_holdout import OODHoldoutValidator

# 机制层目标 (source_patch 自有常量, anchor 存在且不影响其它逻辑)
MECH_MODULE = "huginn.harness.source_patch"
MECH_SYMBOL = "_SOURCE_EVERY_N_GENERATIONS"
# 环层目标 (MetaImprover._SOURCE_TARGET)
RING_MODULE = "huginn.harness.meta_improver"
RING_SYMBOL = "_PROPOSE_EVERY_N"


def _src_on(tmp_path: Path, on_source: bool = True) -> MetaImprover:
    os.environ["HUGINN_CACHE_DIR"] = str(tmp_path)
    for cls in (MetaImprover, SignificanceGate, OODHoldoutValidator, sp.SourcePatchStore):
        cls._instance = None
    sp._ORIGINALS.clear()
    mi._harness_enabled = lambda key, default=False: (
        (key in ("harness_meta_improver", "harness_prompt_patch", "harness_source_patch"))
        if on_source
        else (key in ("harness_meta_improver", "harness_prompt_patch"))
        or default
    )
    return MetaImprover.get_instance()


def _restore_mech() -> None:
    sp.__dict__[MECH_SYMBOL] = 7
    sp._ORIGINALS.clear()


def _restore_ring() -> None:
    # 恢复全部 A3 白名单目标常量 (轮转可能 promote 改任一), 避免污染真实模块
    mi.__dict__["_PROPOSE_EVERY_N"] = 5
    mi.__dict__["_STRATEGIST_EVERY_N_PROPOSALS"] = 3
    mi.__dict__["_REPLAY_MAX"] = 10
    sp._ORIGINALS.clear()


# ── verify 门 ────────────────────────────────────────────────────────────────
def test_verify_rejects_bad_patches() -> None:
    # 非法语法
    bad_syntax = sp.SourcePatch(id="s1", module=MECH_MODULE, symbol=MECH_SYMBOL,
                                new_code=f"def {MECH_SYMBOL}(")
    assert sp.verify_source_patch(bad_syntax)["passed"] is False
    # 缺 anchor
    no_anchor = sp.SourcePatch(id="s2", module=MECH_MODULE,
                               symbol="NONEXISTENT_SYMBOL_XYZ", new_code="NONEXISTENT_SYMBOL_XYZ = 1")
    assert sp.verify_source_patch(no_anchor)["passed"] is False
    # def-time 错误 (赋值引用未定义名)
    def_err = sp.SourcePatch(id="s3", module=MECH_MODULE, symbol=MECH_SYMBOL,
                             new_code=f"{MECH_SYMBOL} = _NOT_DEFINED_ANYWHERE")
    assert sp.verify_source_patch(def_err)["passed"] is False


def test_verify_passes_valid_patch() -> None:
    good = sp.SourcePatch(id="s4", module=MECH_MODULE, symbol=MECH_SYMBOL,
                          new_code=f"{MECH_SYMBOL} = 99")
    v = sp.verify_source_patch(good)
    assert v["passed"] is True, v


# ── apply + 可逆 ─────────────────────────────────────────────────────────────
def test_apply_runtime_selfmod_and_revert(tmp_path: Path) -> None:
    from huginn.security.revertible import RevertibleContext

    sp.SourcePatchStore._instance = None
    os.environ["HUGINN_CACHE_DIR"] = str(tmp_path)
    sp._ORIGINALS.clear()
    orig = sp.__dict__[MECH_SYMBOL]
    ctx = RevertibleContext()
    patch = sp.SourcePatch(id="s5", module=MECH_MODULE, symbol=MECH_SYMBOL,
                           new_code=f"{MECH_SYMBOL} = 12345")
    assert sp.apply_source_patch(patch, ctx) is True
    assert sp.__dict__[MECH_SYMBOL] == 12345, "运行时实现应被换入"
    assert sp.SourcePatchStore.get_instance().get("s5") is not None, "patch 应落盘"
    # 可逆
    ctx.revert_all()
    assert sp.__dict__[MECH_SYMBOL] == orig, "revert_all 应恢复原实现"
    _restore_mech()


def test_apply_rejects_unverified() -> None:
    bad = sp.SourcePatch(id="s6", module=MECH_MODULE, symbol=MECH_SYMBOL,
                         new_code=f"{MECH_SYMBOL} = _UNDEFINED")
    orig = sp.__dict__[MECH_SYMBOL]
    assert sp.apply_source_patch(bad) is False
    assert sp.__dict__[MECH_SYMBOL] == orig, "非法补丁绝不应用"


# ── meta source 环 ───────────────────────────────────────────────────────────
def test_source_ring_off_zero_regression(tmp_path: Path) -> None:
    """harness_source_patch 默认关 → 绝不提议、绝不改代码 (零回归)."""
    meta = _src_on(tmp_path, on_source=False)
    async def fake(prompt: str, task: str = "summarize") -> str:
        return f"{RING_SYMBOL} = 7"
    assert meta.source_enabled() is False
    assert asyncio.run(meta.maybe_propose_source(fake)) is None
    assert meta._SOURCE_TARGET[1] == RING_SYMBOL
    assert mi.__dict__[RING_SYMBOL] == _PROPOSE_EVERY_N, "off 时改任何代码"


def test_maybe_propose_source_parses_valid(tmp_path: Path) -> None:
    """on: LLM 产一行赋值 → 提议合法 source patch 并登记 store + trace."""
    meta = _src_on(tmp_path, on_source=True)
    async def fake(prompt: str, task: str = "summarize") -> str:
        return f"{RING_SYMBOL} = 42"
    pid = asyncio.run(meta.maybe_propose_source(fake))
    assert pid is not None
    p = sp.SourcePatchStore.get_instance().get(pid)
    assert p.symbol == RING_SYMBOL
    assert p.new_code == f"{RING_SYMBOL} = 42"
    # 未 evaluate/promote → 尚未应用
    assert mi.__dict__[RING_SYMBOL] == _PROPOSE_EVERY_N
    # trace: source_propose
    lines = meta._trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(l)["type"] == "source_propose" for l in lines)


def test_maybe_propose_source_rejects_garbage(tmp_path: Path) -> None:
    """on: LLM 产非赋值/非整数 → 提议被拒."""
    meta = _src_on(tmp_path, on_source=True)
    async def fake_bad(prompt: str, task: str = "summarize") -> str:
        return "just prose, no assignment"
    assert asyncio.run(meta.maybe_propose_source(fake_bad)) is None


def test_source_promote_applies_when_green(tmp_path: Path) -> None:
    """好 source 候选 (verify + sig GREEN) → maybe_promote_source 应用, 且可逆."""
    meta = _src_on(tmp_path, on_source=True)
    for i in range(30):
        meta._replay.append({"phase": "hypothesize", "block_names": ["body"],
                             "r_phys": 0.5, "directive": f"hint {i}",
                             "ts": float(300 + i), "probe_id": f"srcp_{i:02d}"})
    async def fake(prompt: str, task: str = "summarize") -> str:
        return f"{RING_SYMBOL} = 77"

    pid = asyncio.run(meta.maybe_propose_source(fake))
    assert pid is not None
    r = asyncio.run(meta.evaluate_source(pid, fake))
    assert r["green"] is True, f"verify+sig 应 GREEN: {r}"
    assert r["verified"] is True
    orig = mi.__dict__[RING_SYMBOL]
    assert meta.maybe_promote_source(pid) is True
    assert mi.__dict__[RING_SYMBOL] == 77, "GREEN source candidate 应被应用"
    lines = meta._trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(l)["type"] == "source_promote" for l in lines)
    _restore_ring()


def test_note_generation_drives_source_ring(tmp_path: Path) -> None:
    """note_generation 每 _SOURCE_EVERY_N_GENERATIONS 代驱动一次 source 环 (生产入口)."""
    meta = _src_on(tmp_path, on_source=True)
    async def ring_llm(prompt: str, task: str = "summarize") -> str:
        # 多目标轮转: 目标符号从当前 _source_target() 动态取, 保证任一目标都提议成功
        if "self-modifying" in prompt:
            return f"{meta._source_target()[1]} = 88"
        return '{"block_name": "mem", "op": "append", "new_text": "x"}'

    n = sp.__dict__[MECH_SYMBOL]  # 7 — source 环触发周期
    for i in range(n * 2):
        asyncio.run(meta.note_generation("h", [("body", "b"), ("mem", "m")],
                                         0.5, f"hint {i}", ring_llm))
    store = sp.SourcePatchStore.get_instance()
    assert store.list_patches(module=RING_MODULE), "source 环应被 note_generation 驱动"
    _restore_ring()