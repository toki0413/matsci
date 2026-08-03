"""剩余断点/死代码补齐 selfcheck — 验证桥 F-K.

桥 F: predict_via_analogy 返回 analogy 格式
桥 G: get_skill_context 注入 hypothesis prompt
桥 H: PMK Memory 路含 failed_directions
桥 I: evolution rules 含 stable_principles
桥 J: EncounterSpace 编码含 surprise
桥 K: _hypothesize 接 recommend 的 avoid_directions
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

_TMPDIR = Path(tempfile.mkdtemp(prefix="loop_remainder_"))


def _run():
    print("=== Loop Remainder Selfcheck ===\n")

    # 桥 F: predict_via_analogy
    print("--- Bridge F: predict_via_analogy ---")
    from huginn.memory.longterm import LongTermMemory
    m = LongTermMemory(enable_semantic=False, db_path=str(_TMPDIR / "test.db"))
    # 空 hypothesis
    r = m.predict_via_analogy({"hypothesis": ""})
    assert r["prediction_type"] == "none", f"empty hypothesis should return none, got {r}"
    assert r["analogy"] == [], "empty hypothesis should return empty analogy"
    print("  F: empty hypothesis → none OK")

    # 写入一条记忆后检索
    m.store(content="GaN bandgap is 3.4 eV", category="fact", importance=0.9)
    r2 = m.predict_via_analogy({"hypothesis": "GaN bandgap"}, top_k=2)
    assert r2["prediction_type"] == "analogy", f"should return analogy, got {r2}"
    assert len(r2["analogy"]) > 0, "should have at least one analogy result"
    assert "content" in r2["analogy"][0], "analogy item should have content"
    assert "score" in r2["analogy"][0], "analogy item should have score"
    print(f"  F: non-empty retrieve → analogy OK ({len(r2['analogy'])} results)")

    # 验证 _build_world_model_block 格式兼容
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.memory = SimpleNamespace(longterm=m)
    os.environ["HUGINN_WORLD_MODEL"] = "1"
    try:
        block = eng._build_world_model_block("GaN bandgap")
        # 有记忆时应该返回非空 block (或空如果 threshold 过滤掉了, 都不崩)
        assert isinstance(block, str), f"block should be str, got {type(block)}"
        print(f"  F: _build_world_model_block format OK (len={len(block)})")
    finally:
        os.environ.pop("HUGINN_WORLD_MODEL", None)

    # 桥 G: get_skill_context 注入
    print("--- Bridge G: SkillEvolutionLayer → prompt ---")
    # 验证 _build_skill_context_block 存在且 flag off 返空串
    assert hasattr(AutoloopEngine, "_build_skill_context_block"), \
        "engine should have _build_skill_context_block"
    os.environ.pop("HUGINN_SKILL_CONTEXT", None)
    block_g = eng._build_skill_context_block()
    assert block_g == "", f"flag off should return empty string, got {block_g!r}"
    print("  G: flag off → empty string OK")

    # flag on 时调 get_skill_context (可能返空串, 但不崩)
    os.environ["HUGINN_SKILL_CONTEXT"] = "1"
    try:
        block_g2 = eng._build_skill_context_block()
        assert isinstance(block_g2, str), f"block should be str, got {type(block_g2)}"
        print(f"  G: flag on → get_skill_context called OK (len={len(block_g2)})")
    finally:
        os.environ.pop("HUGINN_SKILL_CONTEXT", None)

    # 验证 hypothesis prompt blocks 含 skill
    source = inspect.getsource(AutoloopEngine._build_hypothesis_prompt)
    assert '"skill"' in source and "_build_skill_context_block" in source, \
        "hypothesis prompt should inject skill block"
    print("  G: hypothesis prompt injects skill block OK")

    # 桥 H: PMK Memory 路含 failed_directions
    print("--- Bridge H: failed_directions → PMK Memory ---")
    from huginn.autoloop.cognitive_loop import build_pmk_state
    # 验证源码含 recall_failed_directions 和 recall_procedural
    cl_source = inspect.getsource(build_pmk_state)
    assert "recall_failed_directions" in cl_source, \
        "build_pmk_state should call recall_failed_directions"
    assert "recall_procedural" in cl_source, \
        "build_pmk_state should call recall_procedural"
    print("  H: build_pmk_state calls recall_failed_directions + recall_procedural OK")

    # 桥 I: evolution rules 含 stable_principles
    print("--- Bridge I: stable_principles → evolution rules ---")
    from huginn.evolution.engine import EvolutionEngine, EvolutionRule
    evo_source = inspect.getsource(EvolutionEngine.evolve_from_failures)
    assert "load_stable_principles" in evo_source, \
        "evolve_from_failures should call load_stable_principles"
    assert "stable_principle" in evo_source, \
        "evolve_from_failures should create stable_principle rules"
    print("  I: evolve_from_failures loads stable_principles OK")

    # 桥 J: EncounterSpace 编码含 surprise
    print("--- Bridge J: EncounterSpace surprise ---")
    from huginn.metacog.encounter_space import EncounterSpace
    es = EncounterSpace()
    assert es.dim == 21, f"dim should be 21 (16+5), got {es.dim}"
    # 无 surprise 字段 → 默认 0
    v1 = es.encode({"iter": 1, "mode": "explore", "phase": "hypothesize", "val_status": "none"})
    assert v1.shape == (21,), f"vector shape should be (21,), got {v1.shape}"
    assert v1[-1] == 0.0, f"surprise dim should be 0.0 when absent, got {v1[-1]}"
    # 有 surprise 字段 → 编码
    v2 = es.encode({"iter": 1, "mode": "explore", "phase": "hypothesize",
                    "val_status": "none", "surprise": 0.8})
    assert v2[-1] == 0.8, f"surprise dim should be 0.8, got {v2[-1]}"
    # clip 测试
    v3 = es.encode({"surprise": 5.0})
    assert v3[-1] == 1.0, f"surprise should clip to 1.0, got {v3[-1]}"
    print(f"  J: EncounterSpace dim={es.dim}, surprise encoded + clipped OK")

    # 验证 _build_episodic_replay_block cue 含 surprise
    replay_source = inspect.getsource(AutoloopEngine._build_episodic_replay_block)
    assert "_last_surprise" in replay_source and '"surprise"' in replay_source, \
        "_build_episodic_replay_block cue should contain surprise"
    print("  J: _build_episodic_replay_block cue contains surprise OK")

    # 桥 K: _hypothesize 接 recommend
    print("--- Bridge K: recommend → _hypothesize ---")
    from huginn.autoloop.hypothesis_loop import HypothesisMixin
    hyp_source = inspect.getsource(HypothesisMixin._hypothesize)
    assert "EvolutionManager" in hyp_source and "recommend" in hyp_source, \
        "_hypothesize should call EvolutionManager.recommend"
    assert "avoid_directions" in hyp_source, \
        "_hypothesize should use avoid_directions"
    print("  K: _hypothesize calls recommend + uses avoid_directions OK")

    # 验证 recommend 本身可调 (flag off 返空 Recommendation)
    from huginn.evolution.manager import EvolutionManager, _use_evolution_manager
    os.environ["HUGINN_USE_EVOLUTION_MANAGER"] = "1"
    os.environ.pop("HUGINN_DISABLE_EVOLUTION_MANAGER", None)
    mgr = EvolutionManager.shared(None)
    rec = mgr.recommend(hypothesis_context={"test": True})
    assert hasattr(rec, "avoid_directions"), "recommend should return Recommendation"
    assert isinstance(rec.avoid_directions, list), "avoid_directions should be list"
    print(f"  K: recommend() callable, avoid_directions={len(rec.avoid_directions)} items OK")

    # 清理
    shutil.rmtree(_TMPDIR, ignore_errors=True)
    print("\n=== Loop Remainder Selfcheck ALL OK (F/G/H/I/J/K) ===")


if __name__ == "__main__":
    _run()
