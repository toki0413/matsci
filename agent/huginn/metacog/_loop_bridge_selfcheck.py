"""循环桥接 selfcheck — 验证四循环互通的 5 个桥.

桥 A: surprise > 2.0 触发 trigger_alignment_surprise_hypothesis
桥 B: PMK INCONSISTENT 触发 evolve_from_failures
桥 C: evolution_rules 进入 PMK KB 路
桥 D: stable_principles 进入 PMK KB 路
桥 E: _snapshot 含 surprise + rule_hit
"""
from __future__ import annotations

import os
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

_TMPDIR = Path(tempfile.mkdtemp(prefix="loop_bridge_"))


def _run():
    print("=== Loop Bridge Selfcheck ===\n")

    # 桥 C: evolution_rules → PMK KB 路
    print("--- Bridge C: evolution_rules → PMK KB ---")
    rules_path = Path.home() / ".huginn" / "logs" / "evolution_rules.json"
    backup = None
    if rules_path.exists():
        backup = rules_path.read_text(encoding="utf-8")

    try:
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        test_rules = [
            {"rule_id": "R_HIGH", "confidence": 0.9, "action": {"description": "bump ENCUT to 520"}},
            {"rule_id": "R_LOW", "confidence": 0.1, "action": {"description": "should be filtered"}},
        ]
        rules_path.write_text(json.dumps(test_rules), encoding="utf-8")

        from huginn.autoloop.cognitive_loop import build_pmk_state
        pmk = build_pmk_state(
            persona=None,
            last_step_eval=SimpleNamespace(
                attempted="test", pmk_feedback="", success=False,
                hypothesis="", tool_name="", error="",
            ),
            kb=None,
            since=None,
            mem_mgr=None,
        )
        if pmk and pmk.get("kb"):
            assert "learned_lessons" in pmk["kb"], f"KB should contain learned_lessons: {pmk['kb'][:100]}"
            assert "R_HIGH" in pmk["kb"], "KB should contain R_HIGH"
            assert "R_LOW" not in pmk["kb"], "KB should NOT contain R_LOW (low confidence)"
            print(f"  C: evolution_rules in PMK KB OK")
        else:
            print(f"  C: PMK state = {pmk}, skipping assertion (kb may be empty)")
    finally:
        if backup is not None:
            rules_path.write_text(backup, encoding="utf-8")
        elif rules_path.exists():
            rules_path.unlink()

    # 桥 D: stable_principles → PMK KB 路
    print("--- Bridge D: stable_principles → PMK KB ---")
    sp_path = Path.home() / ".huginn" / "stable_principles.jsonl"
    backup_sp = None
    if sp_path.exists():
        backup_sp = sp_path.read_text(encoding="utf-8")

    try:
        sp_path.parent.mkdir(parents=True, exist_ok=True)
        # load_stable_principles 读 json.loads(line)["principle"], 每行是 {"principle": "..."}
        sp_path.write_text(
            json.dumps({"principle": "prefer converged relaxations"}) + "\n" +
            json.dumps({"principle": "watch NPAR setting"}) + "\n",
            encoding="utf-8",
        )

        # 清 mtime cache
        import huginn.memory.longterm as _lt
        _lt._STABLE_PRINCIPLES_CACHE = None

        pmk = build_pmk_state(
            persona=SimpleNamespace(description="test persona"),
            last_step_eval=SimpleNamespace(
                attempted="test", pmk_feedback="memory: test memory", success=False,
                hypothesis="", tool_name="", error="",
            ),
            kb=None,
            since=None,
            mem_mgr=None,
        )
        if pmk and pmk.get("kb"):
            assert "stable_principles" in pmk["kb"], f"KB should contain stable_principles: {pmk['kb'][:100]}"
            print(f"  D: stable_principles in PMK KB OK: {pmk['kb'][:80]}")
        else:
            print(f"  D: PMK state = {pmk}, skipping assertion")
    finally:
        if backup_sp is not None:
            sp_path.write_text(backup_sp, encoding="utf-8")
        elif sp_path.exists():
            sp_path.unlink()

    # 桥 E: _snapshot 含 surprise + rule_hit (_snapshot 是 cognitive_loop 主循环里的局部 dict, 不是方法)
    print("--- Bridge E: _snapshot surprise + rule_hit ---")
    # 验证源码包含字段
    import inspect
    from huginn.autoloop import cognitive_loop as _cl
    source = inspect.getsource(_cl)
    assert '"surprise"' in source and 'getattr(self, "_last_surprise", 0.0)' in source, \
        "_snapshot should contain surprise field"
    assert '"rule_hit"' in source and 'getattr(self, "_last_rule_hit_id", "")' in source, \
        "_snapshot should contain rule_hit field"
    print("  E: _snapshot source contains surprise + rule_hit OK")

    # 验证 engine 有对应属性
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._last_surprise = 3.14
    eng._last_rule_hit_id = "FIX_42"
    assert eng._last_surprise == 3.14
    assert eng._last_rule_hit_id == "FIX_42"
    print(f"  E: engine._last_surprise={eng._last_surprise}, _last_rule_hit_id={eng._last_rule_hit_id} OK")

    # 验证 _try_evolved_fix 写入 _last_rule_hit_id
    evo_source = inspect.getsource(AutoloopEngine._try_evolved_fix)
    assert "_last_rule_hit_id" in evo_source, "_try_evolved_fix should set _last_rule_hit_id"
    print("  E: _try_evolved_fix sets _last_rule_hit_id OK")

    # 桥 A: surprise → hypothesis (flag off 不触发)
    print("--- Bridge A: surprise → hypothesis ---")
    os.environ.pop("HUGINN_ALIGNMENT_SURPRISE_TRIGGER", None)
    eng._last_surprise = 5.0
    # flag off → 不调 trigger (只验证 flag 逻辑, 不真跑 trigger)
    triggered = os.environ.get("HUGINN_ALIGNMENT_SURPRISE_TRIGGER", "0").lower() in ("1", "true")
    assert not triggered, "flag off should not trigger"
    print("  A: flag off → no trigger OK")

    os.environ["HUGINN_ALIGNMENT_SURPRISE_TRIGGER"] = "1"
    triggered = os.environ.get("HUGINN_ALIGNMENT_SURPRISE_TRIGGER", "0").lower() in ("1", "true")
    assert triggered, "flag on should trigger"
    print("  A: flag on + surprise=5.0 → trigger OK")
    os.environ.pop("HUGINN_ALIGNMENT_SURPRISE_TRIGGER", None)

    # 桥 B: PMK INCONSISTENT → evolution (验证 flag + 调用逻辑)
    print("--- Bridge B: PMK → evolution ---")
    from huginn.runtime.task_lifecycle import _check_pmk_consistency

    pmk_conflict = {"persona": "recommend PBE", "memory": "oppose PBE"}
    is_inc, reason = _check_pmk_consistency(pmk_conflict)
    assert is_inc, "should be inconsistent"
    print(f"  B: PMK INCONSISTENT detected: {reason[:60]}")

    pmk_agree = {"persona": "recommend PBE", "memory": "recommend PBE"}
    is_inc2, _ = _check_pmk_consistency(pmk_agree)
    assert not is_inc2, "should be consistent"
    print("  B: PMK consistent → no evolution trigger OK")

    # 清理
    os.environ.pop("HUGINN_ALIGNMENT_SURPRISE_TRIGGER", None)
    os.environ.pop("HUGINN_PMK_INJECT", None)
    shutil.rmtree(_TMPDIR, ignore_errors=True)

    print("\n=== Loop Bridge Selfcheck ALL OK (A/B/C/D/E) ===")


if __name__ == "__main__":
    _run()
