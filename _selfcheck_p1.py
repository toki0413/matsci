"""自检: 6个P1断层的语法和功能验证."""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "agent"))

errors = []

# P1-1 + P1-2: BoundaryState 字段精简 + 可实例化
try:
    from huginn.constraints.boundaries import BoundaryState, BoundaryEvolution
    state = BoundaryState()
    assert not hasattr(state, "allowed_executables"), "allowed_executables should be deleted"
    assert not hasattr(state, "max_timeout"), "max_timeout should be deleted"
    assert not hasattr(state, "allows"), "allows() should be deleted"
    assert hasattr(state, "require_confirmation")
    assert hasattr(state, "max_retries")
    assert hasattr(state, "blocked_tools")
    assert hasattr(state, "blocked_scopes")
    # BoundaryEvolution.update 仍能工作
    from huginn.constraints.reference import ConstraintResult
    r = ConstraintResult(name="test", passed=False, value=0, expected=">0", tolerance=0.1, message="t", severity="block")
    new_state = BoundaryEvolution(state).update([r])
    assert new_state.require_confirmation is True
    assert new_state.max_retries == 1
    print("[OK] P1-1/P1-2: BoundaryState 精简 + BoundaryEvolution.update 可用")
except Exception as e:
    errors.append(f"P1-1/P1-2: {e}")
    import traceback; traceback.print_exc()
    print(f"[FAIL] P1-1/P1-2: {e}")

# P1-3: SubspacePartition lazy R
try:
    from huginn.metacog.hypothesis_manifold import HypothesisManifold, Hypothesis
    m = HypothesisManifold()
    h1 = Hypothesis(h_id="h1", description="t", predictions={"x": 1.0, "y": 2.0})
    m.add(h1)
    assert m._R is None, f"_R should be None (lazy), got {m._R}"
    assert m._R_dirty is True, "_R_dirty should be True after add"
    h2 = Hypothesis(h_id="h2", description="t", predictions={"x": 3.0, "y": 4.0})
    m.add(h2)
    d = m.fisher_distance("h1", "h2")
    expected = math.sqrt((3-1)**2 + (4-2)**2)
    assert abs(d - expected) < 1e-9, f"fisher_distance wrong: {d} vs {expected}"
    print("[OK] P1-3: SubspacePartition lazy R, fisher_distance 走 O(n) 遍历")
except Exception as e:
    errors.append(f"P1-3: {e}")
    import traceback; traceback.print_exc()
    print(f"[FAIL] P1-3: {e}")

# P1-4: vision_describe_tool check_consistency 自动开启
try:
    src = Path("agent/huginn/tools/vision_describe_tool.py").read_text(encoding="utf-8")
    assert "xrd" in src.lower(), "XRD keyword trigger missing"
    assert "check_consistency = input_data.check_consistency" in src
    assert "if not check_consistency:" in src
    print("[OK] P1-4: vision_describe_tool check_consistency 自动开启逻辑存在")
except Exception as e:
    errors.append(f"P1-4: {e}")
    print(f"[FAIL] P1-4: {e}")

# P1-5: streaming.py 读 _effective_recursion_limit
try:
    src = Path("agent/huginn/agent/streaming.py").read_text(encoding="utf-8")
    assert "_effective_recursion_limit()" in src, "streaming.py must call _effective_recursion_limit()"
    assert "max(self._effective_recursion_limit()" in src
    print("[OK] P1-5: streaming.py 读 _effective_recursion_limit() 联动修复")
except Exception as e:
    errors.append(f"P1-5: {e}")
    print(f"[FAIL] P1-5: {e}")

# P1-6: reflection.py 改调 switch_cognitive_mode
try:
    src = Path("agent/huginn/agent/reflection.py").read_text(encoding="utf-8")
    assert "switch_cognitive_mode" in src, "reflection.py must call switch_cognitive_mode"
    assert "from huginn.session_state import CognitiveMode" in src
    assert "self.set_mode(reflection.suggested_mode)" not in src, "old set_mode call should be removed"
    print("[OK] P1-6: reflection.py 改调 switch_cognitive_mode")
except Exception as e:
    errors.append(f"P1-6: {e}")
    print(f"[FAIL] P1-6: {e}")

# P1-1 验证: agent/core.py 注入 boundary_state
try:
    src = Path("agent/huginn/agent/core.py").read_text(encoding="utf-8")
    assert "self._boundary_state = BoundaryState()" in src, "core.py must instantiate BoundaryState"
    count = src.count("boundary_state=self._boundary_state")
    assert count >= 2, f"core.py must pass boundary_state to adapt() in 2 places, got {count}"
    print(f"[OK] P1-1: agent/core.py 注入 boundary_state 到 {count} 处 adapt() 调用")
except Exception as e:
    errors.append(f"P1-1 verify: {e}")
    print(f"[FAIL] P1-1 verify: {e}")

print()
if errors:
    print(f"FAILED: {len(errors)} checks")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL P1 CHECKS PASSED")
