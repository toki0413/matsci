# 临时: 修复后重跑真实 run_cognitive，验证采集真实 plan→actual。跑完即删，不提交。
import os

import pytest

from huginn.autoloop.engine import AutoloopEngine
from huginn.memory.manager import MemoryManager

from tests.test_autoloop_e2e import (
    _DummyTracker,
    _bypass_validate_gate,
    _noop_maybe_clarify,
    _restore_gate,
)


@pytest.mark.asyncio
async def test_real_intern_collect(tmp_path, monkeypatch):
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s %(message)s",
        force=True,
    )
    for _lp in ("huginn.autoloop.engine_reflect",):
        _L = logging.getLogger(_lp)
        _L.setLevel(logging.DEBUG)
    assert os.environ.get("HUGINN_API_KEY"), "需要 HUGINN_API_KEY 等 HUGINN_* env"
    monkeypatch.setattr("huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None)
    monkeypatch.setattr("huginn.autoloop.conjecture.get_kg", lambda *a, **kw: None)
    monkeypatch.setattr(AutoloopEngine, "_maybe_clarify", _noop_maybe_clarify)

    (tmp_path / "pendulum.py").write_text(
        "import numpy as np\nL, g = 1.0, 9.81\nT = 2*np.pi*np.sqrt(L/g)\n"
        "print(f'T={T:.4f} s')\n",
        encoding="utf-8",
    )
    mem = MemoryManager()
    eng = AutoloopEngine(workspace=str(tmp_path), memory_manager=mem)
    eng.progress_tracker = _DummyTracker()
    eng._use_llm_decider = False
    eng.model_router = None
    eng._perceive = lambda: {
        "changed_files": ["pendulum.py"],
        "git_diff": "",
        "timestamp": "2026-09-11T00:00:00Z",
        "goal": "Compute pendulum period",
    }

    gate = _bypass_validate_gate()
    try:
        result = await eng.run_cognitive(
            objective=(
                "Verify the pendulum period T=2*pi*sqrt(L/g) at L=1.0 m, g=9.81 m/s^2; "
                "predict T numerically BEFORE computing, then compute with numpy and report actual T."
            ),
            max_iterations=2,
            progressive_budget=False,
        )
    finally:
        _restore_gate(gate)

    print("\nPHASES", [(p.name, p.status) for p in result.phases])
    print("CURRENT_PRED", repr(eng._current_prediction)[:200])