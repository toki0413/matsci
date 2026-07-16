"""Hard checkpoint gate + skills retention self-check.

Two features landing together:
1. HUGINN_HARD_CHECKPOINT_PHASES — checkpoint 不可超时自动放行
2. _trim_to_budget Pass 3 — skill/composite block 不可清空
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock


def test_hard_checkpoint_config_and_state(monkeypatch):
    monkeypatch.setenv("HUGINN_HARD_CHECKPOINT_PHASES", "plan:execute,validate:learn")
    from huginn.autoloop.phase_gate import (
        _load_hard_checkpoint_config,
        PhaseGateState,
    )

    cfg = _load_hard_checkpoint_config()
    assert cfg == {("plan", "execute"), ("validate", "learn")}

    state = PhaseGateState()
    assert state.is_hard_checkpoint("plan", "execute") is True
    assert state.is_hard_checkpoint("hypothesize", "plan") is False
    # 硬门也算 human checkpoint
    assert state.needs_human_checkpoint("plan", "execute") is True
    # override 仍能覆盖 (option to force proceed)
    state.overrides.add(("plan", "execute"))
    assert state.needs_human_checkpoint("plan", "execute") is False


def test_trim_to_budget_protects_skills(monkeypatch):
    import huginn.autoloop.engine as em

    monkeypatch.setattr(em, "get_model", lambda s: MagicMock())
    monkeypatch.setattr(em, "MemoryManager", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(em, "ProjectKnowledgeGraph", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(em, "BenchmarkRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(em, "CoderRunner", lambda *a, **kw: MagicMock())
    eng = em.AutoloopEngine(workspace=Path("."))
    eng._PROMPT_BUDGET = 200  # 逼 Pass 3 删除

    blocks = [
        ("visual", "x" * 50),
        ("skill", "AVAILABLE SKILLS: band_structure, phonon_calc, dos_plot"),
        ("composite", "COMPOSITE: vasp_workflow"),
        ("body", "y" * 150),
    ]
    result = eng._trim_to_budget(blocks)
    assert "AVAILABLE SKILLS" in result, f"skill block lost: {result[:120]}"
    assert "COMPOSITE" in result, f"composite block lost: {result[:120]}"
