"""能力面汇总注入块的单测 —— weightless: 只测 runtime_probe.capability_note.

注: context._capability_note 是同源薄封装 (仅导入隔离); 其逻辑单一来源在这里,
故单测直接覆盖 capability_note, 不 import huginn.agent (会拉 langchain 重栈).
"""

from __future__ import annotations

from huginn.capabilities import runtime_probe as rp


def test_note_empty_when_all_runnable(monkeypatch):
    monkeypatch.setattr(rp, "probe_runtime", lambda *a, **k: [])
    assert rp.capability_note() == ""


def test_note_includes_degraded_line(monkeypatch):
    monkeypatch.setattr(
        rp,
        "probe_runtime",
        lambda *a, **k: [
            rp.ProbeResult(
                name="RDKitTool",
                kind="tool",
                status=rp.STATUS_MISSING,
                missing_deps=["rdkit"],
                install_hint="pip install rdkit",
            )
        ],
    )
    note = rp.capability_note()
    assert note.startswith("\n\n# Tool Capabilities\n")
    assert "Tool capabilities degraded" in note
    assert "rdkit" in note


def test_note_swallows_probe_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(rp, "probe_runtime", boom)
    assert rp.capability_note() == ""
