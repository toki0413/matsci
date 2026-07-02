"""KB 全链路接入测试 (Phase A).

锁住 6 个新接入点:
  A1. _validate reviewer prompt 注入 KB
  A2. _run_math_validation 写入 reference_principles
  A3. _symreg_hint 拿 KB candidate forms
  A4. _report tutor prompt + report 文件含 Domain Knowledge References
  A5. gap_analysis_tool 标注 kb_covered
  A6. symbolic_math dft/thermodynamics 加 kb_verified_constants

全部用 _FakeKb 替身隔离真实 ChromaDB.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine
from huginn.types import ToolContext, ToolResult


class _FakeKb:
    """最小 KB 替身: 只实现 count() 和 query()."""

    def __init__(self, chunks: list[dict] | None = None):
        self._chunks = chunks or []

    def count(self) -> int:
        return len(self._chunks)

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        return self._chunks[:top_k]


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    """sub-components 全 stub 的 engine, 仿 test_kb_integration."""
    monkeypatch.setattr(
        "huginn.autoloop.engine.get_model", lambda settings: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.MemoryManager", lambda: MagicMock())
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda: MagicMock())
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    return AutoloopEngine(workspace=tmp_path)


class TestA1ValidateKbInjection:
    """A1: _validate reviewer prompt 注入 KB."""

    def test_reviewer_prompt_includes_kb_when_available(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeKb([{"text": "DFT ENCUT > 520 eV for convergence."}])
        monkeypatch.setattr("huginn.knowledge.store.get_knowledge_base", lambda ws: fake)
        prompt = engine._build_reviewer_prompt(
            execution_result={"result_type": "dft"},
            results={"tests_passed": True},
            kb_text=engine._build_kb_text("dft convergence"),
        )
        assert "Domain Knowledge Context" in prompt
        assert "ENCUT" in prompt

    def test_reviewer_prompt_skips_kb_when_empty(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "huginn.knowledge.store.get_knowledge_base", lambda ws: _FakeKb([])
        )
        prompt = engine._build_reviewer_prompt(
            execution_result={},
            results={},
            kb_text=engine._build_kb_text("anything"),
        )
        assert "Domain Knowledge Context" not in prompt

    def test_summarize_for_kb_extracts_keys(self, engine: AutoloopEngine) -> None:
        summary = engine._summarize_for_kb(
            execution_result={"result_type": "dft", "equations": "E = mc^2"},
            results={"tests_passed": True},
        )
        assert "dft" in summary
        assert "E = mc^2" in summary
        assert "tests_passed=True" in summary


class TestA2MathValidationKbReference:
    """A2: _run_math_validation 写入 reference_principles."""

    @pytest.mark.asyncio
    async def test_reference_principles_written_when_kb_has_content(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeKb([{"text": "Mass conservation: d rho/dt + div(rho v) = 0"}])
        monkeypatch.setattr("huginn.knowledge.store.get_knowledge_base", lambda ws: fake)
        # BourbakiTool 调用会失败 (没装), 但 KB 查询独立走, 仍应写 reference_principles
        result = await engine._run_math_validation(
            execution_result={"equations": "mass conservation"}
        )
        assert "reference_principles" in result
        assert len(result["reference_principles"]) >= 1
        assert "Mass conservation" in result["reference_principles"][0]["text"]

    @pytest.mark.asyncio
    async def test_reference_principles_absent_when_kb_empty(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "huginn.knowledge.store.get_knowledge_base", lambda ws: _FakeKb([])
        )
        result = await engine._run_math_validation(
            execution_result={"equations": "mass conservation"}
        )
        assert "reference_principles" not in result

    def test_query_kb_reference_returns_empty_on_no_query(self, engine: AutoloopEngine) -> None:
        assert engine._query_kb_reference("", "") == []


class TestA3SymregHintKbCandidate:
    """A3: _symreg_hint 拿 KB candidate forms."""

    @pytest.mark.asyncio
    async def test_kb_candidate_forms_returned_when_symreg_fails(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeKb([{"text": "Arrhenius: k = A exp(-Ea/RT)"}])
        monkeypatch.setattr("huginn.knowledge.store.get_knowledge_base", lambda ws: fake)
        # observation_data 不含 target_column → symreg 直接退回 KB forms
        hint = await engine._symreg_hint(
            context={"observation_data": {"y": [1, 2, 3]}}
        )
        assert "KB candidate forms" in hint
        assert "Arrhenius" in hint

    @pytest.mark.asyncio
    async def test_symreg_hint_empty_when_kb_empty(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "huginn.knowledge.store.get_knowledge_base", lambda ws: _FakeKb([])
        )
        # data 不全 (没 target_column) 且 KB 空 → 空串
        hint = await engine._symreg_hint(
            context={"observation_data": {"y": [1, 2, 3]}}
        )
        assert hint == ""


class TestA4ReportKbCitation:
    """A4: _report tutor prompt + report 文件含 Domain Knowledge References."""

    def test_tutor_prompt_includes_kb_when_available(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeKb([{"text": "C-S-H gel density ~2.6 g/cm3."}])
        monkeypatch.setattr("huginn.knowledge.store.get_knowledge_base", lambda ws: fake)
        kb_text = engine._build_kb_text("C-S-H density")
        prompt = engine._build_tutor_report_prompt(
            report_data={
                "objective": "study C-S-H",
                "total_time_seconds": 10.0,
                "phases": [],
            },
            kb_text=kb_text,
        )
        assert "Domain Knowledge Context" in prompt
        assert "C-S-H" in prompt

    def test_tutor_prompt_skips_kb_when_empty(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "huginn.knowledge.store.get_knowledge_base", lambda ws: _FakeKb([])
        )
        prompt = engine._build_tutor_report_prompt(
            report_data={
                "objective": "x",
                "total_time_seconds": 1.0,
                "phases": [],
            },
            kb_text="",
        )
        assert "Domain Knowledge Context" not in prompt


class TestA5GapAnalysisKbCoverage:
    """A5: gap_analysis_tool 标注 kb_covered."""

    @pytest.mark.asyncio
    async def test_gap_marked_kb_covered_when_kb_has_content(self) -> None:
        from huginn.tools.design.gap_analysis_tool import (
            GapAnalysisInput,
            GapAnalysisTool,
        )

        fake = _FakeKb([{"text": "DFT PBE band gap of Si is 0.6 eV."}])
        # context 携带 workspace, KB lookup 走 monkeypatch
        ctx = ToolContext(
            session_id="t1", workspace=".", config=None
        )
        tool = GapAnalysisTool()
        # 直接 patch _query_kb_coverage 返回非空
        monkeypatch_target = (
            "huginn.tools.design.gap_analysis_tool.GapAnalysisTool._query_kb_coverage"
        )
        import unittest.mock as _um

        with _um.patch(
            monkeypatch_target,
            return_value=[{"text": "DFT PBE band gap of Si is 0.6 eV.", "source": "kb"}],
        ):
            vr = await tool.call(
                {
                    "action": "analyze_gaps",
                    "topic": "Si band gap",
                    "papers": [
                        {
                            "title": "Paper A",
                            "methods": "dft",
                            "tags": "Si",
                            "results": "improve band gap",
                        }
                    ],
                },
                ctx,
            )
        assert vr.success
        gaps = vr.data["gaps"]
        assert any(g.get("kb_covered") is True for g in gaps)
        assert vr.data["n_kb_covered"] >= 1

    @pytest.mark.asyncio
    async def test_gap_not_marked_when_kb_empty(self) -> None:
        from huginn.tools.design.gap_analysis_tool import (
            GapAnalysisInput,
            GapAnalysisTool,
        )

        tool = GapAnalysisTool()
        ctx = ToolContext(session_id="t2", workspace=".", config=None)
        import unittest.mock as _um

        # 显式 mock KB 查询返回空, 模拟 KB 不可用场景
        with _um.patch(
            "huginn.tools.design.gap_analysis_tool.GapAnalysisTool._query_kb_coverage",
            return_value=[],
        ):
            vr = await tool.call(
                {
                    "action": "analyze_gaps",
                    "topic": "X",
                    "papers": [
                        {
                            "title": "P",
                            "methods": "dft",
                            "tags": "Si",
                            "results": "improve",
                        }
                    ],
                },
                ctx,
            )
        assert vr.success
        # KB 不可用 → 所有 gap kb_covered=False
        assert all(g.get("kb_covered") is False for g in vr.data["gaps"])
        assert vr.data["n_kb_covered"] == 0


class TestA6SymbolicMathKbConstants:
    """A6: symbolic_math dft/thermodynamics 加 kb_verified_constants."""

    @pytest.mark.asyncio
    async def test_dft_result_gets_kb_constants(self) -> None:
        from huginn.tools.symbolic_math.tool import SymbolicMathInput, SymbolicMathTool

        tool = SymbolicMathTool()
        ctx = ToolContext(session_id="t3", workspace=".", config=None)
        import unittest.mock as _um

        with _um.patch(
            "huginn.knowledge.store.get_knowledge_base",
            return_value=_FakeKb([{"text": "hbar = 1.054e-34 J*s"}]),
        ):
            result = await tool.call(
                SymbolicMathInput(action="dft", target="fermi_energy", expression="n=0.05"),
                ctx,
            )
        assert result.success
        assert "kb_verified_constants" in result.data

    @pytest.mark.asyncio
    async def test_thermodynamics_result_gets_kb_constants(self) -> None:
        from huginn.tools.symbolic_math.tool import SymbolicMathInput, SymbolicMathTool

        tool = SymbolicMathTool()
        ctx = ToolContext(session_id="t4", workspace=".", config=None)
        import unittest.mock as _um

        with _um.patch(
            "huginn.knowledge.store.get_knowledge_base",
            return_value=_FakeKb([{"text": "R = 8.314 J/(mol K)"}]),
        ):
            result = await tool.call(
                SymbolicMathInput(action="thermodynamics", target="ideal_gas"),
                ctx,
            )
        assert result.success
        assert "kb_verified_constants" in result.data

    @pytest.mark.asyncio
    async def test_dft_no_kb_constants_when_kb_empty(self) -> None:
        from huginn.tools.symbolic_math.tool import SymbolicMathInput, SymbolicMathTool

        tool = SymbolicMathTool()
        ctx = ToolContext(session_id="t5", workspace=".", config=None)
        import unittest.mock as _um

        with _um.patch(
            "huginn.knowledge.store.get_knowledge_base",
            return_value=_FakeKb([]),
        ):
            result = await tool.call(
                SymbolicMathInput(action="dft", target="fermi_energy"),
                ctx,
            )
        assert result.success
        assert "kb_verified_constants" not in result.data
