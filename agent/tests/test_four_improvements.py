"""Tests for four neuroscience/competitive-research inspired improvements.

P0-1: EventBus ↔ Grader bridge — QUALITY_CHECK event type
P0-2: Memory lint auto_fix — writes contradicts:/crossref: tags
P1-3: PIVOT decision — strategic pivot when refine budget exhausted
P1-4: Tool Retrieval — query-aware top-K tool selection
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ── P0-1: QUALITY_CHECK event type ──────────────────────────────


class TestQualityCheckEvent:
    def test_event_type_exists(self):
        from huginn.events.event_types import QUALITY_CHECK
        assert QUALITY_CHECK == "quality.check"

    def test_in_all_types(self):
        from huginn.events.event_types import ALL_TYPES
        assert "quality.check" in ALL_TYPES


# ── P0-2: Memory lint auto_fix ──────────────────────────────────


class TestMemoryLintAutoFix:
    """Test that lint(auto_fix=True) writes tags back to DB."""

    @pytest.fixture
    def tmp_memory(self, tmp_path):
        from huginn.memory.longterm import LongTermMemory
        db = tmp_path / "test_memory.db"
        mem = LongTermMemory(db_path=db)
        # 添加两个互相矛盾的条目 (同 formula, 不同值)
        mem.store("Fe 的带隙是 0.0 eV", category="fact", tags=["band_gap"], formula="Fe")
        mem.store("Fe 的带隙是 2.1 eV", category="fact", tags=["band_gap"], formula="Fe")
        # 添加一个正常条目
        mem.store("Si 的晶格常数是 5.43 Å", category="fact", tags=["lattice"], formula="Si")
        return mem

    def test_lint_signature_has_auto_fix(self):
        import inspect
        from huginn.memory.longterm import LongTermMemory
        sig = inspect.signature(LongTermMemory.lint)
        assert "auto_fix" in sig.parameters
        assert sig.parameters["auto_fix"].default is False

    def test_lint_reports_contradictions(self, tmp_memory):
        report = tmp_memory.lint()
        assert "contradictions" in report
        assert len(report["contradictions"]) > 0

    def test_auto_fix_writes_tags(self, tmp_memory):
        report = tmp_memory.lint(auto_fix=True)
        assert "auto_fixed" in report
        assert report["auto_fixed"] > 0

    def test_auto_fix_is_idempotent(self, tmp_memory):
        """Running lint(auto_fix=True) twice should fix 0 the second time."""
        tmp_memory.lint(auto_fix=True)
        report2 = tmp_memory.lint(auto_fix=True)
        assert report2["auto_fixed"] == 0


# ── P1-3: PIVOT decision ────────────────────────────────────────


class TestPivotDecision:
    """Test HypothesisGraph.pivot() method."""

    @pytest.fixture
    def graph(self):
        from huginn.autoloop.hypothesis_loop import HypothesisGraph
        return HypothesisGraph()

    def test_pivot_creates_new_node(self, graph):
        # 先加一个假设
        h1 = graph.add_hypothesis("ENCUT=400 足够收敛")
        # pivot
        h2 = graph.pivot(h1, evidence={"error": "convergence not reached"})
        assert h2 is not None
        assert h2 != h1
        # 新节点存在
        assert h2 in graph._nodes

    def test_pivot_creates_edge(self, graph):
        h1 = graph.add_hypothesis("假设 A")
        h2 = graph.pivot(h1, evidence={"error": "failed"})
        # pivot 现在用独立的 pivot edge_type, 不再混入 derive
        edges = [e for e in graph._edges if e.from_id == h1 and e.to_id == h2]
        assert len(edges) >= 1
        assert edges[0].edge_type == "pivot"

    def test_pivot_edge_has_pivot_flag(self, graph):
        h1 = graph.add_hypothesis("假设 A")
        h2 = graph.pivot(h1, evidence={"error": "failed"})
        edge = [e for e in graph._edges if e.from_id == h1 and e.to_id == h2][0]
        # pivot 标记在 edge_type 上, 不在 evidence 里
        assert edge.edge_type == "pivot"

    def test_pivot_without_model_uses_template(self, graph):
        h1 = graph.add_hypothesis("ENCUT=400")
        h2 = graph.pivot(h1, evidence={}, model=None)
        node = graph._nodes[h2]
        # 模板降级: 新假设应该不是空字符串
        assert len(node.statement) > 0

    def test_pivot_new_statement_different(self, graph):
        """Pivot should produce a different statement, not a copy."""
        h1 = graph.add_hypothesis("使用 PBE 泛函计算 Si 带隙")
        h2 = graph.pivot(h1, evidence={"error": "band gap too low"}, model=None)
        node = graph._nodes[h2]
        # 模板降级但仍然包含 pivot 语义
        assert "不同" in node.statement or "pivot" in node.statement.lower() or "排除" in node.statement


# ── P1-4: Tool Retrieval ───────────────────────────────────────


class TestToolRetrieval:
    """Test query-aware tool retrieval in _effective_tools."""

    def _make_mock_tools(self, count: int = 30):
        """Create mock tool objects with name and description."""
        tool_specs = [
            ("vasp_tool", "Run VASP DFT calculation for electronic structure"),
            ("lammps_tool", "Run LAMMPS molecular dynamics simulation"),
            ("structure_tool", "Build and analyze crystal structures"),
            ("memory_tool", "Store and recall memories"),
            ("knowledge_search", "Search the knowledge base"),
            ("periodic_table_tool", "Query periodic table elements"),
            ("search_tool", "Search the web"),
            ("band_structure_tool", "Calculate electronic band structure"),
            ("phonon_tool", "Calculate phonon dispersion"),
            ("elasticity_tool", "Calculate elastic constants tensor"),
            ("xrd_tool", "Predict XRD diffraction pattern"),
            ("molecule_screening_tool", "Screen molecules for properties"),
            ("grain_boundary_tool", "Build grain boundary structures"),
            ("charge_density_tool", "Analyze charge density"),
            ("dos_tool", "Calculate density of states"),
            ("optical_tool", "Calculate optical properties"),
            ("ferroelectric_tool", "Analyze ferroelectric properties"),
            ("surface_tool", "Build surface slab models"),
            ("defect_tool", "Generate defect structures"),
            ("neb_tool", "Run nudged elastic band calculation"),
            ("rdf_tool", "Analyze radial distribution function"),
            ("msd_tool", "Calculate mean square displacement"),
            ("stress_tool", "Analyze stress-strain curve"),
            ("thermo_tool", "Calculate thermodynamic properties"),
            ("wannier_tool", "Wannier interpolation"),
            ("bader_tool", "Bader charge analysis"),
            ("cif_export_tool", "Export CIF crystal structure file"),
            ("plot_tool", "Generate plots and figures"),
            ("convert_tool", "Convert between file formats"),
            ("validate_tool", "Validate calculation results"),
        ]
        tools = []
        for i, (name, desc) in enumerate(tool_specs[:count]):
            tool = type("MockTool", (), {"name": name, "description": desc})()
            tools.append(tool)
        return tools

    def test_retrieval_reduces_tool_count(self):
        """When tools > threshold and query given, fewer tools returned."""
        from huginn.agent.context import ContextMixin

        ctx = ContextMixin.__new__(ContextMixin)
        ctx.langchain_tools = self._make_mock_tools(30)
        ctx._mode = "research"
        ctx._phase_manager = type("P", (), {"tool_filter": staticmethod(lambda: None)})()

        all_tools = ctx._effective_tools()
        assert len(all_tools) == 30  # no query → no retrieval

        filtered = ctx._effective_tools(query="calculate band structure of silicon")
        assert len(filtered) <= 20  # query → retrieval
        assert len(filtered) < 30

    def test_always_on_tools_preserved(self):
        """Always-on tools should appear regardless of query."""
        from huginn.agent.context import ContextMixin

        ctx = ContextMixin.__new__(ContextMixin)
        ctx.langchain_tools = self._make_mock_tools(30)
        ctx._mode = "research"
        ctx._phase_manager = type("P", (), {"tool_filter": staticmethod(lambda: None)})()

        filtered = ctx._effective_tools(query="nonsense query xyz123")
        names = [t.name for t in filtered]
        # memory_tool and search_tool should always be present
        assert "memory_tool" in names
        assert "search_tool" in names

    def test_relevant_tools_ranked_higher(self):
        """Query 'band structure' should keep band_structure_tool."""
        from huginn.agent.context import ContextMixin

        ctx = ContextMixin.__new__(ContextMixin)
        ctx.langchain_tools = self._make_mock_tools(30)
        ctx._mode = "research"
        ctx._phase_manager = type("P", (), {"tool_filter": staticmethod(lambda: None)})()

        filtered = ctx._effective_tools(query="calculate band structure of silicon")
        names = [t.name for t in filtered]
        assert "band_structure_tool" in names
        assert "dos_tool" in names  # related

    def test_no_retrieval_below_threshold(self):
        """Below threshold, all tools returned regardless of query."""
        from huginn.agent.context import ContextMixin

        ctx = ContextMixin.__new__(ContextMixin)
        ctx.langchain_tools = self._make_mock_tools(10)  # < threshold of 25
        ctx._mode = "research"
        ctx._phase_manager = type("P", (), {"tool_filter": staticmethod(lambda: None)})()

        filtered = ctx._effective_tools(query="band structure")
        assert len(filtered) == 10  # all kept

    def test_no_query_returns_all(self):
        """Without a query, no retrieval happens."""
        from huginn.agent.context import ContextMixin

        ctx = ContextMixin.__new__(ContextMixin)
        ctx.langchain_tools = self._make_mock_tools(30)
        ctx._mode = "research"
        ctx._phase_manager = type("P", (), {"tool_filter": staticmethod(lambda: None)})()

        all_tools = ctx._effective_tools(query=None)
        assert len(all_tools) == 30
