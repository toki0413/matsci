"""Integration tests for the enhanced modules.

Covers telemetry memory tracking, context compaction, model-aware token
counting, AutoDiff JAX paths, descriptor graceful degradation, UQ PCE/Morris,
GP Matern kernels / UCB, numerical solvers, symmetry analysis, and unit
tooling.  Optional dependencies are skipped gracefully.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import fields as dataclass_fields
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from huginn.telemetry import TelemetryCollector, TelemetrySpan
from huginn.tools.autodiff_tool import AutoDiffInput, AutoDiffTool
from huginn.tools.descriptor_tool import DescriptorTool
from huginn.tools.gp_tool import GPTool
from huginn.tools.numerical_tool import NumericalTool
from huginn.tools.symmetry_tool import SymmetryTool
from huginn.tools.uq_tool import UQTool
from huginn.tools.unit_tool import UnitTool
from huginn.types import ToolContext
from huginn.utils.context import compact_messages, summarize_compact_messages
from huginn.utils.tokens import count_message_tokens, count_tokens

CTX = ToolContext(session_id="test", workspace=".")


# ──────────────────────────────────────────────────────────────────────
# 1. Telemetry memory tracking
# ──────────────────────────────────────────────────────────────────────


class TestTelemetryMemory:
    def test_span_has_memory_fields(self):
        """TelemetrySpan must expose the memory tracking fields."""
        field_names = {f.name for f in dataclass_fields(TelemetrySpan)}
        assert "memory_start_mb" in field_names
        assert "memory_end_mb" in field_names
        assert "memory_peak_mb" in field_names

    def test_memory_snapshot_returns_rss(self):
        """memory_snapshot() should always include rss_mb."""
        collector = TelemetryCollector()
        snap = collector.memory_snapshot()
        assert isinstance(snap, dict)
        assert "rss_mb" in snap
        assert isinstance(snap["rss_mb"], float)

    def test_summary_includes_memory_stats(self):
        """summary() should report per-span memory delta and peak."""
        collector = TelemetryCollector()
        with collector.span("test_op"):
            # allocate a bit so memory numbers are non-trivial
            _ = bytearray(1024 * 1024)
        summary = collector.summary()
        assert "by_name" in summary
        assert "test_op" in summary["by_name"]
        entry = summary["by_name"]["test_op"]
        assert "avg_memory_delta_mb" in entry
        assert "max_memory_peak_mb" in entry


# ──────────────────────────────────────────────────────────────────────
# 2. Context compaction optimization
# ──────────────────────────────────────────────────────────────────────


class TestContextCompaction:
    def test_compact_messages_linear_performance(self):
        """compact_messages should handle large lists without O(n^2) blowup."""
        # 5000 messages is enough to expose an O(n^2) regression.
        messages = [
            {"role": "user", "content": f"message number {i} " * 20}
            for i in range(5000)
        ]
        start = time.perf_counter()
        result = compact_messages(messages, budget_tokens=500, keep_last_n=4)
        elapsed = time.perf_counter() - start

        assert len(result) <= len(messages)
        # Should finish well under 2s — the old pop-and-recount loop would
        # take orders of magnitude longer on a list this size.
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_summarize_caps_long_summary(self):
        """When the summarizer returns a huge string it should be re-sumpressed."""
        long_text = "detail " * 4000  # well past the 2000-token cap
        short_text = "compressed summary"

        call_count = {"n": 0}

        async def mock_summarizer(transcript: str) -> str:
            call_count["n"] += 1
            # First call produces the oversized summary, second call (the
            # compression pass) returns something manageable.
            if call_count["n"] == 1:
                return long_text
            return short_text

        messages = [
            {"role": "user", "content": f"old message {i} " * 30}
            for i in range(20)
        ] + [
            {"role": "assistant", "content": "recent reply " * 30}
            for _ in range(4)
        ]

        compacted, summary_text = await summarize_compact_messages(
            messages,
            budget_tokens=100,
            keep_last_n=4,
            summarizer=mock_summarizer,
        )

        # The summarizer must have been invoked at least twice — once for
        # the initial summary and once for the compression pass.
        assert call_count["n"] >= 2
        # Final summary should be the compressed version, not the long one.
        assert summary_text == short_text
        assert len(compacted) < len(messages)


# ──────────────────────────────────────────────────────────────────────
# 3. Token counting model-aware
# ──────────────────────────────────────────────────────────────────────


class TestTokenCounting:
    def test_count_tokens_backward_compat(self):
        """count_tokens with no model should still work (cl100k_base default)."""
        n = count_tokens("hello world")
        assert isinstance(n, int)
        assert n > 0

    def test_count_tokens_model_aware(self):
        """Different model families should be able to produce different counts.

        'hello world' happens to tokenize identically across cl100k_base and
        o200k_base, so we use CJK text (which the encoders split differently)
        to confirm the model_name parameter actually switches encoders.
        """
        default_count = count_tokens("hello world")
        gpt4o_count = count_tokens("hello world", model_name="gpt-4o")
        assert isinstance(gpt4o_count, int)
        assert gpt4o_count > 0

        # CJK text is where the two vocabularies diverge.
        cjk = "你好世界"
        cl100k = count_tokens(cjk)
        o200k = count_tokens(cjk, model_name="gpt-4o")
        assert cl100k != o200k, (
            f"Expected different token counts for CJK text, got {cl100k} vs {o200k}"
        )
        # Sanity: the default path should also match the no-arg call.
        assert default_count == count_tokens("hello world")

    def test_count_message_tokens_with_o1(self):
        """count_message_tokens should accept o1 model_name (o200k_base)."""
        n = count_message_tokens("hello", model_name="o1")
        assert isinstance(n, int)
        assert n > 0
        # +4 tokens for role/separators overhead
        assert n >= 4


# ──────────────────────────────────────────────────────────────────────
# 4. AutoDiffTool JAX jacobian
# ──────────────────────────────────────────────────────────────────────


class TestAutoDiffJAX:
    @pytest.fixture
    def tool(self):
        return AutoDiffTool()

    def test_jacobian_vector_function(self, tool):
        """Jacobian of f(x)=[x0^2, x1^3] should give the right shape and values."""
        if not tool._jax_available:
            pytest.skip("jax not installed")

        # Register a vector-valued function the tool can call.
        def vec_fn(x0, x1):
            return [x0 ** 2, x1 ** 3]

        tool._built_in_functions["test_vec"] = vec_fn

        result = asyncio.run(
            tool.call(
                AutoDiffInput(
                    action="jacobian",
                    function_type="test_vec",
                    variables={"x0": [2.0], "x1": [3.0]},
                    use_jax=True,
                ),
                CTX,
            )
        )
        assert result.success, result.error
        jac = result.data["jacobian"]
        # df/dx0 = [2*x0, 0] = [4, 0]
        assert jac["x0"][0] == pytest.approx(4.0, abs=1e-4)
        assert jac["x0"][1] == pytest.approx(0.0, abs=1e-4)
        # df/dx1 = [0, 3*x1^2] = [0, 27]
        assert jac["x1"][0] == pytest.approx(0.0, abs=1e-4)
        assert jac["x1"][1] == pytest.approx(27.0, abs=1e-4)

    def test_optimize_lbfgs_converges(self, tool):
        """L-BFGS should drive (a-3)^2 + (b+1)^2 to zero at a=3, b=-1."""

        # Frame the minimisation as a single-point fit against target=0.
        def opt_fn(x, a, b):
            return (a - 3.0) ** 2 + (b + 1.0) ** 2

        tool._built_in_functions["test_opt"] = opt_fn

        result = asyncio.run(
            tool.call(
                AutoDiffInput(
                    action="optimize",
                    function_type="test_opt",
                    variables={"x": [1.0], "target": [0.0]},
                    function_params={"a": 0.0, "b": 0.0},
                    method="lbfgs",
                    use_jax=False,
                ),
                CTX,
            )
        )
        assert result.success, result.error
        opt = result.data["optimized_params"]
        # L-BFGS on a single data point won't hit machine precision — give
        # it a bit of room.
        assert opt["a"] == pytest.approx(3.0, abs=5e-2)
        assert opt["b"] == pytest.approx(-1.0, abs=5e-2)
        assert result.data["final_loss"] < 1e-3


# ──────────────────────────────────────────────────────────────────────
# 5. DescriptorTool new actions
# ──────────────────────────────────────────────────────────────────────


class TestDescriptorTool:
    @pytest.fixture
    def tool(self):
        return DescriptorTool()

    @pytest.mark.asyncio
    async def test_composition_backward_compat(self, tool):
        """composition action should still work as before."""
        result = await tool.call(
            tool.input_schema(action="composition", formula="H2O"), CTX
        )
        assert result.success
        features = result.data["features"]
        assert features["num_elements"] == 2

    @pytest.mark.asyncio
    async def test_matminer_graceful_without_dep(self, tool):
        """matminer action should return a clear error when matminer is missing."""
        result = await tool.call(
            tool.input_schema(action="matminer", formula="Fe2O3"), CTX
        )
        try:
            import matminer  # noqa: F401
        except ImportError:
            # Expected path on this machine — should fail gracefully.
            assert not result.success
            assert "matminer" in result.error.lower()
        else:
            # If matminer happens to be installed the call should succeed.
            assert result.success

    @pytest.mark.asyncio
    async def test_coulomb_matrix_graceful_without_dscribe(self, tool):
        """coulomb_matrix should report a clean error when dscribe is absent."""
        result = await tool.call(
            tool.input_schema(action="coulomb_matrix", formula="Si"), CTX
        )
        try:
            import dscribe  # noqa: F401
        except ImportError:
            assert not result.success
            assert "dscribe" in result.error.lower()
        else:
            assert result.success


# ──────────────────────────────────────────────────────────────────────
# 6. UQTool PCE and Morris
# ──────────────────────────────────────────────────────────────────────


class TestUQTool:
    @pytest.fixture
    def tool(self):
        return UQTool()

    def test_pce_linear_function(self, tool):
        """PCE on y=a+b with uniform[0,1] variables should recover mean≈1."""
        result = asyncio.run(tool.call(
            {
                "action": "pce",
                "expression": "a + b",
                "variables": [
                    {"name": "a", "distribution": "uniform", "low": 0.0, "high": 1.0},
                    {"name": "b", "distribution": "uniform", "low": 0.0, "high": 1.0},
                ],
                "order": 3,
                "seed": 42,
            }
        ))
        assert result.success, result.error
        data = result.data
        assert data["method"] == "pce"
        # E[a+b] = 0.5 + 0.5 = 1.0
        assert data["mean"] == pytest.approx(1.0, abs=0.15)
        # Var(a+b) = 1/12 + 1/12 ≈ 0.1667
        assert data["variance"] == pytest.approx(1.0 / 6.0, abs=0.05)

    def test_morris_elementary_effects(self, tool):
        """Morris on y=x0+2*x1 should give mu_star ≈ [1, 2]."""
        result = asyncio.run(tool.call(
            {
                "action": "morris",
                "expression": "x0 + 2*x1",
                "variables": [
                    {"name": "x0", "distribution": "uniform", "low": 0.0, "high": 1.0},
                    {"name": "x1", "distribution": "uniform", "low": 0.0, "high": 1.0},
                ],
                "r": 20,
                "levels": 4,
                "seed": 123,
            }
        ))
        assert result.success, result.error
        ee = result.data["elementary_effects"]
        # Linear function → elementary effects are exactly the coefficients.
        assert ee["x0"]["mu_star"] == pytest.approx(1.0, abs=0.15)
        assert ee["x1"]["mu_star"] == pytest.approx(2.0, abs=0.15)


# ──────────────────────────────────────────────────────────────────────
# 7. GPTool Matern kernels and UCB
# ──────────────────────────────────────────────────────────────────────


class TestGPTool:
    @pytest.fixture
    def tool(self):
        return GPTool()

    def test_fit_matern32(self, tool):
        """GP fit with matern32 kernel should succeed."""
        X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        y = [0.0, 0.5, 0.3, 0.8, 0.4]
        result = tool.call(
            {"action": "fit", "X": X, "y": y, "kernel": "matern32"}
        )
        assert result.success, result.error
        assert result.data["kernel"] == "matern32"
        assert result.data["n_train"] == 5

    def test_fit_matern52(self, tool):
        """GP fit with matern52 kernel should succeed."""
        X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        y = [0.0, 0.5, 0.3, 0.8, 0.4]
        result = tool.call(
            {"action": "fit", "X": X, "y": y, "kernel": "matern52"}
        )
        assert result.success, result.error
        assert result.data["kernel"] == "matern52"

    def test_suggest_with_ucb(self, tool):
        """suggest with acquisition='ucb' should return a valid candidate."""
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 0.5, 0.3, 0.8]
        candidates = [[0.5], [1.5], [2.5], [3.5]]
        result = tool.call(
            {
                "action": "suggest",
                "X": X,
                "y": y,
                "X_new": candidates,
                "acquisition": "ucb",
                "kappa": 2.0,
                "maximize": True,
            }
        )
        assert result.success, result.error
        assert 0 <= result.data["suggested_index"] < len(candidates)
        assert result.data["acquisition_type"] == "ucb"


# ──────────────────────────────────────────────────────────────────────
# 8. NumericalTool new actions
# ──────────────────────────────────────────────────────────────────────


class TestNumericalTool:
    @pytest.fixture
    def tool(self):
        return NumericalTool()

    @pytest.mark.asyncio
    async def test_constrained_minimize_with_bounds(self, tool):
        """SLSQP with box bounds should find the minimum inside the feasible region."""
        result = await tool.call(
            {
                "action": "constrained_minimize",
                "func": "(X[0] - 3)**2 + (X[1] + 1)**2",
                "x0": [0.0, 0.0],
                "bounds": [[0.0, 5.0], [-5.0, 0.0]],
                "method": "SLSQP",
            }
        )
        assert result.success, result.error
        x = result.data["values"]["x"]
        # Unconstrained min (3, -1) is inside the bounds, so we should hit it.
        assert x[0] == pytest.approx(3.0, abs=1e-3)
        assert x[1] == pytest.approx(-1.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_svd(self, tool):
        """SVD on a simple matrix should return U, S, Vh."""
        A = [[1.0, 0.0], [0.0, 2.0]]
        result = await tool.call({"action": "svd", "A": A})
        assert result.success, result.error
        S = result.data["S"]
        # Singular values of diag(1,2) are {2, 1}
        assert len(S) == 2
        assert max(S) == pytest.approx(2.0, abs=1e-10)
        assert min(S) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.asyncio
    async def test_matrix_exp_zero_matrix(self, tool):
        """expm of the zero matrix should be the identity."""
        A = [[0.0, 0.0], [0.0, 0.0]]
        result = await tool.call({"action": "matrix_exp", "A": A})
        assert result.success, result.error
        expm = np.array(result.data["expm"])
        expected = np.eye(2)
        np.testing.assert_allclose(expm, expected, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# 9. SymmetryTool new actions
# ──────────────────────────────────────────────────────────────────────


class TestSymmetryTool:
    @pytest.fixture
    def tool(self):
        return SymmetryTool()

    @pytest.fixture
    def fe_poscar(self, tmp_path):
        """Write a BCC Fe POSCAR for the magnetic / subgroup tests."""
        p = tmp_path / "Fe_POSCAR"
        p.write_text(
            "Fe BCC\n1.0\n"
            "2.87 0.0 0.0\n0.0 2.87 0.0\n0.0 0.0 2.87\n"
            "Fe\n2\nDirect\n"
            "0.0 0.0 0.0\n0.5 0.5 0.5\n",
            encoding="utf-8",
        )
        return str(p)

    @pytest.mark.asyncio
    async def test_magnetic_action_detects_fe(self, tool, fe_poscar):
        """magnetic action should flag Fe as a magnetic site."""
        try:
            import pymatgen  # noqa: F401
        except ImportError:
            pytest.skip("pymatgen not installed")

        result = await tool.call({"action": "magnetic", "file_path": fe_poscar})
        assert result.success, result.error
        assert result.data["has_magnetic_sites"] is True
        assert result.data["n_magnetic_sites"] >= 1
        elements = [s["element"] for s in result.data["magnetic_sites"]]
        assert "Fe" in elements

    @pytest.mark.asyncio
    async def test_subgroups_action(self, tool, fe_poscar):
        """subgroups action should return a list of candidate subgroups."""
        try:
            import pymatgen  # noqa: F401
            import spglib  # noqa: F401
        except ImportError:
            pytest.skip("pymatgen or spglib not installed")

        # index=0 means "list everything we find regardless of index"
        result = await tool.call(
            {"action": "subgroups", "file_path": fe_poscar, "index": 0}
        )
        assert result.success, result.error
        assert "subgroups" in result.data
        assert "current_spacegroup" in result.data
        # BCC Fe is Im-3m (229)
        assert result.data["current_spacegroup"] == 229


# ──────────────────────────────────────────────────────────────────────
# 10. UnitTool new actions
# ──────────────────────────────────────────────────────────────────────


class TestUnitTool:
    @pytest.fixture
    def tool(self):
        return UnitTool()

    @pytest.mark.asyncio
    async def test_infer_dimension_force(self, tool):
        """infer_dimension on m*a should give a force dimension."""
        try:
            import pint  # noqa: F401
        except ImportError:
            pytest.skip("pint not installed")

        result = await tool.call(
            {
                "action": "infer_dimension",
                "expression": "m * a",
                "variables": {"m": "kg", "a": "m/s**2"},
            }
        )
        assert result.success, result.error
        dim = result.data["result_dimension"]
        # Force = mass * acceleration → [mass] * [length] / [time]^2
        assert "length" in dim
        assert "mass" in dim
        assert "time" in dim

    @pytest.mark.asyncio
    async def test_unit_arithmetic_multiply(self, tool):
        """2 N * 3 m should give 6 N*m."""
        try:
            import pint  # noqa: F401
        except ImportError:
            pytest.skip("pint not installed (fallback lacks compound-unit multiply)")

        result = await tool.call(
            {
                "action": "unit_arithmetic",
                "operation": "multiply",
                "value1": 2.0,
                "unit1": "N",
                "value2": 3.0,
                "unit2": "m",
            }
        )
        assert result.success, result.error
        assert result.data["result_value"] == pytest.approx(6.0)
        # Result unit should contain both newton and metre components.
        unit_str = result.data["result_unit"]
        assert "newton" in unit_str or "N" in unit_str

    @pytest.mark.asyncio
    async def test_natural_units_hartree_to_si(self, tool):
        """natural_units converting 1 Hartree to SI should give ~4.36e-18 J."""
        result = await tool.call(
            {
                "action": "natural_units",
                "value": 1.0,
                "from_system": "atomic",
                "to_system": "si",
                "quantity": "energy",
            }
        )
        assert result.success, result.error
        # 1 Hartree = 4.3597e-18 J
        assert result.data["value"] == pytest.approx(4.3597e-18, rel=1e-4)
        assert result.data["unit"] == "J"
