"""Tests for the ML potential tool skeleton."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from huginn.core_types import ToolContext
from huginn.tools.sci.ml_potential_tool import MLPotentialTool


class TestMLPotentialTool:
    def test_missing_structure_file(self, tmp_path: Path):
        tool = MLPotentialTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    backend="mace",
                    structure_file=str(tmp_path / "missing.cif"),
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )
        assert result.success is False
        assert "not found" in (result.error or "").lower()

    @pytest.mark.parametrize("backend", ["mace", "chgnet", "nep"])
    def test_missing_dependency_returns_helpful_error(
        self, tmp_path: Path, backend: str
    ):
        # 本测试验证 "backend 未安装时返回 helpful error".
        # 如果 backend 已安装 (如 [all] 依赖装了 chgnet), 测试不适用, 跳过.
        # 之前不跳过 → chgnet 已安装 → 尝试解析 CIF → CIF 格式不对 →
        # 返回 "Unexpected CIF file entry" 而非 "chgnet not installed" → 断言失败.
        backend_modules = {
            "mace": "mace",
            "chgnet": "chgnet",
            "nep": "pynep",
        }
        mod_name = backend_modules.get(backend, backend)
        try:
            __import__(mod_name)
            pytest.skip(f"{backend} 已安装, 'missing dependency' 测试不适用")
        except ImportError:
            pass  # backend 未安装, 继续测试

        tool = MLPotentialTool()
        structure = tmp_path / "Si.cif"
        structure.write_text(
            "data_Si\n_cell_length_a 5.43\n_cell_length_b 5.43\n"
            "_cell_length_c 5.43\n_cell_angle_alpha 90\n"
            "_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            "Si 0 0 0\n",
            encoding="utf-8",
        )

        result = asyncio.run(
            tool.call(
                tool.input_schema(backend=backend, structure_file=str(structure)),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is False
        assert backend in (result.error or "").lower()
