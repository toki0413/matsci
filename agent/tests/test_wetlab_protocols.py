"""E4: 湿实验协议模板扩展 — 5 个结构化协议 (XRD/SEM/TGA_DSC/MECHANICAL/EIS).

覆盖 submit_protocol 和 parse_result 两个本地 action:
- 5 个协议的合法提交 (必填参数 + 样品元数据)
- 默认值填充 (非必填参数自动补全)
- 参数校验: 缺必填、超范围
- 样品元数据校验: 缺必填字段
- 未知协议拒绝
- 结果解析: 抽取 schema 字段、缺失字段填 None
"""

from __future__ import annotations

import pytest

from huginn.tools.wetlab_rpc_tool import (
    PROTOCOLS,
    WetlabInput,
    WetlabRpcTool,
    _parse_protocol_result,
    _validate_protocol_params,
)


@pytest.fixture
def tool():
    return WetlabRpcTool()


# ── submit_protocol: 5 个协议合法提交 ────────────────────────────


class TestSubmitProtocolValid:
    """5 个协议的合法提交路径."""

    @pytest.mark.asyncio
    async def test_xrd_valid_submission(self, tool):
        args = WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            params={
                "scan_range": [10, 80],
                "step_size": 0.02,
                "dwell_time": 1.0,
            },
            sample={
                "sample_id": "S001",
                "composition": "TiO2",
                "preparation_method": "sol-gel",
            },
            lab_id="lab_xrd_1",
        )
        result = await tool.call(args, context=None)
        assert result.success, f"XRD submit failed: {result.error}"
        tpl = result.data["request_template"]
        assert tpl["protocol"] == "XRD"
        assert tpl["lab_id"] == "lab_xrd_1"
        assert tpl["sample"]["sample_id"] == "S001"
        # 非必填参数 wavelength 应该填默认值 1.5406
        assert tpl["params"]["wavelength"] == pytest.approx(1.5406)
        assert tpl["params"]["step_size"] == pytest.approx(0.02)
        assert tpl["lims_format"] is True
        assert result.data["validated"] is True

    @pytest.mark.asyncio
    async def test_sem_valid_with_defaults(self, tool):
        args = WetlabInput(
            action="submit_protocol",
            protocol="SEM",
            params={
                "magnification": 50000,
                "eht": 5.0,
            },
            sample={
                "sample_id": "S002",
                "composition": "LiCoO2",
                "surface_treatment": "polished",
            },
        )
        result = await tool.call(args, context=None)
        assert result.success, f"SEM submit failed: {result.error}"
        params = result.data["request_template"]["params"]
        # working_distance / detector / sample_coating 都有默认值
        assert params["working_distance"] == pytest.approx(8.5)
        assert params["detector"] == "SE2"
        assert params["sample_coating"] == "none"
        assert params["magnification"] == 50000

    @pytest.mark.asyncio
    async def test_tga_dsc_valid_submission(self, tool):
        args = WetlabInput(
            action="submit_protocol",
            protocol="TGA_DSC",
            params={
                "temp_range": [25, 800],
                "ramp_rate": 10.0,
                "atmosphere": "N2",
            },
            sample={
                "sample_id": "S003",
                "composition": "PMMA",
                "pre_treatment": "dried",
            },
        )
        result = await tool.call(args, context=None)
        assert result.success, f"TGA_DSC submit failed: {result.error}"
        params = result.data["request_template"]["params"]
        assert params["atmosphere"] == "N2"
        # sample_mass / flow_rate 有默认值
        assert params["sample_mass"] == pytest.approx(10.0)
        assert params["flow_rate"] == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_mechanical_valid_submission(self, tool):
        args = WetlabInput(
            action="submit_protocol",
            protocol="MECHANICAL",
            params={
                "test_type": "tensile",
                "strain_rate": 1e-3,
            },
            sample={
                "sample_id": "S004",
                "material_grade": "AISI 1045",
                "heat_treatment": "quenched",
            },
        )
        result = await tool.call(args, context=None)
        assert result.success, f"MECHANICAL submit failed: {result.error}"
        params = result.data["request_template"]["params"]
        assert params["test_type"] == "tensile"
        assert params["gauge_length"] == pytest.approx(25.0)
        assert params["specimen_geometry"] == "ASTM E8"
        assert params["temperature"] == pytest.approx(25.0)

    @pytest.mark.asyncio
    async def test_eis_valid_submission(self, tool):
        args = WetlabInput(
            action="submit_protocol",
            protocol="EIS",
            params={
                "frequency_range": [1e5, 1e-2],
                "ac_amplitude": 10.0,
                "electrolyte": "1M LiPF6 in EC/DMC",
            },
            sample={
                "sample_id": "S005",
                "electrode_material": "graphite",
                "electrolyte": "1M LiPF6 in EC/DMC",
            },
        )
        result = await tool.call(args, context=None)
        assert result.success, f"EIS submit failed: {result.error}"
        params = result.data["request_template"]["params"]
        assert params["ac_amplitude"] == pytest.approx(10.0)
        assert params["dc_bias"] == pytest.approx(0.0)
        assert params["points_per_decade"] == 10


# ── submit_protocol: 校验失败路径 ────────────────────────────────


class TestSubmitProtocolValidation:
    """参数 / 样品 / 协议校验."""

    @pytest.mark.asyncio
    async def test_missing_required_param(self, tool):
        # XRD 缺 step_size
        args = WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            params={"scan_range": [10, 80], "dwell_time": 1.0},
            sample={
                "sample_id": "S001",
                "composition": "TiO2",
                "preparation_method": "sol-gel",
            },
        )
        result = await tool.call(args, context=None)
        assert not result.success
        assert "step_size" in result.error
        assert "validation_errors" in result.data

    @pytest.mark.asyncio
    async def test_param_out_of_range(self, tool):
        # step_size 超范围 (上限 1.0)
        args = WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            params={
                "scan_range": [10, 80],
                "step_size": 2.0,
                "dwell_time": 1.0,
            },
            sample={
                "sample_id": "S001",
                "composition": "TiO2",
                "preparation_method": "sol-gel",
            },
        )
        result = await tool.call(args, context=None)
        assert not result.success
        assert "step_size" in result.error
        assert "超出范围" in result.error

    @pytest.mark.asyncio
    async def test_nan_param_rejected(self, tool):
        # NaN 绕过 IEEE 754 比较, 必须用 isfinite 拦截
        args = WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            params={
                "scan_range": [10, 80],
                "step_size": float("nan"),
                "dwell_time": 1.0,
            },
            sample={
                "sample_id": "S001",
                "composition": "TiO2",
                "preparation_method": "sol-gel",
            },
        )
        result = await tool.call(args, context=None)
        assert not result.success, "NaN 参数必须被拒绝"
        assert "step_size" in result.error

    @pytest.mark.asyncio
    async def test_inf_param_rejected(self, tool):
        args = WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            params={
                "scan_range": [10, 80],
                "step_size": float("inf"),
                "dwell_time": 1.0,
            },
            sample={
                "sample_id": "S001",
                "composition": "TiO2",
                "preparation_method": "sol-gel",
            },
        )
        result = await tool.call(args, context=None)
        assert not result.success, "inf 参数必须被拒绝"
        assert "step_size" in result.error

    @pytest.mark.asyncio
    async def test_missing_sample_field(self, tool):
        # 缺 composition
        args = WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            params={
                "scan_range": [10, 80],
                "step_size": 0.02,
                "dwell_time": 1.0,
            },
            sample={
                "sample_id": "S001",
                "preparation_method": "sol-gel",
            },
        )
        result = await tool.call(args, context=None)
        assert not result.success
        assert "composition" in result.error
        assert "missing_sample_fields" in result.data
        assert "composition" in result.data["missing_sample_fields"]

    @pytest.mark.asyncio
    async def test_unknown_protocol(self, tool):
        args = WetlabInput(
            action="submit_protocol",
            protocol="NMR",
            params={},
            sample={"sample_id": "S001"},
        )
        result = await tool.call(args, context=None)
        assert not result.success
        assert "NMR" in result.error or "未知协议" in result.error

    @pytest.mark.asyncio
    async def test_protocol_case_insensitive(self, tool):
        # 小写 "xrd" 应该也能识别
        args = WetlabInput(
            action="submit_protocol",
            protocol="xrd",
            params={
                "scan_range": [10, 80],
                "step_size": 0.02,
                "dwell_time": 1.0,
            },
            sample={
                "sample_id": "S001",
                "composition": "TiO2",
                "preparation_method": "sol-gel",
            },
        )
        result = await tool.call(args, context=None)
        assert result.success, f"case-insensitive failed: {result.error}"
        assert result.data["request_template"]["protocol"] == "XRD"

    @pytest.mark.asyncio
    async def test_missing_protocol_field(self, tool):
        args = WetlabInput(
            action="submit_protocol",
            params={"scan_range": [10, 80]},
            sample={"sample_id": "S001"},
        )
        result = await tool.call(args, context=None)
        assert not result.success
        assert "protocol" in (result.error or "")


# ── parse_result ─────────────────────────────────────────────────


class TestParseResult:
    """结果解析按 schema 抽字段."""

    @pytest.mark.asyncio
    async def test_xrd_result_parsing(self, tool):
        raw = {
            "peaks": [
                {"two_theta": 25.3, "intensity": 100, "fwhm": 0.3},
                {"two_theta": 48.0, "intensity": 60, "fwhm": 0.4},
            ],
            "crystallite_size_nm": 32.5,
            "phase_ids": ["anatase"],
            "extra_field": "ignored",
        }
        args = WetlabInput(
            action="parse_result",
            protocol="XRD",
            raw_result=raw,
        )
        result = await tool.call(args, context=None)
        assert result.success, f"parse failed: {result.error}"
        parsed = result.data["parsed"]
        assert parsed["crystallite_size_nm"] == pytest.approx(32.5)
        assert len(parsed["peaks"]) == 2
        assert parsed["phase_ids"] == ["anatase"]
        assert "extra_field" in parsed["raw"]
        assert result.data["n_fields_extracted"] == 3
        assert result.data["n_fields_total"] == 3

    @pytest.mark.asyncio
    async def test_sem_result_parsing(self, tool):
        raw = {
            "images": [{"filename": "img1.tif", "magnification": 50000, "scale_bar_um": 2}],
            "grain_size_um": 1.2,
            "eds_elements": ["Ti", "O"],
        }
        args = WetlabInput(
            action="parse_result",
            protocol="SEM",
            raw_result=raw,
        )
        result = await tool.call(args, context=None)
        assert result.success
        parsed = result.data["parsed"]
        assert parsed["grain_size_um"] == pytest.approx(1.2)
        assert parsed["eds_elements"] == ["Ti", "O"]

    @pytest.mark.asyncio
    async def test_missing_fields_fill_none(self, tool):
        # raw 缺 peaks 和 phase_ids, 应填 None
        raw = {"crystallite_size_nm": 15.0}
        args = WetlabInput(
            action="parse_result",
            protocol="XRD",
            raw_result=raw,
        )
        result = await tool.call(args, context=None)
        assert result.success
        parsed = result.data["parsed"]
        assert parsed["crystallite_size_nm"] == pytest.approx(15.0)
        assert parsed["peaks"] is None
        assert parsed["phase_ids"] is None
        assert result.data["n_fields_extracted"] == 1
        assert result.data["n_fields_total"] == 3

    @pytest.mark.asyncio
    async def test_parse_unknown_protocol(self, tool):
        args = WetlabInput(
            action="parse_result",
            protocol="NMR",
            raw_result={"some": "data"},
        )
        result = await tool.call(args, context=None)
        assert not result.success
        assert "NMR" in result.error or "未知协议" in result.error

    @pytest.mark.asyncio
    async def test_parse_missing_raw_result(self, tool):
        args = WetlabInput(
            action="parse_result",
            protocol="XRD",
        )
        result = await tool.call(args, context=None)
        assert not result.success
        assert "raw_result" in (result.error or "")


# ── 协议 schema 内联校验 ─────────────────────────────────────────


class TestProtocolSchemas:
    """直接校验 PROTOCOLS schema 和 helper 函数."""

    def test_all_five_protocols_present(self):
        expected = {"XRD", "SEM", "TGA_DSC", "MECHANICAL", "EIS"}
        assert set(PROTOCOLS.keys()) == expected

    def test_each_protocol_has_required_keys(self):
        for name, proto in PROTOCOLS.items():
            assert "name" in proto, f"{name} 缺 name"
            assert "params" in proto, f"{name} 缺 params"
            assert "result_schema" in proto, f"{name} 缺 result_schema"
            assert "sample_fields" in proto, f"{name} 缺 sample_fields"
            assert isinstance(proto["params"], dict)
            assert isinstance(proto["result_schema"], dict)
            assert isinstance(proto["sample_fields"], list)
            assert len(proto["sample_fields"]) >= 2

    def test_validate_protocol_params_helper(self):
        # 合法
        ok, errs = _validate_protocol_params(
            "XRD",
            {"scan_range": [10, 80], "step_size": 0.02, "dwell_time": 1.0},
        )
        assert ok
        assert errs == []
        # 缺必填
        ok2, errs2 = _validate_protocol_params(
            "XRD",
            {"scan_range": [10, 80], "dwell_time": 1.0},
        )
        assert not ok2
        assert any("step_size" in e for e in errs2)
        # 超范围
        ok3, errs3 = _validate_protocol_params(
            "XRD",
            {"scan_range": [10, 80], "step_size": 5.0, "dwell_time": 1.0},
        )
        assert not ok3
        assert any("超出范围" in e for e in errs3)

    def test_parse_protocol_result_helper(self):
        raw = {"youngs_modulus_GPa": 210.0, "yield_strength_MPa": 350.0}
        parsed = _parse_protocol_result("MECHANICAL", raw)
        assert parsed["protocol"] == "MECHANICAL"
        assert parsed["youngs_modulus_GPa"] == pytest.approx(210.0)
        assert parsed["yield_strength_MPa"] == pytest.approx(350.0)
        # 未提供的字段填 None
        assert parsed["ultimate_strength_MPa"] is None
        assert parsed["elongation_at_break_pct"] is None
        assert parsed["stress_strain_curve"] is None
        assert "parsed_fields" in parsed
