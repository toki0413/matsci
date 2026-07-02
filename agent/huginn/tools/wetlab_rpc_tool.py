"""湿实验 RPC 工具 — 调远程湿实验调度服务 + 5 个结构化协议模板.

6 个 action: submit_request / check_status / fetch_result / list_labs /
submit_protocol / parse_result.
端点由 HUGINN_WETLAB_ENDPOINT 环境变量配置. submit_protocol 和 parse_result
是本地校验/解析, 不需要端点, 让 agent 在没有真实 LIMS 时也能用协议模板
规范化湿实验请求与结果.

支持的协议:
- XRD: X 射线衍射 (scan_range/step_size/dwell_time/wavelength)
- SEM: 扫描电镜 (magnification/eht/working_distance/detector)
- TGA_DSC: 热重-差示扫描量热 (temp_range/ramp_rate/atmosphere)
- MECHANICAL: 力学拉伸/压缩/弯曲 (test_type/strain_rate/gauge_length)
- EIS: 电化学阻抗谱 (frequency_range/ac_amplitude/dc_bias)
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


# ── LIMS-style 协议模板 ─────────────────────────────────────────
#
# 每个协议定义:
#   params: 参数名 → {type, unit, range, required, description}
#   result_schema: 结果字段 → {type, unit, description}
#   sample_fields: 样品元数据必填字段


PROTOCOLS: dict[str, dict[str, Any]] = {
    "XRD": {
        "name": "X-ray Diffraction",
        "params": {
            "scan_range": {
                "type": "list[float]",
                "unit": "degree 2θ",
                "range": [5, 150],
                "required": True,
                "description": "[start, end] 扫描角度范围",
            },
            "step_size": {
                "type": "float",
                "unit": "degree",
                "range": [0.005, 1.0],
                "required": True,
                "description": "步进角度",
            },
            "dwell_time": {
                "type": "float",
                "unit": "s",
                "range": [0.1, 100.0],
                "required": True,
                "description": "每步停留时间",
            },
            "wavelength": {
                "type": "float",
                "unit": "Å",
                "range": [0.5, 2.5],
                "required": False,
                "default": 1.5406,
                "description": "X 射线波长 (Cu Kα=1.5406)",
            },
            "sample_mass": {
                "type": "float",
                "unit": "mg",
                "range": [1, 500],
                "required": False,
                "description": "样品质量",
            },
        },
        "result_schema": {
            "peaks": {
                "type": "list[dict]",
                "description": "衍射峰列表 [{two_theta, intensity, fwhm}]",
            },
            "crystallite_size_nm": {
                "type": "float",
                "unit": "nm",
                "description": "Scherrer 公式计算的晶粒尺寸",
            },
            "phase_ids": {
                "type": "list[str]",
                "description": "识别到的物相",
            },
        },
        "sample_fields": ["sample_id", "composition", "preparation_method"],
    },
    "SEM": {
        "name": "Scanning Electron Microscopy",
        "params": {
            "magnification": {
                "type": "int",
                "unit": "x",
                "range": [100, 500000],
                "required": True,
                "description": "放大倍数",
            },
            "eht": {
                "type": "float",
                "unit": "kV",
                "range": [0.1, 30.0],
                "required": True,
                "description": "加速电压",
            },
            "working_distance": {
                "type": "float",
                "unit": "mm",
                "range": [1, 50],
                "required": False,
                "default": 8.5,
                "description": "工作距离",
            },
            "detector": {
                "type": "str",
                "required": False,
                "default": "SE2",
                "description": "探测器类型 (SE2/InLens/ESB/etc.)",
            },
            "sample_coating": {
                "type": "str",
                "required": False,
                "default": "none",
                "description": "导电镀层 (Au/C/Pt/none)",
            },
        },
        "result_schema": {
            "images": {
                "type": "list[dict]",
                "description": "图像列表 [{filename, magnification, scale_bar_um}]",
            },
            "grain_size_um": {
                "type": "float",
                "unit": "μm",
                "description": "图像分析得到的平均晶粒尺寸",
            },
            "eds_elements": {
                "type": "list[str]",
                "description": "EDS 检测到的元素",
            },
        },
        "sample_fields": ["sample_id", "composition", "surface_treatment"],
    },
    "TGA_DSC": {
        "name": "Thermogravimetric Analysis - Differential Scanning Calorimetry",
        "params": {
            "temp_range": {
                "type": "list[float]",
                "unit": "°C",
                "range": [25, 1500],
                "required": True,
                "description": "[start, end] 温度范围",
            },
            "ramp_rate": {
                "type": "float",
                "unit": "°C/min",
                "range": [0.1, 50.0],
                "required": True,
                "description": "升温速率",
            },
            "atmosphere": {
                "type": "str",
                "required": True,
                "description": "气氛 (N2/Air/Ar/He)",
            },
            "sample_mass": {
                "type": "float",
                "unit": "mg",
                "range": [1, 100],
                "required": False,
                "default": 10.0,
                "description": "样品质量",
            },
            "flow_rate": {
                "type": "float",
                "unit": "mL/min",
                "range": [10, 200],
                "required": False,
                "default": 50.0,
                "description": "气体流速",
            },
        },
        "result_schema": {
            "mass_loss_steps": {
                "type": "list[dict]",
                "description": "失重台阶 [{temp_onset, temp_end, mass_loss_pct}]",
            },
            "dsc_peaks": {
                "type": "list[dict]",
                "description": "DSC 峰 [{temp, enthalpy_Jg, type}]",
            },
            "residual_mass_pct": {
                "type": "float",
                "unit": "%",
                "description": "残余质量百分比",
            },
        },
        "sample_fields": ["sample_id", "composition", "pre_treatment"],
    },
    "MECHANICAL": {
        "name": "Mechanical Testing",
        "params": {
            "test_type": {
                "type": "str",
                "required": True,
                "description": "测试类型 (tensile/compression/flexural)",
            },
            "strain_rate": {
                "type": "float",
                "unit": "1/s",
                "range": [1e-6, 1e-1],
                "required": True,
                "description": "应变速率",
            },
            "gauge_length": {
                "type": "float",
                "unit": "mm",
                "range": [5, 100],
                "required": False,
                "default": 25.0,
                "description": "标距长度",
            },
            "specimen_geometry": {
                "type": "str",
                "required": False,
                "default": "ASTM E8",
                "description": "试样几何标准",
            },
            "temperature": {
                "type": "float",
                "unit": "°C",
                "range": [-200, 1000],
                "required": False,
                "default": 25.0,
                "description": "测试温度",
            },
        },
        "result_schema": {
            "youngs_modulus_GPa": {
                "type": "float",
                "unit": "GPa",
                "description": "杨氏模量",
            },
            "yield_strength_MPa": {
                "type": "float",
                "unit": "MPa",
                "description": "屈服强度",
            },
            "ultimate_strength_MPa": {
                "type": "float",
                "unit": "MPa",
                "description": "极限强度",
            },
            "elongation_at_break_pct": {
                "type": "float",
                "unit": "%",
                "description": "断裂延伸率",
            },
            "stress_strain_curve": {
                "type": "list[dict]",
                "description": "应力-应变曲线 [{strain, stress_MPa}]",
            },
        },
        "sample_fields": ["sample_id", "material_grade", "heat_treatment"],
    },
    "EIS": {
        "name": "Electrochemical Impedance Spectroscopy",
        "params": {
            "frequency_range": {
                "type": "list[float]",
                "unit": "Hz",
                "range": [1e-6, 1e7],
                "required": True,
                "description": "[start, end] 频率范围",
            },
            "points_per_decade": {
                "type": "int",
                "unit": "n/dec",
                "range": [5, 20],
                "required": False,
                "default": 10,
                "description": "每十倍频采点数",
            },
            "ac_amplitude": {
                "type": "float",
                "unit": "mV",
                "range": [1, 100],
                "required": True,
                "description": "交流扰动幅值",
            },
            "dc_bias": {
                "type": "float",
                "unit": "V",
                "range": [-3, 3],
                "required": False,
                "default": 0.0,
                "description": "直流偏置",
            },
            "electrolyte": {
                "type": "str",
                "required": True,
                "description": "电解液",
            },
        },
        "result_schema": {
            "nyquist_points": {
                "type": "list[dict]",
                "description": "Nyquist 图点 [{Z_real, Z_imag}]",
            },
            "bode_points": {
                "type": "list[dict]",
                "description": "Bode 图点 [{frequency, abs_Z, phase_deg}]",
            },
            "fitted_circuit": {
                "type": "str",
                "description": "拟合等效电路 (如 Randles)",
            },
            "fitted_params": {
                "type": "dict",
                "description": "拟合参数 {R_ohm, R_ct, C_dl}",
            },
        },
        "sample_fields": ["sample_id", "electrode_material", "electrolyte"],
    },
}


def _validate_protocol_params(protocol: str, params: dict[str, Any]) -> tuple[bool, list[str]]:
    """校验参数是否符合协议 schema. 返回 (ok, errors)."""
    if protocol not in PROTOCOLS:
        return False, [f"未知协议: {protocol}, 支持: {list(PROTOCOLS.keys())}"]
    schema = PROTOCOLS[protocol]["params"]
    errors: list[str] = []
    for name, spec in schema.items():
        if spec.get("required", False) and name not in params:
            errors.append(f"缺少必填参数: {name}")
    # 多检: 传了的参数类型粗检
    for name, value in params.items():
        if name not in schema:
            continue  # 未知参数, 不报错 (允许扩展)
        spec = schema[name]
        rng = spec.get("range")
        if rng and isinstance(value, (int, float)):
            if not math.isfinite(value) or value < rng[0] or value > rng[1]:
                errors.append(
                    f"参数 {name}={value} 超出范围 [{rng[0]}, {rng[1]}]"
                )
    return len(errors) == 0, errors


def _parse_protocol_result(protocol: str, raw: dict[str, Any]) -> dict[str, Any]:
    """按协议 schema 解析原始结果, 抽出结构化字段."""
    if protocol not in PROTOCOLS:
        return {"error": f"未知协议: {protocol}", "raw": raw}
    result_schema = PROTOCOLS[protocol]["result_schema"]
    parsed: dict[str, Any] = {"protocol": protocol}
    for field_name in result_schema:
        if field_name in raw:
            parsed[field_name] = raw[field_name]
        else:
            parsed[field_name] = None
    parsed["raw"] = raw
    parsed["parsed_fields"] = list(result_schema.keys())
    return parsed


class WetlabInput(BaseModel):
    action: Literal[
        "submit_request",
        "check_status",
        "fetch_result",
        "list_labs",
        "submit_protocol",
        "parse_result",
    ] = Field(..., description="湿实验调度动作")
    lab_id: str | None = Field(default=None, description="目标实验室 ID")
    request_type: str | None = Field(
        default=None, description="请求类型, 如 'synthesis' / 'measurement'"
    )
    payload: dict[str, Any] | None = Field(
        default=None, description="请求参数 (样品配方、条件等)"
    )
    request_id: str | None = Field(default=None, description="已提交请求的 ID")
    poll_timeout: int = Field(
        default=30, ge=1, le=600, description="check_status 轮询超时 (秒)"
    )
    # submit_protocol / parse_result 专用
    protocol: str | None = Field(
        default=None,
        description="协议类型: XRD / SEM / TGA_DSC / MECHANICAL / EIS",
    )
    params: dict[str, Any] | None = Field(
        default=None, description="submit_protocol: 协议参数"
    )
    sample: dict[str, Any] | None = Field(
        default=None, description="submit_protocol: 样品元数据"
    )
    raw_result: dict[str, Any] | None = Field(
        default=None, description="parse_result: 原始结果 dict"
    )

    model_config = {"protected_namespaces": ()}


class WetlabRpcTool(HuginnTool):
    """远程湿实验调度 RPC + 5 个结构化协议模板."""

    name = "wetlab_rpc_tool"
    category = "wetlab"
    profile = ToolProfile(phases=frozenset({ResearchPhase.EXECUTION}))
    description = (
        "Submit and track wet-lab experiment requests via a remote scheduling service. "
        "Configure the service URL with the HUGINN_WETLAB_ENDPOINT environment variable. "
        "Includes 5 structured protocols (XRD/SEM/TGA_DSC/MECHANICAL/EIS) for local "
        "validation and result parsing without a remote endpoint."
    )
    input_schema = WetlabInput

    def is_read_only(self, args: WetlabInput) -> bool:
        return args.action in ("check_status", "fetch_result", "list_labs", "parse_result")

    @staticmethod
    def _endpoint() -> str | None:
        return os.environ.get("HUGINN_WETLAB_ENDPOINT")

    async def call(self, args: WetlabInput, context: ToolContext) -> ToolResult:
        try:
            # submit_protocol 和 parse_result 是本地操作, 不需要 endpoint
            if args.action == "submit_protocol":
                return self._do_submit_protocol(args)
            if args.action == "parse_result":
                return self._do_parse_result(args)
            # 其他 4 个 action 需要 endpoint
            endpoint = self._endpoint()
            if not endpoint:
                return ToolResult(
                    data={"error": "wetlab endpoint not configured"},
                    success=False,
                    error="HUGINN_WETLAB_ENDPOINT environment variable not set.",
                )
            if args.action == "submit_request":
                return await self._do_submit_request(endpoint, args)
            if args.action == "check_status":
                return await self._do_check_status(endpoint, args)
            if args.action == "fetch_result":
                return await self._do_fetch_result(endpoint, args)
            if args.action == "list_labs":
                return await self._do_list_labs(endpoint, args)
            return ToolResult(
                data=None, success=False, error=f"unknown action: {args.action}"
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

    # ── 协议操作 (本地, 不需要 endpoint) ──────────────────────────

    def _do_submit_protocol(self, args: WetlabInput) -> ToolResult:
        """校验协议参数 + 样品元数据, 返回 LIMS-style 请求模板."""
        if not args.protocol:
            return ToolResult(
                data=None, success=False, error="submit_protocol 需要 protocol",
            )
        protocol = args.protocol.upper()
        if protocol not in PROTOCOLS:
            return ToolResult(
                data=None,
                success=False,
                error=f"未知协议: {protocol}, 支持: {list(PROTOCOLS.keys())}",
            )
        params = args.params or {}
        sample = args.sample or {}

        ok, errors = _validate_protocol_params(protocol, params)
        if not ok:
            return ToolResult(
                data={"protocol": protocol, "validation_errors": errors},
                success=False,
                error="协议参数校验失败: " + "; ".join(errors),
            )

        # 检查样品元数据必填字段
        required_sample = PROTOCOLS[protocol]["sample_fields"]
        missing_sample = [f for f in required_sample if f not in sample]
        if missing_sample:
            return ToolResult(
                data={
                    "protocol": protocol,
                    "missing_sample_fields": missing_sample,
                    "required_sample_fields": required_sample,
                },
                success=False,
                error=f"样品元数据缺少必填字段: {missing_sample}",
            )

        # 填默认值
        schema_params = PROTOCOLS[protocol]["params"]
        full_params = {}
        for name, spec in schema_params.items():
            if name in params:
                full_params[name] = params[name]
            elif "default" in spec:
                full_params[name] = spec["default"]

        # 构造 LIMS-style 请求模板
        request_template = {
            "protocol": protocol,
            "protocol_name": PROTOCOLS[protocol]["name"],
            "lab_id": args.lab_id,
            "sample": sample,
            "params": full_params,
            "result_schema": PROTOCOLS[protocol]["result_schema"],
            "lims_format": True,
        }
        return ToolResult(
            data={
                "request_template": request_template,
                "protocol": protocol,
                "validated": True,
                "n_params": len(full_params),
                "n_required_sample_fields": len(required_sample),
            },
            success=True,
        )

    def _do_parse_result(self, args: WetlabInput) -> ToolResult:
        """按协议 schema 解析原始结果, 抽出结构化字段."""
        if not args.protocol:
            return ToolResult(
                data=None, success=False, error="parse_result 需要 protocol",
            )
        if not args.raw_result:
            return ToolResult(
                data=None, success=False, error="parse_result 需要 raw_result",
            )
        protocol = args.protocol.upper()
        parsed = _parse_protocol_result(protocol, args.raw_result)
        if "error" in parsed:
            return ToolResult(data=None, success=False, error=parsed["error"])
        return ToolResult(
            data={
                "parsed": parsed,
                "protocol": protocol,
                "n_fields_extracted": sum(
                    1 for k in parsed["parsed_fields"] if parsed.get(k) is not None
                ),
                "n_fields_total": len(parsed["parsed_fields"]),
            },
            success=True,
        )

    async def _do_submit_request(self, endpoint: str, args: WetlabInput) -> ToolResult:
        import aiohttp

        url = f"{endpoint.rstrip('/')}/requests"
        body = {
            "lab_id": args.lab_id,
            "request_type": args.request_type,
            "payload": args.payload or {},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    return ToolResult(
                        data=data, success=False,
                        error=f"submit_request HTTP {resp.status}",
                    )
                return ToolResult(
                    data={"request_id": data.get("request_id"), "raw": data},
                    success=True,
                )

    async def _do_check_status(self, endpoint: str, args: WetlabInput) -> ToolResult:
        import aiohttp

        if not args.request_id:
            return ToolResult(
                data=None, success=False, error="check_status requires request_id.",
            )
        url = f"{endpoint.rstrip('/')}/requests/{args.request_id}"
        deadline = asyncio.get_event_loop().time() + args.poll_timeout
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(url) as resp:
                    data = await resp.json()
                    if resp.status >= 400:
                        return ToolResult(
                            data=data, success=False,
                            error=f"check_status HTTP {resp.status}",
                        )
                    status = data.get("status", "unknown")
                    if status in ("done", "failed", "cancelled"):
                        return ToolResult(
                            data={"status": status, "raw": data}, success=True,
                        )
                if asyncio.get_event_loop().time() >= deadline:
                    return ToolResult(
                        data={"status": "timeout", "raw": data},
                        success=True,
                    )
                await asyncio.sleep(2)

    async def _do_fetch_result(self, endpoint: str, args: WetlabInput) -> ToolResult:
        import aiohttp

        if not args.request_id:
            return ToolResult(
                data=None, success=False, error="fetch_result requires request_id.",
            )
        url = f"{endpoint.rstrip('/')}/requests/{args.request_id}/result"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    return ToolResult(
                        data=data, success=False,
                        error=f"fetch_result HTTP {resp.status}",
                    )
                return ToolResult(data={"result": data}, success=True)

    async def _do_list_labs(self, endpoint: str, args: WetlabInput) -> ToolResult:
        import aiohttp

        url = f"{endpoint.rstrip('/')}/labs"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    return ToolResult(
                        data=data, success=False,
                        error=f"list_labs HTTP {resp.status}",
                    )
                return ToolResult(data={"labs": data}, success=True)
