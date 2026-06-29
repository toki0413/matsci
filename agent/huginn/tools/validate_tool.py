"""Physical validation tool — run all physics validators on a calculation result.

Read-only. Safe to auto-execute.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.constraints import ConstraintAdapter
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult
from huginn.validation.physics import PhysicsValidator


# 按性质定的默认容差, 没传 tolerance 时用这个
TOLERANCES: dict[str, float] = {
    "band_gap": 0.10,          # DFT 带隙问题大, 放宽到 10%
    "lattice_constant": 0.02,  # 晶格常数 DFT 一般准, 2%
    "formation_energy": 0.05,  # 形成能 5%
    "elastic_modulus": 0.15,   # 弹性模量对应变敏感, 15%
    "magnetic_moment": 0.20,   # 磁矩依赖磁序, 20%
    "bulk_modulus": 0.10,      # 体弹模量 10%
}

# 内置实验/DFT 基准值表, key = (structure, property)
# 不在这里的再去查 materials_database_tool 兜底
BENCHMARK_DATA: dict[tuple[str, str], dict[str, Any]] = {
    ("Si", "band_gap"): {"value": 1.12, "uncertainty": 0.01, "source": "experimental",
                          "metadata": {"mp_id": "mp-149", "note": "Si 室温实验带隙 eV"}},
    ("Si", "lattice_constant"): {"value": 5.43, "uncertainty": 0.005, "source": "experimental",
                                  "metadata": {"mp_id": "mp-149", "note": "Si 金刚石晶格常数 Å"}},
    ("Cu", "lattice_constant"): {"value": 3.61, "uncertainty": 0.005, "source": "experimental",
                                  "metadata": {"mp_id": "mp-30", "note": "FCC Cu 晶格常数 Å"}},
    ("Cu", "band_gap"): {"value": 0.0, "uncertainty": 0.0, "source": "experimental",
                          "metadata": {"mp_id": "mp-30", "note": "Cu 金属无带隙"}},
    ("GaN", "band_gap"): {"value": 3.4, "uncertainty": 0.05, "source": "experimental",
                           "metadata": {"mp_id": "mp-804", "note": "GaN 室温带隙 eV"}},
    ("Fe", "lattice_constant"): {"value": 2.87, "uncertainty": 0.005, "source": "experimental",
                                  "metadata": {"mp_id": "mp-13", "note": "BCC Fe 晶格常数 Å"}},
    ("C", "band_gap"): {"value": 5.5, "uncertainty": 0.1, "source": "experimental",
                         "metadata": {"mp_id": "mp-66", "note": "金刚石带隙 eV"}},
    ("SiC", "band_gap"): {"value": 3.26, "uncertainty": 0.05, "source": "experimental",
                           "metadata": {"note": "4H-SiC 带隙 eV"}},
    ("Al", "lattice_constant"): {"value": 4.05, "uncertainty": 0.005, "source": "experimental",
                                  "metadata": {"mp_id": "mp-134", "note": "FCC Al 晶格常数 Å"}},
    ("Au", "lattice_constant"): {"value": 4.08, "uncertainty": 0.005, "source": "experimental",
                                  "metadata": {"mp_id": "mp-81", "note": "FCC Au 晶格常数 Å"}},
    ("MgO", "lattice_constant"): {"value": 4.21, "uncertainty": 0.01, "source": "experimental",
                                   "metadata": {"mp_id": "mp-1265", "note": "MgO 岩盐晶格常数 Å"}},
    ("TiO2", "band_gap"): {"value": 3.0, "uncertainty": 0.05, "source": "experimental",
                            "metadata": {"mp_id": "mp-390", "note": "金红石 TiO2 带隙 eV"}},
    ("ZnO", "band_gap"): {"value": 3.37, "uncertainty": 0.05, "source": "experimental",
                           "metadata": {"mp_id": "mp-2133", "note": "ZnO 室温带隙 eV"}},
    ("GaAs", "band_gap"): {"value": 1.42, "uncertainty": 0.01, "source": "experimental",
                            "metadata": {"mp_id": "mp-2534", "note": "GaAs 室温带隙 eV"}},
    ("NaCl", "lattice_constant"): {"value": 5.64, "uncertainty": 0.01, "source": "experimental",
                                    "metadata": {"mp_id": "mp-22862", "note": "NaCl 岩盐晶格常数 Å"}},
    ("Fe2O3", "band_gap"): {"value": 2.1, "uncertainty": 0.1, "source": "experimental",
                             "metadata": {"mp_id": "mp-19770", "note": "赤铁矿带隙 eV"}},
    ("Ni", "lattice_constant"): {"value": 3.52, "uncertainty": 0.005, "source": "experimental",
                                  "metadata": {"mp_id": "mp-23", "note": "FCC Ni 晶格常数 Å"}},
    ("W", "lattice_constant"): {"value": 3.16, "uncertainty": 0.005, "source": "experimental",
                                 "metadata": {"mp_id": "mp-91", "note": "BCC W 晶格常数 Å"}},
    ("Pt", "lattice_constant"): {"value": 3.92, "uncertainty": 0.005, "source": "experimental",
                                  "metadata": {"mp_id": "mp-126", "note": "FCC Pt 晶格常数 Å"}},
    ("Al", "band_gap"): {"value": 0.0, "uncertainty": 0.0, "source": "experimental",
                          "metadata": {"mp_id": "mp-134", "note": "Al 金属无带隙"}},
}

# benchmark 里 property 名 → MP summary 返回字段名
_MP_FIELD_MAP: dict[str, str] = {
    "band_gap": "band_gap",
    "lattice_constant": "lattice_constant",  # MP 没原生字段, 兜底返回里也没, 走 volume 反推太麻烦, 留着名字
    "formation_energy": "energy_per_atom",
    "elastic_modulus": "bulk_modulus",
    "bulk_modulus": "bulk_modulus",
}


class ValidateToolInput(BaseModel):
    action: Literal["validate", "benchmark"] = Field(default="validate")
    # validate action 用的字段 (action=validate 时必填)
    result_type: Literal["dft", "md", "phonon", "elastic"] | None = Field(default=None)
    result_data: dict | None = Field(default=None, description="Calculation result dict to validate")
    # benchmark action 用的字段
    property: str | None = Field(
        default=None,
        description="要对比的性质: band_gap/lattice_constant/formation_energy/elastic_modulus/magnetic_moment/bulk_modulus",
    )
    computed_value: float | dict | None = Field(
        default=None,
        description="计算值, 单值或 {value, uncertainty}",
    )
    structure: str | None = Field(
        default=None,
        description="结构标识, 化学式 (Si/GaN/Fe2O3) 或 mp_id (mp-149)",
    )
    reference_source: Literal["auto", "materials_project", "experimental", "literature"] = Field(
        default="auto",
        description="基准来源, auto 先内置表再 MP",
    )
    tolerance: float | None = Field(
        default=None,
        description="容差, 不传按性质自动定",
    )
    units: str | None = Field(default=None, description="单位标注")


class ValidateTool(HuginnTool):
    """Validate physical reasonableness of calculation results."""

    name = "validate_tool"
    category = "core"
    description = (
        "Run physical validation checks on calculation results: energy signs, "
        "convergence, band gaps, force thresholds, etc."
    )
    input_schema = ValidateToolInput

    def __init__(self) -> None:
        super().__init__()
        self._adapter = ConstraintAdapter.default()
        # 配一个 PhysicsValidator 兜底，约束系统没覆盖到的物理检查由它补上
        self._physics = PhysicsValidator()

    def is_read_only(self, args: ValidateToolInput) -> bool:
        return True

    async def call(self, args, context: ToolContext | None = None) -> ToolResult:
        # tools.py HTTP 端点传 dict (model_dump), 内部调用传 ValidateToolInput 对象
        if isinstance(args, dict):
            args = ValidateToolInput(**args)
        # benchmark 走独立路径, 不走物理校验那套
        if args.action == "benchmark":
            return await self._run_benchmark(args)
        # 下面是原有 validate 路径, result_type 必须有
        if args.result_type is None:
            return ToolResult(
                data=None, success=False,
                error="result_type is required for validate action",
            )
        if args.result_data is None:
            return ToolResult(
                data=None, success=False,
                error="result_data is required for validate action",
            )
        # 弹性张量校验走独立路径 (Born 稳定性 + Hill 模量)
        if args.result_type == "elastic":
            return self._run_elastic_validation(args.result_data)

        adapter_results = self._adapter.evaluate_all(args.result_type, args.result_data)

        checks: list[dict[str, Any]] = [
            {
                "name": r.name,
                "passed": r.passed,
                "value": r.value,
                "expected": r.expected,
                "message": r.message,
                "severity": r.severity,
                "family": r.family,
                "score": r.score,
            }
            for r in adapter_results
        ]

        # 约束系统已经跑过的检查名，后面用来去重
        covered = {r.name for r in adapter_results}
        for check in self._run_physics_validator(args.result_type, args.result_data):
            if check["name"] in covered:
                # 同名检查以约束系统的结果为准（带 severity），这里跳过
                continue
            checks.append(
                {
                    "name": check["name"],
                    "passed": check["passed"],
                    "value": check["value"],
                    "expected": check["expected"],
                    "message": check["message"],
                    # PhysicsValidator 没有分级，没过的就当 warn，过了标记成 info
                    "severity": "warn" if not check["passed"] else "info",
                    "family": "physics",
                    "score": check.get("score"),
                }
            )
            covered.add(check["name"])

        all_passed = all(c["passed"] for c in checks)
        r_phys = self._aggregate_physics_score(checks)

        return ToolResult(
            data={
                "all_passed": all_passed,
                "r_phys": r_phys,
                "checks": checks,
                "summary": f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed",
            },
            success=True,
        )

    def _run_physics_validator(
        self, result_type: str, result_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """按结果类型分发到 PhysicsValidator 对应方法，吐出扁平的 dict 列表。"""
        if result_type == "dft":
            raw = self._physics.validate_dft_result(result_data)
        elif result_type == "md":
            raw = self._physics.validate_md_result(result_data)
        elif result_type == "phonon":
            raw = self._physics.validate_phonon_result(result_data)
        else:
            return []

        return [
            {
                "name": c.name,
                "passed": c.passed,
                "value": c.value,
                "expected": c.expected,
                "tolerance": c.tolerance,
                "message": c.message,
                "score": c.score,
            }
            for c in raw
        ]

    def _aggregate_physics_score(self, checks: list[dict[str, Any]]) -> float:
        """把校验结果聚合成物理轨数值奖励 R_phys ∈ [0, 1]。

        每个检查的分数: 优先用显式 score, 没有则 passed→1.0/failed→0.0。
        按 severity 加权 (error=3, warn=2, info=1)——物理硬错误必须比
        质量警告更重地拉低 R_phys, 避免 "warn 堆得再多也压不过一个 error"。
        """
        if not checks:
            return 1.0
        weight_map = {"error": 3.0, "warn": 2.0, "info": 1.0}
        total_weight = 0.0
        weighted_score = 0.0
        for c in checks:
            sev = c.get("severity", "warn")
            w = weight_map.get(sev, 1.0)
            s = c.get("score")
            if s is None:
                s = 1.0 if c.get("passed") else 0.0
            total_weight += w
            weighted_score += s * w
        return weighted_score / total_weight if total_weight > 0 else 1.0

    # ── benchmark action: 文献/数据库基准对比 ──────────────────────────

    async def _run_benchmark(self, args: ValidateToolInput) -> ToolResult:
        """把计算值跟内置基准表 / Materials Project 数据对比, 给出 verdict."""
        if args.property is None:
            return ToolResult(data=None, success=False,
                              error="property is required for benchmark action")
        if args.computed_value is None:
            return ToolResult(data=None, success=False,
                              error="computed_value is required for benchmark action")

        # 解析计算值, 支持带不确定度的 dict
        cv = args.computed_value
        if isinstance(cv, dict):
            computed_val = float(cv.get("value", 0.0))
            unc = cv.get("uncertainty")
            computed_unc = float(unc) if unc is not None else None
        else:
            computed_val = float(cv)
            computed_unc = None

        prop = args.property
        # 没传 tolerance 就按性质自动定, 没匹配上的性质给 10% 兜底
        tol = args.tolerance if args.tolerance is not None else TOLERANCES.get(prop, 0.10)

        ref = await self._lookup_reference(prop, args.structure, args.reference_source)

        if ref is None:
            return ToolResult(
                data={
                    "action": "benchmark",
                    "property": prop,
                    "computed": {"value": computed_val, "uncertainty": computed_unc},
                    "reference": None,
                    "difference": None,
                    "relative_error": None,
                    "within_tolerance": None,
                    "verdict": "no_reference",
                    "reference_metadata": None,
                    "tolerance": tol,
                    "units": args.units,
                    "message": f"没找到 {args.structure} 的 {prop} 基准值",
                },
                success=True,
            )

        ref_val = float(ref["value"])
        ref_unc = ref.get("uncertainty")
        diff = computed_val - ref_val
        # 参考值为 0 时相对误差没意义, 返回 None
        rel_err = abs(diff) / abs(ref_val) * 100.0 if ref_val != 0 else None

        within = (rel_err is not None) and (rel_err / 100.0 < tol)

        # verdict 分档: <tol/3 excellent, <tol/2 good, <tol acceptable, 否则 discrepancy
        if rel_err is None:
            # 参考值是 0, 没法算相对误差, 看绝对差是否在容差量级内
            verdict = "acceptable" if abs(diff) < 1e-6 else "discrepancy"
        else:
            rel_frac = rel_err / 100.0
            if rel_frac < tol / 3.0:
                verdict = "excellent"
            elif rel_frac < tol / 2.0:
                verdict = "good"
            elif rel_frac < tol:
                verdict = "acceptable"
            else:
                verdict = "discrepancy"

        return ToolResult(
            data={
                "action": "benchmark",
                "property": prop,
                "computed": {"value": computed_val, "uncertainty": computed_unc},
                "reference": {
                    "value": ref_val,
                    "uncertainty": ref_unc,
                    "source": ref.get("source", "unknown"),
                },
                "difference": diff,
                "relative_error": rel_err,
                "within_tolerance": within,
                "verdict": verdict,
                "reference_metadata": ref.get("metadata", {}),
                "tolerance": tol,
                "units": args.units,
            },
            success=True,
        )

    async def _lookup_reference(
        self,
        prop: str,
        structure: str | None,
        source: str,
    ) -> dict[str, Any] | None:
        """先查内置基准表, 没命中再按 reference_source 决定要不要查 MP."""
        # 内置表里直接命中就走它, 不论 source (内置表全是实验值, experimental/auto/literature 都认)
        if structure:
            key = (structure, prop)
            if key in BENCHMARK_DATA:
                return BENCHMARK_DATA[key]

        # auto / materials_project 才走 MP 兜底, experimental/literature 不查在线库
        if source in ("auto", "materials_project") and structure:
            mp_ref = await self._query_materials_project(prop, structure)
            if mp_ref is not None:
                return mp_ref
        return None

    async def _query_materials_project(
        self, prop: str, structure: str
    ) -> dict[str, Any] | None:
        """调 materials_database_tool 拿一个基准值, 拿不到返回 None.

        走本地结构库 + tool_cache, 没有网络/API key 也大概率能命中本地.
        任何异常都吞掉, 兜底失败就当没参考值.
        """
        try:
            from huginn.tools.materials_database_tool import (
                MaterialsDatabaseInput,
                MaterialsDatabaseTool,
            )

            tool = MaterialsDatabaseTool()
            mp_field = _MP_FIELD_MAP.get(prop, prop)
            mp_args = MaterialsDatabaseInput(
                action="mp_summary",
                query=structure,
                fields=[mp_field],
                limit=1,
            )
            result = await tool.call(mp_args, None)
            if not result.success or not result.data:
                return None
            records = result.data.get("records") or []
            if not records:
                return None
            rec = records[0]
            val = rec.get(mp_field)
            if val is None:
                return None
            return {
                "value": float(val),
                "uncertainty": None,
                "source": f"materials_project ({rec.get('id', '?')})",
                "metadata": {
                    "mp_id": rec.get("id"),
                    "formula": rec.get("formula"),
                    "field": mp_field,
                },
            }
        except Exception:
            # 兜底失败不报错, 让上层走 no_reference
            return None

    def _run_elastic_validation(self, result_data: dict[str, Any]) -> ToolResult:
        """弹性张量校验: Born 稳定性判据 + Voigt/Reuss/Hill 模量."""
        import numpy as np
        from huginn.mechanics import BornStabilityChecker, ElasticTensor

        raw = result_data.get("elastic_tensor") or result_data.get("C")
        if raw is None:
            return ToolResult(
                data=None,
                success=False,
                error="elastic_tensor (6x6 Voigt matrix in GPa) is required",
            )

        try:
            C = np.array(raw, dtype=float)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Cannot parse tensor: {e}"
            )

        if C.shape != (6, 6):
            return ToolResult(
                data=None,
                success=False,
                error=f"Tensor must be 6x6, got {C.shape}",
            )

        crystal_system = result_data.get("crystal_system", "auto")
        checks: list[dict[str, Any]] = []

        # Born 稳定性
        try:
            born = BornStabilityChecker.check(C, crystal_system=crystal_system)
            for c in born.get("criteria", []):
                # numpy.bool_ / numpy.float64 转 Python 原生类型, 否则 FastAPI 序列化炸
                passed = bool(c["passed"])
                checks.append(
                    {
                        "name": c["name"],
                        "passed": passed,
                        "value": float(c["value"]) if c["value"] is not None else None,
                        "expected": f"> {c['threshold']}",
                        "message": f"Born criterion ({born.get('crystal_system', '?')})",
                        "severity": "error" if not passed else "info",
                        "family": "mechanics",
                    }
                )
        except Exception as e:
            checks.append(
                {
                    "name": "born_stability",
                    "passed": False,
                    "value": None,
                    "expected": "valid criteria",
                    "message": f"Born check failed: {e}",
                    "severity": "warn",
                    "family": "mechanics",
                }
            )

        # Hill 平均模量 (Voigt-Reuss 平均)
        try:
            et = ElasticTensor(C)
            moduli = et.hill_moduli()
            for key, label in [
                ("bulk_modulus_hill", "Bulk modulus (Hill)"),
                ("shear_modulus_hill", "Shear modulus (Hill)"),
                ("youngs_modulus", "Young's modulus"),
            ]:
                val = moduli.get(key)
                if val is not None:
                    val = float(val)
                    checks.append(
                        {
                            "name": f"{key} > 0",
                            "passed": val > 0,
                            "value": val,
                            "expected": "> 0 GPa",
                            "message": label,
                            "severity": "warn" if val <= 0 else "info",
                            "family": "mechanics",
                        }
                    )
        except Exception as e:
            checks.append(
                {
                    "name": "moduli_calculation",
                    "passed": False,
                    "value": None,
                    "expected": "invertible tensor",
                    "message": f"Moduli calculation failed: {e}",
                    "severity": "warn",
                    "family": "mechanics",
                }
            )

        all_passed = all(c["passed"] for c in checks)
        r_phys = self._aggregate_physics_score(checks)
        return ToolResult(
            data={
                "all_passed": all_passed,
                "r_phys": r_phys,
                "checks": checks,
                "crystal_system": born.get("crystal_system") if "born" in locals() else None,
                "summary": f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed",
            },
            success=True,
        )
