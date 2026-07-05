"""Thermodynamic data tool — pure-component & mixture property lookups via thermo.

Wraps the ``thermo`` Python library (https://github.com/CalebBell/thermo) to give
the agent access to chemical/physical property data, phase equilibrium, and
mixture thermodynamic properties without hitting an external API.

The ``thermo`` dependency is imported lazily inside :meth:`ThermoTool.call` so the
tool still registers and loads in minimal environments where ``thermo`` is not
installed — callers get a clear error message instead of a startup crash.

All temperatures are in Kelvin, pressures in Pascal, unless noted otherwise.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)

# thermo 是个重量级依赖 (会拉一堆 chemicals/fluids/scipy), 不装也能让 agent 起来.
# 第一次实际调用时才 import, 缺了就返回友好错误而不是崩溃.
_THERMO_AVAILABLE: bool | None = None


def _thermo_available() -> bool:
    """检查 thermo 是否可导入, 结果缓存避免重复探测."""
    global _THERMO_AVAILABLE
    if _THERMO_AVAILABLE is None:
        try:
            import thermo  # noqa: F401

            _THERMO_AVAILABLE = True
        except ImportError:
            _THERMO_AVAILABLE = False
    return _THERMO_AVAILABLE


# 纯组分可查询的物性清单. key 是 thermo Chemical 的属性名, value 是给 LLM 看的
# 人类可读标签 + 单位说明. 某些物性对特定化合物可能为 None, 会被自动跳过.
_PURE_PROPERTIES: dict[str, str] = {
    "CAS": "CAS 号",
    "MW": "摩尔分子量 [g/mol]",
    "Tm": "熔点 [K]",
    "Tb": "沸点 [K]",
    "Tc": "临界温度 [K]",
    "Pc": "临界压力 [Pa]",
    "Vc": "临界摩尔体积 [m^3/mol]",
    "Zc": "临界压缩因子 [-]",
    "omega": "偏心因子 (acentric factor) [-]",
    "phase": "当前相态 (g/l/s)",
    "Hf": "标准生成焓 [J/mol]",
    "Hc": "标准燃烧焓 [J/mol]",
    "Hfus": "熔化焓 [J/mol]",
    "Hvap": "汽化焓 [J/mol]",
    "Cp": "定压热容 [J/mol/K]",
    "Cpg": "气相定压热容 [J/mol/K]",
    "Cpl": "液相定压热容 [J/mol/K]",
    "Cvg": "气相定容热容 [J/mol/K]",
    "H": "焓 (相对参考态) [J/mol]",
    "S": "熵 (相对参考态) [J/mol/K]",
    "G": "吉布斯自由能 (相对参考态) [J/mol]",
    "rho": "密度 [kg/m^3]",
    "Vm": "摩尔体积 [m^3/mol]",
    "mu": "动力粘度 [Pa*s]",
    "k": "导热系数 [W/m/K]",
    "sigma": "表面张力 [N/m]",
    "alpha": "热扩散率 [m^2/s]",
    "isobaric_expansion": "等压膨胀系数 [1/K]",
    "SG": "相对密度 (specific gravity) [-]",
}


class ThermoToolInput(BaseModel):
    """热力学数据工具的输入参数.

    四种 action 共用一套字段, model_validator 会按 action 校验必填项,
    避免到 call() 里才发现少了关键字段.
    """

    action: Literal["properties", "thermodynamic", "phase_equilibrium", "mixture"] = (
        Field(
            ...,
            description=(
                "properties: 查询纯组分基本物性 (CAS/MW/Tb/Tc 等); "
                "thermodynamic: 指定 T/P 下的焓/熵/Gibbs/热容等; "
                "phase_equilibrium: 混合物泡点/露点; "
                "mixture: 指定组成下混合物的物性"
            ),
        )
    )
    # 纯组分查询用 compound; 混合物用 compounds + ws
    compound: str | None = Field(
        default=None,
        description=(
            "纯组分名称, 支持 IUPAC 名 / 通用名 / CAS 号 / 化学式. "
            "如 'water', 'ethanol', '7732-18-5', 'CH3OH'."
        ),
    )
    compounds: list[str] | None = Field(
        default=None,
        description="混合物组分列表 (phase_equilibrium / mixture 必填), 如 ['water', 'ethanol']",
    )
    ws: list[float] | None = Field(
        default=None,
        description="组分质量分数列表, 与 compounds 等长且和为 1, 如 [0.5, 0.5]",
    )
    T: float | None = Field(
        default=None,
        ge=0,
        description="温度 [K], 不填默认 298.15 K (thermodynamic / mixture / phase_equilibrium 可选)",
    )
    P: float | None = Field(
        default=None,
        ge=0,
        description="压力 [Pa], 不填默认 101325 Pa (1 atm)",
    )
    properties: list[str] | None = Field(
        default=None,
        description=(
            "只返回指定物性 (action='properties'/'thermodynamic' 可选). "
            "取值见工具说明, 如 ['MW', 'Tc', 'Pc', 'Cp']"
        ),
    )

    @model_validator(mode="after")
    def _check_required_fields(self) -> "ThermoToolInput":
        """按 action 校验必填字段, 缺了直接在 schema 层报错."""
        if self.action in ("properties", "thermodynamic"):
            if not self.compound:
                raise ValueError(
                    f"action '{self.action}' requires 'compound' (compound name/CAS/formula)"
                )
        else:  # phase_equilibrium / mixture
            if not self.compounds:
                raise ValueError(
                    f"action '{self.action}' requires 'compounds' (non-empty list)"
                )
            if self.ws is not None:
                if len(self.ws) != len(self.compounds):
                    raise ValueError(
                        f"len(ws)={len(self.ws)} != len(compounds)={len(self.compounds)}"
                    )
                total = sum(self.ws)
                if abs(total - 1.0) > 1e-3:
                    raise ValueError(
                        f"mass fractions must sum to 1.0, got {total:.6f}"
                    )
        return self


class ThermoToolOutput(BaseModel):
    """热力学数据工具的输出结构."""

    action: str
    # 纯组分结果: { 物性名: 值 }
    properties: dict[str, Any] | None = None
    # 混合物 / 相平衡结果
    mixture: dict[str, Any] | None = None
    # 查询条件回显
    conditions: dict[str, Any] | None = None
    warnings: list[str] = []


class ThermoTool(HuginnTool):
    """热力学物性查询工具, 封装 thermo 库的 Chemical / Mixture API.

    thermo 库在 import 阶段不做实际加载 (比较重), 只在 call() 第一次
    被调用时才真正 import. 这样即使环境里没装 thermo, agent 也能正常
    启动, 调用本工具时才报清晰的错误.
    """

    name = "thermo_tool"
    category = "materials"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset(
            {
                ResearchPhase.PLANNING,
                ResearchPhase.EXECUTION,
                ResearchPhase.VALIDATION,
            }
        ),
    )
    description = (
        "查询纯组分与混合物的热力学物性: 分子量、沸点/熔点、临界参数、热容、"
        "焓/熵/Gibbs、密度/粘度/导热系数, 以及混合物泡点/露点. "
        "基于 thermo 库 (pip install thermo), 无需联网. "
        "Actions: properties (纯组分基本物性), thermodynamic (指定 T/P 物性), "
        "phase_equilibrium (混合物泡露点), mixture (混合物物性)."
    )
    input_schema = ThermoToolInput
    output_schema = ThermoToolOutput
    read_only = True

    def is_read_only(self, args: ThermoToolInput) -> bool:
        return True

    async def validate_input(
        self, args: ThermoToolInput, context: ToolContext
    ) -> ValidationResult:
        """校验输入. thermo 没装时不在 schema 层拦截, 留到 call() 里返回更清晰."""
        return ValidationResult(result=True)

    async def call(self, args: ThermoToolInput, context: ToolContext) -> ToolResult:
        """分发到对应的 action handler. thermo 在这里才 lazy import."""
        if not _thermo_available():
            return ToolResult(
                data=None,
                success=False,
                error=(
                    "thermo library is not installed. "
                    "Install with: pip install thermo"
                ),
            )

        try:
            from thermo.chemical import Chemical
            from thermo.mixture import Mixture
        except ImportError as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"Failed to import thermo: {exc}",
            )

        handlers = {
            "properties": self._handle_properties,
            "thermodynamic": self._handle_thermodynamic,
            "phase_equilibrium": self._handle_phase_equilibrium,
            "mixture": self._handle_mixture,
        }
        handler = handlers[args.action]
        return await handler(args, Chemical, Mixture)

    # ── action handlers ──────────────────────────────────────────────

    async def _handle_properties(
        self, args: ThermoToolInput, Chemical: Any, Mixture: Any
    ) -> ToolResult:
        """查询纯组分基本物性 (CAS/MW/Tb/Tc/Pc/Vc/Hfus/Hvap 等).

        这些物性大多数是温度无关的常数, T/P 可选填. 如果填了 T/P,
        温度相关的物性 (如 Cp/rho/mu) 会按该 T/P 计算.
        """
        T = args.T if args.T is not None else 298.15
        P = args.P if args.P is not None else 101325.0

        try:
            chem = Chemical(args.compound, T=T, P=P)  # type: ignore[arg-type]
        except Exception as exc:
            return self._compound_not_found(args.compound, exc)

        wanted = args.properties if args.properties else list(_PURE_PROPERTIES.keys())
        props, warnings = self._collect_pure_properties(chem, wanted)

        output = ThermoToolOutput(
            action="properties",
            properties=props,
            conditions={"compound": args.compound, "T": T, "P": P},
            warnings=warnings,
        )
        return ToolResult(data=output.model_dump(), success=True)

    async def _handle_thermodynamic(
        self, args: ThermoToolInput, Chemical: Any, Mixture: Any
    ) -> ToolResult:
        """计算指定 T/P 下的热力学状态函数 (焓/熵/Gibbs/热容/密度/粘度等).

        thermo 的 Chemical 物性是在构造时按 T/P 计算的浮点数, 不是方法,
        所以这里每次都新建一个 Chemical 实例.
        """
        T = args.T if args.T is not None else 298.15
        P = args.P if args.P is not None else 101325.0

        try:
            chem = Chemical(args.compound, T=T, P=P)  # type: ignore[arg-type]
        except Exception as exc:
            return self._compound_not_found(args.compound, exc)

        # 默认返回温度相关的状态函数 + 相态
        default_props = [
            "phase", "MW", "Cp", "Cpg", "Cpl", "H", "S", "G",
            "rho", "Vm", "mu", "k", "sigma", "alpha",
            "isobaric_expansion", "omega",
        ]
        wanted = args.properties if args.properties else default_props
        props, warnings = self._collect_pure_properties(chem, wanted)

        output = ThermoToolOutput(
            action="thermodynamic",
            properties=props,
            conditions={"compound": args.compound, "T": T, "P": P},
            warnings=warnings,
        )
        return ToolResult(data=output.model_dump(), success=True)

    async def _handle_phase_equilibrium(
        self, args: ThermoToolInput, Chemical: Any, Mixture: Any
    ) -> ToolResult:
        """计算混合物的泡点 / 露点.

        返回给定组成下的 Tbubble/Pbubble/Tdew/Pdew. 输入 T 时算饱和压力,
        输入 P 时算饱和温度, 两个都给则用 T 优先.
        """
        T = args.T if args.T is not None else 298.15
        P = args.P if args.P is not None else 101325.0
        ws = args.ws

        try:
            mix = Mixture(args.compounds, ws=ws, T=T, P=P)  # type: ignore[arg-type]
        except Exception as exc:
            return self._compound_not_found(str(args.compounds), exc)

        result: dict[str, Any] = {
            "compounds": mix.names,
            "CASs": list(mix.CASs),
            "ws": list(mix.ws),
            "zs": list(mix.zs),
        }

        warnings: list[str] = []
        # 泡点/露点是 Mixture 的属性 (浮点), 单独 try 以免某一个挂掉全盘皆输
        for key, attr in [
            ("Tbubble", "Tbubble"),
            ("Pbubble", "Pbubble"),
            ("Tdew", "Tdew"),
            ("Pdew", "Pdew"),
        ]:
            val, warn = self._safe_get(mix, attr)
            result[key] = val
            if warn:
                warnings.append(warn)

        # 相态有助于理解 VLE 状态
        result["phase"] = self._safe_get(mix, "phase")[0]

        output = ThermoToolOutput(
            action="phase_equilibrium",
            mixture=result,
            conditions={"T": T, "P": P},
            warnings=warnings,
        )
        return ToolResult(data=output.model_dump(), success=True)

    async def _handle_mixture(
        self, args: ThermoToolInput, Chemical: Any, Mixture: Any
    ) -> ToolResult:
        """计算指定组成下混合物的物性 (MW/rho/Cp/mu/k/H/S/G 等)."""
        T = args.T if args.T is not None else 298.15
        P = args.P if args.P is not None else 101325.0
        ws = args.ws

        try:
            mix = Mixture(args.compounds, ws=ws, T=T, P=P)  # type: ignore[arg-type]
        except Exception as exc:
            return self._compound_not_found(str(args.compounds), exc)

        # 混合物可查物性, 大体与纯组分一致, 但部分纯组分专属物性 (Hfus 等) 不适用
        mix_attrs = [
            "MW", "phase", "Cp", "H", "S", "G",
            "rho", "Vm", "mu", "k", "sigma", "alpha",
            "isobaric_expansion",
        ]

        result: dict[str, Any] = {
            "compounds": list(mix.names),
            "CASs": list(mix.CASs),
            "ws": list(mix.ws),
            "zs": list(mix.zs),
        }
        warnings: list[str] = []
        for attr in mix_attrs:
            val, warn = self._safe_get(mix, attr)
            # None / NaN 一律跳过, 不塞没用的空值给 LLM
            if val is not None and val == val:  # NaN != NaN
                result[attr] = val
            if warn:
                warnings.append(warn)

        output = ThermoToolOutput(
            action="mixture",
            mixture=result,
            conditions={"T": T, "P": P},
            warnings=warnings,
        )
        return ToolResult(data=output.model_dump(), success=True)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _collect_pure_properties(
        chem: Any, wanted: list[str]
    ) -> tuple[dict[str, Any], list[str]]:
        """从 Chemical 实例批量取属性, 跳过不存在的 / None 的."""
        props: dict[str, Any] = {}
        warnings: list[str] = []
        for key in wanted:
            label = _PURE_PROPERTIES.get(key, key)
            val, warn = ThermoTool._safe_get(chem, key)
            if warn:
                warnings.append(warn)
            # None / NaN 不返回, 让 LLM 知道这个物性当前不可用
            if val is not None and val == val:  # NaN check
                props[key] = val
                props[f"{key}_desc"] = label
        return props, warnings

    @staticmethod
    def _safe_get(obj: Any, attr: str) -> tuple[Any, str | None]:
        """安全取属性, 挂了返回 None + warning 而不是抛异常."""
        try:
            val = getattr(obj, attr)
            # 属性可能是 method, 调一下拿值; thermo 大部分是 property 所以通常直接是值
            if callable(val):
                val = val()
            return val, None
        except Exception as exc:
            return None, f"failed to get '{attr}': {exc}"

    @staticmethod
    def _compound_not_found(compound: str, exc: Exception) -> ToolResult:
        """化合物查不到时的统一错误返回."""
        return ToolResult(
            data=None,
            success=False,
            error=(
                f"Could not resolve compound '{compound}': {exc}. "
                "Try IUPAC name, common name, CAS number, or chemical formula."
            ),
        )
