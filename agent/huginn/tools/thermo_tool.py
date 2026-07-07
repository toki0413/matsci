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

    action: Literal[
        "properties",
        "thermodynamic",
        "phase_equilibrium",
        "mixture",
        "phase_diagram",
        "md_thermo",
    ] = (
        Field(
            ...,
            description=(
                "properties: 查询纯组分基本物性 (CAS/MW/Tb/Tc 等); "
                "thermodynamic: 指定 T/P 下的焓/熵/Gibbs/热容等; "
                "phase_equilibrium: 混合物泡点/露点; "
                "mixture: 指定组成下混合物的物性; "
                "phase_diagram: 构建凸包相图 (需 pymatgen); "
                "md_thermo: 从 MD 轨迹时序数据计算热容/自由能等 (涨落公式, 需 numpy)"
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
    # phase_diagram 专用: 一组 entries, 每个包含 composition 和 energy
    entries: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "For phase_diagram: list of {composition, energy} dicts, "
            "e.g. [{'composition': 'Li2O', 'energy': -20.5}, ...]. "
            "Energy should be total energy per atom or per formula unit."
        ),
    )
    output_plot: str | None = Field(
        default=None,
        description=(
            "For phase_diagram: path to save convex hull plot. "
            "If None, no plot is generated."
        ),
    )
    # md_thermo: 从 MD 轨迹时序数据算热力学量, 输入可直接取自 LAMMPS thermo_data
    md_time_series: dict | None = Field(
        default=None,
        description=(
            "MD time series data for md_thermo action. "
            "Expected keys: 'time' (list[float]), 'temperature' (list[float]), "
            "'kinetic_energy' (list[float]), 'potential_energy' (list[float]). "
            "Can be taken from LAMMPS thermo_data output."
        ),
    )
    n_atoms: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Atom count for md_thermo (Cv 涨落公式需要 N 归一化). "
            "If omitted, falls back to md_time_series['n_atoms'] then 1."
        ),
    )

    @model_validator(mode="after")
    def _check_required_fields(self) -> ThermoToolInput:
        """按 action 校验必填字段, 缺了直接在 schema 层报错."""
        if self.action == "phase_diagram":
            if not self.entries:
                raise ValueError(
                    "action 'phase_diagram' requires 'entries' "
                    "(list of {composition, energy} dicts)"
                )
            return self

        if self.action == "md_thermo":
            if not self.md_time_series:
                raise ValueError(
                    "action 'md_thermo' requires 'md_time_series' "
                    "(dict with 'temperature'/'kinetic_energy'/'potential_energy' "
                    "or LAMMPS aliases 'temp'/'kineng'/'poteng')"
                )
            return self

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
        "phase_equilibrium (混合物泡露点), mixture (混合物物性), "
        "phase_diagram (凸包相图, 需 pymatgen), "
        "md_thermo (从 MD 时序算热容/自由能, 需 numpy)."
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
        # phase_diagram 用的是 pymatgen, 不依赖 thermo 库, 单独走
        if args.action == "phase_diagram":
            return await self._handle_phase_diagram(args, context)

        # md_thermo 只用 numpy 涨落公式, 也不依赖 thermo 库
        if args.action == "md_thermo":
            return await self._md_thermo(args, context)

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

    async def _handle_phase_diagram(
        self, args: ThermoToolInput, context: ToolContext
    ) -> ToolResult:
        """构建凸包相图, 返回稳定相 + 分解焓.

        用 pymatgen 的 PhaseDiagram / PDPlotter. pymatgen 没装就优雅降级,
        返回明确的安装提示而不是崩溃.
        """
        try:
            from pymatgen.analysis.phase_diagram import (
                PDPlotter,
                PhaseDiagram,
            )
            from pymatgen.core import Composition
            from pymatgen.entries.computed_entries import ComputedEntry
        except ImportError:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    "pymatgen is required for phase_diagram. "
                    "Install with: pip install pymatgen"
                ),
            )

        # 把 user entries 转成 ComputedEntry
        comp_entries: list[ComputedEntry] = []
        for e in args.entries or []:
            try:
                comp = Composition(e["composition"])
                energy = float(e["energy"])
                comp_entries.append(ComputedEntry(comp, energy))
            except Exception as exc:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Invalid entry {e}: {exc}",
                )

        if not comp_entries:
            return ToolResult(
                data=None,
                success=False,
                error="No valid entries provided for phase diagram",
            )

        try:
            pd = PhaseDiagram(comp_entries)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"Failed to construct phase diagram: {exc}",
            )

        # 稳定相
        stable_phases: list[dict[str, Any]] = []
        for entry in pd.stable_entries:
            stable_phases.append({
                "composition": entry.composition.reduced_formula,
                "energy": entry.energy,
                "energy_per_atom": entry.energy_per_atom,
            })

        # 每个输入 entry 的分解焓和距离凸包的能量
        decomp_info: list[dict[str, Any]] = []
        for entry in comp_entries:
            try:
                decomp, e_above = pd.get_decomp_and_e_above_hull(entry)
                decomp_dict = {
                    e.composition.reduced_formula: v for e, v in decomp.items()
                }
            except Exception:
                e_above = None
                decomp_dict = {}
            decomp_info.append({
                "composition": entry.composition.reduced_formula,
                "energy": entry.energy,
                "energy_above_hull": e_above,
                "is_stable": entry in pd.stable_entries,
                "decomposition": decomp_dict,
            })

        result: dict[str, Any] = {
            "stable_phases": stable_phases,
            "all_entries": decomp_info,
            "n_stable": len(stable_phases),
            "n_total": len(comp_entries),
        }

        # 画图 — 失败不影响数据返回
        if args.output_plot:
            try:
                plotter = PDPlotter(pd)
                fig = plotter.get_plot()
                plot_path = args.output_plot
                # PDPlotter 返回 matplotlib Figure, 直接 savefig
                if hasattr(fig, "savefig"):
                    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                else:
                    # 某些 pymatgen 版本返回 Plotly 对象
                    fig.write_image(plot_path)
                result["plot_path"] = plot_path
            except Exception as exc:
                result["plot_warning"] = f"Failed to generate plot: {exc}"

        output = ThermoToolOutput(
            action="phase_diagram",
            properties=result,
            conditions={"n_entries": len(comp_entries)},
            warnings=[],
        )
        return ToolResult(data=output.model_dump(), success=True)

    async def _md_thermo(
        self, args: ThermoToolInput, context: ToolContext
    ) -> ToolResult:
        """从 MD 轨迹时序数据计算热力学量 (Cv / 自由能 / 能量涨落).

        把 LAMMPS thermo 输出的 KE/PE/T 时序转成统计力学量. 全部基于
        涨落公式, 不做积分也不拟合, 纯 numpy 均值/方差. thermo 库不需要.
        """
        import numpy as np

        # 物理常数: k_B [eV/K], 1 eV/particle -> J/mol
        KB = 8.617333e-5
        EV_TO_J_MOL = 96485.0
        T_REF = 300.0  # Helmholtz 近似里的参考温度

        raw = args.md_time_series or {}

        def _resolve(*keys: str) -> np.ndarray | None:
            # 同一个量在 LAMMPS thermo 里列名不统一, 依次试一遍别名
            for k in keys:
                v = raw.get(k)
                if v is not None:
                    return np.asarray(v, dtype=float)
            return None

        temp = _resolve("temperature", "temp", "t")
        ke = _resolve("kinetic_energy", "kineng", "ke")
        pe = _resolve("potential_energy", "poteng", "pe")
        total = _resolve("total_energy", "toteng", "etotal", "te")
        # time 仅回显, 不参与计算 (step 也能当时间轴用)
        _time = _resolve("time", "step")  # noqa: F841

        if temp is None:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    "md_thermo needs a temperature series "
                    "('temperature' or LAMMPS alias 'temp')"
                ),
            )

        # 总能量: 优先用 LAMMPS 直接给的 TotEng, 没有就 KE+PE 拼
        if total is None and ke is not None and pe is not None:
            total = ke + pe
        elif total is None and pe is not None:
            # NVT thermostat 下 KE 波动由 thermostat 吸收, 用 PE 涨落也算得动
            total = pe

        # 对齐长度 (LAMMPS thermo 列本应等长, 这里防一手长度不齐)
        arrays = [a for a in (temp, ke, pe, total) if a is not None]
        n = min(a.size for a in arrays) if arrays else temp.size
        temp = temp[:n]
        if ke is not None:
            ke = ke[:n]
        if pe is not None:
            pe = pe[:n]
        if total is not None:
            total = total[:n]

        # 原子数: 显式 > md_time_series 内嵌 > 1
        n_atoms = args.n_atoms
        if n_atoms is None:
            n_atoms = int(raw.get("n_atoms", 1) or 1)
        if n_atoms < 1:
            n_atoms = 1

        avg_t = float(np.mean(temp))
        std_t = float(np.std(temp))

        result: dict[str, Any] = {
            "avg_temperature": avg_t,
            "std_temperature": std_t,
            "n_samples": int(n),
            "n_atoms": n_atoms,
        }
        if ke is not None:
            result["avg_kinetic"] = float(np.mean(ke))
        if pe is not None:
            result["avg_potential"] = float(np.mean(pe))

        cv_eV_K: float | None = None
        method = "fluctuation_formula"
        note = ""

        if total is not None and n > 1:
            avg_e = float(np.mean(total))
            var_e = float(np.var(total))  # <E^2> - <E>^2
            std_e = float(np.std(total))
            result["avg_total"] = avg_e
            result["std_total"] = std_e

            # Cv (per atom) = <dE^2> / (k_B * T^2 * N),  eV/K
            if avg_t > 0 and var_e > 0:
                cv_eV_K = var_e / (KB * avg_t * avg_t * n_atoms)
                note = "Cv from energy fluctuations: (<E^2>-<E>^2)/(k_B*T^2*N)"
            else:
                note = (
                    "energy variance or temperature non-positive; "
                    "Cv from energy skipped"
                )
        else:
            result["avg_total"] = None
            result["std_total"] = None

        # 温度涨落兜底: 只在能量涨落不可用时才用
        # per-atom 近似 k_B * <dT^2>/<T>^2; NVT 下温度几乎不涨, 仅作 fallback
        if cv_eV_K is None and std_t > 0 and avg_t > 0:
            cv_eV_K = KB * (std_t * std_t) / (avg_t * avg_t)
            method = "temperature_fluctuation_approx"
            note = (
                "Cv approximated from temperature fluctuations "
                "(k_B*<dT^2>/<T>^2); energy-based value unavailable"
            )

        if cv_eV_K is not None:
            result["cv_eV_K"] = cv_eV_K
            result["cv_j_mol_k"] = cv_eV_K * EV_TO_J_MOL
        else:
            result["cv_eV_K"] = None
            result["cv_j_mol_k"] = None

        # Helmholtz 自由能: F = <E> - T*S, S ~ Cv_total * ln(T/T_ref)
        # 没有配分函数只能粗估; 需要总能量和 (per-atom) Cv 推出总 Cv
        avg_e = result.get("avg_total")
        if (
            avg_e is not None
            and cv_eV_K is not None
            and avg_t > 0
            and avg_t != T_REF  # ln(1)=0 时 S=0, 没意义
        ):
            cv_total = cv_eV_K * n_atoms  # eV/K, 整个体系
            s_est = cv_total * float(np.log(avg_t / T_REF))
            result["helmholtz_free_energy_eV"] = avg_e - avg_t * s_est
        else:
            result["helmholtz_free_energy_eV"] = None

        result["method"] = method
        result["note"] = note

        output = ThermoToolOutput(
            action="md_thermo",
            properties=result,
            conditions={"n_atoms": n_atoms, "n_samples": int(n)},
            warnings=[],
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
