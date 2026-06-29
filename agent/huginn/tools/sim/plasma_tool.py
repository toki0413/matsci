"""等离子体仿真工具 —— 借鉴 ai4plasma 的思路, 用 NumPy + sklearn 做轻量等离子体计算.

ai4plasma (https://github.com/mathboylinlin/ai4plasma) 是首个面向等离子体物理
的 AI 库, 核心是 PINN / DeepONet / CS-PINN 求解等离子体 PDE, 外加 1D 弧等离子体
FVM 仿真和等离子体性质计算. 本工具不依赖 PyTorch, 而是用经典数值方法 + sklearn
代理模型覆盖常见等离子体场景, 适合 agent 内部做快速预估和参数扫描. 真要跑全 PINN
训练还得交给 ai4plasma 本体.

涵盖的物理:
  - PIC (Particle-in-Cell) 一维静电
  - MHD (磁流体) 一维理想
  - 鞘层 (Bohm 判据 + Debye 长度 + 浮动电位)
  - 输运系数 (Spitzer 电阻率 / 热导 / 扩散 / 粘性)
  - 波色散 (Langmuir / 离子声波 / Alfvén / whistler)
  - ML 代理 (sklearn 拟合等离子体性质)
  - 弧等离子体 (1D Elenbaas-Heller 简化)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


# 物理常数 (SI)
_E = 1.602176634e-19       # 电子电荷 (C)
_ME = 9.1093837015e-31     # 电子质量 (kg)
_MP = 1.67262192369e-27    # 质子质量 (kg)
_EPS0 = 8.8541878128e-12   # 真空介电常数 (F/m)
_KB = 1.380649e-23         # 玻尔兹曼常数 (J/K)
_MU0 = 1.25663706212e-6    # 真空磁导率 (N/A^2)
_C = 2.99792458e8          # 光速 (m/s)


class PlasmaToolInput(BaseModel):
    # pydantic v2 默认把 model_ 开头字段当保护命名空间, 这里 model_type 只是
    # ML 模型类型选择, 跟 pydantic 无关, 关掉保护避免警告
    model_config = {"protected_namespaces": ()}

    action: Literal[
        "pic_simulation",
        "fluid_simulation",
        "sheath_model",
        "transport_coefficients",
        "wave_dispersion",
        "ml_surrogate",
        "arc_plasma",
    ] = Field(..., description="等离子体计算动作")

    # ---- 通用参数 ----
    plasma_density: float = Field(default=1e18, gt=0, description="等离子体数密度 n (m^-3)")
    temperature: float = Field(default=1.0, gt=0, description="温度 (eV), 单物种用这个")
    electron_temp: float = Field(default=1.0, gt=0, description="电子温度 Te (eV)")
    ion_temp: float = Field(default=0.1, gt=0, description="离子温度 Ti (eV)")
    B_field: float = Field(default=0.0, ge=0, description="磁场 B (T)")

    # ---- PIC ----
    grid_size: int = Field(default=64, ge=8, description="PIC/MHD 网格点数")
    domain_length: float = Field(default=1e-2, gt=0, description="仿真域长度 (m)")
    time_step: float = Field(default=1e-12, gt=0, description="时间步长 (s)")
    num_steps: int = Field(default=100, ge=1, description="仿真步数")
    num_particles: int = Field(default=2000, ge=10, description="每物种粒子数 (PIC)")
    species: list[str] = Field(default_factory=lambda: ["electron", "ion"])

    # ---- MHD ----
    velocity: float = Field(default=0.0, description="初始流速 (m/s)")
    pressure: float = Field(default=1e3, gt=0, description="热压 (Pa)")
    boundary: Literal["periodic", "outflow"] = Field(default="periodic")

    # ---- sheath ----
    wall_material: str = Field(default="Cu")

    # ---- transport ----
    collision_model: Literal["spitzer", "constant", "lorentz"] = Field(default="spitzer")
    species_name: str = Field(default="electron")
    charge_state: int = Field(default=1, ge=1, description="电离电荷数 Z")
    coulomb_log: float = Field(default=17.0, gt=0, description="库仑对数 lnΛ")

    # ---- wave dispersion ----
    wave_type: Literal["langmuir", "ion_acoustic", "alfven", "whistler"] = Field(
        default="langmuir"
    )
    k_values: list[float] = Field(
        default_factory=lambda: [1.0, 10.0, 100.0],
        description="波数 k (m^-1)",
    )

    # ---- ML surrogate ----
    model_type: Literal["nn", "gp", "RandomForest"] = Field(default="RandomForest")
    target_property: str = Field(default="resistivity")
    X_train: list[list[float]] = Field(default_factory=list)
    y_train: list[float] = Field(default_factory=list)
    X_pred: list[list[float]] = Field(default_factory=list)

    # ---- arc plasma ----
    arc_current: float = Field(default=100.0, gt=0, description="电弧电流 (A)")
    arc_radius: float = Field(default=5e-3, gt=0, description="弧柱半径 (m)")
    wall_temperature: float = Field(default=300.0, gt=0, description="器壁/边界温度 (K)")


class PlasmaTool(HuginnTool):
    """等离子体仿真工具 — PIC / MHD / 鞘层 / 输运 / 波 / ML 代理 / 弧等离子体.

    借鉴 ai4plasma 的方法学, 但用经典数值方法 (NumPy) + sklearn 代理模型实现,
    不依赖 PyTorch. 适合 agent 内部做参数扫描和快速预估.
    """

    name = "plasma_tool"
    category = "sim"
    description = (
        "等离子体物理仿真工具, 借鉴 ai4plasma 的方法. "
        "支持 PIC 粒子模拟、MHD 流体、鞘层模型、输运系数、波色散关系、"
        "ML 代理模型和 1D 弧等离子体. 用 NumPy + sklearn 实现, 不依赖 PyTorch."
    )
    input_schema = PlasmaToolInput
    # 这些 action 都是纯计算, 不写文件, 不调外部服务
    read_only = True

    async def call(self, args: PlasmaToolInput, context: ToolContext) -> ToolResult:
        try:
            if args.action == "pic_simulation":
                return self._pic_simulation(args)
            if args.action == "fluid_simulation":
                return self._fluid_simulation(args)
            if args.action == "sheath_model":
                return self._sheath_model(args)
            if args.action == "transport_coefficients":
                return self._transport_coefficients(args)
            if args.action == "wave_dispersion":
                return self._wave_dispersion(args)
            if args.action == "ml_surrogate":
                return self._ml_surrogate(args)
            if args.action == "arc_plasma":
                return self._arc_plasma(args)
            return ToolResult(
                data=None, success=False, error=f"未知 action: {args.action}"
            )
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"plasma_tool 执行失败: {e}"
            )

    # ============================================================ PIC
    def _pic_simulation(self, args: PlasmaToolInput) -> ToolResult:
        """一维静电 PIC.

        简化版: 双物种 (电子+离子), CIC 电荷沉积, FFT 解泊松, leapfrog 推进.
        没做粒子合并/分裂, 没做碰撞, 适合教学和小规模预估.
        真要跑大规模 PIC 还得上 VPIC / OSIRIS.
        """
        L = args.domain_length
        N = args.grid_size
        dx = L / N
        dt = args.time_step
        n_steps = args.num_steps
        n_part = args.num_particles

        x_grid = np.linspace(0, L, N, endpoint=False)
        rng = np.random.default_rng(42)

        species_list = list(args.species) if args.species else ["electron", "ion"]
        # 背景密度 n0, 每个宏观粒子代表的真实粒子数 macro
        n0 = args.plasma_density
        macro = n0 * L / n_part

        particles: dict[str, dict[str, Any]] = {}
        for sp in species_list:
            x_p = rng.uniform(0, L, n_part)
            if sp.lower().startswith("e"):
                T_J = max(args.electron_temp, args.temperature) * _E
                m, q = _ME, -_E
            else:  # 默认质子
                T_J = max(args.ion_temp, args.temperature * 0.1) * _E
                m, q = _MP, _E
            v_th = np.sqrt(T_J / m)
            v_p = rng.normal(0.0, v_th, n_part)
            particles[sp] = {"x": x_p, "v": v_p, "q": q, "m": m}

        def _deposit() -> np.ndarray:
            rho = np.zeros(N)
            for p in particles.values():
                xn = p["x"] / dx
                i0 = np.floor(xn).astype(int) % N
                i1 = (i0 + 1) % N
                w1 = xn - np.floor(xn)
                w0 = 1.0 - w1
                rho[i0] += w0 * p["q"] * macro / dx
                rho[i1] += w1 * p["q"] * macro / dx
            return rho

        def _poisson(rho: np.ndarray) -> np.ndarray:
            # d²φ/dx² = -ρ/ε0, 周期边界 → FFT
            rho_k = np.fft.fft(rho)
            k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
            k2 = k**2
            k2[0] = 1.0  # 避免 0 除, DC 分量后置 0
            phi_k = rho_k / (_EPS0 * k2)
            phi_k[0] = 0.0
            return np.real(np.fft.ifft(phi_k))

        def _interp(E: np.ndarray, xp: np.ndarray) -> np.ndarray:
            xn = xp / dx
            i0 = np.floor(xn).astype(int) % N
            i1 = (i0 + 1) % N
            w1 = xn - np.floor(xn)
            w0 = 1.0 - w1
            return w0 * E[i0] + w1 * E[i1]

        energy_history: list[dict[str, float]] = []
        rho = phi = E = np.zeros(N)
        for step in range(n_steps):
            rho = _deposit()
            phi = _poisson(rho)
            E = -np.gradient(phi, dx)
            # leapfrog: v_{n-1/2} → v_{n+1/2}, x_n → x_{n+1}
            for p in particles.values():
                Ep = _interp(E, p["x"])
                p["v"] += (p["q"] / p["m"]) * Ep * dt
                p["x"] += p["v"] * dt
                p["x"] %= L  # 周期边界

            KE = sum(
                0.5 * p["m"] * float(np.sum(p["v"] ** 2)) * macro
                for p in particles.values()
            )
            PE = 0.5 * _EPS0 * float(np.sum(E**2)) * dx
            energy_history.append(
                {"step": step, "KE": KE, "PE": PE, "total": KE + PE}
            )

        return ToolResult(
            data={
                "action": "pic_simulation",
                "config": {
                    "grid_size": N,
                    "domain_length": L,
                    "time_step": dt,
                    "num_steps": n_steps,
                    "num_particles": n_part,
                    "species": species_list,
                    "macro_weight": macro,
                },
                "grid": x_grid.tolist(),
                "energy_history": energy_history,
                "final_fields": {
                    "rho": rho.tolist(),
                    "phi": phi.tolist(),
                    "E": E.tolist(),
                },
                "phase_space": {
                    sp: {
                        "x": p["x"].tolist(),
                        "v": p["v"].tolist(),
                    }
                    for sp, p in particles.items()
                },
                "notes": (
                    "一维静电 PIC: CIC 沉积 + FFT 泊松 + leapfrog 推进. "
                    "未含碰撞和电离, 周期边界. 适合教学和参数预估."
                ),
            },
            success=True,
        )

    # ============================================================ MHD
    def _fluid_simulation(self, args: PlasmaToolInput) -> ToolResult:
        """一维理想 MHD (Lax-Friedrichs 显式).

        简化: 等温近似 (γ=1), 一阶 LF 格式, 不含电阻/粘性耗散.
        守恒变量: ρ, ρv, p, B. 通量参考理想 MHD 教科书 (Goedbloed).
        生产级仿真请用 Athena++ / PLUTO.
        """
        L = args.domain_length
        N = args.grid_size
        dx = L / N
        n_steps = args.num_steps

        # MHD 密度通常比实验室稀薄等离子体大, 这里给个下限
        rho0 = max(args.plasma_density, 1e15) * _MP
        p0 = args.pressure
        B0 = args.B_field
        v0 = args.velocity
        gamma = 1.0  # 等温

        c_s = np.sqrt(gamma * p0 / rho0)
        v_A = B0 / np.sqrt(_MU0 * rho0) if B0 > 0 else 0.0
        v_max = abs(v0) + max(c_s, v_A, 1.0)
        dt = 0.4 * dx / v_max  # CFL=0.4

        x = np.linspace(0, L, N, endpoint=False)
        # 初始: 均匀背景 + 小扰动
        rho = rho0 + 0.01 * rho0 * np.sin(2 * np.pi * x / L)
        v = v0 + 1e-3 * np.sin(2 * np.pi * x / L)
        p = p0 + 0.01 * p0 * np.sin(2 * np.pi * x / L)
        B = (
            B0 + 0.001 * B0 * np.sin(2 * np.pi * x / L)
            if B0 > 0
            else np.zeros(N)
        )

        def _roll(a: np.ndarray, shift: int) -> np.ndarray:
            if args.boundary == "periodic":
                return np.roll(a, shift)
            # outflow: 边界值复制
            out = np.empty_like(a)
            out[1:-1] = a[1:-1]
            out[0] = a[0]
            out[-1] = a[-1]
            return out

        history: list[dict[str, Any]] = []
        for step in range(n_steps):
            # 通量 (理想 MHD, 等温)
            F_rho = rho * v
            F_mom = rho * v**2 + p + B**2 / (2 * _MU0)
            F_p = gamma * p * v
            F_B = v * B

            rho_new = 0.5 * (_roll(rho, -1) + _roll(rho, 1)) - 0.5 * dt / dx * (
                _roll(F_rho, -1) - _roll(F_rho, 1)
            )
            mom = rho * v
            mom_new = 0.5 * (_roll(mom, -1) + _roll(mom, 1)) - 0.5 * dt / dx * (
                _roll(F_mom, -1) - _roll(F_mom, 1)
            )
            p_new = 0.5 * (_roll(p, -1) + _roll(p, 1)) - 0.5 * dt / dx * (
                _roll(F_p, -1) - _roll(F_p, 1)
            )
            B_new = 0.5 * (_roll(B, -1) + _roll(B, 1)) - 0.5 * dt / dx * (
                _roll(F_B, -1) - _roll(F_B, 1)
            )

            rho = np.maximum(rho_new, 1e-30 * rho0)
            v = mom_new / rho
            p = np.maximum(p_new, 1e-30 * p0)
            B = B_new

            if step % max(1, n_steps // 20) == 0 or step == n_steps - 1:
                history.append(
                    {
                        "step": step,
                        "t": step * dt,
                        "rho_mean": float(np.mean(rho)),
                        "v_mean": float(np.mean(v)),
                        "p_mean": float(np.mean(p)),
                        "B_mean": float(np.mean(B)) if B0 > 0 else 0.0,
                    }
                )

        return ToolResult(
            data={
                "action": "fluid_simulation",
                "config": {
                    "grid_size": N,
                    "domain_length": L,
                    "dt": dt,
                    "num_steps": n_steps,
                    "boundary": args.boundary,
                    "gamma": gamma,
                    "sound_speed": float(c_s),
                    "alfven_speed": float(v_A),
                },
                "grid": x.tolist(),
                "history": history,
                "final_state": {
                    "rho": rho.tolist(),
                    "velocity": v.tolist(),
                    "pressure": p.tolist(),
                    "B_field": B.tolist(),
                },
                "notes": (
                    "一维理想 MHD, Lax-Friedrichs 一阶显式, 等温近似. "
                    "非守恒形式 (扰动演化OK, 激波捕捉不可靠). "
                    "生产仿真请用 Athena++ / PLUTO."
                ),
            },
            success=True,
        )

    # ============================================================ sheath
    def _sheath_model(self, args: PlasmaToolInput) -> ToolResult:
        """等离子体鞘层模型.

        Bohm 判据: 离子进入鞘层速度 ≥ c_s = sqrt(k*T_e/m_i)
        Debye 长度: λ_D = sqrt(ε0*k*T_e / (n_e*e²))
        浮动电位 (无碰撞, 单一离子种类):
          V_s = (k*T_e/e) * ln(sqrt(m_i/(2π*m_e)))
        离子通量 (鞘层边): Γ_i = n_i * c_s
        电子通量 (打到壁): Γ_e = n_e * v_th,e/4 * exp(-e*V_s/(k*T_e))
        """
        n = args.plasma_density
        Te_eV = max(args.electron_temp, args.temperature)
        Ti_eV = max(args.ion_temp, args.temperature * 0.1)
        Te_J = Te_eV * _E
        Ti_J = Ti_eV * _E
        m_i = _MP  # 默认氢离子, 可扩展

        # Debye 长度
        lambda_D = np.sqrt(_EPS0 * Te_J / (n * _E**2))

        # Bohm 速度 (声速)
        c_s = np.sqrt(Te_J / m_i)

        # 浮动鞘层电位 (kT_e/e 单位是 V, 系数见 Chen 等离子体物理教材)
        V_s = (Te_J / _E) * np.log(np.sqrt(m_i / (2 * np.pi * _ME)))

        # 离子通量 (鞘层边)
        gamma_i = n * c_s

        # 电子热速度
        v_th_e = np.sqrt(8 * Te_J / (np.pi * _ME))
        # 电子通量打到壁 (受鞘层电位排斥)
        gamma_e = 0.25 * n * v_th_e * np.exp(-V_s / Te_eV)

        # 鞘层厚度估算 (Liebig 型经验, ~几个 λ_D)
        sheath_thickness = 5.0 * lambda_D

        return ToolResult(
            data={
                "action": "sheath_model",
                "inputs": {
                    "plasma_density": n,
                    "electron_temp_eV": Te_eV,
                    "ion_temp_eV": Ti_eV,
                    "wall_material": args.wall_material,
                    "ion_mass": m_i,
                },
                "debye_length": float(lambda_D),
                "bohm_velocity": float(c_s),
                "sheath_potential_V": float(V_s),
                "ion_flux": float(gamma_i),
                "electron_flux_to_wall": float(gamma_e),
                "sheath_thickness_estimate": float(sheath_thickness),
                "notes": (
                    "无碰撞鞘层模型, 单一氢离子假设. 浮动电位按 Chen 教材公式. "
                    "鞘层厚度为 ~5λ_D 经验估计, 真实厚度需解 Poisson-Boltzmann."
                ),
            },
            success=True,
        )

    # ============================================================ transport
    def _transport_coefficients(self, args: PlasmaToolInput) -> ToolResult:
        """输运系数 (Spitzer).

        电子-离子碰撞频率 ν_ei (Spitzer):
          ν_ei = (4√(2π)/3) * n_i*Z²*e⁴*lnΛ / ((4πε0)² * m_e^(1/2) * (kT_e)^(3/2))
        电阻率: η = m_e*ν_ei / (n_e*e²)
        电子热导率 (Spitzer-Härm 近似): κ_e ≈ 3.9 * n_e*k_B²*T_e*τ_e/m_e
        离子热导率: κ_i ≈ 3.9 * n_i*k_B²*T_i*τ_i/m_i
        扩散系数 (经典): D = k*T/(m*ν)
        粘性: μ ≈ n*k*T*τ
        """
        n_e = args.plasma_density
        Z = args.charge_state
        n_i = n_e / Z
        lnΛ = args.coulomb_log
        Te_eV = max(args.electron_temp, args.temperature)
        Ti_eV = max(args.ion_temp, args.temperature * 0.1)
        Te_J = Te_eV * _E
        Ti_J = Ti_eV * _E

        if args.collision_model != "spitzer":
            # constant / lorentz 模型给个固定碰撞频率, 这里简化用 spitzer 走通
            # 真要切模型得自己指定 ν
            pass

        # 电子-离子碰撞频率
        coeff = (4.0 * np.sqrt(2.0 * np.pi) / 3.0) * (
            n_i * Z**2 * _E**4 * lnΛ
        ) / ((4.0 * np.pi * _EPS0) ** 2 * np.sqrt(_ME) * Te_J**1.5)
        nu_ei = coeff

        # 离子-离子碰撞频率 (同种离子, ~√2 倍电子公式换质量)
        nu_ii = (
            (4.0 * np.sqrt(np.pi) / 3.0)
            * (n_i * Z**4 * _E**4 * lnΛ)
            / ((4.0 * np.pi * _EPS0) ** 2 * np.sqrt(_MP) * Ti_J**1.5)
        )

        # Spitzer 电阻率 (平行磁场)
        eta = _ME * nu_ei / (n_e * _E**2)

        # 电子热导率 (Spitzer-Härm, 系数 3.9 来自 Braginskii)
        tau_e = 1.0 / nu_ei if nu_ei > 0 else 1.0
        kappa_e = 3.9 * n_e * _KB**2 * Te_eV * _E * tau_e / _ME

        # 离子热导率
        tau_i = 1.0 / nu_ii if nu_ii > 0 else 1.0
        kappa_i = 3.9 * n_i * _KB**2 * Ti_eV * _E * tau_i / _MP

        # 经典扩散系数 (平行磁场)
        D_e = Te_J / (_ME * nu_ei) if nu_ei > 0 else float("inf")
        D_i = Ti_J / (_MP * nu_ii) if nu_ii > 0 else float("inf")

        # 粘性 (Braginskii 离子粘性, ν_ii 主导)
        mu_i = n_i * _KB * Te_eV * _E * tau_i * 0.96  # 0.96 是 Braginskii 系数

        # 经验校验: Spitzer 电阻率简化公式 η ≈ 1.03e-4 * Z * lnΛ / T_eV^1.5
        eta_check = 1.03e-4 * Z * lnΛ / Te_eV**1.5

        return ToolResult(
            data={
                "action": "transport_coefficients",
                "inputs": {
                    "n_e": n_e,
                    "Z": Z,
                    "T_e_eV": Te_eV,
                    "T_i_eV": Ti_eV,
                    "coulomb_log": lnΛ,
                    "collision_model": args.collision_model,
                },
                "electron_ion_collision_freq": float(nu_ei),
                "ion_ion_collision_freq": float(nu_ii),
                "resistivity_Ohm_m": float(eta),
                "resistivity_spitzer_check": float(eta_check),
                "electron_thermal_conductivity": float(kappa_e),
                "ion_thermal_conductivity": float(kappa_i),
                "electron_diffusion_coeff": float(D_e),
                "ion_diffusion_coeff": float(D_i),
                "ion_viscosity": float(mu_i),
                "notes": (
                    "Spitzer-Härm 经典输运, 平行磁场. "
                    "系数参考 Braginskii 'Transport Processes in a Plasma'. "
                    "横场输运需除 (ω_c*τ)², 强磁场下显著减小."
                ),
            },
            success=True,
        )

    # ============================================================ wave dispersion
    def _wave_dispersion(self, args: PlasmaToolInput) -> ToolResult:
        """等离子体波色散关系.

        - Langmuir (Bohm-Gross): ω² = ω_pe² + 3*k²*v_te²
        - 离子声波: ω² = k²*c_s² / (1 + k²*λ_D²)
        - Alfvén (剪切, 平行传播): ω = k*v_A
        - Whistler (平行, 低频): ω = k²*c²*ω_ce/ω_pe²
        """
        n_e = args.plasma_density
        Te_eV = max(args.electron_temp, args.temperature)
        Ti_eV = max(args.ion_temp, args.temperature * 0.1)
        B = args.B_field
        ks = np.array(args.k_values, dtype=float)
        if ks.size == 0:
            ks = np.array([1.0, 10.0, 100.0])

        Te_J = Te_eV * _E
        Ti_J = Ti_eV * _E
        m_i = _MP

        # 等离子体频率
        omega_pe = np.sqrt(n_e * _E**2 / (_EPS0 * _ME))
        omega_pi = np.sqrt(n_e * _E**2 / (_EPS0 * m_i))
        # 热速度
        v_te = np.sqrt(Te_J / _ME)
        v_ti = np.sqrt(Ti_J / m_i)
        # Debye 长度和声速
        lambda_D = np.sqrt(_EPS0 * Te_J / (n_e * _E**2))
        c_s = np.sqrt(Te_J / m_i)  # T_i << T_e 极限
        # Alfven 速度
        rho_m = n_e * m_i  # 质量密度
        v_A = B / np.sqrt(_MU0 * rho_m) if B > 0 else 0.0
        # 回旋频率
        omega_ce = _E * B / _ME if B > 0 else 0.0

        wave_type = args.wave_type
        results: list[dict[str, Any]] = []
        for k in ks:
            k_val = float(k)
            if wave_type == "langmuir":
                omega = np.sqrt(omega_pe**2 + 3.0 * k_val**2 * v_te**2)
                relation = f"ω² = ω_pe² + 3k²v_te², ω_pe={omega_pe:.3e} rad/s"
            elif wave_type == "ion_acoustic":
                denom = 1.0 + (k_val * lambda_D) ** 2
                omega = k_val * c_s / np.sqrt(denom)
                relation = f"ω² = k²c_s²/(1+k²λ_D²), c_s={c_s:.3e} m/s"
            elif wave_type == "alfven":
                if B <= 0:
                    return ToolResult(
                        data=None,
                        success=False,
                        error="Alfvén 波需要非零磁场 B_field",
                    )
                omega = k_val * v_A
                relation = f"ω = k*v_A, v_A={v_A:.3e} m/s"
            else:  # whistler
                if B <= 0:
                    return ToolResult(
                        data=None,
                        success=False,
                        error="whistler 波需要非零磁场 B_field",
                    )
                # ω = k²c²ω_ce/ω_pe² (低频色散, 平行传播)
                omega = k_val**2 * _C**2 * omega_ce / omega_pe**2
                relation = f"ω = k²c²ω_ce/ω_pe², ω_ce={omega_ce:.3e} rad/s"

            v_phase = omega / k_val if k_val != 0 else float("inf")
            # 群速度 dω/dk, 用解析导数
            if wave_type == "langmuir":
                v_group = 3.0 * k_val * v_te**2 / omega if omega > 0 else 0.0
            elif wave_type == "ion_acoustic":
                # ω = k*c_s / sqrt(1+k²λ_D²) → dω/dk = c_s / (1+k²λ_D²)^1.5
                v_group = c_s / (denom**1.5)
            elif wave_type == "alfven":
                v_group = v_A
            else:  # whistler
                v_group = 2.0 * k_val * _C**2 * omega_ce / omega_pe**2

            results.append(
                {
                    "k": k_val,
                    "omega": float(omega),
                    "phase_velocity": float(v_phase),
                    "group_velocity": float(v_group),
                }
            )

        return ToolResult(
            data={
                "action": "wave_dispersion",
                "wave_type": wave_type,
                "relation": relation,
                "plasma_params": {
                    "n_e": n_e,
                    "T_e_eV": Te_eV,
                    "T_i_eV": Ti_eV,
                    "B_T": B,
                    "omega_pe": float(omega_pe),
                    "omega_pi": float(omega_pi),
                    "omega_ce": float(omega_ce),
                    "v_te": float(v_te),
                    "v_A": float(v_A),
                    "c_s": float(c_s),
                    "lambda_D": float(lambda_D),
                },
                "dispersion": results,
                "notes": (
                    f"解析色散关系 ({wave_type}), 无碰撞冷/暖等离子体近似. "
                    "阻尼和耗散未含, 真实情形需解全色散方程 (含碰撞/动力学)."
                ),
            },
            success=True,
        )

    # ============================================================ ML surrogate
    def _ml_surrogate(self, args: PlasmaToolInput) -> ToolResult:
        """ML 代理模型 — 用 sklearn 拟合等离子体性质.

        受 ai4plasma PINN/DeepONet 思路启发, 这里走轻量路线: sklearn 三选一
        (RandomForest / GP / MLP). 输入特征 → 目标性质, 给预测 + 不确定性 (GP) /
        特征重要度 (RF). 训练数据由调用方提供 (X_train, y_train).
        """
        X_train = np.array(args.X_train, dtype=float)
        y_train = np.array(args.y_train, dtype=float)
        X_pred = np.array(args.X_pred, dtype=float) if args.X_pred else X_train

        if X_train.size == 0 or y_train.size == 0:
            return ToolResult(
                data=None,
                success=False,
                error="ml_surrogate 需要 X_train 和 y_train",
            )
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(-1, 1)
        if y_train.shape[0] != X_train.shape[0]:
            return ToolResult(
                data=None,
                success=False,
                error=f"样本数不匹配: X_train={X_train.shape[0]}, y_train={y_train.shape[0]}",
            )

        model_type = args.model_type
        try:
            if model_type == "gp":
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, ConstantKernel

                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
                model = GaussianProcessRegressor(
                    kernel=kernel, n_restarts_optimizer=3, alpha=1e-6
                )
                model.fit(X_train, y_train)
                pred, std = model.predict(X_pred, return_std=True)
                feature_importance = None
            elif model_type == "nn":
                from sklearn.neural_network import MLPRegressor

                # 隐藏层规模按特征数粗调, 不做超参搜索
                n_features = X_train.shape[1]
                hidden = (64, 64) if n_features >= 3 else (32, 32)
                model = MLPRegressor(
                    hidden_layer_sizes=hidden,
                    max_iter=2000,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_pred)
                std = None
                feature_importance = None
            else:  # RandomForest
                from sklearn.ensemble import RandomForestRegressor

                model = RandomForestRegressor(
                    n_estimators=100, random_state=42
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_pred)
                std = None
                feature_importance = model.feature_importances_.tolist()

            # 训练集 R²
            train_r2 = float(model.score(X_train, y_train))

        except ImportError as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"sklearn 不可用: {e}. 请装 scikit-learn",
            )

        result: dict[str, Any] = {
            "action": "ml_surrogate",
            "model_type": model_type,
            "target_property": args.target_property,
            "n_train": int(X_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "n_pred": int(X_pred.shape[0]),
            "prediction": pred.tolist(),
            "train_r2": train_r2,
        }
        if std is not None:
            result["uncertainty_std"] = std.tolist()
        if feature_importance is not None:
            result["feature_importance"] = feature_importance

        result["notes"] = (
            f"sklearn {model_type} 代理模型, 拟合 '{args.target_property}'. "
            "ai4plasma 的 PINN/DeepONet 路线更强但需 PyTorch, 这里走轻量路线. "
            "GP 给不确定性, RF 给特征重要度, NN 适合大样本."
        )

        return ToolResult(data=result, success=True)

    # ============================================================ arc plasma
    def _arc_plasma(self, args: PlasmaToolInput) -> ToolResult:
        """1D 弧等离子体 (Elenbaas-Heller 简化).

        参考 ai4plasma/plasma/arc.py 的弧模型思路. 稳态 1D 能量平衡:
          σ(T)*E² = -d/dx(κ(T)*dT/dx) + P_rad(T)
        这里简化: 忽略辐射, 用拟稳态迭代解 1D 平板热平衡:
          κ(T)*d²T/dx² + σ(T)*E² = 0
        电导率 σ(T) 和热导率 κ(T) 用简化模型 (类氩弧, Saha 弱电离近似):
          σ ∝ T^1.5, κ ∝ T^0.5 (量级估算用)
        电流密度 J = I/(πR²), E = J/σ(T_center) 自洽迭代.
        """
        I = args.arc_current
        R = args.arc_radius
        N = args.grid_size
        dx = 2.0 * R / (N - 1)
        x = np.linspace(-R, R, N)
        Tw = args.wall_temperature

        J = I / (np.pi * R**2)

        # 初始温度剖面 (抛物线, 中心高, 壁低)
        T = Tw + 1e4 * (1.0 - (x / R) ** 2)
        T = np.maximum(T, Tw)

        # 迭代解非线性扩散方程
        n_iter = 50
        # 参考点电导率 (10000K 氩弧约 ~1000 S/m, 这里归一化)
        sigma_ref = 1000.0  # S/m at T=10000K
        kappa_ref = 0.5  # W/(m·K) at T=10000K
        T_ref = 10000.0

        for _ in range(n_iter):
            sigma = sigma_ref * (T / T_ref) ** 1.5
            kappa = kappa_ref * (T / T_ref) ** 0.5
            # 自洽电场 E = J/σ_mean
            sigma_mean = float(np.mean(sigma))
            E_field = J / max(sigma_mean, 1e-6)
            # 源项 σE²
            src = sigma * E_field**2
            # 隐式松驰更新 T: κ d²T/dx² + src = 0
            T_new = T.copy()
            omega = 0.3  # 松弛因子, 防发散
            for i in range(1, N - 1):
                lap = (T[i + 1] - 2 * T[i] + T[i - 1]) / dx**2
                # dT/dt = (κ*lap + src) / (rho*cp), 稳态 = 0
                # 用松弛: T_new = T + ω * (κ*lap + src)/(κ/dx²) * dx²
                # 简化为 T_new[i] = T[i] + ω*( (T[i+1]+T[i-1])/2 + src*dx²/(2κ) - T[i] )
                rhs = 0.5 * (T[i + 1] + T[i - 1]) + src[i] * dx**2 / (2.0 * kappa[i])
                T_new[i] = (1.0 - omega) * T[i] + omega * rhs
            # 边界 T = Tw
            T_new[0] = Tw
            T_new[-1] = Tw
            T = np.maximum(T_new, Tw)

        sigma = sigma_ref * (T / T_ref) ** 1.5
        kappa = kappa_ref * (T / T_ref) ** 0.5
        sigma_mean = float(np.mean(sigma))
        E_final = J / max(sigma_mean, 1e-6)
        T_center = float(np.max(T))

        # 估算功率耗散 (单位长度弧柱)
        power_per_length = float(np.mean(sigma * E_final**2) * 2 * R)  # W/m

        return ToolResult(
            data={
                "action": "arc_plasma",
                "config": {
                    "arc_current_A": I,
                    "arc_radius_m": R,
                    "grid_size": N,
                    "wall_temperature_K": Tw,
                    "iterations": n_iter,
                },
                "current_density": float(J),
                "electric_field": float(E_final),
                "T_profile_K": T.tolist(),
                "sigma_profile": sigma.tolist(),
                "kappa_profile": kappa.tolist(),
                "T_center_K": T_center,
                "power_dissipation_per_length": power_per_length,
                "grid_x": x.tolist(),
                "notes": (
                    "1D Elenbaas-Heller 简化, 忽略辐射, σ/κ 用幂律模型. "
                    "参考 ai4plasma/plasma/arc.py 的弧模型思路. "
                    "真实弧仿真需解辐射 + Saha 电离 + 净发射系数, 用 ai4plasma 的 CS-PINN 更靠谱."
                ),
            },
            success=True,
        )
