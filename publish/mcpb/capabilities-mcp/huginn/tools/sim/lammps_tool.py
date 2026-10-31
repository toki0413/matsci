"""LAMMPS molecular dynamics tool — real execution via subprocess.

Uses the installed lmp.exe for actual MD simulations.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from huginn.core_types import HandleType, ToolContext, ToolResult, ValidationResult
from huginn.env_access import env_path
from huginn.security import SandboxExecutor
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.tools.sim._async_job import submit_background
from huginn.validation.handle_validator import HandleValidator

logger = logging.getLogger(__name__)

try:
    import huginn_ext

    _HAS_HUGINN_EXT = True
except ImportError:
    logger.debug("best-effort op failed", exc_info=True)
    huginn_ext = None
    _HAS_HUGINN_EXT = False

# submit_async 实际可以跑的 LAMMPS 计算类型
_LAMMPS_COMPUTE_ACTIONS = ("run", "minimize", "equilibrate")


class LammpsToolInput(BaseModel):
    action: Literal[
        "run",
        "minimize",
        "equilibrate",
        "analyze_trajectory",
        "submit_async",
        "poll_job",
        "wait_job",
        "equilibrium_check",
        "dem_packing",
    ] = Field(...)
    input_script: str = Field(
        default="", description="LAMMPS input script content or file path"
    )
    structure_file: str | None = Field(
        default=None, description="Structure file path (data, xyz, etc.)"
    )
    potentials: list[str] = Field(
        default_factory=list, description="List of potential file paths"
    )
    trajectory_file: str | None = Field(
        default=None, description="Trajectory file to analyze (for analyze_trajectory)"
    )
    output_prefix: str = Field(default="lammps_out")
    num_processes: int = Field(default=1, ge=1)
    working_dir: str | None = Field(default=None)
    fixes: dict[str, str] = Field(
        default_factory=dict,
        description="Auto-applied fixes from diagnosis (e.g., {'timestep': '0.5'})",
    )
    # submit_async 专用: 指定实际跑哪种计算 (run/minimize/equilibrate)
    compute_action: Literal["run", "minimize", "equilibrate"] | None = Field(
        default=None,
        description="For submit_async: which computation to run (run/minimize/equilibrate)",
    )
    # poll_job / wait_job 专用
    job_id: str | None = Field(
        default=None,
        description="For poll_job/wait_job: the job_id returned by submit_async",
    )
    # wait_job 专用: 最长等多久 (秒)
    timeout: float = Field(
        default=3600.0,
        ge=1.0,
        description="For wait_job: max seconds to wait before returning (default 3600)",
    )
    # 计算失败 / 物理审计报错时自动诊断 + 改脚本重试的次数. 0 = 关闭自愈.
    max_auto_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="On failure or physics-audit error, auto-diagnose + patch script and retry up to N times",
    )
    # equilibrium_check 专用
    log_file_path: str | None = Field(
        default=None,
        description="For equilibrium_check: path to log.lammps (defaults to working_dir/log.lammps)",
    )
    target_temp: float | None = Field(
        default=None,
        description="For equilibrium_check: target temperature in K",
    )
    target_pressure: float | None = Field(
        default=None,
        description="For equilibrium_check: target pressure in bar",
    )
    window: float = Field(
        default=30.0,
        ge=1.0,
        le=100.0,
        description="For equilibrium_check: percentage of trailing steps to use (default 30%)",
    )
    # dem_packing 专用: 离散元颗粒碰撞模拟
    dem_box: list[float] = Field(
        default=[100.0, 100.0, 100.0],
        description="DEM simulation box dimensions in LAMMPS units (Å for real, µm for si)",
    )
    dem_n_particles: int = Field(
        default=1000, ge=1, le=100000,
        description="Number of particles for DEM packing",
    )
    dem_radius: float = Field(
        default=5.0, gt=0,
        description="Particle radius (in LAMMPS length unit)",
    )
    dem_radius_std: float = Field(
        default=0.0, ge=0.0,
        description="Particle radius standard deviation (for polydisperse packing)",
    )
    dem_density: float = Field(
        default=1.0, gt=0,
        description="Particle density (mass/volume in LAMMPS units)",
    )
    dem_youngs: float = Field(
        default=1e6, gt=0,
        description="Young's modulus for Hertzian contact (Pa for si, energy/length³ for real)",
    )
    dem_poisson: float = Field(
        default=0.3, ge=0.0, lt=0.5,
        description="Poisson's ratio for contact model",
    )
    dem_friction: float = Field(
        default=0.5, ge=0.0, le=2.0,
        description="Sliding friction coefficient",
    )
    dem_restitution: float = Field(
        default=0.8, gt=0.0, le=1.0,
        description="Restitution coefficient (bounciness, 1=perfectly elastic)",
    )
    dem_n_steps: int = Field(
        default=100000, ge=100,
        description="Number of DEM simulation steps",
    )
    dem_gravity: float = Field(
        default=0.0,
        description="Gravity acceleration (m/s² for si, or 0 for no gravity)",
    )

    @model_validator(mode="after")
    def _check_action_fields(self) -> LammpsToolInput:
        """不同 action 需要不同字段, schema 层兜底."""
        if self.action == "submit_async":
            if not self.compute_action:
                raise ValueError(
                    "submit_async requires 'compute_action' (run/minimize/equilibrate)"
                )
            if not self.input_script:
                raise ValueError(
                    "submit_async requires 'input_script' (script content or file path)"
                )
        if self.action in ("poll_job", "wait_job") and not self.job_id:
            raise ValueError(f"action '{self.action}' requires 'job_id'")
        return self


class LammpsToolOutput(BaseModel):
    log_path: str | None = None
    trajectory_path: str | None = None
    thermo_data: dict | None = None
    final_energy: float | None = None
    warnings: list[str] = []


# ── MSD (mean squared displacement) helpers ──────────────────────────


def _msd_coord_mode(atom0: dict) -> str | None:
    """判定轨迹坐标系来源: xu/yu/zu > ix/iy/iz > 包裹坐标 x/y/z."""
    if all(k in atom0 for k in ("xu", "yu", "zu")):
        return "unwrapped"
    if all(k in atom0 for k in ("ix", "iy", "iz")):
        return "image"
    if all(k in atom0 for k in ("x", "y", "z")):
        return "wrapped"
    return None


def _msd_box_lengths(frame: dict) -> tuple[float, float, float]:
    """取首帧正交盒长; 缺 box 字段时退化为单位盒."""
    box = frame.get("box", [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    return (
        box[0][1] - box[0][0],
        box[1][1] - box[1][0],
        box[2][1] - box[2][0],
    )


def _msd_ref_positions(
    atoms: list[dict], mode: str, lengths: tuple[float, float, float]
) -> list[tuple[float, float, float]]:
    """把首帧原子坐标统一成"未包裹"参考位置."""
    lx, ly, lz = lengths
    ref_pos: list[tuple[float, float, float]] = []
    for atom in atoms:
        if mode == "unwrapped":
            ref_pos.append((atom["xu"], atom["yu"], atom["zu"]))
        elif mode == "image":
            ref_pos.append(
                (
                    atom["x"] + atom["ix"] * lx,
                    atom["y"] + atom["iy"] * ly,
                    atom["z"] + atom["iz"] * lz,
                )
            )
        else:
            ref_pos.append((atom["x"], atom["y"], atom["z"]))
    return ref_pos


def _msd_wrapped_position(
    atom: dict,
    i: int,
    lengths: tuple[float, float, float],
    running_pos: list[list[float]],
    prev_wrapped: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], float]:
    """单原子逐帧 MIC 增量解包; 返回 (未包裹位置, pre-MIC 位移/箱长)."""
    lx, ly, lz = lengths
    px, py, pz = prev_wrapped[i]
    dx = atom["x"] - px
    dy = atom["y"] - py
    dz = atom["z"] - pz
    pre_mic = max(abs(dx) / lx, abs(dy) / ly, abs(dz) / lz)
    dx -= lx * round(dx / lx)
    dy -= ly * round(dy / ly)
    dz -= lz * round(dz / lz)
    # 累积成未包裹位置
    cx = running_pos[i][0] + dx
    cy = running_pos[i][1] + dy
    cz = running_pos[i][2] + dz
    running_pos[i] = [cx, cy, cz]
    prev_wrapped[i] = (atom["x"], atom["y"], atom["z"])
    return (cx, cy, cz), pre_mic


def _msd_frame(
    atoms: list[dict],
    ref_pos: list[tuple[float, float, float]],
    mode: str,
    lengths: tuple[float, float, float],
    running_pos: list[list[float]] | None,
    prev_wrapped: list[tuple[float, float, float]] | None,
) -> tuple[float, float]:
    """单帧 MSD 与 pre-MIC 最大位移 (归一化到箱长).

    wrapped-only 模式下就地更新 running_pos / prev_wrapped, 第二返回值供上层
    判断 MIC 是否歧义.
    """
    lx, ly, lz = lengths
    n_atoms = len(ref_pos)
    sum_sq = 0.0
    max_pre_mic = 0.0
    for i in range(n_atoms):
        atom = atoms[i]
        rx, ry, rz = ref_pos[i]
        if mode == "unwrapped":
            cx, cy, cz = atom["xu"], atom["yu"], atom["zu"]
        elif mode == "image":
            cx = atom["x"] + atom["ix"] * lx
            cy = atom["y"] + atom["iy"] * ly
            cz = atom["z"] + atom["iz"] * lz
        else:
            # wrapped-only: 用上一帧的包裹坐标算逐帧位移再做 MIC 折叠.
            # pre-MIC 位移归一化后若超 L/2, MIC 有歧义 (无法区分 "+0.51L"
            # 与 "-0.49L"), 可能漏掉一次 wrap, 上抛给调用方告警.
            (cx, cy, cz), pre_mic = _msd_wrapped_position(
                atom, i, lengths, running_pos, prev_wrapped  # type: ignore[arg-type]
            )
            max_pre_mic = max(max_pre_mic, pre_mic)
        dx = cx - rx
        dy = cy - ry
        dz = cz - rz
        sum_sq += dx * dx + dy * dy + dz * dz
    return sum_sq / n_atoms, max_pre_mic


_MSD_WRAPPED_ONLY_WARNING = (
    "Trajectory has only wrapped coords (no xu/yu/zu or ix/iy/iz). "
    "MSD computed via frame-by-frame MIC incremental unwrap. If a "
    "particle moved > L/2 between consecutive frames, the unwrap "
    "will miss wrapping events and MSD will be underestimated. "
    "For reliable long-time diffusion coefficients, re-dump with "
    "xu yu zu columns."
)


def _msd_saturation_warning(fi: int, max_pre_mic: float) -> str:
    """pre-MIC 位移超过 L/2 时的歧义告警."""
    return (
        f"Frame {fi}: per-frame pre-MIC displacement reached "
        f"{max_pre_mic:.3f}×L (>0.5×L). Incremental MIC unwrap is "
        f"ambiguous at this scale — a wrap event may have been "
        f"missed and MSD underestimated for this and later frames. "
        f"Diffusion coefficient D derived from this MSD is NOT reliable."
    )


def _msd_series(
    frames: list[dict],
    ref_pos: list[tuple[float, float, float]],
    mode: str,
    lengths: tuple[float, float, float],
    running_pos: list[list[float]] | None,
    prev_wrapped: list[tuple[float, float, float]] | None,
) -> tuple[list[dict], tuple[int, float] | None]:
    """逐帧算 MSD; 返回 (msd_data, 首个饱和告警的 (frame_idx, max_pre_mic)).

    Saturation guard: wrapped-only 模式下若某帧 pre-MIC 位移超 L/2, 说明 MIC
    有歧义 (可能漏掉 wrap), 之后各帧 MSD 与扩散系数都不可信. 只报第一次.
    """
    msd_data: list[dict] = []
    first_saturation: tuple[int, float] | None = None
    for fi, frame in enumerate(frames[1:], start=1):
        msd, max_pre_mic = _msd_frame(
            frame["atoms"], ref_pos, mode, lengths, running_pos, prev_wrapped
        )
        msd_data.append({"timestep": frame.get("timestep", fi), "msd": msd})
        if mode == "wrapped" and first_saturation is None and max_pre_mic > 0.5:
            first_saturation = (fi, max_pre_mic)
    return msd_data, first_saturation


# ── correlation-function helpers (van Hove / F(q,t)) ────────────────────

def _traj_positions(frame: dict) -> Any:
    """取一帧全部原子坐标 (N, 3): xu/yu/zu 优先, 否则包裹 x/y/z."""
    import numpy as np

    atoms = frame["atoms"]
    if atoms and "xu" in atoms[0]:
        return np.array(
            [[a["xu"], a["yu"], a["zu"]] for a in atoms], dtype=np.float64
        )
    return np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=np.float64)


def _vh_series(
    frames: list[dict],
    r0: Any,
    top_idx: Any,
    r_edges: Any,
    r_centers: list[float],
    k: int,
) -> list[dict]:
    """逐帧统计 top-K 原子的位移分布 G_s(r, t)."""
    import numpy as np

    out: list[dict] = []
    for fi, frame in enumerate(frames):
        disp = _traj_positions(frame)[top_idx] - r0[top_idx]  # (K, 3)
        disp_mag = np.sqrt((disp**2).sum(axis=1))  # (K,)
        hist, _ = np.histogram(disp_mag, bins=r_edges)
        out.append(
            {
                "frame_index": fi,
                "timestep": frame.get("timestep", fi),
                "r": r_centers,
                "G_s": (hist / k).tolist(),  # 归一化为概率密度
            }
        )
    return out


def _fqt_series(
    frames: list[dict], r0: Any, q_values: list[float]
) -> list[dict]:
    """逐帧算 F(q, t) = <cos(q·Δr)> 的三方向均值."""
    import numpy as np

    out: list[dict] = []
    for fi, frame in enumerate(frames):
        delta = _traj_positions(frame) - r0  # (N, 3)
        f_entry: dict[str, Any] = {
            "frame_index": fi,
            "timestep": frame.get("timestep", fi),
            "q_values": list(q_values),
            "F": [],  # 与 q_values 对齐
        }
        for q in q_values:
            f_real = np.cos(q * delta).mean()  # 沿每列取均值再合并
            if hasattr(f_real, "shape") and f_real.shape == (3,):
                f_entry["F"].append(float(f_real.mean()))
            else:
                f_entry["F"].append(float(f_real))
        out.append(f_entry)
    return out


def _find_trajectory(work_dir: Path) -> str | None:
    """work_dir 里第一个轨迹文件 (.lammpstrj / .dump / .xyz)."""
    for ext in (".lammpstrj", ".dump", ".xyz"):
        candidates = sorted(work_dir.glob(f"*{ext}"))
        if candidates:
            return str(candidates[0])
    return None


def _run_outcome(result: Any, soft_failure_msg: str | None) -> tuple[bool, str | None]:
    """成功判定: returncode==0 且没有遗留软失败 (物理审计报错)."""
    if result.returncode == 0 and soft_failure_msg is None:
        return True, None
    if result.returncode != 0:
        return False, result.stderr[:500]
    return False, soft_failure_msg


def _attach_provenance(args: LammpsToolInput, data: dict) -> None:
    """给结果附加 provenance 快照; 溯源失败不阻断结果返回."""
    try:
        from huginn.provenance import capture

        data["provenance"] = capture(
            "lammps_tool", args.model_dump(), output=dict(data)
        ).to_dict()
    except Exception:  # 防御: 溯源失败不阻断结果返回
        logger.debug("provenance 失败不能把计算结果带挂", exc_info=True)


def _new_run_outcome() -> dict[str, Any]:
    """单次 run 的累积结果骨架 (跨 autofix 重试复用)."""
    return {
        "result": None,
        "thermo_data": {},
        "final_energy": None,
        "warnings": [],
        "audit_report": None,
        "soft_failure_msg": None,
        "autoheal_log": [],
    }


def _record_autofix(
    outcome: dict[str, Any], attempt: int, error: str, fixed: dict
) -> None:
    """把一次自动修复记进 autoheal_log, 供结果里回放."""
    outcome["autoheal_log"].append(
        {
            "attempt": attempt + 1,
            "error": error[:300],
            "fixes_applied": fixed["fixes"],
            "reasoning": fixed["reasoning"],
        }
    )


def _validate_path(
    context: ToolContext, path: str, label: str, error_code: int = 404
) -> ValidationResult | None:
    """文件路径门禁; 不通过返回 ValidationResult, 通过返回 None."""
    vr = HandleValidator.validate(HandleType.FILE_PATH, path, context)
    if vr.result:
        return None
    return ValidationResult(
        result=False, message=f"{label}: {path}", error_code=error_code
    )


def _dump_handle_timestep(
    lines: list[str], i: int, frame: dict[str, Any], result: dict[str, Any]
) -> int:
    """处理 'ITEM: TIMESTEP': 读下一行步数; 返回消费后的行号."""
    i += 1
    if i < len(lines):
        frame["timestep"] = int(lines[i].strip())
        result["timesteps"].append(frame["timestep"])
    return i


def _dump_handle_n_atoms(
    lines: list[str], i: int, frame: dict[str, Any], result: dict[str, Any]
) -> int:
    """处理 'ITEM: NUMBER OF ATOMS': 录原子数; 返回消费后的行号."""
    i += 1
    if i < len(lines):
        if frame is None:
            raise ValueError("dump truncated before ITEM: TIMESTEP")
        frame["n_atoms"] = int(lines[i].strip())
        result["n_atoms"] = frame["n_atoms"]
    return i


def _dump_handle_box(
    lines: list[str], i: int, frame: dict[str, Any] | None, result: dict[str, Any]
) -> int:
    """处理 'ITEM: BOX BOUNDS': 读三行边界; 返回消费后的行号."""
    bounds: list[list[float]] = []
    for _ in range(3):
        i += 1
        if i < len(lines):
            bounds.append([float(x) for x in lines[i].strip().split()])
    if frame is not None:
        frame["box"] = bounds
    if not result["box_bounds"]:
        result["box_bounds"] = bounds
    return i


def _velocities(frame: dict) -> Any:
    """取一帧全部原子速度 (N, 3)."""
    import numpy as np

    return np.array(
        [[a["vx"], a["vy"], a["vz"]] for a in frame["atoms"]], dtype=np.float64
    )


def _vacf_series(frames: list[dict], v0: Any) -> list[dict]:
    """逐帧算 <v(0)·v(t)>, 并按 C(0) 标准化便于跨体系比较."""
    out: list[dict] = []
    for fi, frame in enumerate(frames):
        dot = (v0 * _velocities(frame)).sum(axis=1)  # (n_atoms,)
        out.append(
            {
                "frame_index": fi,
                "timestep": frame.get("timestep", fi),
                "vacf": float(dot.mean()),
            }
        )
    c0 = out[0]["vacf"]
    for d in out:
        d["vacf_normalized"] = d["vacf"] / c0 if abs(c0) > 1e-12 else 0.0
    return out


def _dem_alpha(e: float) -> float:
    """Tsuji 阻尼系数 α = −ln(e)/sqrt(π²+ln²(e)); e→1 无阻尼, e→0 全塑性."""
    if e >= 1.0:
        return 0.0
    if e <= 0.0:
        return 1e9  # 完全塑性, 极大阻尼
    return -math.log(e) / math.sqrt(math.pi**2 + math.log(e) ** 2)


def _dem_polydispersion(r: float, r_std: float) -> str:
    """多分散粒径的 set 语句块; r_std<=0 时退化为单分散."""
    if r_std <= 0:
        return "# 单分散粒径"
    return (
        f"# 多分散粒径 r~N({r}, {r_std})\n"
        f"variable       r_var normal {r} {r_std}\n"
        f"set             type 1 diameter v_r_var"
    )


def _dem_params(args: LammpsToolInput) -> dict[str, Any]:
    """DEM 派生量: 质量 / 剪切模量 / Rayleigh 时间步 / Tsuji 阻尼 / 多分散块."""
    r = args.dem_radius
    rho = args.dem_density
    E = args.dem_youngs
    nu = args.dem_poisson

    # 颗粒质量: m = ρ * (4/3)πr³
    mass = rho * (4.0 / 3.0) * 3.14159265358979 * r**3

    # 剪切模量 G = E / (2(1+v)), Mindlin 切向模型用
    G = E / (2 * (1 + nu))

    # Rayleigh 接触时间 t_R = π·r·sqrt(ρ/G) / (0.1631ν + 0.8766): 表面波在颗粒上
    # 转一圈的时间, 是 DEM 时间步上限. 取 10% t_R 是业界标准稳定裕度
    # (Li et al., Pow.Tech. 2005); 旧实现硬编码 1e-6 量级偏小 ~10³.
    rayleigh_t = math.pi * r * math.sqrt(rho / G) / (0.1631 * nu + 0.8766)

    return {
        "bx": args.dem_box[0],
        "by": args.dem_box[1],
        "bz": args.dem_box[2],
        "r": r,
        "r_std": args.dem_radius_std,
        "rho": rho,
        "E": E,
        "nu": nu,
        "mu": args.dem_friction,
        "e": args.dem_restitution,
        "n_steps": args.dem_n_steps,
        "g": args.dem_gravity,
        "n_particles": args.dem_n_particles,
        "diameter": 2 * r,
        "mass": mass,
        "G": G,
        "rayleigh_t": rayleigh_t,
        "dt": 0.1 * rayleigh_t,
        "alpha": _dem_alpha(args.dem_restitution),
        "polydispersion_block": _dem_polydispersion(r, args.dem_radius_std),
    }


def _dem_header(p: dict[str, Any]) -> str:
    """DEM 脚本前半段: 单位/盒子/建粒子/邻域/接触模型."""
    return f"""# LAMMPS DEM (Discrete Element Method) — Granular Packing Simulation
# 接触模型: Hertz-Mindlin with Tsuji damping (restitution e={p["e"]})
# 生成方式: lammps_tool action=dem_packing
# 粒子数: {p["n_particles"]}, 粒径: {p["r"]} ± {p["r_std"]}
# 时间步 dt = 0.1 * t_R = {p["dt"]:.6e} s  (t_R = πr·sqrt(ρ/G)/(0.1631ν+0.8766) = {p["rayleigh_t"]:.6e} s)

# ── Units & Atom Style ──────────────────────────────────────────
# si: meters/seconds/kg; real: Angstroms/fs/g (需按比例换算)
units           si
atom_style      sphere
boundary        f f f

# ── Simulation Box ──────────────────────────────────────────────
region          box block 0 {p["bx"]} 0 {p["by"]} 0 {p["bz"]}
create_box      1 box

# ── Create Particles ────────────────────────────────────────────
# 必须先 create_atoms 再 set diameter/density:
# LAMMPS set 只作用于已存在原子, create_box 后无原子时 set 静默无效.
create_atoms    1 random {p["n_particles"]} 12345 box \\
                overlap {p["diameter"]} maxtry 10000

# ── Particle Properties ─────────────────────────────────────────
# 每个粒子: position + diameter + density (LAMMPS 内部按 sphere 算 mass)
set             type 1 diameter {p["diameter"]} density {p["rho"]}
{p["polydispersion_block"]}

# ── Neighbor & Communication ───────────────────────────────────
neighbor        {p["diameter"]} bin
neigh_modify    delay 0

# ── Pair Style: Hertz-Mindlin Granular Contact ──────────────────
# LAMMPS GRANULAR package, 官方语法 (https://docs.lammps.org/pair_granular.html):
#   pair_coeff * * hertz/material E v gamma_n tangential mindlin_rescale G_t mu_t x_t damping tsuji gamma_n
# 注意: tangential 关键字必须前置 (旧脚本把 normal/tangential/rolling/twisting
# 当 pair_coeff 末尾位置参数, 是非法语法). damping_coeff 是臆造关键字,
# 正确是 "damping tsuji gamma_n".
pair_style      granular
pair_coeff      * * hertz/material {p["E"]:.6e} {p["nu"]} {p["alpha"]:.6e} \\
                tangential mindlin_rescale {p["G"]:.6e} {p["mu"]} 1.0 \\
                damping tsuji {p["alpha"]:.6e}

"""


def _dem_physics(p: dict[str, Any]) -> str:
    """DEM 脚本中段: 重力/积分器/热力学与 dump 输出."""
    gravity_line = (
        f"fix            gravity all gravity {p['g']} vector 0 0 -1"
        if p["g"] > 0
        else "# no gravity"
    )
    return f"""# ── Physics: Gravity (optional) ─────────────────────────────────
{gravity_line}

# ── Integration: NVE + Granular Temperature ────────────────────
fix             integrate all nve/sphere
fix             freeze_property all setforce 0 0 0

# ── Output: Thermo + Dump ──────────────────────────────────────
thermo          1000
thermo_style    custom step atoms ke pe etotal press
thermo_modify   lost warn

dump            particles all custom 10000 dump.particles id type x y z vx vy vz \\
                radius mass
dump_modify     particles sort id

"""


def _dem_run(p: dict[str, Any]) -> str:
    """DEM 脚本尾段: 最小化 + 主循环 + 配位数/填充率统计."""
    return f"""# ── Run ─────────────────────────────────────────────────────────
# 初始能量最小化 (消除 create_atoms overlap 残留)
minimize        1e-6 1e-8 1000 10000

# 释放 freeze, 跑 DEM. dt 取 10% Rayleigh 接触时间保证稳定.
unfix           freeze_property
timestep        {p["dt"]:.6e}
run             {p["n_steps"]}

# ── Post: Compute Coordination Number & Packing Fraction ───────
compute         cn all contact/atom
compute         cn_avg all reduce ave c_cn
variable        phi equal count(all) * {p["mass"]:.6e} / (vol * {p["rho"]})
variable        mean_cn equal c_cn_avg

print           "Packing fraction (phi): ${{phi}}"
print           "Mean coordination number: ${{mean_cn}}"

write_restart   restart.dem
"""


def _resolve_log_path(args: LammpsToolInput) -> Path | None:
    """按 log_file_path / working_dir/log.lammps 定位日志."""
    if args.log_file_path:
        return Path(args.log_file_path)
    if args.working_dir:
        return Path(args.working_dir) / "log.lammps"
    return None


def _missing_log_result() -> ToolResult:
    """找不到 LAMMPS 日志时的统一返回."""
    return ToolResult(
        data=None,
        success=False,
        error=(
            "No log file found. Provide log_file_path or working_dir "
            "with log.lammps"
        ),
    )


def _equilibrium_empty_result(reason: str) -> ToolResult:
    """没有可用 thermo 数据时的统一返回 (未平衡 + 原因)."""
    return ToolResult(
        data={
            "equilibrated": False,
            "avg_temp": None,
            "avg_pressure": None,
            "temp_drift": None,
            "pressure_drift": None,
            "recommendation": reason,
        },
        success=True,
    )


def _equilibrium_tail(
    thermo: dict, window: float
) -> tuple[list, list, list, int, int]:
    """取时序尾部 window% 数据; 返回 (temps, press, steps, n_tail, n_total)."""
    temps = thermo.get("temp", [])
    press = thermo.get("press", [])
    steps = thermo.get("step", [])
    n_total = len(temps)
    n_tail = max(1, int(n_total * window / 100.0))
    tail_press = press[-n_tail:] if press else []
    tail_steps = steps[-n_tail:] if steps else list(range(n_tail))
    return temps[-n_tail:], tail_press, tail_steps, n_tail, n_total


def _dem_work_dir(args: LammpsToolInput, context: ToolContext) -> Path:
    """DEM 工作目录: 显式 working_dir 优先, 否则 workspace 下按输出前缀建."""
    if args.working_dir:
        return Path(args.working_dir)
    return Path(context.workspace) / f"lammps_dem_{args.output_prefix}"


def _dem_payload(args: LammpsToolInput, script_path: Path, script: str) -> dict:
    """DEM 结果载荷骨架 (执行后原地补充 returncode/status 等字段)."""
    return {
        "action": "dem_packing",
        "script_path": str(script_path),
        "script_content": script,
        "n_particles": args.dem_n_particles,
        "particle_radius": args.dem_radius,
        "box_size": args.dem_box,
        "contact_model": "hertz/material",
        "n_steps": args.dem_n_steps,
        "friction_coeff": args.dem_friction,
        "restitution": args.dem_restitution,
    }


class LammpsTool(HuginnTool):
    """Execute LAMMPS molecular dynamics simulations."""

    name = "lammps_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        constraint_scope="md",
        light_alternatives=("symbolic_math_tool", "numerical_tool"),
    )
    description = (
        "Run LAMMPS molecular dynamics simulations (minimization, equilibration, production). "
        "Supports async submission via submit_async / poll_job / wait_job for long-running jobs."
    )
    input_schema = LammpsToolInput
    _init_kwargs_map = {"lammps_executable": "lammps_executable"}

    # 异步作业注册表: job_id -> {status, task, result, error, started_at, finished_at}
    # 类级别共享, 跟 VaspTool 一样的模式. 进程重启后状态丢失.
    _async_jobs: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        lammps_executable: str | None = None,
        sandbox: SandboxExecutor | None = None,
    ):
        super().__init__()
        self.lammps_executable = lammps_executable or self._find_lammps()
        self.sandbox = sandbox or SandboxExecutor()

    def _find_lammps(self) -> str | None:
        """Find LAMMPS executable on the system."""
        import glob

        # Check environment variable
        exe_from_env = env_path("LAMMPS_EXECUTABLE")
        if exe_from_env and exe_from_env.exists():
            return str(exe_from_env)

        # Check PATH
        try:
            import shutil

            exe = shutil.which("lmp")
            if exe:
                return exe
        except Exception:  # 防御: PATH 探测失败继续下一步查找
            logger.debug("suppressed in _find_lammps", exc_info=True)

        # Check common Windows locations (with glob for unicode paths)
        patterns = [
            r"C:\Users\*\OneDrive\*\LAMMPS*\bin\lmp.exe",
            r"C:\Program Files*\LAMMPS*\bin\lmp.exe",
            r"C:\ProgramData\*\LAMMPS*\bin\lmp.exe",
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            for m in matches:
                if Path(m).exists():
                    return m

        return None

    def estimate_cost(self, args: LammpsToolInput) -> dict[str, float] | None:
        # poll_job / wait_job 是查询操作, 不消耗计算资源
        if args.action in ("poll_job", "wait_job"):
            return None
        return {"cpu_hours": 2, "walltime_hours": 2}

    async def validate_input(
        self, args: LammpsToolInput, context: ToolContext
    ) -> ValidationResult:
        """Pre-flight: verify required files based on action type.

        poll_job / wait_job 只需要 job_id, 不检查文件. submit_async 跟
        普通计算 action 一样检查 input_script / structure_file / potentials.
        """
        # poll_job / wait_job 不需要文件检查
        if args.action in ("poll_job", "wait_job"):
            return ValidationResult(result=True)

        if args.action == "analyze_trajectory":
            traj = args.trajectory_file or args.input_script
            if not traj:
                return ValidationResult(
                    result=False,
                    message="Trajectory file not specified",
                    error_code=400,
                )
            bad = _validate_path(context, traj, "Trajectory file not found")
            if bad is not None:
                return bad

        checks = [(args.structure_file, "Structure file not found", 404)]
        if args.input_script and Path(args.input_script).suffix in (
            ".lammps",
            ".in",
            ".lmp",
        ):
            checks.append((args.input_script, "Input script file not found", 404))
        checks.extend(
            (pot, "Potential file not found", 404) for pot in args.potentials
        )
        for path, label, code in checks:
            if not path:
                continue
            bad = _validate_path(context, path, label, code)
            if bad is not None:
                return bad
        return ValidationResult(result=True)

    async def call(self, args: LammpsToolInput, context: ToolContext) -> ToolResult:
        handled = await self._dispatch_aux_action(args, context)
        if handled is not None:
            return handled

        exe_error = self._ensure_executable()
        if exe_error is not None:
            return exe_error

        work_dir, input_path = self._prepare_run_inputs(args, context)
        work_dir_abs = work_dir.resolve()
        input_path_abs = input_path.resolve()
        log_path_abs = (work_dir_abs / "log.lammps").resolve()
        cmd = self._build_command(input_path_abs, log_path_abs, args)

        try:
            outcome = self._run_with_autofix(
                cmd, work_dir_abs, work_dir, input_path_abs, args
            )
            return self._build_run_result(outcome, args, work_dir)
        except subprocess.TimeoutExpired:
            return ToolResult(
                data=None, success=False, error="LAMMPS execution timed out (3600s)"
            )
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"LAMMPS execution failed: {e}"
            )

    async def _dispatch_aux_action(
        self, args: LammpsToolInput, context: ToolContext
    ) -> ToolResult | None:
        """Actions that never spawn a LAMMPS run (job API / analysis / checks)."""
        # 异步作业管理动作: 不跑实际计算, 只查/等作业状态
        if args.action == "submit_async":
            return await self._handle_submit_async(args, context)
        if args.action == "poll_job":
            return self._handle_poll_job(args)
        if args.action == "wait_job":
            return await self._handle_wait_job(args)
        # Equilibrium check: analyze thermo data from a log file, no LAMMPS run
        if args.action == "equilibrium_check":
            return self._run_equilibrium_check(args)
        # DEM packing: generate granular input script, then optionally run
        if args.action == "dem_packing":
            return await self._handle_dem_packing(args, context)
        if args.action == "analyze_trajectory":
            return self._analyze_trajectory_only(args)
        return None

    def _analyze_trajectory_only(self, args: LammpsToolInput) -> ToolResult:
        """Handle trajectory analysis without running LAMMPS."""
        traj_file = args.trajectory_file or args.input_script
        if not traj_file or not Path(traj_file).exists():
            return ToolResult(
                data=None,
                success=False,
                error="Trajectory file not specified or not found",
            )
        analysis = self.parse_trajectory(traj_file)
        # 分析也算一次计算, 带 provenance
        try:
            from huginn.provenance import capture

            analysis["provenance"] = capture(
                "lammps_tool", args.model_dump(), output=dict(analysis)
            ).to_dict()
        except Exception:  # 防御: 溯源记录失败不阻断分析结果
            logger.debug("suppressed in estimate_cost", exc_info=True)
        # 轨迹里有 MSD, 提示 agent 用 GP 量化扩散系数的不确定性
        analysis["uq_hint"] = self._uq_hint()
        return ToolResult(
            data=analysis,
            success="error" not in analysis,
            error=analysis.get("error"),
        )

    def _ensure_executable(self) -> ToolResult | None:
        """Resolve the LAMMPS binary; returns an error result when unavailable."""
        if self.lammps_executable:
            return None
        from huginn.tools.sim.executable_resolver import resolve_executable

        resolution = resolve_executable("lammps")
        if isinstance(resolution, str):
            self.lammps_executable = resolution
            return None
        return ToolResult(
            data=None,
            success=False,
            error=f"LAMMPS executable not found. {resolution.install_hint}",
            metadata={
                "needs_resolution": True,
                "resolution_request": resolution.to_dict(),
            },
        )

    def _prepare_run_inputs(
        self, args: LammpsToolInput, context: ToolContext
    ) -> tuple[Path, Path]:
        """Create the run directory, write input.lammps, and copy potentials."""
        # Determine working directory
        if args.working_dir:
            work_dir = Path(args.working_dir)
        else:
            work_dir = Path(context.workspace) / f"lammps_{args.output_prefix}"
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / "input.lammps"
        input_path.write_text(self._resolve_script_content(args), encoding="utf-8")
        self._copy_potentials(args.potentials, work_dir)
        return work_dir, input_path

    def _resolve_script_content(self, args: LammpsToolInput) -> str:
        """Read the input script from disk or inline, then prepend/auto-fix."""
        # Check if input_script is a file path or content
        script_path = Path(args.input_script)
        if script_path.exists():
            script_content = script_path.read_text(encoding="utf-8")
        else:
            script_content = args.input_script

        # Prepend structure read if structure_file provided
        if args.structure_file:
            structure_path = Path(args.structure_file)
            if structure_path.exists():
                # Detect format and prepend read command
                prefix = (
                    f"read_xyz {structure_path}\n"
                    if structure_path.suffix == ".xyz"
                    else f"read_data {structure_path}\n"
                )
                if (
                    "read_data" not in script_content
                    and "read_xyz" not in script_content
                ):
                    script_content = prefix + script_content

        # Apply auto-fixes from diagnosis to input script
        if args.fixes:
            script_content = self._apply_script_fixes(script_content, args.fixes)
        return script_content

    @staticmethod
    def _copy_potentials(potentials: list[str], work_dir: Path) -> None:
        """Copy potential files into the run directory (skip existing)."""
        for pot in potentials:
            pot_path = Path(pot)
            if not pot_path.exists():
                continue
            dest = work_dir / pot_path.name
            if dest.exists():
                continue
            import shutil

            shutil.copy2(pot_path, dest)

    def _build_command(
        self, input_path_abs: Path, log_path_abs: Path, args: LammpsToolInput
    ) -> list[str]:
        """Assemble the LAMMPS command line (with optional MPI prefix)."""
        cmd = [
            self.lammps_executable,
            "-in",
            str(input_path_abs),
            "-log",
            str(log_path_abs),
        ]
        if args.num_processes > 1:
            cmd = ["mpiexec", "-n", str(args.num_processes)] + cmd
        return cmd

    def _run_with_autofix(
        self,
        cmd: list[str],
        work_dir_abs: Path,
        work_dir: Path,
        input_path_abs: Path,
        args: LammpsToolInput,
    ) -> dict[str, Any]:
        """Run LAMMPS — 带 autofix 重试.

        硬失败 (returncode!=0) 或物理审计报错 (温度爆炸 / 能量漂移) 都重试.
        """
        outcome = _new_run_outcome()

        for attempt in range(args.max_auto_retries + 1):
            outcome["result"] = self.sandbox.run(
                cmd, cwd=str(work_dir_abs), timeout=3600
            )
            # Parse log file for thermo data
            log_path = work_dir / "log.lammps"
            (
                outcome["thermo_data"],
                outcome["final_energy"],
                outcome["warnings"],
            ) = self._parse_log(log_path)

            error, soft_msg, report = self._assess_attempt(outcome, args)
            if report is not None:
                outcome["audit_report"] = report
            outcome["soft_failure_msg"] = soft_msg

            if error is None:
                break  # 真正成功

            # 失败了 (硬失败或软失败), 看还有没有重试额度 + 能不能自动修
            fixed = (
                self._try_autofix(input_path_abs, error)
                if attempt < args.max_auto_retries
                else None
            )
            if not fixed:
                break  # 没修动或重试耗尽
            _record_autofix(outcome, attempt, error, fixed)
        return outcome

    def _assess_attempt(
        self, outcome: dict[str, Any], args: LammpsToolInput
    ) -> tuple[str | None, str | None, Any]:
        """Judge one attempt: (error, soft_failure_msg, audit_report).

        - returncode != 0 → 硬失败, 从 stderr 取诊断
        - returncode == 0 但物理审计报错 → 软失败,
          LAMMPS 经常 exit=0 但轨迹是垃圾 (温度爆炸 / 能量漂移)
        """
        result = outcome["result"]
        if result.returncode != 0:
            return result.stderr or "", None, None
        try:
            from huginn.execution.physics_auditor import PhysicsAuditor

            report = PhysicsAuditor().audit(
                "lammps_tool",
                args.compute_action or args.action,
                {
                    "thermo_data": outcome["thermo_data"],
                    "final_energy": outcome["final_energy"],
                },
                args.model_dump(),
            )
        except Exception:  # 防御: 审计异常不阻断重试流程
            logger.debug("审计本身挂了不能阻塞结果", exc_info=True)
            return None, None, None
        if not report.has_errors:
            return None, None, report
        errs = [f.message for f in report.findings if f.severity == "error"]
        error = f"Physics audit found errors: {errs}"
        return error, error, report

    def _build_run_result(
        self, outcome: dict[str, Any], args: LammpsToolInput, work_dir: Path
    ) -> ToolResult:
        """Assemble the ToolResult from the last attempt plus on-disk artifacts."""
        result = outcome["result"]
        traj_path = _find_trajectory(work_dir)

        output = LammpsToolOutput(
            log_path=str(work_dir / "log.lammps"),
            trajectory_path=traj_path,
            thermo_data=outcome["thermo_data"],
            final_energy=outcome["final_energy"],
            warnings=outcome["warnings"],
        )

        ok, error_out = _run_outcome(result, outcome["soft_failure_msg"])
        data = output.model_dump()

        # Auto-parse trajectory if available
        if traj_path:
            traj_analysis = self.parse_trajectory(traj_path)
            if "error" not in traj_analysis:
                data["trajectory_analysis"] = traj_analysis

        # Physics audit — check thermo data for unphysical values.
        # LAMMPS can exit cleanly while the trajectory itself is garbage
        # (e.g. exploded temperatures, runaway pressure). Flag those here.
        # 循环里跑过审计就复用, 没跑过 (比如硬失败) 就补跑一次兜底.
        audit_dict = self._physics_audit_dict(outcome, args)
        if audit_dict is not None:
            data["physics_audit"] = audit_dict
        if outcome["autoheal_log"]:
            data["autoheal_attempts"] = outcome["autoheal_log"]

        # 注意: 在加 provenance 字段前先 snapshot 输出, 避免 output_hash 自指
        _attach_provenance(args, data)

        # 提示 agent 用 GP 拟合 MSD-vs-time, 量化扩散系数不确定性
        data["uq_hint"] = self._uq_hint()

        return ToolResult(data=data, success=ok, error=error_out)

    @staticmethod
    def _physics_audit_dict(
        outcome: dict[str, Any], args: LammpsToolInput
    ) -> dict | None:
        """Reuse the loop's audit report; run one as a fallback when absent."""
        report = outcome["audit_report"]
        if report is not None:
            return report.to_dict()
        try:
            from huginn.execution.physics_auditor import PhysicsAuditor

            report = PhysicsAuditor().audit(
                "lammps_tool",
                args.compute_action or args.action,
                {
                    "thermo_data": outcome["thermo_data"],
                    "final_energy": outcome["final_energy"],
                },
                args.model_dump(),
            )
            return report.to_dict()
        except Exception:  # 防御: 审计失败不阻断结果返回
            logger.debug("audit is best-effort, never block the result", exc_info=True)
            return None

    # ------------------------------------------------------------------ async job API

    async def _handle_submit_async(
        self, args: LammpsToolInput, context: ToolContext
    ) -> ToolResult:
        """异步提交 LAMMPS 计算, 立即返回 job_id, 不阻塞.

        内部用 asyncio.create_task 在后台跑实际计算 (run/minimize/equilibrate),
        计算完成后把结果塞进 _async_jobs[job_id]. 进程重启后作业状态会丢,
        长跑作业建议走 job_tool 提交到 HPC.

        Returns:
            ToolResult.data = {"job_id": str, "status": "running", "compute_action": str}
        """
        # 构造同步调用的 args: 用 compute_action 作为 action, 透传其它字段
        sync_args = LammpsToolInput(
            action=args.compute_action,
            input_script=args.input_script,
            structure_file=args.structure_file,
            potentials=args.potentials,
            output_prefix=args.output_prefix,
            num_processes=args.num_processes,
            working_dir=args.working_dir,
            fixes=args.fixes,
        )

        job_id = await submit_background(
            "lammps",
            compute_action=args.compute_action,
            working_dir=args.working_dir,
            jobs=LammpsTool._async_jobs,
            run=lambda: self.call(sync_args, context),
        )

        return ToolResult(
            data={
                "job_id": job_id,
                "status": "running",
                "compute_action": args.compute_action,
                "working_dir": args.working_dir,
            },
            success=True,
        )

    def _handle_poll_job(self, args: LammpsToolInput) -> ToolResult:
        """查作业状态, 立即返回, 不阻塞.

        Returns:
            ToolResult.data = {
                "job_id": str,
                "status": "running" | "done" | "failed",
                "progress": 0-100,
                "partial_result": ... | None,
                "error": str | None,
                "elapsed": float,
            }
        """
        job_id = args.job_id
        job = LammpsTool._async_jobs.get(job_id)
        if job is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown job_id: {job_id}",
            )

        elapsed = time.time() - job["started_at"]
        status = job["status"]
        # 进度估算: running 给 50 (LAMMPS 内部进度没法简单跟踪),
        # done/failed 给 100. 真要精确进度得解析 log.lammps 的 thermo step,
        # 这里先做粗略估计, 跟 VaspTool 保持一致.
        progress = 100 if status in ("done", "failed") else 50

        return ToolResult(
            data={
                "job_id": job_id,
                "status": status,
                "progress": progress,
                "partial_result": job["result"] if status == "done" else None,
                "error": job["error"],
                "elapsed": round(elapsed, 2),
                "compute_action": job.get("compute_action"),
            },
            success=True,
        )

    async def _handle_wait_job(self, args: LammpsToolInput) -> ToolResult:
        """阻塞等待作业完成或超时.

        内部用 asyncio.wait_for 等后台 task, 超时返回当前状态 (status=running).
        作业完成返回最终结果, 失败返回错误. 超时不取消 task, 让它继续在后台跑.
        """
        job_id = args.job_id
        timeout = args.timeout
        job = LammpsTool._async_jobs.get(job_id)
        if job is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown job_id: {job_id}",
            )

        task = job.get("task")
        if task is None:
            # 同步退化路径下没有 task, 直接返回当前状态
            return self._handle_poll_job(args)

        # 已经完成的作业直接返回, 不再 wait
        if job["status"] in ("done", "failed"):
            return self._handle_poll_job(args)

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except TimeoutError:
            # 超时: 作业还在跑, 返回当前状态 (不取消 task, 让它继续)
            logger.debug("best-effort op failed", exc_info=True)
        except Exception as exc:
            # task 本身挂了 (不是超时), 把错误记下来
            job["status"] = "failed"
            job["error"] = str(exc)
            job["finished_at"] = time.time()

        return self._handle_poll_job(args)

    def _uq_hint(self) -> dict[str, Any]:
        """提示 agent 用 gp_tool 拟合 MSD-vs-time 做不确定性量化."""
        return {
            "tool": "gp_tool",
            "action": "fit",
            "suggestion": (
                "Consider calling gp_tool with action='fit' to fit a Gaussian "
                "Process to the MSD-vs-time trajectory for uncertainty quantification "
                "of the diffusion coefficient."
            ),
            "data_mapping": {
                "X": "time (trajectory timesteps)",
                "y": "msd (mean squared displacement)",
            },
        }

    def _parse_log(self, log_path: Path) -> tuple[dict, float | None, list[str]]:
        """Parse LAMMPS log file for thermodynamic data."""
        if not log_path.exists():
            return {}, None, ["Log file not found"]

        thermo_data: dict = {}
        final_energy: float | None = None
        warnings: list[str] = []

        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")
            columns = self._thermo_columns(content)
            data_rows = self._thermo_rows(content)
            if data_rows and columns:
                thermo_data = self._thermo_series(columns, data_rows)
            final_energy = self._final_toteng(thermo_data, content)
            warnings.extend(self._log_diagnostics(content))
        except Exception as e:
            warnings.append(f"Failed to parse log: {e}")

        return thermo_data, final_energy, warnings

    @staticmethod
    def _thermo_columns(content: str) -> list[str]:
        """Thermo table header columns (``Step Temp Press TotEng ...``)."""
        header_match = re.search(r"^(Step\s+.*?)$", content, re.MULTILINE)
        return header_match.group(1).split() if header_match else []

    def _thermo_rows(self, content: str) -> list[list]:
        """Numeric data rows: integer step followed by (almost) all numeric fields."""
        data_rows: list[list] = []
        for line in content.split("\n"):
            parts = line.split()
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            try:
                # Verify most parts are numeric
                numeric_count = sum(1 for p in parts if self._is_float(p))
                if numeric_count >= len(parts) - 1:
                    data_rows.append([self._to_float_or_str(p) for p in parts])
            except ValueError:
                logger.debug("suppressed in _parse_log", exc_info=True)
        return data_rows

    def _thermo_series(self, columns: list[str], data_rows: list[list]) -> dict:
        """Transpose rows into ``{column: [floats]}`` (non-numeric cells dropped)."""
        thermo_data: dict = {}
        for col_idx, col_name in enumerate(columns):
            if col_idx >= len(data_rows[0]):
                continue
            values = [row[col_idx] for row in data_rows if col_idx < len(row)]
            float_values = [
                v if isinstance(v, float) else float(v)
                for v in values
                if isinstance(v, float) or (isinstance(v, str) and self._is_float(v))
            ]
            if float_values:
                thermo_data[col_name.lower()] = float_values
        return thermo_data

    @staticmethod
    def _final_toteng(thermo_data: dict, content: str) -> float | None:
        """Last TotEng from the thermo table, else a regex fallback."""
        if thermo_data.get("toteng"):
            return thermo_data["toteng"][-1]
        if "toteng" in thermo_data:
            return None
        energy_match = re.findall(r"TotEng\s+([-\d.eE]+)", content)
        if not energy_match:
            return None
        with contextlib.suppress(ValueError):
            return float(energy_match[-1])
        return None

    @staticmethod
    def _log_diagnostics(content: str) -> list[str]:
        """Up to 5 WARNING lines followed by up to 3 ERROR lines."""
        warnings = [
            line.strip() for line in content.split("\n") if "WARNING" in line
        ][:5]
        warnings.extend(
            [line.strip() for line in content.split("\n") if "ERROR" in line][:3]
        )
        return warnings

    def _run_equilibrium_check(self, args: LammpsToolInput) -> ToolResult:
        """Check if an MD run has reached thermal/mechanical equilibrium.

        取 log 里尾部 window% 的 thermo 步: 温度需在 target_temp 5% 内, 线性漂移
        需 < 0.01 K/step; 未收敛时给出继续运行的 recommendation.
        """
        log_path = _resolve_log_path(args)
        if log_path is None or not log_path.exists():
            return _missing_log_result()

        thermo_data, _, _ = self._parse_log(log_path)
        if not thermo_data:
            return _equilibrium_empty_result(
                "Log file contains no thermo data. Check the log for errors."
            )
        if not thermo_data.get("temp"):
            return _equilibrium_empty_result(
                "No temperature data found in log. Check thermo_style."
            )

        tail_temps, tail_press, tail_steps, n_tail, n_total = _equilibrium_tail(
            thermo_data, args.window
        )
        metrics = self._equilibrium_metrics(
            tail_temps, tail_press, tail_steps, n_tail, n_total, args
        )
        equilibrated, avg_temp, avg_press, temp_drift, pressure_drift = metrics

        recommendation = self._build_equilibrium_recommendation(
            equilibrated, avg_temp, args.target_temp, temp_drift,
            avg_press, args.target_pressure, n_tail, n_total,
        )

        return ToolResult(
            data={
                "equilibrated": equilibrated,
                "avg_temp": avg_temp,
                "avg_pressure": avg_press,
                "temp_drift": temp_drift,
                "pressure_drift": pressure_drift,
                "recommendation": recommendation,
                "window_steps": n_tail,
                "total_steps": n_total,
            },
            success=True,
        )

    def _equilibrium_metrics(
        self,
        tail_temps: list,
        tail_press: list,
        tail_steps: list,
        n_tail: int,
        n_total: int,
        args: LammpsToolInput,
    ) -> tuple[bool, float, float | None, float, float | None]:
        """尾部窗口均值/漂移 + 平衡判定 (温度 5% 内且漂移 < 0.01 K/step)."""
        avg_temp = sum(tail_temps) / len(tail_temps)
        avg_press = sum(tail_press) / len(tail_press) if tail_press else None
        temp_drift = self._linear_slope(tail_steps, tail_temps)
        pressure_drift = (
            self._linear_slope(tail_steps, tail_press) if tail_press else None
        )

        temp_ok = True
        if args.target_temp is not None and args.target_temp > 0:
            temp_ok = abs(avg_temp - args.target_temp) / args.target_temp <= 0.05

        # drift threshold: ~1 K per 100 steps is a reasonable cutoff for
        # "still drifting". ponytail: this is heuristic and system-dependent;
        # for production runs, tune based on the specific thermostat/barostat.
        equilibrated = temp_ok and abs(temp_drift) < 0.01
        return equilibrated, avg_temp, avg_press, temp_drift, pressure_drift

    @staticmethod
    def _build_equilibrium_recommendation(
        equilibrated: bool,
        avg_temp: float,
        target_temp: float | None,
        temp_drift: float,
        avg_press: float | None,
        target_pressure: float | None,
        n_tail: int,
        n_total: int,
    ) -> str:
        if equilibrated:
            return "System has reached equilibrium. Proceed with production run."

        reasons: list[str] = []

        if target_temp is not None and target_temp > 0:
            rel_err = abs(avg_temp - target_temp) / target_temp
            if rel_err > 0.05:
                reasons.append(
                    f"avg temp {avg_temp:.1f} K deviates {rel_err*100:.1f}% from target {target_temp:.1f} K"
                )

        if abs(temp_drift) >= 0.01:
            reasons.append(f"temperature drift {temp_drift:.4f} K/step is too high")

        if target_pressure is not None and avg_press is not None and abs(avg_press - target_pressure) > max(abs(target_pressure) * 0.1, 100.0):
                reasons.append(
                    f"avg pressure {avg_press:.1f} bar is far from target {target_pressure:.1f} bar"
                )

        if not reasons:
            return "System is close to equilibrium. Extend equilibration to confirm stability."

        # suggest extending by 50% more steps or halving the timestep
        extend_by = max(int(n_total * 0.5), 1000)
        rec = "Not equilibrated: " + "; ".join(reasons) + "."
        rec += f" Extend run by ~{extend_by} steps or reduce timestep by half."
        return rec

    @staticmethod
    def _linear_slope(x: list[float], y: list[float]) -> float:
        """Least-squares slope of y vs x. Returns 0 for degenerate input."""
        n = len(y)
        if n < 2:
            return 0.0
        # use list indices as x when x is empty or mismatched length
        if len(x) != n:
            x = list(range(n))
        sx = sum(x)
        sy = sum(y)
        sxy = sum(xi * yi for xi, yi in zip(x, y))
        sxx = sum(xi * xi for xi in x)
        denom = n * sxx - sx * sx
        if denom == 0:
            return 0.0
        return (n * sxy - sx * sy) / denom

    def _apply_script_fixes(self, script: str, fixes: dict[str, str]) -> str:
        """Apply diagnosed fixes to LAMMPS input script.

        Replaces command parameters like 'timestep 1.0' with 'timestep 0.5'.
        """
        lines = script.split("\n")
        modified = []
        applied = set()

        for line in lines:
            stripped = line.strip().lower()
            # Skip comments and blank lines
            if not stripped or stripped.startswith("#"):
                modified.append(line)
                continue

            # Check each fix key
            for key, new_value in fixes.items():
                key_lower = key.lower()
                # Match command at start of line (allow leading whitespace)
                parts = stripped.split()
                if parts and parts[0] == key_lower:
                    # Replace the value part(s)
                    # e.g., 'timestep 1.0' → 'timestep 0.5'
                    # e.g., 'fix nvt all temp 300 300 0.1' → more complex
                    indent = line[: len(line) - len(line.lstrip())]
                    modified_line = f"{indent}{key} {new_value}"
                    modified.append(modified_line)
                    applied.add(key)
                    break
            else:
                modified.append(line)

        # If any fix wasn't applied, append it at the end
        for key, new_value in fixes.items():
            if key.lower() not in applied:
                modified.append(f"{key} {new_value}")

        return "\n".join(modified)

    def _read_script_params(self, input_path: Path) -> dict[str, Any]:
        """读 input.lammps 解析关键参数 (timestep 等), 给 AutoFixLoop 当上下文."""
        params: dict[str, Any] = {}
        try:
            for line in input_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if not parts:
                    continue
                cmd = parts[0].lower()
                # 只抓 AutoFixLoop 关心的几个命令: timestep / neighbor
                if cmd == "timestep" and len(parts) > 1:
                    try:
                        params["timestep"] = float(parts[1])
                    except ValueError:
                        params["timestep"] = parts[1]
                elif cmd == "neighbor" and len(parts) > 1:
                    try:
                        params["neighbor"] = float(parts[1])
                    except ValueError:
                        params["neighbor"] = parts[1]
        except Exception:  # 防御: 参数解析失败返回空配置
            logger.debug("suppressed in _read_script_params", exc_info=True)
        return params

    def _try_autofix(
        self, input_path: Path, error: str
    ) -> dict[str, Any] | None:
        """跑一次 AutoFixLoop, 命中规则就改 input.lammps 返回修了啥. 没命中返回 None."""
        try:
            from huginn.execution.autofix import AutoFixLoop

            current = self._read_script_params(input_path)
            fixed = AutoFixLoop().apply_fix("lammps_tool", error, current)
            if not fixed:
                return None
            reasoning = fixed.pop("__auto_fix", None)
            fixed.pop("__auto_fix_patterns_matched", None)
            if not fixed:
                return None
            # _apply_script_fixes 会整行替换, 只喂实际变化的参数,
            # 避免把无关命令行 (如 'neighbor 2.0 bin') 重写丢参数
            changed = {k: v for k, v in fixed.items() if current.get(k) != v}
            if not changed:
                return None
            str_fixes = {k: str(v) for k, v in changed.items()}
            new_script = self._apply_script_fixes(
                input_path.read_text(encoding="utf-8"), str_fixes
            )
            input_path.write_text(new_script, encoding="utf-8")
            return {"fixes": changed, "reasoning": reasoning}
        except Exception:  # 防御: 自动修复失败回退无修复
            logger.debug("best-effort op failed", exc_info=True)
            return None

    def _is_float(self, s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            logger.debug("best-effort op failed", exc_info=True)
            return False

    def _to_float_or_str(self, s: str):
        try:
            return float(s)
        except ValueError:
            return s

    def parse_trajectory(self, traj_path: str | Path) -> dict[str, Any]:
        """Parse LAMMPS trajectory file and compute basic analyses.

        Supports .lammpstrj and .dump formats. Always uses pure-Python path
        (see inline comment for why Rust fast-path is disabled).
        """
        from pathlib import Path

        traj_path = Path(traj_path)
        if not traj_path.exists():
            return {"error": "Trajectory file not found"}

        # ponytail: skip Rust fast-path (huginn_ext.parse_lammps_dump with
        # compute_msd=True) — it applies MIC to displacement-from-ref which
        # saturates at L²/6 for long-time diffusion (audit_20260717/14 P1-4,
        # pyext/src/analysis.rs:14-15 self-documents this). Python path
        # handles xu/yu/zu, ix/iy/iz, and wrapped-only incremental unwrap
        # correctly. Rust RDF is single-frame MIC (correct) but we lose that
        # perf here for simplicity.
        # PERF: Rust fast path 暂禁用 — 启用条件: Rust 端检测 xu/yu/zu 列后
        # 跳过 MIC clamping (pyext/src/analysis.rs:14-15). 当前 python fallback
        # 正确但慢. 这是有意识的性能债, 不是临时绕过.
        return self._parse_trajectory_python(traj_path)

    def _parse_trajectory_python(self, traj_path: str | Path) -> dict[str, Any]:
        """Pure-Python LAMMPS trajectory parser (baseline/fallback)."""
        from pathlib import Path

        traj_path = Path(traj_path)

        result = {
            "n_frames": 0,
            "n_atoms": 0,
            "atom_types": set(),
            "box_bounds": [],
            "timesteps": [],
        }

        try:
            frames = self._read_dump_frames(traj_path, result)
            result["n_frames"] = len(frames)
            result["atom_types"] = sorted(result["atom_types"])

            self._attach_trajectory_analyses(result, frames)

        except Exception as e:
            result["error"] = str(e)

        return result

    def _read_dump_frames(
        self, traj_path: Path, result: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Parse a LAMMPS dump file into per-frame atom dicts.

        Fills the ``result`` aggregates (n_atoms / atom_types / box_bounds /
        timesteps) in place while scanning, and returns the frame list.
        """
        frames: list[dict[str, Any]] = []
        current_frame: dict[str, Any] | None = None
        lines = traj_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "ITEM: TIMESTEP":
                if current_frame is not None:
                    frames.append(current_frame)
                current_frame = {"atoms": []}
                i = _dump_handle_timestep(lines, i, current_frame, result)
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                i = _dump_handle_n_atoms(lines, i, current_frame, result)
            elif line.startswith("ITEM: BOX BOUNDS"):
                i = _dump_handle_box(lines, i, current_frame, result)
            elif line.startswith("ITEM: ATOMS") and current_frame is not None:
                atoms, i = self._parse_atom_rows(
                    lines,
                    i,
                    line.replace("ITEM: ATOMS ", "").split(),
                    current_frame.get("n_atoms", 0),
                    result["atom_types"],
                )
                current_frame["atoms"] = atoms
            i += 1

        if current_frame is not None:
            frames.append(current_frame)
        return frames

    @staticmethod
    def _parse_atom_rows(
        lines: list[str],
        header_idx: int,
        headers: list[str],
        n_atoms: int,
        atom_types: set[int],
    ) -> tuple[list[dict[str, Any]], int]:
        """Parse ``n_atoms`` atom rows following the ``ITEM: ATOMS`` header line.

        Returns the atom dicts plus the index of the last consumed line.
        """
        atoms: list[dict[str, Any]] = []
        i = header_idx
        for _ in range(n_atoms):
            i += 1
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            atom: dict[str, Any] = {}
            for header, part in zip(headers, parts):
                try:
                    atom[header] = float(part)
                except ValueError:
                    atom[header] = part
            atoms.append(atom)
            if "type" in atom:
                atom_types.add(int(atom["type"]))
        return atoms, i

    def _attach_trajectory_analyses(
        self, result: dict[str, Any], frames: list[dict[str, Any]]
    ) -> None:
        """Attach MSD / RDF / VACF / van Hove / F(q,t) and the standardized timeseries."""
        if not frames:
            return
        first_atoms = frames[0]["atoms"]
        has_positions = all("x" in a for a in first_atoms)
        if has_positions and len(frames) > 1:
            self._attach_msd(result, frames)
        if has_positions:
            self._attach_rdf_series(result, frames)
        if len(frames) > 1 and all(
            "vx" in a and "vy" in a and "vz" in a for a in first_atoms
        ):
            self._attach_vacf(result, frames)

        timeseries: list[dict] = self._collect_scalar_timeseries(result)
        if has_positions and len(frames) > 1:
            timeseries.extend(self._collect_spatial_timeseries(result, frames))
        if timeseries:
            result["_physical_timeseries"] = timeseries

    def _attach_msd(self, result: dict[str, Any], frames: list[dict[str, Any]]) -> None:
        """Compute MSD and surface ``_msd_warnings`` (saturated diffusion guard)."""
        # _compute_msd sets self._msd_warnings — surface to caller so
        # downstream agents don't trust a saturated diffusion coeff.
        msd = self._compute_msd(frames)
        if msd:
            result["msd"] = msd
        if getattr(self, "_msd_warnings", None):
            result["msd_warnings"] = list(self._msd_warnings)
            self._msd_warnings = []

    def _attach_rdf_series(
        self, result: dict[str, Any], frames: list[dict[str, Any]]
    ) -> None:
        """Per-frame g(r) series (uniformly subsampled for large trajectories).

        ponytail: 这不是真正的 van Hove g(r,t), 只是按帧采样的 g(r) 序列.
        真 van Hove 见 _compute_van_hove — 那才是 spatio-temporal correlation.
        """
        # 采样帧数限制: 大轨迹取均匀采样避免 O(n_frames * n_atoms²) 爆炸
        max_rdf_frames = 20
        if len(frames) <= max_rdf_frames:
            sample_frames = frames
            idx = list(range(len(frames)))
        else:
            idx = [
                int(round(i * (len(frames) - 1) / (max_rdf_frames - 1)))
                for i in range(max_rdf_frames)
            ]
            sample_frames = [frames[i] for i in idx]

        rdf_series = []
        for fi, frame in enumerate(sample_frames):
            rdf = self._compute_rdf(frame)
            if not rdf:
                continue
            frame_index = idx[fi] if len(frames) > max_rdf_frames else fi
            rdf_series.append({
                "frame_index": frame_index,
                "timestep": frame.get("timestep", frame_index),
                **rdf,
            })
        if not rdf_series:
            return
        result["rdf_series"] = rdf_series
        # 末帧 RDF 保留兼容老字段
        result["rdf"] = rdf_series[-1]

    def _attach_vacf(self, result: dict[str, Any], frames: list[dict[str, Any]]) -> None:
        """VACF + Green-Kubo diffusion (needs vx/vy/vz columns).

        第一性原理方法, 比 Einstein 关系在短时模拟上更准.
        """
        vacf = self._compute_vacf(frames)
        if not vacf:
            return
        result["vacf"] = vacf
        gk = self._green_kubo_diffusion(vacf)
        if gk is not None:
            result["diffusion_green_kubo"] = gk

    @staticmethod
    def _collect_scalar_timeseries(result: dict[str, Any]) -> list[dict]:
        """MSD / RDF-first-peak / VACF as standardized ``(t, v)`` channels."""
        # P3: 把物理时序统一注册为标准化通道, 让 cognition loop 收集后注入 prompt.
        # 每条时序: {name, unit, data: [(t, v), ...], meaning, source}.
        # engine 算 trend (rising/decaying/flat) 判动力学 regime:
        #   - VACF decaying + MSD rising → 扩散型 (同一物理两种表征, 可交叉验证)
        #   - VACF 振荡 + MSD flat → 束缚态
        #   - RDF 峰位漂移 → 结构弛豫/相变 (与 VACF 解耦的独立维度)
        timeseries: list[dict] = []
        if result.get("msd"):
            timeseries.append({
                "name": "MSD",
                "unit": "Å²",
                "data": [
                    (d.get("timestep", i), d.get("msd", 0.0))
                    for i, d in enumerate(result["msd"])
                ],
                "meaning": "mean squared displacement <Δr²>",
                "source": "lammps",
            })
        peak_data = LammpsTool._rdf_first_peaks(result.get("rdf_series") or [])
        if peak_data:
            timeseries.append({
                "name": "RDF_first_peak",
                "unit": "Å",
                "data": peak_data,
                "meaning": "RDF first peak position (structural relaxation)",
                "source": "lammps",
            })
        if result.get("vacf"):
            timeseries.append({
                "name": "VACF",
                "unit": "Å²/ps²",
                "data": [
                    (d.get("timestep", i), d.get("vacf", 0.0))
                    for i, d in enumerate(result["vacf"])
                ],
                "meaning": "velocity autocorrelation <v(0)·v(t)>",
                "source": "lammps",
            })
        return timeseries

    @staticmethod
    def _rdf_first_peaks(rdf_series: list[dict]) -> list[tuple]:
        """g(r) 第一峰位置序列 — 峰位漂移直接对应结构弛豫.

        ponytail: argmax 取最高峰, 对液体/固体第一峰通常就是最高峰;
        多峰体系升级路径是找第一个局部最大.
        """
        peak_data: list[tuple] = []
        for fi, fr in enumerate(rdf_series):
            g = fr.get("g") or (fr.get("rdf") or {}).get("g")
            r = fr.get("r") or (fr.get("rdf") or {}).get("r")
            if not (g and r and len(g) == len(r) and len(g) > 0):
                continue
            try:
                peak_idx = max(range(len(g)), key=lambda k: g[k])
                peak_data.append((fr.get("timestep", fi), r[peak_idx]))
            except (ValueError, TypeError):
                logger.debug("best-effort op failed", exc_info=True)
                continue
        return peak_data

    def _collect_spatial_timeseries(
        self, result: dict[str, Any], frames: list[dict[str, Any]]
    ) -> list[dict]:
        """van Hove G_s(r,t) and F(q,t) as ``(t, r, v)`` spatial channels."""
        # S1: van Hove G_s(r,t) — spatio-temporal correlation, 三元组 (t, r, G_s).
        # 这是时空表征的数学根: MSD 是它的二阶矩, F(q,t) 是它的空间傅立叶变换.
        timeseries: list[dict] = []
        van_hove = self._compute_van_hove(frames)
        if van_hove:
            result["van_hove"] = van_hove
            # 展开为 (t, r, G_s) 三元组让 _format_timeseries_context 识别空间维度
            timeseries.append({
                "name": "van_hove_G_s",
                "unit": "1/Å³",
                "data": [
                    (entry["timestep"], r, gs)
                    for entry in van_hove
                    for r, gs in zip(entry["r"], entry["G_s"])
                ],
                "meaning": "van Hove self-part G_s(r,t) — single-atom displacement distribution",
                "source": "lammps",
                "spatial": True,  # 标记: data 是 (t, r, v) 三元组, 不是 (t, v) 二元组
            })
        # S1: F(q,t) — 中间散射函数, G(r,t) 的空间傅立叶变换, 与原位 XRD 帧间峰位漂移对偶
        fqt = self._compute_F_q_t(frames)
        if fqt:
            result["F_q_t"] = fqt
            timeseries.append({
                "name": "F_q_t",
                "unit": "1",
                "data": [
                    (entry["timestep"], q, f)
                    for entry in fqt
                    for q, f in zip(entry["q_values"], entry["F"])
                ],
                "meaning": "intermediate scattering function F(q,t) — Fourier transform of G(r,t)",
                "source": "lammps",
                "spatial": True,
            })
        return timeseries

    def _compute_msd(self, frames: list[dict]) -> list[dict] | None:
        """Compute mean squared displacement across frames.

        三档解包策略 (优先级): xu/yu/zu → ix/iy/iz 重建 → 逐帧 MIC 增量解包.
        旧实现直接对包裹坐标做 frame[0] 差值, 跨边界跳变 L² 会让 MSD 爆刺;
        Rust 版对位移做 MIC 截断则长时饱和在 L²/6. 两种都只在各自场景错.
        返回 [{"timestep", "msd"}]; 饱和告警写入 self._msd_warnings 供上层附着.
        """
        self._msd_warnings = []
        if not frames or len(frames) < 2:
            return None
        n_atoms = len(frames[0]["atoms"])
        if n_atoms == 0:
            return None

        mode = _msd_coord_mode(frames[0]["atoms"][0])
        if mode is None:
            return None

        # ponytail: assume orthorhombic constant box — NPT variable-box not
        # handled; for that case recompute L per frame (upgrade path).
        lengths = _msd_box_lengths(frames[0])

        if mode == "wrapped":
            self._msd_warnings.append(_MSD_WRAPPED_ONLY_WARNING)

        ref_pos = _msd_ref_positions(frames[0]["atoms"], mode, lengths)

        # For wrapped-only path: track running unwrapped position per atom
        # AND previous wrapped position (for computing per-frame wrapped
        # displacement, which is what MIC operates on).
        running_pos: list[list[float]] | None = None
        prev_wrapped: list[tuple[float, float, float]] | None = None
        if mode == "wrapped":
            running_pos = [list(p) for p in ref_pos]
            prev_wrapped = [tuple(p) for p in ref_pos]

        msd_data, saturation = _msd_series(
            frames, ref_pos, mode, lengths, running_pos, prev_wrapped
        )
        if saturation is not None:
            self._msd_warnings.append(_msd_saturation_warning(*saturation))
        return msd_data

    def _compute_rdf(
        self, frame: dict, bins: int = 100, r_max: float | None = None
    ) -> dict | None:
        """Compute radial distribution function for a single frame."""
        try:
            import numpy as np

            atoms = frame["atoms"]
            pos = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=np.float64)
            n = len(pos)

            # Estimate r_max from box
            box = frame.get("box", [[0, 10], [0, 10], [0, 10]])
            lx = box[0][1] - box[0][0]
            ly = box[1][1] - box[1][0]
            lz = box[2][1] - box[2][0]
            if r_max is None:
                r_max = min(lx, ly, lz) / 2

            r_max / bins
            box_vec = np.array([lx, ly, lz])

            # Compute pairwise distances with minimum image convention.
            # ponytail: O(n²) memory — fine for n<10k, chunk for larger systems.
            d = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (n, n, 3)
            d -= box_vec * np.round(d / box_vec)
            r = np.sqrt((d ** 2).sum(axis=2))  # (n, n)
            mask = (r > 0) & (r < r_max)
            distances = r[mask]

            # Bin distances — full matrix counts each pair twice (i,j)+(j,i).
            g, _ = np.histogram(distances, bins=bins, range=(0, r_max))
            g = g.astype(np.float64)

            # Normalize
            volume = lx * ly * lz
            rho = n / volume
            r_edges = np.linspace(0, r_max, bins + 1)
            r_inner = r_edges[:-1]
            r_outer = r_edges[1:]
            shell_vol = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
            shell_vol = np.where(shell_vol > 0, shell_vol, 1.0)
            g /= n * rho * shell_vol

            r_values = ((r_edges[:-1] + r_edges[1:]) / 2).tolist()
            return {"r": r_values, "g": g.tolist(), "bins": bins, "r_max": r_max}
        except Exception:  # 防御: RDF 计算失败返回空
            logger.debug("best-effort op failed", exc_info=True)
            return None

    def _compute_vacf(self, frames: list[dict]) -> list[dict] | None:
        """速度自相关函数 (VACF): <v(0)·v(t)>.

        VACF 是 Green-Kubo 扩散系数积分的核心. 与 MSD 互补:
        - MSD (Einstein): 长时极限 D = MSD/(6t), 需要长轨迹
        - VACF (Green-Kubo): D = (1/3) ∫₀^∞ <v(0)·v(t)> dt, 短时模拟更准

        实现要点:
        - 参考速度 v(0) 取 frame[0], 每帧算 <v_i(0)·v_i(t)>_i 均值
        - dt 从相邻 timestep 差推断 (假设等间隔)
        - 速度列必须存在 (vx vy vz), 没有返回 None
        """
        try:
            if not frames or len(frames) < 2:
                return None
            if len(frames[0]["atoms"]) == 0:
                return None
            return _vacf_series(frames, _velocities(frames[0]))
        except Exception:  # 防御: VACF 计算失败返回空
            logger.debug("best-effort op failed", exc_info=True)
            return None

    def _green_kubo_diffusion(self, vacf_data: list[dict]) -> float | None:
        """Green-Kubo 扩散系数: D = (1/3) ∫₀^∞ C(t) dt.

        C(t) = <v(0)·v(t)>, 积分到轨迹末尾.
        3D 体系中 <v_x²> = k_B T / m, D = (1/3) ∫ C(t) dt.

        实现用梯形法离散积分. 单位:
        - 速度: Å/ps (LAMMPS real+metal units 默认) → D: Å²/ps
        - 速度: Å/fs (LAMMPS real units, dt=fs) → D: Å²/fs, ×1e3 → Å²/ps

        不自动做单位转换, 调用方根据 LAMMPS units 判断.
        ponytail: 等间隔假设, 非等间隔用 numpy.trapz(x=timesteps).
        """
        try:
            if not vacf_data or len(vacf_data) < 2:
                return None

            # 从 timestep 差推断 dt
            t0 = vacf_data[0]["timestep"]
            t1 = vacf_data[1]["timestep"]
            dt = t1 - t0
            if dt <= 0:
                return None

            # 梯形法积分 C(t) dt, 除以 3 (3D)
            vacf = [d["vacf"] for d in vacf_data]
            integral = 0.0
            for i in range(1, len(vacf)):
                integral += 0.5 * (vacf[i] + vacf[i - 1]) * dt

            d_gk = integral / 3.0
            return float(d_gk)
        except Exception:  # 防御: 扩散系数计算失败返回空
            logger.debug("best-effort op failed", exc_info=True)
            return None

    # ------------------------------------------------------------------ van Hove

    _VAN_HOVE_TOP_K = 32  # ponytail: 32 起步, 平衡存储和统计噪声. 升级路径: 按 cell 分区抽样.

    def _compute_van_hove(
        self, frames: list[dict], r_max: float = 10.0, bins: int = 20,
    ) -> list[dict] | None:
        """van Hove 函数 self-part G_s(r, t) — 真正的 spatio-temporal correlation.

        G_s(r, t) = (1/N) Σ_i δ(r - |r_i(t) - r_i(0)|)
        描述单原子在时间 t 内位移到 r 处的概率分布.

        与 MSD 的关系: MSD(t) = ∫ r² G_s(r, t) d³r  (G_s 是 MSD 的母函数)
        与 F(q,t) 的关系: F(q,t) = ∫ exp(iq·r) G(r,t) d³r  (F 是 G 的空间傅立叶变换)

        算法:
        1. 取每帧原子的 unwrapped position (xu/yu/zu 优先, 否则 x/y/z)
        2. 按 |Δr(t_last)|² 排序选 top-K=32 原子做稀疏采样, 避免 O(N²) 爆炸
        3. 对每个采样原子, 算 |Δr(t)| = |r_i(t) - r_i(0)|, 然后 histogram 给出 G_s 在 r 上的分布

        ponytail: 只算 self-part (G_s), distinct-part (G_d) 需要 pair 统计, 存储翻倍.
        升级路径: 同时算 G_d, 输出完整 G(r,t).
        天花板: top-K 采样下统计噪声更大, 但动力学 regime 识别够用.
        """
        try:
            import numpy as np

            if not frames or len(frames) < 2:
                return None

            r0 = _traj_positions(frames[0])  # (N, 3)
            n_atoms = len(r0)
            if n_atoms == 0:
                return None

            # 按 |Δr(t_last)|² 排序选 top-K 原子
            disp_sq_last = ((_traj_positions(frames[-1]) - r0) ** 2).sum(axis=1)
            k = min(self._VAN_HOVE_TOP_K, n_atoms)
            top_idx = np.argsort(disp_sq_last)[-k:]

            # r bins
            r_edges = np.linspace(0, r_max, bins + 1)
            r_centers = ((r_edges[:-1] + r_edges[1:]) / 2).tolist()

            return _vh_series(frames, r0, top_idx, r_edges, r_centers, k)
        except Exception:  # 防御: 关联函数计算失败返回空
            logger.debug("best-effort op failed", exc_info=True)
            return None

    def _compute_F_q_t(  # noqa: N802
        self, frames: list[dict], q_values: list[float] | None = None,
    ) -> list[dict] | None:
        """中间散射函数 F(q, t) — G(r,t) 的空间傅立叶变换.

        F(q, t) = (1/N) Σ_i <exp(i q · (r_i(t) - r_i(0)))>
                 = (1/N) Σ_i cos(q · Δr_i(t))   (虚部平均为 0, 取实部)

        与 VACF 的关系: F(q→0, t) 是 MSD 的傅立叶对偶; F(q_峰, t) 衰减
        直接对应实验可测的 inelastic neutron scattering / XPCS 信号.

        与视觉的耦合: 原位 XRD 帧间峰位漂移可视化 ≈ F(q_峰, t) 衰减.

        算法:
        1. 取 unwrapped positions
        2. 对每个 q (默认 3 个典型值 [0.5, 1.0, 2.0] 1/Å, 覆盖典型第一峰),
           算 F(q, t) = <cos(q · Δr(t))>
        3. 三个 q 方向取 (1,0,0), (0,1,0), (0,0,1) 各自 q·Δr, 再三方向均值

        ponytail: 只算 self-part, q 固定 3 个方向取均值. 升级路径: 球面积分.
        """
        try:
            if not frames or len(frames) < 2:
                return None

            r0 = _traj_positions(frames[0])
            if len(r0) == 0:
                return None

            if q_values is None:
                # 默认 3 个典型 q 值, 覆盖典型第一峰
                # ponytail: 静态选, 不从 RDF 自动抽峰. 升级路径: 抽 RDF 第一峰位.
                q_values = [0.5, 1.0, 2.0]

            return _fqt_series(frames, r0, q_values)
        except Exception:  # 防御: 散射函数计算失败返回空
            logger.debug("best-effort op failed", exc_info=True)
            return None

    # ------------------------------------------------------------------ DEM

    async def _handle_dem_packing(
        self, args: LammpsToolInput, context: ToolContext
    ) -> ToolResult:
        """DEM 颗粒碰撞模拟: 生成 LAMMPS granular 输入脚本 + 可选执行."""
        script = self._generate_dem_input_script(args)
        work_dir = _dem_work_dir(args, context)
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "input.dem.lammps"
        script_path.write_text(script, encoding="utf-8")

        data = _dem_payload(args, script_path, script)

        early = self._dem_ensure_executable(data)
        if early is not None:
            return early

        return self._dem_execute(args, data, work_dir, script_path)

    def _dem_ensure_executable(self, data: dict) -> ToolResult | None:
        """没有 LAMMPS 可执行文件就只返回脚本; 有则返回 None."""
        if self.lammps_executable:
            return None
        from huginn.tools.sim.executable_resolver import resolve_executable

        resolution = resolve_executable("lammps")
        if isinstance(resolution, str):
            self.lammps_executable = resolution
            return None
        data["needs_resolution"] = True
        data["resolution_request"] = resolution.to_dict()
        return ToolResult(
            data=data,
            success=True,
            error="LAMMPS executable not found. Script generated only.",
        )

    def _dem_execute(
        self, args: LammpsToolInput, data: dict, work_dir: Path, script_path: Path
    ) -> ToolResult:
        """执行 LAMMPS — 走 sandbox, 不能裸 subprocess."""
        try:
            cmd = [str(self.lammps_executable), "-in", str(script_path)]
            result = self.sandbox.run(cmd, cwd=str(work_dir), timeout=3600)
            ok = result.returncode == 0
            data["returncode"] = result.returncode
            data["stdout_tail"] = result.stdout[-2000:] if result.stdout else ""
            data["stderr_tail"] = result.stderr[-2000:] if result.stderr else ""

            if ok:
                data["output_dir"] = str(work_dir)
                data["status"] = "completed"
                _attach_provenance(args, data)
            else:
                data["status"] = "failed"

            return ToolResult(
                data=data,
                success=ok,
                error=None if ok else f"LAMMPS exited with code {result.returncode}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                data=data, success=False, error="LAMMPS DEM execution timed out (3600s)"
            )
        except Exception as e:
            return ToolResult(
                data=data, success=False, error=f"DEM execution failed: {e}"
            )

    @staticmethod
    def _generate_dem_input_script(args: LammpsToolInput) -> str:
        """生成 LAMMPS DEM (Discrete Element Method) 颗粒碰撞脚本.

        Hertz-Mindlin 接触模型 (pair_style granular): 法向 Hertzian, 切向
        Mindlin, Tsuji 阻尼 (由恢复系数反算 γ_n), Coulomb 摩擦. 需 GRANULAR
        package; 语法依据 https://docs.lammps.org/pair_granular.html
        """
        p = _dem_params(args)
        return _dem_header(p) + _dem_physics(p) + _dem_run(p)
