"""能量 / 力评估器 — LJ / ML 势 / VASP 三路后端.

empirical (LJ) 是内置兜底, ml_potential 和 vasp 通过 ToolRegistry 互调,
不硬 import, 没注册就退到 LJ.
"""
from __future__ import annotations

import math
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from huginn.types import ToolContext, ToolResult
from huginn.tools.neb._io import write_xyz, write_poscar

if TYPE_CHECKING:
    from huginn.tools.neb.tool import NEBToolInput


async def eval_images(
    images: list[np.ndarray],
    atomic_numbers: list[int],
    cell: np.ndarray | None,
    args: "NEBToolInput",
    context: ToolContext,
) -> tuple[list[float], list[np.ndarray]]:
    """批量评估每个 image 的能量和力."""
    energies: list[float] = []
    forces: list[np.ndarray] = []
    for img in images:
        e, f = await eval_single(
            img, atomic_numbers, cell, args, context
        )
        energies.append(e)
        forces.append(f)
    return energies, forces


async def eval_single(
    positions: np.ndarray,
    atomic_numbers: list[int],
    cell: np.ndarray | None,
    args: "NEBToolInput",
    context: ToolContext,
) -> tuple[float, np.ndarray]:
    """单点能量 + 力评估, 按能量评估后端路由."""
    if args.energy_evaluator == "empirical":
        return eval_lj(
            positions, atomic_numbers, args.lj_epsilon, args.lj_sigma
        )
    if args.energy_evaluator == "ml_potential":
        return await eval_via_ml_potential(
            positions, atomic_numbers, cell, args, context
        )
    if args.energy_evaluator == "vasp":
        return await eval_via_vasp(
            positions, atomic_numbers, cell, args, context
        )
    raise ValueError(f"未知 energy_evaluator: {args.energy_evaluator}")


def eval_lj(
    positions: np.ndarray,
    atomic_numbers: list[int],
    epsilon: float,
    sigma: float,
) -> tuple[float, np.ndarray]:
    """Lennard-Jones 势 (autodiff_tool 同款公式) + 解析力.

    E = 4ε[(σ/r)^12 - (σ/r)^6]
    F_j = dE/dr_jk 方向投影, 解析导数避免数值差分误差.
    截断半径 2.5σ, 避免远距离噪声.
    """
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    rcut = 2.5 * sigma
    rcut2 = rcut * rcut

    energy = 0.0
    forces = np.zeros_like(pos)

    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r2 = float(np.dot(rij, rij))
            if r2 < 1e-12 or r2 > rcut2:
                continue
            r = math.sqrt(r2)
            sr6 = (sigma / r) ** 6
            sr12 = sr6 * sr6
            energy += 4.0 * epsilon * (sr12 - sr6)
            # dE/dr = 4ε[-12 σ^12/r^13 + 6 σ^6/r^7]
            dE_dr = 4.0 * epsilon * (-12.0 * sr12 / r + 6.0 * sr6 / r)
            # F_i = dE/dr * (r_i - r_j)/r, F_j = -F_i
            fvec = dE_dr * rij / r
            forces[i] += fvec
            forces[j] -= fvec

    return float(energy), forces


async def eval_via_ml_potential(
    positions: np.ndarray,
    atomic_numbers: list[int],
    cell: np.ndarray | None,
    args: "NEBToolInput",
    context: ToolContext,
) -> tuple[float, np.ndarray]:
    """写临时结构文件, 调 ml_potential_tool.predict 算能量 + 力."""
    try:
        from huginn.tools.registry import ToolRegistry

        ml_tool = ToolRegistry.get("ml_potential_tool")
    except Exception:
        ml_tool = None

    if ml_tool is None:
        # ML 势没注册, 退到 LJ 并标记
        return eval_lj(
            positions, atomic_numbers, args.lj_epsilon, args.lj_sigma
        )

    # 写临时 XYZ (ml_potential_tool 用 ASE 读, XYZ 最通用)
    tmp_dir = tempfile.mkdtemp(prefix="neb_eval_")
    tmp_xyz = os.path.join(tmp_dir, "structure.xyz")
    write_xyz(tmp_xyz, atomic_numbers, positions, cell)

    ml_input = ml_tool.input_schema(
        backend=args.ml_backend,
        action="predict",
        structure_file=tmp_xyz,
        model_path=args.ml_model_path,
    )
    result = await ml_tool.call(ml_input, context)
    if not result.success or not result.data:
        raise RuntimeError(
            f"ml_potential_tool 评估失败: {result.error}"
        )

    energy = float(result.data.get("energy", 0.0))
    forces_raw = result.data.get("forces") or []
    forces = np.asarray(forces_raw, dtype=float)
    if forces.size == 0:
        # 没力就退化为零力, NEB 自然不会动, 但不挂
        forces = np.zeros_like(positions)
    return energy, forces


async def eval_via_vasp(
    positions: np.ndarray,
    atomic_numbers: list[int],
    cell: np.ndarray | None,
    args: "NEBToolInput",
    context: ToolContext,
) -> tuple[float, np.ndarray]:
    """调 vasp_tool.scf 做单点能.

    vasp_tool 需要 working_dir 里有 INCAR/POTCAR/KPOINTS, 这里从
    args.vasp_working_dir 拷一份, 把 POSCAR 替换成当前 image.
    没传 vasp_working_dir 就报错.
    """
    if not args.vasp_working_dir:
        raise RuntimeError(
            "vasp 评估器需要 vasp_working_dir (含 INCAR/POTCAR/KPOINTS)"
        )

    try:
        from huginn.tools.registry import ToolRegistry

        vasp_tool = ToolRegistry.get("vasp_tool")
    except Exception:
        vasp_tool = None

    if vasp_tool is None:
        return eval_lj(
            positions, atomic_numbers, args.lj_epsilon, args.lj_sigma
        )

    import shutil
    from pathlib import Path

    src = Path(args.vasp_working_dir)
    tmp_dir = tempfile.mkdtemp(prefix="neb_vasp_")
    dst = Path(tmp_dir)
    # 拷模板文件
    for fname in ("INCAR", "POTCAR", "KPOINTS"):
        f = src / fname
        if f.exists():
            shutil.copy2(f, dst / fname)
    # 写 POSCAR
    write_poscar(dst / "POSCAR", atomic_numbers, positions, cell)

    vasp_input = vasp_tool.input_schema(
        action="scf",
        working_dir=tmp_dir,
    )
    result = await vasp_tool.call(vasp_input, context)
    if not result.success or not result.data:
        raise RuntimeError(f"vasp_tool.scf 评估失败: {result.error}")

    data = result.data
    energy = float(data.get("energy") or 0.0)
    # 解析力 — vasp_tool 在 parsed.forces 里返回 list of {force: [...]}
    forces = np.zeros_like(positions)
    parsed = data.get("parsed") or {}
    forces_raw = parsed.get("forces") or []
    if forces_raw:
        for i, fitem in enumerate(forces_raw[: positions.shape[0]]):
            forces[i] = np.asarray(fitem.get("force", [0.0, 0.0, 0.0]))
    return energy, forces
