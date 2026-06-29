"""NEBTool 主体 — 势能面探索工具.

action 入口:
  - neb: Nudged Elastic Band 最小能量路径
  - pes_scan: 沿坐标轴网格扫描势能面
  - saddle_search: Dimer 方法找一阶鞍点
  - mep_analyze: 分析 NEB 给出的最小能量路径
  - landscape_topology: 势能面地形拓扑摘要

能量评估通过 ToolRegistry 互调 ml_potential_tool / vasp_tool, 不硬 import.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult
from huginn.tools.neb._io import read_structure
from huginn.tools.neb._evaluators import eval_images, eval_single
from huginn.tools.neb._neb_core import (
    idpp_initial_path,
    compute_neb_forces,
    compute_barriers,
    compute_path_length,
)
from huginn.tools.neb._dimer import dimer_rotate, estimate_hessian_along_mode
from huginn.tools.neb._topology import topology_via_tda, basin_analysis, find_extrema


# ---------------------------------------------------------------------------
# 输入 / 输出 schema
# ---------------------------------------------------------------------------


class NEBToolInput(BaseModel):
    """NEB / PES 工具的统一输入.

    structure 类字段支持两种写法:
      1. 文件路径字符串 (POSCAR / CIF / XYZ / 任何 ASE 能读的格式)
      2. 内联 dict: {"atomic_numbers": [...], "positions": [[x,y,z],...],
                     "cell": [[...]] (可选)}
    """

    action: Literal[
        "neb",
        "pes_scan",
        "saddle_search",
        "mep_analyze",
        "landscape_topology",
    ] = Field(..., description="NEB/PES 子任务类型")

    # ---- neb ----
    initial_structure: str | dict | None = Field(
        default=None, description="NEB 起点结构 (路径或内联 dict)"
    )
    final_structure: str | dict | None = Field(
        default=None, description="NEB 终点结构 (路径或内联 dict)"
    )
    n_images: int = Field(default=7, ge=3, description="NEB 图像数 (含首尾)")
    spring_constant: float = Field(
        default=5.0, description="弹簧常数 k (eV/Å^2)"
    )
    max_iter: int = Field(default=300, ge=1, description="最大迭代步数")
    tolerance: float = Field(
        default=0.05, description="收敛阈值 (最大原子受力, eV/Å)"
    )
    climbing_image: bool = Field(
        default=True, description="启用 climbing image NEB (CI-NEB)"
    )

    # ---- pes_scan / saddle_search ----
    structure: str | dict | None = Field(
        default=None, description="PES 扫描 / 鞍点搜索的初始结构"
    )
    # pes_scan
    scan_coords: list[list[int]] | None = Field(
        default=None,
        description="扫描坐标列表, 每项 [atom_idx, axis(0/1/2)]",
    )
    scan_range: list[float] | None = Field(
        default=None,
        description="扫描位移范围 (Å), 如 [-1.0, 1.0]",
    )
    n_points: int = Field(default=11, ge=2, description="每维扫描点数")
    # saddle_search
    search_method: Literal["dimer", "growing_string", "eigenvalue"] = Field(
        default="dimer", description="鞍点搜索方法"
    )
    dimer_length: float = Field(default=0.01, description="Dimer 长度 (Å)")
    dimer_rotations: int = Field(
        default=20, description="每步 Dimer 旋转次数"
    )

    # ---- mep_analyze ----
    neb_result: dict | None = Field(
        default=None, description="neb action 返回的结果 dict"
    )
    analysis_type: Literal[
        "energy_profile", "barrier", "decomposition"
    ] = Field(default="energy_profile")

    # ---- landscape_topology ----
    pes_data: dict | None = Field(
        default=None, description="pes_scan 返回的结果 dict"
    )
    method: Literal["tda", "basin_analysis"] = Field(
        default="basin_analysis", description="拓扑分析方法"
    )

    # ---- 共享:能量评估 ----
    energy_evaluator: Literal[
        "ml_potential", "vasp", "empirical"
    ] = Field(
        default="empirical",
        description="能量评估后端: ml_potential / vasp / empirical (内置 LJ)",
    )
    ml_backend: Literal["mace", "chgnet", "nep"] = Field(
        default="mace", description="ml_potential 评估器后端"
    )
    ml_model_path: str | None = Field(
        default=None, description="ML 势模型路径 (None 用预训练)"
    )
    vasp_working_dir: str | None = Field(
        default=None,
        description="vasp 评估器的工作目录模板 (含 INCAR/POTCAR/KPOINTS)",
    )
    # empirical (LJ) 参数, 方便调势阱深浅
    lj_epsilon: float = Field(default=1.0, description="LJ ε (eV)")
    lj_sigma: float = Field(default=1.0, description="LJ σ (Å)")

    @model_validator(mode="after")
    def _check_action_fields(self) -> "NEBToolInput":
        """不同 action 校验必填字段, 别等 call() 才挂."""
        if self.action == "neb":
            if not self.initial_structure or not self.final_structure:
                raise ValueError("neb action 需要 initial_structure 和 final_structure")
        if self.action in ("pes_scan", "saddle_search"):
            if not self.structure:
                raise ValueError(f"{self.action} action 需要 structure")
        if self.action == "pes_scan":
            if not self.scan_coords or self.scan_range is None:
                raise ValueError("pes_scan action 需要 scan_coords 和 scan_range")
        if self.action == "mep_analyze" and not self.neb_result:
            raise ValueError("mep_analyze action 需要 neb_result")
        if self.action == "landscape_topology" and not self.pes_data:
            raise ValueError("landscape_topology action 需要 pes_data")
        return self


class NEBToolOutput(BaseModel):
    """松散输出信封, 真实 payload 在 ToolResult.data."""

    action: str
    data: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# 工具主体
# ---------------------------------------------------------------------------


class NEBTool(HuginnTool):
    """势能面探索工具: NEB / PES 扫描 / 鞍点搜索 / MEP 分析 / 地形拓扑."""

    name = "neb_tool"
    category = "sim"
    description = (
        "势函数地形图工具. 提供 Nudged Elastic Band (NEB) 最小能量路径、"
        "PES 网格扫描、Dimer 鞍点搜索、MEP 分析、地形拓扑摘要. "
        "能量评估通过 ToolRegistry 互调 ml_potential_tool / vasp_tool, "
        "也支持内置 empirical (LJ) 评估器."
    )
    input_schema = NEBToolInput
    # 不写文件 (除了临时文件), 不修改用户输入; 默认按只读处理
    read_only = True

    async def call(
        self, args: NEBToolInput, context: ToolContext
    ) -> ToolResult:
        try:
            if args.action == "neb":
                return await self._action_neb(args, context)
            if args.action == "pes_scan":
                return await self._action_pes_scan(args, context)
            if args.action == "saddle_search":
                return await self._action_saddle_search(args, context)
            if args.action == "mep_analyze":
                return self._action_mep_analyze(args)
            if args.action == "landscape_topology":
                return await self._action_landscape_topology(args, context)
            return ToolResult(
                data=None, success=False, error=f"未知 action: {args.action}"
            )
        except Exception as exc:
            return ToolResult(
                data=None, success=False, error=f"NEBTool 错误: {exc}"
            )

    # =====================================================================
    # 1. NEB
    # =====================================================================

    async def _action_neb(
        self, args: NEBToolInput, context: ToolContext
    ) -> ToolResult:
        """Nudged Elastic Band 最小能量路径."""
        init = read_structure(args.initial_structure)
        final = read_structure(args.final_structure)

        if init["atomic_numbers"] != final["atomic_numbers"]:
            return ToolResult(
                data=None,
                success=False,
                error="初末结构原子种类/顺序不一致, NEB 要求一致",
            )

        # 1. IDPP 初猜
        images = idpp_initial_path(
            init["positions"], final["positions"], args.n_images
        )

        # 2. NEB 优化
        energies: list[float] = []
        forces_history: list[float] = []

        atomic_numbers = init["atomic_numbers"]
        cell = init.get("cell")

        step = 0
        max_force = float("inf")
        converged = False
        while step < args.max_iter and max_force > args.tolerance:
            # 评估每个 image 的能量和力
            energies, forces = await eval_images(
                images, atomic_numbers, cell, args, context
            )

            # NEB 力投影
            neb_forces = compute_neb_forces(
                images, forces, energies, args.spring_constant, args.climbing_image
            )

            # 更新位置 (quick-min 风格: 用当前力做一步最速下降)
            # 简化版, 没做线搜索; 为了避免 LJ 等陡峭势导致原子重叠爆炸,
            # 加两道保险: (1) 力大就小步走 (2) 单原子单步位移硬上限 0.05 Å.
            # 真正的生产级 NEB 会用 FIRE / LBFGS, 这里够 demo / 教学用.
            dt = 0.05
            max_atom_step = 0.05  # Å, 单原子单步位移上限
            for i in range(1, args.n_images - 1):
                f_norm = np.linalg.norm(neb_forces[i])
                if f_norm < 1e-12:
                    continue
                step_size = min(dt, 0.5 / (f_norm + 1e-8))
                delta = step_size * neb_forces[i]
                # 每个原子的位移范数封顶, 防止力爆炸时位置也爆炸
                atom_norms = np.linalg.norm(delta, axis=1, keepdims=True)
                scale = np.where(
                    atom_norms > max_atom_step,
                    max_atom_step / (atom_norms + 1e-12),
                    1.0,
                )
                images[i] = images[i] + delta * scale

            max_force = float(
                max(
                    np.linalg.norm(neb_forces[i])
                    for i in range(1, args.n_images - 1)
                )
            )
            forces_history.append(max_force)
            step += 1

        converged = max_force <= args.tolerance

        # 重算一次能量, 确保输出对应当前位置
        energies, _ = await eval_images(
            images, atomic_numbers, cell, args, context
        )

        # 势垒 / 鞍点
        forward_barrier, reverse_barrier, saddle_idx = compute_barriers(
            energies
        )
        saddle_structure = {
            "atomic_numbers": atomic_numbers,
            "positions": images[saddle_idx].tolist(),
            "cell": cell.tolist() if cell is not None else None,
        }
        path_length = compute_path_length(images)

        return ToolResult(
            data={
                "action": "neb",
                "images": [
                    {
                        "atomic_numbers": atomic_numbers,
                        "positions": img.tolist(),
                        "cell": cell.tolist() if cell is not None else None,
                    }
                    for img in images
                ],
                "energies": [float(e) for e in energies],
                "forward_barrier": float(forward_barrier),
                "reverse_barrier": float(reverse_barrier),
                "saddle_point": {
                    "image_index": int(saddle_idx),
                    "energy": float(energies[saddle_idx]),
                    "structure": saddle_structure,
                },
                "path_length": float(path_length),
                "converged": bool(converged),
                "iterations": int(step),
                "final_max_force": float(max_force),
            },
            success=True,
        )

    # =====================================================================
    # 2. PES 扫描
    # =====================================================================

    async def _action_pes_scan(
        self, args: NEBToolInput, context: ToolContext
    ) -> ToolResult:
        """沿指定坐标轴网格扫描势能面."""
        struct = read_structure(args.structure)
        positions0 = np.asarray(struct["positions"], dtype=float)
        atomic_numbers = struct["atomic_numbers"]
        cell = struct.get("cell")

        scan_coords = args.scan_coords or []
        n_dims = len(scan_coords)
        if n_dims == 0:
            return ToolResult(
                data=None, success=False, error="scan_coords 不能为空"
            )

        lo, hi = args.scan_range[0], args.scan_range[1]
        grid_1d = np.linspace(lo, hi, args.n_points)
        # n 维网格
        mesh = np.meshgrid(*([grid_1d] * n_dims), indexing="ij")
        grid_shape = mesh[0].shape
        n_total = int(np.prod(grid_shape))
        energies = np.zeros(grid_shape)

        # 遍历每个网格点
        flat_idx = 0
        for idx in np.ndindex(*grid_shape):
            disp = np.zeros_like(positions0)
            for d, (atom_idx, axis) in enumerate(scan_coords):
                disp[atom_idx, axis] = mesh[d][idx]
            displaced = positions0 + disp
            e, _ = await eval_single(
                displaced, atomic_numbers, cell, args, context
            )
            energies[idx] = e
            flat_idx += 1

        # 找极小 / 鞍点 (网格上的局部极值)
        minima, saddles = find_extrema(energies, mesh, scan_coords)

        return ToolResult(
            data={
                "action": "pes_scan",
                "scan_coords": scan_coords,
                "scan_range": [float(lo), float(hi)],
                "n_points": args.n_points,
                "n_dims": n_dims,
                "grid_shape": list(grid_shape),
                "grid_axes": [grid_1d.tolist()] * n_dims,
                "energies": energies.tolist(),
                "contour_data": {
                    "x": mesh[0].tolist() if n_dims >= 1 else None,
                    "y": mesh[1].tolist() if n_dims >= 2 else None,
                    "z": energies.tolist(),
                },
                "minima": minima,
                "saddle_points": saddles,
                "n_evaluations": n_total,
            },
            success=True,
        )

    # =====================================================================
    # 3. 鞍点搜索 (Dimer)
    # =====================================================================

    async def _action_saddle_search(
        self, args: NEBToolInput, context: ToolContext
    ) -> ToolResult:
        """Dimer 方法找一阶鞍点.

        Dimer 由两个相近点 R ± d*τ 组成, 通过旋转 τ 找最低曲率方向
        (最负本征值对应的本征矢), 然后沿 τ 把 R 平移到鞍点.
        """
        if args.search_method != "dimer":
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"search_method={args.search_method} 暂未实现, "
                    "目前只支持 dimer"
                ),
            )

        struct = read_structure(args.structure)
        R = np.asarray(struct["positions"], dtype=float).flatten()
        atomic_numbers = struct["atomic_numbers"]
        cell = struct.get("cell")

        n = R.size
        rng = np.random.default_rng(42)
        # 初始模式: 随机单位矢量, 后续会被旋转校准
        tau = rng.standard_normal(n)
        tau /= np.linalg.norm(tau) + 1e-12

        d = args.dimer_length
        max_iter = args.max_iter
        converged = False
        step = 0
        last_eigval = 0.0

        while step < max_iter:
            # 1. 旋转阶段: 找最低曲率方向
            tau, curvature = await dimer_rotate(
                R, tau, d, atomic_numbers, cell, args, context,
                n_rot=args.dimer_rotations,
            )

            # 2. 平移阶段: 沿 tau 方向爬向鞍点
            # 在 R 处算力 F, 取其平行 tau 分量的反号 + 垂直分量
            e0, f0 = await eval_single(
                R.reshape(-1, 3), atomic_numbers, cell, args, context
            )
            F = f0.flatten()
            F_parallel = np.dot(F, tau) * tau
            F_perp = F - F_parallel
            # saddle 力: -F_parallel + F_perp (把平行分量翻转, 顺 tau 上坡)
            F_saddle = -F_parallel + F_perp

            step_size = min(0.05, 0.5 / (np.linalg.norm(F_saddle) + 1e-8))
            R = R + step_size * F_saddle / (np.linalg.norm(F_saddle) + 1e-12)

            last_eigval = float(curvature)
            if np.linalg.norm(F_saddle) < args.tolerance and curvature < 0:
                converged = True
                break
            step += 1

        # 最终结构 / 能量 / Hessian 沿 tau 的本征值
        e_final, _ = await eval_single(
            R.reshape(-1, 3), atomic_numbers, cell, args, context
        )
        hess_eigvals, hess_mode = await estimate_hessian_along_mode(
            R, tau, d, atomic_numbers, cell, args, context
        )

        return ToolResult(
            data={
                "action": "saddle_search",
                "search_method": "dimer",
                "saddle_structure": {
                    "atomic_numbers": atomic_numbers,
                    "positions": R.reshape(-1, 3).tolist(),
                    "cell": cell.tolist() if cell is not None else None,
                },
                "energy": float(e_final),
                "hessian_eigenvalues": hess_eigvals,
                "mode": hess_mode.tolist(),
                "lowest_curvature": last_eigval,
                "converged": bool(converged),
                "iterations": int(step),
            },
            success=True,
        )

    # =====================================================================
    # 4. MEP 分析
    # =====================================================================

    def _action_mep_analyze(self, args: NEBToolInput) -> ToolResult:
        """分析 NEB 给出的最小能量路径."""
        neb = args.neb_result or {}
        energies = neb.get("energies") or []
        if not energies:
            return ToolResult(
                data=None, success=False, error="neb_result.energies 为空"
            )

        e_arr = np.asarray(energies, dtype=float)
        n = len(e_arr)
        # 反应坐标: 累计路径长度 (用 images 位置算, 没有就退化为均匀)
        images = neb.get("images") or []
        if images and len(images) == n:
            coords = [np.asarray(im["positions"]) for im in images]
            cum = [0.0]
            for i in range(1, n):
                cum.append(cum[-1] + float(np.linalg.norm(coords[i] - coords[i - 1])))
            reaction_coord = np.asarray(cum)
            total_len = reaction_coord[-1] if reaction_coord.size else 0.0
        else:
            reaction_coord = np.linspace(0.0, 1.0, n)
            total_len = float(1.0)

        # 势垒
        e_init = float(e_arr[0])
        e_final = float(e_arr[-1])
        e_max = float(np.max(e_arr))
        forward_barrier = e_max - e_init
        reverse_barrier = e_max - e_final
        thermodynamic_driving_force = e_init - e_final

        if args.analysis_type == "barrier":
            return ToolResult(
                data={
                    "action": "mep_analyze",
                    "analysis_type": "barrier",
                    "forward_barrier": forward_barrier,
                    "reverse_barrier": reverse_barrier,
                    "saddle_energy": e_max,
                    "initial_energy": e_init,
                    "final_energy": e_final,
                    "thermodynamic_driving_force": thermodynamic_driving_force,
                },
                success=True,
            )

        if args.analysis_type == "decomposition":
            # 把能量分解成: 化学势 (端点均值) + 路径起伏 + 端点偏置
            e_mean = 0.5 * (e_init + e_final)
            path_deviation = (e_arr - np.linspace(e_init, e_final, n)).tolist()
            return ToolResult(
                data={
                    "action": "mep_analyze",
                    "analysis_type": "decomposition",
                    "chemical_potential_mean": float(e_mean),
                    "path_deviation": path_deviation,
                    "initial_energy": e_init,
                    "final_energy": e_final,
                    "asymmetry": float(e_init - e_final),
                    "max_deviation": float(np.max(np.abs(path_deviation))),
                },
                success=True,
            )

        # 默认 energy_profile
        return ToolResult(
            data={
                "action": "mep_analyze",
                "analysis_type": "energy_profile",
                "reaction_coordinate": reaction_coord.tolist(),
                "energies": e_arr.tolist(),
                "path_length": float(total_len),
                "forward_barrier": forward_barrier,
                "reverse_barrier": reverse_barrier,
                "thermodynamic_driving_force": thermodynamic_driving_force,
                "saddle_index": int(np.argmax(e_arr)),
            },
            success=True,
        )

    # =====================================================================
    # 5. 地形拓扑
    # =====================================================================

    async def _action_landscape_topology(
        self, args: NEBToolInput, context: ToolContext
    ) -> ToolResult:
        """势能面地形拓扑摘要 — 调 tda_tool 或自己做 basin 分析."""
        pes = args.pes_data or {}
        energies = np.asarray(pes.get("energies") or [], dtype=float)
        if energies.size == 0:
            return ToolResult(
                data=None, success=False, error="pes_data.energies 为空"
            )

        if args.method == "tda":
            return await topology_via_tda(pes, energies, context)

        # basin_analysis: 把能量从最低到阈值分层, 数连通分量
        return basin_analysis(pes, energies)
