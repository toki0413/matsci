"""材料科学内置 hook — 覆盖主流仿真工具的收敛/终止/物理合理性检查.

通过 register_science_hooks(hook_manager) 注册, 复用现有的
HookContext + run_pre/run_post 机制, 不创建并行系统.

与 __init__.py 里的 AnomalyDetectionHook (登记异常) 互补:
  - AnomalyDetectionHook: 只登记, 不阻断
  - 这里的 hook: 可以 block (如 VASP 未收敛) 或 warn (如能量超范围),
    保护 agent 不基于坏数据继续走

覆盖的仿真工具族:
  - DFT:      VASP / QE / CP2K
  - 量子化学:  Gaussian / ORCA
  - MD:       LAMMPS / GROMACS
  - FEM/CFD:  Abaqus / COMSOL / Elmer / FEniCS / OpenFOAM
  - 物理检查:  弹性常数正定性 / 打包成功 / 能量合理范围
  - 通用:     输出文件存在性
"""

from __future__ import annotations

import logging
import os
from typing import Any

from huginn.hooks import POST_TOOL_USE, HookContext, HookManager

logger = logging.getLogger(__name__)


# ── 辅助函数 ─────────────────────────────────────────────────────


def _extract_text(ctx: HookContext) -> str:
    """从 ctx.result 提取小写文本, 用于关键词匹配.

    序列化后的结构: 成功 {"result": {...}}, 失败 {"error": "..."}.
    把 result 里的数据序列化成字符串再 lower, stdout/stderr 等子字段也能扫到.
    """
    result = ctx.result if isinstance(ctx.result, dict) else {}
    data = result.get("result", result)
    if isinstance(data, dict):
        import json

        try:
            return json.dumps(data, ensure_ascii=False, default=str).lower()
        except Exception:
            return str(data).lower()
    return str(data).lower()


def _result_data(ctx: HookContext) -> dict:
    """取 ctx.result 里的 result 数据 dict, 拿不到就返回空 dict."""
    result = ctx.result if isinstance(ctx.result, dict) else {}
    data = result.get("result", result)
    return data if isinstance(data, dict) else {}


def _block(ctx: HookContext, reason: str) -> HookContext:
    """设置 block 标记并返回 ctx."""
    ctx.metadata["blocked_by_hook"] = True
    ctx.metadata["block_reason"] = reason
    return ctx


def _warn(ctx: HookContext, warning: str) -> None:
    """加一条 warning 到 metadata, 不 block."""
    warnings = ctx.metadata.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
        ctx.metadata["warnings"] = warnings
    warnings.append(warning)


# ── VASP 收敛性检查 ─────────────────────────────────────────────


async def vasp_convergence_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 VASP 输出是否收敛. 不收敛时 block.

    block 方式: ctx.metadata['block_reason'] = '...', 上层的 adapter
    检查这个字段并替换输出为错误消息.
    """
    if ctx.tool_name != "vasp_tool":
        return None

    result = ctx.result if isinstance(ctx.result, dict) else {}
    text = ""
    if "result" in result and isinstance(result["result"], dict):
        data = result["result"]
        converged = data.get("converged")
        if converged is True:
            return None  # 明确收敛
        if converged is False:
            ctx.metadata["block_reason"] = (
                "VASP calculation did not converge. Consider increasing "
                "NSW (max ionic steps) or adjusting EDIFF/EDIFFG."
            )
            ctx.metadata["blocked_by_hook"] = True
            return ctx
        # converged 字段不存在, 从输出文本判断
        text = str(data)
    else:
        text = str(result)

    text_lower = text.lower()
    if "reached required accuracy" in text_lower:
        return None  # 收敛了
    if "aborting loop" in text_lower or "brions" in text_lower:
        ctx.metadata["block_reason"] = (
            "VASP ionic relaxation did not reach required accuracy. "
            "Consider increasing NSW or adjusting EDIFFG."
        )
        ctx.metadata["blocked_by_hook"] = True
        return ctx
    return None


# ── LAMMPS 稳定性检查 ───────────────────────────────────────────


async def lammps_stability_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 LAMMPS 是否丢原子或不稳定."""
    if ctx.tool_name != "lammps_tool":
        return None

    result = ctx.result if isinstance(ctx.result, dict) else {}
    text = str(result.get("result", result)).lower()

    if "lost atoms" in text:
        ctx.metadata["block_reason"] = (
            "LAMMPS simulation lost atoms — system became unstable. "
            "Consider reducing timestep or using a softer potential."
        )
        ctx.metadata["blocked_by_hook"] = True
        return ctx
    return None


# ── 结构合理性检查 ───────────────────────────────────────────────


async def structure_sanity_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查结构工具返回的原子间距是否物理合理."""
    if ctx.tool_name != "structure_tool":
        return None

    result = ctx.result if isinstance(ctx.result, dict) else {}
    data = result.get("result", {})
    if not isinstance(data, dict):
        return None

    positions = data.get("positions") or data.get("frac_positions")
    if not positions or not isinstance(positions, (list, tuple)) or len(positions) < 2:
        return None

    import math

    for i in range(len(positions)):
        for j in range(i + 1, min(i + 5, len(positions))):
            p1, p2 = positions[i], positions[j]
            if not (isinstance(p1, (list, tuple)) and len(p1) >= 3):
                continue
            dx = float(p1[0]) - float(p2[0])
            dy = float(p1[1]) - float(p2[1])
            dz = float(p1[2]) - float(p2[2])
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if 0 < dist < 0.5:
                ctx.metadata["block_reason"] = (
                    f"Structure sanity check: atoms {i} and {j} are "
                    f"only {dist:.2f} Å apart (min 0.5 Å). Check coordinates."
                )
                ctx.metadata["blocked_by_hook"] = True
                return ctx
    return None


# ── DFT 收敛检查族 ───────────────────────────────────────────────


async def qe_convergence_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 Quantum ESPRESSO SCF 是否收敛."""
    if ctx.tool_name != "qe_tool":
        return None

    text = _extract_text(ctx)
    if "convergence has been achieved" in text or "scf correction compared to forces" in text:
        return None
    if "not converged" in text or "too many iterations" in text:
        return _block(
            ctx,
            "QE SCF 未收敛. 考虑增加 electron_maxstep 或调整 mixing_beta.",
        )
    return None


async def cp2k_convergence_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 CP2K SCF 是否收敛."""
    if ctx.tool_name != "cp2k_tool":
        return None

    text = _extract_text(ctx)
    if "scf run converged" in text:
        return None
    if "scf not converged" in text:
        return _block(
            ctx,
            "CP2K SCF 未收敛. 考虑增加 max_scf 或调整 OT 方法参数.",
        )
    return None


# ── 量子化学终止检查族 ───────────────────────────────────────────


async def gaussian_termination_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 Gaussian 是否正常终止."""
    if ctx.tool_name != "gaussian_tool":
        return None

    text = _extract_text(ctx)
    if "normal termination" in text:
        return None
    if "error termination" in text or "l9999.exe" in text:
        return _block(
            ctx,
            "Gaussian 异常终止. 检查输入文件/基组/内存/收敛阈值设置.",
        )
    return None


async def orca_termination_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 ORCA 是否正常终止."""
    if ctx.tool_name != "orca_tool":
        return None

    text = _extract_text(ctx)
    if "orca terminated normally" in text:
        return None
    if "aborted" in text or "fatal error" in text:
        return _block(
            ctx,
            "ORCA 异常终止. 检查输入文件/基组/内存/SCF 设置.",
        )
    return None


# ── MD 稳定性检查族 ──────────────────────────────────────────────


async def gromacs_stability_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 GROMACS 是否丢原子或不稳定.

    "Finished mdrun" 表示正常完成; "Lost atoms" / "System too unstable" 则 block.
    """
    if ctx.tool_name != "gromacs_tool":
        return None

    text = _extract_text(ctx)
    if "lost atoms" in text or "system too unstable" in text:
        return _block(
            ctx,
            "GROMACS 模拟不稳定 (丢原子). 考虑减小时间步或软化势能参数.",
        )
    return None


# ── FEM/CFD 求解器检查族 ─────────────────────────────────────────

_FEM_TOOLS = frozenset({"abaqus_tool", "comsol_tool", "elmer_tool", "fenics_tool"})


async def fem_solver_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 FEM 求解器 (Abaqus/COMSOL/Elmer/FEniCS) 是否收敛."""
    if ctx.tool_name not in _FEM_TOOLS:
        return None

    text = _extract_text(ctx)
    if "not converged" in text or "singular matrix" in text or "too many iterations" in text:
        return _block(
            ctx,
            f"{ctx.tool_name} 求解未收敛. 检查网格质量/边界条件/材料参数/载荷步长.",
        )
    if "convergence" in text or "solution completed" in text:
        return None
    return None


async def openfoam_residual_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 OpenFOAM 是否正常结束或出现致命错误."""
    if ctx.tool_name != "openfoam_tool":
        return None

    text = _extract_text(ctx)
    if "floating point exception" in text or "foam fatal error" in text:
        return _block(
            ctx,
            "OpenFOAM 求解异常. 检查网格质量/边界条件/时间步长/离散格式.",
        )
    return None


# ── 物理合理性检查族 ─────────────────────────────────────────────

_ELASTIC_KEYS = (
    "elastic_constants",
    "stiffness_matrix",
    "elastic_matrix",
    "c_matrix",
    "stiffness",
)


async def mechanical_property_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查弹性常数矩阵是否正定 (对角元素全正).

    只在结果包含弹性常数矩阵时检查; 没有矩阵就不处理.
    """
    if ctx.tool_name != "mechanical_tool":
        return None

    data = _result_data(ctx)
    if not data:
        return None

    # 找弹性常数矩阵 (list of lists)
    matrix = None
    for key in _ELASTIC_KEYS:
        val = data.get(key)
        if isinstance(val, (list, tuple)) and val and isinstance(val[0], (list, tuple)):
            matrix = val
            break

    if matrix is None:
        return None  # 没有弹性常数矩阵, 不检查

    # 检查对角元素是否全正
    for i, row in enumerate(matrix):
        if not isinstance(row, (list, tuple)) or i >= len(row):
            continue
        try:
            diag = float(row[i])
        except (TypeError, ValueError, IndexError):
            continue
        if diag <= 0:
            return _block(
                ctx,
                f"弹性常数矩阵非正定: C[{i}][{i}] = {diag:.4f} <= 0. "
                "检查计算参数或结构对称性.",
            )
    return None


async def packing_success_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 Packmol/packing 是否成功打包."""
    if ctx.tool_name != "packing_tool":
        return None

    text = _extract_text(ctx)
    if "overlap" in text or "failed to pack" in text:
        return _block(
            ctx,
            "Packmol 打包失败 (原子重叠或无法打包). 调整盒子尺寸或分子数量.",
        )
    if "success" in text:
        return None
    return None


# 能量扫描用的原子数字段名 (小写匹配)
_ATOM_COUNT_KEYS = frozenset(
    {"n_atoms", "natoms", "num_atoms", "atom_count", "total_atoms", "natom"}
)


def _scan_energy_fields(data: Any, found: dict) -> None:
    """递归扫描 data, 收集能量值和原子数到 found dict.

    found 结构: {"total": [float...], "per_atom": [float...], "n_atoms": float}
    """
    if isinstance(data, dict):
        for key, val in data.items():
            kl = key.lower()
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                if "energy" in kl and "per_atom" in kl:
                    found["per_atom"].append(float(val))
                elif "energy" in kl:
                    found["total"].append(float(val))
                elif kl in _ATOM_COUNT_KEYS:
                    found["n_atoms"] = float(val)
            elif isinstance(val, (dict, list)):
                _scan_energy_fields(val, found)
    elif isinstance(data, list):
        for item in data:
            _scan_energy_fields(item, found)


async def energy_bound_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查能量值是否在物理合理范围内 (所有工具, 只 warn 不 block).

    判据:
      - 总能量 < 0 (束缚态应为负值), 非负则 warn
      - 单原子能量在 -1e4 到 1e4 eV 之间, 超出则 warn
    """
    data = _result_data(ctx)
    if not data:
        return None

    found: dict = {"total": [], "per_atom": [], "n_atoms": 0.0}
    _scan_energy_fields(data, found)

    if not found["total"] and not found["per_atom"]:
        return None  # 没有能量值, 不处理

    total_energy = found["total"][0] if found["total"] else None
    per_atom = found["per_atom"][0] if found["per_atom"] else None

    # 有总能量和原子数但没 per_atom, 自己算一下
    if per_atom is None and total_energy is not None and found["n_atoms"] > 0:
        per_atom = total_energy / found["n_atoms"]

    if total_energy is not None and total_energy >= 0:
        _warn(
            ctx,
            f"总能量 {total_energy:.4f} eV 非负, 物理上束缚态总能量应为负值.",
        )

    if per_atom is not None and (per_atom < -1e4 or per_atom > 1e4):
        _warn(
            ctx,
            f"单原子能量 {per_atom:.4f} eV/atom 超出物理合理范围 [-1e4, 1e4] eV.",
        )

    return None


# ── 文件存在性验证 ───────────────────────────────────────────────


def _extract_output_paths(data: Any) -> list[str]:
    """从结果数据里提取声明的输出文件路径, 只取绝对路径.

    匹配字段名含 "output" 且含 "file"/"path" 的键, 以及 "output_files"
    (可能是 list 或 dict). 相对路径/纯文件名无法可靠校验, 跳过.
    """
    paths: list[str] = []
    if isinstance(data, dict):
        for key, val in data.items():
            kl = key.lower()
            if kl == "output_files":
                if isinstance(val, list):
                    paths.extend(v for v in val if isinstance(v, str) and os.path.isabs(v))
                elif isinstance(val, dict):
                    paths.extend(
                        v for v in val.values() if isinstance(v, str) and os.path.isabs(v)
                    )
            elif "output" in kl and ("file" in kl or "path" in kl):
                if isinstance(val, str) and os.path.isabs(val):
                    paths.append(val)
                elif isinstance(val, list):
                    paths.extend(
                        v for v in val if isinstance(v, str) and os.path.isabs(v)
                    )
            elif isinstance(val, (dict, list)):
                paths.extend(_extract_output_paths(val))
    elif isinstance(data, list):
        for item in data:
            paths.extend(_extract_output_paths(item))
    return paths


async def output_file_existence_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 验证工具声称产出的文件是否真的存在 (所有工具)."""
    data = _result_data(ctx)
    if not data:
        return None

    paths = _extract_output_paths(data)
    if not paths:
        return None

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        preview = "; ".join(missing[:5])
        return _block(ctx, f"工具声称产出的文件不存在: {preview}")
    return None


# ── 注册函数 ─────────────────────────────────────────────────────


def register_science_hooks(hm: HookManager) -> None:
    """把材料科学 hook 注册到 HookManager. 在 agent 初始化时调用.

    幂等: 检查是否已注册, 避免重复.
    """
    if getattr(hm, "_science_hooks_registered", False):
        return

    # 原有三个 hook
    hm.register(POST_TOOL_USE, vasp_convergence_hook)
    hm.register(POST_TOOL_USE, lammps_stability_hook)
    hm.register(POST_TOOL_USE, structure_sanity_hook)
    # DFT 收敛检查族
    hm.register(POST_TOOL_USE, qe_convergence_hook)
    hm.register(POST_TOOL_USE, cp2k_convergence_hook)
    # 量子化学终止检查族
    hm.register(POST_TOOL_USE, gaussian_termination_hook)
    hm.register(POST_TOOL_USE, orca_termination_hook)
    # MD 稳定性检查族
    hm.register(POST_TOOL_USE, gromacs_stability_hook)
    # FEM/CFD 求解器检查族
    hm.register(POST_TOOL_USE, fem_solver_hook)
    hm.register(POST_TOOL_USE, openfoam_residual_hook)
    # 物理合理性检查族
    hm.register(POST_TOOL_USE, mechanical_property_hook)
    hm.register(POST_TOOL_USE, packing_success_hook)
    hm.register(POST_TOOL_USE, energy_bound_hook)
    # 文件存在性验证
    hm.register(POST_TOOL_USE, output_file_existence_hook)

    # 事件驱动管线 hook — 成功完成后建议下一步
    try:
        from huginn.provenance import pipeline_hook
        hm.register(POST_TOOL_USE, pipeline_hook)
    except ImportError:
        logger.debug("pipeline_hook not available (non-fatal)")

    hm._science_hooks_registered = True
    logger.info(
        "Science hooks registered: vasp_convergence, lammps_stability, "
        "structure_sanity, qe_convergence, cp2k_convergence, "
        "gaussian_termination, orca_termination, gromacs_stability, "
        "fem_solver, openfoam_residual, mechanical_property, "
        "packing_success, energy_bound, output_file_existence, pipeline"
    )
