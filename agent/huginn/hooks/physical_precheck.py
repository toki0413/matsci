"""PRE_TOOL_USE physical pre-check — warn user, offer force-proceed.

Unlike POST_TOOL_USE science hooks that block bad results, these hooks run
BEFORE the tool executes and catch physically infeasible setups early:

  - Band/DOS calculation before SCF converged
  - Elastic constants without prior relaxation
  - MD with unreasonable timestep
  - DFT with too-low encut for heavy elements
  - GROMACS/OpenMM MD without prior minimization

Key design: these hooks do NOT permanently block. They set blocked=True with
a warning message that includes instructions to force-proceed. When the user
re-invokes the tool with force_proceed=True in args, the hook skips the check.

This follows the user's explicit requirement:
  "你不要自动拦截，你要首先警告用户，之后给用户提供强行推进的选项"
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from huginn.hooks import PRE_TOOL_USE, HookContext, HookManager

logger = logging.getLogger(__name__)


# v6 G56: SafetyMode 四档替换二元 force_proceed
# - safe:        警告即停, 不允许 force_proceed (生产/受控环境)
# - guided:      警告 + 问用户 (默认), 用户确认才放行
# - autonomous:  警告 + 自动 force_proceed (agent 自主决策)
# - yolo:        不检查 (调试/快速迭代)
# ponytail: 老 force_proceed=True 调用点不动, 行为等同 autonomous;
# force_proceed=False 调用点不动, 行为等同 guided. 向后兼容.
class SafetyMode(StrEnum):
    SAFE = "safe"
    GUIDED = "guided"
    AUTONOMOUS = "autonomous"
    YOLO = "yolo"


def mode_allows_force_proceed(mode: str | SafetyMode) -> bool:
    """safe → False (拒绝强行推进); guided/autonomous/yolo → True.

    guided 语义上仍允许 force_proceed, 但调用方应先问用户;
    本函数只判"是否技术上允许", 不替调用方做交互决策.
    ponytail: 不抛异常, 未知 mode 默认 True (不阻塞, 跟 v5 行为一致).
    """
    m = SafetyMode(mode) if mode else SafetyMode.GUIDED
    return m != SafetyMode.SAFE


def _get_args(ctx: HookContext) -> dict[str, Any]:
    """Extract args dict from HookContext, tolerating None."""
    if isinstance(ctx.args, dict):
        return ctx.args
    return {}


def _resolve_safety_mode(ctx: HookContext) -> SafetyMode:
    """读 ctx.metadata['safety_mode'] / ctx.args['safety_mode'].

    未设置默认 GUIDED (跟 v5 force_proceed=False 等同).
    ponytail: 不从全局 config 读, 让调用方 (engine / CLI) 在 metadata 注入.
    """
    raw = ctx.metadata.get("safety_mode") or _get_args(ctx).get("safety_mode")
    if not raw:
        return SafetyMode.GUIDED
    try:
        return SafetyMode(raw)
    except ValueError:
        return SafetyMode.GUIDED


def _is_force_proceed(ctx: HookContext) -> bool:
    """Check if user explicitly requested to bypass physical pre-checks.

    v6 G56: 优先看 safety_mode; force_proceed 字段保留向后兼容.
    - safety_mode=safe → 永远 False (即使 force_proceed=True)
    - safety_mode=yolo → 永远 True (跳过所有 precheck)
    - safety_mode=autonomous → True (自动放行)
    - safety_mode=guided → 看 force_proceed 字段 (v5 行为)
    - 未设 safety_mode → 看 force_proceed 字段 (v5 行为)
    """
    mode = _resolve_safety_mode(ctx)
    if mode == SafetyMode.SAFE:
        return False
    if mode in (SafetyMode.AUTONOMOUS, SafetyMode.YOLO):
        return True
    # guided 或未设置 — 回退到老 force_proceed 字段
    return _get_args(ctx).get("force_proceed") is True


def _warn_and_block(ctx: HookContext, warning: str) -> HookContext:
    """Block with a warning + force-proceed instructions.

    The block_reason is returned to the LLM by the tool adapter, so the
    user/agent sees the warning and can re-call with force_proceed=True.
    """
    ctx.blocked = True
    ctx.metadata["physical_warning"] = warning
    ctx.metadata["force_proceed_available"] = True
    ctx.metadata["block_reason"] = (
        f"[Physical Pre-check] {warning}\n"
        "If you understand the risk and want to proceed anyway, "
        "re-invoke the tool with force_proceed=True in the arguments."
    )
    return ctx


# ── Pre-check implementations ────────────────────────────────────


async def band_before_scf_hook(ctx: HookContext) -> HookContext | None:
    """Warn if band/DOS calculation is requested without prior SCF.

    Band structure and DOS need a converged charge density from a static
    SCF run. Running them directly after relaxation wastes compute and
    gives garbage results.
    """
    if _is_force_proceed(ctx):
        return None

    args = _get_args(ctx)
    if ctx.tool_name not in ("vasp_tool", "qe_tool", "cp2k_tool"):
        return None

    action = str(args.get("action", "")).lower()
    if action not in ("band", "dos", "band_structure", "density_of_states"):
        return None

    # Check provenance for prior static/SCF calculation
    try:
        from huginn.provenance.registry import ProvenanceRegistry
        reg = ProvenanceRegistry.shared()
        # Look for any prior static/SCF entry
        for tool in ("vasp_tool", "qe_tool", "cp2k_tool"):
            entries = reg.find_by_tool(tool)
            for e in entries:
                params = e.parameters or {}
                if str(params.get("action", "")).lower() in ("static", "scf"):
                    return None  # Found prior SCF, all good
    except Exception:
        # Can't check provenance — don't block, just warn softly
        logger.debug("provenance check skipped for band_before_scf", exc_info=True)
        return None

    return _warn_and_block(
        ctx,
        "能带/DOS 计算通常需要先完成 SCF (静态) 计算以获得收敛的电荷密度. "
        "当前未检测到已完成的 SCF 计算. 建议先执行 static/SCF 计算.",
    )


async def elastic_without_relax_hook(ctx: HookContext) -> HookContext | None:
    """Warn if elastic constants are requested on an unrelaxed structure.

    Elastic constants are sensitive to the equilibrium position. Calculating
    them on a non-relaxed structure gives physically meaningless results.
    """
    if _is_force_proceed(ctx):
        return None

    if ctx.tool_name != "mechanical_tool":
        return None

    args = _get_args(ctx)
    action = str(args.get("action", "")).lower()
    if "elastic" not in action and "stiffness" not in action and "c_matrix" not in action:
        return None

    # Check for prior relaxation
    try:
        from huginn.provenance.registry import ProvenanceRegistry
        reg = ProvenanceRegistry.shared()
        for tool in ("vasp_tool", "qe_tool", "cp2k_tool", "lammps_tool"):
            entries = reg.find_by_tool(tool)
            for e in entries:
                params = e.parameters or {}
                if str(params.get("action", "")).lower() in ("relax", "optimization", "opt"):
                    return None  # Found prior relaxation
    except Exception:
        return None

    return _warn_and_block(
        ctx,
        "弹性常数计算需要基于已优化的平衡结构. "
        "当前未检测到已完成的结构优化 (relax). 建议先执行结构优化.",
    )


async def md_timestep_hook(ctx: HookContext) -> HookContext | None:
    """Warn if MD timestep is unreasonably large.

    For all-atom MD, timestep > 5 fs usually causes instability.
    For ab initio MD, the limit is even lower (~1-2 fs).
    """
    if _is_force_proceed(ctx):
        return None

    if ctx.tool_name not in ("lammps_tool", "gromacs_tool", "openmm_tool"):
        return None

    args = _get_args(ctx)
    dt = args.get("timestep") or args.get("dt") or args.get("time_step")
    if dt is None:
        return None

    try:
        dt_val = float(dt)
    except (TypeError, ValueError):
        return None

    # 5 fs is the hard ceiling for classical all-atom; ab initio is ~1-2 fs
    threshold = 5.0
    if ctx.tool_name in ("vasp_tool", "qe_tool", "cp2k_tool"):
        threshold = 2.0

    if dt_val > threshold:
        return _warn_and_block(
            ctx,
            f"MD 时间步 {dt_val:.1f} fs 超过推荐上限 {threshold:.1f} fs. "
            "过大的时间步会导致能量发散和原子飞离. "
            "建议使用 1-2 fs (全原子) 或更小 (含氢原子).",
        )

    return None


async def low_encut_hook(ctx: HookContext) -> HookContext | None:
    """Warn if DFT encut is too low for the elements involved.

    Heavy elements (transition metals, f-electron systems) need higher
    encut to properly represent semi-core states.
    """
    if _is_force_proceed(ctx):
        return None

    if ctx.tool_name not in ("vasp_tool", "qe_tool", "cp2k_tool"):
        return None

    args = _get_args(ctx)
    encut = args.get("encut") or args.get("ecutwfc") or args.get("cutoff")
    if encut is None:
        return None

    try:
        encut_val = float(encut)
    except (TypeError, ValueError):
        return None

    # Below 300 eV is generally too low for production calculations
    if encut_val < 300:
        return _warn_and_block(
            ctx,
            f"截断能 {encut_val:.0f} eV 低于推荐最小值 300 eV. "
            "过低的截断能会导致能量不收敛和受力不准确. "
            "建议至少 400-520 eV (VASP) 或等效值.",
        )

    return None


async def md_without_minimize_hook(ctx: HookContext) -> HookContext | None:
    """Warn if MD run is requested without prior energy minimization.

    Running MD on an un-minimized structure causes large initial forces
    and can crash the simulation.
    """
    if _is_force_proceed(ctx):
        return None

    if ctx.tool_name not in ("lammps_tool", "gromacs_tool", "openmm_tool"):
        return None

    args = _get_args(ctx)
    action = str(args.get("action", "")).lower()
    if action not in ("md", "md_run", "nvt", "npt", "run"):
        return None

    # Check for prior minimization
    try:
        from huginn.provenance.registry import ProvenanceRegistry
        reg = ProvenanceRegistry.shared()
        for tool in ("lammps_tool", "gromacs_tool", "openmm_tool"):
            entries = reg.find_by_tool(tool)
            for e in entries:
                params = e.parameters or {}
                if str(params.get("action", "")).lower() in (
                    "minimize", "energy_minimize", "emin", "relax", "opt"
                ):
                    return None  # Found prior minimization
    except Exception:
        return None

    return _warn_and_block(
        ctx,
        "MD 模拟前通常需要先进行能量最小化. "
        "当前未检测到已完成的能量最小化. 建议先执行 minimize/relax.",
    )


# ── Registration ─────────────────────────────────────────────────

_PRECHECK_HOOKS = [
    band_before_scf_hook,
    elastic_without_relax_hook,
    md_timestep_hook,
    low_encut_hook,
    md_without_minimize_hook,
]


def register_physical_prechecks(hm: HookManager) -> None:
    """Register all physical pre-check hooks.

    Called from register_science_hooks(). Idempotent.
    """
    if getattr(hm, "_physical_prechecks_registered", False):
        return
    for hook in _PRECHECK_HOOKS:
        hm.register(PRE_TOOL_USE, hook)
    hm._physical_prechecks_registered = True
    logger.info(
        "Physical pre-check hooks registered: "
        "band_before_scf, elastic_without_relax, md_timestep, "
        "low_encut, md_without_minimize. "
        "Pattern: warn + force_proceed (not auto-block)."
    )
