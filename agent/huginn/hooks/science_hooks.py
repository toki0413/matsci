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
    """设置 block 标记并返回 ctx. severity=blocking."""
    ctx.metadata["blocked_by_hook"] = True
    ctx.metadata["block_reason"] = reason
    ctx.metadata["severity"] = "blocking"
    return ctx


def _warn(ctx: HookContext, warning: str, severity: str = "minor") -> None:
    """加一条 warning 到 metadata, 不 block.

    severity 按 OpenScience provenance_review 的四级:
      - blocking: 严重到应停止 (用 _block 代替)
      - major:    影响结果可靠性, agent 应考虑修正
      - minor:    值得注意但通常不影响结论
      - info:     纯信息性, 记录但不需行动
    """
    warnings = ctx.metadata.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
        ctx.metadata["warnings"] = warnings
    warnings.append({"severity": severity, "message": warning})


# 声明式 hook 规则表 — 替代 9 个结构相同的模式匹配函数
# ponytail: rule table, switch to per-tool class if hook logic diverges
_PATTERN_HOOK_RULES: list[dict[str, Any]] = [
    {
        "tools": ["qe_tool"],
        "success_keywords": ["convergence has been achieved", "scf correction compared to forces"],
        "fail_keywords": ["not converged", "too many iterations"],
        "block_msg": "QE SCF 未收敛. 考虑增加 electron_maxstep 或调整 mixing_beta.",
    },
    {
        "tools": ["cp2k_tool"],
        "success_keywords": ["scf run converged"],
        "fail_keywords": ["scf not converged"],
        "block_msg": "CP2K SCF 未收敛. 考虑增加 max_scf 或调整 OT 方法参数.",
    },
    {
        "tools": ["gaussian_tool"],
        "success_keywords": ["normal termination"],
        "fail_keywords": ["error termination", "l9999.exe"],
        "block_msg": "Gaussian 异常终止. 检查输入文件/基组/内存/收敛阈值设置.",
    },
    {
        "tools": ["orca_tool"],
        "success_keywords": ["orca terminated normally"],
        "fail_keywords": ["aborted", "fatal error"],
        "block_msg": "ORCA 异常终止. 检查输入文件/基组/内存/SCF 设置.",
    },
    {
        "tools": ["gromacs_tool"],
        "success_keywords": [],
        "fail_keywords": ["lost atoms", "system too unstable"],
        "block_msg": "GROMACS 模拟不稳定 (丢原子). 考虑减小时间步或软化势能参数.",
    },
    {
        "tools": ["abaqus_tool", "comsol_tool", "elmer_tool", "fenics_tool"],
        "success_keywords": ["convergence", "solution completed"],
        "fail_keywords": ["not converged", "singular matrix", "too many iterations"],
        "block_msg": "{tool} 求解未收敛. 检查网格质量/边界条件/材料参数/载荷步长.",
    },
    {
        "tools": ["openfoam_tool"],
        "success_keywords": [],
        "fail_keywords": ["floating point exception", "foam fatal error"],
        "block_msg": "OpenFOAM 求解异常. 检查网格质量/边界条件/时间步长/离散格式.",
    },
    {
        "tools": ["packing_tool"],
        "success_keywords": ["success"],
        "fail_keywords": ["overlap", "failed to pack"],
        "block_msg": "Packmol 打包失败 (原子重叠或无法打包). 调整盒子尺寸或分子数量.",
    },
    {
        "tools": ["lammps_tool"],
        "success_keywords": [],
        "fail_keywords": ["lost atoms"],
        "block_msg": (
            "LAMMPS simulation lost atoms — system became unstable. "
            "Consider reducing timestep or using a softer potential."
        ),
    },
]


def _make_pattern_hook(rule: dict[str, Any]):
    """根据规则 dict 生成一个 async POST_TOOL_USE hook.

    逻辑: tool_name 匹配 → 提取文本 → 命中 fail 关键词则 block,
    否则命中 success 关键词则放行. fail 优先于 success (避免 "not converged"
    误匹配 success 的 "convergence" 子串).
    """
    tools = frozenset(rule["tools"])
    success_kw = rule["success_keywords"]
    fail_kw = rule["fail_keywords"]
    raw_msg = rule["block_msg"]

    async def _hook(ctx: HookContext) -> HookContext | None:
        if ctx.tool_name not in tools:
            return None
        text = _extract_text(ctx)
        for kw in fail_kw:
            if kw in text:
                msg = raw_msg.format(tool=ctx.tool_name) if "{tool}" in raw_msg else raw_msg
                return _block(ctx, msg)
        for kw in success_kw:
            if kw in text:
                return None
        return None

    return _hook


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


# ── 生物医药检查族 ───────────────────────────────────────────────


async def vina_docking_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 AutoDock Vina 对接结果的合理性.

    - 对接无结果时 block
    - 最佳亲和力 > 0 kcal/mol 时 block (物理上不合理)
    - 亲和力过弱 (> -3) 时 warn (几乎无结合)
    """
    if ctx.tool_name != "vina_tool":
        return None

    data = _result_data(ctx)
    if not data or data.get("action") != "dock":
        return None

    poses = data.get("poses", [])
    if not poses:
        return _block(ctx, "Vina 对接未产生任何结合构象. 检查格点盒子是否覆盖活性位点.")

    best = data.get("best_affinity")
    if best is not None:
        if best > 0:
            return _block(
                ctx,
                f"对接亲和力 {best:.2f} kcal/mol 为正值, 物理上不合理. 检查受体/配体准备.",
            )
        if best > -3.0:
            _warn(ctx, f"最佳亲和力 {best:.2f} kcal/mol 偏弱, 可能无显著结合.")

    return None


async def openmm_stability_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 OpenMM MD 运行稳定性.

    - 温度严重偏离设定值时 warn
    - 能量为 NaN/Inf 时 block
    - 能量不降反升 (minimization) 时 warn
    """
    if ctx.tool_name != "openmm_tool":
        return None

    data = _result_data(ctx)
    if not data:
        return None

    text = _extract_text(ctx)
    if "nan" in text or "inf" in text:
        return _block(ctx, "OpenMM 输出包含 NaN/Inf, 模拟可能发散. 检查时间步长和约束设置.")

    action = data.get("action")

    # Minimization: energy should decrease
    if action == "energy_minimize":
        e0 = data.get("initial_energy_kj_mol")
        ef = data.get("final_energy_kj_mol")
        if e0 is not None and ef is not None and ef > e0:
            _warn(ctx, f"能量最小化后能量升高 ({e0:.1f} → {ef:.1f} kJ/mol), 可能未收敛.")

    # MD run: check temperature drift
    if action == "md_run":
        series = data.get("time_series", {})
        temps = series.get("temperatures", [])
        target = data.get("temperature_k", 300.0)
        if temps:
            mean_t = sum(temps) / len(temps)
            if abs(mean_t - target) > 50:
                _warn(
                    ctx,
                    f"平均温度 {mean_t:.1f}K 偏离设定值 {target:.1f}K 超过 50K. "
                    "检查热浴耦合强度或时间步长.",
                )

    return None


# ── 跨学科工具检查族 (loop engineering 补全) ─────────────────────


async def fep_validation_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查自由能计算结果的合理性."""
    if ctx.tool_name != "fep_tool":
        return None

    data = _result_data(ctx)
    if not data:
        return None

    text = _extract_text(ctx)
    if "nan" in text or "inf" in text:
        return _block(ctx, "FEP 输出包含 NaN/Inf, 自由能计算可能发散.")

    # 自由能差值过大时 warn (超过 100 kcal/mol 物理上可疑)
    dG = data.get("delta_g_kcal_mol")
    if dG is not None and abs(dG) > 100:
        _warn(ctx, f"自由能变化 {dG:.2f} kcal/mol 过大, 检查 lambda 窗口和耦合参数.")

    # TI/FEP 需要 lambda 窗口, 太少不靠谱
    lambdas = data.get("lambdas", [])
    if isinstance(lambdas, list) and len(lambdas) < 3:
        _warn(ctx, f"仅 {len(lambdas)} 个 lambda 窗口, 建议至少 5-10 个提高精度.")

    return None


async def enhanced_sampling_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查增强采样结果的合理性."""
    if ctx.tool_name != "enhanced_sampling_tool":
        return None

    data = _result_data(ctx)
    if not data:
        return None

    text = _extract_text(ctx)
    if "nan" in text or "inf" in text:
        return _block(ctx, "增强采样输出包含 NaN/Inf, FES 重构可能失败.")

    # WHAM 未收敛时 warn
    converged = data.get("converged")
    if converged is False:
        _warn(ctx, "WHAM 迭代未收敛, FES 可能不准确. 考虑增加 bin 数或迭代次数.", "major")

    # FES 自由能范围过大时 warn
    fes_min = data.get("fes_min")
    fes_max = data.get("fes_max")
    if fes_min is not None and fes_max is not None:
        rng = abs(fes_max - fes_min)
        if rng > 200:
            _warn(ctx, f"FES 自由能范围 {rng:.1f} kcal/mol 过大, 检查 CV 选择和采样充分性.")

    return None


async def msm_validation_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 MSM 构建的有效性."""
    if ctx.tool_name != "msm_tool":
        return None

    data = _result_data(ctx)
    if not data:
        return None

    text = _extract_text(ctx)
    if "nan" in text or "inf" in text:
        return _block(ctx, "MSM 输出包含 NaN/Inf, 转移矩阵可能无效.")

    # 转移矩阵行和应接近 1 (随机矩阵性质)
    t_matrix = data.get("transition_matrix")
    if isinstance(t_matrix, list) and t_matrix:
        for i, row in enumerate(t_matrix):
            if isinstance(row, list) and row:
                row_sum = sum(row)
                if row_sum < 0.9 or row_sum > 1.1:
                    _warn(ctx, f"转移矩阵第 {i} 行和为 {row_sum:.4f}, 偏离 1.0, 检查微扰态划分.", "major")
                    break

    # 隐含弛豫时间过短说明微扰态划分太粗
    timescales = data.get("implied_timescales", [])
    if isinstance(timescales, list) and timescales:
        max_ts = max(timescales) if timescales else 0
        if max_ts < 2:
            _warn(ctx, f"最大隐含弛豫时间 {max_ts:.1f} 步, 微扰态划分可能过粗.")

    return None


async def inverse_design_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查逆向设计优化结果."""
    if ctx.tool_name != "inverse_design_tool":
        return None

    data = _result_data(ctx)
    if not data:
        return None

    # Pareto 前沿为空时 block
    pareto = data.get("pareto_frontier", data.get("pareto_indices"))
    if pareto is not None and isinstance(pareto, list) and len(pareto) == 0:
        return _block(ctx, "Pareto 前沿为空, 可能所有解都被支配. 检查目标函数方向.")

    # 优化没改善时 warn
    best = data.get("best_score")
    initial = data.get("initial_score")
    if best is not None and initial is not None and best >= initial:
        _warn(ctx, f"优化后最佳得分 {best:.4f} 未优于初始 {initial:.4f}, 可能陷入局部最优.", "major")

    return None


async def motif_mining_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查结构 motif 挖掘结果."""
    if ctx.tool_name != "motif_mining_tool":
        return None

    data = _result_data(ctx)
    if not data:
        return None

    motifs = data.get("motifs", data.get("found_motifs", []))
    if isinstance(motifs, list) and len(motifs) == 0:
        _warn(ctx, "未检测到任何结构 motif, 检查截断半径或输入结构.")

    return None


async def consensus_scoring_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查共识打分结果的一致性."""
    if ctx.tool_name != "consensus_scoring_tool":
        return None

    data = _result_data(ctx)
    if not data:
        return None

    # 排名列表为空时 block
    ranked = data.get("ranked_items", data.get("ranking", []))
    if isinstance(ranked, list) and len(ranked) == 0:
        return _block(ctx, "共识打分结果为空, 检查输入分数列表.")

    # 排名一致性 (Kendall W 太低说明各模型分歧大)
    w = data.get("kendall_w")
    if w is not None and isinstance(w, (int, float)) and w < 0.3:
        _warn(ctx, f"Kendall W = {w:.3f} 偏低, 各打分模型分歧较大, 共识排名可靠性有限.")

    return None


async def rdkit_validation_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 RDKit 分子操作的有效性."""
    if ctx.tool_name != "rdkit_tool":
        return None

    data = _result_data(ctx)
    if not data:
        return None

    # SMILES 解析失败
    if data.get("valid") is False or data.get("error"):
        return _block(ctx, f"RDKit 分子解析失败: {data.get('error', '未知错误')}")

    # 分子描述符异常
    mw = data.get("molecular_weight")
    if mw is not None and isinstance(mw, (int, float)):
        if mw > 900:
            _warn(ctx, f"分子量 {mw:.1f} Da 超过 900, 可能不适合类药物性 (Lipinski 规则).")
        if mw < 30:
            _warn(ctx, f"分子量 {mw:.1f} Da 过小, 可能不是有效的药物分子.")

    return None


async def neb_convergence_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 NEB 路径收敛性."""
    if ctx.tool_name != "neb_tool":
        return None

    data = _result_data(ctx)
    if not data:
        return None

    converged = data.get("converged")
    if converged is False:
        _warn(ctx, "NEB 路径未收敛, 考虑增加最大迭代次数或调整弹簧常数.", "major")

    # 能垒为负说明起点比终点高, 可能路径方向反了
    barrier = data.get("barrier_ev")
    if barrier is not None and isinstance(barrier, (int, float)) and barrier < 0:
        _warn(ctx, f"NEB 能垒 {barrier:.4f} eV 为负值, 检查初末态方向是否正确.")

    return None


async def gp_model_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 GP 模型拟合质量."""
    if ctx.tool_name != "gp_tool":
        return None

    data = _result_data(ctx)
    if not data:
        return None

    # R² 过低说明模型拟合差
    r2 = data.get("r2_score")
    if r2 is not None and isinstance(r2, (int, float)) and r2 < 0.5:
        _warn(ctx, f"GP 模型 R² = {r2:.4f} 偏低, 考虑增加核函数复杂度或训练数据.", "major")

    # 超参数异常 (length_scale 过大说明模型没学到任何结构)
    ls = data.get("length_scale")
    if ls is not None and isinstance(ls, (int, float)) and ls > 100:
        _warn(ctx, f"GP length_scale = {ls:.2f} 过大, 模型可能退化为常数预测.", "major")

    return None


# ── 注册函数 ─────────────────────────────────────────────────────


def register_science_hooks(hm: HookManager) -> None:
    """把材料科学 hook 注册到 HookManager. 在 agent 初始化时调用.

    幂等: 检查是否已注册, 避免重复.
    """
    if getattr(hm, "_science_hooks_registered", False):
        return

    # 原有三个 hook (vasp/structure 保留, 其余模式匹配走规则表)
    hm.register(POST_TOOL_USE, vasp_convergence_hook)
    hm.register(POST_TOOL_USE, structure_sanity_hook)
    # 模式匹配 hook: qe/cp2k/gaussian/orca/gromacs/fem/openfoam/packing/lammps
    for rule in _PATTERN_HOOK_RULES:
        hm.register(POST_TOOL_USE, _make_pattern_hook(rule))
    # 物理合理性检查族
    hm.register(POST_TOOL_USE, mechanical_property_hook)
    hm.register(POST_TOOL_USE, energy_bound_hook)
    # 文件存在性验证
    hm.register(POST_TOOL_USE, output_file_existence_hook)
    # 生物医药检查族
    hm.register(POST_TOOL_USE, vina_docking_hook)
    hm.register(POST_TOOL_USE, openmm_stability_hook)
    # 跨学科工具检查族 (loop engineering 补全)
    hm.register(POST_TOOL_USE, fep_validation_hook)
    hm.register(POST_TOOL_USE, enhanced_sampling_hook)
    hm.register(POST_TOOL_USE, msm_validation_hook)
    hm.register(POST_TOOL_USE, inverse_design_hook)
    hm.register(POST_TOOL_USE, motif_mining_hook)
    hm.register(POST_TOOL_USE, consensus_scoring_hook)
    hm.register(POST_TOOL_USE, rdkit_validation_hook)
    hm.register(POST_TOOL_USE, neb_convergence_hook)
    hm.register(POST_TOOL_USE, gp_model_hook)

    # 事件驱动管线 hook — 成功完成后建议下一步
    try:
        from huginn.provenance import pipeline_hook
        hm.register(POST_TOOL_USE, pipeline_hook)
    except ImportError:
        logger.debug("pipeline_hook not available (non-fatal)")

    # RAG 检索结果增强: 搜索知识库时附带 provenance 上下文
    try:
        from huginn.rag.provenance_enhancer import rag_provenance_hook
        hm.register(POST_TOOL_USE, rag_provenance_hook)
    except ImportError:
        logger.debug("rag_provenance_hook not available (non-fatal)")

    # 领域管线建议: 搜索知识库时根据领域推荐计算流程
    try:
        from huginn.knowledge.domain_pipeline import workflow_guidance_hook
        hm.register(POST_TOOL_USE, workflow_guidance_hook)
    except ImportError:
        logger.debug("workflow_guidance_hook not available (non-fatal)")

    # 计算结果自动摄入知识库 + hook 失败蒸馏联动
    try:
        from huginn.knowledge.auto_ingest import (
            calculation_ingest_hook,
            hook_failure_hook,
        )
        hm.register(POST_TOOL_USE, calculation_ingest_hook)
        hm.register(POST_TOOL_USE, hook_failure_hook)
    except ImportError:
        logger.debug("auto_ingest hooks not available (non-fatal)")

    # 检索质量反馈循环: rag_tool 搜索结果 -> 后续工具成败 -> 文档置信度调权
    try:
        from huginn.rag.feedback import rag_feedback_hook, rag_track_hook

        hm.register(POST_TOOL_USE, rag_track_hook)
        hm.register(POST_TOOL_USE, rag_feedback_hook)
    except ImportError:
        logger.debug("rag feedback hooks not available (non-fatal)")

    # 文件系统级 undo: 仿真/文件工具执行前后拍快照, 支持 revert/unrevert.
    # 幂等由 register_snapshot_hooks 内部的 _snapshot_hooks_registered 守护.
    try:
        from huginn.snapshot.integration import register_snapshot_hooks

        register_snapshot_hooks(hm)
    except ImportError:
        logger.debug("snapshot hooks not available (non-fatal)")

    hm._science_hooks_registered = True
    logger.info(
        "Science hooks registered: vasp_convergence, lammps_stability, "
        "structure_sanity, qe_convergence, cp2k_convergence, "
        "gaussian_termination, orca_termination, gromacs_stability, "
        "fem_solver, openfoam_residual, mechanical_property, "
        "packing_success, energy_bound, output_file_existence, "
        "vina_docking, openmm_stability, "
        "fep_validation, enhanced_sampling, msm_validation, "
        "inverse_design, motif_mining, consensus_scoring, "
        "rdkit_validation, neb_convergence, gp_model, "
        "pipeline, "
        "rag_provenance, workflow_guidance, calculation_ingest, "
        "hook_failure, rag_track, rag_feedback. "
        "Pipeline stages: structure→relax→static→properties→"
        "mechanical→md→cheminfo→docking/biomd→"
        "free_energy/enhanced_sampling→kinetics/motif/inverse→"
        "consensus→analysis"
    )

