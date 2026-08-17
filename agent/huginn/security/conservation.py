"""材料计算的守恒不变量对账 — 独立于 LLM 的机械守恒审计.

对标"朗兰兹纲领: 对偶不变量"的落地: 不给 LLM 打分中间过程(那是 step_verifier),
而是对解析出的原子受力/位置做**机械守恒校验**. 对一个孤立的 DFT 原子构型, 无净
外力时, 平移平衡 (ΣF ≈ 0) 与旋转平衡 (Σ r_i × F_i ≈ 0) 是**不依赖模型的两条刚性
物理约束** — 一旦被破坏, 说明几何未收敛 / 约束设置错误 / 对称性被破坏.

输入来源: vasp_tool._parse_outcar 的 result dict
  forces: [{"position":[x, y, z], "force":[fx, fy, fz]}, ...]
    - regex 兜底路径: position 是 OUTCAR TOTAL-FORCE 的真实坐标
    - pymatgen 路径:   position 是占位 [0.0, 0.0, 0.0] (pymatgen 不返回坐标)
  净力 (翻译平衡) 总能算; 净扭矩 (旋转平衡) 仅在 position 为真实坐标时算, 否则
  安全跳过 (返回 None), 不误告警.

本模块是纯函数, 无 IO / 无 LLM / 无单例, 可直接单测.
"""

from __future__ import annotations

import math
from typing import Any

_TORQUE_PLACEHOLDER_TOL = 1e-9  # position 全为占位 [0,0,0] 时视为无真实坐标

# 对账阈值 (可被 audit_material_conservation 参数覆盖).
# 取值依据: 对完全平衡的构型, 净力应远小于任何单个原子力.
_FORCE_BALANCE_WARN = 0.10   # 净力达到最强原子力的 10% → warn
_FORCE_BALANCE_FAIL = 0.50   # 净力达到最强原子力的 50% → fail
_TORQUE_BALANCE_WARN = 0.10
_TORQUE_BALANCE_FAIL = 0.50
# 宣称已收敛 (relax 离子收敛) 但残余力仍很大 = 自相矛盾 (Ev/Å).
_CONVERGED_FORCE_MAX = 1.0


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(c * c for c in v))


def _usable_forces(forces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保留含合法数值 3 维力的原子; 畸形条目 (缺 / 非数值) 直接跳过.

    保证审计对脏数据健壮: 坏条目不拖垮整体, 也不静默污染净力.
    """
    out: list[dict[str, Any]] = []
    for atom in forces:
        vec = atom.get("force")
        if not isinstance(vec, (list, tuple)) or len(vec) < 3:
            continue
        try:
            fv = [float(x) for x in vec[:3]]
        except (TypeError, ValueError):
            continue
        position = atom.get("position")
        out.append({"position": position, "force": fv})
    return out


def _net_force_vec(forces: list[dict[str, Any]]) -> list[float]:
    f = [0.0, 0.0, 0.0]
    for atom in _usable_forces(forces):
        vec = atom.get("force")  # type: ignore[assignment]
        f[0] += vec[0]  # type: ignore[index]
        f[1] += vec[1]  # type: ignore[index]
        f[2] += vec[2]  # type: ignore[index]
    return f


def net_force(forces: list[dict[str, Any]]) -> float:
    """净合力幅值 (平移平衡量). 无外力时不应为零附近."""
    return _norm(_net_force_vec(forces))


def _has_real_positions(forces: list[dict[str, Any]]) -> bool:
    """position 是否为真实坐标. pymatgen 路径给 [0,0,0] 占位, 视为无真实坐标."""
    for atom in _usable_forces(forces):
        pos = atom.get("position")
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            continue
        try:
            if _norm([float(c) for c in pos[:3]]) > _TORQUE_PLACEHOLDER_TOL:
                return True
        except (TypeError, ValueError):
            continue
    return False


def net_torque(forces: list[dict[str, Any]]) -> float | None:
    """净扭矩幅值 |Σ r_i × F_i|. 无真实坐标时返回 None (旋转平衡不可判)."""
    if not _has_real_positions(forces):
        return None
    t = [0.0, 0.0, 0.0]
    for atom in _usable_forces(forces):
        pos = atom.get("position")
        f = atom.get("force")  # type: ignore[assignment]
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            continue
        rx, ry, rz = (float(c) for c in pos[:3])
        fx, fy, fz = f[0], f[1], f[2]  # type: ignore[index]
        # cross product r × F
        t[0] += ry * fz - rz * fy
        t[1] += rz * fx - rx * fz
        t[2] += rx * fy - ry * fx
    return _norm(t)


def _max_force(forces: list[dict[str, Any]]) -> float:
    m = 0.0
    for atom in _usable_forces(forces):
        f = atom.get("force")  # type: ignore[assignment]
        m = max(m, _norm([float(c) for c in f[:3]]))  # type: ignore[index]
    return m


def _mean_scale(fmax: float, n: int) -> float:
    """力尺度参考. 用最强原子力兜底 (平移不变, 不依赖坐标)."""
    return fmax if fmax > 0 else 0.0


def audit_material_conservation(
    result: dict[str, Any],
    *,
    force_balance_warn: float = _FORCE_BALANCE_WARN,
    force_balance_fail: float = _FORCE_BALANCE_FAIL,
    torque_balance_warn: float = _TORQUE_BALANCE_WARN,
    torque_balance_fail: float = _TORQUE_BALANCE_FAIL,
    converged_force_max: float = _CONVERGED_FORCE_MAX,
) -> dict[str, Any]:
    """对解析出的结果做守恒对账, 返回结构化审计 (纯函数).

    result 至少含 ``forces`` (list[dict]). 其余字段 (converged) 可选.

    Returns:
        {
          "invariants": [...],          # 实际校验的不变量名
          "n_atoms": int,
          "net_force_gap": float,       # |ΣF|
          "force_balance_ratio": float, # |ΣF| / max|F|  (0 兜底)
          "torque_gap": float|None,     # 有无真实坐标才有值
          "torque_ratio": float|None,
          "max_force": float,
          "forces_relaxed": bool,       # max|F| 落在收敛量级 (< converged_force_max)
          "verdict": "pass"|"warn"|"fail"|"skip",
          "reasons": [str, ...],
        }
    """
    forces = result.get("forces") or []
    usable = _usable_forces(forces)
    n = len(usable)
    if n == 0:
        return {
            "invariants": [],
            "n_atoms": 0,
            "net_force_gap": 0.0,
            "force_balance_ratio": 0.0,
            "torque_gap": None,
            "torque_ratio": None,
            "max_force": 0.0,
            "forces_relaxed": True,
            "verdict": "skip",
            "reasons": ["无可用受力数据, 守恒审计跳过"],
        }

    invariants = ["translational_equilibrium"]
    fmax = _max_force(usable)
    f_scale = _mean_scale(fmax, n)

    net_f = net_force(usable)
    force_ratio = (net_f / f_scale) if f_scale > 0 else 0.0

    # 旋转平衡: 仅当有真实坐标
    torque_gap = net_torque(usable)
    torque_ratio: float | None = None
    if torque_gap is not None:
        invariants.append("rotational_equilibrium")
        # 参考尺度: (质心基准力臂) 近似用坐标一阶矩强度 * 最大力
        lever = _max_position_scale(forces)
        ref = (lever * fmax) if lever > 0 else f_scale
        torque_ratio = (torque_gap / ref) if ref > 0 else 0.0

    reasons: list[str] = []
    verdict = "pass"

    if force_ratio > force_balance_fail:
        verdict = "fail"
        reasons.append(
            f"平移平衡破坏: 净力 |ΣF|={net_f:.4g} 达最强原子力 {force_ratio*100:.0f}% "
            "(>50%), 几何可能未收敛或约束错误"
        )
    elif force_ratio > force_balance_warn:
        verdict = "warn"
        reasons.append(
            f"平移平衡可疑: 净力 |ΣF|={net_f:.4g} 为最强原子力的 {force_ratio*100:.0f}%"
        )

    if torque_ratio is not None:
        if torque_ratio > torque_balance_fail:
            verdict = "fail" if verdict != "fail" else verdict
            reasons.append(
                f"旋转平衡破坏: 净扭矩 {torque_gap:.4g} 达参考量级 {torque_ratio*100:.0f}% "
                "(>50%)"
            )
        elif torque_ratio > torque_balance_warn:
            verdict = "warn" if verdict == "pass" else verdict
            reasons.append(
                f"旋转平衡可疑: 净扭矩为参考量级的 {torque_ratio*100:.0f}%"
            )

    # 跨信号对账: 宣称离子收敛 (relax) 却残留大残余力 = 自相矛盾.
    forces_relaxed = fmax <= converged_force_max
    if result.get("converged") and not forces_relaxed:
        verdict = "warn" if verdict == "pass" else verdict
        reasons.append(
            f"收敛标记自相矛盾: 宣称 converged 但 max|F|={fmax:.4g} Ev/Å "
            f"> {converged_force_max:g} (relax 离子收敛本应力极小)"
        )

    if not reasons:
        reasons.append("平移/旋转平衡通过, 未见守恒破坏")

    return {
        "invariants": invariants,
        "n_atoms": n,
        "net_force_gap": net_f,
        "force_balance_ratio": round(force_ratio, 4),
        "torque_gap": torque_gap,
        "torque_ratio": round(torque_ratio, 4) if torque_ratio is not None else None,
        "max_force": fmax,
        "forces_relaxed": forces_relaxed,
        "verdict": verdict,
        "reasons": reasons,
    }


def _max_position_scale(forces: list[dict[str, Any]]) -> float:
    """当前位置尺度的保守估计 (用于扭矩参考量级)."""
    m = 0.0
    for atom in _usable_forces(forces):
        pos = atom.get("position")
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            continue
        try:
            m = max(m, _norm([float(c) for c in pos[:3]]))
        except (TypeError, ValueError):
            continue
    return m


__all__ = [
    "net_force",
    "net_torque",
    "audit_material_conservation",
]