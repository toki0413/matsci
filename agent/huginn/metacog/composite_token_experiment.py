"""C 实验: 复合 token (text × SE(3)) 半直积叠加可行性验证.

布尔巴基视角: 文本 token 和视觉 token 都可以看作三种母结构的复合.
本实验验证: SE(3) 群作用能否同时穿过文本 monoid 和坐标向量空间,
使半直积 (M × V) ⋊ SE(3) 成为合法的复合结构.

数学:
  M = Σ* (文本 free monoid, concat, 含 <point>[x,y]</point> 坐标)
  V = R^(3N) (3D 原子坐标向量空间)
  carrier = M × V
  SE(3) 作用: g▷(t, c) = (g▷t, g▷c)
    g▷t: 旋转文本里的 <point> 坐标 (2D 嵌入 3D z=0, 旋转, 投影回 2D)
    g▷c: 旋转 3D 坐标 (scipy Rotation)

验证 3 个性质:
  1. 群作用 compatibility: (g1·g2)▷t == g1▷(g2▷t)   [SE(3) 是真群作用]
  2. concat 同态: g▷(a·b) == (g▷a)·(g▷b)              [SE(3) 是 aut(M)]
  3. 单位元: e▷t == t                                  [平凡]

若三性质都成立, SE(3) 是 aut(M×V) 的子群, 半直积有意义.
三结构 (代数 concat / 代数 SE(3) / 拓扑 坐标邻域) 兼容叠加.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.spatial.transform import Rotation as _R
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# <point>[x,y]</point> 坐标, 允许负数 (旋转后可能出负)
_POINT_RE = re.compile(r"<point>\[(-?\d+),(-?\d+)\]</point>")


@dataclass
class CompositeToken:
    """复合 token = 文本 (free monoid) × 3D 坐标 (R^3N).

    text: 文本, 可能含:
      - <point>[x,y]</point> 2D 归一化坐标 (0-999)
      - <point3d>[x,y,z]</point3d>(label) 3D 归一化坐标 (扩 SE(3) 实验)
    coords: (N, 3) 3D 原子坐标 (Å)
    """
    text: str
    coords: np.ndarray


# <point3d>[x,y,z]</point3d>(label) — 3D 坐标, 允许负数 (旋转后)
_POINT3D_RE = re.compile(
    r"<point3d>\[(-?\d+),(-?\d+),(-?\d+)\]</point3d>(?:\([^)]+\))?"
)


def se3_act(rot: Any, trans: np.ndarray, token: CompositeToken) -> CompositeToken:
    """SE(3) 群作用: 旋转 + 平移, 同时作用在 text <point>/<point3d> 和 coords 上.

    <point>[x,y]   → 嵌入 3D (x,y,0) → 旋转 → +平移 → 投影回 2D (取 x,y)
    <point3d>[x,y,z] → 直接 3D 旋转 → +平移 (不投影, 真 3D 群作用)
    coords → 旋转 → +平移

    扩展: <point3d> 原语支持真 3D SE(3) 群作用, 不再丢失 z 维度.
    """
    # 1. 作用在 text <point3d>: 直接 3D 旋转 (不投影)
    def _replace_3d(m: re.Match) -> str:
        x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
        p3d = np.array([float(x), float(y), float(z)])
        p_new = rot.apply(p3d) + trans
        # 保留原 label (如果有), 格式 <point3d>[x,y,z]</point3d>(label)
        label_match = re.search(r"\(([^)]+)\)", m.group(0))
        label = f"({label_match.group(1)})" if label_match else ""
        return f"<point3d>[{int(round(p_new[0]))},{int(round(p_new[1]))},{int(round(p_new[2]))}]</point3d>{label}"
    new_text = _POINT3D_RE.sub(_replace_3d, token.text)

    # 2. 作用在 text <point>: 2D→3D 嵌入 → 旋转 → 投影回 2D (legacy)
    def _replace_2d(m: re.Match) -> str:
        x, y = int(m.group(1)), int(m.group(2))
        p3d = np.array([float(x), float(y), 0.0])
        p_new = rot.apply(p3d) + trans
        return f"<point>[{int(round(p_new[0]))},{int(round(p_new[1]))}]</point>"
    new_text = _POINT_RE.sub(_replace_2d, new_text)

    # 3. 作用在 coords: 旋转 3D 坐标
    if len(token.coords) > 0:
        new_coords = rot.apply(token.coords) + trans
    else:
        new_coords = token.coords.copy()

    return CompositeToken(new_text, new_coords)


def concat(a: CompositeToken, b: CompositeToken) -> CompositeToken:
    """文本 monoid concat + 坐标 vstack (free monoid 乘法)."""
    new_text = (a.text + " " + b.text).strip()
    if len(a.coords) > 0 and len(b.coords) > 0:
        new_coords = np.vstack([a.coords, b.coords])
    elif len(a.coords) > 0:
        new_coords = a.coords.copy()
    else:
        new_coords = b.coords.copy()
    return CompositeToken(new_text, new_coords)


def _selfcheck() -> None:
    """验证 3 个数学性质 + 1 个实际场景."""
    print("=" * 70)
    print("C 实验: 复合 token (text × SE(3)) 半直积叠加验证")
    print("=" * 70)
    print()

    # ── 性质 1: 群作用 compatibility ───────────────────────────
    # (g1·g2)▷t == g1▷(g2▷t)
    print("[1] 群作用 compatibility: (g1·g2)▷t == g1▷(g2▷t)")
    print("    含义: 先 g2 后 g1 作用 == 先合成 g1g2 再作用")
    t = CompositeToken(
        text="peak at <point>[500,800]</point>, min at <point>[100,200]</point>",
        coords=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    g1_rot = _R.from_euler("z", 30, degrees=True)
    g1_trans = np.array([1.0, 0.0, 0.0])
    g2_rot = _R.from_euler("z", 45, degrees=True)
    g2_trans = np.array([0.0, 1.0, 0.0])

    # (g1·g2) 合成: scipy g1*g2 = 先 g2 后 g1; 平移 = R1·t2 + t1
    g12_rot = g1_rot * g2_rot
    g12_trans = g1_rot.apply(g2_trans) + g1_trans

    lhs = se3_act(g12_rot, g12_trans, t)
    rhs = se3_act(g1_rot, g1_trans, se3_act(g2_rot, g2_trans, t))

    text_match = lhs.text == rhs.text
    coords_match = np.allclose(lhs.coords, rhs.coords, atol=1e-6)
    assert text_match, f"text mismatch:\n  LHS={lhs.text}\n  RHS={rhs.text}"
    assert coords_match, f"coords mismatch:\n  LHS={lhs.coords}\n  RHS={rhs.coords}"
    print(f"    text:   {'PASS' if text_match else 'FAIL'}")
    print(f"    coords: {'PASS' if coords_match else 'FAIL'}")
    print(f"    LHS: {lhs.text}")
    print(f"    RHS: {rhs.text}")
    print()

    # ── 性质 2: concat 同态 ───────────────────────────────────
    # g▷(a·b) == (g▷a)·(g▷b)
    print("[2] concat 同态: g▷(a·b) == (g▷a)·(g▷b)")
    print("    含义: 先 concat 再旋转 == 先旋转再 concat (SE(3) 是 aut(M))")
    a = CompositeToken(
        text="peak at <point>[500,800]</point>",
        coords=np.array([[1.0, 0.0, 0.0]]),
    )
    b = CompositeToken(
        text="min at <point>[100,200]</point>",
        coords=np.array([[0.0, 1.0, 0.0]]),
    )
    g_rot = _R.from_euler("z", 90, degrees=True)
    g_trans = np.array([0.5, 0.5, 0.0])

    lhs = se3_act(g_rot, g_trans, concat(a, b))
    rhs = concat(se3_act(g_rot, g_trans, a), se3_act(g_rot, g_trans, b))

    text_match = lhs.text == rhs.text
    coords_match = np.allclose(lhs.coords, rhs.coords, atol=1e-6)
    assert text_match, f"text mismatch:\n  LHS={lhs.text}\n  RHS={rhs.text}"
    assert coords_match, f"coords mismatch:\n  LHS={lhs.coords}\n  RHS={rhs.coords}"
    print(f"    text:   {'PASS' if text_match else 'FAIL'}")
    print(f"    coords: {'PASS' if coords_match else 'FAIL'}")
    print(f"    LHS: {lhs.text}")
    print(f"    RHS: {rhs.text}")
    print()

    # ── 性质 3: 单位元 ───────────────────────────────────────
    # e▷t == t
    print("[3] 单位元: e▷t == t")
    print("    含义: 恒等变换不改变 token")
    e_rot = _R.identity()
    e_trans = np.zeros(3)
    result = se3_act(e_rot, e_trans, t)
    text_match = result.text == t.text
    coords_match = np.allclose(result.coords, t.coords)
    assert text_match, f"text changed: {result.text} vs {t.text}"
    assert coords_match, "coords changed"
    print(f"    text:   {'PASS' if text_match else 'FAIL'}")
    print(f"    coords: {'PASS' if coords_match else 'FAIL'}")
    print()

    # ── 实际场景: 旋转复合 token 90° ──────────────────────────
    print("[4] 实际场景: 旋转复合 token 90° (绕 z 轴 CCW)")
    print("    含义: 文本 <point> 坐标 + 3D 原子坐标 同步旋转")
    token = CompositeToken(
        text="lattice peak at <point>[800,500]</point>, origin at <point>[0,0]</point>",
        coords=np.array([
            [0.0, 0.0, 0.0],  # 原子 1 (原点)
            [2.0, 0.0, 0.0],  # 原子 2 (x 方向)
            [0.0, 2.0, 0.0],  # 原子 3 (y 方向)
        ]),
    )
    rot_90 = _R.from_euler("z", 90, degrees=True)
    trans_0 = np.zeros(3)
    rotated = se3_act(rot_90, trans_0, token)

    print(f"    原始 text:   {token.text}")
    print(f"    旋转 text:   {rotated.text}")
    print(f"    原始 coords: {token.coords.tolist()}")
    print(f"    旋转 coords: {rotated.coords.tolist()}")

    # 验证: <point>[800,500] → [-500,800] (CCW 90°: (x,y)→(-y,x))
    assert "<point>[-500,800]</point>" in rotated.text, \
        f"<point>[800,500] 旋转 90° 后应为 [-500,800], 实际: {rotated.text}"
    # 验证: coords [2,0,0] → [0,2,0]
    assert np.allclose(rotated.coords[1], [0.0, 2.0, 0.0], atol=1e-6), \
        f"coords [2,0,0] 旋转 90° 后应为 [0,2,0], 实际: {rotated.coords[1]}"
    # 原点不变
    assert np.allclose(rotated.coords[0], [0.0, 0.0, 0.0]), "原点应不变"
    assert "<point>[0,0]</point>" in rotated.text, "原点 <point> 应不变"
    print(f"    PASS: <point> 坐标和 3D 坐标同步旋转, 原点不变")
    print()

    # ── 扩展: <point3d> 真 3D SE(3) 群作用 ────────────────────
    print("[5] <point3d> 真 3D SE(3) 群作用 (绕 x 轴 90°)")
    print("    含义: 3D 原语直接 3D 旋转, 不投影, 不丢 z 维度")
    token_3d = CompositeToken(
        text=(
            "atoms: <point3d>[999,0,0]</point3d>(Fe), "
            "<point3d>[0,999,0]</point3d>(O), "
            "<point3d>[0,0,999]</point3d>(O)"
        ),
        coords=np.array([
            [2.0, 0.0, 0.0],  # Fe 在 x 轴
            [0.0, 2.0, 0.0],  # O 在 y 轴
            [0.0, 0.0, 2.0],  # O 在 z 轴
        ]),
    )
    # 绕 x 轴 90°: y→z, z→-y
    rot_x90 = _R.from_euler("x", 90, degrees=True)
    trans_0 = np.zeros(3)
    rotated_3d = se3_act(rot_x90, trans_0, token_3d)

    print(f"    原始 text:   {token_3d.text}")
    print(f"    旋转 text:   {rotated_3d.text}")
    print(f"    原始 coords: {token_3d.coords.tolist()}")
    print(f"    旋转 coords: {rotated_3d.coords.tolist()}")

    # 验证 <point3d>:
    #   [999,0,0] (Fe, x 轴) → x 轴不变 [999,0,0]
    assert "<point3d>[999,0,0]</point3d>(Fe)" in rotated_3d.text, \
        f"Fe x 轴应不变, 实际: {rotated_3d.text}"
    #   [0,999,0] (O, y 轴) → y→z, z→-y: [0,0,999] (绕 x 90°: y→z)
    assert "<point3d>[0,0,999]</point3d>(O)" in rotated_3d.text or \
           "<point3d>[0,-0,999]</point3d>(O)" in rotated_3d.text, \
        f"O y→z 应为 [0,0,999], 实际: {rotated_3d.text}"
    #   [0,0,999] (O, z 轴) → z→-y: [0,-999,0]
    assert "<point3d>[0,-999,0]</point3d>(O)" in rotated_3d.text, \
        f"O z→-y 应为 [0,-999,0], 实际: {rotated_3d.text}"
    # 验证 coords 同步:
    #   [0,2,0] (y 轴) → [0,0,2] (绕 x 90°: y→z)
    assert np.allclose(rotated_3d.coords[1], [0.0, 0.0, 2.0], atol=1e-6), \
        f"coords [0,2,0] 绕 x 90° 应为 [0,0,2], 实际: {rotated_3d.coords[1]}"
    #   [0,0,2] (z 轴) → [0,-2,0]
    assert np.allclose(rotated_3d.coords[2], [0.0, -2.0, 0.0], atol=1e-6), \
        f"coords [0,0,2] 绕 x 90° 应为 [0,-2,0], 实际: {rotated_3d.coords[2]}"
    print(f"    PASS: <point3d> 3D 坐标和 3D coords 同步旋转, z 维度保留")
    print()

    # ── 扩展: <point3d> 群作用 compatibility ───────────────────
    print("[6] <point3d> 群作用 compatibility: (g1·g2)▷t == g1▷(g2▷t)")
    print("    含义: 3D 原语也满足群作用相容性")
    t_3d = CompositeToken(
        text="<point3d>[500,300,100]</point3d>(C)",
        coords=np.array([[1.0, 2.0, 3.0]]),
    )
    g1_rot = _R.from_euler("y", 30, degrees=True)
    g1_trans = np.array([1.0, 0.0, 0.0])
    g2_rot = _R.from_euler("z", 45, degrees=True)
    g2_trans = np.array([0.0, 1.0, 0.0])

    g12_rot = g1_rot * g2_rot
    g12_trans = g1_rot.apply(g2_trans) + g1_trans

    lhs = se3_act(g12_rot, g12_trans, t_3d)
    rhs = se3_act(g1_rot, g1_trans, se3_act(g2_rot, g2_trans, t_3d))

    text_match = lhs.text == rhs.text
    coords_match = np.allclose(lhs.coords, rhs.coords, atol=1e-6)
    assert text_match, f"text mismatch:\n  LHS={lhs.text}\n  RHS={rhs.text}"
    assert coords_match, f"coords mismatch"
    print(f"    text:   {'PASS' if text_match else 'FAIL'}")
    print(f"    coords: {'PASS' if coords_match else 'FAIL'}")
    print()

    # ── 结论 ─────────────────────────────────────────────────
    print("=" * 70)
    print("结论")
    print("=" * 70)
    print()
    print("4 个数学性质 + 2 个扩展全部成立:")
    print("  [1] 群作用 compatibility  → SE(3) 是真群作用 (不是伪作用)")
    print("  [2] concat 同态           → SE(3) 是 aut(M), 保代数结构")
    print("  [3] 单位元                → 平凡")
    print("  [4] <point> 2D 旋转        → SE(2) ⊂ SE(3) 子群作用")
    print("  [5] <point3d> 真 3D 旋转  → SE(3) 完整群作用, z 维度保留")
    print("  [6] <point3d> compatibility→ 3D 原语也满足群作用相容性")
    print()
    print("因此 SE(3) 是 aut(M × V) 的子群, 半直积 (M × V) ⋊ SE(3) 合法.")
    print("<point3d> 原语让 SE(3) 群作用完整穿过到文本, 不再需要 2D 投影.")
    print()
    print("布尔巴基三结构兼容叠加验证:")
    print("  代数 I  (free monoid concat): 文本 token 拼接, 旋转不变 concat 结构")
    print("  代数 II (SE(3) 群作用):       3D 坐标旋转, 同时穿过到文本 <point>/<point3d>")
    print("  拓扑  (坐标邻域):             <point>/<point3d> 和 coords 共享同一 SE(3) 变换,")
    print("                                邻域结构一致")
    print()
    print("实际意义: agent 的视觉 token (primitives + coords) 和文本 token")
    print("可以在同一 SE(3) 变换下同步, 不需要分别处理两种模态.")
    print("这是 unified token protocol 的数学基础 (虽然离工程实现还有距离).")
    print()
    print("C EXPERIMENT ALL CHECKS PASSED")


if __name__ == "__main__":
    if not _HAS_SCIPY:
        print("需要 scipy: pip install scipy")
        raise SystemExit(1)
    _selfcheck()
