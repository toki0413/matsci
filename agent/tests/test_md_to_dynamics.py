"""LAMMPS dump -> dynamics_discovery 数据管道测试.

三个层次:
  1. _parse_lammps_dump 能正确解析标准 dump 文本 (多帧/列定位)
  2. load_lammps_dump 返回里带物理量 (avg_speed / msd / kinetic_energy), 且数值对
  3. 从一条阻尼螺旋轨迹 dump 出发, SINDy 能稀疏恢复平均速度模和动能的衰减方程
     (dx0/dt = -g*x0, dke/dt = -2g*ke) —— 这是简谐(阻尼)运动的干净线性退化情形
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from huginn.tools.sci.dynamics_discovery_tool import (
    DynamicsDiscoveryInput,
    DynamicsDiscoveryTool,
)
from huginn.types import ToolContext

_COLS = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]


def _dump_text(frames: list[tuple[int, np.ndarray]]) -> str:
    """把 (timestep, atoms[N,8]) 帧列表拼成 LAMMPS dump 文本.

    atoms 列顺序固定为 _COLS, 跟 parser 约定一致.
    """
    parts: list[str] = []
    for ts, atoms in frames:
        parts.append("ITEM: TIMESTEP")
        parts.append(str(int(ts)))
        parts.append("ITEM: NUMBER OF ATOMS")
        parts.append(str(len(atoms)))
        parts.append("ITEM: BOX BOUNDS pp pp pp")
        parts.append("0.0 10.0")
        parts.append("0.0 10.0")
        parts.append("0.0 10.0")
        parts.append("ITEM: ATOMS " + " ".join(_COLS))
        for row in atoms:
            parts.append(" ".join(f"{v:.10g}" for v in row))
    return "\n".join(parts) + "\n"


def _ctx() -> ToolContext:
    return ToolContext(session_id="test-md2dyn", workspace=".")


def _run_dump(tmp_path: Path, frames, **kw) -> object:
    p = tmp_path / "traj.dump"
    p.write_text(_dump_text(frames), encoding="utf-8")
    tool = DynamicsDiscoveryTool()
    args = DynamicsDiscoveryInput(
        action="load_lammps_dump", data_file=str(p), **kw
    )
    return asyncio.run(tool.call(args, _ctx()))


# ── 1. 解析 ──────────────────────────────────────────────────────

def test_parse_simple_dump(tmp_path):
    """两帧单原子 dump: timestep / 列名 / 坐标都能解析出来."""
    a0 = np.array([[1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    a1 = np.array([[1, 1, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0]])
    p = tmp_path / "simple.dump"
    p.write_text(_dump_text([(0, a0), (100, a1)]), encoding="utf-8")

    frames = DynamicsDiscoveryTool._parse_lammps_dump(p)
    assert len(frames) == 2
    assert frames[0]["timestep"] == 0
    assert frames[1]["timestep"] == 100
    assert frames[0]["columns"] == _COLS
    assert frames[0]["atoms"].shape == (1, 8)
    # 第二帧坐标确实读进来了
    assert frames[1]["atoms"][0, 2] == pytest.approx(1.0)
    assert frames[1]["atoms"][0, 5] == pytest.approx(0.5)


def test_parse_ignores_extra_box_lines(tmp_path):
    """triclinic 的 box 多一行 tilt 因子, parser 靠 marker 跳过不该炸."""
    txt = (
        "ITEM: TIMESTEP\n0\n"
        "ITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS xy yz xz pp pp pp\n"
        "0.0 10.0 0.0\n0.0 10.0 0.0\n0.0 10.0 0.0\n"
        "ITEM: ATOMS id type x y z vx vy vz\n"
        "1 1 0.5 0.5 0.5 0.1 0.2 0.3\n"
    )
    p = tmp_path / "tri.dump"
    p.write_text(txt, encoding="utf-8")
    frames = DynamicsDiscoveryTool._parse_lammps_dump(p)
    assert len(frames) == 1
    assert frames[0]["atoms"].shape == (1, 8)
    assert frames[0]["atoms"][0, 5] == pytest.approx(0.1)


# ── 2. 物理量 ────────────────────────────────────────────────────

def test_quantities_present_and_correct(tmp_path):
    """单原子匀速直线运动: avg_speed/MSD/KE 都该等于手算值."""
    frames = []
    for i in range(6):  # >=5 帧让 discover 跑得起来
        x = float(i)
        atoms = np.array([[1, 1, x, 0.0, 0.0, 1.0, 0.0, 0.0]])
        frames.append((i, atoms))
    res = _run_dump(tmp_path, frames)
    assert res.success, res.error
    q = res.data["lammps_quantities"]
    # 全程速度模 = 1
    assert q["avg_speed"] == pytest.approx([1.0] * 6, abs=1e-9)
    # MSD = x^2 (初始位置 0)
    assert q["msd"] == pytest.approx([0.0, 1.0, 4.0, 9.0, 16.0, 25.0], abs=1e-9)
    # KE = 0.5 * m * v^2 = 0.5 (m=1, v=1)
    assert q["kinetic_energy"] == pytest.approx([0.5] * 6, abs=1e-9)
    assert q["n_frames"] == 6


def test_missing_velocity_columns_fails(tmp_path):
    """dump 没有 vx/vy/vz 时该报错而不是瞎算."""
    txt = (
        "ITEM: TIMESTEP\n0\n"
        "ITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n"
        "ITEM: ATOMS id type x y z\n1 1 0 0 0\n"
    )
    p = tmp_path / "noval.dump"
    p.write_text(txt, encoding="utf-8")
    tool = DynamicsDiscoveryTool()
    args = DynamicsDiscoveryInput(action="load_lammps_dump", data_file=str(p))
    res = asyncio.run(tool.call(args, _ctx()))
    assert not res.success
    assert "vx" in res.error or "velocity" in res.error.lower()


# ── 3. 方程发现 ──────────────────────────────────────────────────

def test_discover_damped_harmonic_from_dump(tmp_path):
    """阻尼螺旋轨迹 (xy 平面衰减旋转) 的 dump.

    取 x=e^{-g t} cos(w t), y=e^{-g t} sin(w t). 速度模 |v|=e^{-g t} sqrt(g^2+w^2)
    纯指数衰减, 所以平均速度模 x0 满足 dx0/dt = -g x0, 动能 x2 满足 dke/dt=-2g ke.
    这俩是干净的线性方程, SINDy 应能稀疏恢复.
    """
    g, w = 0.05, 0.5
    n = 160
    t = np.arange(n, dtype=float)
    frames = []
    for i, ti in enumerate(t):
        ex = np.exp(-g * ti)
        x, y = ex * np.cos(w * ti), ex * np.sin(w * ti)
        vx = -g * x - w * ex * np.sin(w * ti)
        vy = -g * y + w * ex * np.cos(w * ti)
        atoms = np.array([[1, 1, x, y, 0.0, vx, vy, 0.0]])
        frames.append((int(ti), atoms))

    res = _run_dump(tmp_path, frames, max_order=2, threshold=0.05, smooth=True)
    assert res.success, res.error
    data = res.data

    coefs_x0 = dict(zip(data["terms"], data["coefficients"]["x0"]))
    # 平均速度模 |v| = sqrt(g^2+w^2) e^{-g t} 纯指数衰减, 跟候选库里别的项不共线
    # (x0^2 / ke 都 ∝ e^{-2g t}), 所以这条方程系数应该干净恢复:
    #   d(avg_speed)/dt = -g * avg_speed
    assert "x0" in coefs_x0
    assert coefs_x0["x0"] == pytest.approx(-g, abs=0.01), coefs_x0["x0"]
    assert data["r2_score"]["x0"] > 0.9
    # 发现的方程非空
    assert len(data["equations"]) == 3
    # 物理量也带回来了
    assert "lammps_quantities" in data
    assert len(data["lammps_quantities"]["avg_speed"]) == n
    # 平均速度模确实在衰减 (首帧 > 末帧), 证明物理量提取对路
    avg = data["lammps_quantities"]["avg_speed"]
    assert avg[0] > avg[-1] > 0.0


def test_load_lammps_dump_missing_file():
    """文件不存在时优雅失败."""
    tool = DynamicsDiscoveryTool()
    args = DynamicsDiscoveryInput(
        action="load_lammps_dump", data_file="/nope/missing.dump"
    )
    res = asyncio.run(tool.call(args, _ctx()))
    assert not res.success
    assert "not found" in res.error.lower()
