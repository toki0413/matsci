"""Real-time 3D molecular viewer endpoints.

Inspired by MolGame: streams atom positions and live telemetry while the
simulation runs, and lets the user "steer" atoms by sending force vectors
through the WebSocket. Also exposes a small REST surface for loading static
structures / trajectories and querying the element table.

Endpoints
---------
* ``POST /viewer3d/load``        - parse a structure file (POSCAR/CIF/XYZ)
* ``POST /viewer3d/trajectory``   - parse a trajectory (XDATCAR/XYZ traj)
* ``GET  /viewer3d/elements``    - element table (colors, radii)
* ``WS   /ws/viewer3d``           - real-time stream + force steering

The heavy lifting (pymatgen / ase) is lazy-imported so the module loads
even when those packages aren't installed; we fall back to a tiny built-in
parser for POSCAR/XYZ in that case.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from huginn.security.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/viewer3d", tags=["viewer3d"])

# WebSocket 前缀和路由分开注册: /ws/viewer3d 不会被 /viewer3d 前缀影响
ws_router = APIRouter(tags=["viewer3d-ws"])


# ── 元素数据: CPK 颜色 + 共价半径 ────────────────────────────────
# 数据来源: Jmol 颜色表 + Cordero 共价半径表, 手工摘录常用元素
_ELEMENT_COLORS: dict[str, list[float]] = {
    "H": [1.00, 1.00, 1.00], "He": [0.85, 1.00, 1.00],
    "Li": [0.80, 0.50, 0.25], "Be": [0.76, 1.00, 0.00],
    "B": [1.00, 0.71, 0.71], "C": [0.30, 0.30, 0.30],
    "N": [0.19, 0.31, 0.97], "O": [1.00, 0.05, 0.05],
    "F": [0.56, 0.87, 0.31], "Ne": [0.70, 0.89, 0.96],
    "Na": [0.67, 0.36, 0.94], "Mg": [0.54, 1.00, 0.00],
    "Al": [0.75, 0.65, 0.65], "Si": [0.94, 0.78, 0.63],
    "P": [1.00, 0.50, 0.00], "S": [1.00, 1.00, 0.19],
    "Cl": [0.12, 0.94, 0.12], "Ar": [0.50, 0.81, 0.89],
    "K": [0.56, 0.25, 0.83], "Ca": [0.24, 1.00, 0.00],
    "Sc": [0.90, 0.90, 0.90], "Ti": [0.75, 0.76, 0.78],
    "V": [0.65, 0.65, 0.67], "Cr": [0.54, 0.60, 0.78],
    "Mn": [0.61, 0.48, 0.78], "Fe": [0.88, 0.40, 0.20],
    "Co": [0.94, 0.56, 0.63], "Ni": [0.31, 0.82, 0.31],
    "Cu": [0.78, 0.50, 0.20], "Zn": [0.49, 0.50, 0.69],
    "Ga": [0.76, 0.56, 0.56], "Ge": [0.40, 0.56, 0.56],
    "As": [0.74, 0.50, 0.89], "Se": [1.00, 0.63, 0.00],
    "Br": [0.65, 0.16, 0.16], "Kr": [0.36, 0.72, 0.82],
    "Rb": [0.44, 0.18, 0.69], "Sr": [0.00, 1.00, 0.00],
    "Y": [0.58, 1.00, 1.00], "Zr": [0.58, 0.88, 0.88],
    "Nb": [0.45, 0.76, 0.79], "Mo": [0.33, 0.71, 0.71],
    "Tc": [0.23, 0.62, 0.62], "Ru": [0.14, 0.56, 0.56],
    "Rh": [0.04, 0.49, 0.55], "Pd": [0.00, 0.41, 0.52],
    "Ag": [0.75, 0.75, 0.75], "Cd": [1.00, 0.85, 0.56],
    "In": [0.65, 0.46, 0.45], "Sn": [0.40, 0.50, 0.50],
    "Sb": [0.62, 0.39, 0.71], "Te": [0.83, 0.48, 0.00],
    "I": [0.58, 0.00, 0.58], "Xe": [0.26, 0.62, 0.69],
    "Cs": [0.34, 0.09, 0.56], "Ba": [0.00, 0.79, 0.00],
    "La": [0.44, 0.83, 1.00], "Ce": [1.00, 1.00, 0.78],
    "Pr": [0.85, 1.00, 0.78], "Nd": [0.78, 1.00, 0.78],
    "Pm": [0.64, 1.00, 0.78], "Sm": [0.56, 1.00, 0.78],
    "Eu": [0.38, 1.00, 0.78], "Gd": [0.27, 1.00, 0.78],
    "Tb": [0.19, 1.00, 0.78], "Dy": [0.12, 1.00, 0.78],
    "Ho": [0.00, 1.00, 0.61], "Er": [0.00, 0.90, 0.46],
    "Tm": [0.00, 0.83, 0.32], "Yb": [0.00, 0.75, 0.22],
    "Lu": [0.00, 0.67, 0.14], "Hf": [0.30, 0.76, 1.00],
    "Ta": [0.30, 0.65, 1.00], "W": [0.13, 0.58, 0.84],
    "Re": [0.15, 0.49, 0.67], "Os": [0.15, 0.40, 0.59],
    "Ir": [0.09, 0.33, 0.53], "Pt": [0.82, 0.82, 0.88],
    "Au": [1.00, 0.82, 0.14], "Hg": [0.72, 0.72, 0.82],
    "Tl": [0.65, 0.33, 0.30], "Pb": [0.34, 0.35, 0.38],
    "Bi": [0.62, 0.31, 0.71], "U": [0.00, 0.56, 1.00],
}

# 共价半径 (Å), Cordero 2008; 未知元素回退到 1.5
_COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84,
    "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
    "Kr": 1.16, "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75,
    "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42,
    "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39,
    "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44,
    "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
    "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94,
    "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87,
    "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51,
    "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50,
    "Rn": 1.50, "U": 1.70,
}

_DEFAULT_COLOR = [1.0, 0.41, 0.71]  # 未知元素用粉色
_DEFAULT_RADIUS = 1.50
_BOND_TOLERANCE = 1.3  # 共价半径之和的放大系数, 用于成键判定


def _element_color(symbol: str) -> list[float]:
    return _ELEMENT_COLORS.get(symbol.capitalize(), _DEFAULT_COLOR)


def _covalent_radius(symbol: str) -> float:
    return _COVALENT_RADII.get(symbol.capitalize(), _DEFAULT_RADIUS)


# ── 轻量结构解析器 ────────────────────────────────────────────
# 优先使用 pymatgen, 不可用时回退到手写解析器, 保证模块可独立加载

def _parse_structure(text: str, fmt: str) -> dict[str, Any]:
    """Parse a structure string into a JSON-serializable dict.

    Returns: {atoms, bonds, cell, title}
    """
    fmt = fmt.lower()
    if fmt == "cif":
        atoms, cell, title = _parse_cif(text)
    elif fmt == "xyz":
        atoms, cell, title = _parse_xyz(text), None, "XYZ structure"
    elif fmt in ("poscar", "vasp"):
        atoms, cell, title = _parse_poscar(text)
    else:
        # 让 pymatgen 试试猜测格式
        atoms, cell, title = _parse_via_pymatgen(text, fmt)

    bonds = _detect_bonds(atoms)
    return {
        "atoms": atoms,
        "bonds": bonds,
        "cell": cell,
        "title": title,
        "n_atoms": len(atoms),
    }


def _parse_xyz(text: str) -> list[dict[str, Any]]:
    lines = text.strip().splitlines()
    if not lines:
        return []
    try:
        n = int(lines[0].strip())
    except ValueError:
        n = len(lines) - 2
    atoms: list[dict[str, Any]] = []
    for line in lines[2 : 2 + max(n, 0)]:
        parts = line.strip().split()
        if len(parts) >= 4:
            sym = parts[0]
            atoms.append({
                "element": sym,
                "position": [float(parts[1]), float(parts[2]), float(parts[3])],
                "radius": _covalent_radius(sym),
                "color": _element_color(sym),
            })
    return atoms


def _parse_poscar(text: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    lines = text.strip().splitlines()
    if len(lines) < 8:
        return [], None, "invalid POSCAR"
    title = lines[0].strip()
    scale = float(lines[1].strip() or "1.0")
    a = [float(x) * scale for x in lines[2].strip().split()]
    b = [float(x) * scale for x in lines[3].strip().split()]
    c = [float(x) * scale for x in lines[4].strip().split()]
    cell = {"a": a, "b": b, "c": c, "scale": scale}

    species_line = lines[5].strip().split()
    counts_line = lines[6].strip().split()

    # 处理没有元素行的情况: 第 5 行是数量, 第 6 行是坐标类型
    if all(p.isdigit() for p in species_line):
        counts = [int(x) for x in species_line]
        species = [f"X{i}" for i in range(len(counts))]
        coord_line = lines[6].strip().lower()
        first_coord_idx = 7
    else:
        species = species_line
        counts = [int(x) for x in counts_line]
        coord_line = lines[7].strip().lower() if len(lines) > 7 else "direct"
        first_coord_idx = 8

    is_direct = coord_line.startswith("d") or coord_line.startswith("f")
    atoms: list[dict[str, Any]] = []
    idx = first_coord_idx
    for sp, cnt in zip(species, counts):
        for _ in range(cnt):
            if idx >= len(lines):
                break
            parts = lines[idx].strip().split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                if is_direct:
                    # 笛卡尔坐标 = x*a + y*b + z*c
                    cx = x * a[0] + y * b[0] + z * c[0]
                    cy = x * a[1] + y * b[1] + z * c[1]
                    cz = x * a[2] + y * b[2] + z * c[2]
                else:
                    cx, cy, cz = x, y, z
                atoms.append({
                    "element": sp,
                    "position": [cx, cy, cz],
                    "radius": _covalent_radius(sp),
                    "color": _element_color(sp),
                })
            idx += 1
    return atoms, cell, title


def _parse_cif(text: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    """解析 CIF, 优先用 pymatgen, 失败回退到正则提取原子坐标."""
    try:
        return _parse_via_pymatgen(text, "cif")
    except Exception:
        pass

    # 正则兜底: 提取 _cell_length_a 等和 _atom_site 数据
    def _g(pat: str) -> float | None:
        m = re.search(pat, text, re.IGNORECASE)
        return float(m.group(1)) if m else None

    la, lb, lc = _g(r"_cell_length_a\s+([\d.]+)"), _g(r"_cell_length_b\s+([\d.]+)"), _g(r"_cell_length_c\s+([\d.]+)")
    an, bn, cn = _g(r"_cell_angle_alpha\s+([\d.]+)"), _g(r"_cell_angle_beta\s+([\d.]+)"), _g(r"_cell_angle_gamma\s+([\d.]+)")
    cell = None
    if la and lb and lc:
        cell = {"a": [la, 0, 0], "b": [0, lb, 0], "c": [0, 0, lc], "angles": [an or 90, bn or 90, cn or 90]}

    atoms: list[dict[str, Any]] = []
    # 找到 _atom_site 行后开始读
    in_atoms = False
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("_atom_site.type_symbol"):
            in_atoms = True
            continue
        if in_atoms and s.startswith("_atom_site"):
            continue
        if in_atoms and s and not s.startswith("_") and not s.startswith("loop_"):
            parts = s.split()
            if len(parts) >= 4:
                sym = parts[0]
                # 简化处理: 假设坐标在固定列, 失败就跳过
                try:
                    fx, fy, fz = float(parts[-3]), float(parts[-2]), float(parts[-1])
                except ValueError:
                    continue
                pos = [fx, fy, fz]  # 分数坐标
                if cell:
                    # 笛卡尔近似 (正交晶系), 严格的话需要 pymatgen
                    pos = [fx * la, fy * lb, fz * lc] if (la and lb and lc) else [fx, fy, fz]
                atoms.append({
                    "element": sym,
                    "position": pos,
                    "radius": _covalent_radius(sym),
                    "color": _element_color(sym),
                })
    return atoms, cell, "CIF structure"


def _parse_via_pymatgen(text: str, fmt: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    """用 pymatgen 解析结构, 不可用则抛错."""
    from pymatgen.core import Structure  # 延迟导入

    from io import StringIO
    s = Structure.from_str(text, fmt=fmt if fmt != "poscar" else "poscar")
    atoms: list[dict[str, Any]] = []
    for site in s:
        sym = site.specie.symbol
        atoms.append({
            "element": sym,
            "position": list(site.coords),
            "radius": _covalent_radius(sym),
            "color": _element_color(sym),
        })
    cell = None
    if s.lattice is not None:
        mat = s.lattice.matrix
        cell = {
            "a": list(mat[0]),
            "b": list(mat[1]),
            "c": list(mat[2]),
            "scale": 1.0,
        }
    return atoms, cell, "pymatgen structure"


def _detect_bonds(atoms: list[dict[str, Any]]) -> list[list[int]]:
    """根据共价半径判定成键, 距离 < (ri + rj) * tolerance."""
    bonds: list[list[int]] = []
    n = len(atoms)
    # 大体系用简单 O(n^2), 量级够用; 真正大体系前端会切到 instanced mesh
    if n > 5000:
        return bonds
    for i in range(n):
        ri = atoms[i].get("radius", _DEFAULT_RADIUS)
        pi = atoms[i]["position"]
        for j in range(i + 1, n):
            rj = atoms[j].get("radius", _DEFAULT_RADIUS)
            pj = atoms[j]["position"]
            dx = pi[0] - pj[0]
            dy = pi[1] - pj[1]
            dz = pi[2] - pj[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            cutoff = (ri + rj) * _BOND_TOLERANCE
            if 0.1 < dist < cutoff:
                bonds.append([i, j])
    return bonds


# ── 轨迹解析 ────────────────────────────────────────────────────

def _parse_trajectory(text: str, fmt: str) -> dict[str, Any]:
    """Parse a trajectory file into a list of frames."""
    fmt = fmt.lower()
    if fmt == "xdatcar":
        return _parse_xdatcar(text)
    if fmt in ("xyz", "traj"):
        return _parse_xyz_trajectory(text)
    return {"frames": [], "error": f"unsupported trajectory format: {fmt}"}


def _parse_xdatcar(text: str) -> dict[str, Any]:
    """解析 VASP XDATCAR, 每一帧是 Direct 坐标."""
    lines = text.strip().splitlines()
    if len(lines) < 8:
        return {"frames": [], "error": "XDATCAR too short"}
    title = lines[0].strip()
    scale = float(lines[1].strip() or "1.0")
    a = [float(x) * scale for x in lines[2].split()]
    b = [float(x) * scale for x in lines[3].split()]
    c = [float(x) * scale for x in lines[4].split()]
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    n_atoms = sum(counts)
    cell = {"a": a, "b": b, "c": c, "scale": scale}

    frames: list[dict[str, Any]] = []
    i = 7
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # 帧头类似 "X 1.0" 或者直接是坐标
        if line.lower().startswith("direct") or re.match(r"^[A-Za-z]+\s+[\d.]+$", line):
            i += 1
            positions: list[list[float]] = []
            for _ in range(n_atoms):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    fx, fy, fz = float(parts[0]), float(parts[1]), float(parts[2])
                    cx = fx * a[0] + fy * b[0] + fz * c[0]
                    cy = fx * a[1] + fy * b[1] + fz * c[1]
                    cz = fx * a[2] + fy * b[2] + fz * c[2]
                    positions.append([cx, cy, cz])
                i += 1
            if positions:
                frames.append({
                    "positions": positions,
                    "energy": None,
                    "step": len(frames),
                })
        else:
            i += 1
    return {"frames": frames, "n_atoms": n_atoms, "cell": cell, "species": species, "title": title}


def _parse_xyz_trajectory(text: str) -> dict[str, Any]:
    """多帧 XYZ 轨迹: 每帧一个标准 XYZ 块, 注释行可以带能量."""
    lines = text.strip().splitlines()
    frames: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        try:
            n = int(lines[i].strip())
        except (ValueError, IndexError):
            break
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        # 注释行里找 energy=... 或直接是浮点数
        energy = None
        m = re.search(r"energy\s*=\s*(-?\d+\.?\d*)", comment, re.IGNORECASE)
        if m:
            energy = float(m.group(1))
        else:
            try:
                energy = float(comment.strip())
            except ValueError:
                pass
        positions: list[list[float]] = []
        for k in range(n):
            idx = i + 2 + k
            if idx >= len(lines):
                break
            parts = lines[idx].strip().split()
            if len(parts) >= 4:
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if positions:
            frames.append({"positions": positions, "energy": energy, "step": len(frames)})
        i += 2 + n
    return {"frames": frames, "n_atoms": len(frames[0]["positions"]) if frames else 0}


# ── Pydantic 请求模型 ────────────────────────────────────────────

class LoadRequest(BaseModel):
    file_path: str | None = Field(None, description="结构文件绝对路径")
    content: str | None = Field(None, description="内联结构文本, 与 file_path 二选一")
    format: Literal["auto", "poscar", "vasp", "cif", "xyz"] = Field(
        "auto", description="结构格式, auto 会按文件后缀猜测"
    )


class TrajectoryRequest(BaseModel):
    file_path: str | None = None
    content: str | None = None
    format: Literal["xdatcar", "xyz", "traj"] = "xdatcar"


# ── REST 端点 ────────────────────────────────────────────────────

@router.get("/elements")
async def get_elements() -> dict[str, Any]:
    """返回元素表 (CPK 颜色 + 共价半径), 供前端初始化用."""
    elements = []
    for sym, color in _ELEMENT_COLORS.items():
        elements.append({
            "symbol": sym,
            "color": color,
            "covalent_radius": _COVALENT_RADII.get(sym, _DEFAULT_RADIUS),
        })
    return {
        "elements": elements,
        "default_color": _DEFAULT_COLOR,
        "default_radius": _DEFAULT_RADIUS,
        "bond_tolerance": _BOND_TOLERANCE,
    }


@router.post("/load", dependencies=[Depends(require_api_key)])
async def load_structure(req: LoadRequest) -> dict[str, Any]:
    """加载结构文件, 返回原子坐标 / 键 / 晶胞."""
    if req.content:
        text = req.content
        fmt = req.format if req.format != "auto" else "xyz"
    elif req.file_path:
        p = Path(req.file_path).resolve()
        # ponytail: path traversal guard — only allow files under workspace
        # Upgrade: if multi-tenant needed, check per-user sandbox root.
        workspace = Path.cwd().resolve()
        try:
            p.relative_to(workspace)
        except ValueError:
            return {"error": "file_path must be within workspace"}
        if not p.exists():
            return {"error": f"file not found: {req.file_path}"}
        text = p.read_text(encoding="utf-8", errors="replace")
        if req.format != "auto":
            fmt = req.format
        else:
            suf = p.suffix.lower().lstrip(".")
            fmt = {"vasp": "poscar", "poscar": "poscar", "cif": "cif"}.get(suf, "xyz")
    else:
        return {"error": "either file_path or content must be provided"}

    try:
        result = _parse_structure(text, fmt)
        return result
    except Exception as e:
        logger.exception("failed to parse structure")
        return {"error": f"parse failed: {e}"}


@router.post("/trajectory", dependencies=[Depends(require_api_key)])
async def load_trajectory(req: TrajectoryRequest) -> dict[str, Any]:
    """加载轨迹文件, 返回帧列表."""
    if req.content:
        text = req.content
    elif req.file_path:
        p = Path(req.file_path).resolve()
        workspace = Path.cwd().resolve()
        try:
            p.relative_to(workspace)
        except ValueError:
            return {"error": "file_path must be within workspace"}
        if not p.exists():
            return {"error": f"file not found: {req.file_path}"}
        text = p.read_text(encoding="utf-8", errors="replace")
    else:
        return {"error": "either file_path or content must be provided"}

    try:
        return _parse_trajectory(text, req.format)
    except Exception as e:
        logger.exception("failed to parse trajectory")
        return {"error": f"trajectory parse failed: {e}"}


# ── WebSocket: 实时原子位置 + 力的回传 ──────────────────────────────
# 协议设计参考 MolGame: 双向 JSON 消息, 客户端发 force, 服务端推 frame/telemetry

# 每个 session 维护一个 force 队列, 模拟器消费后回推位置
# 这里实现一个内置的 mock 模拟器 (谐振子), 真实 MD 引擎可以替换 _SIM_HOST
class _MockSim:
    """简单的谐振子模拟器, 用于在没有真实 MD 引擎时演示实时流."""

    def __init__(self, positions: list[list[float]], masses: list[float]) -> None:
        self.positions = [list(p) for p in positions]
        self.velocities = [[0.0, 0.0, 0.0] for _ in positions]
        self.masses = masses
        self.forces: list[list[float]] = [[0.0, 0.0, 0.0] for _ in positions]
        self.energy = 0.0
        self.temperature = 0.0
        self.step = 0

    def apply_force(self, atom_idx: int, force: list[float]) -> None:
        if 0 <= atom_idx < len(self.forces):
            self.forces[atom_idx] = list(force)

    def step_once(self, dt: float = 0.5) -> None:
        # 弹簧力 + 外力, k=1, 平衡位置 = 初始位置
        for i, (p, v, m, f) in enumerate(zip(self.positions, self.velocities, self.masses, self.forces)):
            ax = (-p[0] + f[0]) / m
            ay = (-p[1] + f[1]) / m
            az = (-p[2] + f[2]) / m
            v[0] += ax * dt
            v[1] += ay * dt
            v[2] += az * dt
            p[0] += v[0] * dt
            p[1] += v[1] * dt
            p[2] += v[2] * dt
            # 力消耗一次后衰减
            f[0] *= 0.5
            f[1] *= 0.5
            f[2] *= 0.5
        # 能量 = 0.5 * (m * v^2 + k * x^2)
        ke = sum(0.5 * m * (v[0]**2 + v[1]**2 + v[2]**2)
                 for m, v in zip(self.masses, self.velocities))
        pe = sum(0.5 * (p[0]**2 + p[1]**2 + p[2]**2) for p in self.positions)
        self.energy = ke + pe
        # 温度 ~ <KE> / kB, 这里只给个量级
        n = max(len(self.masses), 1)
        self.temperature = (2.0 / 3.0) * ke / n
        self.step += 1


@ws_router.websocket("/ws/viewer3d")
async def viewer3d_websocket(websocket: WebSocket) -> None:
    """实时 3D 查看器的 WebSocket 端点.

    客户端连接后发送 hello 消息指定结构, 服务端进入循环:
    * 推送 frame (原子位置 + 能量 + 温度)
    * 接收 force 事件并转发到模拟器
    """
    await websocket.accept()
    sim: _MockSim | None = None
    stream_task: asyncio.Task | None = None
    # 用户施加的力队列: 原子索引 -> 力向量
    force_queue: asyncio.Queue[tuple[int, list[float]]] = asyncio.Queue()

    async def _stream_frames() -> None:
        """每 50ms 推一帧, 模拟实时 MD."""
        try:
            while sim is not None:
                # 先消费用户施加的力
                while not force_queue.empty():
                    idx, f = await force_queue.get()
                    sim.apply_force(idx, f)
                sim.step_once(dt=0.05)
                msg = {
                    "type": "frame",
                    "positions": sim.positions,
                    "energy": sim.energy,
                    "temperature": sim.temperature,
                    "step": sim.step,
                    "timestamp": time.time(),
                }
                await websocket.send_text(json.dumps(msg))
                await asyncio.sleep(0.05)  # 20 fps
        except Exception as e:
            logger.warning("viewer3d stream stopped: %s", e)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "invalid JSON"}))
                continue

            mtype = msg.get("type", "")

            if mtype == "hello":
                # 初始化: 解析结构并启动模拟器
                content = msg.get("content")
                fmt = msg.get("format", "xyz")
                file_path = msg.get("file_path")
                if file_path and not content:
                    p = Path(file_path)
                    content = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
                if not content:
                    await websocket.send_text(json.dumps({"type": "error", "message": "no structure content"}))
                    continue
                try:
                    struct = _parse_structure(content, fmt)
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": f"parse failed: {e}"}))
                    continue
                atoms = struct["atoms"]
                # 质量用原子序数近似, H=1, C=12 之类, 这里简化为 1.0
                masses = [1.0 for _ in atoms]
                sim = _MockSim([a["position"] for a in atoms], masses)
                # 发送初始结构
                await websocket.send_text(json.dumps({
                    "type": "structure",
                    "atoms": atoms,
                    "bonds": struct["bonds"],
                    "cell": struct["cell"],
                    "title": struct["title"],
                    "streaming": msg.get("stream", True),
                }))
                if msg.get("stream", True):
                    stream_task = asyncio.create_task(_stream_frames())

            elif mtype == "force":
                # 用户拖拽原子施加力
                if sim is None:
                    await websocket.send_text(json.dumps({"type": "error", "message": "no simulation running"}))
                    continue
                atom_idx = int(msg.get("atom_idx", -1))
                force = msg.get("force", [0.0, 0.0, 0.0])
                await force_queue.put((atom_idx, force))
                await websocket.send_text(json.dumps({
                    "type": "force_ack",
                    "atom_idx": atom_idx,
                    "force": force,
                }))

            elif mtype == "pause":
                if stream_task and not stream_task.done():
                    stream_task.cancel()
                    stream_task = None
                await websocket.send_text(json.dumps({"type": "paused"}))

            elif mtype == "resume":
                if sim and (stream_task is None or stream_task.done()):
                    stream_task = asyncio.create_task(_stream_frames())
                await websocket.send_text(json.dumps({"type": "resumed"}))

            elif mtype == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

            else:
                await websocket.send_text(json.dumps({"type": "error", "message": f"unknown type: {mtype}"}))

    except WebSocketDisconnect:
        logger.info("viewer3d websocket disconnected")
    finally:
        if stream_task and not stream_task.done():
            stream_task.cancel()
