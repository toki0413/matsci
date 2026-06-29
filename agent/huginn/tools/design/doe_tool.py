"""实验设计 (Design of Experiments) 工具.

纯算法实现，不调用 LLM。支持：
- 全因子设计 (factorial)
- 部分因子设计 2^(k-p) (fractional)
- 正交表 L8/L16/L9/L27 (orthogonal)
- 响应面设计 CCD / Box-Behnken (rsm)
- 实验顺序随机化 (randomize)

只依赖标准库 (itertools / math / random)，所有正交表与生成元
都是预定义的标准表，构造过程确定性强。
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


# ── 预定义正交表 ──────────────────────────────────────────────────────

# L8(2^7)：8 runs × 7 columns，2 水平
_L8: list[list[int]] = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 2, 2, 2, 2],
    [1, 2, 2, 1, 1, 2, 2],
    [1, 2, 2, 2, 2, 1, 1],
    [2, 1, 2, 1, 2, 1, 2],
    [2, 1, 2, 2, 1, 2, 1],
    [2, 2, 1, 1, 2, 2, 1],
    [2, 2, 1, 2, 1, 1, 2],
]

# L9(3^4)：9 runs × 4 columns，3 水平
_L9: list[list[int]] = [
    [1, 1, 1, 1],
    [1, 2, 2, 2],
    [1, 3, 3, 3],
    [2, 1, 2, 3],
    [2, 2, 3, 1],
    [2, 3, 1, 2],
    [3, 1, 3, 2],
    [3, 2, 1, 3],
    [3, 3, 2, 1],
]

# L16(2^15)：16 runs × 15 columns，2 水平
# 列序对应 2^4 全因子 Yates 序的全部 15 个非空交互列
# (A, B, AB, C, AC, BC, ABC, D, AD, BD, ABD, CD, ACD, BCD, ABCD)
_L16: list[list[int]] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
    [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
    [1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1],
    [1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1],
    [1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
    [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    [2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1],
    [2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1],
    [2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2],
    [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
    [2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2],
    [2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2],
    [2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1],
]


def _build_l27() -> list[list[int]]:
    """构造标准 Taguchi L27(3^13) 正交表.

    用 GF(3) 上 13 个两两线性无关的线性型生成，等价于标准正交表
    OA(27, 13, 3, 2)。水平取 {1,2,3}，共 27 runs × 13 columns。
    """
    # 13 个线性型 (a1,a2,a3) 对应 a1*x1 + a2*x2 + a3*x3 (mod 3)
    forms = [
        (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 2, 0),
        (0, 0, 1), (1, 0, 1), (1, 0, 2), (0, 1, 1), (0, 1, 2),
        (1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2),
    ]
    table: list[list[int]] = []
    for x1 in range(3):
        for x2 in range(3):
            for x3 in range(3):
                row = [
                    (a1 * x1 + a2 * x2 + a3 * x3) % 3 + 1
                    for a1, a2, a3 in forms
                ]
                table.append(row)
    return table


_L27: list[list[int]] = _build_l27()

_ORTHOGONAL_TABLES: dict[str, list[list[int]]] = {
    "L8": _L8, "L9": _L9, "L16": _L16, "L27": _L27,
}


# ── 部分因子生成元表 ────────────────────────────────────────────────
# 键 (k, p)：k 因子、p 个生成元；值为 (生成元列表, 分辨度)
# 生成元用基础因子索引的元组表示，对应列 = 这些基础列的乘积 (±1)
_FRACTIONAL_DESIGNS: dict[tuple[int, int], tuple[list[tuple[int, ...]], str]] = {
    (3, 1): ([(0, 1)], "III"),
    (4, 1): ([(0, 1, 2)], "IV"),
    (5, 1): ([(0, 1, 2, 3)], "V"),
    (5, 2): ([(0, 1), (0, 2)], "III"),
    (6, 1): ([(0, 1, 2, 3, 4)], "VI"),
    (6, 2): ([(0, 1, 2), (1, 2, 3)], "IV"),
    (6, 3): ([(0, 1), (0, 2), (1, 2)], "III"),
    (7, 1): ([(0, 1, 2, 3, 4, 5)], "VII"),
    (7, 2): ([(0, 1, 2), (0, 3, 4)], "IV"),
    (7, 3): ([(0, 1, 2), (0, 3, 4), (1, 3, 5)], "IV"),
    (7, 4): ([(0, 1), (0, 2), (1, 2), (0, 1, 2)], "III"),
    (8, 2): ([(0, 1, 2, 3), (0, 1, 4, 5)], "V"),
    (8, 3): ([(0, 1, 2), (0, 3, 4), (1, 3, 5)], "IV"),
    (8, 4): ([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], "IV"),
}


class DOEInput(BaseModel):
    action: Literal["factorial", "fractional", "orthogonal", "rsm", "randomize"] = Field(
        ..., description="实验设计动作类型"
    )
    factors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="因子列表，每个含 name/type/levels/low/high",
    )
    design_type: str = Field(
        default="",
        description="设计子类型: full/half/quarter/L8/L16/L9/L27/CCD/Box-Behnken",
    )
    runs: int | None = Field(
        default=None, description="期望实验次数 (部分因子设计用)"
    )
    center_points: int = Field(
        default=3, ge=0, description="中心点重复次数 (RSM 用，默认 3)"
    )
    randomize: bool = Field(
        default=True, description="是否随机化实验顺序 (默认 True)"
    )
    seed: int | None = Field(default=None, description="随机种子")
    design_matrix: list[dict[str, Any]] | None = Field(
        default=None, description="已有实验矩阵 (randomize 动作用)",
    )


def _factor_levels(f: dict[str, Any]) -> list[Any]:
    """从因子描述里取出水平列表，缺 levels 时退化为 [low, high]."""
    if f.get("levels") is not None:
        return list(f["levels"])
    low = f.get("low")
    high = f.get("high")
    if low is not None and high is not None:
        return [low, high]
    raise ValueError(f"因子 {f.get('name')!r} 缺少 levels 或 low/high")


def _maybe_shuffle(rows: list[Any], args: DOEInput) -> None:
    """按 args.randomize / args.seed 原地打乱行序."""
    if args.randomize:
        rng = random.Random(args.seed)
        rng.shuffle(rows)


class DOETool(HuginnTool):
    """实验设计工具 —— 生成全因子 / 部分因子 / 正交表 / 响应面方案矩阵."""

    name = "doe_tool"
    category = "design"
    description = (
        "实验设计工具，支持全因子/部分因子/正交表/响应面设计，"
        "生成实验方案矩阵。纯算法，不调 LLM。"
    )
    input_schema = DOEInput
    read_only = True

    def is_read_only(self, args: DOEInput) -> bool:
        return True

    async def call(self, args: DOEInput, context: ToolContext) -> ToolResult:
        # 兼容调用方直接传 dict 的情况
        if isinstance(args, dict):
            args = DOEInput(**args)

        try:
            if args.action == "factorial":
                return self._factorial(args)
            if args.action == "fractional":
                return self._fractional(args)
            if args.action == "orthogonal":
                return self._orthogonal(args)
            if args.action == "rsm":
                return self._rsm(args)
            if args.action == "randomize":
                return self._randomize_action(args)
            raise ValueError(f"未知 action: {args.action}")
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"DOE 工具失败: {exc}")

    # ── 全因子 ────────────────────────────────────────────────────
    def _factorial(self, args: DOEInput) -> ToolResult:
        factors = args.factors
        if not factors:
            raise ValueError("至少需要一个因子")
        names = [f["name"] for f in factors]
        levels_lists = [_factor_levels(f) for f in factors]

        rows = [dict(zip(names, combo)) for combo in itertools.product(*levels_lists)]
        _maybe_shuffle(rows, args)

        n_runs = len(rows)
        return ToolResult(
            data={
                "action": "factorial",
                "design_matrix": rows,
                "n_runs": n_runs,
                "design_info": {
                    "design_type": args.design_type or "full",
                    "n_factors": len(factors),
                    "levels_per_factor": [len(lv) for lv in levels_lists],
                },
                "summary": f"全因子设计：{len(factors)} 因子，共 {n_runs} runs",
            },
            success=True,
        )

    # ── 部分因子 2^(k-p) ──────────────────────────────────────────
    def _fractional(self, args: DOEInput) -> ToolResult:
        factors = args.factors
        k = len(factors)
        if k < 3:
            raise ValueError("部分因子设计至少需要 3 个因子")

        # 确定 p：优先看 design_type，再看 runs
        dt = (args.design_type or "").lower()
        if "half" in dt:
            p = 1
        elif "quarter" in dt:
            p = 2
        elif args.runs is not None:
            log2_runs = math.log2(args.runs)
            if not log2_runs.is_integer():
                raise ValueError(f"runs={args.runs} 不是 2 的幂")
            p = k - int(log2_runs)
        else:
            p = 1  # 默认半因子

        if p < 0:
            raise ValueError("runs 大于全因子实验数，请改用 factorial")
        if k - p < 3:
            raise ValueError("基础设计 2^(k-p) 至少需要 8 runs (k-p>=3)")

        key = (k, p)
        if key not in _FRACTIONAL_DESIGNS:
            available = ", ".join(f"2^{a}-{b}" for a, b in sorted(_FRACTIONAL_DESIGNS))
            raise ValueError(f"没有预定义的 2^({k}-{p}) 设计；可选: {available}")
        generators, resolution = _FRACTIONAL_DESIGNS[key]

        n_base = k - p
        runs = 2 ** n_base

        # 基础设计 ±1 矩阵 (Yates 序，第一个基础因子变化最慢)
        base: list[list[int]] = []
        for r in range(runs):
            row = []
            for b in range(n_base):
                period = 2 ** (n_base - b - 1)
                row.append(-1 if (r // period) % 2 == 0 else 1)
            base.append(row)

        # 生成元列 = 对应基础列的乘积
        full = [row[:] for row in base]
        for gen in generators:
            for r in range(runs):
                val = 1
                for idx in gen:
                    val *= base[r][idx]
                full[r].append(val)

        # ±1 映射到因子实际水平
        rows: list[dict[str, Any]] = []
        for r in range(runs):
            run: dict[str, Any] = {}
            for i, f in enumerate(factors):
                levels = _factor_levels(f)
                if len(levels) != 2:
                    raise ValueError(f"部分因子设计要求因子 {f['name']!r} 为 2 水平")
                run[f["name"]] = levels[0] if full[r][i] == -1 else levels[1]
            rows.append(run)

        _maybe_shuffle(rows, args)

        base_names = [factors[i]["name"] for i in range(n_base)]
        gen_names = [factors[n_base + i]["name"] for i in range(p)]
        gen_strs = []
        for i, gen in enumerate(generators):
            rhs = "".join(base_names[idx] for idx in gen)
            gen_strs.append(f"{gen_names[i]}={rhs}")

        return ToolResult(
            data={
                "action": "fractional",
                "design_matrix": rows,
                "n_runs": len(rows),
                "generators": gen_strs,
                "resolution": resolution,
                "design_info": {
                    "design_type": f"2^({k}-{p})",
                    "k": k,
                    "p": p,
                    "resolution": resolution,
                    "base_factors": base_names,
                    "generators": gen_strs,
                },
                "summary": (
                    f"2^({k}-{p}) 部分因子设计，{len(rows)} runs，"
                    f"分辨度 {resolution}"
                ),
            },
            success=True,
        )

    # ── 正交表 ────────────────────────────────────────────────────
    def _orthogonal(self, args: DOEInput) -> ToolResult:
        factors = args.factors
        n = len(factors)
        if n == 0:
            raise ValueError("至少需要一个因子")

        dt = (args.design_type or "").upper().replace(" ", "")
        if dt in _ORTHOGONAL_TABLES:
            table_name = dt
        else:
            # 按因子水平数自动选表
            levels_counts = [len(_factor_levels(f)) for f in factors]
            if all(c == 2 for c in levels_counts):
                if n <= 7:
                    table_name = "L8"
                elif n <= 15:
                    table_name = "L16"
                else:
                    raise ValueError(f"2 水平因子数 {n} 超出 L16 容量 (15)")
            elif all(c == 3 for c in levels_counts):
                if n <= 4:
                    table_name = "L9"
                elif n <= 13:
                    table_name = "L27"
                else:
                    raise ValueError(f"3 水平因子数 {n} 超出 L27 容量 (13)")
            else:
                raise ValueError("正交表要求所有因子同为 2 水平或 3 水平")

        table = _ORTHOGONAL_TABLES[table_name]
        n_cols = len(table[0])
        if n > n_cols:
            raise ValueError(f"{table_name} 只有 {n_cols} 列，容纳不了 {n} 个因子")

        rows: list[dict[str, Any]] = []
        for table_row in table:
            run: dict[str, Any] = {}
            for i, f in enumerate(factors):
                levels = _factor_levels(f)
                lvl_idx = table_row[i]  # 1-based
                if lvl_idx > len(levels):
                    raise ValueError(
                        f"因子 {f['name']!r} 水平数 {len(levels)} 少于 {table_name} 要求"
                    )
                run[f["name"]] = levels[lvl_idx - 1]
            rows.append(run)

        _maybe_shuffle(rows, args)

        return ToolResult(
            data={
                "action": "orthogonal",
                "design_matrix": rows,
                "n_runs": len(rows),
                "table_name": table_name,
                "design_info": {
                    "table": table_name,
                    "n_factors": n,
                    "n_runs": len(rows),
                    "n_columns": n_cols,
                },
                "summary": f"正交表 {table_name}：{len(rows)} runs × {n} 因子",
            },
            success=True,
        )

    # ── 响应面 ────────────────────────────────────────────────────
    def _rsm(self, args: DOEInput) -> ToolResult:
        factors = args.factors
        k = len(factors)
        if k < 2:
            raise ValueError("响应面设计至少需要 2 个因子")

        design_type = (args.design_type or "CCD").upper().replace("-", "").replace("_", "")
        if design_type in ("CCD", "CENTRALCOMPOSITE"):
            return self._ccd(args, k, factors)
        if design_type in ("BB", "BOXBEHNKEN"):
            return self._box_behnken(args, k, factors)
        raise ValueError(f"不支持的 RSM 设计类型: {args.design_type!r}")

    def _ccd(self, args: DOEInput, k: int, factors: list[dict]) -> ToolResult:
        centers, half_ranges = self._factor_centers(factors)
        # 可旋转 α = (2^k)^(1/4)
        alpha = (2.0 ** k) ** 0.25

        def to_actual(coded: list[float]) -> dict[str, Any]:
            return {
                factors[i]["name"]: centers[i] + coded[i] * half_ranges[i]
                for i in range(k)
            }

        paired: list[tuple[dict[str, Any], str]] = []
        # 角点 2^k
        for combo in itertools.product([-1.0, 1.0], repeat=k):
            paired.append((to_actual(list(combo)), "factorial"))
        # 轴点 2k
        for i in range(k):
            for sign in (-1.0, 1.0):
                coded = [0.0] * k
                coded[i] = sign * alpha
                paired.append((to_actual(coded), "axial"))
        # 中心点
        for _ in range(args.center_points):
            paired.append((to_actual([0.0] * k), "center"))

        if args.randomize:
            rng = random.Random(args.seed)
            rng.shuffle(paired)

        rows = [p[0] for p in paired]
        point_types = [p[1] for p in paired]

        return ToolResult(
            data={
                "action": "rsm",
                "design_type": "CCD",
                "design_matrix": rows,
                "n_runs": len(rows),
                "alpha": alpha,
                "n_center": args.center_points,
                "design_info": {
                    "design_type": "CCD",
                    "alpha": alpha,
                    "n_center": args.center_points,
                    "n_factorial": 2 ** k,
                    "n_axial": 2 * k,
                    "rotatable": True,
                    "point_types": point_types,
                },
                "summary": (
                    f"CCD 中心复合设计：2^{k}+{2*k}+{args.center_points}"
                    f" = {len(rows)} runs，α={alpha:.4f}"
                ),
            },
            success=True,
        )

    def _box_behnken(self, args: DOEInput, k: int, factors: list[dict]) -> ToolResult:
        if k < 3:
            raise ValueError("Box-Behnken 设计至少需要 3 个因子")
        centers, half_ranges = self._factor_centers(factors)

        def to_actual(coded: list[float]) -> dict[str, Any]:
            return {
                factors[i]["name"]: centers[i] + coded[i] * half_ranges[i]
                for i in range(k)
            }

        paired: list[tuple[dict[str, Any], str]] = []
        # 每对因子做 2^2 子设计，其余因子保持中心
        for i, j in itertools.combinations(range(k), 2):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    coded = [0.0] * k
                    coded[i] = si
                    coded[j] = sj
                    paired.append((to_actual(coded), "edge"))
        for _ in range(args.center_points):
            paired.append((to_actual([0.0] * k), "center"))

        if args.randomize:
            rng = random.Random(args.seed)
            rng.shuffle(paired)

        rows = [p[0] for p in paired]
        point_types = [p[1] for p in paired]
        n_edge = 4 * (k * (k - 1) // 2)

        return ToolResult(
            data={
                "action": "rsm",
                "design_type": "Box-Behnken",
                "design_matrix": rows,
                "n_runs": len(rows),
                "alpha": None,
                "n_center": args.center_points,
                "design_info": {
                    "design_type": "Box-Behnken",
                    "alpha": None,
                    "n_center": args.center_points,
                    "n_edge_points": n_edge,
                    "point_types": point_types,
                },
                "summary": (
                    f"Box-Behnken 设计：{n_edge}+{args.center_points}"
                    f" = {len(rows)} runs"
                ),
            },
            success=True,
        )

    @staticmethod
    def _factor_centers(factors: list[dict]) -> tuple[list[float], list[float]]:
        """取每个因子的中心值和半幅，缺 low/high 时退化为中心 0、半幅 1."""
        centers: list[float] = []
        half_ranges: list[float] = []
        for f in factors:
            low = f.get("low")
            high = f.get("high")
            if low is None or high is None:
                centers.append(0.0)
                half_ranges.append(1.0)
            else:
                low_f = float(low)
                high_f = float(high)
                centers.append((low_f + high_f) / 2.0)
                half_ranges.append((high_f - low_f) / 2.0)
        return centers, half_ranges

    # ── 随机化已有方案 ────────────────────────────────────────────
    def _randomize_action(self, args: DOEInput) -> ToolResult:
        matrix = args.design_matrix
        if not matrix:
            raise ValueError("randomize 动作需要传入 design_matrix")
        n = len(matrix)
        rng = random.Random(args.seed)
        order = list(range(n))
        rng.shuffle(order)
        randomized = [matrix[i] for i in order]

        return ToolResult(
            data={
                "action": "randomize",
                "design_matrix": randomized,
                "randomized_matrix": randomized,
                "original_order": list(range(n)),
                "new_order": order,
                "n_runs": n,
                "design_info": {"seed": args.seed, "n_runs": n},
                "summary": f"已随机化 {n} 个实验的运行顺序",
            },
            success=True,
        )
