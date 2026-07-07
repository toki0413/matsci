"""随机非线性动力学发现工具 —— 从含噪时间序列里反推控制方程.

思路取自 Nandor et al. "A machine learning framework for uncovering
stochastic nonlinear dynamics from noisy data" (Chaos 2024): 先把含噪
轨迹的数值导数估出来, 再在候选函数库上做稀疏回归筛出少数重要项,
拼回 dx/dt = ... 的符号形式.

这里不引入深度符号回归 (太重), 改用经典 SINDy (Brunton & Kutz, PNAS 2016):
  - 候选库: 多项式 (含交叉项) + sin/cos + exp
  - 稀疏回归: STLSQ (顺序阈值最小二乘), 只依赖 numpy lstsq;
    若环境里有 sklearn 则优先用 Lasso, threshold 作 alpha 初值.
"""

from __future__ import annotations

import csv
from collections import Counter
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class DynamicsDiscoveryInput(BaseModel):
    action: str = Field(
        default="discover",
        description="discover | validate | load_lammps_dump",
    )
    # 数据来源: 二选一
    data_file: str | None = Field(
        default=None,
        description="CSV 或 .npy 文件路径. .npy 视作 2D 数组, 列号当列名用",
    )
    data_json: dict[str, list[float]] | None = Field(
        default=None,
        description="内联数据 {列名: [值]}, 含时间列",
    )
    # 列指定
    time_column: str = Field(
        default="",
        description="时间列名 (CSV/data_json) 或列号字符串 (.npy). 留空则取第一列",
    )
    value_columns: list[str] | None = Field(
        default=None,
        description="状态变量列名/列号. 留空则取时间列之外的全部列",
    )
    # 候选库控制
    max_order: int = Field(
        default=2, ge=1, le=5,
        description="多项式最高次数 (含交叉项, 如 2 会生成 x0*x1)",
    )
    include_trig: bool = Field(
        default=False,
        description="候选库加入 sin/cos. 小幅度数据下 sin(x)≈x 会与多项式共线, 按需开启",
    )
    include_exp: bool = Field(
        default=False,
        description="候选库加入 exp. exp(x) 对小 x 近似常数, 易与常数项共线, 按需开启",
    )
    # 稀疏化 / 导数
    threshold: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description=(
            "稀疏阈值, 相对量: 每个变量保留系数 >= threshold * 该变量最大系数的项 "
            "(0.05 = 砍掉不到最大系数 5% 的项). 同时映射成 Lasso alpha"
        ),
    )
    smooth: bool = Field(
        default=True,
        description="用 Savitzky-Golay 平滑求导, 抑制噪声; 点太少时自动退回 np.gradient",
    )
    # validate 专用: discover 返回的 terms + coefficients
    terms: list[str] | None = Field(
        default=None, description="discover 返回的候选项名列表",
    )
    coefficients: dict[str, list[float]] | None = Field(
        default=None,
        description="各状态变量对 terms 的系数, {变量名: [与 terms 对齐的系数]}",
    )


class DynamicsDiscoveryTool(HuginnTool):
    """从含噪时间序列发现非线性 ODE 控制方程 (轻量 SINDy)."""

    name = "dynamics_discovery_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.HYPOTHESIS, ResearchPhase.VALIDATION}),
    )
    description = (
        "Discover governing ODEs from noisy time-series via sparse "
        "identification (SINDy). action=discover fits dx/dt=f(x); "
        "action=validate integrates the found equations and scores against data."
    )
    input_schema = DynamicsDiscoveryInput

    # ── 数据加载 ───────────────────────────────────────────────────

    def _load_series(
        self, args: DynamicsDiscoveryInput
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """返回 (t, X, col_names). t 形状 (n,), X 形状 (n, m)."""
        if args.data_json:
            d = args.data_json
            tkey = args.time_column or "t"
            if tkey not in d:
                raise ValueError(f"time column '{tkey}' not in data_json keys {list(d)}")
            t = np.asarray(d[tkey], dtype=np.float64)
            cols = args.value_columns or [k for k in d if k != tkey]
            if not cols:
                raise ValueError("no value columns found (only time column present)")
            X = np.column_stack([np.asarray(d[c], dtype=np.float64) for c in cols])
            return t, X, list(cols)

        if not args.data_file:
            raise ValueError("either data_file or data_json must be provided")

        path = Path(args.data_file)
        if not path.exists():
            raise FileNotFoundError(f"data file not found: {path}")

        if path.suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
            if arr.ndim != 2:
                raise ValueError(f".npy must be 2D, got {arr.ndim}D")
            tidx = int(args.time_column) if args.time_column else 0
            t = arr[:, tidx].astype(np.float64)
            if args.value_columns:
                vidx = [int(c) for c in args.value_columns]
            else:
                vidx = [i for i in range(arr.shape[1]) if i != tidx]
            X = arr[:, vidx].astype(np.float64)
            return t, X, [f"x{i}" for i in vidx]

        # CSV
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise ValueError("CSV is empty")
        fields = reader.fieldnames or []
        tkey = args.time_column or fields[0]
        if tkey not in fields:
            raise ValueError(f"time column '{tkey}' not in CSV header {fields}")
        t = np.array([float(r[tkey]) for r in rows], dtype=np.float64)
        cols = args.value_columns or [c for c in fields if c != tkey]
        if not cols:
            raise ValueError("no value columns (only time column present)")
        X = np.column_stack(
            [np.array([float(r[c]) for r in rows], dtype=np.float64) for c in cols]
        )
        return t, X, list(cols)

    # ── 候选函数库 ──────────────────────────────────────────────────

    @staticmethod
    def _collapse_powers(name: str) -> str:
        """'x0*x0' -> 'x0^2' 之类, 方便人读."""
        c = Counter(name.split("*"))
        out = []
        for var, k in c.items():
            out.append(var if k == 1 else f"{var}^{k}")
        return "*".join(out)

    def _build_library(
        self, X: np.ndarray, max_order: int, trig: bool, exp: bool
    ) -> tuple[list[str], np.ndarray]:
        """构造候选库 (names, Theta). Theta 形状 (n_samples, n_terms)."""
        n, m = X.shape
        names: list[str] = ["1"]
        cols: list[np.ndarray] = [np.ones(n)]

        for deg in range(1, max_order + 1):
            for combo in combinations_with_replacement(range(m), deg):
                col = np.prod(X[:, list(combo)], axis=1)
                names.append(self._collapse_powers("*".join(f"x{i}" for i in combo)))
                cols.append(col)

        if trig:
            for i in range(m):
                names.append(f"sin(x{i})")
                cols.append(np.sin(X[:, i]))
                names.append(f"cos(x{i})")
                cols.append(np.cos(X[:, i]))
        if exp:
            for i in range(m):
                names.append(f"exp(x{i})")
                cols.append(np.exp(X[:, i]))

        return names, np.column_stack(cols)

    # ── 导数估计 ────────────────────────────────────────────────────

    def _derivatives(self, X: np.ndarray, t: np.ndarray, smooth: bool) -> np.ndarray:
        """估 dX/dt. 含噪数据走 Savitzky-Golay (顺手平滑+求导), 否则 np.gradient."""
        from scipy.signal import savgol_filter

        n, m = X.shape
        # 假设近似均匀采样, 用平均 dt; SINDy 对微小不均匀不敏感
        dt = float(np.mean(np.diff(t)))
        if dt <= 0:
            raise ValueError("time must be strictly increasing")

        if smooth and n >= 7:
            # 窗口必须是奇数且 > polyorder(=2), 但不能超过样本数
            win = min(n if n % 2 == 1 else n - 1, 7)
            win = max(win, 5)
            dXdt = np.empty_like(X)
            for j in range(m):
                dXdt[:, j] = savgol_filter(X[:, j], win, 2, deriv=1, delta=dt)
            return dXdt
        # 点太少或关了平滑: 非均匀间距的中央差分
        return np.gradient(X, t, axis=0)

    # ── 稀疏回归 ────────────────────────────────────────────────────

    def _sparse_regression(
        self, Theta: np.ndarray, dXdt: np.ndarray, threshold: float
    ) -> np.ndarray:
        """解 dXdt ≈ Theta @ Xi, 返回稀疏 Xi (n_terms, n_vars).

        先把候选库每列归一化到单位 L2 范数再回归, 否则 exp 项数值范围远大于
        多项式项会撑爆条件数, 把真项淹没掉 (SINDy 的经典坑). 回归完再把系数
        缩回原始尺度, 方程里看到的系数就是原始单位的.

        优先 sklearn.Lasso (threshold 映射成 alpha); 不可用就退回 STLSQ,
        也就是 SINDy 论文里的顺序阈值最小二乘, 只要 numpy.
        """
        scales = np.linalg.norm(Theta, axis=0)
        scales[scales == 0] = 1.0
        Theta_n = Theta / scales
        Xi_n = self._solve(Theta_n, dXdt, threshold)
        # 撤回归一化: Xi_original = Xi_normalized / scale
        return Xi_n / scales[:, None]

    def _solve(
        self, Theta: np.ndarray, dXdt: np.ndarray, threshold: float
    ) -> np.ndarray:
        """在已归一化的库上跑稀疏回归 (Lasso 优先, STLSQ 兜底).

        threshold 是相对阈值 (保留 |coef| >= threshold * max|coef| 的项),
        Lasso 只用很小的 alpha 做数值稳定, 真正的稀疏化交给 _threshold().
        之前把 threshold 直接当 Lasso alpha 用, 0.05 的 alpha 对 y=2x 这种
        简单关系都过度收缩到 1.48.
        """
        try:
            from sklearn.linear_model import Lasso  # type: ignore

            # alpha 要小: Lasso 只负责 mild regularization, 真正的稀疏化
            # 由 _threshold() 按相对阈值做. 1e-4 是 sklearn 文档推荐的
            # 下限, 再小数值上和 OLS 没区别.
            alpha = 1e-4
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=50000)
            Xi = np.zeros((Theta.shape[1], dXdt.shape[1]))
            for j in range(dXdt.shape[1]):
                model.fit(Theta, dXdt[:, j])
                Xi[:, j] = model.coef_
            return self._threshold(Xi, threshold)
        except ImportError:
            return self._stlsq(Theta, dXdt, threshold)

    @staticmethod
    def _keep_mask(coefs: np.ndarray, threshold: float) -> np.ndarray:
        """相对阈值: 每列保留 |coef| >= threshold * max(|coef|) 的项.

        绝对阈值不跨尺度通用 (导数量级一变就失效), 改成相对最大系数的比例,
        0.05 就是砍掉不到最大系数 5% 的项. 这是稳健 SINDy 实现的常见做法.
        """
        mask = np.zeros(coefs.shape, dtype=bool)
        for j in range(coefs.shape[1]):
            col = np.abs(coefs[:, j])
            m = float(col.max())
            mask[:, j] = col >= threshold * m if m > 0 else False
        return mask

    @staticmethod
    def _threshold(Xi: np.ndarray, threshold: float) -> np.ndarray:
        """按相对阈值把小系数清零."""
        mask = DynamicsDiscoveryTool._keep_mask(Xi, threshold)
        Xi = Xi.copy()
        Xi[~mask] = 0.0
        return Xi

    @staticmethod
    def _stlsq(
        Theta: np.ndarray, dXdt: np.ndarray, threshold: float, max_iter: int = 20
    ) -> np.ndarray:
        """顺序阈值最小二乘 (Brunton 2016 SINDy 原版算法).

        每轮: 最小二乘解全量 -> 按相对阈值砍掉小系数 -> 只留幸存项重解 ->
        收敛即停. 相对阈值让幸存项重解后尺度变化也不会反复误杀.
        """
        Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]
        for _ in range(max_iter):
            big = DynamicsDiscoveryTool._keep_mask(Xi, threshold)
            Xi_new = np.zeros_like(Xi)
            for j in range(dXdt.shape[1]):
                mask = big[:, j]
                if mask.any():
                    Xi_new[mask, j] = np.linalg.lstsq(
                        Theta[:, mask], dXdt[:, j], rcond=None
                    )[0]
            if np.allclose(Xi_new, Xi):
                break
            Xi = Xi_new
        return Xi

    # ── 项求值 (供 validate 积分用) ──────────────────────────────────

    @staticmethod
    def _eval_term(name: str, x: np.ndarray) -> float:
        """在单点 x (一维状态向量) 上求一个候选项的值.

        支持的写法: '1', 'x0', 'x0*x1', 'x0^2', 'x1^3*x0', 'sin(x0)', 'cos(x0)', 'exp(x0)'.
        """
        if name == "1":
            return 1.0
        for fn in ("sin", "cos", "exp"):
            tag = f"{fn}(x"
            if name.startswith(tag) and name.endswith(")"):
                idx = int(name[len(tag):-1])
                return float(getattr(np, fn)(x[idx]))
        result = 1.0
        for factor in name.split("*"):
            if "^" in factor:
                var, exp = factor.split("^")
                idx = int(var[1:])
                result *= x[idx] ** int(exp)
            else:
                idx = int(factor[1:])
                result *= x[idx]
        return float(result)

    def _equation_string(
        self, var_idx: int, terms: list[str], coefs: np.ndarray
    ) -> str:
        """拼 'dx{i}/dt = 1.23*x0 - 0.50*sin(x1)' 这种字符串."""
        parts = []
        for name, c in zip(terms, coefs):
            if abs(c) < 1e-12:
                continue
            sign = "-" if c < 0 else "+"
            mag = abs(c)
            if name == "1":
                parts.append(f"{sign}{mag:.4f}")
            else:
                parts.append(f"{sign}{mag:.4f}*{name}")
        body = " ".join(parts).strip()
        if body.startswith("+"):
            body = body[1:].lstrip()
        if not body:
            body = "0"
        return f"dx{var_idx}/dt = {body}"

    # ── 入口 ────────────────────────────────────────────────────────

    async def call(
        self, args: DynamicsDiscoveryInput, context: ToolContext
    ) -> ToolResult:
        if args.action == "discover":
            return self._discover(args)
        if args.action == "validate":
            return self._validate(args)
        if args.action == "load_lammps_dump":
            return self._load_lammps_dump(args)
        return ToolResult(
            data=None, success=False, error=f"unknown action: {args.action}"
        )

    def _discover(self, args: DynamicsDiscoveryInput) -> ToolResult:
        try:
            t, X, col_names = self._load_series(args)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"data loading failed: {e}")

        if X.shape[0] < 5:
            return ToolResult(
                data=None, success=False,
                error=f"need >=5 samples for derivatives, got {X.shape[0]}",
            )

        try:
            dXdt = self._derivatives(X, t, args.smooth)
            terms, Theta = self._build_library(
                X, args.max_order, args.include_trig, args.include_exp
            )
            Xi = self._sparse_regression(Theta, dXdt, args.threshold)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"discovery failed: {e}")

        # 拟合质量: 每个状态变量一个 R² (基于残差 dXdt - Theta@Xi)
        r2: dict[str, float] = {}
        residuals: dict[str, list[float]] = {}
        equations: list[str] = []
        for j in range(X.shape[1]):
            pred = Theta @ Xi[:, j]
            ss_res = float(np.sum((dXdt[:, j] - pred) ** 2))
            ss_tot = float(np.sum((dXdt[:, j] - np.mean(dXdt[:, j])) ** 2))
            r2[f"x{j}"] = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
            residuals[f"x{j}"] = (dXdt[:, j] - pred).tolist()
            equations.append(self._equation_string(j, terms, Xi[:, j]))

        return ToolResult(
            data={
                "equations": equations,
                "terms": terms,
                "coefficients": {
                    f"x{j}": Xi[:, j].tolist() for j in range(X.shape[1])
                },
                "r2_score": r2,
                "residuals": residuals,
                "variables": col_names,
                "n_samples": int(X.shape[0]),
                "dt": float(np.mean(np.diff(t))),
                "library_size": len(terms),
            },
            success=True,
        )

    def _validate(self, args: DynamicsDiscoveryInput) -> ToolResult:
        """用 discover 出的方程积分, 跟真实数据对比."""
        if not args.terms or not args.coefficients:
            return ToolResult(
                data=None, success=False,
                error="validate needs terms + coefficients from discover",
            )
        try:
            t, X, col_names = self._load_series(args)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"data loading failed: {e}")

        from scipy.integrate import solve_ivp

        terms = args.terms
        var_names = list(args.coefficients.keys())  # x0, x1, ...
        n_vars = len(var_names)
        if X.shape[1] != n_vars:
            return ToolResult(
                data=None, success=False,
                error=f"value columns ({X.shape[1]}) != coefficients vars ({n_vars})",
            )

        # 每个变量的系数向量, 对齐到 terms
        coef_rows = [np.asarray(args.coefficients[v], dtype=np.float64) for v in var_names]
        if any(len(c) != len(terms) for c in coef_rows):
            return ToolResult(
                data=None, success=False,
                error="coefficient length must match terms length",
            )

        def rhs(_t, x) -> list:
            vals = np.array([self._eval_term(name, x) for name in terms])
            return [float(coef @ vals) for coef in coef_rows]

        sol = solve_ivp(
            rhs, (float(t[0]), float(t[-1])), X[0].tolist(),
            t_eval=t.tolist(), method="RK45", max_step=float(np.mean(np.diff(t))),
        )
        if not sol.success:
            return ToolResult(
                data=None, success=False,
                error=f"integration failed: {sol.message}",
            )

        pred = sol.y.T  # (n, n_vars)
        r2: dict[str, float] = {}
        residuals: dict[str, list[float]] = {}
        for j in range(n_vars):
            ss_res = float(np.sum((X[:, j] - pred[:, j]) ** 2))
            ss_tot = float(np.sum((X[:, j] - np.mean(X[:, j])) ** 2))
            r2[var_names[j]] = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
            residuals[var_names[j]] = (X[:, j] - pred[:, j]).tolist()

        return ToolResult(
            data={
                "r2_score": r2,
                "residuals": residuals,
                "predicted": pred.tolist(),
                "variables": col_names,
                "n_samples": int(X.shape[0]),
                "integration_status": sol.message,
            },
            success=True,
        )

    # ── LAMMPS dump → 动力学发现 ─────────────────────────────────────

    @staticmethod
    def _parse_lammps_dump(path: Path) -> list[dict]:
        """解析标准 LAMMPS dump 轨迹, 返回每帧 {timestep, columns, atoms}.

        dump 帧内小节顺序固定: TIMESTEP / NUMBER OF ATOMS / BOX BOUNDS /
        ATOMS. 用 marker 跳过 BOX BOUNDS 那几行, 不假设 box 行数 (偶尔有
        triclinic 的 tilt 因子多出一行). 只依赖 numpy + 文本解析.
        """
        lines = path.read_text(encoding="utf-8").splitlines()
        frames: list[dict] = []
        i, n = 0, len(lines)
        while i < n:
            if not lines[i].startswith("ITEM: TIMESTEP"):
                i += 1
                continue
            ts = int(lines[i + 1])
            # 找 NUMBER OF ATOMS
            j = i + 2
            while j < n and "NUMBER OF ATOMS" not in lines[j]:
                j += 1
            if j >= n:
                break
            n_atoms = int(lines[j + 1])
            # 找 ATOMS header (跳过 BOX BOUNDS, 行数不固定)
            k = j + 2
            while k < n and "ITEM: ATOMS" not in lines[k]:
                k += 1
            if k >= n:
                break
            cols = lines[k].split("ATOMS", 1)[1].strip().split()
            start = k + 1
            rows = lines[start : start + n_atoms]
            atoms = np.array(
                [[float(v) for v in r.split()] for r in rows], dtype=np.float64
            )
            frames.append({"timestep": ts, "columns": cols, "atoms": atoms})
            i = start + n_atoms
        return frames

    @staticmethod
    def _extract_lammps_quantities(
        frames: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """从轨迹帧抽 (t, Y, col_names).

        Y 列 = [avg_speed, msd, kinetic_energy]:
          - avg_speed: 平均速度模 <|v|>
          - msd: 均方位移 <|r(t)-r(0)|^2>
          - kinetic_energy: 0.5 * sum(m * |v|^2), 无 mass 列则 m=1 (LJ 约化单位常见)
        """
        if not frames:
            raise ValueError("no frames parsed from dump")
        cols = frames[0]["columns"]

        def idx(name: str) -> int:
            return cols.index(name) if name in cols else -1

        vx_i, vy_i, vz_i = idx("vx"), idx("vy"), idx("vz")
        if min(vx_i, vy_i, vz_i) < 0:
            raise ValueError(
                f"dump missing vx/vy/vz columns; got {cols}. "
                "dump 需带速度 (dump custom ... vx vy vz)"
            )
        # MSD 优先用 unwrapped 坐标, 否则周期边界会让位移跳变
        # ponytail: 没有 xu/yu/zu 时用 wrapped x/y/z, 大位移下 MSD 会偏小,
        #           升级路径: 让用户在 LAMMPS 里 dump xu yu zu
        px = "xu" if "xu" in cols else "x"
        py = "yu" if "yu" in cols else "y"
        pz = "zu" if "zu" in cols else "z"
        px_i, py_i, pz_i = idx(px), idx(py), idx(pz)
        mass_i = idx("mass")

        ref = frames[0]["atoms"][:, [px_i, py_i, pz_i]]
        nf = len(frames)
        t = np.empty(nf)
        Y = np.empty((nf, 3))
        for fi, fr in enumerate(frames):
            a = fr["atoms"]
            v = a[:, [vx_i, vy_i, vz_i]]
            t[fi] = float(fr["timestep"])
            Y[fi, 0] = float(np.mean(np.sqrt(np.sum(v * v, axis=1))))
            disp = a[:, [px_i, py_i, pz_i]] - ref
            Y[fi, 1] = float(np.mean(np.sum(disp * disp, axis=1)))
            m = a[:, mass_i] if mass_i >= 0 else 1.0
            Y[fi, 2] = float(0.5 * np.sum(m * np.sum(v * v, axis=1)))
        return t, Y, ["avg_speed", "msd", "kinetic_energy"]

    def _load_lammps_dump(self, args: DynamicsDiscoveryInput) -> ToolResult:
        """读 LAMMPS dump -> 抽物理量 -> 复用 _discover 跑 SINDy 方程发现."""
        if not args.data_file:
            return ToolResult(
                data=None, success=False,
                error="load_lammps_dump needs data_file (dump 路径)",
            )
        path = Path(args.data_file)
        if not path.exists():
            return ToolResult(
                data=None, success=False, error=f"dump file not found: {path}"
            )
        try:
            frames = self._parse_lammps_dump(path)
            t, Y, col_names = self._extract_lammps_quantities(frames)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"dump parse failed: {e}"
            )

        # 提取出的物理量塞进 data_json, 复用 discover 的 SINDy 流程
        payload: dict[str, list[float]] = {"t": t.tolist()}
        for j, name in enumerate(col_names):
            payload[name] = Y[:, j].tolist()
        discover_args = args.model_copy(
            update={"action": "discover", "data_json": payload, "data_file": None}
        )
        res = self._discover(discover_args)
        if not res.success:
            return res
        # 补上原始物理量, 下游可以直接拿来画图/分析
        res.data["lammps_quantities"] = {
            "timesteps": t.tolist(),
            "avg_speed": Y[:, 0].tolist(),
            "msd": Y[:, 1].tolist(),
            "kinetic_energy": Y[:, 2].tolist(),
            "n_frames": len(frames),
            "note": (
                "t 取 LAMMPS timestep 值; 无 mass 列时动能按 m=1 算. "
                "MSD 用 unwrapped 坐标 (xu/yu/zu), 没有则退回 wrapped."
            ),
        }
        return res

    def estimate_cost(self, args: DynamicsDiscoveryInput) -> dict[str, float] | None:
        # 纯本地最小二乘, 开销可忽略
        return {"cpu_hours": 0.0, "gpu_hours": 0.0, "walltime_hours": 0.0}


# ── 自检: 阻尼振荡器 + 噪声, 验证能稀疏恢复控制方程 ───────────────────
# dx0/dt = x1            (真值: 仅 x1 项, 系数 1.0)
# dx1/dt = -1.0*x0 - 0.2*x1
if __name__ == "__main__":
    from scipy.integrate import solve_ivp as _ivp

    rng = np.random.default_rng(0)
    n = 600
    t = np.linspace(0, 30, n)

    def true_rhs(_t, x) -> list:
        return [x[1], -1.0 * x[0] - 0.2 * x[1]]

    sol = _ivp(true_rhs, (t[0], t[-1]), [1.0, 0.0], t_eval=t, rtol=1e-8, atol=1e-10)
    X = sol.y.T + rng.normal(0, 0.01, (n, 2))  # 加 1% 噪声

    tool = DynamicsDiscoveryTool()
    res = tool._discover(DynamicsDiscoveryInput(
        data_json={"t": t.tolist(), "x0": X[:, 0].tolist(), "x1": X[:, 1].tolist()},
        max_order=2, threshold=0.05, smooth=True,
    ))
    assert res.success, res.error
    data = res.data
    print("equations:")
    for eq in data["equations"]:
        print(" ", eq)
    print("R2:", {k: round(v, 4) for k, v in data["r2_score"].items()})

    # dx0/dt = x1  -> 系数应接近 1.0, 且没有别的项
    coefs_x0 = dict(zip(data["terms"], data["coefficients"]["x0"]))
    assert abs(coefs_x0["x1"] - 1.0) < 0.05, f"dx0/dt x1 coef {coefs_x0['x1']} != 1.0"
    # dx1/dt = -x0 - 0.2*x1
    coefs_x1 = dict(zip(data["terms"], data["coefficients"]["x1"]))
    assert abs(coefs_x1["x0"] + 1.0) < 0.05, f"dx1/dt x0 coef {coefs_x1['x0']} != -1.0"
    assert abs(coefs_x1["x1"] + 0.2) < 0.03, f"dx1/dt x1 coef {coefs_x1['x1']} != -0.2"
    assert data["r2_score"]["x0"] > 0.9, f"x0 R2 low: {data['r2_score']['x0']}"
    assert data["r2_score"]["x1"] > 0.9, f"x1 R2 low: {data['r2_score']['x1']}"
    print("OK: recovered dx0/dt~=x1, dx1/dt~-x0-0.2x1, R2>0.9")

    # 用发现的方程积分, 跟原轨迹对比
    vres = tool._validate(DynamicsDiscoveryInput(
        action="validate",
        data_json={"t": t.tolist(), "x0": X[:, 0].tolist(), "x1": X[:, 1].tolist()},
        terms=data["terms"], coefficients=data["coefficients"],
    ))
    assert vres.success, vres.error
    print("validate R2:", {k: round(v, 4) for k, v in vres.data["r2_score"].items()})
    assert vres.data["r2_score"]["x0"] > 0.9, f"validate x0 R2 low"
    print("OK: validation integration tracks data")
