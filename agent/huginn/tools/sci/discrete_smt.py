"""DiscreteSMT — SAT/SMT 求解 + 稀疏约束图分析.

互补人类"lift 到连续"偏置的根工具. 离散约束求解的通用底座,
对应连续优化里的 scipy.optimize.

7 个 action:
  solve_sat          布尔可满足性
  solve_smt          SMT 理论求解 (LIA / BV / 数论)
  optimize           离散优化 (min/max)
  all_solutions      枚举所有解 (最多 N 个)
  verify_implication A ⟹ B 反例搜索
  analyze_sparsity   约束图稀疏性分析 (treewidth/连通分量)
  solve_decomposed   树分解分治求解 (大稀疏约束图加速)

稀疏计算结合:
  - analyze_sparsity: 用 scipy.sparse.csgraph 算连通分量 + degeneracy
    (treewidth 下界). networkx 构 primal graph.
  - solve_decomposed: 约束图连通分量独立求解再合并. 树宽 ≤ k 的图
    可以 O(n) 分解, z3 在每个连通分量上独立跑.

安全:
  - constraints 字符串走 z3 Python API eval, 白名单 namespace, 禁 builtins
  - timeout 强制 (z3 set_param)
  - 变量 ≤ 100, 约束 ≤ 500 (防 DoS)

接入点:
  - RedTeamReviewer._discrete_counterexample_scan 调 verify_implication
  - hypothesis_generator 后续可调 solve_sat 验证假设可满足性

设计原则 (ponytail):
  - z3 优先, sympy 不够才上 (z3 自带 BV/Arith/Array)
  - 不引 SageMath/GAP (重依赖)
  - 稀疏结构作为一等公民, 大实例自适应分解
  - ponytail: 树宽 > 10 走 LLM 跨域类比, 不强求解 (z3 会爆)

升级路径:
  - SyGuS (syntax-guided synthesis) — z3 已支持, 留后续 action
  -并行求解 — z3 parallel mode, 留后续
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

import z3
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


# ── 安全: z3 eval 白名单 ────────────────────────────────────
# ponytail: 白名单足够覆盖 SAT/SMT 常用构造, 不暴露任意 Python.
# 升级路径: 接 AST 解析器替代 eval (彻底无注入风险).
_Z3_NAMES = frozenset({
    # 逻辑
    "And", "Or", "Not", "Implies", "If", "Xor", "BoolVal",
    # 变量声明
    "Bool", "Int", "Real", "BitVec", "Const", "Array", "Function",
    # 算术
    "Sum", "Product", "Abs", "Mod", "Pow", "Sqrt",
    # 量词
    "ForAll", "Exists",
    # 转换
    "ToInt", "ToReal", "BV2Int", "Int2BV",
    # BV 操作
    "UGE", "ULE", "UGT", "ULT", "LShR", "RotateLeft", "RotateRight",
    "SignExt", "ZeroExt", "Extract", "Concat", "BVAdd", "BVMul",
    "BVUDiv", "BVURem", "BVSub", "BVAnd", "BVOr", "BVXor", "BVNot",
    "BVShl", "BVShr", "BVNeg",
    # 求解器辅助
    "Solver", "Optimize", "sat", "unsat", "unknown",
})


def _safe_eval(expr: str, variables: dict[str, Any]) -> Any:
    """在受限 namespace 里 eval z3 表达式字符串."""
    # ponytail: 禁 builtins + 只暴露白名单 z3 符号 + 用户变量.
    # 天花板: 字符串解析仍可能被构造恶意语法绕过 (但白名单足以挡常规注入).
    namespace: dict[str, Any] = {name: getattr(z3, name) for name in _Z3_NAMES if hasattr(z3, name)}
    namespace.update(variables)
    return eval(expr, {"__builtins__": {}}, namespace)


# ── 稀疏: 约束图构造 ───────────────────────────────────────

def _build_primal_graph(
    variables: list[dict], constraints: list[str]
) -> tuple["set[str]", "list[tuple[str, str]]"]:
    """从约束列表构造 primal graph (变量交互图).

    ponytail: 扫每个约束字符串里的变量名, 共现则连边.
    天花板: 字符串扫描不解析 AST, 可能漏/误 (如变量名是另一变量子串).
    升级路径: 用 z3 AST visitor 精确提取变量.
    """
    var_names = [v["name"] for v in variables]
    var_set = set(var_names)
    edges: list[tuple[str, str]] = []
    for c in constraints:
        # 简单 token 扫描: 找出 c 里出现的变量名
        present = [v for v in var_names if v in c]
        # 全连 (clique on present vars)
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                edges.append((present[i], present[j]))
    # 去重
    edges = list(set(edges))
    return var_set, edges


def _analyze_graph_topology(
    var_set: set[str], edges: list[tuple[str, str]]
) -> dict[str, Any]:
    """用 scipy.sparse.csgraph 算稀疏拓扑指标."""
    try:
        import networkx as nx
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        return {"error": "networkx/scipy required for sparsity analysis"}

    G = nx.Graph()
    G.add_nodes_from(var_set)
    G.add_edges_from(edges)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # 连通分量 (scipy.sparse.csgraph)
    nodes_list = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes_list)}
    row, col = [], []
    for u, v in edges:
        row.append(idx[u]); col.append(idx[v])
        row.append(idx[v]); col.append(idx[u])
    data = [1] * len(row)
    A = csr_matrix((data, (row, col)), shape=(n, n))
    n_comp, labels = connected_components(A, directed=False)

    # degeneracy = treewidth 下界 (近似)
    degeneracy = max(dict(G.degree()).values()) if n > 0 else 0

    # networkx treewidth_min_degree 上界 (小图才算, 大图太慢)
    tw_upper = None
    if n <= 50:
        try:
            tw_upper, _ = nx.algorithms.approximation.treewidth_min_degree(G)
        except Exception:
            tw_upper = None

    return {
        "n_vars": n,
        "n_edges": m,
        "density": (2 * m) / (n * (n - 1)) if n > 1 else 0.0,
        "avg_degree": (2 * m) / n if n > 0 else 0.0,
        "n_components": int(n_comp),
        "largest_component_size": int(max(np.bincount(labels))) if n > 0 else 0,
        "treewidth_lower": int(degeneracy),
        "treewidth_upper": int(tw_upper) if tw_upper is not None else None,
        "is_tree": n > 0 and m == n - n_comp,
        "is_chordal": nx.is_chordal(G) if n <= 100 else None,
    }


# ── 核心求解 ────────────────────────────────────────────────

def _parse_variables(var_specs: list[dict]) -> dict[str, Any]:
    """把变量声明转成 z3 变量对象."""
    z3_vars: dict[str, Any] = {}
    for v in var_specs:
        name = v["name"]
        vtype = v.get("type", "int")
        if vtype == "bool":
            z3_vars[name] = z3.Bool(name)
        elif vtype == "int":
            z3_vars[name] = z3.Int(name)
        elif vtype == "real":
            z3_vars[name] = z3.Real(name)
        elif vtype == "bv":
            width = int(v.get("domain") or v.get("width") or 8)
            z3_vars[name] = z3.BitVec(name, width)
        else:
            raise ValueError(f"unknown var type: {vtype}")
    return z3_vars


def _add_domain_constraints(
    solver: Any, var_specs: list[dict], z3_vars: dict[str, Any]
) -> None:
    """整数/BV 变量的值域约束."""
    for v in var_specs:
        name = v["name"]
        vtype = v.get("type", "int")
        if vtype == "int" and v.get("domain"):
            lo, hi = v["domain"]
            solver.add(z3_vars[name] >= int(lo), z3_vars[name] <= int(hi))


def _extract_model(model: Any, z3_vars: dict[str, Any]) -> dict[str, Any]:
    """从 z3 Model 提取变量赋值."""
    if model is None:
        return {}
    out: dict[str, Any] = {}
    for name, var in z3_vars.items():
        val = model.eval(var, model_completion=True)
        # z3 BoolRef / ArithRef / BitVecRef → Python 值
        if z3.is_bool(val):
            out[name] = bool(val)
        elif z3.is_bv_value(val):
            out[name] = val.as_long()
        elif z3.is_int_value(val):
            out[name] = val.as_long()
        elif z3.is_rational_value(val):
            n = val.numerator_as_long()
            d = val.denominator_as_long()
            out[name] = n / d if d != 1 else n
        else:
            out[name] = str(val)
    return out


def _solve_smt(
    var_specs: list[dict],
    constraints: list[str],
    timeout_ms: int = 5000,
) -> dict[str, Any]:
    """solve_smt / solve_sat 共用."""
    z3_vars = _parse_variables(var_specs)
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    _add_domain_constraints(solver, var_specs, z3_vars)
    for c in constraints:
        solver.add(_safe_eval(c, z3_vars))
    result = solver.check()
    if result == z3.sat:
        model = solver.model()
        return {
            "sat": True,
            "status": "sat",
            "model": _extract_model(model, z3_vars),
            "n_constraints": len(constraints),
        }
    elif result == z3.unsat:
        return {"sat": False, "status": "unsat", "model": None, "n_constraints": len(constraints)}
    else:
        return {"sat": None, "status": "unknown", "model": None, "n_constraints": len(constraints)}


def _optimize(
    var_specs: list[dict],
    constraints: list[str],
    objective: str,
    timeout_ms: int = 5000,
) -> dict[str, Any]:
    """离散优化."""
    z3_vars = _parse_variables(var_specs)
    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)
    _add_domain_constraints(opt, var_specs, z3_vars)
    for c in constraints:
        opt.add(_safe_eval(c, z3_vars))
    # objective: "minimize x+y" / "maximize x" — 拆关键字和表达式
    obj_stripped = objective.strip()
    direction = None
    expr_str = obj_stripped
    for kw in ("minimize", "maximize"):
        if obj_stripped.lower().startswith(kw):
            direction = kw
            expr_str = obj_stripped[len(kw):].strip()
            break
    if direction is None:
        return {"error": f"objective must start with 'minimize' or 'maximize', got: {objective}"}
    obj_expr = _safe_eval(expr_str, z3_vars)
    handle = opt.minimize(obj_expr) if direction == "minimize" else opt.maximize(obj_expr)
    result = opt.check()
    if result == z3.sat:
        model = opt.model()
        model_vals = _extract_model(model, z3_vars)
        # 用 model.eval 算目标值 (避免字符串解析目标表达式)
        opt_value = None
        try:
            evaled = model.eval(obj_expr, model_completion=True)
            if z3.is_int_value(evaled):
                opt_value = evaled.as_long()
            elif z3.is_rational_value(evaled):
                opt_value = float(evaled.as_decimal(6).rstrip("?"))
        except Exception:
            pass
        return {
            "optimal": True,
            "status": "sat",
            "opt_value": opt_value,
            "model": model_vals,
        }
    elif result == z3.unsat:
        return {"optimal": False, "status": "unsat", "opt_value": None, "model": None}
    else:
        return {"optimal": None, "status": "unknown", "opt_value": None, "model": None}


def _all_solutions(
    var_specs: list[dict],
    constraints: list[str],
    max_solutions: int = 10,
    timeout_ms: int = 5000,
) -> dict[str, Any]:
    """枚举所有解 (加 blocker 排除已找到的)."""
    z3_vars = _parse_variables(var_specs)
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    _add_domain_constraints(solver, var_specs, z3_vars)
    for c in constraints:
        solver.add(_safe_eval(c, z3_vars))
    solutions: list[dict[str, Any]] = []
    while len(solutions) < max_solutions:
        if solver.check() != z3.sat:
            break
        model = solver.model()
        sol = _extract_model(model, z3_vars)
        solutions.append(sol)
        # blocker: 排除当前解
        blocker = z3.Or([z3_vars[k] != v for k, v in sol.items() if not isinstance(v, str)])
        solver.add(blocker)
    return {
        "solutions": solutions,
        "n_found": len(solutions),
        "truncated": len(solutions) >= max_solutions,
    }


def _verify_implication(
    var_specs: list[dict],
    premises: list[str],
    conclusion: str,
    timeout_ms: int = 5000,
) -> dict[str, Any]:
    """验证 premises ⟹ conclusion, 找反例 (premises ∧ ¬conclusion)."""
    z3_vars = _parse_variables(var_specs)
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    _add_domain_constraints(solver, var_specs, z3_vars)
    for p in premises:
        solver.add(_safe_eval(p, z3_vars))
    concl = _safe_eval(conclusion, z3_vars)
    solver.add(z3.Not(concl))
    result = solver.check()
    if result == z3.sat:
        model = solver.model()
        return {
            "holds": False,
            "counterexample": _extract_model(model, z3_vars),
        }
    elif result == z3.unsat:
        return {"holds": True, "counterexample": None}
    else:
        return {"holds": None, "counterexample": None, "status": "unknown"}


def _solve_decomposed(
    var_specs: list[dict],
    constraints: list[str],
    timeout_ms: int = 5000,
) -> dict[str, Any]:
    """按约束图连通分量分治求解.

    ponytail: 只做连通分量分解 (零成本), 不做完整树分解 (实现复杂).
    每个分量独立 z3 求解, 合并 model. 全部分量 sat → 全局 sat.
    天花板: 不识别耦合分量内的稀疏子结构 (留 treewidth 分解).
    升级路径: min-fill treewidth 分解 + bucket elimination.
    """
    import networkx as nx

    var_set, edges = _build_primal_graph(var_specs, constraints)
    G = nx.Graph()
    G.add_nodes_from(var_set)
    G.add_edges_from(edges)
    components = list(nx.connected_components(G))

    # 每个分量: 只取该分量变量涉及的约束
    var_to_comp: dict[str, int] = {}
    for i, comp in enumerate(components):
        for v in comp:
            var_to_comp[v] = i

    component_results: list[dict[str, Any]] = []
    global_model: dict[str, Any] = {}
    all_sat = True
    any_unknown = False
    for i, comp in enumerate(components):
        comp_vars = [v for v in var_specs if v["name"] in comp]
        # 约束 c 属于分量 i 当且仅当 c 涉及的所有变量都在 comp 里
        comp_constraints = []
        for c in constraints:
            present = [v["name"] for v in var_specs if v["name"] in c]
            if present and all(var_to_contained := [var_to_comp.get(p, -1) == i for p in present]):
                comp_constraints.append(c)
        r = _solve_smt(comp_vars, comp_constraints, timeout_ms)
        component_results.append({
            "component_id": i,
            "n_vars": len(comp),
            "n_constraints": len(comp_constraints),
            "result": r["status"],
            "model": r.get("model"),
        })
        if r["status"] == "unsat":
            all_sat = False
            break  # 一个分量 unsat 全局就 unsat
        if r["status"] == "unknown":
            any_unknown = True
        if r["status"] == "sat" and r.get("model"):
            global_model.update(r["model"])

    if not all_sat:
        status = "unsat"
    elif any_unknown:
        status = "unknown"
    else:
        status = "sat"

    return {
        "status": status,
        "sat": status == "sat" if status != "unknown" else None,
        "model": global_model if status == "sat" else None,
        "n_components": len(components),
        "component_results": component_results,
        "speedup": "每个连通分量独立求解, 大稀疏约束图可线性加速",
    }


# ── HuginnTool 包装 ────────────────────────────────────────

class DiscreteSMTInput(BaseModel):
    action: Literal[
        "solve_sat",
        "solve_smt",
        "optimize",
        "all_solutions",
        "verify_implication",
        "analyze_sparsity",
        "solve_decomposed",
    ] = Field(..., description="求解模式")
    variables: list[dict] = Field(
        ...,
        description=(
            "变量声明列表, 每项 {name, type, domain}. "
            "type: 'bool' | 'int' | 'real' | 'bv'. "
            "domain: None (bool) | [lo, hi] (int) | width (bv). "
            "例: [{name:'x', type:'int', domain:[0, 10]}]"
        ),
    )
    constraints: list[str] = Field(
        default_factory=list,
        description=(
            "z3 Python API 表达式字符串列表. "
            "白名单: And/Or/Not/Implies/If/Sum/ForAll 等 z3 符号 + 变量名. "
            "例: ['x + y > 10', 'x != y']"
        ),
    )
    objective: str | None = Field(
        default=None,
        description="optimize action 的目标, 例: 'minimize x+y' / 'maximize x'",
    )
    premises: list[str] = Field(
        default_factory=list,
        description="verify_implication 的前提约束列表 (同 constraints 格式)",
    )
    conclusion: str | None = Field(
        default=None,
        description="verify_implication 的结论表达式",
    )
    max_solutions: int = Field(default=10, ge=1, le=1000)
    timeout_ms: int = Field(default=5000, ge=100, le=60000)


class DiscreteSMTTool(HuginnTool):
    """SAT/SMT 求解 + 稀疏约束图分析.

    离散约束求解的通用底座. 7 个 action 覆盖:
      - SAT/SMT 求解 + 优化 + 枚举
      - 反例搜索 (verify_implication)
      - 约束图稀疏性分析 (analyze_sparsity)
      - 大稀疏约束图分治求解 (solve_decomposed)

    用 z3 求解, scipy.sparse.csgraph + networkx 做稀疏分析.
    """

    name = "discrete_smt"
    category = "sci"
    profile = ToolProfile(
        phases=frozenset({ResearchPhase.HYPOTHESIS, ResearchPhase.VALIDATION}),
    )
    description = (
        "SAT/SMT solver with sparse constraint graph analysis. "
        "Solves boolean satisfiability, integer/real/bitvector SMT, "
        "discrete optimization, solution enumeration, implication "
        "verification (counterexample search), and sparse constraint "
        "graph topology analysis (connected components, treewidth). "
        "Use this for discrete problems instead of lifting to continuous."
    )
    input_schema = DiscreteSMTInput
    read_only = True

    def is_read_only(self, args: DiscreteSMTInput) -> bool:
        return True

    async def validate_input(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ValidationResult:
        args_obj = args if isinstance(args, DiscreteSMTInput) else DiscreteSMTInput(**args)
        if len(args_obj.variables) > 100:
            return ValidationResult(result=False, message="变量数 > 100 上限")
        if len(args_obj.constraints) > 500:
            return ValidationResult(result=False, message="约束数 > 500 上限")
        if args_obj.action == "optimize" and not args_obj.objective:
            return ValidationResult(result=False, message="optimize 需要 objective")
        if args_obj.action == "verify_implication" and not args_obj.conclusion:
            return ValidationResult(result=False, message="verify_implication 需要 conclusion")
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        args_obj = args if isinstance(args, DiscreteSMTInput) else DiscreteSMTInput(**args)
        try:
            if args_obj.action in ("solve_sat", "solve_smt"):
                r = _solve_smt(args_obj.variables, args_obj.constraints, args_obj.timeout_ms)
            elif args_obj.action == "optimize":
                r = _optimize(args_obj.variables, args_obj.constraints, args_obj.objective or "", args_obj.timeout_ms)
            elif args_obj.action == "all_solutions":
                r = _all_solutions(args_obj.variables, args_obj.constraints, args_obj.max_solutions, args_obj.timeout_ms)
            elif args_obj.action == "verify_implication":
                r = _verify_implication(args_obj.variables, args_obj.premises, args_obj.conclusion or "", args_obj.timeout_ms)
            elif args_obj.action == "analyze_sparsity":
                var_set, edges = _build_primal_graph(args_obj.variables, args_obj.constraints)
                r = _analyze_graph_topology(var_set, edges)
            elif args_obj.action == "solve_decomposed":
                r = _solve_decomposed(args_obj.variables, args_obj.constraints, args_obj.timeout_ms)
            else:
                return ToolResult(data=None, success=False, error=f"unknown action: {args_obj.action}")
            return ToolResult(data=r, success="error" not in r)
        except Exception as exc:
            logger.warning("discrete_smt failed: %s", exc, exc_info=True)
            return ToolResult(data=None, success=False, error=str(exc))


# ── selfcheck ──────────────────────────────────────────────

def _selfcheck() -> None:
    """12 项 assert 验证 SMT + 稀疏分析核心行为."""
    print("[discrete_smt] running self-check...")

    # 1. SAT 简单
    r = _solve_smt(
        [{"name": "x", "type": "bool"}, {"name": "y", "type": "bool"}],
        ["And(x, Not(y))"],
    )
    assert r["sat"] is True, f"1. expected sat, got {r}"
    assert r["model"]["x"] is True and r["model"]["y"] is False, f"1. model wrong: {r['model']}"

    # 2. UNSAT
    r = _solve_smt(
        [{"name": "x", "type": "bool"}],
        ["And(x, Not(x))"],
    )
    assert r["sat"] is False, f"2. expected unsat, got {r}"

    # 3. SMT 整数
    r = _solve_smt(
        [{"name": "x", "type": "int", "domain": [0, 20]},
         {"name": "y", "type": "int", "domain": [0, 20]}],
        ["x + y == 10", "x > 3", "y > 3"],
    )
    assert r["sat"] is True, f"3. expected sat, got {r}"
    assert r["model"]["x"] + r["model"]["y"] == 10, f"3. model violates x+y=10: {r['model']}"

    # 4. SMT UNSAT (整数)
    r = _solve_smt(
        [{"name": "x", "type": "int", "domain": [0, 100]},
         {"name": "y", "type": "int", "domain": [0, 100]}],
        ["x + y == 10", "x > 8", "y > 8"],
    )
    assert r["sat"] is False, f"4. expected unsat, got {r}"

    # 5. 优化 minimize
    r = _optimize(
        [{"name": "x", "type": "int", "domain": [0, 100]},
         {"name": "y", "type": "int", "domain": [0, 100]}],
        ["x + y >= 10"],
        "minimize x + y",
    )
    assert r["optimal"] is True, f"5. expected optimal, got {r}"
    assert r["model"]["x"] + r["model"]["y"] == 10, f"5. opt should be 10, model: {r['model']}"

    # 6. all_solutions
    r = _all_solutions(
        [{"name": "x", "type": "int", "domain": [0, 3]},
         {"name": "y", "type": "int", "domain": [0, 3]}],
        ["x != y"],
        max_solutions=20,
    )
    assert r["n_found"] == 12, f"6. expected 12 (4*3), got {r['n_found']}"
    assert r["truncated"] is False, f"6. should not be truncated"

    # 7. verify_implication 真
    r = _verify_implication(
        [{"name": "x", "type": "int", "domain": [0, 100]}],
        premises=["x > 5"],
        conclusion="x > 3",
    )
    assert r["holds"] is True, f"7. (x>5) ⟹ (x>3) should hold, got {r}"

    # 8. verify_implication 反例
    r = _verify_implication(
        [{"name": "x", "type": "int", "domain": [0, 100]}],
        premises=["x > 3"],
        conclusion="x > 5",
    )
    assert r["holds"] is False, f"8. (x>3) ⟹ (x>5) should NOT hold, got {r}"
    assert r["counterexample"]["x"] == 4, f"8. ce should be x=4, got {r['counterexample']}"

    # 9. N-queens N=4 (经典 SAT)
    queens = [{"name": f"q_{i}", "type": "int", "domain": [0, 3]} for i in range(4)]
    qcons = []
    for i in range(4):
        for j in range(i + 1, 4):
            qcons.append(f"q_{i} != q_{j}")  # 不同列
            qcons.append(f"Abs(q_{i} - q_{j}) != {j - i}")  # 不在对角
    r = _solve_smt(queens, qcons)
    assert r["sat"] is True, f"9. 4-queens should have solution, got {r}"
    # 验证模型合法
    qs = [r["model"][f"q_{i}"] for i in range(4)]
    assert len(set(qs)) == 4, f"9. queens same column: {qs}"
    for i in range(4):
        for j in range(i + 1, 4):
            assert abs(qs[i] - qs[j]) != j - i, f"9. diagonal conflict: {qs}"

    # 10. analyze_sparsity 基础
    r = _analyze_graph_topology(
        {"x", "y", "z", "w"},
        [("x", "y"), ("y", "z"), ("z", "w")],  # 路径图 P4
    )
    assert r["n_vars"] == 4 and r["n_edges"] == 3, f"10. graph stats wrong: {r}"
    assert r["is_tree"] is True, f"10. P4 should be tree, got {r}"
    assert r["n_components"] == 1, f"10. P4 is connected"

    # 11. analyze_sparsity 多连通分量 (孤立点也算分量)
    r = _analyze_graph_topology(
        {"a", "b", "c", "d", "e"},
        [("a", "b"), ("c", "d")],  # 2 个二元分量 + 1 个孤立点 e = 3 分量
    )
    assert r["n_components"] == 3, f"11. expected 3 components (incl. isolated), got {r['n_components']}"
    assert r["largest_component_size"] == 2, f"11. largest comp should be 2, got {r['largest_component_size']}"

    # 12. solve_decomposed: 两块独立约束
    r = _solve_decomposed(
        [{"name": "x", "type": "int", "domain": [0, 10]},
         {"name": "y", "type": "int", "domain": [0, 10]},
         {"name": "u", "type": "int", "domain": [0, 10]},
         {"name": "v", "type": "int", "domain": [0, 10]}],
        ["x + y == 5", "u + v == 7"],  # x,y 与 u,v 不耦合 → 2 分量
    )
    assert r["status"] == "sat", f"12. decomposed should be sat, got {r['status']}"
    assert r["n_components"] == 2, f"12. expected 2 components, got {r['n_components']}"
    assert r["model"]["x"] + r["model"]["y"] == 5, f"12. component 1 model wrong: {r['model']}"
    assert r["model"]["u"] + r["model"]["v"] == 7, f"12. component 2 model wrong: {r['model']}"

    print("[discrete_smt] self-check OK (12/12)")


if __name__ == "__main__":
    _selfcheck()
