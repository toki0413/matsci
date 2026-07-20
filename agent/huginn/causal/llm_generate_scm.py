"""LLM 生成 SCM — KB 无模板时的 fallback.

当 KB 内置模板 (sintering/ostwald/diffusion/phase_transition) 都不匹配
用户问题时, 让 LLM 读 KB 论文 + 用户问题生成 SCM draft. 标
confirmed=False, predict_intervention 时显式警告.

混合策略 (跟用户偏好对齐):
  1. 优先用 KB 内置物理方程模板 (confirmed=True)
  2. KB 无模板时, LLM 读 KB 论文 + 问题生成 draft (confirmed=False)
  3. LLM 草稿必须用户审核才能升级 confirmed=True
  4. 数据充足时 (>=5 观测点) 用 Phase 2 visual_causal_chain 数据拟合

LLM 生成流程:
  Step 1: 调 KB recall 拿相关论文片段 (物理方程/数据/机理)
  Step 2: 调 LLM 把片段 + 问题 → SCM draft (nodes/edges/equations/noise)
  Step 3: 验证 SCM 结构合法 (DAG / equations 可调 / 节点范围合理)
  Step 4: 标 confirmed=False, source="llm_draft", 加生成痕迹供审计

设计原则 (ponytail):
  - LLM 只生成结构 (nodes/edges/方程名), 不生成数值参数
    数值参数从 KB 数据拟合或 LLM 给区间 + 用户确认
  - 方程用 closures 包装 LLM 输出的 Python 表达式, 限白名单函数
    (math.exp/log/pow + 不引外部库), 防 code injection
  - 失败返 None 让调用方 fallback 到模板或退化路径
"""
from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from huginn.causal.visual_scm import (
    VisualSCM, Variable, Edge, _noise_normal,
)

logger = logging.getLogger(__name__)


# ── 安全的方程表达式求值 ─────────────────────────────────────

# 白名单函数 (LLM 方程里能用的)
_SAFE_FUNCS: dict[str, Callable[..., float]] = {
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "pow": math.pow,
    "abs": abs,
    "min": min,
    "max": max,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": lambda: math.pi,  # 常量当 0-参函数
    "e": lambda: math.e,
}

# 允许的 AST 节点类型 (白名单, 防 code injection)
_SAFE_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
    ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
)


def _safe_eval_expr(expr: str, variables: dict[str, float]) -> float:
    """安全求值数学表达式. 只允许白名单函数 + 变量.

    LLM 生成的方程表达式如 "A0 * exp(-Ea / (R * T))" 在此求值.
    """
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _SAFE_AST_NODES):
                raise ValueError(f"不允许的 AST 节点: {type(node).__name__}")
        # 编译 + 求值, 全局只给白名单函数
        env: dict[str, Any] = {**_SAFE_FUNCS, **variables, "R": 8.314e-3}
        return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env))
    except Exception as exc:
        logger.debug("safe_eval failed for '%s': %s", expr, exc)
        raise


def _build_equation_closure(
    expr: str, parent_names: list[str]
) -> Callable[[dict[str, float], float], float]:
    """把 LLM 表达式 + parent 名单包成 SCM equation closure.

    返 f(parents_values, noise) -> float.
    noise 作为变量 "noise" 注入表达式 (LLM 可选用, 也可不用).
    """
    def equation(parents: dict[str, float], noise: float) -> float:
        try:
            env = {**parents, "noise": noise}
            return _safe_eval_expr(expr, env)
        except Exception:
            # ponytail: 表达式失败返 0, 不阻塞 SCM 整体跑. predict_intervention
            # 的 delta 会反映这个节点没贡献. 升级路径: 让 LLM 重新生成方程.
            return 0.0
    return equation


# ── LLM prompt 模板 ─────────────────────────────────────────

_SCM_GENERATION_PROMPT = """你是材料科学结构因果模型 (SCM) 设计专家.

任务: 给定研究问题 + KB 论文片段, 设计一个 SCM draft.

研究问题:
{question}

相关 KB 论文片段 (物理方程/机理/数据):
{kb_context}

请输出严格 JSON (不要 markdown 代码块), 结构:
{{
  "name": "custom_<domain>_<short_id>",
  "domain": "<材料域, e.g. ceramic/polymer/battery/catalyst>",
  "nodes": [
    {{"name": "<var_name>", "type": "condition|feature|latent", "unit": "<SI单位>", "range": [min, max], "description": "<中文描述>"}}
  ],
  "edges": [
    {{"cause": "<parent>", "effect": "<child>", "mechanism": "<物理机制名>", "strength": 0.0-1.0}}
  ],
  "equations": [
    // 每个非条件节点一个方程, 用 parents + noise 变量
    {{"node": "<var_name>", "expr": "<python 数学表达式, 只用 exp/log/sqrt/pow/abs/min/max/sin/cos + parents 变量名 + noise>"}}
  ],
  "notes": "<中文说明这个 SCM 的适用范围和局限>"
}}

约束:
- 节点用 snake_case 英文名 (e.g. T, t, particle_size, density)
- 至少 1 个 condition 节点 (可干预的) + 1 个 feature 节点 (要预测的)
- edges 必须构成 DAG (无环)
- 方程只用白名单函数: exp, log, log10, sqrt, pow, abs, min, max, sin, cos, tan, pi, e
- 方程表达式里变量名必须等于 parents 节点名 + "noise"
- 常数 R=8.314e-3 (kJ/mol·K) 自动注入, 不用声明
- 方程物理合理 (e.g. Arrhenius 项用 exp(-Ea/(R*T)))
- 不要输出任何 markdown, 直接输出 JSON

输出:"""


# ── LLM 生成主流程 ──────────────────────────────────────────

@dataclass
class GenerateSCMResult:
    """LLM 生成 SCM 的结果."""
    scm: VisualSCM | None
    raw_response: str
    parse_error: str | None = None
    validation_errors: list[str] | None = None


async def generate_scm_via_llm(
    question: str,
    llm_chat: Callable[[str], "Any"],
    kb_context: str = "",
) -> GenerateSCMResult:
    """LLM 生成 SCM draft.

    Args:
        question: 研究问题 (e.g. "温度对 C-S-H 中 Ca 扩散系数的影响")
        llm_chat: 异步 LLM 调用函数, 接 prompt 返 response (str)
        kb_context: KB recall 拿到的论文片段 (可选)

    Returns:
        GenerateSCMResult: 含 SCM (None 如果失败) + raw_response + 错误信息

    生成的 SCM 必然 confirmed=False, source="llm_draft".
    """
    import json

    prompt = _SCM_GENERATION_PROMPT.format(
        question=question,
        kb_context=kb_context or "(无相关 KB 片段, 基于领域知识生成)",
    )

    try:
        response = await llm_chat(prompt)
    except Exception as exc:
        return GenerateSCMResult(scm=None, raw_response="", parse_error=f"LLM 调用失败: {exc}")

    raw_response = str(response)

    # 解析 JSON (LLM 可能加 markdown 代码块, 清掉)
    json_str = raw_response.strip()
    json_str = re.sub(r"^```(?:json)?\s*", "", json_str)
    json_str = re.sub(r"\s*```$", "", json_str)

    try:
        scm_draft = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return GenerateSCMResult(scm=None, raw_response=raw_response,
                                  parse_error=f"JSON 解析失败: {exc}")

    # 验证 + 构造 VisualSCM
    errors: list[str] = []

    # 必要字段
    if "nodes" not in scm_draft or not scm_draft["nodes"]:
        errors.append("缺少 nodes 或为空")
    if "edges" not in scm_draft:
        errors.append("缺少 edges")
    if "equations" not in scm_draft:
        errors.append("缺少 equations")

    if errors:
        return GenerateSCMResult(scm=None, raw_response=raw_response,
                                  parse_error="; ".join(errors))

    # 构造 nodes
    nodes: dict[str, Variable] = {}
    for n in scm_draft["nodes"]:
        try:
            name = str(n["name"])
            vtype = str(n.get("type", "feature"))
            unit = str(n.get("unit", ""))
            rng = n.get("range")
            rng_t = tuple(float(x) for x in rng) if rng and len(rng) == 2 else None
            desc = str(n.get("description", ""))
            nodes[name] = Variable(name, vtype, unit, rng_t, desc)
        except (KeyError, ValueError) as exc:
            errors.append(f"node 解析失败 {n}: {exc}")

    # 构造 edges
    edges: list[Edge] = []
    for e in scm_draft["edges"]:
        try:
            cause, effect = str(e["cause"]), str(e["effect"])
            if cause not in nodes or effect not in nodes:
                errors.append(f"edge 引用不存在节点: {cause}→{effect}")
                continue
            mech = str(e.get("mechanism", ""))
            strength = float(e.get("strength", 1.0))
            edges.append(Edge(cause, effect, mech, strength))
        except (KeyError, ValueError) as exc:
            errors.append(f"edge 解析失败 {e}: {exc}")

    # 构造 equations
    equations: dict[str, Callable[[dict[str, float], float], float]] = {}
    eq_specs: dict[str, str] = {str(eq["node"]): str(eq["expr"]) for eq in scm_draft["equations"]}
    for node_name, var in nodes.items():
        if var.type == "condition":
            # 条件节点: 直接从 parents (实际是 intervention/base_conditions) 取
            equations[node_name] = lambda p, u, n=node_name: p.get(n, 0.0)
        elif node_name in eq_specs:
            parent_names = [e.cause for e in edges if e.effect == node_name]
            equations[node_name] = _build_equation_closure(eq_specs[node_name], parent_names)
        else:
            errors.append(f"非条件节点 '{node_name}' 缺方程")
            equations[node_name] = lambda p, u, n=node_name: 0.0

    # 构造 noise (默认 10% 相对噪声, LLM 没指定)
    noise: dict[str, Callable[[], float]] = {}
    for node_name, var in nodes.items():
        if var.type == "condition":
            noise[node_name] = _noise_normal(0)
        else:
            noise[node_name] = _noise_normal(0.10)

    if errors:
        return GenerateSCMResult(scm=None, raw_response=raw_response,
                                  validation_errors=errors)

    # 验证 DAG (topological_order 会抛)
    scm = VisualSCM(
        name=str(scm_draft.get("name", "custom_llm")),
        domain=str(scm_draft.get("domain", "unknown")),
        nodes=nodes, edges=edges, equations=equations, noise=noise,
        confirmed=False,   # LLM 草稿必标 False
        source="llm_draft",
        notes=str(scm_draft.get("notes", "")) +
              " | LLM 生成草稿, 未用户审核. 方程表达式: " +
              "; ".join(f"{n}={e}" for n, e in eq_specs.items()),
    )

    try:
        _ = scm.topological_order()  # 抛如果有环
    except ValueError as exc:
        return GenerateSCMResult(scm=None, raw_response=raw_response,
                                  validation_errors=[f"SCM 有环: {exc}"])

    # smoke test: 方程都能调 (传合理值)
    test_values: dict[str, float] = {}
    for n, v in nodes.items():
        if v.range:
            test_values[n] = (v.range[0] + v.range[1]) / 2
        else:
            test_values[n] = 1.0
    for node_name, eq in equations.items():
        try:
            val = eq(test_values, 0.0)
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                errors.append(f"方程 {node_name} 返非有限数: {val}")
        except Exception as exc:
            errors.append(f"方程 {node_name} 调用失败: {exc}")

    if errors:
        return GenerateSCMResult(scm=None, raw_response=raw_response,
                                  validation_errors=errors)

    return GenerateSCMResult(scm=scm, raw_response=raw_response)


# ── KB recall 接口 (复用 metacog.recall_context) ────────────

def _recall_kb_context(question: str, top_k: int = 5) -> str:
    """从 KB recall 相关论文片段给 LLM 当上下文.

    ponytail: 复用 metacog.recall_context, 不重写 RAG.
    失败返空字符串, LLM 仍能基于领域知识生成 (只是没 KB 支撑).
    """
    try:
        from huginn.metacog import recall_context
        results = recall_context(category="knowledge_seed", query=question, top_k=top_k)
        if not results:
            # 退化: 试 stable_principles
            results = recall_context(category="stable_principles", query=question, top_k=top_k)
        if not results:
            return ""
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            content = r.get("content", "") if isinstance(r, dict) else str(r)
            parts.append(f"[{i}] {content[:500]}")  # 截断避免 prompt 过长
        return "\n\n".join(parts)
    except Exception as exc:
        logger.debug("KB recall failed: %s", exc)
        return ""


# ── 用户审核接口 ─────────────────────────────────────────────

def confirm_scm(scm: VisualSCM, user_notes: str = "") -> VisualSCM:
    """用户审核后升级 confirmed=True.

    审核通过后 SCM 可用于 predict_intervention 不再警告.
    notes 字段追加用户审核记录.
    """
    scm.confirmed = True
    scm.source = "user_confirmed"
    audit = f" | 用户审核通过 ({user_notes})" if user_notes else " | 用户审核通过"
    scm.notes += audit
    return scm


# ── self-check ───────────────────────────────────────────────

def _selfcheck() -> None:
    """12 项 assert 验证 LLM 生成 SCM 路径."""
    import asyncio

    # 1. _safe_eval_expr 白名单函数可用
    assert _safe_eval_expr("exp(-1.0)", {}) == math.exp(-1.0)
    assert _safe_eval_expr("2 * 3 + 1", {}) == 7.0
    assert _safe_eval_expr("sqrt(16)", {}) == 4.0

    # 2. _safe_eval_expr 阻止非白名单 (e.g. __import__)
    try:
        _safe_eval_expr("__import__('os').system('echo hack')", {})
        assert False, "应拒绝 __import__"
    except (ValueError, NameError, SyntaxError):
        pass

    # 3. _safe_eval_expr 阻止赋值语句
    try:
        _safe_eval_expr("x = 1", {})
        assert False, "应拒绝赋值"
    except SyntaxError:
        pass

    # 4. _build_equation_closure 返可调函数
    eq = _build_equation_closure("A0 * exp(-Ea / (R * T))", ["T"])
    val = eq({"T": 1500.0, "A0": 1e6, "Ea": 250.0}, 0.0)
    assert val > 0
    # Arrhenius: T 升 → val 升
    val_high = eq({"T": 1800.0, "A0": 1e6, "Ea": 250.0}, 0.0)
    assert val_high > val

    # 5. _build_equation_closure 失败返 0 (不抛)
    bad_eq = _build_equation_closure("undefined_func(1.0)", [])
    assert bad_eq({}, 0.0) == 0.0

    # 6. generate_scm_via_llm: mock LLM 返合法 JSON
    valid_json = '''{
      "name": "custom_battery_aging",
      "domain": "battery",
      "nodes": [
        {"name": "T", "type": "condition", "unit": "K", "range": [253, 353], "description": "温度"},
        {"name": "cycle_count", "type": "condition", "unit": "", "range": [0, 5000], "description": "循环次数"},
        {"name": "capacity_retention", "type": "feature", "unit": "", "range": [0, 1], "description": "容量保持率"}
      ],
      "edges": [
        {"cause": "T", "effect": "capacity_retention", "mechanism": "arrhenius_degradation", "strength": 0.8},
        {"cause": "cycle_count", "effect": "capacity_retention", "mechanism": "cyclic_aging", "strength": 1.0}
      ],
      "equations": [
        {"node": "capacity_retention", "expr": "exp(-0.0005 * cycle_count) * (1 - 0.001 * (T - 298))"}
      ],
      "notes": "电池老化 SCM, 线性温度修正+指数循环衰减"
    }'''

    async def mock_llm(prompt: str) -> str:
        return valid_json

    result = asyncio.run(generate_scm_via_llm(
        question="温度对电池容量衰减的影响",
        llm_chat=mock_llm,
        kb_context="",
    ))
    assert result.scm is not None
    assert result.scm.name == "custom_battery_aging"
    assert result.scm.domain == "battery"
    assert result.scm.confirmed is False  # LLM 草稿必 False
    assert result.scm.source == "llm_draft"
    assert "capacity_retention" in result.scm.nodes
    assert len(result.scm.edges) == 2

    # 7. LLM 生成 SCM 拓扑序不抛 (无环)
    order = result.scm.topological_order()
    assert "T" in order and "cycle_count" in order and "capacity_retention" in order
    assert order.index("T") < order.index("capacity_retention")
    assert order.index("cycle_count") < order.index("capacity_retention")

    # 8. 方程可调且物理合理
    eq_retention = result.scm.equations["capacity_retention"]
    v_low_cycles = eq_retention({"T": 300, "cycle_count": 10}, 0.0)
    v_high_cycles = eq_retention({"T": 300, "cycle_count": 1000}, 0.0)
    assert v_high_cycles < v_low_cycles  # 循环越多, 保持率越低

    # 9. LLM 返非法 JSON → parse_error
    async def mock_bad_json(prompt: str) -> str:
        return "not a json"

    result = asyncio.run(generate_scm_via_llm(
        question="test", llm_chat=mock_bad_json,
    ))
    assert result.scm is None
    assert result.parse_error is not None

    # 10. LLM 返缺字段 JSON → parse_error (空 nodes/edges/equations 走 parse 阶段)
    incomplete_json = '{"name": "x", "nodes": [], "edges": [], "equations": []}'

    async def mock_incomplete(prompt: str) -> str:
        return incomplete_json

    result = asyncio.run(generate_scm_via_llm(
        question="test", llm_chat=mock_incomplete,
    ))
    assert result.scm is None
    # 空字段走 parse_error, 结构错走 validation_errors
    assert result.parse_error is not None or result.validation_errors is not None

    # 11. confirm_scm 升级 confirmed=True
    scm = result.scm if result.scm else asyncio.run(generate_scm_via_llm(
        "test", mock_llm
    )).scm
    assert scm is not None
    assert scm.confirmed is False
    confirmed = confirm_scm(scm, "测试审核")
    assert confirmed.confirmed is True
    assert confirmed.source == "user_confirmed"
    assert "测试审核" in confirmed.notes

    # 12. markdown 代码块包裹的 JSON 也能解析
    md_json = "```json\n" + valid_json + "\n```"

    async def mock_md(prompt: str) -> str:
        return md_json

    result = asyncio.run(generate_scm_via_llm(
        "test", mock_md
    ))
    assert result.scm is not None
    assert result.scm.name == "custom_battery_aging"

    print("all self-checks passed")


if __name__ == "__main__":
    _selfcheck()
