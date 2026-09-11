"""Deep-research HTTP endpoint (ADR-0001 single-gateway migration target).

让外部消费者 (scripts/ examples/ servers/) 用纯 HTTP 跑完整深研管线, 而不必
``import huginn.research.program``。设计取舍 (pack contract):

- **闭包不过 HTTP**：``run_research_program`` 的 ``Experiment.run`` 是本地闭包
  (domain 真实计算), 无法序列化。本端点把实验建模成**声明的数值模型** —— 每个
  实验由一个 ``objective_expr`` (数学表达式) + ``params`` (变量取值) 描述,
  服务端用**受控 AST 数学求值器**计算 objectives (只放行 数字/常量/算术/白名单
  math 函数/命名变量, 不 ``exec``, 不碰文件/网络/模块)。
- **确定性综合**：不传 ``client`` → ``run_research_program`` 走确定性组装 +
  claim_grounding 门禁, 产出 pareto 前沿 / pruning / 批判综合 / verdict, 全程
  数值来自服务端对声明的真实计算。
- **迁移目标**：example 里只用纯数值目标函数 + 参数空间的深研, 可改走本端点,
  从 ``ALLOWED_EXTERNAL_IMPORTS`` 移除; 内嵌 ODE/FEM/沙箱的闭合模型示例仍无法
  换皮, 保持清单登记 (诚实边界, 见 test_arch_single_gateway).
"""

from __future__ import annotations

import ast
import asyncio
import logging
import math
import numbers
import operator
from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from huginn.research.program import Experiment, run_research_program

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


def _bad(msg: str) -> HTTPException:
    """客户端输入错误 → 400 (不是 500: 错在调用方, 不在服务端)."""
    return HTTPException(status_code=400, detail=msg)


# ── 受控数学求值器 (安全 AST, 非 exec) ──────────────────────────────
# 只允许: 数字/命名变量/算术/比较逻辑/白名单 math 函数. 不 import, 不求值字面串.
_ALLOWED_FUNCS = {
    "abs": abs, "min": min, "max": max, "round": round,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "floor": math.floor, "ceil": math.ceil, "fabs": math.fabs, "pow": math.pow,
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
}
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval_tail(node: ast.AST, ctx: dict[str, float], depth: int) -> float:
    """受控求值节点树; 任何超出白名单的节点抛 ValueError."""
    if depth > 64:
        raise ValueError("expression too deep")
    if isinstance(node, ast.Expression):
        return _safe_eval_tail(node.body, ctx, depth + 1)
    if isinstance(node, ast.Constant):
        # Py3.8+ 数值统一走 Constant (ast.Num 已在 3.14 移除)
        if isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node.value, numbers.Real):
            return float(node.value)
        raise ValueError("literal not supported")
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_FUNCS and isinstance(_ALLOWED_FUNCS[node.id], numbers.Real):
            return float(_ALLOWED_FUNCS[node.id])
        if node.id in ctx:
            return float(ctx[node.id])
        raise ValueError(f"unknown variable: {node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](
            _safe_eval_tail(node.left, ctx, depth + 1),
            _safe_eval_tail(node.right, ctx, depth + 1),
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval_tail(node.operand, ctx, depth + 1))
    if isinstance(node, ast.Call):
        fn = node.func
        if not isinstance(fn, ast.Name) or fn.id not in _ALLOWED_FUNCS:
            raise ValueError("only whitelisted math functions")
        callable_fn = _ALLOWED_FUNCS[fn.id]
        if not callable(callable_fn):
            raise ValueError("function call target not callable")
        if node.keywords:
            raise ValueError("keyword arguments not allowed")
        args = [_safe_eval_tail(a, ctx, depth + 1) for a in node.args]
        return float(callable_fn(*args))
    if isinstance(node, ast.Compare):
        lb = _safe_eval_tail(node.left, ctx, depth + 1)
        for opn, cmp_node in zip(node.ops, node.comparators):
            rb = _safe_eval_tail(cmp_node, ctx, depth + 1)
            if type(opn) is ast.Lt and not (lb < rb):
                return 0.0
            if type(opn) is ast.LtE and not (lb <= rb):
                return 0.0
            if type(opn) is ast.Gt and not (lb > rb):
                return 0.0
            if type(opn) is ast.GtE and not (lb >= rb):
                return 0.0
            if type(opn) is ast.Eq and lb != rb:
                return 0.0
            if type(opn) is ast.NotEq and lb == rb:
                return 0.0
            lb = rb
        return 1.0
    raise ValueError(f"expression node not allowed: {type(node).__name__}")


def _safe_eval_expr(expr: str, ctx: dict[str, float]) -> float:
    """把开根表达式解析成 AST 后走白名单求值 (不 exec)."""
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("objective_expr must be a non-empty string")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as syn:
        raise ValueError(f"expression not parseable: {syn.msg}") from syn
    return float(_safe_eval_tail(tree, ctx, 0))


# ── 请求/响应模型 ──────────────────────────────────────────────────


class _ExperimentSpec:
    """声明的单实验: 名字/假说 + 数值目标函数 + 变量取值."""

    def __init__(
        self,
        name: str,
        hypothesis: str,
        objectives: dict[str, str],
        params: dict[str, float] | None = None,
    ) -> None:
        self.name = name
        self.hypothesis = hypothesis
        # {objective_name: objective_expr}, expr 里的变量从 params 取
        self.objectives = objectives or {}
        self._params = dict(params or {})

    def make_run(self):
        def _run():
            summary = {}
            scores: dict[str, float] = {}
            for obj, expr in self.objectives.items():
                v = _safe_eval_expr(expr, self._params)
                scores[obj] = v
                summary[obj] = v
            return {"summary": summary, "objectives": scores}

        return _run


# 只收 JSON 可序列化字段, 闭包/回调用例由深研程序化 API 本地提供.
_PAYLOAD_KEYS = {
    "goal", "objectives_config", "experiments", "max_iterations",
    "min_iterations", "max_parallel", "model", "base_url",
    "human_review", "debate", "strictness", "harness_agent", "harness_machine",
}
_INT_KEYS = {"max_iterations", "min_iterations", "max_parallel", "strictness"}
_BOOL_KEYS = {"debate"}
_STR_KEYS = {"goal", "model", "base_url", "harness_agent", "harness_machine"}


def _build_experiments(specs: list[dict]) -> list[Experiment]:
    if not isinstance(specs, list) or not specs:
        raise _bad("experiments must be a non-empty list")
    exps: list[Experiment] = []
    for sp in specs:
        name = sp.get("name")
        hypothesis = sp.get("hypothesis", "")
        objectives = sp.get("objectives")  # {obj: expr}
        params = sp.get("params")
        if not isinstance(name, str) or not name:
            raise _bad("each experiment needs a non-empty string name")
        if not isinstance(objectives, dict) or not objectives:
            raise _bad(f"experiment {name}: objectives dict needed")
        for obj, expr in objectives.items():
            if not isinstance(expr, str):
                raise _bad(f"experiment {name}/{obj}: objective_expr must be a string")
            # 前置校验: 表达式白名单 + 变量都在 params 里 → 坏配置**立即 400**,
            # 而不是被 orchestrator 静默吞成 explored=0 的空成功.
            try:
                _safe_eval_expr(expr, dict(params or {}))
            except Exception as exc:
                raise _bad(
                    f"experiment {name}/{obj}: bad objective_expr (or missing param): "
                    f"{exc}"
                ) from exc
        spec = _ExperimentSpec(name, hypothesis, objectives, params)
        exps.append(Experiment(spec.name, spec.hypothesis, spec.make_run()))
    return exps


def _normalize_options(body: dict, exps: list[Experiment]) -> dict:
    opts: dict = {}
    for k in _STR_KEYS:
        v = body.get(k)
        if isinstance(v, str) and v:
            opts[k] = v
    for k in _INT_KEYS:
        v = body.get(k)
        if isinstance(v, int) and not isinstance(v, bool):
            opts[k] = v
    for k in _BOOL_KEYS:
        v = body.get(k)
        if isinstance(v, bool):
            opts[k] = v
    human = body.get("human_review")
    if isinstance(human, str) and human.lower() in ("keep", "drop", "guidance"):
        opts["human_review"] = (lambda card, _gdec=human: {"decision": _gdec, "card": card})
    # 默认小预算: 确定性综合不必长跑; 调用方可调高复现完整演化
    opts.setdefault("max_iterations", 12)
    opts.setdefault("min_iterations", 2)
    opts.setdefault("max_parallel", 2)
    return opts


@router.post("/run_program")
async def run_program(body: dict) -> dict:
    """跑一条完整(确定性)深研管线, 返回 ResearchOutcome 的 JSON 视图.

    请求:
    .. code-block:: json

        {
          "goal": "find optimal a,b",
          "objectives_config": {"score": "maximize"},
          "experiments": [
            {"name": "e0", "hypothesis": "baseline",
             "objectives": {"score": "a + b"}, "params": {"a": 1.0, "b": 2.0}},
            {"name": "e1", "hypothesis": "candidate", "objectives":
             {"score": "2*a + b"}, "params": {"a": 1.5, "b": 2.0}}
          ],
          "max_iterations": 8, "min_iterations": 2
        }

    响应: ResearchOutcome 序列化 (含 pareto_front / report / verdict / consolidated).
    """
    if not isinstance(body, dict):
        raise _bad("body must be a JSON object")
    unknown = set(body) - _PAYLOAD_KEYS
    if unknown:
        raise _bad(f"unknown fields: {sorted(unknown)}")
    goal = body.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise _bad("goal must be a non-empty string")
    objectives_config = body.get("objectives_config")
    if not isinstance(objectives_config, dict) or not objectives_config:
        raise _bad("objectives_config must be a non-empty object")

    exps = _build_experiments(body.get("experiments"))
    opts = _normalize_options(body, exps)

    try:
        # run_research_program 是同步阻塞(还跑 orchestrator 循环), 放进线程隔离,
        # 不卡 FastAPI 事件循环.
        out = await asyncio.to_thread(
            run_research_program,
            goal,
            exps,
            objectives_config,
            max_iterations=opts.get("max_iterations", 12),
            min_iterations=opts.get("min_iterations", 2),
            max_parallel=opts.get("max_parallel", 2),
            client=None,
            model=opts.get("model", "intern-s2-preview"),
            base_url=opts.get("base_url"),
            human_review=opts.get("human_review"),
            debate=opts.get("debate", False),
            strictness=opts.get("strictness", 0),
            harness_agent=opts.get("harness_agent", ""),
            harness_machine=opts.get("harness_machine", ""),
        )
    except Exception:
        logger.exception("deep research run_program failed")
        raise

    result = asdict(out)
    return result
