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

import asyncio
import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from huginn.research.program import Experiment, grounding_verifier, run_research_program
from huginn.research.safe_expr import safe_math_eval as _safe_eval_expr

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


def _bad(msg: str) -> HTTPException:
    """客户端输入错误 → 400 (不是 500: 错在调用方, 不在服务端)."""
    return HTTPException(status_code=400, detail=msg)


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


@router.post("/grounding")
async def grounding(body: dict) -> dict:
    """声明门禁唯一实现的门禁 (claim_grounding) 经 HTTP 暴露。

    让脚本/示例在**不 import huginn.validation** 时即可对成文报告做结论证伪:
    ::

        {"report": "…", "trace": ["tool1 -> …", …]}

    返回 ``{"verdict": "pass"|"needs_grounding", "unsubstantiated": [...]}``。
    等价物: ``huginn.research.grounding_verifier()(report, trace)``。
    """
    if not isinstance(body, dict):
        raise _bad("body must be a JSON object")
    report = body.get("report")
    trace = body.get("trace")
    if not isinstance(report, str):
        raise _bad("report must be a string")
    if trace is None:
        trace = []
    if not isinstance(trace, list) or not all(isinstance(t, str) for t in trace):
        raise _bad("trace must be a list of strings")
    try:
        verify = grounding_verifier()
        result = await asyncio.to_thread(verify, report, trace)
    except Exception:
        logger.exception("deep research grounding failed")
        raise
    return result or {"verdict": "needs_grounding", "unsubstantiated": []}
