"""世界模型能力 —— 把 LawModel(数学定律前向模型)装箱成可发现/可调用的能力.

治理意义 (对标 具身图灵测试 + 世界模型多元论):
  具身原则: 完整智能须依托物理实体交互。在 AI4S agent 里"物理实体交互"即**真实科学
  执行**(compute/sim/experiment)。因此世界模型能力**必须可证伪**: 它的预告(predict)
  只是假说, 只有与真实执行 reconcile 对账后才算被证实。凡不满足该契约的对象
  (无 predict/law/seed, 或非可对账的 LawModel/鸭子型) 一律拒绝装箱 —— 绝不把
  "不可证伪的预判"伪装成"世界知识"。

  去混淆原则: 世界模型不是单一范式。能力带 `world_model_card`, 显式声明 worldview
  (物理-行动-因果 / 预测感知 / 隐态转移 / RL 策略), 防止把不同'世界'的理解混为一谈。

对外 op 面 (单一能力, 多操作 —— 别让碎片工具淹没 LLM):
  - "law"        返回数学定律(方程串) —— 规划/预告/验证的公共语言
  - "predict"    给定 state + action → 预告下继状态 (带 falsifiable 标注 = 假说待对账)
  - "reconcile"  给定 predicted + actual → 数值对账, 判定律被证实/证伪 (真相检验)

纯标准库依赖 (law_model / interaction_explain 均不拉重栈); 经 CapabilityRegistry 自动
装箱 → mcp_export 对 MCP host 可见, 也能被研究管线 resolve_diagnostic_tools 引用。
"""
from __future__ import annotations

from typing import Any

from huginn.capabilities.base import Capability, CapabilityResult
from huginn.core_types import ToolContext
from huginn.research.law_model import LawAction, LawModel, LawState, reconcile
from huginn.research.law_model import world_model_card  # noqa: F401  (复用卡片, 供外部取)

# 对外 op 面的输入 schema (MCP / OpenAI function 形态), 让 host 知道怎么调.
_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["law", "predict", "reconcile"],
               "description": "操作: 数学定律 / 预告下继状态 / 与真实执行数值对账"},
        "state": {"type": "object", "description": "世界状态(特征向量), predict/reconcile 用"},
        "action": {"type": "object", "description": "动作/实验配置, predict 用"},
        "label": {"type": "string", "description": "动作标签(可选)"},
        "predicted": {"type": "object", "description": "预告结果(as_dict), reconcile 用"},
        "actual": {"type": "object", "description": "真实执行结果数值, reconcile 用"},
        "tol": {"type": "number", "description": "对账相对误差容差, 默认 0.03"},
    },
    "required": ["op"],
}


def _falsifiable(model: Any) -> bool:
    """治理契约: 该对象能否作为*世界模型能力*? 需具备『预告 + 数学定律 + 播种』三件套,
    且可对账(LawModel 子类自动可 reconcile; 鸭子型须自带 reconcile 方法)."""
    if not all(hasattr(model, m) for m in ("predict", "law", "seed")):
        return False
    return isinstance(model, LawModel) or callable(getattr(model, "reconcile", None))


class WorldModelCapability(Capability):
    """把 LawModel 装箱成只读世界模型能力 (law / predict / reconcile)."""

    category = "world_model"
    read_only = True

    def __init__(self, model: LawModel, *, name: str | None = None) -> None:
        if not _falsifiable(model):
            raise TypeError(
                f"不可证伪对象不能装箱为世界模型能力: {type(model).__name__} — "
                "须具备 predict/law/seed 且可 reconcile(具身可信: 无对账的预判不是世界知识)")
        self.model = model
        self.domain = getattr(model, "domain", "")
        self.name = name or f"world_model.{self.domain}"
        self._card = world_model_card(model)
        self.description = (
            f"世界模型能力[{self.name}] — {self.domain} 域, 世界观 "
            f"(world model) 站在 '{self._card['worldview']}' 一极。可问它的『数学定律』、"
            "『预告下一状态』, 或把预告与真实执行『数值对账』(证伪式可信)。"
            "世界模型关闭时仍须能对账, 否则视为失效。"
        )

    def is_available(self) -> bool:
        return True

    @property
    def input_json_schema(self) -> dict[str, Any] | None:
        return _INPUT_SCHEMA

    async def _run(self, args: Any, context: ToolContext | None) -> CapabilityResult:
        if not isinstance(args, dict):
            return CapabilityResult.fail("世界模型能力参数须是 dict(含 op)", code="invalid_input")
        op = args.get("op")
        if op == "law":
            # 数学定律 + 治理卡片(世界观/可证伪性), 供规划与审计取公共语言.
            return CapabilityResult.ok({"law": self.model.law(), "card": self._card})
        if op == "predict":
            state = LawState(dict(args.get("state") or {}), domain=self.domain)
            action = LawAction(dict(args.get("action") or {}),
                               label=args.get("label") or "")
            pred = self.model.predict(state, action).as_dict()
            # 预告是假说: 回执显式标注 falsifiable, 提醒下游须对账, 不被误认为真实.
            return CapabilityResult.ok({
                "predicted": pred,
                "falsifiable": True,
                "note": "预告是假说(可证伪), 需 reconcile 对真实执行验证后才算被证实",
            })
        if op == "reconcile":
            pred = args.get("predicted")
            actual = args.get("actual")
            tol = float(args.get("tol", 0.03))
            # predicted / actual 都可能是 Cap 内 predict 的嵌套形式({"domain","state"}),
            # 归一化成 flat 数值 dict 再对账, 兼容 MCP 调用方的两种传法.
            pred_vec = (pred.get("state") if isinstance(pred, dict) and "state" in pred
                        else pred)
            act_vec = (actual.get("state") if isinstance(actual, dict) and "state" in actual
                       else actual)
            resp = reconcile(LawState(dict(pred_vec or {})), dict(act_vec or {}), tol=tol)
            return CapabilityResult.ok(resp)
        return CapabilityResult.fail(f"未知 op '{op}'(law|predict|reconcile)",
                                     code="invalid_input")


def register_world_model_capabilities(model: LawModel, registry=None) -> str:
    """把世界模型装箱并注册进能力注册表(默认 CapabilityRegistry), 返回注册名.

    接线链: CapabilityRegistry → scan_tool_registry → mcp_export(tools/list + call)。
    注册后该世界模型既服务深研管线(LawModel), 也对外 MCP host 可调(law/predict/reconcile)。
    """
    cap = WorldModelCapability(model)
    from huginn.capabilities.registry import CapabilityRegistry

    (registry or CapabilityRegistry).register(cap)
    return cap.name