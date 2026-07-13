"""信息隔离层 — 控制哪些上下文下发给哪些探索 agent.

核心机制: 每轮探索开始前, 根 agent 显式决定 "当前偏好假设" 的可见范围.
默认对探索 agent 隐藏, 仅根 agent + 对抗 agent 可见.

为什么需要: 不隔离的话, 所有 agent 都会顺着当前偏好假设产出措辞不同
但思想同源的方案. 直觉应该先独立孵化, 再聚合, 而不是先广播再探索.

与 "捕捉研究者模糊直觉" 偏好的关系:
- researcher_intuition 进入 ContextBundle, 但不下发给探索 agent
- 根 agent 用 intuition 做"调度权重" (决定哪些思想族入场), 不做"探索约束"
- exploratory 状态 (current_preferred_hypothesis=None) 时自动放宽隔离
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


AgentRole = Literal[
    "root",           # 根 agent / 元认知层
    "exploration",    # 探索 agent (默认隔离)
    "adversarial",    # 对抗 agent (可见全部, 用于审计)
    "executor",       # 执行 agent (只看任务定义 + 计划)
]


@dataclass
class ContextBundle:
    """一轮探索的完整上下文, 按可见性分级."""

    # 全局共享 (所有角色可见)
    global_math_background: str = ""
    task_definition: str = ""

    # 仅 root + adversarial 可见
    current_preferred_hypothesis: str | None = None
    researcher_intuition: str = ""  # 用户的模糊直觉, 不下发

    # 仅 root 可见 (避免 agent 看到全局后自我归类)
    method_family_registry: dict[str, Any] = field(default_factory=dict)

    # exploratory 标记: current_preferred_hypothesis=None 时为 True
    @property
    def is_exploratory(self) -> bool:
        return self.current_preferred_hypothesis is None


@dataclass
class IsolationPolicy:
    """可见性规则. 默认实现 = 严格隔离.

    重写 allowed_fields 可放宽, 但 exploratory 状态下自动放宽为 partial.
    """

    # 每个角色能看到的 ContextBundle 字段名
    # ponytail: 用字段名清单而不是角色继承树, 避免过度抽象
    visibility: dict[AgentRole, frozenset[str]] = field(default_factory=lambda: {
        "root": frozenset(ContextBundle.__dataclass_fields__.keys()),
        "adversarial": frozenset({
            "global_math_background",
            "task_definition",
            "current_preferred_hypothesis",
            "researcher_intuition",
        }),
        "exploration": frozenset({
            "global_math_background",
            "task_definition",
        }),
        "executor": frozenset({
            "task_definition",
        }),
    })

    def fields_for(self, role: AgentRole, bundle: ContextBundle) -> frozenset[str]:
        """返回某角色实际能看到的字段集合.

        exploratory 状态下, exploration 自动放宽为 partial:
        能看到 method_family_registry 的 key 列表 (用于自我归类), 但看不到
        current_preferred_hypothesis (因为没有) 和 researcher_intuition.
        """
        base = self.visibility.get(role, frozenset())
        if role == "exploration" and bundle.is_exploratory:
            # 放宽: 允许看到方法族 id 列表 (不是完整 registry)
            return base | {"_method_family_ids"}
        return base


def isolate(
    bundle: ContextBundle,
    role: AgentRole,
    policy: IsolationPolicy | None = None,
) -> dict[str, Any]:
    """按角色裁剪 ContextBundle, 返回该角色可见的上下文 dict.

    这是信息隔离层的唯一入口. 探索 agent 拿到的就是这个 dict, 不是完整 bundle.
    """
    pol = policy or IsolationPolicy()
    allowed = pol.fields_for(role, bundle)

    out: dict[str, Any] = {}
    for field_name in allowed:
        # _method_family_ids 是派生字段, 不是 ContextBundle 的真实字段
        if field_name == "_method_family_ids":
            out[field_name] = list(bundle.method_family_registry.keys())
            continue
        if hasattr(bundle, field_name):
            value = getattr(bundle, field_name)
            # 跳过 None 和空字符串, 不污染下游 prompt
            if value is None or value == "" or value == {}:
                continue
            out[field_name] = value
    return out


# ── 自检 ─────────────────────────────────────────────────────────

def _selfcheck() -> None:
    bundle = ContextBundle(
        global_math_background="DFT + GP",
        task_definition="predict formation energy",
        current_preferred_hypothesis="GPR with compositional features",
        researcher_intuition="maybe enthalpy matters more",
        method_family_registry={"dft-direct": {}, "gp": {}},
    )

    # 1. exploration agent 看不到 hypothesis 和 intuition
    exp_ctx = isolate(bundle, "exploration")
    assert "current_preferred_hypothesis" not in exp_ctx, (
        f"exploration 不应看到 hypothesis, got keys={list(exp_ctx)}"
    )
    assert "researcher_intuition" not in exp_ctx
    assert "global_math_background" in exp_ctx

    # 2. adversarial agent 能看到 hypothesis (用于审计)
    adv_ctx = isolate(bundle, "adversarial")
    assert "current_preferred_hypothesis" in adv_ctx
    assert "researcher_intuition" in adv_ctx
    assert "method_family_registry" not in adv_ctx, "adversarial 不需要 registry"

    # 3. exploratory 状态自动放宽: exploration 能看到 family ids
    bundle_expl = ContextBundle(
        global_math_background="DFT",
        task_definition="explore",
        current_preferred_hypothesis=None,  # exploratory
        method_family_registry={"dft": {}, "gp": {}},
    )
    exp_ctx2 = isolate(bundle_expl, "exploration")
    assert "_method_family_ids" in exp_ctx2, "exploratory 应放宽到看到 family ids"
    assert exp_ctx2["_method_family_ids"] == ["dft", "gp"]

    # 4. root 看到全部
    root_ctx = isolate(bundle, "root")
    assert "method_family_registry" in root_ctx
    assert "researcher_intuition" in root_ctx

    print("context_isolation selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
