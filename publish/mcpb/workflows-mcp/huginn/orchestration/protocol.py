"""Unified protocol for all orchestrators.

v23 P2-2: 5 个 Orchestrator 之前各自为政, 入口方法不统一 (run / explore /
plan+execute), 上层 (DecisionArbiter / UnifiedBus / 诊断端点) 没有统一接口
来调用任意 orchestrator. 本模块定义 OrchestratorProtocol, 让所有
orchestrator 结构化地满足同一契约.

设计原则 (ponytail / YAGNI):
  - Protocol 只要求 "有 async run 方法". 5 个 orchestrator 的 run 签名差异极大
    (objective / stages / script / initial_prompt), 强行统一签名会引入 *args/**kwargs
    黑盒, 失去类型检查价值. runtime_checkable 只验证方法存在, 不验证签名 —
    这正好匹配现状: 上层需要的是 "能 await run()" 的鸭子类型, 不是签名一致性.
  - 不定义 OrchestratorResultProtocol. 5 个 result 类
    (OrchestratorResult / ExplorationResult / WorkflowExecutionRecord /
    WorkflowResult / str) 字段差异极大 (success / overall_success / status=='completed' /
    无 success 字段), 强行统一 success/summary 字段会让每个 result 类加无意义字段.
    若上层需要统一序列化, 走 routes 层的 to_dict 适配即可, 不需要 Protocol 强制.

落地承诺 (兑现):
  - OrchestratorProtocol: 所有 5 个 orchestrator 都满足 (有 async run 方法).
    - agents.Orchestrator.run(objective, ...)  ✓ 原生
    - ExplorationOrchestrator.run(objective, ...)  ✓ 加门面包装 explore()
    - ExecutionOrchestrator.run(stages, ...)  ✓ 原生
    - BenchmarkOrchestrator.run(initial_prompt)  ✓ 原生
    - WorkflowOrchestrator.run(script, ...)  ✓ 原生
  - 不再声明 OrchestratorResultProtocol (无消费方, 删除避免死代码).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OrchestratorProtocol(Protocol):
    """统一 orchestrator 契约: 必须有 async run 方法.

    runtime_checkable 让 isinstance(obj, OrchestratorProtocol) 可用,
    但只验证方法存在, 不验证签名 (这是有意为之 — 见模块 docstring).
    """

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the orchestrator's main workflow.

        各 orchestrator 的具体签名由子类决定. 调用方应查阅子类 docstring.
        """
        ...


__all__ = ["OrchestratorProtocol"]
