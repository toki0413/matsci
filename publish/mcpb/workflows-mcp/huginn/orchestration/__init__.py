"""Orchestration protocol — unified contract for all orchestrators.

v23 P2-2: 5 个 Orchestrator 之前各自为政, 入口方法不统一 (run / explore /
plan+execute), 上层 (DecisionArbiter / UnifiedBus / 诊断端点) 没有统一接口
来调用任意 orchestrator. 本模块定义统一 Protocol, 让所有 orchestrator
结构化地满足同一契约, 便于上层用统一接口调用.

采用 typing.Protocol (结构化子类型) 而非 ABC:
  - 5 个 orchestrator 构造器签名差异极大, 共享 __init__ 不现实
  - Protocol 不要求显式继承, 鸭子类型即可
  - runtime_checkable 让 isinstance 检查可用 (只验证方法存在, 不验证签名)

落地承诺 (兑现):
  所有 5 个 orchestrator 都满足 OrchestratorProtocol (有 async run 方法):
    - agents.Orchestrator.run(objective, ...)  ✓ 原生
    - ExplorationOrchestrator.run(objective, ...)  ✓ 加门面包装 explore()
    - ExecutionOrchestrator.run(stages, ...)  ✓ 原生
    - BenchmarkOrchestrator.run(initial_prompt)  ✓ 原生
    - WorkflowOrchestrator.run(script, ...)  ✓ 原生

  不定义 OrchestratorResultProtocol — 5 个 result 类字段差异极大
  (success / overall_success / status=='completed' / 无 success 字段),
  强行统一会让每个类加无意义字段. 若上层需要统一序列化, 走 routes 层适配.
"""
from __future__ import annotations

from huginn.orchestration.protocol import OrchestratorProtocol

__all__ = ["OrchestratorProtocol"]
