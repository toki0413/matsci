"""OrchestratorProtocol 一致性测试 — 验证 5 个 Orchestrator 满足统一契约.

v23 P2-2: huginn.orchestration.OrchestratorProtocol 是统一 orchestrator 契约.
本测试静态验证所有 5 个 Orchestrator 类都有 async run 方法, 满足协议.
不实例化 (各 Orchestrator 构造器签名差异极大), 只做结构化子类型检查.
"""

from __future__ import annotations

import inspect

from huginn.orchestration import OrchestratorProtocol


def _all_orchestrator_classes() -> dict[str, type]:
    """收集所有 5 个 Orchestrator 类.

    延迟 import 避免模块加载顺序问题.
    """
    from huginn.agents.orchestrator import Orchestrator as AgentsOrchestrator
    from huginn.autoloop.dynamic_workflow import WorkflowOrchestrator
    from huginn.bench.orchestrator import BenchmarkOrchestrator
    from huginn.execution.orchestrator import ExecutionOrchestrator
    from huginn.exploration.orchestrator import ExplorationOrchestrator

    return {
        "agents.Orchestrator": AgentsOrchestrator,
        "ExplorationOrchestrator": ExplorationOrchestrator,
        "ExecutionOrchestrator": ExecutionOrchestrator,
        "BenchmarkOrchestrator": BenchmarkOrchestrator,
        "WorkflowOrchestrator": WorkflowOrchestrator,
    }


class TestOrchestratorProtocolImportable:
    """Protocol 自身可导入, runtime_checkable 可用."""

    def test_protocol_importable(self):
        from huginn.orchestration.protocol import OrchestratorProtocol

        assert OrchestratorProtocol is not None

    def test_protocol_exported_from_package(self):
        import huginn.orchestration as pkg

        assert hasattr(pkg, "OrchestratorProtocol")
        assert "OrchestratorProtocol" in pkg.__all__

    def test_protocol_is_runtime_checkable(self):
        """runtime_checkable 让 isinstance 可用 (只验证方法存在)."""
        # Protocol 加了 @runtime_checkable 装饰器才能 isinstance
        # 直接验证 hasattr(OrchestratorProtocol, '_is_runtime_protocol')
        assert getattr(OrchestratorProtocol, "_is_runtime_protocol", False) is True


class TestAllOrchestratorsSatisfyProtocol:
    """5 个 Orchestrator 类都满足 OrchestratorProtocol (有 async run 方法)."""

    def test_all_have_async_run(self):
        orchs = _all_orchestrator_classes()
        assert len(orchs) == 5, f"期望 5 个 Orchestrator, 实际 {len(orchs)}"

        failures: list[str] = []
        for name, cls in orchs.items():
            if not hasattr(cls, "run"):
                failures.append(f"{name} 没有 run 方法")
                continue
            if not inspect.iscoroutinefunction(cls.run):
                failures.append(f"{name}.run 不是 async")
        assert not failures, "协议违约: " + "; ".join(failures)

    def test_each_orchestrator_satisfies_protocol(self):
        """逐个验证, 失败时能精确定位哪个 Orchestrator 不满足."""
        orchs = _all_orchestrator_classes()
        for name, cls in orchs.items():
            assert hasattr(cls, "run"), f"{name} 缺 run 方法"
            assert inspect.iscoroutinefunction(cls.run), f"{name}.run 必须 async"


class TestExplorationOrchestratorRunFacade:
    """ExplorationOrchestrator.run 是 P2-2 新加的门面, 单独验证其语义."""

    def test_run_method_exists_and_is_async(self):
        from huginn.exploration.orchestrator import ExplorationOrchestrator

        assert hasattr(ExplorationOrchestrator, "run")
        assert inspect.iscoroutinefunction(ExplorationOrchestrator.run)

    def test_run_signature_accepts_objective(self):
        """run 门面签名: (objective, initial_branches=None, **kwargs).

        initial_branches 默认 None, 让上层能 await orch.run("obj") 直接调用.
        """
        from huginn.exploration.orchestrator import ExplorationOrchestrator

        sig = inspect.signature(ExplorationOrchestrator.run)
        params = sig.parameters
        assert "objective" in params
        assert "initial_branches" in params
        # initial_branches 默认 None (非必填)
        assert params["initial_branches"].default is None

    def test_run_returns_exploration_result(self):
        """run 门面转发到 explore(), 返回类型一致."""
        from huginn.exploration.orchestrator import (
            ExplorationOrchestrator,
            ExplorationResult,
        )

        sig = inspect.signature(ExplorationOrchestrator.run)
        # 返回注解是 ExplorationResult
        assert sig.return_annotation in (ExplorationResult, "ExplorationResult")


class TestProtocolNotExportingResultProtocol:
    """P2-2 设计决策: 不定义 OrchestratorResultProtocol (YAGNI).

    5 个 result 类字段差异极大, 强行统一会让每个类加无意义字段.
    本测试固化该决策 — 防止后续误加 ResultProtocol.
    """

    def test_result_protocol_not_exported(self):
        import huginn.orchestration as pkg

        assert "OrchestratorResultProtocol" not in pkg.__all__
        assert not hasattr(pkg, "OrchestratorResultProtocol")
