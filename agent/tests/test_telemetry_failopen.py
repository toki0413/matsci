"""telemetry 必须 fail-open — 导出器异常不得传播给 agent 主链。

结构不变量(对应架构缝"可观测性不得阻塞/拖垮主 agent"):`TelemetryCollector` 的
span 退出、flush、shutdown 三个**对外可见的调用点**都必须吞掉 exporter 抛出的任何
异常, 只记 debug 日志。若任一 exporter 环节能 propagate, 一次 OTLP 网络故障就会
把一个本来成功的 agent 调用打成失败 —— 这是可观测性反向劫持(observer darkening
call path), 由本门钉死。

对齐 ApxInf 的 fail-open 精神: 工程旁路不得成为运行时的故障源。
"""
from __future__ import annotations

import pytest

from huginn.telemetry import TelemetryCollector


class _RaisingExporter:
    """每个导出环节都抛异常, 模拟导出后端完全不可用."""

    def emit(self, span) -> None:
        raise RuntimeError("boom-emit")

    def flush(self) -> None:
        raise RuntimeError("boom-flush")

    def shutdown(self, block: bool = False) -> None:
        raise RuntimeError("boom-shutdown")


def test_span_contextmanager_swallows_exporter_emit() -> None:
    collector = TelemetryCollector(exporter=_RaisingExporter())
    # 根 span 退出时会把 span 交给 exporter.emit —— 不能抛到测试之外.
    with collector.span("root"):
        collector.span("child")  # 非根 span 不触发 emit, 只是多一条路径
    # 若 emit 异常 propagated, 上行 with 块早已 raise.


def test_flush_fail_open_does_not_raise() -> None:
    collector = TelemetryCollector(exporter=_RaisingExporter())
    with collector.span("root"):
        pass
    collector.flush()  # 必须吞掉 exporter.flush 的异常


def test_shutdown_fail_open_does_not_raise() -> None:
    collector = TelemetryCollector(exporter=_RaisingExporter())
    with collector.span("root"):
        pass
    collector.shutdown(block=True)  # 必须吞掉 exporter.shutdown 的异常


def test_span_output_preserved_even_when_exporter_raises() -> None:
    """fail-open 的同时, 主链路数据(这里指 span 记录)不受导出异常影响."""
    collector = TelemetryCollector(exporter=_RaisingExporter())
    with collector.span("root", phase="observe"), collector.span("child"):
        pass
    summary = collector.summary()
    assert summary["by_name"]["root"]["count"] == 1
    assert summary["by_name"]["child"]["count"] == 1


@pytest.mark.parametrize("method", ["emit", "flush", "shutdown"])
def test_all_export_hooks_survive_operational_exception(method: str) -> None:
    """任一导出钩子抛**操作性 Exception**(网络/后端/内存类故障)都不冒泡。

    注意 `except Exception` 不会吞 SystemExit/KeyboardInterrupt(进程控制流),
    这是刻意为之 —— fail-open 只覆盖"可观测性后端故障", 不吞进程控制权。
    仍要吞掉的最大真实风险源是导出挂起, 因此这里用工况主要是运行期 Exception。
    """
    class _HostileExporter:
        def emit(self, span): raise MemoryError("nope")
        def flush(self): raise TimeoutError("unreachable")
        def shutdown(self, block=False): raise OSError("closed")

    collector = TelemetryCollector(exporter=_HostileExporter())
    with collector.span("root"):
        pass
    if method == "flush":
        collector.flush()
    elif method == "shutdown":
        collector.shutdown()
