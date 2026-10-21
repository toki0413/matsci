"""VIGIL 独立监控进程入口（``python -m huginn.metacog.vigil_daemon``）.

crash-only：只读事件/写心跳与诊断，不执行用户任务、不写 ``authorize``、
不改 ``PersonaKernel``。父进程消失即退出；任何未捕获异常都写 L4 心跳后退出，
由 watchdog 决定是否退避重启。
"""
from __future__ import annotations

import logging
import os
import signal
import threading

from huginn.metacog.vigil import MonitorStatus, VigilMonitor
from huginn.metacog.vigil_process import heartbeat_path, write_heartbeat

logger = logging.getLogger(__name__)


def _l4_status(reason: str) -> MonitorStatus:
    return MonitorStatus(
        monitor_id="vigil",
        tenant_id="default",
        state="L4",
        trust="unknown",
        coverage={},
        last_heartbeat="",
        degraded_reason=reason[:200],
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from huginn.env_access import env_float, env_int

    interval = max(1.0, float(env_int("HUGINN_VIGIL_HEARTBEAT_S")))
    monitor = VigilMonitor(
        "vigil", error_threshold=env_float("HUGINN_VIGIL_ERROR_THRESHOLD")
    )
    path = heartbeat_path()
    parent_pid = os.getppid()
    stop = threading.Event()

    def _on_signal(signum, frame) -> None:  # noqa: ARG001
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    try:
        while not stop.is_set():
            status = monitor.heartbeat()
            status = monitor.tick()
            write_heartbeat(status, path=path)
            if os.getppid() != parent_pid:
                break  # 父进程消失 → crash-only 退出
            stop.wait(interval)
    except Exception as exc:  # 未捕获异常 → L4 可见降级，不静默
        logger.exception("vigil daemon crashed")
        write_heartbeat(_l4_status(f"unhandled: {exc}"), path=path)
        return 1

    write_heartbeat(
        {
            **monitor.status().to_dict(),
            "state": "L3",
            "trust": "unknown",
            "degraded_reason": "daemon stopped",
        },
        path=path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
