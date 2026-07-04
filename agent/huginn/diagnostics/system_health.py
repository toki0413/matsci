"""系统资源健康监控 —— CPU / 内存 / 磁盘负载异常自动诊断。

telemetry.py 只管进程级 RSS，health_dashboard 只管 per-tool 成功率，
中间缺了一层：系统资源被打满时谁也看不到。这个模块补上：

  * 后台守护线程每隔几秒采样一次 psutil
  * 超过阈值（CPU 需持续 N 秒，内存/磁盘瞬时）就生成 AnomalyEvent
  * diagnose() 做根因分析：列出吃资源最多的进程，给出针对性建议
  * auto_fix 开了的话，自动熔断处于不健康状态的工具、清缓存

psutil 是可选依赖，没装就优雅降级——snapshot 返回 psutil_available=False，
后台线程不启动，调用方自己决定怎么处理。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# psutil 是可选依赖，整个模块不能因为它 import 失败就崩
try:
    import psutil

    _PSUTIL_OK = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _PSUTIL_OK = False


# ── 数据结构 ────────────────────────────────────────────────────────


@dataclass
class SystemMetrics:
    """某一时刻的系统资源快照。"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    swap_percent: float
    swap_used_mb: float
    # {mountpoint: {"percent": float, "used_gb": float, "free_gb": float, "total_gb": float}}
    disk: dict[str, dict[str, float]] = field(default_factory=dict)
    # (1min, 5min, 15min)，Windows 上是 None
    load_avg: tuple[float, float, float] | None = None
    psutil_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "memory_used_mb": round(self.memory_used_mb, 1),
            "memory_total_mb": round(self.memory_total_mb, 1),
            "swap_percent": round(self.swap_percent, 1),
            "swap_used_mb": round(self.swap_used_mb, 1),
            "disk": {
                mp: {k: round(v, 2) for k, v in d.items()}
                for mp, d in self.disk.items()
            },
            "load_avg": self.load_avg,
            "psutil_available": self.psutil_available,
        }


@dataclass
class AnomalyEvent:
    """一次诊断出的资源异常。"""

    timestamp: float
    resource: str  # "cpu" | "memory" | "disk" | "swap"
    severity: str  # "warning" | "critical"
    value: float
    threshold: float
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    auto_fixed: bool = False
    auto_fix_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "resource": self.resource,
            "severity": self.severity,
            "value": round(self.value, 1),
            "threshold": round(self.threshold, 1),
            "message": self.message,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "auto_fixed": self.auto_fixed,
            "auto_fix_detail": self.auto_fix_detail,
        }


@dataclass
class ThresholdPolicy:
    """资源异常阈值。均衡策略是默认值。"""

    cpu_percent: float = 85.0
    cpu_sustained_seconds: float = 30.0
    memory_percent: float = 85.0
    disk_percent: float = 90.0
    swap_percent: float = 80.0
    # critical 比 warning 高一档，区分严重程度
    cpu_critical_percent: float = 95.0
    memory_critical_percent: float = 95.0
    disk_critical_percent: float = 97.0


# ── 监控器 ──────────────────────────────────────────────────────────


class SystemHealthMonitor:
    """进程级单例，后台守护线程采样系统资源。

    用法::

        monitor = SystemHealthMonitor.shared()
        monitor.start()                  # 启动后台线程

        metrics = monitor.snapshot()     # 拿当前快照
        events = monitor.diagnose()      # 诊断当前异常 + 根因
        top = monitor.top_processes(5)   # 吃 CPU 最多的 5 个进程
    """

    _singleton_lock = threading.Lock()
    _singleton: SystemHealthMonitor | None = None

    def __init__(
        self,
        policy: ThresholdPolicy | None = None,
        poll_interval: float = 5.0,
    ) -> None:
        self._policy = policy or ThresholdPolicy()
        self._poll_interval = poll_interval
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest: SystemMetrics | None = None
        self._anomalies: deque[AnomalyEvent] = deque(maxlen=100)
        # CPU 持续窗口：存最近 N 个采样点，判断是否持续超阈值
        window_size = max(1, int(self._policy.cpu_sustained_seconds / poll_interval))
        self._cpu_window: deque[float] = deque(maxlen=window_size)
        # 缓存的 top processes，避免每次 top_processes() 都遍历进程列表
        self._top_cpu_cache: list[dict[str, Any]] = []
        self._top_mem_cache: list[dict[str, Any]] = []
        # 防止同一个异常反复触发 auto_fix
        self._last_auto_fix_ts: dict[str, float] = {}
        self._auto_fix_cooldown = 120.0  # 同一 resource 2 分钟内不重复修

    @classmethod
    def shared(cls) -> SystemHealthMonitor:
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    # -------------------------------------------------- 生命周期

    def start(self) -> None:
        """启动后台监控线程。已启动则 no-op。psutil 没装也不报错。"""
        if not _PSUTIL_OK:
            logger.info("psutil not installed, system health monitor disabled")
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._monitor_loop,
                name="huginn-system-health",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """停掉后台线程。"""
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -------------------------------------------------- 公开查询

    def snapshot(self) -> SystemMetrics:
        """当前系统资源快照。后台线程没跑就现场采一次。"""
        if not _PSUTIL_OK:
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
                swap_percent=0.0,
                swap_used_mb=0.0,
                psutil_available=False,
            )
        with self._lock:
            if self._latest is not None:
                return self._latest
        # 后台线程没启动，现场采一次
        metrics = self._collect()
        with self._lock:
            self._latest = metrics
        return metrics

    def diagnose(self) -> list[AnomalyEvent]:
        """诊断当前资源状态，返回所有活跃异常（含根因和建议）。"""
        metrics = self.snapshot()
        if not metrics.psutil_available:
            return []
        events: list[AnomalyEvent] = []
        cpu_event = self._diagnose_cpu(metrics)
        if cpu_event:
            events.append(cpu_event)
        mem_event = self._diagnose_memory(metrics)
        if mem_event:
            events.append(mem_event)
        events.extend(self._diagnose_disk(metrics))
        swap_event = self._diagnose_swap(metrics)
        if swap_event:
            events.append(swap_event)
        return events

    def recent_anomalies(self, limit: int = 20) -> list[AnomalyEvent]:
        """最近记录到的异常事件（后台线程监测到的）。"""
        with self._lock:
            items = list(self._anomalies)
        return items[-limit:][::-1]  # 最新的在前

    def top_processes(self, n: int = 10, by: str = "cpu") -> list[dict[str, Any]]:
        """返回吃资源最多的 n 个进程。by="cpu" 或 "memory"。

        返回缓存的采样结果（后台线程每轮更新一次），避免调用方阻塞。
        """
        with self._lock:
            cache = self._top_cpu_cache if by == "cpu" else self._top_mem_cache
            return list(cache[:n])

    # -------------------------------------------------- 后台循环

    def _monitor_loop(self) -> None:
        # 第一次 cpu_percent 返回 0，先暖一下
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        while not self._stop_event.is_set():
            try:
                metrics = self._collect()
                with self._lock:
                    self._latest = metrics
                    self._cpu_window.append(metrics.cpu_percent)
                    self._refresh_process_cache()
                self._check_and_record(metrics)
            except Exception:
                # 后台线程不能因为一次采样失败就挂
                logger.debug("system health sample failed", exc_info=True)
            self._stop_event.wait(self._poll_interval)

    def _collect(self) -> SystemMetrics:
        """采一次系统资源快照。调用方保证 psutil 可用。"""
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        cpu = psutil.cpu_percent(interval=None)

        # 磁盘：监控所有分区，但跳过没有 total 的（比如光驱/网络盘）
        disk: dict[str, dict[str, float]] = {}
        try:
            for part in psutil.disk_partitions(all=False):
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                except (PermissionError, OSError):
                    continue
                disk[part.mountpoint] = {
                    "percent": usage.percent,
                    "used_gb": usage.used / 1024**3,
                    "free_gb": usage.free / 1024**3,
                    "total_gb": usage.total / 1024**3,
                }
        except Exception:
            pass

        # loadavg 在 Windows 上不存在
        load_avg: tuple[float, float, float] | None = None
        try:
            load = os.getloadavg()
            load_avg = (load[0], load[1], load[2])
        except (AttributeError, OSError):
            pass

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu,
            memory_percent=vm.percent,
            memory_used_mb=vm.used / 1024**2,
            memory_total_mb=vm.total / 1024**2,
            swap_percent=sm.percent,
            swap_used_mb=sm.used / 1024**2,
            disk=disk,
            load_avg=load_avg,
            psutil_available=True,
        )

    def _refresh_process_cache(self) -> None:
        """遍历进程列表，更新 top CPU / memory 缓存。调用方持锁。"""
        cpu_list: list[dict[str, Any]] = []
        mem_list: list[dict[str, Any]] = []
        own_pid = os.getpid()
        for proc in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent", "memory_info"]):
            try:
                info = proc.info
                if info["pid"] == own_pid:
                    continue
                cpu_pct = info.get("cpu_percent") or 0.0
                mem_pct = info.get("memory_percent") or 0.0
                rss = info.get("memory_info")
                rss_mb = rss.rss / 1024**2 if rss else 0.0
                entry = {
                    "pid": info["pid"],
                    "name": info.get("name", "?"),
                    "cpu_percent": round(cpu_pct, 1),
                    "memory_percent": round(mem_pct, 2),
                    "rss_mb": round(rss_mb, 1),
                }
                cpu_list.append(entry)
                mem_list.append(entry)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        cpu_list.sort(key=lambda x: x["cpu_percent"], reverse=True)
        mem_list.sort(key=lambda x: x["memory_percent"], reverse=True)
        self._top_cpu_cache = cpu_list[:20]
        self._top_mem_cache = mem_list[:20]

    # -------------------------------------------------- 阈值检测

    def _check_and_record(self, metrics: SystemMetrics) -> None:
        """后台线程用：检查阈值，记录异常事件，按需 auto_fix。"""
        events: list[AnomalyEvent] = []

        # CPU 需要持续超阈值才算异常
        with self._lock:
            window = list(self._cpu_window)
        if len(window) >= self._cpu_window.maxlen and all(
            v >= self._policy.cpu_percent for v in window
        ):
            ev = self._diagnose_cpu(metrics)
            if ev:
                events.append(ev)

        mem_ev = self._diagnose_memory(metrics)
        if mem_ev:
            events.append(mem_ev)
        events.extend(self._diagnose_disk(metrics))
        swap_ev = self._diagnose_swap(metrics)
        if swap_ev:
            events.append(swap_ev)

        for ev in events:
            self._maybe_auto_fix(ev)
            with self._lock:
                self._anomalies.append(ev)

        if events:
            self._fire_alert_webhook(events)

    def _fire_alert_webhook(self, events: list[AnomalyEvent]) -> None:
        """有异常就往外部 webhook 推一条告警，没配 URL 就跳过。"""
        url = os.environ.get("HUGINN_ALERT_WEBHOOK_URL")
        if not url:
            return
        for ev in events:
            payload = json.dumps(
                {
                    "event": "anomaly",
                    "resource": ev.resource,
                    "severity": ev.severity,
                    "message": ev.message,
                    "value": ev.value,
                    "threshold": ev.threshold,
                    "timestamp": ev.timestamp,
                }
            ).encode("utf-8")
            try:
                req = urllib.request.Request(
                    url, data=payload, headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                # webhook 挂了不能把监控线程也带崩
                logger.debug("alert webhook POST failed", exc_info=True)

    def _diagnose_cpu(self, m: SystemMetrics) -> AnomalyEvent | None:
        if m.cpu_percent < self._policy.cpu_percent:
            return None
        critical = m.cpu_percent >= self._policy.cpu_critical_percent
        severity = "critical" if critical else "warning"
        top = self.top_processes(5, by="cpu")
        return AnomalyEvent(
            timestamp=m.timestamp,
            resource="cpu",
            severity=severity,
            value=m.cpu_percent,
            threshold=self._policy.cpu_percent,
            message=f"CPU usage {m.cpu_percent:.1f}% above {self._policy.cpu_percent:.0f}%",
            evidence={"top_processes": top, "load_avg": m.load_avg},
            recommendations=[
                "降低并行工具数 (HUGINN_MAX_PARALLEL_TOOLS)",
                "检查是否有模拟任务卡在死循环",
                "VASP 任务可降低 NPAR / NSIM 减轻 CPU 压力",
                "LAMMPS 任务检查是否用了全邻居列表 (neigh_modify one)",
            ],
        )

    def _diagnose_memory(self, m: SystemMetrics) -> AnomalyEvent | None:
        if m.memory_percent < self._policy.memory_percent:
            return None
        critical = m.memory_percent >= self._policy.memory_critical_percent
        severity = "critical" if critical else "warning"
        top = self.top_processes(5, by="memory")
        return AnomalyEvent(
            timestamp=m.timestamp,
            resource="memory",
            severity=severity,
            value=m.memory_percent,
            threshold=self._policy.memory_percent,
            message=f"Memory usage {m.memory_percent:.1f}% above {self._policy.memory_percent:.0f}%",
            evidence={
                "top_processes": top,
                "used_mb": m.memory_used_mb,
                "total_mb": m.memory_total_mb,
                "swap_percent": m.swap_percent,
            },
            recommendations=[
                "缩小工具缓存 (HUGINN_TOOL_CACHE_SIZE)",
                "关闭不再使用的大结构对象",
                "降低 k 点密度或截断能减少 DFT 内存占用",
                "MD 任务减小 timestep 或原子数",
                "检查是否有内存泄漏 (long-running agent 会累积 span)",
            ],
        )

    def _diagnose_disk(self, m: SystemMetrics) -> list[AnomalyEvent]:
        events: list[AnomalyEvent] = []
        for mountpoint, d in m.disk.items():
            if d["percent"] < self._policy.disk_percent:
                continue
            critical = d["percent"] >= self._policy.disk_critical_percent
            severity = "critical" if critical else "warning"
            events.append(AnomalyEvent(
                timestamp=m.timestamp,
                resource="disk",
                severity=severity,
                value=d["percent"],
                threshold=self._policy.disk_percent,
                message=f"Disk {mountpoint} usage {d['percent']:.1f}% above {self._policy.disk_percent:.0f}%",
                evidence={
                    "mountpoint": mountpoint,
                    "used_gb": d["used_gb"],
                    "free_gb": d["free_gb"],
                    "total_gb": d["total_gb"],
                },
                recommendations=[
                    "清理 HUGINN_CACHE_DIR 下的缓存文件",
                    "归档旧的作业输出 (job_tool 的结果文件)",
                    "删除临时文件: find $HUGINN_CACHE_DIR -name '*.tmp' -delete",
                    "VASP: 清理 WAVECAR/CHGCAR (大文件)",
                    "检查日志文件是否过大",
                ],
            ))
        return events

    def _diagnose_swap(self, m: SystemMetrics) -> AnomalyEvent | None:
        if m.swap_percent < self._policy.swap_percent:
            return None
        return AnomalyEvent(
            timestamp=m.timestamp,
            resource="swap",
            severity="warning",
            value=m.swap_percent,
            threshold=self._policy.swap_percent,
            message=f"Swap usage {m.swap_percent:.1f}% above {self._policy.swap_percent:.0f}%",
            evidence={"swap_used_mb": m.swap_used_mb},
            recommendations=[
                "降低内存压力 (见 memory 建议)",
                "如果持续 swap thrashing，考虑增加物理内存",
                "减少并发工具数",
            ],
        )

    # -------------------------------------------------- 自动修复

    def _maybe_auto_fix(self, event: AnomalyEvent) -> None:
        """根据 feature flag 决定要不要自动修复。只动安全的东西。"""
        from huginn.feature_flags import FeatureFlags

        if not FeatureFlags.shared().is_enabled("system_health_auto_fix"):
            return
        if not FeatureFlags.shared().is_enabled("system_health_monitor"):
            return

        # 同一资源类型在冷却期内不重复修
        now = time.time()
        with self._lock:
            last = self._last_auto_fix_ts.get(event.resource, 0)
            if now - last < self._auto_fix_cooldown:
                return
            self._last_auto_fix_ts[event.resource] = now

        detail = self._auto_fix_action(event)
        if detail:
            event.auto_fixed = True
            event.auto_fix_detail = detail
            logger.warning(
                "auto-fix applied for %s anomaly: %s", event.resource, detail
            )

    def _auto_fix_action(self, event: AnomalyEvent) -> str:
        """执行具体的自动修复，返回描述。只做安全操作。"""
        if event.resource in ("cpu", "memory"):
            # 系统资源吃紧时，把处于不健康状态的工具强制熔断，阻止它继续吃资源
            try:
                from huginn.agents.circuit_breaker import CircuitBreaker
                from huginn.agents.health_dashboard import HealthDashboard

                breaker = CircuitBreaker.shared()
                dash = HealthDashboard.shared()
                tripped: list[str] = []
                for tool_stat in dash.get_all():
                    verdict = tool_stat.get("verdict", "")
                    tool_name = tool_stat.get("tool", "")
                    if verdict in ("unhealthy", "degraded") and tool_name:
                        breaker.force_open(
                            tool_name,
                            reason=f"auto-fix: {event.resource} {event.value:.1f}% (threshold {event.threshold:.0f}%)",
                        )
                        tripped.append(tool_name)
                if tripped:
                    return f"force-opened circuit breakers for unhealthy tools: {', '.join(tripped)}"
            except Exception as exc:
                logger.debug("auto-fix circuit breaker failed: %s", exc)
                return ""

        if event.resource == "disk":
            # 磁盘满了清工具缓存，缓存是可重建的
            cache_dir = os.environ.get("HUGINN_CACHE_DIR")
            if not cache_dir:
                return ""
            try:
                import shutil
                from pathlib import Path

                cache_path = Path(cache_dir)
                if not cache_path.exists():
                    return ""
                # 只清 cache 子目录下的 .tmp 和 tool_cache，不动配置和审计日志
                freed = 0.0
                for sub in ("tool_cache", "tmp"):
                    target = cache_path / sub
                    if target.exists() and target.is_dir():
                        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                        shutil.rmtree(target, ignore_errors=True)
                        target.mkdir(parents=True, exist_ok=True)
                        freed += size / 1024**3
                if freed > 0:
                    return f"cleared tool cache, freed {freed:.2f} GB"
            except Exception as exc:
                logger.debug("auto-fix disk cleanup failed: %s", exc)

        return ""
