"""统一成本账本 (CostLedger) — 把散落各处的成本归一成单一货币 + 维度标签.

为什么要统一: 现有成本机制各自为政 (research_budget 按次数、rate_limiter 按美元、
TokenBudget 按 token、BudgetPolicy 按 CPU/GPU 资源), 无法回答"这个任务到底花了多少、
花在哪、值不值"。CostLedger 把所有记录归一到 **usd** (单一可比货币), 并打上
**维度标签** (推理/计算/外部) 与归因 (tool/phase/session), 是"精细成本控制"的地基。

三种单位混入的换算由**换算率**配置决定 (provider 的 usd_per_1k_tokens 等), 未配置的
率默认 0 → 该单位无法折算美元时记原始量, usd 记为 0 (量纲仍可统计次数/机时)。

用法::

    from huginn.cost_ledger import CostLedger, CostDimension, CostUnit

    ledger = CostLedger.from_env()          # 从 HUGINN_COST_* 建, 拿全局率
    ledger.record(CostDimension.LLM, 1200, CostUnit.TOKENS, tool="vasp_tool", phase="explore")
    total = ledger.total_usd()              # 统一货币累计
    by_dim = ledger.by_dimension()          # 花在哪 (推理/计算/外部)
    decision, reason = ledger.check_budget(10.0)  # ALLOW/WARN/DENY
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
import time

try:
    from huginn.env_defaults import get_bool, get_float
except ImportError:  # pragma: no cover - env_defaults 始终存在
    def get_float(*a, **k):  # type: ignore
        return float(k.get("default", 0.0))

    def get_bool(*a, **k):  # type: ignore
        return bool(k.get("default", False))


__all__ = [
    "CostDimension",
    "CostUnit",
    "CostEntry",
    "CostLedger",
    "get_cost_ledger",
    "reset_cost_ledger",
]


class CostDimension(StrEnum):
    """成本维度标签 — 回答"花在哪"的粗分类."""

    LLM = "llm"  # 模型推理 token 成本
    COMPUTE = "compute"  # 工具计算 / HPC 机时 (CPU/GPU)
    EXTERNAL = "external"  # 外部 API / 服务调用
    OTHER = "other"  # 其它 (审批/人工/兜底)


class CostUnit(StrEnum):
    """原始计量单位 — 记录时保持原量纲, 另存归一后的 usd."""

    USD = "usd"
    TOKENS = "tokens"
    CPU_HOURS = "cpu_hours"
    GPU_HOURS = "gpu_hours"
    CALLS = "calls"


@dataclass(frozen=True)
class CostEntry:
    """一条成本记录: 原始量 + 归一美元 + 维度标签 + 归因."""

    dimension: CostDimension
    amount: float
    unit: CostUnit
    usd: float
    tool: str = ""
    phase: str = ""
    session_id: str = "default"
    label: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "amount": self.amount,
            "unit": self.unit.value,
            "usd": self.usd,
            "tool": self.tool,
            "phase": self.phase,
            "session_id": self.session_id,
            "label": self.label,
            "ts": self.ts,
        }


class CostLedger:
    """进程内统一成本账本. 线程安全 (append 原子, 读时快照)."""

    def __init__(
        self,
        *,
        usd_per_1k_tokens: float = 0.0,
        usd_per_cpu_hour: float = 0.0,
        usd_per_gpu_hour: float = 0.0,
        usd_per_call: float = 0.0,
    ) -> None:
        self.usd_per_1k_tokens = usd_per_1k_tokens
        self.usd_per_cpu_hour = usd_per_cpu_hour
        self.usd_per_gpu_hour = usd_per_gpu_hour
        self.usd_per_call = usd_per_call
        self._entries: list[CostEntry] = []

    @classmethod
    def from_env(cls) -> "CostLedger":
        """从 HUGINN_COST_* env 读取换算率构建."""
        return cls(
            usd_per_1k_tokens=get_float("HUGINN_COST_USD_PER_1K_TOKENS", default=0.0),
            usd_per_cpu_hour=get_float("HUGINN_COST_USD_PER_CPU_HOUR", default=0.0),
            usd_per_gpu_hour=get_float("HUGINN_COST_USD_PER_GPU_HOUR", default=0.0),
            usd_per_call=get_float("HUGINN_COST_USD_PER_CALL", default=0.0),
        )

    # ── 记账 ─────────────────────────────────────────────────────
    def record(
        self,
        dimension: CostDimension | str,
        amount: float,
        unit: CostUnit | str,
        *,
        usd: float | None = None,
        tool: str = "",
        phase: str = "",
        session_id: str = "default",
        label: str = "",
    ) -> CostEntry:
        """记一笔账. usd 显式传则直接用; 否则按换算率折算 (率未配则 0)."""
        dim = CostDimension(dimension)
        un = CostUnit(unit)
        if usd is None:
            usd = self._to_usd(amount, un)
        entry = CostEntry(
            dimension=dim,
            amount=amount,
            unit=un,
            usd=usd,
            tool=tool,
            phase=phase,
            session_id=session_id,
            label=label,
        )
        self._entries.append(entry)
        return entry

    def _to_usd(self, amount: float, unit: CostUnit) -> float:
        rate = {
            CostUnit.USD: 1.0,
            CostUnit.TOKENS: self.usd_per_1k_tokens / 1000.0,
            CostUnit.CPU_HOURS: self.usd_per_cpu_hour,
            CostUnit.GPU_HOURS: self.usd_per_gpu_hour,
            CostUnit.CALLS: self.usd_per_call,
        }[unit]
        return round(amount * rate, 6)

    # ── 查询 ─────────────────────────────────────────────────────
    def entries(self) -> list[CostEntry]:
        return list(self._entries)

    def total_usd(self) -> float:
        """统一货币累计 (所有已折美元条目)."""
        return round(sum(e.usd for e in self._entries), 6)

    def by_dimension(self) -> dict[str, float]:
        """花在哪: 按维度标签聚合 usd."""
        out: dict[str, float] = {}
        for e in self._entries:
            out[e.dimension.value] = round(out.get(e.dimension.value, 0.0) + e.usd, 6)
        return out

    def by_tool(self, session_id: str | None = None) -> dict[str, float]:
        """按工具聚合 usd (可限定 session)."""
        out: dict[str, float] = {}
        for e in self._entries:
            if session_id is not None and e.session_id != session_id:
                continue
            key = e.tool or "—"
            out[key] = round(out.get(key, 0.0) + e.usd, 6)
        return out

    def by_phase(self, session_id: str | None = None) -> dict[str, float]:
        """按阶段聚合 usd (可限定 session)."""
        out: dict[str, float] = {}
        for e in self._entries:
            if session_id is not None and e.session_id != session_id:
                continue
            key = e.phase or "—"
            out[key] = round(out.get(key, 0.0) + e.usd, 6)
        return out

    def session_total(self, session_id: str) -> float:
        return round(
            sum(e.usd for e in self._entries if e.session_id == session_id), 6
        )

    # ── 预算判定 ─────────────────────────────────────────────────
    def check_budget(
        self, budget_usd: float, *, warn_ratio: float = 0.8
    ) -> tuple[str, str]:
        """对预算做三档判定 (参考 BudgetDecision): ALLOW / WARN / DENY.

        返回 (decision, reason). decision ∈ {allow, warn, deny}.
        budget_usd <= 0 视为不限制 → 恒 allow.
        """
        if budget_usd <= 0:
            return "allow", f"不限制; 已花 self.total_usd() 美元"
        total = self.total_usd()
        reason = f"已花 ${total:.4f} / 预算 ${budget_usd:.2f}"
        if total >= budget_usd:
            return "deny", reason
        if total >= budget_usd * warn_ratio:
            return "warn", reason
        return "allow", reason

    def remaining(self, budget_usd: float) -> float:
        """相对预算的剩余额度 (不限制返回 inf)."""
        if budget_usd <= 0:
            return float("inf")
        return max(0.0, round(budget_usd - self.total_usd(), 6))

    def snapshot(self) -> dict[str, Any]:
        """可观测快照 (审计/前端用)."""
        return {
            "total_usd": self.total_usd(),
            "by_dimension": self.by_dimension(),
            "by_phase": self.by_phase(),
            "entry_count": len(self._entries),
            "rates": {
                "usd_per_1k_tokens": self.usd_per_1k_tokens,
                "usd_per_cpu_hour": self.usd_per_cpu_hour,
                "usd_per_gpu_hour": self.usd_per_gpu_hour,
                "usd_per_call": self.usd_per_call,
            },
        }


# 进程级单例 (一个 agent 进程一本账). 升级路径: 持久化到 state_store.
_ledger: CostLedger | None = None


def get_cost_ledger() -> CostLedger:
    """全局单例 (首次从 env 换算率构建)."""
    global _ledger
    if _ledger is None:
        _ledger = CostLedger.from_env()
    return _ledger


def reset_cost_ledger() -> None:
    """测试辅助: 重建单例 (清空账本)."""
    global _ledger
    _ledger = None


if __name__ == "__main__":
    # 自检
    l = CostLedger(usd_per_1k_tokens=2.0, usd_per_cpu_hour=0.5)
    l.record(CostDimension.LLM, 1000, CostUnit.TOKENS, tool="vasp_tool", phase="explore")
    l.record(CostDimension.COMPUTE, 2.0, CostUnit.CPU_HOURS, tool="vasp_tool", phase="explore")
    l.record(CostDimension.EXTERNAL, 0.3, CostUnit.USD, tool="web_search", phase="review")
    assert l.total_usd() == 2.0 + 1.0 + 0.3, l.snapshot()
    assert l.by_dimension()["llm"] == 2.0
    assert l.by_phase()["explore"] == 3.0
    assert l.check_budget(10.0)[0] == "allow"
    assert l.check_budget(2.0)[0] == "deny"
    assert l.check_budget(4.0)[0] == "warn"
    print("cost_ledger self-check passed")