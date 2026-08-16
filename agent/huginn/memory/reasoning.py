"""Structured reasoning — the deepened external-thinking core.

external thinking 原实现 (deep_think_tool + session.reasoning_trace) 是**被动捕获**:
模型写自由文本分析 → 扁平字符串列表。本模块把它升级为**结构化推理协议**:

- 每条推理是一条 ``ReasoningRecord``: 带阶段 (phase)、核心论断 (claim)、依据
  (evidence)、量化预估 (estimate)、不确定性 (uncertainty)、下一步 (plan)。
- 阶段化编排 (think → plan → pre_action → reflect) 让推理按研究阶段可追踪。
- 执行后回填 ``outcome`` (confirmed / refuted / partial) — 自校验闭环的地基。
- 高分记录 (confirmed + 有 estimate) 可被蒸馏成可复用知识 (reasoning distillation)。

设计原则 (Everything is a Plugin 同理): 结构化通道是**新增**侧信道, 不破坏
session.reasoning_trace 的扁平字符串通道 (原生 reasoning_content + 旧 deep_think
仍写那里), 下游蒸馏/反思消费方可平滑迁移。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class ReasoningPhase(StrEnum):
    """推理阶段 — think→plan→pre_action→reflect 编排.

    Attributes:
        THINK: 假设 / 论断 — 提出核心主张.
        PLAN: 方案 — 达成主张的行动序列.
        PRE_ACTION: 动手前预估 — 量化预测, 供执行后对照.
        REFLECT: 回映 — 执行后复盘主张对错.
    """

    THINK = "think"
    PLAN = "plan"
    PRE_ACTION = "pre_action"
    REFLECT = "reflect"


class ReasoningOutcome(StrEnum):
    """回映结果 (自校验闭环填充)."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    PARTIAL = "partial"


@dataclass
class ReasoningRecord:
    """一条结构化推理记录.

    字段刻意从宽 (str, 默认空) — 模型自由文本填, 不做强 schema 校验; 结构化
    只为让下游可蒸馏/校验/追溯, 不为约束模型写法而设门槛.
    """

    claim: str = ""  # 核心论断 / 假设
    phase: ReasoningPhase = ReasoningPhase.THINK
    evidence: str = ""  # 依据
    estimate: str = ""  # 量化预估 (字符串, 因含单位/范围)
    uncertainty: str = ""  # 不确定性与边界条件
    plan: str = ""  # 下一步动作
    outcome: ReasoningOutcome = ReasoningOutcome.PENDING
    verified_by: str = ""  # 回映来源 (tool_name / reflect stage)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def is_distillable(self) -> bool:
        """确认且带量化预估的记录可蒸馏成可复用知识."""
        return (
            self.outcome is ReasoningOutcome.CONFIRMED
            and bool(self.claim)
            and bool(self.estimate)
        )

    @property
    def is_pending(self) -> bool:
        return self.outcome is ReasoningOutcome.PENDING


class ReasoningTrace:
    """结构化推理侧信道: append / recent / pending / mark_outcome.

    与 session.reasoning_trace (扁平字符串) 并存; 封装 cap 截断, 避免调用方
    各自维护长度.
    """

    def __init__(self, max_records: int = 200) -> None:
        self._records: list[ReasoningRecord] = []
        self.max_records = max_records

    def append(self, record: ReasoningRecord) -> None:
        self._records.append(record)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]

    def recent(self, n: int = 10) -> list[ReasoningRecord]:
        return self._records[-n:]

    def last_pending(self) -> ReasoningRecord | None:
        """返回最近一条未回映的记录 (自校验从它开始回填)."""
        for r in reversed(self._records):
            if r.is_pending and r.phase in (
                ReasoningPhase.PRE_ACTION,
                ReasoningPhase.PLAN,
            ):
                return r
        return None

    def mark_outcome(
        self,
        record: ReasoningRecord,
        outcome: ReasoningOutcome,
        verified_by: str = "",
    ) -> None:
        record.outcome = outcome
        record.verified_by = verified_by

    def distillable(self) -> list[ReasoningRecord]:
        return [r for r in self._records if r.is_distillable]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)
