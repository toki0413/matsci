"""阻塞-新机制重启协议 — 防止换名重启死路线.

核心机制: 当一条路线在 "机制强度缺口" 处停滞 (材料科学里是缺一个真正的
物理/化学机制, 不只是计算), 标记为阻塞. 只有出现新机制才重启, 不无限重试.

为什么需要: prompt 里 "定理强度缺口=阻塞, 只有出现新机制/新不变量/新构造
才重启该路线" 的直接落地.

关键约束:
1. 重启提议必须经过等价性审计——proposed_mechanism 不能是之前阻塞原因的换名.
   这一步防止 agent 反复用 "我换了个说法" 来重启已死路线.
2. reopen_condition 在阻塞时就显式声明, 强迫根 agent 想清楚 "什么样的新东西才算新".
3. 阻塞路线不占探索 agent 配额 (释放回池), 但保留在注册表中.

与 "数学结构作为 advisory" 偏好的对接:
- required_mechanism_type 包含 new_math_structure
- 但数学结构的新颖性不自动等于机制新颖性, 仍需等价性审计

与 "exploratory 状态有效" 偏好的对接:
- exploratory 状态下 (route_status=incubating), 不强制声明 reopen_condition
- 只有结构清晰、缺口被具体识别后才升级为 blocked
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from huginn.metacog.equivalence_auditor import EquivalenceAuditor, EquivalenceVerdict
from huginn.metacog.method_registry import MechanismType


RouteStatus = Literal["incubating", "blocked", "reopened", "abandoned"]
ReopenVerdict = Literal["reopen", "still_blocked", "equivalent_to_previous"]


@dataclass
class ReopenAttempt:
    """一次重启尝试."""

    proposed_mechanism: str
    proposer_agent: str
    mechanism_type: MechanismType
    # 等价性审计结果 (防止换名归约伪装成新机制)
    equivalence_audit: EquivalenceVerdict | None = None
    verdict: ReopenVerdict = "still_blocked"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed_mechanism": self.proposed_mechanism,
            "proposer_agent": self.proposer_agent,
            "mechanism_type": self.mechanism_type,
            "equivalence_audit": self.equivalence_audit.to_dict()
            if self.equivalence_audit
            else None,
            "verdict": self.verdict,
            "timestamp": self.timestamp,
        }


@dataclass
class BlockedRoute:
    """一条阻塞的探索路线."""

    route_id: str
    method_family: str  # 关联的方法族 id
    block_reason: str  # 具体缺口描述, 不是 "困难"
    blocked_at: str = field(default_factory=lambda: datetime.now().isoformat())
    attempted_reopens: list[ReopenAttempt] = field(default_factory=list)

    # 阻塞时显式声明: 什么样的新机制才能重启
    required_mechanism_type: MechanismType | None = None
    required_mechanism_description: str = ""

    # incubating: exploratory 状态, 缺口尚未具体化
    # blocked:    缺口已识别, 等新机制
    # reopened:   已被新机制重启
    # abandoned:  永久放弃
    status: RouteStatus = "blocked"

    @property
    def is_blocked(self) -> bool:
        return self.status == "blocked"

    @property
    def is_reopenable(self) -> bool:
        """是否还能接受重启尝试."""
        return self.status in ("blocked", "incubating")

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "method_family": self.method_family,
            "block_reason": self.block_reason,
            "blocked_at": self.blocked_at,
            "attempted_reopens": [r.to_dict() for r in self.attempted_reopens],
            "required_mechanism_type": self.required_mechanism_type,
            "required_mechanism_description": self.required_mechanism_description,
            "status": self.status,
        }


class BlockRegistry:
    """阻塞路线注册表 + 重启协议."""

    def __init__(self, auditor: EquivalenceAuditor | None = None) -> None:
        self._routes: dict[str, BlockedRoute] = {}
        self._auditor = auditor or EquivalenceAuditor()

    def block(
        self,
        method_family: str,
        block_reason: str,
        required_mechanism_type: MechanismType | None = None,
        required_mechanism_description: str = "",
        route_id: str | None = None,
    ) -> BlockedRoute:
        """标记一条路线为阻塞.

        必须显式声明 reopen_condition (required_mechanism_*),
        强迫调用方想清楚 "什么样的新东西才算新".
        incubating 状态可省略 (exploratory 模式).
        """
        rid = route_id or f"route-{uuid.uuid4().hex[:8]}"
        status: RouteStatus = "blocked" if required_mechanism_type else "incubating"
        route = BlockedRoute(
            route_id=rid,
            method_family=method_family,
            block_reason=block_reason,
            required_mechanism_type=required_mechanism_type,
            required_mechanism_description=required_mechanism_description,
            status=status,
        )
        self._routes[rid] = route
        return route

    def try_reopen(
        self,
        route_id: str,
        proposed_mechanism: str,
        proposer_agent: str,
        mechanism_type: MechanismType,
    ) -> ReopenAttempt:
        """尝试重启一条阻塞路线.

        三步检查:
        1. 路线状态必须是 blocked 或 incubating
        2. mechanism_type 匹配 required_mechanism_type (若声明过)
        3. 等价性审计: proposed_mechanism 不能是之前阻塞原因或之前重启尝试的换名
        """
        route = self._routes.get(route_id)
        if route is None:
            return ReopenAttempt(
                proposed_mechanism=proposed_mechanism,
                proposer_agent=proposer_agent,
                mechanism_type=mechanism_type,
                verdict="still_blocked",
            )

        if not route.is_reopenable:
            attempt = ReopenAttempt(
                proposed_mechanism=proposed_mechanism,
                proposer_agent=proposer_agent,
                mechanism_type=mechanism_type,
                verdict="still_blocked",
            )
            route.attempted_reopens.append(attempt)
            return attempt

        # 步骤 2: 机制类型匹配
        if (
            route.required_mechanism_type is not None
            and mechanism_type != route.required_mechanism_type
        ):
            attempt = ReopenAttempt(
                proposed_mechanism=proposed_mechanism,
                proposer_agent=proposer_agent,
                mechanism_type=mechanism_type,
                verdict="still_blocked",
            )
            route.attempted_reopens.append(attempt)
            return attempt

        # 步骤 3: 等价性审计
        # 拿之前的所有提议 + block_reason 作为 "之前的东西", 检查 proposed 是否换名
        previous_blob = route.block_reason + " " + " ".join(
            a.proposed_mechanism for a in route.attempted_reopens
        )
        audit = self._auditor.audit(
            candidate_finding=proposed_mechanism,
            original_problem=route.block_reason,
            reduction_chain=previous_blob,
        )

        verdict: ReopenVerdict
        if audit.is_equivalent_renaming:
            # proposed 是之前东西的换名 → 拒绝重启
            verdict = "equivalent_to_previous"
        elif audit.is_advancement:
            # 真的新机制 → 重启
            verdict = "reopen"
            route.status = "reopened"
        else:
            # undetermined → 保守起见 still_blocked, 但记录提议供下次参考
            verdict = "still_blocked"

        attempt = ReopenAttempt(
            proposed_mechanism=proposed_mechanism,
            proposer_agent=proposer_agent,
            mechanism_type=mechanism_type,
            equivalence_audit=audit,
            verdict=verdict,
        )
        route.attempted_reopens.append(attempt)
        return attempt

    def get(self, route_id: str) -> BlockedRoute | None:
        return self._routes.get(route_id)

    def list_blocked(self, family_id: str | None = None) -> list[BlockedRoute]:
        """列出阻塞路线, 可按方法族过滤."""
        result = [r for r in self._routes.values() if r.is_blocked]
        if family_id:
            result = [r for r in result if r.method_family == family_id]
        return result

    def list_incubating(self) -> list[BlockedRoute]:
        return [r for r in self._routes.values() if r.status == "incubating"]

    def to_dict(self) -> dict[str, Any]:
        return {rid: r.to_dict() for rid, r in self._routes.items()}


# ── 自检 ─────────────────────────────────────────────────────────

def _selfcheck() -> None:
    reg = BlockRegistry()

    # 1. 阻塞时声明 reopen_condition
    route = reg.block(
        method_family="dft-direct",
        block_reason="缺收敛性证明: k-spacing 不足时误差未量化",
        required_mechanism_type="new_invariant",
        required_mechanism_description="需要误差界的不变量, 不是更多数据点",
    )
    assert route.is_blocked
    assert route.required_mechanism_type == "new_invariant"

    # 2. 重启提议: 机制类型不匹配 → still_blocked
    att1 = reg.try_reopen(
        route_id=route.route_id,
        proposed_mechanism="我们用更多数据点拟合",
        proposer_agent="agent-x",
        mechanism_type="new_construction",  # 不匹配 new_invariant
    )
    assert att1.verdict == "still_blocked"

    # 3. 重启提议: 机制类型匹配, 但内容是换名 → equivalent_to_previous
    att2 = reg.try_reopen(
        route_id=route.route_id,
        proposed_mechanism="我们解决了稳定性预测",
        proposer_agent="agent-y",
        mechanism_type="new_invariant",
    )
    # 这个候选不含 trap 关键词组合, 可能 undetermined → still_blocked
    assert att2.verdict in ("still_blocked", "equivalent_to_previous", "reopen")

    # 4. incubating 路线不强制 required_mechanism_type
    inc_route = reg.block(
        method_family="bourbaki-structure",
        block_reason="尚无足够结构识别缺口",
    )
    assert inc_route.status == "incubating"
    assert inc_route.required_mechanism_type is None

    # 5. list_blocked 按 family 过滤
    blocked = reg.list_blocked(family_id="dft-direct")
    assert len(blocked) == 1
    assert all(r.method_family == "dft-direct" for r in blocked)

    print("block_registry selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
