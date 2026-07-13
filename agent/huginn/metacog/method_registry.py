"""方法族注册表 — 按思想本质聚类探索 agent, 监控收敛度.

核心机制: 不按 agent 措辞分组, 按其使用的数学/物理思想分组. 当某族占
总 agent 数超阈值时, 强制重定向新 agent 到探索不足的族.

为什么需要: prompt 里 "不要让一种方法仅仅因为它给出优雅的归约就占据主导"
的直接落地. 不监控的话, 所有 agent 会向同一个 attractor 收敛.

初始族清单针对材料科学:
- dft-direct, ml-potential, symbolic-regression, gaussian-process,
  calphad-thermo, phase-field,
  bourbaki-structure (advisory), extreme-argument, computational-check
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


MechanismType = Literal[
    "new_invariant",       # 新的不变量 (守恒律 / 对称性)
    "new_construction",    # 新的构造 (算法 / 数据结构)
    "new_reduction",       # 新的归约 (问题等价转换)
    "new_data_source",     # 新的数据源 (实验 / 数据库)
    "new_math_structure",  # 新的数学结构 (Bourbaki 视角)
]


@dataclass
class MethodFamily:
    """一个方法族 = 一类思想本质相同的探索路线."""

    id: str
    essence: str  # 一句话描述思想本质, 不是表面措辞
    member_agent_ids: list[str] = field(default_factory=list)
    last_block_reason: str | None = None
    # 该族是否豁免第一性原理淘汰 (经验方法可标记, 如迁移学习)
    exempt_from_fp_check: bool = False

    @property
    def is_blocked(self) -> bool:
        return self.last_block_reason is not None

    def member_count(self, total_agents: int) -> float:
        """该族占总 agent 数的比例 (convergence pressure)."""
        if total_agents <= 0:
            return 0.0
        return len(self.member_agent_ids) / total_agents


# 初始族清单. 应随研究领域扩充.
_DEFAULT_FAMILIES: list[MethodFamily] = [
    MethodFamily(id="dft-direct", essence="从第一性原理直接计算目标性质"),
    MethodFamily(id="ml-potential", essence="ML 势函数 + MD 模拟"),
    MethodFamily(id="symbolic-regression", essence="符号回归找解析式"),
    MethodFamily(id="gaussian-process", essence="GP + 不确定性量化"),
    MethodFamily(id="calphad-thermo", essence="CALPHAD 热力学相图"),
    MethodFamily(id="phase-field", essence="相场模拟演化"),
    MethodFamily(id="bourbaki-structure", essence="数学结构映射 (advisory)"),
    MethodFamily(id="extreme-argument", essence="极值/反例论证"),
    MethodFamily(id="computational-check", essence="计算合理性检查 (独立于其他族)"),
]


@dataclass
class _RedirectSuggestion:
    """重定向建议: 把新 agent 分到哪个族."""

    target_family: str
    reason: str


class MethodRegistry:
    """方法族注册表 + 收敛度监控."""

    # 某族占比超此阈值 → 强制重定向新 agent 到冷门族
    CONVERGENCE_PRESSURE_LIMIT = 0.4
    # 冷门族定义: 占比低于此值
    COLD_FAMILY_THRESHOLD = 0.1

    def __init__(self, families: list[MethodFamily] | None = None) -> None:
        self._families: dict[str, MethodFamily] = {
            f.id: f for f in (families or _DEFAULT_FAMILIES)
        }

    def all(self) -> list[MethodFamily]:
        return list(self._families.values())

    def by_id(self, family_id: str) -> MethodFamily | None:
        return self._families.get(family_id)

    def register_agent(self, family_id: str, agent_id: str) -> None:
        """把 agent 登记到某族. 重复登记会先从原族移除."""
        for f in self._families.values():
            if agent_id in f.member_agent_ids:
                f.member_agent_ids.remove(agent_id)
        fam = self._families.get(family_id)
        if fam is None:
            raise KeyError(f"unknown method family: {family_id}")
        fam.member_agent_ids.append(agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        for f in self._families.values():
            if agent_id in f.member_agent_ids:
                f.member_agent_ids.remove(agent_id)

    def total_agents(self) -> int:
        return sum(len(f.member_agent_ids) for f in self._families.values())

    def pressure(self, family_id: str) -> float:
        """某族的 convergence pressure = member_count / total."""
        fam = self._families.get(family_id)
        if fam is None:
            return 0.0
        return fam.member_count(self.total_agents())

    def suggest_redirect(self) -> _RedirectSuggestion | None:
        """建议新 agent 该分到哪个族.

        优先级: 已阻塞族跳过; 否则选当前最冷门的非阻塞族.
        若最热族 pressure 超阈值且存在冷门族, 返回冷门族建议;
        否则返回 None (让调用方自由分配).
        """
        active = [f for f in self._families.values() if not f.is_blocked]
        if not active:
            return None

        total = self.total_agents()
        # 没有 agent 时, 默认分到 dft-direct
        if total == 0:
            return _RedirectSuggestion(
                target_family="dft-direct",
                reason="no agents yet, default to first-principles baseline",
            )

        # 找最热和最冷
        hottest = max(active, key=lambda f: f.member_count(total))
        coldest = min(active, key=lambda f: f.member_count(total))

        if (
            hottest.member_count(total) > self.CONVERGENCE_PRESSURE_LIMIT
            and coldest.member_count(total) < self.COLD_FAMILY_THRESHOLD
            and hottest.id != coldest.id
        ):
            return _RedirectSuggestion(
                target_family=coldest.id,
                reason=(
                    f"{hottest.id} 占比 {hottest.member_count(total):.0%} 超阈值, "
                    f"重定向到冷门族 {coldest.id}"
                ),
            )
        return None

    def mark_blocked(self, family_id: str, reason: str) -> None:
        fam = self._families.get(family_id)
        if fam is None:
            raise KeyError(f"unknown method family: {family_id}")
        fam.last_block_reason = reason

    def mark_unblocked(self, family_id: str) -> None:
        fam = self._families.get(family_id)
        if fam is None:
            return
        fam.last_block_reason = None

    def to_dict(self) -> dict[str, Any]:
        return {
            fid: {
                "essence": f.essence,
                "member_count": len(f.member_agent_ids),
                "pressure": round(f.member_count(max(self.total_agents(), 1)), 3),
                "blocked": f.is_blocked,
                "block_reason": f.last_block_reason,
                "exempt_from_fp_check": f.exempt_from_fp_check,
            }
            for fid, f in self._families.items()
        }


# ponytail: to_dict 用 Any 是为了不引入额外类型, 保持模块自洽
from typing import Any  # noqa: E402


# ── 自检 ─────────────────────────────────────────────────────────

def _selfcheck() -> None:
    reg = MethodRegistry()

    # 1. 初始状态无 agent, suggest_redirect 给默认族
    sug = reg.suggest_redirect()
    assert sug is not None and sug.target_family == "dft-direct"

    # 2. 4 个 agent 都分到 dft-direct (5 个里占 80%) → 应重定向到冷门族
    for i in range(4):
        reg.register_agent("dft-direct", f"agent-{i}")
    sug2 = reg.suggest_redirect()
    assert sug2 is not None, "过热应触发重定向"
    assert sug2.target_family != "dft-direct", "应重定向到冷门族, 不是 dft-direct"
    assert reg.pressure("dft-direct") > 0.4

    # 3. 阻塞族不参与重定向建议
    reg.mark_blocked("calphad-thermo", "test block")
    blocked_fam = reg.by_id("calphad-thermo")
    assert blocked_fam is not None and blocked_fam.is_blocked

    # 4. exempt_from_fp_check 可手动标记
    from huginn.metacog.method_registry import MethodFamily as _MF
    custom_reg = MethodRegistry([_MF(id="empirical-fit", essence="经验拟合", exempt_from_fp_check=True)])
    assert custom_reg.by_id("empirical-fit").exempt_from_fp_check

    # 5. unregister 后 pressure 下降
    reg.unregister_agent("agent-0")
    assert len(reg.by_id("dft-direct").member_agent_ids) == 3

    print("method_registry selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
