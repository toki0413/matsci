"""P2 转用门 — 把『同源异流』先验固化为跨工具复用开关.

P2 (同源异流): 噪声动力学与相变动力学共享同一数学母结构
(SDE → 路径积分/MSR → 活化律), 但相变是该母结构在空间扩展、集体关联与对称性
破缺下的特殊实现. 跨现象**借模型必须过三条门**, 依据如下"转用规则":

  - ``is_subcase_of`` (⊂ 包含)  : dst 是 src 的子情形 → 可**特化** src 的求解器,
    但不整段搬运 src 的校准参数.
  - ``isomorphic_to``  (≅ MSR)  : 细致平衡下随机动力学↔平衡统计场论只在算法骨架
    层面等价 → 可**只借算法**(路径积分求值 / 涨落-耗散 / 响应), 不借参数.
  - ``independent_of`` (≢ 不等价) : 普适类不同 (相变有标度律/普适类, Kramers 逃逸
    无) → **失效安全门**: 禁止跨普适类迁移定量参数与求解算法, 只留定性结构
    (守恒/对称已固化在 ``PhysicsValidator``).

本模块是 P2 的软件接缝: 把"跨现象能借什么"从对话约定变成机器可判定、可审计的门.
"""

from __future__ import annotations

from dataclasses import dataclass

# 三种结构关系 (转用规则的可枚举值).
SUBTYPE = "is_subcase_of"       # ⊂  包含 — 相变 ⊂ 噪声动力学
ISOMORPHIC = "isomorphic_to"    # ≅  MSR — 细致平衡下结构同构
INDEPENDENT = "independent_of"  # ≢  物理不等价 — 普适类不同


@dataclass(frozen=True)
class TransferRelation:
    """一个工具相对另一个工具的 P2 结构关系 (带方向: ``with_tool`` 是被指向方)."""

    kind: str
    with_tool: str
    note: str = ""


@dataclass(frozen=True)
class TransferVerdict:
    """一次"借用"判定的结果 — 是否允许复用目标方的算法与参数."""

    allow_machinery: bool   # 是否可复用求解算法/骨架 (⊂ 特化 / ≅ 借骨架 → 真)
    allow_parameters: bool  # 是否可复用拟合/标定参数 (一律需显式, 默认假)
    reason: str


def resolve_transfer(relation: TransferRelation | None) -> TransferVerdict:
    """按三条转用规则判定一次借用; ``None`` (未声明关系) 按隔离失效安全处理."""
    kind = relation.kind if relation is not None else None
    target = relation.with_tool if relation is not None else "(isolated)"

    if kind == SUBTYPE:
        return TransferVerdict(
            allow_machinery=True,
            allow_parameters=False,
            reason=f"⊂ {target}: 子情形可特化父求解器; 参数须按本域重标定, 不整段搬父参数",
        )
    if kind == ISOMORPHIC:
        return TransferVerdict(
            allow_machinery=True,
            allow_parameters=False,
            reason=f"≅ {target}: 仅借算法骨架 (MSR); 参数因现象而异, 不转移",
        )
    # independent_of / unknown / None → 一律失效安全: 禁转.
    return TransferVerdict(
        allow_machinery=False,
        allow_parameters=False,
        reason=f"≢ {target}: 普适类不同/未声明关系, 禁止跨域转移 (失效安全)",
    )
