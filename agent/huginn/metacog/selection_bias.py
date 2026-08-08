"""选择偏差检测 — 对抗 agent 从有偏子样本推因果结论.

触发场景 (CausalGame 的 antenna_trap 系): 系统只在幸存者子集上暴露数据,
失败/被删样本对 agent 不可见. agent 若拿"幸存者设计 → 高存活"直接推因果,
结论就是幸存者偏差 (survivorship bias) 驱动的, 不是真实因果.

本模块给一个通用检测器: 喂一组观察记录 (每条可带 outcome + 可选分组),
启发式判断样本是否"系统性缺了一类". 命中时返回 flag + 可读 hint, 调用方
(validate 阶段 / RCB TaskComplete 审计) 把它注入下一轮 prompt, 让 agent
意识到自己在用有偏样本下结论.

ponytail: 纯启发式, 不做统计检验. ceiling: 对 outcome 做二项检验 + 分组
卡方检验, 需要每类期望频数, 本模块拿不到就先不接.

不依赖 engine / LLM / hypothesis_graph. 传进来啥就验啥, 出错返回不命中
(advisory, 不阻断).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SelectionBiasVerdict:
    """一次选择偏差检测的结果."""

    biased: bool = False
    bias_type: str = ""  # closed_outcome / single_group / unbalanced_group
    outcome_values: list[str] = field(default_factory=list)
    group_values: list[str] = field(default_factory=list)
    n: int = 0
    hint: str = ""


def _obs_fields(obs: Any) -> tuple[str | None, str | None]:
    """从观察记录里取 (outcome, group). 兼容 dataclass / dict."""
    if isinstance(obs, dict):
        return obs.get("outcome"), obs.get("group")
    return getattr(obs, "outcome", None), getattr(obs, "group", None)


def detect_selection_bias(
    observations: list,
    *,
    min_n: int = 3,
    single_group_min_n: int = 4,
) -> SelectionBiasVerdict:
    """检测观察样本是否系统性缺了一类.

    Args:
        observations: 观察记录列表, 每条可含 outcome / group 字段.
        min_n: 启动 outcome 检测的最小样本数 (太少不算偏差).
        single_group_min_n: 启动分组检测的最小样本数.

    检测三种模式 (按优先级):
    1. closed_outcome — 样本够多但 outcome 只有一个值 (全成功或全失败),
       疑似只看单一结局 -> 幸存者偏差.
    2. single_group — 带分组字段但只出现一个组, 疑似子群体被隐藏.
    3. unbalanced_group — 分组出现但某组占比极端 (>= 0.9), 子群体代表性不足.

    返回 SelectionBiasVerdict. 任一带解析失败都返回不命中.
    """
    if not observations:
        return SelectionBiasVerdict()

    outcomes: list[str | None] = []
    groups: list[str | None] = []
    for obs in observations:
        try:
            oc, gr = _obs_fields(obs)
        except Exception:
            return SelectionBiasVerdict()
        outcomes.append(oc)
        groups.append(gr)

    n = len(observations)

    # 模式 1: 有 outcome 字段, 样本够多但值域坍缩到单值
    present_outcomes = [o for o in outcomes if o is not None]
    if present_outcomes and n >= min_n:
        unique_outcomes = list(dict.fromkeys(str(o) for o in present_outcomes))
        if len(unique_outcomes) == 1:
            return SelectionBiasVerdict(
                biased=True,
                bias_type="closed_outcome",
                outcome_values=unique_outcomes,
                n=n,
                hint=(
                    f"[selection bias] 你有 {n} 条观测, 但 outcome 全是 "
                    f"'{unique_outcomes[0]}'. 如果失败/被删样本对你不可见, 拿这份 "
                    "单侧数据推因果结论是幸存者偏差. 先确认样本是否截断, 再下结论."
                ),
            )

    # 模式 2 & 3: 有 group 字段, 样本够多时看分组分布
    present_groups = [g for g in groups if g is not None]
    if present_groups and n >= single_group_min_n:
        unique_groups = list(dict.fromkeys(str(g) for g in present_groups))
        if len(unique_groups) == 1:
            return SelectionBiasVerdict(
                biased=True,
                bias_type="single_group",
                group_values=unique_groups,
                n=n,
                hint=(
                    f"[selection bias] {n} 条观测只来自一个子群体 "
                    f"'{unique_groups[0]}'. 其他子群体可能被系统性排除. "
                    "别把单群体的规律外推到全体."
                ),
            )
        # 组占比极端: 最大的组占了 >= 90%, 其他组代表性不足
        counts = {g: 0 for g in unique_groups}
        for g in unique_groups:
            counts[g] = sum(1 for x in present_groups if str(x) == g)
        top_group = max(counts, key=counts.get)
        top_ratio = counts[top_group] / n
        if top_ratio >= 0.9:
            return SelectionBiasVerdict(
                biased=True,
                bias_type="unbalanced_group",
                group_values=unique_groups,
                n=n,
                hint=(
                    f"[selection bias] 子群体'{top_group}'占了 {top_ratio:.0%} "
                    f"({counts[top_group]}/{n}) 的观测, 其他组代表性不足. "
                    "结论可能偏向优势子群体."
                ),
            )

    return SelectionBiasVerdict()


# ── 自检 ─────────────────────────────────────────────────────────

def _selfcheck() -> None:
    # 1) 空输入 -> 不命中
    assert not detect_selection_bias([]).biased

    # 2) 样本够但 outcome 单值 -> closed_outcome
    v = detect_selection_bias(
        [{"outcome": "survived"} for _ in range(5)], min_n=3
    )
    assert v.biased and v.bias_type == "closed_outcome"
    assert v.hint and "幸存者偏差" in v.hint

    # 3) 样本太少 -> 不命中
    assert not detect_selection_bias(
        [{"outcome": "survived"}, {"outcome": "survived"}], min_n=3
    ).biased

    # 4) 单分组 -> single_group (样本够, 且 outcome 不只单值避免先命中 closed_outcome)
    v = detect_selection_bias(
        [{"outcome": "a", "group": "zone_a"}] * 2 +
        [{"outcome": "b", "group": "zone_a"}] * 2,
        single_group_min_n=4,
    )
    assert v.biased and v.bias_type == "single_group"

    # 5) 分组不平衡 -> unbalanced_group (outcome 非单值避免先命中 closed_outcome)
    obs = [{"outcome": "a", "group": "zone_a"}] * 5 + \
        [{"outcome": "b", "group": "zone_a"}] * 4 + \
        [{"outcome": "a", "group": "zone_b"}]
    v = detect_selection_bias(obs, min_n=3, single_group_min_n=4)
    assert v.biased and v.bias_type == "unbalanced_group"

    # 6) 均衡分组 -> 不命中 (outcome 非单值避免先命中 closed_outcome)
    balanced = [{"outcome": "a", "group": "zone_a"}] * 3 + \
        [{"outcome": "b", "group": "zone_b"}] * 3 + \
        [{"outcome": "c", "group": "zone_c"}] * 3
    assert not detect_selection_bias(balanced).biased

    # 7) 无 outcome/group 字段 -> 不命中
    assert not detect_selection_bias([{"a": 1}, {"a": 2}, {"a": 3}]).biased

    # 8) dataclass 对象也兼容
    from dataclasses import dataclass as _dc

    @_dc
    class _Row:
        outcome: str
        group: str = "g"

    rows = [_Row(outcome="failed") for _ in range(4)]
    v = detect_selection_bias(rows, min_n=3)
    assert v.biased and v.bias_type == "closed_outcome"

    print("selection_bias selfcheck OK")


if __name__ == "__main__":
    _selfcheck()