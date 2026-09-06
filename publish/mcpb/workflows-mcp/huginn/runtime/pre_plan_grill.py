"""grill-me 风格的开工前需求澄清.

参考 Matt Pocock 的 grill-me skill (https://github.com/mattpocock/skills).
4 条核心约束纠正 4 个 LLM 默认偏置:

| 偏置             | grill-me 反向约束                                   |
|------------------|-----------------------------------------------------|
| 行动偏置         | 用户确认 shared understanding 前不得实施            |
| 顺从偏置         | 持续追问遍历决策树, 每题附推荐答案                 |
| 隐性填坑         | 一次只问一个问题, 等用户答完再问下一个              |
| 事实/决策混淆    | 事实 (代码库里能查到的) 自己查; 决策归用户          |

和 G71 paused_asking_decision 的关系: G71 是"出问题后停", grill 是"开工前
深挖". 两者共享 pause 机制 (should_pause_for_decision 条件 0) 但不冲突:
grill 优先级最高, 在其他触发条件之前检查.

升级路径: 当前是 prompt-level 约束, 非权限锁. 不可逆操作 (删数据/部署/付款)
仍由 sandbox / permissions / 7-layer defense 兜底, 不能只靠 grill.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# === LLM system prompt (grill 模式专用) ===
# Engine 在进入 grill 模式时把这段切到 LLM 的 system prompt.
# 中英对照: 短句好 token, 英文给海外模型, 中文给国产模型都看得懂.
GRILL_SYSTEM_PROMPT = """You are now in GRILL MODE before planning.

Goal: surface hidden assumptions BEFORE producing anything. The current
target is NOT to ship — it is to reach a shared understanding with the user.

Follow these four rules strictly:

1. ONE QUESTION AT A TIME. Ask, then STOP. Wait for the user's answer
   before asking the next one. Multiple questions in one message is
   forbidden — it hides unresolved decisions under a fluent plan.

2. EACH QUESTION MUST COME WITH YOUR RECOMMENDED ANSWER. The recommendation
   is a suggestion, not a decision. You are not the decider; the user is.
   The recommendation lowers the user's decision cost but does not authorize
   you to proceed.

3. IF A FACT CAN BE FOUND BY EXPLORING THE CODEBASE OR DOCS, LOOK IT UP
   YOURSELF. Do not ask the user things you can verify by reading files,
   checking configs, or grepping. Only genuine decisions belong to the user
   (risk tolerance, business priorities, irreversibility acceptance).

4. DO NOT ENACT THE PLAN until the user explicitly confirms shared
   understanding. No code writes, no file mutations, no tool calls that
   change external state. Read-only exploration is fine.

When you believe the design tree has been walked through end-to-end, stop
asking and produce a CONFIRMED DECISIONS bullet list, then ask:
"Have we reached shared understanding? If yes, I'll proceed to plan."

Exit grill mode only after the user replies affirmatively.
"""


GRILL_SYSTEM_PROMPT_CN = """你现在是开工前的 GRILL 模式.

目标: 在产出任何东西之前, 把隐藏假设摊开. 当前目标不是交付, 是和用户达成
shared understanding.

严格遵守以下四条:

1. 一次只问一个问题. 问完就停, 等用户答完再问下一个. 一条消息里抛多个问题
   等于把未决决策藏在一份流畅计划里, 这是隐性填坑.

2. 每个问题必须附上你的推荐答案. 推荐是建议, 不是决策. 你不是决策者, 用户
   才是. 推荐降低用户的决策成本, 但不等于授权你开工.

3. 能在代码库 / 文档 / 配置里查到的事实, 自己去查, 不要问用户. 只有真正的
   决策才归用户 (风险偏好 / 业务优先级 / 不可逆操作是否接受).

4. 用户明确确认 shared understanding 前, 不得实施计划. 不写代码, 不改
   文件, 不调用任何会改外部状态的工具. 只读探索可以.

当你认为设计树的每个分支都走过了, 停止提问, 输出一份 CONFIRMED DECISIONS
要点列表, 然后问: "我们是否达成了 shared understanding? 是的话我开始 plan."

只有用户回答 yes 才退出 grill 模式.
"""


# === Grill 会话状态 ===


@dataclass
class GrillSession:
    """一次 grill 会话的最小状态.

    ponytail: 只持有 LLM-driven 流程真正需要的状态. Engine 持有
    GrillSession 实例 (挂到 thread-local 或 lifecycle), LLM 自己负责
    "一次一题" 和 "何时退出". Engine 只用 questions_asked 和
    completed 做进度日志和退出检查.
    """
    started_at: float = 0.0
    questions_asked: int = 0
    confirmed_decisions: list[tuple[str, str]] = field(default_factory=list)
    # (question_summary, user_answer_summary)
    completed: bool = False  # 用户确认 shared understanding

    def next_question(self, question_summary: str = "") -> None:
        """记录开始问下一个问题. 实际问题文本由 LLM 生成."""
        self.questions_asked += 1
        if question_summary:
            self.confirmed_decisions.append((question_summary, ""))

    def record_answer(self, answer_summary: str) -> None:
        """用户回答完当前问题. answer_summary 由 Engine 抽."""
        if not self.confirmed_decisions:
            return
        q, _ = self.confirmed_decisions[-1]
        self.confirmed_decisions[-1] = (q, answer_summary)

    def confirm_shared_understanding(self) -> None:
        """用户明确确认, 退出 grill."""
        self.completed = True
        logger.info(
            "Grill 完成: %d 个决策确认", len(self.confirmed_decisions)
        )

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "questions_asked": self.questions_asked,
            "confirmed_decisions": list(self.confirmed_decisions),
            "completed": self.completed,
        }


# === Grill 启动判定 ===
#
# 触发条件参考 grill-me 原作者建议 (juejin 2026-07-13 文):
#   意图越模糊 / 决策分支越多 / 做错后返工成本越高 → grill 越有价值.
# 改一行配置 / 改一个明确函数 → 不必 grill.
#
# ponytail: 阈值简单硬编码, 不上 LLM 评估. 升级路径: LLM 给 ambiguity_score.


# 歧义分数阈值: >= 0.6 就该 grill
_AMBIGUITY_THRESHOLD = 0.6
# 触发 grill 的复杂度等级 (A=最高, D=最低, 参考 plan_check 的 tier 分级)
_GRILL_TIERS = frozenset({"A", "B"})
# scene_tag 中含这些关键词时认为歧义
_AMBIGUOUS_SCENE_KEYWORDS = (
    "ambiguous", "unclear", "vague", "multi_branch",
    "歧义", "模糊", "多分支", "未明确",
)


def should_start_grill(
    *,
    has_grilled: bool = False,
    ambiguity_score: float = 0.0,
    tier: str = "",
    scene_tag: str = "",
    plan_is_empty: bool = False,
) -> tuple[bool, str]:
    """判断该不该启动 grill 模式.

    触发条件 (任一):
    1. has_grilled=False 且 ambiguity_score >= 0.6
       → 显式歧义分够高, 直接 grill
    2. has_grilled=False 且 tier in {A, B} 且 scene_tag 命中歧义关键词
       → 复杂度高 + 场景歧义, 双触发
    3. has_grilled=False 且 plan_is_empty
       → 没计划就开始, grill 摊开假设

    Args:
        has_grilled: 本任务是否已经 grill 过 (避免循环触发)
        ambiguity_score: 0.0-1.0, LLM 或规则给的歧义度
        tier: plan_check 给的复杂度等级 ("A".."D")
        scene_tag: plan_check 给的场景标签 (任意字符串)
        plan_is_empty: 当前 plan 是否为空 / None

    Returns:
        (should_start, reason). 不触发返回 (False, "").
    """
    if has_grilled:
        return (False, "")

    if ambiguity_score >= _AMBIGUITY_THRESHOLD:
        return (
            True,
            f"歧义分 {ambiguity_score:.2f} >= {_AMBIGUITY_THRESHOLD}",
        )

    if tier in _GRILL_TIERS:
        scene_lower = (scene_tag or "").lower()
        if any(kw in scene_lower for kw in _AMBIGUOUS_SCENE_KEYWORDS):
            return (
                True,
                f"tier={tier} + scene='{scene_tag}' 双触发",
            )

    if plan_is_empty:
        return (True, "plan 为空, grill 摊开假设")

    return (False, "")


def grill_pause_options() -> list[dict]:
    """grill pause 时给用户的选项.

    复用 should_pause_for_decision 的 option dict schema
    ({"id", "label", "pros", "cons"}).
    """
    return [
        {
            "id": "A",
            "label": "进入 grill 模式, 一次一问",
            "pros": "把隐藏假设摊开, 避免带病开工",
            "cons": "需要几轮问答, 不能立即产出",
        },
        {
            "id": "B",
            "label": "跳过 grill 直接 plan",
            "pros": "快, 适合明确任务",
            "cons": "Agent 可能用默认值补齐规格缺口",
        },
        {
            "id": "C",
            "label": "取消任务",
            "pros": "用户改变主意",
            "cons": "已收集的上下文作废",
        },
    ]


# === 自检 (ponytail: 非平凡逻辑留一个 runnable check) ===

if __name__ == "__main__":
    import time

    # 1. should_start_grill — 条件 1: 高歧义分触发
    ok, reason = should_start_grill(ambiguity_score=0.8)
    assert ok, f"ambiguity=0.8 → grill, got {ok}/{reason}"
    assert "0.80" in reason, f"reason 应含分数: {reason}"

    # 1b. 边界: 刚好 0.6 触发
    ok, _ = should_start_grill(ambiguity_score=0.6)
    assert ok, "ambiguity=0.6 阈值边界应触发"

    # 1c. 边界: 0.59 不触发
    ok, _ = should_start_grill(ambiguity_score=0.59)
    assert not ok, "ambiguity=0.59 不到阈值不应触发"

    # 2. 条件 2: tier A/B + 歧义 scene
    ok, reason = should_start_grill(tier="A", scene_tag="ambiguous_requirement")
    assert ok, f"tier=A + ambiguous scene → grill, got {ok}/{reason}"
    assert "tier=A" in reason

    # 2b. tier A 但 scene 无歧义关键词 → 不触发
    ok, _ = should_start_grill(tier="A", scene_tag="clear_task")
    assert not ok, "tier=A 但 scene 无歧义词 → 不应触发"

    # 2c. tier C (低复杂度) + 歧义 scene → 不触发
    ok, _ = should_start_grill(tier="C", scene_tag="ambiguous")
    assert not ok, "tier=C 即使歧义也不该 grill"

    # 3. 条件 3: plan 为空
    ok, reason = should_start_grill(plan_is_empty=True)
    assert ok, f"plan 空 → grill, got {ok}/{reason}"
    assert "plan 为空" in reason

    # 4. has_grilled=True 时不重复触发
    ok, _ = should_start_grill(
        ambiguity_score=0.9, has_grilled=True)
    assert not ok, "已 grill 过不应重复触发"

    # 5. 完全没触发条件 → False
    ok, _ = should_start_grill(
        ambiguity_score=0.1, tier="C", scene_tag="clear",
        plan_is_empty=False)
    assert not ok, "无触发条件 → False"

    # 6. GrillSession 状态流转
    gs = GrillSession(started_at=time.time())
    assert not gs.completed
    assert gs.questions_asked == 0
    assert len(gs.confirmed_decisions) == 0

    gs.next_question("缓存粒度?")
    assert gs.questions_asked == 1
    assert gs.confirmed_decisions[0] == ("缓存粒度?", "")

    gs.record_answer("函数级别")
    assert gs.confirmed_decisions[0] == ("缓存粒度?", "函数级别")

    gs.next_question("冲突策略?")
    gs.record_answer("服务端版本为准")
    assert gs.questions_asked == 2
    assert len(gs.confirmed_decisions) == 2
    assert gs.confirmed_decisions[1] == ("冲突策略?", "服务端版本为准")

    # to_dict 往返
    d = gs.to_dict()
    assert d["questions_asked"] == 2
    assert d["completed"] is False
    assert len(d["confirmed_decisions"]) == 2

    gs.confirm_shared_understanding()
    assert gs.completed
    d = gs.to_dict()
    assert d["completed"] is True

    # 7. grill_pause_options schema
    opts = grill_pause_options()
    assert len(opts) == 3
    for opt in opts:
        assert set(opt.keys()) == {"id", "label", "pros", "cons"}
    assert opts[0]["id"] == "A"
    assert opts[1]["id"] == "B"

    # 8. prompts 不空, 含 4 条核心规则 (4 条规则的英文短语可能在行内换行)
    assert "ONE QUESTION AT A TIME" in GRILL_SYSTEM_PROMPT
    assert "RECOMMENDED ANSWER" in GRILL_SYSTEM_PROMPT
    # 第 3 条 "look it up yourself" 在 prompt 里跨行, 去空白再匹配
    prompt_flat = " ".join(GRILL_SYSTEM_PROMPT.split())
    assert "LOOK IT UP YOURSELF" in prompt_flat.upper() or \
           "look it up yourself" in prompt_flat.lower()
    assert "shared understanding" in GRILL_SYSTEM_PROMPT.lower()

    assert "一次只问一个" in GRILL_SYSTEM_PROMPT_CN
    assert "推荐答案" in GRILL_SYSTEM_PROMPT_CN
    assert "查到的事实" in GRILL_SYSTEM_PROMPT_CN
    assert "shared understanding" in GRILL_SYSTEM_PROMPT_CN

    print("pre_plan_grill selfcheck All passed")
