"""用户 taste profile — 首次使用时的精简问卷, 提取思维模式和想象力偏好.

灵感来自「让陌生人迅速相爱的 36 个问题」: 用最少的题数提取最核心的 taste.
12 题, 3 个维度 (思维模式 / 想象力 / 研究风格), 每题 4 选项.
可跳过, 跳过时用 balanced/mixed 默认值.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ── 维度枚举 ──────────────────────────────────────────────────────


class ThinkingMode(str, Enum):
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    VISUAL = "visual"
    SYSTEMS = "systems"


class Imagination(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    SPECULATIVE = "speculative"
    VISIONARY = "visionary"


class ResearchStyle(str, Enum):
    THEORETICAL = "theoretical"
    EMPIRICAL = "empirical"
    ENGINEERING = "engineering"
    MIXED = "mixed"


# ── TasteProfile ──────────────────────────────────────────────────


@dataclass
class TasteProfile:
    """从问卷答案提取的用户 taste, 注入 system prompt 定制 agent 行为."""

    thinking_mode: ThinkingMode = ThinkingMode.SYSTEMS
    imagination: Imagination = Imagination.BALANCED
    research_style: ResearchStyle = ResearchStyle.MIXED
    completed: bool = False  # 是否完整回答了问卷
    skipped: bool = False  # 是否主动跳过

    def to_directive(self) -> str:
        """生成 system prompt 片段, 告诉 agent 这个用户的思维偏好."""
        if self.skipped:
            return ""

        parts: list[str] = []

        tm = self.thinking_mode
        if tm == ThinkingMode.ANALYTICAL:
            parts.append(
                "用户偏好严谨的逻辑分析, 回答时给出推导步骤和数据支撑, "
                "用精确的定义而非模糊的类比."
            )
        elif tm == ThinkingMode.INTUITIVE:
            parts.append(
                "用户偏好直觉式理解, 回答时先用类比和直觉解释, "
                "再补充细节. 避免一上来就堆公式."
            )
        elif tm == ThinkingMode.VISUAL:
            parts.append(
                "用户偏好可视化思维, 回答时尽量用图表/示意图描述, "
                "用空间关系而非纯文字表达结构."
            )
        elif tm == ThinkingMode.SYSTEMS:
            parts.append(
                "用户偏好系统性思考, 回答时先展示整体架构和模块关系, "
                "再深入细节. 强调连接和交互."
            )

        img = self.imagination
        if img == Imagination.CONSERVATIVE:
            parts.append(
                "用户偏好保守推断, 只给有数据支撑的结论, "
                "明确标注假设的不确定性."
            )
        elif img == Imagination.BALANCED:
            parts.append(
                "用户偏好平衡的推测, 既给保守结论也提合理外推, "
                "区分已验证和推测部分."
            )
        elif img == Imagination.SPECULATIVE:
            parts.append(
                "用户偏好大胆推测, 鼓励提出前瞻性假设和非常规思路, "
                "即使证据不完全也值得探索."
            )
        elif img == Imagination.VISIONARY:
            parts.append(
                "用户偏好颠覆性想象, 鼓励提出范式转换级的想法, "
                "不必受现有框架约束."
            )

        rs = self.research_style
        if rs == ResearchStyle.THEORETICAL:
            parts.append(
                "用户偏好理论驱动的研究, 从第一性原理出发推导, "
                "关注数学优美和物理本质."
            )
        elif rs == ResearchStyle.EMPIRICAL:
            parts.append(
                "用户偏好实验驱动的研究, 从数据出发归纳规律, "
                "关注可复现性和统计显著性."
            )
        elif rs == ResearchStyle.ENGINEERING:
            parts.append(
                "用户偏好工程导向的研究, 从应用需求出发, "
                "关注可行性和性能优化."
            )
        elif rs == ResearchStyle.MIXED:
            parts.append(
                "用户偏好混合风格, 理论+实验+工程并重, "
                "根据问题灵活切换."
            )

        return "用户 taste profile:\n" + "\n".join(f"  - {p}" for p in parts)


# ── 问卷设计 ──────────────────────────────────────────────────────
# 12 题, 每题 4 选项. 选项索引 (0-3) 直接映射到对应维度的枚举值.


@dataclass
class Question:
    id: int
    dimension: str  # "thinking_mode" / "imagination" / "research_style"
    text: str
    options: list[str]  # 4 个选项, 索引 0-3 对应枚举顺序


_QUESTIONS: list[Question] = [
    # ── 思维模式 (Q1-Q4) ──
    Question(
        id=1,
        dimension="thinking_mode",
        text="面对一个新问题, 你倾向于:",
        options=[
            "拆成逻辑步骤, 逐步分析每个部分",
            "凭直觉感受答案的方向",
            "画图或可视化问题的结构",
            "看各部分如何作为一个系统连接",
        ],
    ),
    Question(
        id=2,
        dimension="thinking_mode",
        text="读论文时, 你最先看:",
        options=[
            "公式和数学推导",
            "摘要和结论, 抓大意",
            "图表和可视化",
            "方法论, 理解整个系统",
        ],
    ),
    Question(
        id=3,
        dimension="thinking_mode",
        text="解释一个概念时, 你自然会用:",
        options=[
            "精确定义和逻辑论证",
            "类比和直觉描述",
            "图表或视觉模型",
            "系统关系图展示连接",
        ],
    ),
    Question(
        id=4,
        dimension="thinking_mode",
        text="调试代码时, 你会:",
        options=[
            "逐行跟踪逻辑",
            "凭感觉定位 bug",
            "可视化数据流",
            "检查模块间的交互",
        ],
    ),
    # ── 想象力 (Q5-Q8) ──
    Question(
        id=5,
        dimension="imagination",
        text="思考未来可能性时, 你倾向于:",
        options=[
            "基于已证实的方法, 稳健推进",
            "在验证和新想法间找平衡",
            "提出大胆但合理的推测",
            "想象颠覆性的、范式转换的场景",
        ],
    ),
    Question(
        id=6,
        dimension="imagination",
        text="面对不确定的数据, 你:",
        options=[
            "只采信高置信度部分, 明确标注不确定",
            "区分已验证和推测, 两者都给",
            "大胆外推, 探索可能含义",
            "提出全新解读框架, 不受旧范式束缚",
        ],
    ),
    Question(
        id=7,
        dimension="imagination",
        text="做研究计划时, 你偏好:",
        options=[
            "沿成熟路径推进, 降低风险",
            "主线稳妥 + 支线探索",
            "尝试非常规方法, 接受失败风险",
            "追求范式突破, 不在乎短期产出",
        ],
    ),
    Question(
        id=8,
        dimension="imagination",
        text="看到意外结果, 你的第一反应:",
        options=[
            "检查实验是否有误, 谨慎对待",
            "考虑多种解释, 逐一排查",
            "思考这可能暗示什么新机制",
            "想这是否能推翻现有理论",
        ],
    ),
    # ── 研究风格 (Q9-Q12) ──
    Question(
        id=9,
        dimension="research_style",
        text="你的研究驱动力是:",
        options=[
            "数学优美和物理本质",
            "数据和实验现象",
            "解决实际工程问题",
            "根据问题灵活切换",
        ],
    ),
    Question(
        id=10,
        dimension="research_style",
        text="选择方法时, 你看重:",
        options=[
            "理论严谨性和推导正确",
            "实验可复现性和统计显著",
            "工程可行性和性能",
            "能否适配当前问题",
        ],
    ),
    Question(
        id=11,
        dimension="research_style",
        text="评价一个结果的好坏, 你看:",
        options=[
            "是否从第一性原理严格推出",
            "是否有足够的实验验证",
            "是否能解决实际问题",
            "是否在理论/实验/工程上都有价值",
        ],
    ),
    Question(
        id=12,
        dimension="research_style",
        text="你最享受的研究环节:",
        options=[
            "推导公式, 建立理论模型",
            "做实验, 分析数据",
            "把理论变成可用的东西",
            "在理论/实验/工程间穿梭",
        ],
    ),
]

_DIMENSION_ENUMS = {
    "thinking_mode": ThinkingMode,
    "imagination": Imagination,
    "research_style": ResearchStyle,
}


class OnboardingQuestionnaire:
    """12 题 onboarding 问卷, 逐题收集答案并提取 taste."""

    def __init__(self) -> None:
        self._questions = list(_QUESTIONS)
        self._answers: dict[int, int] = {}  # question_id -> option_index (0-3)

    @property
    def total_questions(self) -> int:
        return len(self._questions)

    def get_question(self, index: int) -> dict[str, Any] | None:
        """获取第 index 题 (0-based). 返回 None 表示超出范围."""
        if index < 0 or index >= len(self._questions):
            return None
        q = self._questions[index]
        return {
            "index": index,
            "id": q.id,
            "dimension": q.dimension,
            "text": q.text,
            "options": q.options,
        }

    def submit_answer(self, question_id: int, option_index: int) -> bool:
        """提交一题的答案. option_index 0-3. 返回是否成功."""
        if option_index < 0 or option_index > 3:
            return False
        # 验证 question_id 存在
        if not any(q.id == question_id for q in self._questions):
            return False
        self._answers[question_id] = option_index
        return True

    @property
    def answered_count(self) -> int:
        return len(self._answers)

    @property
    def is_complete(self) -> bool:
        return len(self._answers) == len(self._questions)

    def next_index(self) -> int | None:
        """返回下一个未答题的 index, 全答完返回 None."""
        for i, q in enumerate(self._questions):
            if q.id not in self._answers:
                return i
        return None

    def extract(self) -> TasteProfile:
        """从已收集的答案提取 TasteProfile. 未答完的维度用默认值."""
        if not self._answers:
            return TasteProfile(completed=False, skipped=True)

        # 按维度统计投票
        dim_votes: dict[str, Counter] = {
            "thinking_mode": Counter(),
            "imagination": Counter(),
            "research_style": Counter(),
        }
        for q in self._questions:
            if q.id in self._answers:
                opt_idx = self._answers[q.id]
                enum_list = list(_DIMENSION_ENUMS[q.dimension])
                if opt_idx < len(enum_list):
                    dim_votes[q.dimension][enum_list[opt_idx]] += 1

        # 取每维度票数最多的
        profile = TasteProfile(completed=self.is_complete)
        for dim_name, counter in dim_votes.items():
            if counter:
                setattr(profile, dim_name, counter.most_common(1)[0][0])

        return profile


# ── 持久化 ────────────────────────────────────────────────────────


def _default_profile_path() -> Path:
    cache_dir = os.environ.get("HUGINN_CACHE_DIR", "")
    if cache_dir:
        p = Path(cache_dir) / "taste_profile.json"
    else:
        p = Path.home() / ".huginn" / "taste_profile.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_profile(profile: TasteProfile, path: Path | None = None) -> Path:
    """把 profile 存到 JSON. 返回实际存储路径."""
    p = path or _default_profile_path()
    data = {
        "thinking_mode": profile.thinking_mode.value,
        "imagination": profile.imagination.value,
        "research_style": profile.research_style.value,
        "completed": profile.completed,
        "skipped": profile.skipped,
    }
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_profile(path: Path | None = None) -> TasteProfile | None:
    """从 JSON 加载 profile. 文件不存在返回 None."""
    p = path or _default_profile_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return TasteProfile(
            thinking_mode=ThinkingMode(data.get("thinking_mode", "systems")),
            imagination=Imagination(data.get("imagination", "balanced")),
            research_style=ResearchStyle(data.get("research_style", "mixed")),
            completed=data.get("completed", False),
            skipped=data.get("skipped", False),
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        return None
