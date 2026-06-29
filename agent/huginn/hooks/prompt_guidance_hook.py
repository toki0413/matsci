"""USER_PROMPT_SUBMIT 钩子: 关键词命中后强制要求走工具, 并按任务类型推荐工具.

背景: 第九轮测试里 Q'2 agent 直接拿 LLM 内部知识答了"硅带隙 1.1 eV",
根本没碰 validate_tool. 这种问题在材料计算场景里很危险——模型记忆
里的常数不一定对得上当前结构/参数. 这里在用户提问环节做一次纯规则
匹配, 命中"验证/计算/求解/检索/对比"之类动词就把"必须调工具"的指令
塞进 ctx.metadata, 由 agent 拼到 system message 里.

设计要点 (v2):
1. 关键词分中文/英文两组, 覆盖"读/查/对比/拟合/优化/可视化/检索/校验"
   等材料计算常见动词, 不局限于 validate/numerical 三个工具.
2. 任务→工具映射: 按关键词推断推荐工具, 给 agent 更精准的引导,
   避免用户问"对比"被引导到 validate_tool (validate 不做对比).
3. 反模式识别: 用户明确说"用你的知识回答/不用调工具"时不强制,
   尊重用户意图.
4. 多工具候选: 用户提问里点名了具体工具时, 优先按用户点名走,
   不覆盖用户意图.
5. 纯字符串匹配, 不调 LLM, 零成本. 默认开, 不需要环境变量开关.
"""

from __future__ import annotations

import logging
import re

from huginn.hooks import HookContext

logger = logging.getLogger(__name__)

# 中文动词关键词. 命中任一就触发"必须调工具"引导.
# 覆盖材料计算的常见动作: 验证/计算/检索/对比/分析/构建.
_KEYWORDS_CN = (
    "验证", "校验", "检查", "分析", "计算", "求解",
    "解方程", "拟合", "优化", "最小化", "积分",
    "检索", "查询", "查找", "搜索", "对比", "比较",
    "构建", "建模", "读取", "导入", "可视化",
    "评估", "判定", "判断", "识别",
)

# 英文关键词. 用词边界匹配, 避免 "checkpoint" 里的 "check" 误判.
_KEYWORDS_EN = (
    "check", "verify", "validate", "calculate", "solve",
    "fit", "optimize", "minimize", "integrate",
    "search", "query", "retrieve", "compare", "analyze",
    "build", "construct", "read", "load", "visualize",
    "evaluate", "assess", "identify",
)

# 用户明确说"不用调工具"时, 不强制. 这些是反模式关键词.
_SKIP_CN = ("用你的知识", "不用调工具", "不用工具", "直接回答", "凭记忆")
_SKIP_EN = (r"from (?:your |the )?memory", r"without (?:using )?tools?")

# 任务→推荐工具映射. 按 keyword 命中推断, 给 agent 更精准的引导.
# 不在这里列的工具 (bash_tool/code_tool 等) 不主动推荐, 因为它们
# 是通用工具, 用户不会在 prompt 里点名.
_TASK_TOOL_MAP = {
    # 校验类 → validate_tool
    "validate_tool": ("验证", "校验", "检查", "verify", "validate", "check"),
    # 计算/求解/拟合类 → numerical_tool
    "numerical_tool": (
        "计算", "求解", "解方程", "拟合", "优化", "最小化", "积分",
        "calculate", "solve", "fit", "optimize", "minimize", "integrate",
    ),
    # 结构类 → structure_tool
    "structure_tool": (
        "构建", "建模", "读取结构", "分析结构",
        "build", "construct", "read structure",
    ),
    # 检索类 → rag_tool
    "rag_tool": ("检索", "查询文献", "查找文档", "search", "retrieve"),
    # 数据库类 → materials_database_tool
    "materials_database_tool": (
        "查询材料", "查材料", "材料数据库",
        "query material",
    ),
    # 可视化类 → 不主动推荐具体工具, 让 agent 自己选
}

# 基础引导语. 命中关键词就追加.
_GUIDANCE_BASE = (
    "这个问题需要调用对应工具验证/计算/检索, 不要直接用知识回答。"
    "请先调用工具获取结果, 再基于工具结果分析。"
)


class PromptGuidanceHook:
    """USER_PROMPT_SUBMIT 钩子: 关键词命中后追加"必须用工具"的引导.

    纯规则匹配, 不花 LLM 钱. 匹配失败 / 抛异常都不影响 agent 主流程.
    """

    def __init__(self) -> None:
        # 英文关键词预编译成正则, 词边界匹配
        self._en_pattern = re.compile(
            r"\b(" + "|".join(_KEYWORDS_EN) + r")\b", re.IGNORECASE
        )
        # 反模式英文也预编译
        self._skip_en_pattern = re.compile(
            r"|".join(_SKIP_EN), re.IGNORECASE
        )

    async def __call__(self, ctx: HookContext) -> HookContext | None:
        try:
            message = ctx.metadata.get("user_message")
            if not isinstance(message, str) or not message.strip():
                return None

            # 反模式: 用户明确说"不用调工具", 尊重用户意图
            if self._match_skip(message):
                return None

            if not self._match_keywords(message):
                return None

            guidance = _GUIDANCE_BASE

            # 1) 用户点名了具体工具, 优先按用户点名走
            mentioned = [
                t for t in _TASK_TOOL_MAP
                if t in message
            ]
            if mentioned:
                tools_text = "、".join(mentioned)
                guidance = f"{guidance}\n请确保调用 {tools_text}。"
            else:
                # 2) 没点名工具, 按 keyword 推断推荐工具
                recommended = self._recommend_tools(message)
                if recommended:
                    tools_text = "、".join(recommended)
                    guidance = (
                        f"{guidance}\n根据问题类型, 推荐调用 {tools_text} "
                        f"(如果该工具不适用, 可换其他工具, 但不要直接用知识回答)。"
                    )

            # 写进 metadata, 由 agent 在拼 messages 时取走.
            # 用 list 累加而不是覆盖, 给后续可能挂上来的同事件钩子留兼容.
            existing = ctx.metadata.get("prompt_guidance")
            if isinstance(existing, list):
                existing.append(guidance)
            elif isinstance(existing, str) and existing:
                ctx.metadata["prompt_guidance"] = [existing, guidance]
            else:
                ctx.metadata["prompt_guidance"] = guidance
        except Exception:
            # 兜底: 规则匹配本身不该挂, 真挂了也别拖垮 agent
            logger.warning("PromptGuidanceHook raised", exc_info=True)
        return None

    def _match_keywords(self, text: str) -> bool:
        """中文字面命中或英文词边界命中即返回 True."""
        for kw in _KEYWORDS_CN:
            if kw in text:
                return True
        if self._en_pattern.search(text):
            return True
        return False

    def _match_skip(self, text: str) -> bool:
        """用户明确说"不用调工具"时返回 True, 跳过引导."""
        for kw in _SKIP_CN:
            if kw in text:
                return True
        if self._skip_en_pattern.search(text):
            return True
        return False

    def _recommend_tools(self, text: str) -> list[str]:
        """按 keyword 命中推断推荐工具, 最多 3 个, 按命中数排序."""
        scores: dict[str, int] = {}
        text_lower = text.lower()
        for tool, keywords in _TASK_TOOL_MAP.items():
            score = 0
            for kw in keywords:
                # 中文按字面, 英文按词边界
                if kw.isascii():
                    if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                        score += 1
                else:
                    if kw in text:
                        score += 1
            if score > 0:
                scores[tool] = score
        # 按命中数降序, 最多取 3 个
        return [
            tool for tool, _ in sorted(
                scores.items(), key=lambda x: -x[1]
            )[:3]
        ]
