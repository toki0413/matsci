"""USER_PROMPT_SUBMIT 钩子: 校验用户消息里点名的工具是否真实存在.

背景: 用户有时会把工具名拼错(比如 structure_tools 多了个 s, 或 struct_tool
拼错), 或者干脆编一个不存在的名字(比如 calculator_tool). 当前 agent 碰到
这种消息会默默调 ls 探索或者 web_search 搜索, 而不是直接告诉用户这工具
不存在. 这里在用户消息进 LLM 之前做一次正则扫描, 把消息里形如
xxx_tool / xxx_database 的词跟 agent 实际可用工具列表对一遍, 对不上的就
在 prompt_guidance 里塞一段提示, 让 agent 明确告知用户工具不存在, 并建议
最接近的正确名字.

设计要点:
1. 纯正则匹配, 不调 LLM, 零成本. 默认开, 不需要环境变量.
2. 只抓 xxx_tool / xxx_database 后缀, 避免误伤普通英文词.
3. available_tools 拿不到就跳过, 不报错 —— 上下文没给工具列表没法判.
4. 多个错误工具名合并成一条提示, 避免刷屏.
5. 钩子抛异常不能拖垮主流程, 全包在 try/except 里.
"""

from __future__ import annotations

import logging
import re

from huginn.hooks import HookContext

logger = logging.getLogger(__name__)

# 用户消息里常出现的工具名后缀模式. 只抓小写开头 + 下划线 + tool/database,
# 避免误伤普通单词. 例: structure_tool / materials_database / vasp_tool.
_TOOL_NAME_PATTERN = re.compile(
    r"\b[a-z][a-z0-9_]*_tool\b|\b[a-z][a-z0-9_]*_database\b"
)


class ToolNameValidationHook:
    """USER_PROMPT_SUBMIT 钩子: 校验用户消息里点名的工具是否真实存在.

    纯正则匹配, 不调 LLM. 拿不到 available_tools 就跳过, 不报错.
    """

    def __init__(self) -> None:
        # 编译好的正则, 复用避免每次 __call__ 都重编一遍
        self._pattern = _TOOL_NAME_PATTERN

    async def __call__(self, ctx: HookContext) -> HookContext | None:
        try:
            message = ctx.metadata.get("user_message")
            if not isinstance(message, str) or not message.strip():
                return None

            # available_tools 拿不到就没法判, 直接跳过不报错
            available = ctx.metadata.get("available_tools")
            if not available:
                return None
            # 统一成 set 方便比对, 容错 list/set/tuple 各种序列类型
            try:
                valid_names = set(available)
            except Exception:
                valid_names = set()
            if not valid_names:
                return None

            # 扫消息里所有 xxx_tool / xxx_database 形态的词
            mentioned = set(self._pattern.findall(message))
            if not mentioned:
                return None

            # 对不上的就是可疑工具名, 排个序输出稳定点
            bad_names = sorted(n for n in mentioned if n not in valid_names)
            if not bad_names:
                return None

            # 多个错误名合并成一条提示, 避免刷屏
            bad_text = ", ".join(bad_names)
            valid_text = ", ".join(sorted(valid_names))
            guidance = (
                f"提示: 用户消息里提到的工具 '{bad_text}' 不在可用工具列表里。"
                f"可用工具包括: {valid_text}。"
                f"请在回复里明确告知用户该工具不存在, 并建议最接近的正确工具名。"
                f"不要默默调 ls 或 web_search 探索。"
            )

            # 跟其他同事件钩子保持一致: 用 list 累加而不是覆盖,
            # 给先后挂上来的 PromptGuidanceHook / 别的钩子留兼容
            existing = ctx.metadata.get("prompt_guidance")
            if isinstance(existing, list):
                existing.append(guidance)
            elif isinstance(existing, str) and existing:
                ctx.metadata["prompt_guidance"] = [existing, guidance]
            else:
                ctx.metadata["prompt_guidance"] = guidance
        except Exception:
            # 校验逻辑本身不该挂, 真挂了也别拖垮 agent
            logger.warning("ToolNameValidationHook raised", exc_info=True)
        return None
