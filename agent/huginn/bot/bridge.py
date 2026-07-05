"""OneBot v11 bot bridge —— 把 QQ / WeChat 消息接入 Huginn agent.

设计思路:
- 不依赖 nonebot / go-cqhttp SDK, 纯标准库 + aiohttp
- 通过 FastAPI 的 POST 端点接收 OneBot v11 事件 (HTTP POST 上报)
- 通过 aiohttp 向 OneBot HTTP API 发送消息 (send_msg)
- agent.chat() 是 async generator, 这里把流式输出收集成一条完整消息
- 限流: 每用户 / 每群 分别计数, 全局冷却防止刷屏

OneBot v11 协议参考:
https://github.com/botuniverse/onebot-11

典型部署:
  go-cqhttp / Lagrange / NapCat
       │ HTTP POST 上报事件
       ▼
  Huginn FastAPI  (POST /onebot/v11/event)
       │ agent.chat()
       ▼
  Huginn Agent → 流式回复
       │ aiohttp POST
       ▼
  go-cqhttp HTTP API (send_msg) → 发回 QQ/WeChat
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── 配置 ────────────────────────────────────────────────────────────


@dataclass
class BotConfig:
    """Bot 桥接配置.

    通过环境变量或 REST API 注入, enabled=False 时桥接完全不启动.
    """

    enabled: bool = False
    platform: str = "qq"  # "qq" 或 "wechat"
    bot_id: str = ""  # 机器人 QQ 号或微信 ID
    http_port: int = 8080  # 监听 OneBot 事件的端口 (预留, 实际由 FastAPI 统一监听)
    ws_url: str = ""  # 可选: 反向 WebSocket 地址 (暂未实现)
    api_url: str = ""  # OneBot HTTP API 地址, 如 http://127.0.0.1:5700
    admin_users: list[str] = field(default_factory=list)  # 管理员 QQ/微信 ID
    allowed_groups: list[str] = field(default_factory=list)  # 群白名单, 空列表 = 不限
    rate_limit_per_user: int = 5  # 每用户每分钟最大消息数
    rate_limit_per_group: int = 20  # 每群每分钟最大消息数
    response_cooldown: float = 3.0  # 同一用户/群两次回复之间的冷却秒数
    max_response_length: int = 2000  # 单条消息最大长度 (QQ 限制约 2000 字)


# ── BotBridge ──────────────────────────────────────────────────────


class BotBridge:
    """消息平台与 Huginn agent 之间的桥接器.

    从 QQ/WeChat (经 OneBot v11 HTTP POST) 接收消息,
    转发给 agent.chat() 并把回复发回平台.
    """

    def __init__(self, agent: Any, config: BotConfig):
        self.agent = agent
        self.config = config
        self._running = False
        self._session: Any = None  # aiohttp.ClientSession, start() 时创建

        # 限流: 记录最近一分钟内的消息时间戳
        self._user_msg_times: dict[str, list[float]] = {}
        self._group_msg_times: dict[str, list[float]] = {}
        # 冷却: 同一用户/群上次回复的时间
        self._last_reply: dict[str, float] = {}

        # 统计
        self._stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "commands_handled": 0,
            "last_message_time": 0.0,
        }

        # 防止并发调 agent 时互相踩 thread_id
        self._agent_lock = asyncio.Lock()

    # ── 生命周期 ──────────────────────────────────────────────

    async def start(self) -> None:
        """启动桥接器 —— 创建 HTTP 会话, 标记为运行中."""
        import aiohttp

        if self._running:
            logger.warning("bot bridge 已经在运行, 跳过重复启动")
            return

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Content-Type": "application/json"},
        )
        self._running = True
        logger.info(
            "bot bridge 已启动 (platform=%s, bot_id=%s, api_url=%s)",
            self.config.platform,
            self.config.bot_id or "(未设置)",
            self.config.api_url or "(未设置, 仅本地回复)",
        )

    async def stop(self) -> None:
        """停止桥接器 —— 关闭 HTTP 会话."""
        self._running = False
        if self._session is not None:
            await self._session.close()
            self._session = None
        logger.info("bot bridge 已停止")

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict[str, Any]:
        """返回当前运行状态和统计信息."""
        return {
            "running": self._running,
            "platform": self.config.platform,
            "bot_id": self.config.bot_id,
            "api_url": self.config.api_url,
            "stats": dict(self._stats),
        }

    # ── 核心: 处理 OneBot 事件 ────────────────────────────────

    async def handle_message(self, event: dict) -> str | None:
        """处理一条 OneBot v11 事件.

        解析消息 → 限流检查 → 命令分发或调 agent → 格式化 → 发回平台.
        返回回复文本 (供调试 / 快速回复), 无需回复时返回 None.
        """
        self._stats["messages_received"] += 1
        self._stats["last_message_time"] = time.time()

        # 只处理消息事件, notice / request / meta_event 直接跳过
        post_type = event.get("post_type")
        if post_type != "message":
            return None

        # 解析事件字段
        info = self._parse_event(event)
        if info is None:
            return None

        user_id = info["user_id"]
        group_id = info.get("group_id")
        is_group = info["is_group"]

        # 群消息: 检查白名单
        if is_group and self.config.allowed_groups:
            if group_id not in self.config.allowed_groups:
                return None

        # 群消息: 检查是否 @了机器人 (bot_id 为空时不限制)
        if is_group and self.config.bot_id:
            if not info["at_bot"]:
                return None
            # 去掉消息开头的 @bot
            info["text"] = self._strip_at_mention(info["text"])

        text = info["text"].strip()
        if not text:
            return None

        # 限流检查
        if not self._check_rate_limit(user_id, group_id):
            logger.info("限流命中, 忽略 user=%s group=%s", user_id, group_id)
            return None

        # 命令处理 (/help /reset /model /persona)
        if text.startswith("/"):
            response = self._handle_command(text, user_id, group_id)
            if response:
                self._stats["commands_handled"] += 1
                await self._send_response(info, response)
                return response
            return None

        # 调 agent
        thread_id = self._make_thread_id(info)
        try:
            response = await self._call_agent(text, thread_id)
        except Exception:
            logger.error("agent 调用失败", exc_info=True)
            self._stats["messages_failed"] += 1
            error_msg = "内部错误, 稍后再试."
            await self._send_response(info, error_msg)
            return error_msg

        if not response:
            return None

        # 格式化 + 分段发送
        for chunk in self._format_response(response):
            await self._send_response(info, chunk)

        self._stats["messages_processed"] += 1
        return response

    # ── 事件解析 ──────────────────────────────────────────────

    def _parse_event(self, event: dict) -> dict[str, Any] | None:
        """从 OneBot v11 事件中提取关键字段.

        返回 dict 包含: user_id, group_id(可选), is_group, text, at_bot, sender_name.
        无法解析返回 None.
        """
        msg_type = event.get("message_type")
        user_id = str(event.get("user_id", ""))
        if not user_id:
            return None

        is_group = msg_type == "group"
        group_id = str(event.get("group_id", "")) if is_group else None

        # 提取纯文本 —— message 字段可能是字符串或消息段数组
        message = event.get("message", "")
        raw = event.get("raw_message", "")
        text, at_bot = self._extract_text(message, raw)

        sender = event.get("sender", {})
        sender_name = (
            sender.get("card") or sender.get("nickname") or user_id
        )

        return {
            "user_id": user_id,
            "group_id": group_id,
            "is_group": is_group,
            "text": text,
            "at_bot": at_bot,
            "sender_name": sender_name,
            "message_id": event.get("message_id"),
        }

    def _extract_text(self, message: Any, raw: str) -> tuple[str, bool]:
        """从 OneBot 消息段中提取纯文本.

        message 可能是:
        - 字符串 (CQ 码格式): "[CQ:at,qq=123] hello"
        - 数组 (消息段格式): [{"type":"text","data":{"text":"hello"}}, ...]

        返回 (纯文本, 是否@了机器人).
        """
        at_bot = False
        bot_id = self.config.bot_id

        if isinstance(message, list):
            # 消息段数组 —— 逐段提取 text, 检查 at 段
            parts: list[str] = []
            for seg in message:
                if not isinstance(seg, dict):
                    continue
                seg_type = seg.get("type", "")
                data = seg.get("data", {})
                if seg_type == "text":
                    parts.append(data.get("text", ""))
                elif seg_type == "at":
                    qq = str(data.get("qq", ""))
                    if bot_id and qq == bot_id:
                        at_bot = True
                    # at 段不加入文本, 后面统一处理
                elif seg_type == "image":
                    url = data.get("url", "")
                    if url:
                        parts.append(f"[图片:{url}]")
                    else:
                        parts.append("[图片]")
                elif seg_type == "face":
                    parts.append(f"[表情{data.get('id', '')}]")
                elif seg_type == "reply":
                    # 回复段, 不加入文本
                    pass
                else:
                    # 其他段 (record/video 等) 用占位
                    parts.append(f"[{seg_type}]")
            text = "".join(parts).strip()
        else:
            # 字符串格式 —— 去掉 CQ 码
            text = str(message) if message else raw
            text = self._strip_cq_codes(text)
            if bot_id:
                at_bot = f"[CQ:at,qq={bot_id}]" in text or f"[CQ:at,qq={bot_id}]" in raw

        return text, at_bot

    def _strip_cq_codes(self, text: str) -> str:
        """去掉字符串里的 CQ 码 ([CQ:...]), 只保留文本."""
        # 匹配 [CQ:type,key=value,...] 格式
        cleaned = re.sub(r"\[CQ:[^\]]+\]", "", text)
        return cleaned.strip()

    def _strip_at_mention(self, text: str) -> str:
        """去掉消息开头的 @bot 文本.

        有些客户端会把 @ 转成纯文本 "@" 加昵称, 这里一并清理.
        """
        # 去掉可能残留的 CQ at 码
        text = re.sub(r"\[CQ:at,qq=\d+\]", "", text)
        # 去掉开头可能出现的 "@昵称 " 或 "@昵称\u2005" (全角空格)
        text = re.sub(r"^@\S+[\s\u2005]*", "", text)
        return text.strip()

    # ── 限流 / 冷却 ──────────────────────────────────────────

    def _check_rate_limit(self, user_id: str, group_id: str | None) -> bool:
        """检查是否超过限流阈值.

        每用户每分钟 max rate_limit_per_user 条,
        每群每分钟 max rate_limit_per_group 条,
        同一用户/群两次回复间隔不小于 response_cooldown.
        """
        now = time.time()
        window = 60.0  # 一分钟窗口

        # 用户级限流
        user_times = self._user_msg_times.setdefault(user_id, [])
        user_times[:] = [t for t in user_times if now - t < window]
        if len(user_times) >= self.config.rate_limit_per_user:
            return False

        # 群级限流
        if group_id:
            group_times = self._group_msg_times.setdefault(group_id, [])
            group_times[:] = [t for t in group_times if now - t < window]
            if len(group_times) >= self.config.rate_limit_per_group:
                return False

        # 冷却: 同一用户/群
        cooldown_key = group_id or user_id
        last = self._last_reply.get(cooldown_key, 0)
        if now - last < self.config.response_cooldown:
            return False

        # 通过检查 —— 记录时间戳
        user_times.append(now)
        if group_id:
            group_times.append(now)
        self._last_reply[cooldown_key] = now
        return True

    # ── 命令处理 ──────────────────────────────────────────────

    def _handle_command(
        self, text: str, user_id: str, group_id: str | None
    ) -> str | None:
        """处理斜杠命令, 返回回复文本或 None."""
        # 拆出命令名和参数
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        is_admin = user_id in self.config.admin_users

        if cmd == "/help":
            return self._cmd_help()

        if cmd == "/reset":
            return self._cmd_reset(group_id or user_id)

        if cmd == "/model":
            return self._cmd_model(arg, is_admin)

        if cmd == "/persona":
            return self._cmd_persona(arg, is_admin)

        # 未知命令
        return f"未知命令: {cmd}\n输入 /help 查看可用命令"

    def _cmd_help(self) -> str:
        """显示帮助."""
        return (
            "Huginn Bot 命令列表\n"
            "──────────────────\n"
            "/help    —— 显示本帮助\n"
            "/reset   —— 清空当前会话历史\n"
            "/model   —— 查看当前模型\n"
            "/persona —— 查看当前人格设定\n"
            "──────────────────\n"
            "直接发送消息即可与 Agent 对话."
        )

    def _cmd_reset(self, thread_key: str) -> str:
        """清空会话历史."""
        try:
            self.agent.memory.clear_session()
            # 重置 agent 内部状态, 让下一轮从干净上下文开始
            if hasattr(self.agent, "_conversation_summary"):
                self.agent._conversation_summary = ""
            if hasattr(self.agent, "_agent_graph"):
                self.agent._agent_graph = None
            logger.info("会话已重置 (key=%s)", thread_key)
            return "会话已重置, 可以重新开始对话了."
        except Exception:
            logger.error("重置会话失败", exc_info=True)
            return "重置失败, 请稍后重试."

    def _cmd_model(self, arg: str, is_admin: bool) -> str:
        """查看 / 切换当前模型."""
        current = self._get_model_name()
        if not arg:
            return f"当前模型: {current}"

        if not is_admin:
            return f"当前模型: {current}\n(仅管理员可切换模型)"

        # 切换模型需要通过配置系统, 这里只做提示
        return (
            f"当前模型: {current}\n"
            f"切换模型请通过 REST API /bot/config 或配置文件修改."
        )

    def _cmd_persona(self, arg: str, is_admin: bool) -> str:
        """查看 / 切换当前人格."""
        current = self._get_persona_name()
        if not arg:
            return f"当前人格: {current}"

        if not is_admin:
            return f"当前人格: {current}\n(仅管理员可切换人格)"

        # 尝试通过 PersonaManager 查找并切换
        try:
            from huginn.personas import PersonaManager

            manager = PersonaManager()
            available = manager.list()
            if arg not in available:
                names = ", ".join(available[:10])
                return f"未找到人格: {arg}\n可用人格: {names}"

            persona = manager.get(arg)
            self.agent.set_persona(persona=persona)
            logger.info("人格已切换: %s -> %s", current, arg)
            return f"人格已切换为: {arg}"
        except ImportError:
            return f"当前人格: {current}\n(人格切换功能不可用)"
        except Exception:
            logger.error("切换人格失败", exc_info=True)
            return f"当前人格: {current}\n(切换失败, 请稍后重试.)"

    # ── Agent 调用 ───────────────────────────────────────────

    def _make_thread_id(self, info: dict[str, Any]) -> str:
        """根据消息来源生成 thread_id, 用于会话隔离.

        群消息用 group_id, 私聊用 user_id, 加平台前缀避免跟 Web/CLI 会话冲突.
        """
        platform = self.config.platform
        if info["is_group"]:
            return f"bot_{platform}_group_{info['group_id']}"
        return f"bot_{platform}_user_{info['user_id']}"

    async def _call_agent(self, content: str, thread_id: str) -> str:
        """调用 agent.chat() 并把流式输出收集成一条完整回复.

        agent.chat() 是 async generator, 每轮 yield 一个 state dict.
        我们取最后一条 AI 消息的 content 作为最终回复.
        """
        final_text = ""
        async with self._agent_lock:
            async for state in self.agent.chat(content, thread_id=thread_id):
                if not isinstance(state, dict):
                    continue
                msgs = state.get("messages", [])
                for msg in msgs:
                    msg_type = getattr(msg, "type", "")
                    msg_content = getattr(msg, "content", "")
                    # 只收集 AI 消息的文本内容, 工具消息跳过
                    if msg_type == "ai" and isinstance(msg_content, str) and msg_content:
                        final_text = msg_content
        return final_text

    # ── 消息格式化 ───────────────────────────────────────────

    def _format_response(self, text: str) -> list[str]:
        """把 agent 回复格式化为适合 QQ/WeChat 的纯文本消息.

        1. 去掉 QQ/WeChat 无法渲染的 markdown 语法
        2. 按最大长度分段
        """
        cleaned = self._strip_markdown(text)
        return self._split_message(cleaned, self.config.max_response_length)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """去掉 markdown 格式标记, 保留纯文本内容."""
        # 代码块 ```lang ... ``` -> 内容
        text = re.sub(r"```[^\n]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)
        # 行内代码 `code` -> code
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # 图片 ![alt](url) -> [图片]
        text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "[图片]", text)
        # 链接 [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        # 粗体 **text** / __text__
        text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        # 斜体 *text* / _text_
        text = re.sub(r"(?<!\*)\*([^\*]+)\*(?!\*)", r"\1", text)
        text = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"\1", text)
        # 删除线 ~~text~~
        text = re.sub(r"~~([^~]+)~~", r"\1", text)
        # 标题 # / ## / ### -> 去掉井号
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # 引用 > text -> text
        text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
        # 无序列表 - / * 开头 -> •
        text = re.sub(r"^[\-\*]\s+", "• ", text, flags=re.MULTILINE)
        # 有序列表 1. -> 去掉数字
        text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
        # 水平线 --- / *** -> ───
        text = re.sub(r"^[\-\*]{3,}$", "───", text, flags=re.MULTILINE)
        # 多余的空行压缩
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _split_message(text: str, max_length: int) -> list[str]:
        """把长文本按 max_length 切成多段.

        优先在换行处切, 其次在空格处切, 都不行就硬切.
        """
        if len(text) <= max_length:
            return [text]

        chunks: list[str] = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break

            chunk = text[:max_length]
            # 优先在换行处断开
            cut = chunk.rfind("\n")
            if cut > max_length // 2:
                chunks.append(text[:cut])
                text = text[cut + 1:]
                continue
            # 其次在空格处断开
            cut = chunk.rfind(" ")
            if cut > max_length // 2:
                chunks.append(text[:cut])
                text = text[cut + 1:]
                continue
            # 找不到合适的断点, 硬切
            chunks.append(chunk)
            text = text[max_length:]

        return chunks

    # ── 发送消息 ──────────────────────────────────────────────

    async def _send_response(self, info: dict[str, Any], message: str) -> bool:
        """通过 OneBot HTTP API 把消息发回平台.

        需要 config.api_url 配置了 go-cqhttp/Lagrange 的 HTTP 地址.
        没配的话只记日志, 不发 (方便本地测试).
        """
        if not self.config.api_url:
            logger.debug("api_url 未配置, 跳过发送: %s", message[:80])
            return False

        if self._session is None:
            logger.warning("HTTP 会话未初始化, 无法发送消息")
            return False

        # 构造 OneBot send_msg 请求体
        if info["is_group"]:
            payload = {
                "message_type": "group",
                "group_id": int(info["group_id"]),
                "message": message,
            }
        else:
            payload = {
                "message_type": "private",
                "user_id": int(info["user_id"]),
                "message": message,
            }

        try:
            url = f"{self.config.api_url.rstrip('/')}/send_msg"
            async with self._session.post(url, json=payload) as resp:
                data = await resp.json()
                if data.get("status") == "ok":
                    return True
                logger.warning(
                    "发送消息失败 retcode=%s: %s",
                    data.get("retcode"),
                    data.get("msg", ""),
                )
                return False
        except Exception:
            logger.error("发送消息异常", exc_info=True)
            return False

    # ── 辅助: 读取 agent 状态 ─────────────────────────────────

    def _get_model_name(self) -> str:
        """获取当前模型名称."""
        model = getattr(self.agent, "model", None)
        if model is None:
            return "mock (无模型)"
        name = getattr(model, "model_name", None) or getattr(model, "model", None)
        return str(name) if name else "unknown"

    def _get_persona_name(self) -> str:
        """获取当前人格名称."""
        name = getattr(self.agent, "persona_name", None)
        return str(name) if name else "default"


# ── 全局单例管理 ────────────────────────────────────────────────────
#
# bridge 需要持有 agent 引用, agent 通过 server_core.get_agent() 获取.
# 这里用模块级变量管理生命周期, routes/bot.py 通过这些函数操作.

_bridge: BotBridge | None = None
_config: BotConfig = BotConfig()


def get_config() -> BotConfig:
    """获取当前 bot 配置 (可能被 REST API 修改过)."""
    return _config


def set_config(cfg: BotConfig) -> None:
    """更新全局配置, 同时同步到已存在的 bridge 实例."""
    global _config
    _config = cfg
    if _bridge is not None:
        _bridge.config = cfg
    logger.info(
        "bot 配置已更新 (enabled=%s, platform=%s)",
        cfg.enabled,
        cfg.platform,
    )


def get_bridge() -> BotBridge | None:
    """获取当前 bridge 实例 (可能为 None)."""
    return _bridge


async def start_bridge(agent: Any = None) -> BotBridge:
    """创建并启动 bridge.

    如果 agent 为 None, 尝试从 server_core 获取全局 agent.
    重复调用时如果已经在运行, 直接返回现有实例.
    """
    global _bridge

    if _bridge is not None and _bridge.is_running:
        return _bridge

    if agent is None:
        from huginn.server_core import get_agent

        agent = await get_agent()

    _bridge = BotBridge(agent, _config)
    await _bridge.start()
    return _bridge


async def stop_bridge() -> None:
    """停止并销毁当前 bridge."""
    global _bridge
    if _bridge is not None:
        await _bridge.stop()
        _bridge = None
    logger.info("bridge 已销毁")
