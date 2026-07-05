"""WeChat (iLink Bot API) 桥接器 —— 把个人微信消息接入 Huginn agent.

iLink 是腾讯开放的个人微信 Bot API, 本地 daemon 默认监听 9011 端口.
本模块通过 HTTP 长轮询拉取消息, 转成 OneBot v11 事件格式后复用
BotBridge 的全部处理逻辑 (限流 / 命令分发 / agent 调用 / 格式化).

数据流:
  iLink daemon (localhost:9011)
       |  HTTP long-poll  POST /ilink/bot/getupdates
       v
  WeChatBridge._poll_loop()
       |  转成 OneBot v11 event dict
       v
  BotBridge.handle_message()  (继承, 零改动)
       |  agent.chat()
       v
  WeChatBridge._send_response()  (重写, 走 iLink /ilink/bot/send)
       v
  iLink daemon -> 微信

iLink 消息结构参考:
  message_id, from_user_id, message_type, item_list[]
  item_list 里 type=1 是文本 (text_item.text), type=2 图片, type=4 文件 ...
  群消息带 chat_room_id 字段.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from huginn.bot.bridge import BotBridge, BotConfig

logger = logging.getLogger(__name__)


# ── 配置 ────────────────────────────────────────────────────────────


@dataclass
class WeChatConfig:
    """iLink Bot API 桥接配置."""

    enabled: bool = False
    api_url: str = "http://localhost:9011"  # iLink 本地 daemon 地址
    bot_id: str = ""  # 微信 bot ID (留空则群消息不做过滤)
    admin_users: list[str] = field(default_factory=list)
    rate_limit_per_user: int = 5  # 每用户每分钟
    response_cooldown: float = 3.0  # 同一用户两次回复间隔
    max_response_length: int = 2000


def _to_bot_config(cfg: WeChatConfig) -> BotConfig:
    """WeChatConfig -> BotConfig, 复用 BotBridge 的全部参数."""
    return BotConfig(
        enabled=True,
        platform="wechat",
        bot_id=cfg.bot_id,
        api_url=cfg.api_url,
        admin_users=list(cfg.admin_users),
        rate_limit_per_user=cfg.rate_limit_per_user,
        rate_limit_per_group=20,
        response_cooldown=cfg.response_cooldown,
        max_response_length=cfg.max_response_length,
    )


# ── WeChatBridge ────────────────────────────────────────────────────


class WeChatBridge(BotBridge):
    """iLink Bot API -> Huginn agent 桥接器.

    继承 BotBridge 获取限流 / 命令 / agent 调用 / 消息格式化能力,
    只重写两件事:
    - 收: 长轮询 iLink /ilink/bot/getupdates, 转成 OneBot v11 事件
    - 发: 通过 iLink /ilink/bot/send 回复 (替代 OneBot send_msg)
    """

    def __init__(self, agent: Any, wechat_config: WeChatConfig):
        super().__init__(agent, _to_bot_config(wechat_config))
        self.wechat_config = wechat_config
        self._client: Any = None  # httpx.AsyncClient, start() 时创建
        self._poll_task: asyncio.Task | None = None

    # ── 生命周期 ──────────────────────────────────────────────

    async def start(self) -> None:
        import httpx

        if self._running:
            logger.warning("wechat bridge 已在运行, 跳过重复启动")
            return

        # 长轮询 timeout 设比 daemon 持有时间长, 避免客户端先超时
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(65.0))
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(
            "wechat bridge 已启动 (api_url=%s, bot_id=%s)",
            self.config.api_url,
            self.config.bot_id or "(未设置)",
        )

    async def stop(self) -> None:
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        logger.info("wechat bridge 已停止")

    # ── 长轮询 ────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """持续拉取 iLink 消息, 转成 OneBot v11 事件后丢给 handle_message."""
        url = f"{self.config.api_url.rstrip('/')}/ilink/bot/getupdates"
        while self._running:
            try:
                resp = await self._client.post(
                    url, json={"timeout": 30}, timeout=60.0
                )
                if resp.status_code != 200:
                    logger.warning("iLink getupdates 返回 %s", resp.status_code)
                    await asyncio.sleep(5)
                    continue

                data = resp.json()
                # 响应体可能是列表或 {"messages": [...]}
                messages = (
                    data if isinstance(data, list) else data.get("messages", [])
                )
                for msg in messages:
                    event = self._convert_ilink_event(msg)
                    if event is not None:
                        await self.handle_message(event)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("iLink 轮询异常", exc_info=True)
                await asyncio.sleep(5)

    def _convert_ilink_event(self, msg: dict) -> dict[str, Any] | None:
        """iLink 消息 -> OneBot v11 事件 dict, 供 handle_message 消费.

        item_list 里 type=1 是文本, 其他类型暂不处理.
        群消息带 chat_room_id / group_id.
        """
        user_id = str(msg.get("from_user_id", ""))
        if not user_id:
            return None

        # 群消息: iLink 用 chat_room_id 或 group_id 标识群
        group_raw = (
            msg.get("chat_room_id")
            or msg.get("group_id")
            or msg.get("chatroom_id")
        )
        is_group = bool(group_raw)

        # 从 item_list 提取文本 (type=1)
        parts: list[str] = []
        for item in msg.get("item_list") or []:
            if item.get("type") == 1 and item.get("text_item"):
                parts.append(item["text_item"].get("text", ""))
        text = "".join(parts).strip()
        if not text:
            return None

        return {
            "post_type": "message",
            "message_type": "group" if is_group else "private",
            "user_id": user_id,
            "group_id": str(group_raw) if is_group else None,
            "message": text,
            "raw_message": text,
            "message_id": msg.get("message_id"),
            "sender": {"nickname": msg.get("from_nickname", user_id)},
        }

    # ── 发送回复 (重写) ──────────────────────────────────────

    async def _send_response(self, info: dict[str, Any], message: str) -> bool:
        """通过 iLink /ilink/bot/send 发送回复, 替代 OneBot send_msg."""
        if not self.config.api_url or self._client is None:
            logger.debug("iLink 未就绪, 跳过发送: %s", message[:80])
            return False

        target = info.get("group_id") or info["user_id"]
        payload = {"to": target, "message": message}
        try:
            url = f"{self.config.api_url.rstrip('/')}/ilink/bot/send"
            resp = await self._client.post(url, json=payload, timeout=15.0)
            if resp.status_code == 200:
                return True
            logger.warning(
                "iLink 发送失败 status=%s: %s", resp.status_code, resp.text[:200]
            )
            return False
        except Exception:
            logger.error("iLink 发送异常", exc_info=True)
            return False


# ── 单例管理 (跟 bridge.py 同一套模式) ───────────────────────────────


_wechat_bridge: WeChatBridge | None = None
_wechat_config: WeChatConfig = WeChatConfig()


def get_wechat_config() -> WeChatConfig:
    return _wechat_config


def set_wechat_config(cfg: WeChatConfig) -> None:
    global _wechat_config
    _wechat_config = cfg
    if _wechat_bridge is not None:
        _wechat_bridge.wechat_config = cfg
        _wechat_bridge.config = _to_bot_config(cfg)
    logger.info(
        "wechat 配置已更新 (enabled=%s, api_url=%s)", cfg.enabled, cfg.api_url
    )


def get_wechat_bridge() -> WeChatBridge | None:
    return _wechat_bridge


async def start_wechat_bridge(agent: Any = None) -> WeChatBridge:
    global _wechat_bridge
    if _wechat_bridge is not None and _wechat_bridge.is_running:
        return _wechat_bridge
    if agent is None:
        from huginn.server_core import get_agent

        agent = await get_agent()
    _wechat_bridge = WeChatBridge(agent, _wechat_config)
    await _wechat_bridge.start()
    return _wechat_bridge


async def stop_wechat_bridge() -> None:
    global _wechat_bridge
    if _wechat_bridge is not None:
        await _wechat_bridge.stop()
        _wechat_bridge = None
    logger.info("wechat bridge 已销毁")


# ── 自检 ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # 快速验证 iLink -> OneBot 事件转换逻辑
    class _FakeAgent:
        pass

    bridge = WeChatBridge(_FakeAgent(), WeChatConfig())

    # 纯文本私聊
    e = bridge._convert_ilink_event(
        {
            "message_id": "m1",
            "from_user_id": "wx_user_1",
            "item_list": [{"type": 1, "text_item": {"text": "hello"}}],
        }
    )
    assert e is not None and e["user_id"] == "wx_user_1"
    assert e["message_type"] == "private" and e["message"] == "hello"

    # 群消息
    e2 = bridge._convert_ilink_event(
        {
            "message_id": "m2",
            "from_user_id": "wx_user_1",
            "chat_room_id": "group_123",
            "item_list": [{"type": 1, "text_item": {"text": "hi group"}}],
        }
    )
    assert e2 is not None and e2["message_type"] == "group"
    assert e2["group_id"] == "group_123"

    # 非文本 / 无 user_id -> None
    assert (
        bridge._convert_ilink_event(
            {"from_user_id": "u1", "item_list": [{"type": 2}]}
        )
        is None
    )
    assert bridge._convert_ilink_event({"item_list": []}) is None

    print("wechat_bridge self-check passed")
