"""Bot 桥接管理路由.

提供 REST 端点管理 bot bridge 的生命周期和配置,
同时暴露 OneBot v11 事件接收端点给 go-cqhttp / Lagrange / NapCat 上报.

端点:
- GET  /bot/status           查看运行状态 + 统计
- POST /bot/start            启动 bridge
- POST /bot/stop             停止 bridge
- PUT  /bot/config           更新配置
- GET  /bot/config           查看当前配置
- POST /onebot/v11/event     OneBot v11 事件上报入口 (go-cqhttp POST 到这里)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse

from huginn.bot.bridge import (
    BotConfig,
    get_bridge,
    get_config,
    set_config,
    start_bridge,
    stop_bridge,
)
from huginn.bot.wechat_bridge import (
    WeChatConfig,
    get_wechat_bridge,
    get_wechat_config,
    set_wechat_config,
    start_wechat_bridge,
    stop_wechat_bridge,
)
from huginn.security.auth import require_admin_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["bot"])


# ── 管理端点 ────────────────────────────────────────────────────────


@router.get("/bot/status", dependencies=[Depends(require_admin_key)])
async def bot_status() -> dict[str, Any]:
    """获取 bot 运行状态和统计信息."""
    bridge = get_bridge()
    if bridge is None:
        return {
            "running": False,
            "enabled": get_config().enabled,
            "platform": get_config().platform,
            "stats": None,
        }
    return bridge.get_status()


@router.post("/bot/start", dependencies=[Depends(require_admin_key)])
async def bot_start() -> dict[str, Any]:
    """启动 bot bridge.

    会从 server_core 获取全局 agent 实例, 然后创建 bridge 并开始监听.
    需要先在配置里把 enabled 设为 True 并填好 api_url / bot_id.
    """
    cfg = get_config()
    if not cfg.enabled:
        return {
            "success": False,
            "error": "bot 未启用, 请先 PUT /bot/config 设置 enabled=true",
        }

    bridge = get_bridge()
    if bridge is not None and bridge.is_running:
        return {"success": True, "message": "bot 已在运行", "status": bridge.get_status()}

    try:
        bridge = await start_bridge()
        return {"success": True, "status": bridge.get_status()}
    except Exception as e:
        logger.error("启动 bot 失败", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/bot/stop", dependencies=[Depends(require_admin_key)])
async def bot_stop() -> dict[str, Any]:
    """停止 bot bridge."""
    bridge = get_bridge()
    if bridge is None:
        return {"success": True, "message": "bot 未在运行"}
    try:
        await stop_bridge()
        return {"success": True, "message": "bot 已停止"}
    except Exception as e:
        logger.error("停止 bot 失败", exc_info=True)
        return {"success": False, "error": str(e)}


@router.get("/bot/config", dependencies=[Depends(require_admin_key)])
async def get_bot_config() -> dict[str, Any]:
    """查看当前 bot 配置."""
    cfg = get_config()
    return {"success": True, "config": asdict(cfg)}


@router.put("/bot/config", dependencies=[Depends(require_admin_key)])
async def update_bot_config(params: dict[str, Any]) -> dict[str, Any]:
    """更新 bot 配置.

    支持部分更新 —— 只传需要改的字段即可. 如果 bridge 正在运行, 配置会实时生效
    (api_url / 限流参数等), 但 platform / bot_id 变更可能需要重启 bridge.

    请求体示例:
    {
        "enabled": true,
        "platform": "qq",
        "bot_id": "123456789",
        "api_url": "http://127.0.0.1:5700",
        "admin_users": ["111", "222"],
        "allowed_groups": ["999999"],
        "rate_limit_per_user": 5,
        "rate_limit_per_group": 20,
        "response_cooldown": 3.0,
        "max_response_length": 2000
    }
    """
    current = get_config()
    current_dict = asdict(current)

    # 合并: 只更新传入的字段
    for key, value in params.items():
        if key in current_dict:
            current_dict[key] = value
        else:
            logger.warning("忽略未知配置字段: %s", key)

    try:
        new_cfg = BotConfig(**current_dict)
        set_config(new_cfg)
        return {"success": True, "config": asdict(new_cfg)}
    except TypeError as e:
        return {"success": False, "error": f"配置字段类型错误: {e}"}


# ── OneBot v11 事件接收 ─────────────────────────────────────────────


@router.post("/onebot/v11/event")
async def onebot_event(event: dict[str, Any]) -> dict[str, Any]:
    """OneBot v11 事件上报入口.

    go-cqhttp / Lagrange / NapCat 配置 HTTP POST 上报到这个地址,
    每条消息 / 通知 / 请求事件都会 POST 过来.

    返回 200 + {"status":"ok"} 确认收到, 实际回复通过 OneBot HTTP API
    (send_msg) 异步发回. 如果 bridge 未启动, 事件会被丢弃但仍返回 200,
    避免 go-cqhttp 重试.
    """
    bridge = get_bridge()
    if bridge is None or not bridge.is_running:
        # bridge 没跑, 吞掉事件但返回 200
        return {"status": "ok", "message": "bridge not running"}

    try:
        reply = await bridge.handle_message(event)
        return {"status": "ok", "reply": reply}
    except Exception as e:
        logger.error("处理 OneBot 事件失败", exc_info=True)
        # 返回 200 避免 go-cqhttp 反复重试
        return {"status": "error", "error": str(e)}


# ── WeChat (iLink) 桥接管理 ─────────────────────────────────────────


@router.get("/bot/wechat/status", dependencies=[Depends(require_admin_key)])
async def wechat_status() -> dict[str, Any]:
    """查看 wechat bridge 运行状态."""
    bridge = get_wechat_bridge()
    if bridge is None:
        return {
            "running": False,
            "enabled": get_wechat_config().enabled,
            "stats": None,
        }
    return bridge.get_status()


@router.post("/bot/wechat/start", dependencies=[Depends(require_admin_key)])
async def wechat_start() -> dict[str, Any]:
    """启动 wechat bridge (iLink 长轮询)."""
    cfg = get_wechat_config()
    if not cfg.enabled:
        return {
            "success": False,
            "error": "wechat 未启用, 请先 PUT /bot/wechat/config 设 enabled=true",
        }
    bridge = get_wechat_bridge()
    if bridge is not None and bridge.is_running:
        return {"success": True, "message": "wechat 已在运行", "status": bridge.get_status()}
    try:
        bridge = await start_wechat_bridge()
        return {"success": True, "status": bridge.get_status()}
    except Exception as e:
        logger.error("启动 wechat 失败", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/bot/wechat/stop", dependencies=[Depends(require_admin_key)])
async def wechat_stop() -> dict[str, Any]:
    """停止 wechat bridge."""
    bridge = get_wechat_bridge()
    if bridge is None:
        return {"success": True, "message": "wechat 未在运行"}
    try:
        await stop_wechat_bridge()
        return {"success": True, "message": "wechat 已停止"}
    except Exception as e:
        logger.error("停止 wechat 失败", exc_info=True)
        return {"success": False, "error": str(e)}


@router.get("/bot/wechat/config", dependencies=[Depends(require_admin_key)])
async def wechat_get_config() -> dict[str, Any]:
    """查看当前 wechat 配置."""
    return {"success": True, "config": asdict(get_wechat_config())}


@router.put("/bot/wechat/config", dependencies=[Depends(require_admin_key)])
async def wechat_update_config(params: dict[str, Any]) -> dict[str, Any]:
    """更新 wechat 配置 (部分更新, bridge 运行中实时生效)."""
    current = get_wechat_config()
    current_dict = asdict(current)
    for key, value in params.items():
        if key in current_dict:
            current_dict[key] = value
        else:
            logger.warning("忽略未知 wechat 配置字段: %s", key)
    try:
        new_cfg = WeChatConfig(**current_dict)
        set_wechat_config(new_cfg)
        return {"success": True, "config": asdict(new_cfg)}
    except TypeError as e:
        return {"success": False, "error": f"配置字段类型错误: {e}"}


# ── 企业微信回调 ─────────────────────────────────────────────────────
#
# 企业微信在后台配置回调 URL 后, 会向 /wechat/event 发 GET (验证) 和 POST (消息).
# 签名验证: sha1("".join(sorted([token, timestamp, nonce]))) == msg_signature
# token 通过环境变量 HUGINN_WECOM_TOKEN 配置.


def _verify_wecom_signature(
    token: str, signature: str, timestamp: str, nonce: str
) -> bool:
    if not token:
        return False
    computed = hashlib.sha1(
        "".join(sorted([token, timestamp, nonce])).encode()
    ).hexdigest()
    return computed == signature


@router.get("/wechat/event")
async def wechat_event_verify(
    msg_signature: str = "",
    timestamp: str = "",
    nonce: str = "",
    echostr: str = "",
):
    """企业微信回调 URL 验证 (GET).

    首次配置回调时企业微信发 GET, 验签通过后原样返回 echostr.
    """
    token = os.environ.get("HUGINN_WECOM_TOKEN", "")
    if not _verify_wecom_signature(token, msg_signature, timestamp, nonce):
        return {"error": "invalid signature"}
    return PlainTextResponse(echostr)


@router.post("/wechat/event")
async def wechat_event(
    request: Request,
    msg_signature: str = "",
    timestamp: str = "",
    nonce: str = "",
):
    """企业微信消息回调 (POST).

    验签后解析 XML, 转成 OneBot v11 事件交给 bridge 处理.
    企业微信要求 5 秒内响应, 所以异步处理消息后立即回 success.
    """
    token = os.environ.get("HUGINN_WECOM_TOKEN", "")
    if not _verify_wecom_signature(token, msg_signature, timestamp, nonce):
        return {"error": "invalid signature"}

    body = await request.body()
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        logger.warning("企业微信回调 XML 解析失败")
        return PlainTextResponse("success")

    # 目前只处理文本消息
    if root.findtext("MsgType", "") != "text":
        return PlainTextResponse("success")

    from_user = root.findtext("FromUserName", "")
    content = root.findtext("Content", "")
    if not from_user or not content:
        return PlainTextResponse("success")

    event = {
        "post_type": "message",
        "message_type": "private",
        "user_id": from_user,
        "message": content,
        "raw_message": content,
        "message_id": root.findtext("MsgId", ""),
        "sender": {"nickname": from_user},
    }

    # 优先 wechat bridge, 没有就退回 QQ bridge
    bridge = get_wechat_bridge() or get_bridge()
    if bridge is not None and bridge.is_running:
        asyncio.create_task(bridge.handle_message(event))

    return PlainTextResponse("success")


# ponytail: Telegram/Discord/Slack/DingTalk/Feishu channels skipped —
# no users on those platforms yet. When needed, add a ChannelBase
# abstraction + per-platform adapter (like WeChatBridge extends BotBridge).
# The existing BotBridge + OneBot v11 forwarding pattern is the template.
