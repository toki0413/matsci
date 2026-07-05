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

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter

from huginn.bot.bridge import (
    BotConfig,
    get_bridge,
    get_config,
    set_config,
    start_bridge,
    stop_bridge,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["bot"])


# ── 管理端点 ────────────────────────────────────────────────────────


@router.get("/bot/status")
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


@router.post("/bot/start")
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


@router.post("/bot/stop")
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


@router.get("/bot/config")
async def get_bot_config() -> dict[str, Any]:
    """查看当前 bot 配置."""
    cfg = get_config()
    return {"success": True, "config": asdict(cfg)}


@router.put("/bot/config")
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
