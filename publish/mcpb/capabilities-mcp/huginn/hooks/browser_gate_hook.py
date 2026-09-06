"""浏览器操作确认门控 hook — 受 BrowserAct confirmation gate 启发.

敏感浏览器操作 (login, form submit, file upload, navigation to payment)
需要用户确认才能执行, 不是自动拦截, 而是发 warning 让用户决定.

与用户偏好一致: "物理 precheck 检测到问题时, 先收到警告, 可以选择强制继续"
"""

from __future__ import annotations

import logging

from huginn.hooks import PRE_TOOL_USE, HookContext, HookManager

logger = logging.getLogger(__name__)

# 需要确认的操作
_SENSITIVE_ACTIONS = {
    "login",
    "fill_form",
}

# 需要确认的 URL 模式 (支付/账户/删除等)
_SENSITIVE_URL_PATTERNS = [
    "payment", "checkout", "delete", "remove", "logout",
    "settings/security", "account/close",
]


def _is_sensitive_browser_call(ctx: HookContext) -> tuple[bool, str]:
    """检查是否是敏感浏览器操作, 返回 (是否敏感, 原因)."""
    if ctx.tool_name != "browser_tool":
        return False, ""

    args = ctx.tool_args
    if not isinstance(args, dict):
        return False, ""

    action = (args.get("action") or "").lower()

    # login / fill_form 需要确认
    if action in _SENSITIVE_ACTIONS:
        return True, f"sensitive browser action: {action}"

    # navigate 到敏感 URL
    if action == "navigate":
        url = (args.get("url") or "").lower()
        for pattern in _SENSITIVE_URL_PATTERNS:
            if pattern in url:
                return True, f"navigating to sensitive URL (matched '{pattern}')"

    return False, ""


async def browser_gate_hook(ctx: HookContext) -> HookContext:
    """PRE_TOOL_USE hook: 检测敏感浏览器操作, 标记需要确认.

    不直接 block (用户偏好: 先警告可强制继续), 而是在 metadata 里加
    'requires_confirmation' 标记, agent 主循环可以检测到后询问用户.
    """
    is_sensitive, reason = _is_sensitive_browser_call(ctx)
    if is_sensitive:
        ctx.metadata["requires_confirmation"] = True
        ctx.metadata["confirmation_reason"] = reason
        logger.info("browser gate: %s", reason)

    return ctx


def register_browser_gate_hooks(hm: HookManager) -> None:
    """注册浏览器门控 hook. 幂等."""

    flag = "_browser_gate_registered"
    if getattr(hm, flag, False):
        return
    setattr(hm, flag, True)

    hm.register(PRE_TOOL_USE, browser_gate_hook)
    logger.info("browser gate hooks registered")
