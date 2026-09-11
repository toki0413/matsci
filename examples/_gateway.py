"""examples/ 共享的 Huginn 单网关 HTTP 客户端 (ADR-0001 单网关注入).

让示例在**不 import huginn.*** 的前提下调用 Huginn server 的 HTTP API。
仅标准库; 不 import 任何 huginn 模块, 因此 arch 单网关门禁不受影响。

常见用法::

    from _gateway import ground, set_server
    set_server(url)                     # 可选; 默认 HUGINN_SERVER_URL 或 127.0.0.1:8765
    ground(report, trace)               # -> {"verdict": ..., "unsubstantiated": [...]}

等价物: ``huginn.research.grounding_verifier()(report, trace)``。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error  # HTTP/网络错误类型: 只按"网关不可达"降级, 不误报证据缺陷
import urllib.request  # 仅做 HTTP POST, 不引入第三方业务依赖

# server 地址: 可用环境变量 HUGINN_SERVER_URL 覆盖, 或调用 set_server().
_DEFAULT_SERVER = os.environ.get("HUGINN_SERVER_URL", "http://127.0.0.1:8765")
_SERVER: dict[str, str] = {"url": _DEFAULT_SERVER}


def set_server(url: str) -> None:
    """切换 Huginn 网关地址 (可用 --server / HUGINN_SERVER_URL)."""
    if url:
        _SERVER["url"] = url


def _post(path: str, payload: dict) -> dict:
    url = _SERVER["url"].rstrip("/") + path
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ground(report: str, trace: list[str] | None = None) -> dict:
    """结论证伪门禁: 报告每个数值必须落在工具执行轨迹中.

    等价物: ``grounding_verifier()(report, trace)``。server 不可达时如实降级
    为 needs_grounding 并区分**网关存活失败**与**证据缺陷**, 绝不伪装成通过:

    - 网关不可达(网络/HTTP/解析错误) → ``gateway_unreachable=True``, 不是数值
      无法溯源, 调用方不应把它当``unsubstantiated``误报给用户。
    - 其余异常 → 防御性兜底, 仍按 needs_grounding 处理, 但带类型指纹便于排查。
    """
    _body = {"report": report, "trace": list(trace or [])}
    _net = (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError)
    try:
        return _post("/v1/research/grounding", _body)
    except _net as exc:  # 网络/存活问题: 与"数值无法溯源"是两回事
        print(f"[gateway] 门禁服务不可达({exc}); 请先启动 huginn server 再重跑", file=sys.stderr)
        return {"verdict": "needs_grounding", "unsubstantiated": [],
                "gateway_unreachable": True, "reason": f"gateway unreachable: {exc}"}
    except json.JSONDecodeError as exc:  # 服务返回了非 JSON 的坏负载
        print(f"[gateway] 门禁返回无法解析的响应({exc}); 按不可达处理", file=sys.stderr)
        return {"verdict": "needs_grounding", "unsubstantiated": [],
                "gateway_unreachable": True, "reason": f"bad response: {exc}"}
    except Exception as exc:  # noqa: BLE001 — 防御性兜底: 任一异常都如实降级, 不伪装通过, 留类型指纹
        print(f"[gateway] 门禁调用内部异常({type(exc).__name__}: {exc}); 按 needs_grounding 处理",
              file=sys.stderr)
        return {"verdict": "needs_grounding", "unsubstantiated": [],
                "gateway_unreachable": True, "reason": f"{type(exc).__name__}: {exc}"}