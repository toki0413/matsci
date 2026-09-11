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
    为 needs_grounding 并在 stderr 说明, 绝不伪装成通过。
    """
    try:
        return _post("/v1/research/grounding", {"report": report, "trace": list(trace or [])})
    except Exception as exc:  # noqa: BLE001 — 例如 server 未启动; 如实上报给用户
        print(f"[gateway] 门禁 HTTP 调用失败({exc}); 按 needs_grounding 处理", file=sys.stderr)
        return {"verdict": "needs_grounding", "unsubstantiated": ["门禁不可达"]}