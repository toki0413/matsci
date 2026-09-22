"""JEV MCP server 接入路径契约测试.

独立加载 ``servers/jev-mcp/server.py`` (自包含, 不 import huginn.*), 验证:

  - 暴露 3 个 MCP 工具 (jev_noul / jev_choice / jev_score)
  - 注入假 transport 的 round-trip: 返回值可解析出答案字段
  - fail-open: 无 API key / 传输异常 → 显式错误文本, 不抛穿

零网络: transport 一律注入 mock. 用 asyncio.run 直调 call_tool。
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SERVER = _REPO / "servers" / "jev-mcp" / "server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("_jev_mcp", str(_SERVER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeTransport:
    """可注入 transport: 记录请求, 返回给定响应或抛异常."""

    def __init__(self, payload=None, exc=None):
        self._payload = payload
        self._exc = exc
        self.calls: list[tuple] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        if self._exc is not None:
            raise self._exc
        return _FakeResp(self._payload)


def _text_of(result) -> str:
    return result[0].text


def _loads(result) -> dict:
    return json.loads(_text_of(result))


def test_exposes_three_primitives():
    server = _load_server()
    names = [t.name for t in server.TOOLS]
    assert {"jev_noul", "jev_choice", "jev_score"} <= set(names)


def test_noul_roundtrip_with_injected_transport():
    server = _load_server()
    transport = _FakeTransport(
        payload={"answers": {"is_urgent": {"noul": 0.87, "confidence": 0.8}}}
    )
    server._api = server.JevApi(api_key="k-test", transport=transport)
    result = asyncio.run(
        server.call_tool("jev_noul", {"state": {"text": "…"}, "questions": {"is_urgent": "urgent?"}})
    )
    out = _loads(result)
    assert out["success"] is True
    assert out["answers"]["is_urgent"]["noul"] == 0.87
    # 请求里的 body 结构: questions{name:{type:noul}}
    body = transport.calls[0]["json"]
    assert body["questions"]["is_urgent"]["type"] == "noul"


def test_choice_roundtrip():
    server = _load_server()
    transport = _FakeTransport(
        payload={"answers": {"c": {"choice": "billing", "probabilities": {"billing": 0.84}, "confidence": 0.6}}}
    )
    server._api = server.JevApi(api_key="k-test", transport=transport)
    result = asyncio.run(
        server.call_tool(
            "jev_choice",
            {"state": {}, "instruction": "which dept?", "options": ["billing", "technical"], "question_name": "c"},
        )
    )
    out = _loads(result)
    assert out["answer"]["choice"] == "billing"


def test_score_roundtrip():
    server = _load_server()
    transport = _FakeTransport(
        # 默认 question_name="score", 故 answers 键须为 "score"
        payload={"answers": {"score": {"score": 3.4, "confidence": 0.9}}}
    )
    server._api = server.JevApi(api_key="k-test", transport=transport)
    result = asyncio.run(
        server.call_tool("jev_score", {"state": {}, "instruction": "rate severity"})
    )
    out = _loads(result)
    assert out["answer"]["score"] == 3.4


def test_fail_open_without_api_key():
    server = _load_server()
    server._api = server.JevApi(api_key=None)  # 无 key
    result = asyncio.run(
        server.call_tool("jev_noul", {"state": {}, "questions": {"a": "rel?"}})
    )
    out = _loads(result)
    assert out["success"] is False
    assert "unavailable" in out["error"]


def test_fail_open_on_transport_error():
    server = _load_server()
    server._api = server.JevApi(api_key="k-test", transport=_FakeTransport(exc=RuntimeError("down")))
    result = asyncio.run(
        server.call_tool("jev_noul", {"state": {}, "questions": {"a": "rel?"}})
    )
    out = _loads(result)
    assert out["success"] is False
    assert "unavailable" in out["error"]
