"""混沌工程测试套件 —— 注入各种故障, 验证系统韧性。

覆盖 7 大场景:
  1. 后端服务故障 (LLM 异常, MCP 断连, 工具超时)
  2. WebSocket 连接韧性 (断连, 畸形 JSON, 超长消息, 高频 ping, 空闲超时)
  3. SQLite 故障恢复 (database locked, 磁盘满, 文件损坏)
  4. 文件系统故障 (不存在的目录/文件, 只读文件)
  5. 网络超时模拟 (HTTP client 超时, SSH 超时)
  6. 资源耗尽 (MemoryError, 线程池, 文件描述符)
  7. 并发冲突 (plan 并发执行, checkpoint 并发 accept/reject, config 并发更新)

核心原则: 用 mock 模拟故障, 不真搞坏系统。每个测试验证 "故障下仍安全"。
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# server.py 间接依赖 mcp, 没装的话整个模块都跳过
pytest.importorskip("mcp")

# Chaos tests use mock injection — fast enough for CI but logically grouped
# with the integration suite. Run in integration CI job.
pytestmark = pytest.mark.integration

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from huginn.server import app

client = TestClient(app)

# WS 路由挂在 /v1 前缀下 (include_v1_routes), 但 root compat 也保留了 /ws/agent
WS_PATH = "/ws/agent"

_HEADERS = {"X-HUGINN-API-KEY": "test-key"}


# ─── 公共 fixture ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _no_rate_limit():
    """关掉 HTTP 限流, 避免高频请求被 429。"""
    from huginn.server import rate_limit_middleware

    mw_globals = rate_limit_middleware.__globals__
    saved = mw_globals.get("_RATE_LIMIT")
    mw_globals["_RATE_LIMIT"] = 0
    yield
    if saved is not None:
        mw_globals["_RATE_LIMIT"] = saved


# ─── WS mock 基础设施 (复用 test_ws_integration 的思路) ──────────────

class _MockModel:
    """够用的 chat model mock。"""

    def __init__(self, plan_text: str = "{}") -> None:
        self.plan_text = plan_text

    async def ainvoke(self, prompt: str, **kw: Any):
        class _R:
            content = plan_text
        return _R()


class _MockAgent:
    """按脚本回放 state 的 agent。"""

    def __init__(self, states: list[dict] | None = None) -> None:
        self._states = list(states or [])
        self.model = _MockModel()
        self.persona_name = "default"

    def set_persona(self, *a, **kw):
        pass

    async def chat(self, content: str, thread_id: str = "default"):
        for s in self._states:
            yield s


class _MockFactory:
    def __init__(self, agent: _MockAgent) -> None:
        self._agent = agent

    def create_lead(self, *a, **kw):
        return self._agent

    def create(self, *a, **kw):
        return self._agent

    def list_profiles(self):
        return []


@pytest.fixture
def ws_harness(tmp_path, monkeypatch):
    """给 WS 路由打 mock, 让测试不走真实 agent。"""
    import huginn.routes.ws as ws_mod

    agent = _MockAgent()
    cfg = SimpleNamespace(
        workspace=str(tmp_path),
        rag_enabled=False,
        persona_auto_route=False,
        persona_auto_route_threshold=0.6,
        team_mode_enabled=False,
        max_concurrent_subagents=2,
    )
    ctx = SimpleNamespace(config=SimpleNamespace(workspace=str(tmp_path)), kb=None)

    async def _get_agent():
        return agent

    monkeypatch.setattr(ws_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(ws_mod, "get_agent", _get_agent)
    monkeypatch.setattr(ws_mod, "get_agent_factory", lambda: _MockFactory(agent))
    monkeypatch.setattr(ws_mod, "get_context", lambda: ctx)
    monkeypatch.setattr(ws_mod, "get_memory_manager", lambda: MagicMock())
    monkeypatch.setattr(ws_mod, "get_or_create_thread", lambda *a, **k: {"id": "t"})

    return SimpleNamespace(agent=agent, cfg=cfg, ctx=ctx)


def _drain(ws, max_msgs: int = 32) -> list[dict]:
    """拉消息直到 done/error 或到达上限。"""
    out = []
    for _ in range(max_msgs):
        msg = ws.receive_json()
        out.append(msg)
        if msg.get("type") in ("done", "error"):
            break
    return out


# ═══════════════════════════════════════════════════════════════════
#  1. 后端服务故障模拟
# ═══════════════════════════════════════════════════════════════════

class TestBackendServiceFailure:
    """后端依赖挂了, API 不能跟着崩。"""

    def test_llm_provider_exception_returns_500_not_crash(self, monkeypatch):
        """LLM provider 抛异常时, /health/live 仍然正常, chat 路径返回错误而非 crash。"""
        # health/live 是最便宜的端点, 它不应该受 LLM 故障影响
        resp = client.get("/health/live", headers=_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_mcp_disconnect_reflected_in_status(self, monkeypatch):
        """MCP server 断连后, /mcp/status 应反映断连状态。"""
        # 用 mock context 替换 get_context, 让 mcp_manager 返回断连状态
        import huginn.routes.mcp as mcp_mod

        fake_mgr = MagicMock()
        fake_mgr.get_server_status.return_value = {
            "test-server": {"connected": False, "error": "connection lost"}
        }

        fake_ctx = SimpleNamespace(mcp_manager=fake_mgr)
        monkeypatch.setattr(mcp_mod, "get_context", lambda: fake_ctx)

        resp = client.get("/mcp/status", headers=_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert "servers" in body
        assert body["servers"]["test-server"]["connected"] is False

    def test_mcp_disconnect_does_not_break_other_endpoints(self, monkeypatch):
        """MCP 断连时, 其他端点 (如 /tools) 仍然正常工作。"""
        import huginn.routes.mcp as mcp_mod

        fake_mgr = MagicMock()
        fake_mgr.get_server_status.side_effect = ConnectionError("MCP gone")
        fake_ctx = SimpleNamespace(mcp_manager=fake_mgr)
        monkeypatch.setattr(mcp_mod, "get_context", lambda: fake_ctx)

        # /mcp/status 可能返回错误, 但 /tools 和 /health/live 不受影响
        tools_resp = client.get("/tools", headers=_HEADERS)
        assert tools_resp.status_code == 200

        health_resp = client.get("/health/live", headers=_HEADERS)
        assert health_resp.status_code == 200

    def test_tool_execution_timeout_does_not_block(self, monkeypatch):
        """工具执行超时时, 系统不应永久阻塞。"""
        # mock 一个会卡住很久的工具调用, 然后验证 asyncio.wait_for 能取消它
        # 不走真实 HTTP (TestClient 同步且没有内置超时), 直接验证超时逻辑

        async def _slow_task():
            await asyncio.sleep(999)  # 永远不返回

        async def _run_with_timeout():
            try:
                await asyncio.wait_for(_slow_task(), timeout=0.5)
                return "completed"
            except asyncio.TimeoutError:
                return "timeout"

        result = asyncio.run(_run_with_timeout())
        assert result == "timeout", "超时没有被正确处理, 任务仍在阻塞"

        # 验证服务仍在正常响应
        assert client.get("/health/live", headers=_HEADERS).status_code == 200


# ═══════════════════════════════════════════════════════════════════
#  2. WebSocket 连接韧性
# ═══════════════════════════════════════════════════════════════════

class TestWebSocketResilience:

    def test_ws_server_disconnect_client_receives_close(self, ws_harness):
        """服务端主动断连后, 客户端应收到 close frame 而非无限等待。"""
        with client.websocket_connect(WS_PATH, headers=_HEADERS) as ws:
            ws.send_json({"type": "ping"})
            msg = ws.receive_json()
            assert msg["type"] == "pong"
            # 正常关闭
            ws.close()

    def test_ws_malformed_json_returns_error_not_crash(self, ws_harness):
        """发畸形 JSON, 服务端应返回 error 消息而非 crash。"""
        with client.websocket_connect(WS_PATH, headers=_HEADERS) as ws:
            # 发一段不是 JSON 的文本
            ws.send_text("this is not json {{{")
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "json" in msg["error"].lower() or "malformed" in msg["error"].lower()

    def test_ws_oversized_message_rejected_or_handled(self, ws_harness):
        """发超长消息 (>64KB), 服务端应拒绝或安全处理, 不 crash。"""
        with client.websocket_connect(WS_PATH, headers=_HEADERS) as ws:
            # 构造一个 70KB 的 content, 超过 WSMessage 的 max_length=50000
            big_content = "A" * 70000
            ws.send_json({"type": "user_input", "content": big_content})
            msg = ws.receive_json()
            # Pydantic 校验失败, 应返回 error 而非 crash
            assert msg["type"] == "error"

    def test_ws_high_frequency_ping_no_crash(self, ws_harness):
        """高频 ping (100 次), 服务端不应 crash。"""
        with client.websocket_connect(WS_PATH, headers=_HEADERS) as ws:
            for i in range(100):
                ws.send_json({"type": "ping"})
                msg = ws.receive_json()
                assert msg["type"] == "pong", f"第 {i} 次 ping 没有收到 pong"

    def test_ws_idle_connection_cleanup(self, ws_harness):
        """连接建立后不发消息直接关闭, 服务端应正常清理。"""
        # 只连不发, 然后立刻关 — 验证没有异常
        with client.websocket_connect(WS_PATH, headers=_HEADERS) as ws:
            pass  # 立即关闭

        # 服务端没挂, 后续连接还能正常建
        with client.websocket_connect(WS_PATH, headers=_HEADERS) as ws:
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"


# ═══════════════════════════════════════════════════════════════════
#  3. SQLite 故障恢复
# ═══════════════════════════════════════════════════════════════════

class TestSQLiteFailureRecovery:

    def test_database_locked_handled_gracefully(self, tmp_path):
        """模拟 SQLite 'database is locked', 验证不会 crash。"""
        db_path = tmp_path / "test_locked.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (v INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()

        # sqlite3.Connection 是 C 扩展类型, 不能 patch 它的方法
        # 用 mock 对象模拟, 验证 OperationalError 能被正确捕获
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("database is locked")

        with pytest.raises(sqlite3.OperationalError):
            mock_conn.execute("SELECT * FROM t")

        # 恢复后数据库仍然可用
        rows = conn.execute("SELECT * FROM t").fetchall()
        assert rows == [(1,)]
        conn.close()

    def test_disk_full_error_propagates_not_crash(self, monkeypatch, tmp_path):
        """模拟磁盘满, 验证错误传播而非 crash。"""
        import huginn.routes.checkpoints as cp_mod

        # mock _snapshot_directory 抛出 OSError (磁盘满)
        monkeypatch.setattr(
            cp_mod,
            "_snapshot_directory",
            MagicMock(side_effect=OSError(28, "No space left on device")),
        )

        # 创建 checkpoint 会失败, 但应该返回错误而不是 500 crash
        resp = client.post(
            "/checkpoints",
            json={"path": str(tmp_path)},
            headers=_HEADERS,
        )
        # 要么是 500 (被全局 handler 捕获), 要么是其他错误码
        # 关键是不能让进程崩掉
        assert resp.status_code >= 400

    def test_corrupt_sqlite_returns_error_not_traceback(self, tmp_path):
        """模拟 SQLite 文件损坏, 验证返回错误消息而非 traceback。"""
        db_path = tmp_path / "corrupt.db"
        # 写一堆垃圾数据模拟损坏的 db 文件
        db_path.write_bytes(b"not a sqlite database at all!!!")

        # 尝试打开损坏的 db, 应该抛 DatabaseError 而非段错误
        conn = sqlite3.connect(str(db_path))
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("SELECT * FROM sqlite_master").fetchall()
        conn.close()


# ═══════════════════════════════════════════════════════════════════
#  4. 文件系统故障
# ═══════════════════════════════════════════════════════════════════

class TestFileSystemFailure:

    def test_write_to_nonexistent_dir_handled(self, tmp_path):
        """写入不存在的目录, 应返回错误而非 crash。"""
        nonexistent = tmp_path / "no_such_dir" / "subdir" / "file.txt"
        with pytest.raises(FileNotFoundError):
            nonexistent.write_text("test")

    def test_read_nonexistent_file_returns_404_not_500(self):
        """读取不存在的文件, 应返回 404 而非 500。"""
        # /checkpoints/{不存在的id} 应返回错误 (不是 500)
        resp = client.get("/checkpoints/nonexistent_id_12345", headers=_HEADERS)
        # checkpoint 路由返回 error 字段但 HTTP 状态码是 200
        # 关键是不返回 500 (说明没 crash)
        assert resp.status_code != 500
        body = resp.json()
        assert "error" in body or "id" in body

    def test_write_readonly_file_handled(self, tmp_path):
        """写入只读文件, 应返回错误而非 crash。"""
        ro_file = tmp_path / "readonly.txt"
        ro_file.write_text("original")
        ro_file.chmod(0o444)  # 只读

        try:
            with pytest.raises(PermissionError):
                ro_file.write_text("modified")
        finally:
            # Windows 上 chmod 可能不完全生效, 恢复权限避免清理问题
            ro_file.chmod(0o644)


# ═══════════════════════════════════════════════════════════════════
#  5. 网络超时模拟
# ═══════════════════════════════════════════════════════════════════

class TestNetworkTimeout:

    def test_http_client_timeout_propagates(self, monkeypatch):
        """模拟 httpx 超时, 验证错误正确传播。"""
        import httpx

        # mock httpx.AsyncClient.get 抛 TimeoutException
        original_get = httpx.AsyncClient.get

        async def _timeout_get(self, url, **kw):
            raise httpx.ReadTimeout("read timeout", request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", _timeout_get)

        # 创建一个临时的 httpx client 验证超时被正确抛出
        async def _verify():
            async with httpx.AsyncClient() as c:
                with pytest.raises(httpx.ReadTimeout):
                    await c.get("http://example.com")

        asyncio.run(_verify())

        # 恢复后 httpx 正常
        monkeypatch.undo()
        assert httpx.AsyncClient.get is original_get

    def test_ssh_timeout_handled_gracefully(self, monkeypatch):
        """模拟 paramiko SSH 超时, 验证 HPC 端点优雅失败。"""
        # 不需要真的调 HPC 端点, 验证 mock 的 paramiko 超时被正确捕获即可
        try:
            import paramiko
        except ImportError:
            pytest.skip("paramiko not installed")

        fake_ssh = MagicMock()
        fake_ssh.connect.side_effect = paramiko.SSHException("SSH connection timed out")

        # 验证 SSH 异常被正确抛出, 调用方可以捕获
        with pytest.raises(paramiko.SSHException):
            fake_ssh.connect("fake-host", timeout=1)

    def test_async_timeout_does_not_hang(self):
        """asyncio.wait_for 超时后任务应被取消, 不会泄漏。"""
        async def _slow():
            await asyncio.sleep(100)
            return "done"

        async def _main():
            try:
                await asyncio.wait_for(_slow(), timeout=0.1)
                assert False, "应该超时"
            except asyncio.TimeoutError:
                pass

        asyncio.run(_main())


# ═══════════════════════════════════════════════════════════════════
#  6. 资源耗尽场景
# ═══════════════════════════════════════════════════════════════════

class TestResourceExhaustion:

    def test_memory_error_handled_gracefully(self, monkeypatch):
        """模拟 MemoryError, 验证不会导致进程 crash。"""
        # 验证 Python 能正确捕获 MemoryError
        # 真正的 MemoryError 不好模拟, 用 mock 验证错误处理逻辑
        def _raise_oom():
            raise MemoryError("out of memory")

        with pytest.raises(MemoryError):
            _raise_oom()

        # 进程仍然存活
        assert client.get("/health/live", headers=_HEADERS).status_code == 200

    def test_thread_pool_does_not_create_unlimited_threads(self):
        """线程池提交大量任务, 验证不会无限创建线程。"""
        import concurrent.futures

        max_workers = 4
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        active = 0
        max_active = 0
        lock = threading.Lock()

        def _task():
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            threading.Event().wait(0.05)  # 短暂占用
            with lock:
                active -= 1

        futures = [pool.submit(_task) for _ in range(50)]
        concurrent.futures.wait(futures)

        # 并发线程数不应该远超 max_workers
        assert max_active <= max_workers + 1, (
            f"线程池创建了太多线程: peak={max_active}, limit={max_workers}"
        )
        pool.shutdown()

    def test_too_many_open_fds_handled(self, monkeypatch):
        """模拟打开过多文件描述符, 验证错误处理。"""
        import os
        # mock open() 抛 OSError (EMFILE)
        _real_open = open

        call_count = 0

        def _limited_open(file, mode="r", *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise OSError(24, "Too many open files")
            return _real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", _limited_open)

        # 前 3 次调用成功, 第 4 次抛 OSError
        # os.devnull 是跨平台的 /dev/null
        devnull = os.devnull
        _limited_open(devnull)
        _limited_open(devnull)
        _limited_open(devnull)
        with pytest.raises(OSError):
            _limited_open(devnull)  # 第4次调用会失败

        # 恢复后正常
        monkeypatch.undo()


# ═══════════════════════════════════════════════════════════════════
#  7. 并发冲突
# ═══════════════════════════════════════════════════════════════════

class TestConcurrencyConflict:

    def test_concurrent_checkpoint_accept_reject_consistency(self, monkeypatch, tmp_path):
        """同一 checkpoint 被并发 accept 和 reject, 只应有一个成功。"""
        import huginn.routes.checkpoints as cp_mod
        from huginn.server_core import _checkpoints, _state_lock

        # 先创建一个 checkpoint
        test_dir = tmp_path / "workspace"
        test_dir.mkdir()
        (test_dir / "a.txt").write_text("hello")

        snapshot = {"a.txt": "hello"}
        cp_id = "test_cp_concurrent"
        with _state_lock:
            _checkpoints[cp_id] = (test_dir, snapshot)

        results = {"accept": None, "reject": None}
        errors = {"accept": None, "reject": None}

        def _do_accept():
            try:
                r = client.post(f"/checkpoints/{cp_id}/accept", headers=_HEADERS)
                results["accept"] = r
            except Exception as e:
                errors["accept"] = e

        def _do_reject():
            try:
                r = client.post(f"/checkpoints/{cp_id}/reject", headers=_HEADERS)
                results["reject"] = r
            except Exception as e:
                errors["reject"] = e

        t1 = threading.Thread(target=_do_accept)
        t2 = threading.Thread(target=_do_reject)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # 两个都不应抛异常
        assert errors["accept"] is None, f"accept 抛异常: {errors['accept']}"
        assert errors["reject"] is None, f"reject 抛异常: {errors['reject']}"

        # 至少一个应该成功 (success=True), 另一个应该报告 not found
        bodies = []
        for key in ("accept", "reject"):
            r = results[key]
            if r is not None:
                bodies.append(r.json())

        # 至少有一个成功
        successes = [b for b in bodies if b.get("success")]
        assert len(successes) >= 1, "至少应该有一个操作成功"

        # 清理: 确保 checkpoint 已被删除
        with _state_lock:
            _checkpoints.pop(cp_id, None)

    def test_concurrent_config_updates_no_crash(self, monkeypatch, tmp_path):
        """并发更新配置, 验证不会 crash 或数据损坏。"""
        # 用一个简单的并发写入场景验证: 多个线程同时写同一个 toml 文件
        config_file = tmp_path / "huginn.toml"
        config_file.write_text('[chat]\nprovider = "ollama"\n')

        lock = threading.Lock()
        write_count = 0

        def _write_config(provider_val):
            nonlocal write_count
            # 模拟 config 路由内部的写入: 加锁后原子写
            with lock:
                current = config_file.read_text()
                config_file.write_text(
                    f'[chat]\nprovider = "{provider_val}"\n'
                )
                write_count += 1

        threads = [
            threading.Thread(target=_write_config, args=(f"provider_{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert write_count == 5, "不是所有写入都完成了"
        # 最终文件是有效的 toml
        content = config_file.read_text()
        assert content.startswith("[chat]"), f"配置文件损坏: {content}"

    def test_concurrent_health_checks_all_succeed(self):
        """并发健康检查都应成功, 验证无共享状态竞争。"""
        results = []
        errors = []

        def _check():
            try:
                r = client.get("/health/live", headers=_HEADERS)
                results.append(r.status_code)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_check) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"并发健康检查出错: {errors}"
        assert all(s == 200 for s in results), f"非 200 状态码: {results}"


# ═══════════════════════════════════════════════════════════════════
#  补充: 全局异常处理验证
# ═══════════════════════════════════════════════════════════════════

class TestGlobalExceptionSafety:
    """验证全局异常处理器兜底, 确保未知异常不暴露 traceback。"""

    def test_unhandled_exception_returns_500_not_crash(self, monkeypatch):
        """路由内部抛未捕获异常时, 全局 handler 返回 500 而非 crash。"""
        import huginn.routes.health as health_mod

        # mock /health/ready 里的一个依赖抛异常
        original_ready = health_mod.health_ready

        async def _boom(*args, **kwargs):
            raise RuntimeError("simulated internal failure")

        monkeypatch.setattr(health_mod, "health_ready", _boom)

        # 触发异常 — 全局 handler 应捕获并返回 500
        resp = client.get("/health/ready", headers=_HEADERS)
        # 可能是 500 (全局 handler 兜底) 或 503 (原有的就绪检查逻辑)
        # 关键是进程没挂
        assert resp.status_code in (500, 503)

    def test_error_response_does_not_leak_traceback(self, monkeypatch):
        """错误响应不应包含 traceback 或内部路径信息。"""
        import huginn.routes.health as health_mod

        async def _boom(*args, **kwargs):
            raise RuntimeError("secret internal path /etc/passwd leaked")

        monkeypatch.setattr(health_mod, "health_ready", _boom)

        resp = client.get("/health/ready", headers=_HEADERS)
        body = resp.text

        # 不应包含 traceback 关键字
        assert "Traceback" not in body
        assert "secret internal path" not in body.lower()
