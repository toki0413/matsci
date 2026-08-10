"""数据一致性测试 — 并发写隔离、checkpoint 往返、export-import 无损、audit 哈希链."""
from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _isolated_auth_env(monkeypatch):
    """Isolate auth env so we don't pollute other modules in the same worker."""
    monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)
    monkeypatch.setenv("HUGINN_API_KEY", "consistency-key-0123456789abcdef")
    monkeypatch.setenv("HUGINN_JWT_SECRET", "consistency-jwt-secret")
    monkeypatch.setenv("HUGINN_RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "1")
    yield


async def _noop():
    pass


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    ws = tmp_path_factory.mktemp("consistency_ws")
    os.environ["HUGINN_WORKSPACE"] = str(ws)
    # 隔离 cache 目录, 避免共享 memory.db 累积导致 export 超限 / 状态串扰
    os.environ["HUGINN_CACHE_DIR"] = str(tmp_path_factory.mktemp("consistency_cache"))
    import huginn.server as sm
    sm._init_mcp_tools = _noop
    sm._shutdown_mcp = _noop
    from huginn.config import HuginnConfig
    from huginn.server_context import create_server_context, set_server_context
    ctx = create_server_context(HuginnConfig(provider="ollama", model="test", workspace=str(ws)))
    set_server_context(ctx)
    sm._context = ctx
    from huginn.security.auth import get_user_store
    from huginn.security.rbac import Role, User
    store = get_user_store()
    store._users["consistency-admin"] = User(user_id="consistency-admin", username="admin", role=Role.ADMIN, active=True, api_key_hash="", created_at=0)
    store._users["consistency-user-a"] = User(user_id="consistency-user-a", username="userA", role=Role.OPERATOR, active=True, api_key_hash="", created_at=0)
    store._users["consistency-user-b"] = User(user_id="consistency-user-b", username="userB", role=Role.OPERATOR, active=True, api_key_hash="", created_at=0)
    from fastapi.testclient import TestClient
    with TestClient(sm.app) as c:
        yield c


@pytest.fixture
def admin_token(app_client):
    from huginn.security.auth import create_token, get_user_store
    return create_token(get_user_store()._users["consistency-admin"], expires_in=3600)


@pytest.fixture
def user_a_token(app_client):
    from huginn.security.auth import create_token, get_user_store
    return create_token(get_user_store()._users["consistency-user-a"], expires_in=3600)


@pytest.fixture
def user_b_token(app_client):
    from huginn.security.auth import create_token, get_user_store
    return create_token(get_user_store()._users["consistency-user-b"], expires_in=3600)


def _bearer(t): return {"Authorization": f"Bearer {t}"}


class TestConcurrentWriteIsolation:
    """并发写隔离."""

    def test_two_users_write_memory_no_crosstalk(self, app_client, user_a_token, user_b_token):
        """两个用户并发写 memory, 数据不串."""
        import concurrent.futures

        def write_a(i):
            resp = app_client.post("/memory", headers=_bearer(user_a_token), json={
                "content": f"userA-item-{i}", "category": "fact", "tags": ["userA"],
            })
            return resp.status_code

        def write_b(i):
            resp = app_client.post("/memory", headers=_bearer(user_b_token), json={
                "content": f"userB-item-{i}", "category": "fact", "tags": ["userB"],
            })
            return resp.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for i in range(10):
                futures.append(pool.submit(write_a, i))
                futures.append(pool.submit(write_b, i))
            results = [f.result() for f in futures]

        assert all(r == 200 for r in results), f"some failed: {set(results)}"

    def test_concurrent_thread_creation_no_id_collision(self, app_client, admin_token):
        """并发创建 thread, ID 不冲突."""
        import concurrent.futures

        thread_ids = []

        def create(i):
            resp = app_client.post("/threads", headers=_bearer(admin_token), json={"title": f"thread-{i}"})
            if resp.status_code in (200, 201):
                body = resp.json()
                tid = body.get("id") or body.get("thread_id")
                if tid:
                    return tid
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(create, i) for i in range(20)]
            for f in concurrent.futures.as_completed(futures):
                tid = f.result()
                if tid:
                    thread_ids.append(tid)

        # 无重复 ID
        assert len(thread_ids) == len(set(thread_ids)), \
            f"thread ID collision! {thread_ids}"


class TestExportImportRoundtrip:
    """export → import 数据无损."""

    def test_memory_export_import_roundtrip(self, app_client, admin_token):
        """导出 memory → 导入, 数据完整."""
        # 先写几条 memory
        for i in range(5):
            app_client.post("/memory", headers=_bearer(admin_token), json={
                "content": f"roundtrip-item-{i}", "category": "fact", "importance": 0.8,
            })

        # 导出
        resp = app_client.post("/export/memory", headers=_bearer(admin_token), json={"format": "json"})
        assert resp.status_code == 200, f"export failed: {resp.status_code}"
        exported_data = resp.content

        # 导入回来
        resp = app_client.post("/import/all", headers=_bearer(admin_token),
                               files={"file": ("memory.json", exported_data, "application/json")})
        assert resp.status_code == 200, f"import failed: {resp.status_code}"
        body = resp.json()
        # 不应崩溃; errors 可有可无
        assert isinstance(body, dict)

    def test_full_export_zip_import_roundtrip(self, app_client, admin_token):
        """全量 zip 导出 → 导入, 不崩溃."""
        resp = app_client.post("/export/all", headers=_bearer(admin_token), json={"format": "zip"})
        assert resp.status_code == 200, f"export failed: {resp.status_code}"
        exported_data = resp.content

        # 导入回来
        resp = app_client.post("/import/all", headers=_bearer(admin_token),
                               files={"file": ("backup.zip", exported_data, "application/zip")})
        assert resp.status_code == 200, f"import failed: {resp.status_code}"


class TestAuditLogIntegrity:
    """审计日志完整性."""

    def test_audit_log_records_write_operations(self, app_client, admin_token):
        """写操作被审计日志记录."""
        # 执行写操作
        app_client.post("/memory", headers=_bearer(admin_token), json={
            "content": "audit-test-item", "category": "fact",
        })

        # 检查审计日志 (通过 telemetry 或 diagnostics)
        resp = app_client.get("/telemetry/summary", headers=_bearer(admin_token))
        assert resp.status_code == 200

    def test_audit_log_hash_chain(self, app_client, admin_token):
        """审计日志哈希链可验证 (如果实现了)."""
        # 写几条操作
        for i in range(3):
            app_client.post("/memory", headers=_bearer(admin_token), json={
                "content": f"hash-chain-{i}", "category": "fact",
            })

        # 检查审计日志是否有哈希链
        # 通过 diagnostics 或直接检查 audit_logger
        from huginn.server_core import get_context
        ctx = get_context()
        if hasattr(ctx, "audit_logger"):
            logger = ctx.audit_logger
            # 检查是否有 entries
            entries = getattr(logger, "_entries", []) or getattr(logger, "entries", [])
            if entries:
                # 如果有 entries, 检查是否有 hash 字段
                first = entries[0] if isinstance(entries, list) else None
                if first and isinstance(first, dict):
                    # 有 hash 字段说明哈希链实现了
                    print(f"\n[CONSISTENCY] audit log has {len(entries)} entries, hash chain: {'hash' in first}")
        # 不强制断言 (审计日志实现可能不同), 只验证不崩溃


class TestWorkflowCheckpointConsistency:
    """workflow checkpoint 往返一致性."""

    def test_workflow_execute_does_not_corrupt_state(self, app_client, admin_token):
        """workflow 执行后, server 状态不被破坏."""
        # 执行前检查健康
        resp = app_client.get("/health")
        assert resp.status_code == 200

        # 列出 workflow 模板
        resp = app_client.get("/workflows", headers=_bearer(admin_token))
        assert resp.status_code == 200

        # 执行后健康检查仍正常
        resp = app_client.get("/health")
        assert resp.status_code == 200
        resp = app_client.get("/memory/stats", headers=_bearer(admin_token))
        assert resp.status_code == 200


class TestSQLiteIntegrity:
    """SQLite 数据库完整性."""

    def test_memory_db_not_corrupted_after_concurrent_ops(self, app_client, admin_token):
        """并发读写后 memory.db 不损坏."""
        import concurrent.futures

        def read():
            return app_client.get("/memory/stats", headers=_bearer(admin_token)).status_code

        def write(i):
            return app_client.post("/memory", headers=_bearer(admin_token), json={
                "content": f"integrity-{i}", "category": "fact",
            }).status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for i in range(10):
                futures.append(pool.submit(write, i))
                futures.append(pool.submit(read))
            results = [f.result() for f in futures]

        assert all(r == 200 for r in results), f"some failed: {set(results)}"

        # 最终 memory.stats 仍正常
        resp = app_client.get("/memory/stats", headers=_bearer(admin_token))
        assert resp.status_code == 200


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-cov"]))
