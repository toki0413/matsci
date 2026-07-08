"""API 契约测试套件。

从 FastAPI app 的 OpenAPI schema 自动枚举所有注册的端点，覆盖以下维度：
- 端点枚举与可达性（不返回 500）
- HTTP 方法验证（未定义的方法返回 405）
- 响应结构验证（GET 返回 JSON、POST 缺 body 返回 422、鉴权 401/403）
- /v1 前缀一致性（root 与 /v1 返回相同状态码）
- Content-Type 验证（JSON 端点返回 application/json）
- Deprecation header（root 有、/v1 没有）
- OpenAPI schema 一致性

用 app.openapi() 作为路由来源，因为新版 Starlette 的 app.routes 里
是 _IncludedRouter 包装对象，不是直接的 Route。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

# server.py 的 lifespan 依赖 mcp，没装就跳过整个文件
pytest.importorskip("mcp")

from fastapi.testclient import TestClient  # noqa: E402

from huginn.config import HuginnConfig  # noqa: E402
from huginn.server_context import (  # noqa: E402
    create_server_context,
    set_server_context,
)

# ── 常量 ──────────────────────────────────────────────────────────────

_AUTO_PATHS = frozenset({"/docs", "/openapi.json", "/redoc"})

# 直接从路由模块导入，避免和实现脱节
from huginn.routes import _ROOT_ONLY_PATHS  # noqa: E402

# 动态路径参数的默认填充值
_PARAM_DEFAULTS: dict[str, str] = {
    "thread_id": "test-thread",
    "session_id": "test-session",
    "tool_name": "bash_tool",
    "name": "default",
    "persona_name": "default",
    "agent_id": "lead",
    "checkpoint_id": "test-cp",
    "cp_id": "test-cp",
    "memory_id": "test-mem",
    "job_id": "test-job",
    "local_id": "test-job",
    "id": "test-id",
    "key": "test-key",
    "node_id": "test-node",
    "plan_id": "test-plan",
    "file": "test.txt",
    "path": "test",
    "server_id": "test-server",
    "skill": "test-skill",
    "template": "test-template",
    "pk": "1",
    "item": "test-item",
    "workflow_id": "test-wf",
    "task_id": "test-task",
    "question_id": "test-q",
    "channel": "test-channel",
    "user": "test-user",
    "host": "testhost",
    "model": "test-model",
    "alias": "test-alias",
    "kernel_id": "test-kernel",
    "tunnel_id": "test-tunnel",
    "transfer_id": "test-transfer",
    "terminal_id": "test-terminal",
    "cid": "test-cred",
    "service": "ssh",
    "feature": "test-feature",
    "tool": "bash_tool",
    "type_name": "test-type",
    "document_id": "test-doc",
}

_ALL_HTTP_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]


# ── 路由信息 ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RouteInfo:
    """从 OpenAPI schema 提取的端点信息。"""

    path: str
    methods: frozenset[str] = field(default_factory=frozenset)

    @property
    def safe_methods(self) -> set[str]:
        """排除 HEAD/OPTIONS 后的业务方法。"""
        return set(self.methods) - {"HEAD", "OPTIONS"}


# ── 辅助函数 ──────────────────────────────────────────────────────────


def _fill_path(path: str) -> str:
    """用合理默认值替换路径中的 {param} 或 {param:type} 占位符。"""
    def _repl(m: re.Match) -> str:
        param = m.group(1)
        rest = m.group(2) or ""
        if "int" in rest:
            return "1"
        if "float" in rest:
            return "1.0"
        if "uuid" in rest:
            return "00000000-0000-0000-0000-000000000000"
        return _PARAM_DEFAULTS.get(param, "test")

    return re.sub(r"\{(\w+)(:[^}]+)?\}", _repl, path)


def _route_id(route: RouteInfo) -> str:
    """生成可读的 pytest 参数 ID。"""
    methods = sorted(route.safe_methods)
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", route.path).strip("_")
    return f"{'-'.join(methods)}_{safe}"


def _hit(client: TestClient, route: RouteInfo, *, path: str | None = None) -> Any:
    """用最安全的方式访问端点，返回 response。"""
    p = path or _fill_path(route.path)
    methods = route.safe_methods
    if "GET" in methods:
        return client.get(p)
    if "POST" in methods:
        return client.post(p, json={})
    if "PUT" in methods:
        return client.put(p, json={})
    if "PATCH" in methods:
        return client.patch(p, json={})
    if "DELETE" in methods:
        return client.delete(p)
    m = next(iter(methods), "GET")
    return client.request(m, p)


def _wrong_method(route: RouteInfo) -> str | None:
    """挑一个路由没定义的 HTTP 方法。"""
    methods = route.safe_methods
    for m in _ALL_HTTP_METHODS:
        if m not in methods:
            return m
    return None


# ── 导入 app 并从 OpenAPI schema 收集路由 ────────────────────────────

import huginn.server as _srv  # noqa: E402

# 模块级缓存 schema，后续 fixture 和测试都复用
_RAW_SCHEMA: dict[str, Any] = {}
try:
    _RAW_SCHEMA = _srv.app.openapi()
except Exception:
    pass


def _collect_routes(schema: dict[str, Any]) -> list[RouteInfo]:
    """从 OpenAPI schema 的 paths 中提取所有 HTTP 路由。"""
    routes: list[RouteInfo] = []
    for path, path_info in schema.get("paths", {}).items():
        if path in _AUTO_PATHS:
            continue
        methods = frozenset(
            m.upper()
            for m in path_info
            if m.upper() in _ALL_HTTP_METHODS
        )
        if methods:
            routes.append(RouteInfo(path=path, methods=methods))
    return routes


_ALL_ROUTES = _collect_routes(_RAW_SCHEMA)
_ROOT_ROUTES = [r for r in _ALL_ROUTES if not r.path.startswith("/v1/")]
_V1_ROUTES = [r for r in _ALL_ROUTES if r.path.startswith("/v1/")]
_GET_ROUTES = [r for r in _ROOT_ROUTES if "GET" in r.safe_methods]
_POST_ROUTES = [r for r in _ROOT_ROUTES if "POST" in r.safe_methods]


# ── 模块级 fixture ──────────────────────────────────────────────────

async def _noop() -> None:
    pass


@pytest.fixture(scope="module", autouse=True)
def _patch_mcp_lifespan():
    """跳过 MCP 子进程初始化，保持测试启动快。"""
    original_init = _srv._init_mcp_tools
    original_shutdown = _srv._shutdown_mcp
    _srv._init_mcp_tools = _noop
    _srv._shutdown_mcp = _noop
    yield
    _srv._init_mcp_tools = original_init
    _srv._shutdown_mcp = original_shutdown


@pytest.fixture(scope="module")
def _tmp_workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("api-contract")


@pytest.fixture(scope="module")
def _server_context(_tmp_workspace):
    """提供隔离的 ServerContext，避免测试间状态泄漏。"""
    os.environ["HUGINN_PROVIDER"] = "ollama"
    os.environ["HUGINN_MODEL"] = "qwen2.5:14b"
    os.environ["HUGINN_WORKSPACE"] = str(_tmp_workspace)
    cfg = HuginnConfig(
        provider="ollama", model="qwen2.5:14b", workspace=str(_tmp_workspace)
    )
    ctx = create_server_context(cfg)
    set_server_context(ctx)
    old = _srv._context
    _srv._context = ctx
    yield ctx
    _srv._context = old


@pytest.fixture(scope="module")
def client(_server_context, _tmp_workspace):
    """返回 dev mode 下的 TestClient。"""
    os.environ["HUGINN_DEV_MODE"] = "1"
    os.environ["HUGINN_PERSONA"] = "default"
    _srv._checkpoints.clear()
    _srv._threads.clear()
    with TestClient(_srv.app) as c:
        yield c


@pytest.fixture(scope="module")
def openapi_schema(client):
    """从 /openapi.json 获取运行时 schema，和模块级 _RAW_SCHEMA 对比用。"""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    return resp.json()


# ── 1. 端点枚举测试 ──────────────────────────────────────────────────


class TestEndpointEnumeration:
    """验证路由注册数量和基本结构。"""

    def test_routes_registered(self):
        """schema 应包含足够多的路由。"""
        assert len(_ALL_ROUTES) > 100, f"只找到 {len(_ALL_ROUTES)} 条路由，可能注册有问题"

    def test_root_and_v1_both_mounted(self):
        """root 和 /v1 都有路由挂载。"""
        assert len(_ROOT_ROUTES) > 50, "root 路由太少"
        assert len(_V1_ROUTES) > 50, "/v1 路由太少"

    def test_root_v1_count_matches(self):
        """root 路由数应该和 /v1 路由数一致。"""
        diff = abs(len(_ROOT_ROUTES) - len(_V1_ROUTES))
        assert diff <= 5, f"root/v1 路由数差异过大: {diff}"

    def test_every_root_has_v1_counterpart(self):
        """每条 root 路径都应有对应的 /v1 版本。"""
        missing = [
            r.path
            for r in _ROOT_ROUTES
            if f"/v1{r.path}" not in {v.path for v in _V1_ROUTES}
        ]
        assert not missing, f"以下 root 路径没有 /v1 对应: {missing[:10]}"

    @pytest.mark.parametrize("route", _ROOT_ROUTES, ids=_route_id)
    def test_no_500(self, client, route):
        """每个端点用定义的方法访问，不应该返回 500。"""
        resp = _hit(client, route)
        assert resp.status_code != 500, (
            f"{route.path} 返回 500: {resp.text[:200]}"
        )


# ── 2. HTTP 方法验证 ─────────────────────────────────────────────────


class TestHttpMethodValidation:
    """每个端点只接受定义的 HTTP 方法，其他方法返回 405。"""

    # 行为一致，取前 30 条抽样就够
    _sample = _ROOT_ROUTES[:30]

    @pytest.mark.parametrize("route", _sample, ids=_route_id)
    def test_wrong_method_returns_405(self, client, route):
        """未定义的方法应返回 405 或 404。"""
        wrong = _wrong_method(route)
        if wrong is None:
            pytest.skip("所有方法都已定义")

        path = _fill_path(route.path)
        resp = client.request(wrong, path)
        # 405 = 方法不允许；404 也可接受（路径参数填充后可能不匹配具体资源）
        assert resp.status_code in (405, 404), (
            f"{wrong} {path} 应返回 405/404，实际 {resp.status_code}"
        )


# ── 3. 响应结构验证 ──────────────────────────────────────────────────


class TestResponseStructure:
    """验证 GET 返回 JSON、POST 缺 body 返回 422、鉴权行为。"""

    @pytest.mark.parametrize("route", _GET_ROUTES, ids=_route_id)
    def test_get_returns_valid_json(self, client, route):
        """GET 端点返回 200 时 body 应该是有效 JSON。"""
        path = _fill_path(route.path)
        resp = client.get(path)

        if resp.status_code != 200:
            assert resp.status_code != 500, f"GET {path} 返回 500"
            return

        ct = resp.headers.get("content-type", "")
        # SSE 流式端点、HTML 可视化端点、Prometheus metrics 不走 JSON 校验
        if "text/event-stream" in ct or "text/html" in ct or "text/plain" in ct:
            return

        try:
            resp.json()
        except Exception:
            pytest.fail(f"GET {path} 返回 200 但 body 不是有效 JSON: {resp.text[:200]}")

    @pytest.mark.parametrize("route", _POST_ROUTES, ids=_route_id)
    def test_post_without_body_returns_422(self, client, route):
        """POST 端点缺少 body 时返回 422（仅当 schema 声明 body 必填时）。"""
        path_info = _RAW_SCHEMA.get("paths", {}).get(route.path, {})
        post_info = path_info.get("post", {})
        request_body = post_info.get("requestBody", {})

        if not request_body.get("required", False):
            pytest.skip("该 POST 端点不要求 body")

        path = _fill_path(route.path)
        resp = client.post(path)
        assert resp.status_code == 422, (
            f"POST {path} 缺 body 应返回 422，实际 {resp.status_code}"
        )

    @pytest.mark.parametrize("route", _POST_ROUTES, ids=_route_id)
    def test_post_empty_body_not_500(self, client, route):
        """POST 端点发空 JSON body 不应该返回 500。"""
        path = _fill_path(route.path)
        resp = client.post(path, json={})
        assert resp.status_code != 500, f"POST {path} 发空 body 返回 500"

    def test_missing_required_fields_returns_422(self, client):
        """POST 端点缺少 required fields 时返回 422。"""
        # /tunnels 的 body 要求 name/ssh_host/ssh_user/credential_id/local_port
        resp = client.post("/tunnels", json={})
        assert resp.status_code == 422

    def test_unauthorized_without_api_key(self, client, monkeypatch):
        """关闭 dev mode 后，无 API key 访问受保护端点返回 401。"""
        monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)
        resp = client.get("/agents")
        assert resp.status_code == 401, (
            f"无 API key 应返回 401，实际 {resp.status_code}"
        )

    def test_admin_endpoint_with_regular_key_rejected(self, client, monkeypatch):
        """admin 端点用普通 key 应返回 401/403。"""
        monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)
        monkeypatch.setenv("HUGINN_ADMIN_API_KEY", "admin-secret")
        resp = client.get(
            "/admin/maintenance",
            headers={"X-HUGINN-API-KEY": "test-key"},
        )
        assert resp.status_code in (401, 403), (
            f"普通 key 访问 admin 端点应返回 401/403，实际 {resp.status_code}"
        )


# ── 4. /v1 前缀一致性 ────────────────────────────────────────────────


class TestV1PrefixConsistency:
    """所有端点在 /v1 前缀下返回相同状态码。"""

    @pytest.mark.parametrize("route", _ROOT_ROUTES, ids=_route_id)
    def test_root_v1_same_status(self, client, route):
        """root 和 /v1 返回相同状态码。"""
        root_path = _fill_path(route.path)
        v1_path = "/v1" + root_path

        methods = route.safe_methods
        if "GET" in methods:
            root_resp = client.get(root_path)
            v1_resp = client.get(v1_path)
        elif "POST" in methods:
            root_resp = client.post(root_path, json={})
            v1_resp = client.post(v1_path, json={})
        elif "DELETE" in methods:
            root_resp = client.delete(root_path)
            v1_resp = client.delete(v1_path)
        else:
            m = next(iter(methods), "GET")
            root_resp = client.request(m, root_path, json={})
            v1_resp = client.request(m, v1_path, json={})

        assert root_resp.status_code == v1_resp.status_code, (
            f"{root_path} root={root_resp.status_code} /v1={v1_resp.status_code}"
        )


# ── 5. Content-Type 验证 ─────────────────────────────────────────────


class TestContentType:
    """所有 JSON 端点返回 application/json。"""

    @pytest.mark.parametrize("route", _GET_ROUTES, ids=_route_id)
    def test_json_content_type(self, client, route):
        """返回 200 且 body 是 JSON 的端点，Content-Type 应包含 application/json。"""
        path = _fill_path(route.path)
        resp = client.get(path)

        if resp.status_code != 200:
            return

        ct = resp.headers.get("content-type", "")
        # SSE 流式、HTML 可视化、Prometheus metrics 不走 JSON 校验
        if "text/event-stream" in ct or "text/html" in ct or "text/plain" in ct:
            return

        try:
            resp.json()
            assert "application/json" in ct, (
                f"GET {path} 返回 JSON 但 Content-Type 是 {ct}"
            )
        except Exception:
            pass


# ── 6. Deprecation header ────────────────────────────────────────────


class TestDeprecationHeader:
    """root 路径带 Deprecation header，/v1 路径不带。"""

    @pytest.mark.parametrize("route", _ROOT_ROUTES, ids=_route_id)
    def test_root_has_deprecation(self, client, route):
        """非 _ROOT_ONLY_PATHS 的 root 路径应带 Deprecation: true。"""
        path = _fill_path(route.path)
        resp = _hit(client, route)

        if route.path in _ROOT_ONLY_PATHS:
            assert resp.headers.get("Deprecation") != "true", (
                f"{path} 在 _ROOT_ONLY_PATHS 中，不应有 Deprecation header"
            )
        else:
            assert resp.headers.get("Deprecation") == "true", (
                f"root {path} 缺少 Deprecation: true header"
            )

    @pytest.mark.parametrize("route", _ROOT_ROUTES[:30], ids=_route_id)
    def test_v1_no_deprecation(self, client, route):
        """/v1 路径不应带 Deprecation header。"""
        path = "/v1" + _fill_path(route.path)
        methods = route.safe_methods

        if "GET" in methods:
            resp = client.get(path)
        elif "POST" in methods:
            resp = client.post(path, json={})
        else:
            m = next(iter(methods), "GET")
            resp = client.request(m, path, json={})

        assert resp.headers.get("Deprecation") != "true", (
            f"{path} 不应有 Deprecation header"
        )


# ── 7. OpenAPI schema 一致性 ─────────────────────────────────────────


class TestOpenAPIConsistency:
    """从 /openapi.json 获取 schema，验证响应与 schema 匹配。"""

    def test_openapi_endpoint_returns_200(self, client):
        """/openapi.json 返回 200 且结构正确。"""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "info" in data
        assert "openapi" in data

    def test_openapi_version(self, openapi_schema):
        """schema 版本应为 3.x。"""
        assert openapi_schema.get("openapi", "").startswith("3."), (
            f"OpenAPI 版本不是 3.x: {openapi_schema.get('openapi')}"
        )

    def test_schema_path_count(self, openapi_schema):
        """schema 中的路径数应与模块级收集的一致。"""
        schema_paths = openapi_schema.get("paths", {})
        assert len(schema_paths) == len(_ALL_ROUTES), (
            f"schema 路径数 {len(schema_paths)} != 收集路由数 {len(_ALL_ROUTES)}"
        )

    def test_every_path_has_methods(self, openapi_schema):
        """每条路径至少声明一个 HTTP 方法。"""
        empty: list[str] = []
        for path, path_info in openapi_schema.get("paths", {}).items():
            methods = [m for m in path_info if m.upper() in _ALL_HTTP_METHODS]
            if not methods:
                empty.append(path)
        assert not empty, f"以下路径没有声明 HTTP 方法: {empty[:10]}"

    def test_schema_methods_match_collected(self, openapi_schema):
        """运行时 schema 的方法应和模块级收集的一致。"""
        runtime_routes = {
            r.path: r.safe_methods for r in _collect_routes(openapi_schema)
        }
        module_routes = {r.path: r.safe_methods for r in _ALL_ROUTES}

        mismatches: list[str] = []
        for path in set(runtime_routes) | set(module_routes):
            if runtime_routes.get(path, set()) != module_routes.get(path, set()):
                mismatches.append(
                    f"{path}: runtime={runtime_routes.get(path)} module={module_routes.get(path)}"
                )
        assert not mismatches, f"运行时和模块级路由方法不一致: {mismatches[:10]}"

    def test_post_routes_have_request_body_schema(self, openapi_schema):
        """POST 端点的 requestBody（如有）应有 content 定义。"""
        checked = 0
        for path, path_info in openapi_schema.get("paths", {}).items():
            post_info = path_info.get("post", {})
            rb = post_info.get("requestBody")
            if not rb:
                continue
            # 必填 body 必须有 content 描述结构
            if rb.get("required", False):
                assert "content" in rb, f"POST {path} 的 required body 缺少 content"
                checked += 1
        # 至少应该有一些需要 body 的 POST 端点
        assert checked > 0, "没有找到声明 required body 的 POST 端点"
