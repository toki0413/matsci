"""User-facing interaction tests for the Huginn API.

These cover the REST + WebSocket surface the React desktop app talks to:
health probes, the tools/skills panel, credential CRUD, export/share,
bot status, the 3D viewer element table, and the chat websocket.

We only assert that endpoints *respond* (no 500). Whether they return
perfect data depends on the agent being fully initialised, which is not
guaranteed under the test fixture. Dev mode (set in conftest.py) bypasses
API-key / admin-key checks so we can hit every route unauthenticated.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from huginn.server import app

# One shared client covers both HTTP and WS — TestClient supports both,
# and reusing it avoids spinning up a second lifespan.
client = TestClient(app)

WS_PATH = "/v1/ws/agent"


class TestHealthEndpoints:
    """Liveness / readiness / diagnostics — the probes k8s and the
    desktop status bar poll."""

    def test_health_live(self):
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert "status" in resp.json()

    def test_health_ready(self):
        # Readiness can legitimately fail (503) when a dependency is down,
        # e.g. no LLM provider configured in CI. Either way the per-check
        # status lives under "checks".
        resp = client.get("/health/ready")
        assert resp.status_code in (200, 503)
        body = resp.json()
        assert "checks" in body or "status" in body

    def test_diagnostics(self):
        resp = client.get("/diagnostics")
        assert resp.status_code == 200
        # diagnostics aggregates a bunch of subsystems; just make sure we
        # got a structured report back, not an empty 200.
        assert isinstance(resp.json(), dict)


class TestToolAndSkillPanel:
    """The left-panel lists in the desktop UI."""

    def test_list_tools(self):
        resp = client.get("/tools")
        assert resp.status_code < 500
        body = resp.json()
        # ToolRegistry may be empty if presets haven't loaded yet, but the
        # shape is always a list.
        assert isinstance(body, (list, dict))

    def test_list_skills(self):
        resp = client.get("/skills")
        assert resp.status_code < 500
        body = resp.json()
        assert isinstance(body, (list, dict))

    def test_get_skill_detail(self):
        # There's no /skills/{name} detail route today — the panel hits it
        # optimistically, so a 404 is fine. We just need to confirm the
        # server doesn't blow up with a 500.
        resp = client.get("/skills/literature_search")
        assert resp.status_code < 500


class TestCredentialManagement:
    """The settings → credentials screen.

    Service API keys go through POST /credentials/{service} (service name
    in the path, not the body), so we hit that rather than the SSH/LLM
    CRUD at POST /credentials.
    """

    def test_create_credential(self):
        resp = client.post("/credentials/openai", json={"api_key": "sk-test"})
        assert resp.status_code in (200, 201)

    def test_list_credentials(self):
        resp = client.get("/credentials")
        assert resp.status_code < 500
        assert isinstance(resp.json(), dict)

    def test_delete_credential(self):
        # 200 if we created it above (or in a prior run), 404-ish otherwise.
        # The handler actually returns 200 with success=False when missing,
        # so < 500 is the robust bar.
        resp = client.delete("/credentials/openai")
        assert resp.status_code < 500


class TestExportShare:
    """Memory / knowledge export buttons.

    Both routes are POST (they accept a format/include body), not GET —
    hitting them with GET returns 405, so we POST an empty body to
    exercise the real download path.
    """

    def test_export_memory(self):
        resp = client.post("/export/memory", json={})
        # Success is either a 200 with a download header, or a 200 with a
        # JSON error body if memory isn't initialised. Anything else means
        # the route itself broke.
        assert resp.status_code < 500
        if resp.status_code == 200:
            cd = resp.headers.get("content-disposition", "")
            # Download responses carry an attachment filename; JSON error
            # responses don't. Either is acceptable.
            assert cd.startswith("attachment") or "application" in resp.headers.get(
                "content-type", ""
            )

    def test_export_knowledge(self):
        resp = client.post("/export/knowledge", json={})
        assert resp.status_code < 500
        if resp.status_code == 200:
            cd = resp.headers.get("content-disposition", "")
            assert cd.startswith("attachment") or "application" in resp.headers.get(
                "content-type", ""
            )


class TestBotManagement:
    """The bot bridge status indicator."""

    def test_bot_status(self):
        # require_admin_key is short-circuited by HUGINN_DEV_MODE=1.
        resp = client.get("/bot/status")
        assert resp.status_code == 200
        body = resp.json()
        # When no bridge is running we still get a minimal status dict.
        assert "running" in body or "status" in body


class TestViewer3D:
    """The 3D structure viewer's element table fetch on boot."""

    def test_element_table(self):
        resp = client.get("/viewer3d/elements")
        assert resp.status_code == 200
        body = resp.json()
        assert "elements" in body
        assert isinstance(body["elements"], list)
        # Sanity: hydrogen should be in any element table we ship.
        symbols = {e.get("symbol") for e in body["elements"]}
        assert "H" in symbols


class TestWebSocketConnection:
    """Smoke test for the chat websocket the desktop app opens."""

    def test_ws_connect_and_disconnect(self):
        # Ping/pong is the lightest round-trip — it doesn't need the LLM
        # agent or a configured thread, so it isolates the transport from
        # the agent layer. Sending a "user_input" here would block on a
        # real model call; sending an unhandled type would hang until the
        # 30s heartbeat fires. Ping is the right probe.
        with client.websocket_connect(WS_PATH) as websocket:
            websocket.send_text(json.dumps({"type": "ping"}))
            data = websocket.receive_text()
            msg = json.loads(data)
            assert msg.get("type") == "pong"
        # Exiting the context manager closes the socket cleanly — if the
        # server held the connection open we'd hang on context exit.
