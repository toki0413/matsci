"""End-to-end test for Huginn using webbridge + local server.

Starts the Huginn backend (port 8000), serves the frontend (port 1420),
and uses webbridge to verify the full stack in a real browser.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
FRONTEND_DIR = PROJECT_ROOT.parent / "desktop" / "dist"
HUGINN_EXE = PROJECT_ROOT / "dist" / "huginn-agent" / "huginn-agent.exe"
WEBBRIDGE_URL = "http://127.0.0.1:10086/command"
SESSION = "huginn-e2e-test"


def webbridge(action: str, args: dict | None = None) -> dict:
    """Call the Kimi webbridge daemon."""
    body = json.dumps({
        "action": action,
        "args": args or {},
        "session": SESSION,
    }).encode("utf-8")
    req = urllib.request.Request(
        WEBBRIDGE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": str(e)}


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll until an HTTP server responds (any HTTP status = server is up)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except urllib.error.HTTPError as e:
            if e.code in (200, 404, 405, 401, 403, 500):
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_backend() -> subprocess.Popen:
    """Start Huginn serve in a subprocess."""
    if HUGINN_EXE.exists():
        cmd = [str(HUGINN_EXE), "serve", "--port", "8000"]
    else:
        cmd = [sys.executable, "-m", "huginn.cli", "serve", "--port", "8000"]
    print(f"[BACKEND] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def start_frontend() -> subprocess.Popen:
    """Serve the frontend dist directory on port 1420."""
    cmd = [sys.executable, "-m", "http.server", "1420", "--directory", str(FRONTEND_DIR)]
    print(f"[FRONTEND] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def test_backend_api() -> dict:
    """Test key Huginn API endpoints via HTTP."""
    results = {}
    endpoints = [
        ("GET", "http://localhost:8000/tools", "tools_list"),
        ("POST", "http://localhost:8000/plan", "plan"),
    ]
    for method, url, name in endpoints:
        try:
            if method == "GET":
                req = urllib.request.Request(url, method="GET")
            else:
                req = urllib.request.Request(
                    url,
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
            with urllib.request.urlopen(req, timeout=10) as resp:
                results[name] = {"status": resp.status, "body": resp.read(1024).decode()}
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


def test_webbridge_frontend() -> dict:
    """Use webbridge to load the frontend in a real browser and snapshot."""
    results = {}

    # 1. Navigate to frontend
    nav = webbridge("navigate", {"url": "http://localhost:1420", "newTab": True, "group_title": "Huginn E2E Test"})
    results["navigate"] = nav
    if not nav.get("ok"):
        return results

    time.sleep(2)  # Let React hydrate

    # 2. Snapshot the page
    snap = webbridge("snapshot")
    results["snapshot"] = snap

    # 3. Screenshot for visual verification
    screenshot = webbridge("screenshot", {"format": "png"})
    results["screenshot"] = screenshot

    # 4. Close the session tab
    close = webbridge("close_tab")
    results["close_tab"] = close

    return results


def main() -> int:
    print("=" * 60)
    print("Huginn End-to-End Test (WebBridge + Local Servers)")
    print("=" * 60)

    # Start servers
    backend = start_backend()
    frontend = start_frontend()

    try:
        # Wait for both servers
        print("[WAIT] Waiting for backend (localhost:8000)...")
        backend_ok = wait_for_server("http://localhost:8000", timeout=30)
        print(f"[WAIT] Backend: {'OK' if backend_ok else 'TIMEOUT'}")

        print("[WAIT] Waiting for frontend (localhost:1420)...")
        frontend_ok = wait_for_server("http://localhost:1420", timeout=15)
        print(f"[WAIT] Frontend: {'OK' if frontend_ok else 'TIMEOUT'}")

        if not backend_ok or not frontend_ok:
            print("[FAIL] Servers did not start in time.")
            return 1

        # API tests
        print("\n[TEST] Backend API endpoints...")
        api_results = test_backend_api()
        for name, res in api_results.items():
            status = res.get("status", "ERR")
            print(f"  {name}: {status}")

        # WebBridge tests
        print("\n[TEST] WebBridge frontend test...")
        wb_results = test_webbridge_frontend()
        for name, res in wb_results.items():
            ok = res.get("ok", False)
            print(f"  {name}: {'OK' if ok else 'FAIL'} - {res.get('error', '')[:100]}")

        # Summary
        all_ok = all(
            r.get("status", 0) in (200, 404) or r.get("ok", False)
            for r in list(api_results.values()) + list(wb_results.values())
        )
        print("\n" + "=" * 60)
        if all_ok:
            print("[PASS] All tests passed.")
        else:
            print("[FAIL] Some tests failed.")
        print("=" * 60)
        return 0 if all_ok else 1

    finally:
        print("[CLEANUP] Stopping servers...")
        backend.terminate()
        frontend.terminate()
        backend.wait(timeout=5)
        frontend.wait(timeout=5)
        # Close webbridge session
        webbridge("close_session")
        print("[CLEANUP] Done.")


if __name__ == "__main__":
    sys.exit(main())
