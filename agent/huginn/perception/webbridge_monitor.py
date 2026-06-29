"""WebBridge monitor — monitor browser state via Kimi WebBridge daemon.

Connects to localhost:10086 to capture browser tab state, DOM snapshots,
and user interactions. Used for multi-modal perception layer.

Graceful degradation: if WebBridge is not running, silently disables.

Usage:
    monitor = WebBridgeMonitor()
    if monitor.available():
        snapshot = monitor.snapshot()
        print(snapshot.url, snapshot.title)
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class BrowserSnapshot:
    """Snapshot of a browser tab."""
    url: str
    title: str
    tree: str = ""
    active: bool = False


class WebBridgeMonitor:
    """Monitor browser state via WebBridge daemon.

    URL: http://127.0.0.1:10086
    No persistent connection. Polling-based.
    """

    ENDPOINT = "http://127.0.0.1:10086"
    TIMEOUT = 5.0

    def __init__(self) -> None:
        self._available: bool | None = None

    def available(self) -> bool:
        """Check if WebBridge daemon is running."""
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(
                f"{self.ENDPOINT}/command",
                data=b'{}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
        return self._available

    def _post(self, action: str, args: dict[str, Any] | None = None, session: str = "huginn") -> dict[str, Any]:
        """Send a command to WebBridge."""
        payload = {
            "action": action,
            "args": args or {},
            "session": session,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ENDPOINT}/command",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_tabs(self) -> list[BrowserSnapshot]:
        """List all browser tabs."""
        if not self.available():
            return []
        result = self._post("list_tabs")
        tabs = result.get("tabs", [])
        return [
            BrowserSnapshot(
                url=t.get("url", ""),
                title=t.get("title", ""),
                active=t.get("active", False),
            )
            for t in tabs
        ]

    def snapshot(self, url: str | None = None) -> BrowserSnapshot | None:
        """Take a snapshot of the current or specified tab."""
        if not self.available():
            return None
        try:
            if url:
                self._post("navigate", {"url": url})
            result = self._post("snapshot")
            return BrowserSnapshot(
                url=result.get("url", ""),
                title=result.get("title", ""),
                tree=result.get("tree", ""),
                active=True,
            )
        except Exception:
            return None

    def screenshot(self, path: str | None = None) -> str | None:
        """Take a screenshot, return file path."""
        if not self.available():
            return None
        args = {}
        if path:
            args["path"] = path
        result = self._post("screenshot", args)
        return result.get("path")
