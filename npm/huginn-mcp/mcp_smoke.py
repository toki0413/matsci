#!/usr/bin/env python3
"""MCP stdio 冒烟测试：对 huginn capabilities-mcp / workflows-mcp 做完整握手."""
import json
import subprocess
import sys
import threading
import time

OKG, OKR = "\033[32m", "\033[31m"
END = "\033[0m"
LOG = None  # captured stderr lines


def reader(proc):
    global LOG
    LOG = proc.stderr.read().decode("utf-8", "replace")


def run_handshake(cmd, label, want_tools_gt=0):
    print(f"\n== {label}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    t = threading.Thread(target=reader, args=(proc,))
    t.daemon = True
    t.start()

    results = {}

    def exchange(msg_id, method, params):
        req = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        proc.stdin.write((json.dumps(req) + "\n").encode())
        proc.stdin.flush()
        deadline = time.time() + 90
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("id") == msg_id and "result" in obj:
                return obj["result"]
        return None

    try:
        init = exchange(1, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "smoke", "version": "1.0"},
        })
        results["serverInfo"] = init.get("serverInfo") if init else None
        results["protocolVersion"] = init.get("protocolVersion") if init else None

        proc.stdin.write((json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode())
        proc.stdin.flush()

        tools = exchange(3, "tools/list", {})
        results["tools"] = tools.get("tools", []) if isinstance(tools, dict) else []
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        time.sleep(0.3)
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(5)
        except Exception:
            pass

    si = results.get("serverInfo")
    pv = results.get("protocolVersion")
    n = len(results.get("tools") or [])
    print(f"  serverInfo  : {si}")
    print(f"  protocolVer : {pv}")
    print(f"  tools/list  : {n} 项")
    fname = (si or {}).get("name") if si else None
    ok = bool(fname) and pv and n > want_tools_gt
    print(f"  判定: {OKG}PASS{END}" if ok else f"  {OKR}FAIL{END}")
    if not ok and LOG:
        print(f"  [stderr] {LOG[:800]}")
    return ok


def main():
    which = ["capabilities-mcp", "workflows-mcp"]
    allok = True
    for w in which:
        try:
            ok = run_handshake(["huginn-agent", w], "huginn-agent " + w)
        except Exception as exc:
            print(f"  {OKR}异常: {exc}{END}")
            ok = False
        allok = allok and ok

    wrapper = "/workspace/npm/huginn-mcp/bin/capabilities-mcp.js"
    import os
    if os.path.exists(wrapper):
        try:
            ok = run_handshake(["node", wrapper], "node bin/capabilities-mcp.js (包装器)")
            allok = allok and ok
        except Exception as exc:
            print(f"  {OKR}包装器异常: {exc}{END}")
            allok = False

    print("\n" + ("全部 PASS OK" if allok else "存在 FAIL"))
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()