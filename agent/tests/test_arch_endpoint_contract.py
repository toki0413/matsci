"""端点契约门禁：CLI 引用的 /v1 端点必须存在于后端 OpenAPI schema。

规则 (CI fast-fail)：
  R1  CLI 客户端（cli/src/http.rs）里引用的每个 `/v1/*` 路径，必须能在后端
      OpenAPI schema 中找到对应的路径模板（方法与路径都要对得上）。
      后端是权威，CLI 是客户端 —— 后端路由一旦改名/删除，CLI 若还引用旧路径
      就会静默 404，本门禁在 CI 里立刻拦截，防止"端点改了但没人知道"。
  R2  反向不强制：后端有大量端点，CLI 只消费其中一部分，不要求 CLI 覆盖全部。

设计：
  - 后端路径模板从 huginn.server 的 OpenAPI schema 提取（含 `{param}` 占位符）。
  - CLI 路径从 cli/src/http.rs 提取字符串字面量，按方法与后端模板双向校验。
  - CLI 全具体路径需能匹配后端含占位符的模板（如 /v1/agents/lead/chat/stream
    匹配 /v1/agents/{agent_id}/chat/stream）。
  - 只读扫描 + 导入 app 生成 schema，不改任何源码。

迁移须知：
  后端新增端点不用动本文件；后端改端点是破坏性变更，必须同步改 CLI http.rs，
  让本门禁保持绿。CLI 新增端点引用，其后端必须真实存在，否则 CI 变红。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# 仓库根 = 本文件 (agent/tests/test_arch_endpoint_contract.py) 向上二级
REPO_ROOT = Path(__file__).resolve().parents[2]


# ── CLI 侧：从 cli/src/http.rs 提取引用的 /v1 路径与方法 ──────────────
def _extract_cli_bound_endpoints(http_rs: Path) -> list[dict]:
    """提取 http.rs 里所有 `/v1/...` 字符串字面量及其推断的 HTTP 方法。

    返回 [{path, method}]。method 为 GET/POST 或 None(无法从上下文唯一推断)。
    """
    text = http_rs.read_text(encoding="utf-8", errors="ignore")

    # 只保留『纯路径』字面量：只含 /v1 + [A-Za-z0-9_-/{}]。
    # 这样能排除 .context("读取后端 /v1/tools 响应失败") 这类错误消息字符串，
    # 它们不是真实的端点引用。
    path_pat = re.compile(r'"(/v1/[A-Za-z0-9_\-/{}]+)"')
    endpoints: list[dict] = []
    for m in path_pat.finditer(text):
        path = m.group(1)
        method = _infer_method(text, m.start())
        endpoints.append({"path": path, "method": method})
    return endpoints


# 关键字的相对优先级：越靠后出现的越近，作为当前路径的调用者。
_METHOD_MARKERS = [
    ("ureq::get(", "GET"),
    ("ureq::post(", "POST"),
    ("ureq::head(", "HEAD"),
    ("ureq::delete(", "DELETE"),
    ("post_json(", "POST"),
    ("backend_available", None),  # url 只是探测，不参与契约
]


def _infer_method(text: str, pos: int) -> str | None:
    head = text[max(0, pos - 400):pos]
    best_pos = -1
    best_meth: str | None = None
    for marker, meth in _METHOD_MARKERS:
        idx = head.rfind(marker)
        if idx > best_pos:
            best_pos = idx
            best_meth = meth
    return best_meth


# ── 后端侧：从 OpenAPI schema 提取路径模板与方法 ─────────────────────
def _backend_endpoints() -> list[dict]:
    """从 huginn.server 的 OpenAPI schema 提取所有 /v1 路径模板与方法。"""
    from huginn.server import app  # noqa: PLC0415

    schema = app.openapi()
    out: list[dict] = []
    for template, methods in schema.get("paths", {}).items():
        if not template.startswith("/v1"):
            continue
        for meth in ("get", "post", "put", "delete", "patch"):
            if meth in methods:
                out.append({"path": template, "method": meth.upper()})
    return out


# ── 匹配：CLI 全具体路径 vs 后端含占位符模板 ─────────────────────────
def _segments(path: str) -> list[str]:
    return [s for s in path.strip("/").split("/") if s]


def _cli_path_matches_backend(cli_path: str, backend_template: str) -> bool:
    """CLI 具体路径能否匹配后端模板（{param} 处视为通配）。"""
    cli_seg = _segments(cli_path)
    be_seg = _segments(backend_template)
    if len(cli_seg) != len(be_seg):
        return False
    for c, b in zip(cli_seg, be_seg):
        if b.startswith("{") and b.endswith("}"):
            continue  # 占位符通配
        if c != b:
            return False
    return True


def _find_backend_match(cli_path: str, cli_method: str | None, backend: list[dict]) -> dict | None:
    """返回能匹配 cli_path 的后端端点；若 cli_method 已知，优先方法与路径都匹配。"""
    candidates = [b for b in backend if _cli_path_matches_backend(cli_path, b["path"])]
    if not candidates:
        return None
    if cli_method:
        for c in candidates:
            if c["method"] == cli_method:
                return c
        return candidates[0]  # 路径在但方法对不上 → 交给调用方判定
    return candidates[0]


# ── 测试 ────────────────────────────────────────────────────────────
def test_cli_referenced_endpoints_exist_in_backend():
    """R1：CLI 引用的每个 /v1 端点，后端必须存在（方法与路径都匹配）。"""
    http_rs = REPO_ROOT / "cli" / "src" / "http.rs"
    if not http_rs.exists():
        pytest.skip("cli/src/http.rs 不存在，跳过端点契约检查")

    cli_endpoints = _extract_cli_bound_endpoints(http_rs)
    assert cli_endpoints, "http.rs 里没找到任何 /v1 路径引用，契约门禁失效"

    backend = _backend_endpoints()
    assert backend, "后端 OpenAPI schema 里没有 /v1 端点，契约门禁失效"

    backend_keys = {b["path"] for b in backend}
    failures: list[str] = []
    for ep in cli_endpoints:
        match = _find_backend_match(ep["path"], ep["method"], backend)
        if match is None:
            failures.append(f"  不存在后端端点: {ep['path']} (method={ep['method']})")
            continue
        if ep["method"] and ep["method"] != match["method"]:
            failures.append(
                f"  方法不匹配: CLI {ep['method']} {ep['path']} "
                f"→ 后端 {match['method']} {match['path']}"
            )
        # 记录命中的模板，便于人工核对
        if match["path"] != ep["path"]:
            _ = backend_keys  # (占位符展开，路径不同属正常)

    assert not failures, (
        "CLI 引用了后端不存在的端点（或方法不匹配），违反端点契约。"
        "后端路由变更后，CLI 会静默 404。请同步修改 cli/src/http.rs。\n"
        + "\n".join(failures)
    )


def test_cli_endpoint_contract_self_test():
    """门禁自检：确认匹配逻辑能识别已知端点，防止门禁自己失效。"""
    # 已知正确的配对
    assert _cli_path_matches_backend(
        "/v1/agents/lead/chat/stream", "/v1/agents/{agent_id}/chat/stream"
    )
    assert _cli_path_matches_backend("/v1/tools", "/v1/tools")
    assert _cli_path_matches_backend("/v1/hpc/test", "/v1/hpc/test")
    # 已知错误配对
    assert not _cli_path_matches_backend("/v1/tools", "/v1/tools/{name}")
    assert not _cli_path_matches_backend("/v1/execute", "/v1/explore")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-o", "addopts="]))
