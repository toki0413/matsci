"""防漂移: ponytail 技术债清单 (docs/TECHNICAL_DEBT_PONYTAIL.md) 第 4/5 节
登记的每条 **open** S1/S2 条目, 都必须在代码里找到对应事实锚点.

双向防漂移:
- 代码改了但清单没删 → 锚点消失, 本测试红 (提醒销账).
- 清单写了但代码里没有该捷径 → 锚点不存在, 本测试红 (提醒清单是幻影).

只锚定"锚点子串"而非行号, 对行漂移免疫. 收敛后的条目 (第 6 节) 不在此列.
"""

from __future__ import annotations

import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[1]
_ROOT = _REPO.parent  # /workspace
_DOC = _REPO / "docs" / "TECHNICAL_DEBT_PONYTAIL.md"

# ID -> (相对仓库根的路径, 代码里的稳定锚点子串)
_OPEN_ANCHORS: dict[str, tuple[str, str]] = {
    "DEBT-S1-01": ("desktop/src/lib/api-client.ts", "localStorage.getItem('huginn:auth_token')"),
    "DEBT-S1-02": ("desktop/src/lib/api.ts", 'localStorage.getItem("huginn:api_key")'),
    "DEBT-S1-03": ("desktop/src/lib/config-store.ts", "stored in localStorage for simplicity"),
    "DEBT-S1-04": ("agent/huginn/tools/sci/numerical_tool.py", "eval with __builtins__={}"),
    "DEBT-S1-05": ("agent/huginn/security/code_act_sandbox.py", "__subclasses__()"),
    "DEBT-S2-01": ("agent/huginn/security/sandbox.py", "cmd /c 单层包裹"),
    "DEBT-S2-02": ("agent/huginn/security/docker_sandbox.py", "三档够覆盖科研工作流"),
    "DEBT-S2-03": ("agent/huginn/security/policy_engine.py", "we lose path scoping"),
}


def test_debt_registry_exists():
    assert _DOC.is_file(), "技术债清单 docs/TECHNICAL_DEBT_PONYTAIL.md 缺失"


def test_registry_lists_every_tracked_item():
    """清单正文必须逐条列出被锚定的 open 条目 ID (防止清单漏项)."""
    text = _DOC.read_text(encoding="utf-8")
    missing = [i for i in _OPEN_ANCHORS if i not in text]
    assert not missing, f"清单未列出以下被跟踪条目: {missing}"


def test_open_items_anchor_real_code():
    """每条 open 条目在代码里都有对应事实锚点."""
    failures = []
    for debt_id, (rel, anchor) in _OPEN_ANCHORS.items():
        f = _ROOT / rel
        if not f.is_file():
            failures.append(f"{debt_id}: 文件不存在 {rel}")
            continue
        if anchor not in f.read_text(encoding="utf-8"):
            failures.append(f"{debt_id}: {rel} 中找不到锚点 {anchor!r} (已修复? 请销账)")
    assert not failures, "技术债清单与代码漂移:\n" + "\n".join(failures)