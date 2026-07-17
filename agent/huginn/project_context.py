"""Project-level context loader.

Reads a `.huginn.md` file from the workspace root and injects it into the
agent system prompt. Falls back to `AGENTS.md` if present.

v7: workspace root 通过 .git marker 向上递归定位 (跟 Claude Code 一致),
然后收集 root → cwd 路径上所有 AGENTS.md 按序拼接, 子目录覆盖父目录.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

DEFAULT_FILENAME = ".huginn.md"
FALLBACK_FILENAME = "AGENTS.md"


def _find_workspace_root(start: Path) -> Path:
    """从 start 向上找 .git 文件或目录, 找到就返回其父目录; 找不到回退 start.

    ponytail: 20 层上限防止死循环, 实际项目深度很少超过 10.
    """
    p = start.resolve()
    for _ in range(20):
        if (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start.resolve()


def _find_context_file(workspace: Path) -> Path | None:
    """Return the primary project context file (.huginn.md), root only."""
    root = _find_workspace_root(workspace)
    primary = root / DEFAULT_FILENAME
    if primary.exists() and primary.is_file():
        return primary
    return None


def _collect_agents_md(workspace: Path) -> list[Path]:
    """收集 root → workspace 路径上每一级目录里的 AGENTS.md, 按从浅到深排序.

    子目录的指令在列表末尾, 调用方拼接后放在 system prompt 后部, 让 LLM
    优先采纳更具体 (子目录) 的指令. 如果 workspace 本身就是 root, 最多 1 个.
    """
    root = _find_workspace_root(workspace)
    target = workspace.resolve()
    # 收集 root, root/child, ..., target 这条链上每一级
    chain: list[Path] = []
    p = target
    while True:
        chain.append(p)
        if p == root or p.parent == p:
            break
        p = p.parent
    chain.reverse()  # root 在前, target 在末尾
    return [
        p / FALLBACK_FILENAME
        for p in chain
        if (p / FALLBACK_FILENAME).is_file()
    ]


def load_project_context(workspace: str | Path) -> str:
    """Load project context markdown from the workspace.

    优先级: .huginn.md (root 独占) > AGENTS.md (root → cwd 沿途收集拼接).
    两者都存在时, .huginn.md 完全覆盖 AGENTS.md (用户显式覆盖意图).
    """
    ws = Path(workspace)
    primary = _find_context_file(ws)
    if primary is not None:
        try:
            return primary.read_text(encoding="utf-8")
        except Exception:
            return ""
    # 没有 .huginn.md, 拼接所有 AGENTS.md
    parts: list[str] = []
    for path in _collect_agents_md(ws):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if text:
            parts.append(f"<!-- {path.name} @ {path.parent.name or path.parent} -->\n{text}")
    return "\n\n---\n\n".join(parts)


def save_project_context(workspace: str | Path, content: str) -> dict:
    """Write project context markdown to `.huginn.md` (in workspace, not root)."""
    path = Path(workspace) / DEFAULT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "bytes": len(content.encode("utf-8"))}


def project_context_path(workspace: str | Path) -> Path:
    """Return the primary project context file path (root .huginn.md, fallback workspace)."""
    root = _find_workspace_root(Path(workspace))
    primary = root / DEFAULT_FILENAME
    if primary.exists() and primary.is_file():
        return primary
    return Path(workspace) / DEFAULT_FILENAME


def context_source(workspace: str | Path) -> Literal[".huginn.md", "AGENTS.md", "none"]:
    """Indicate which context file is being used."""
    ws = Path(workspace)
    if _find_context_file(ws) is not None:
        return ".huginn.md"
    if _collect_agents_md(ws):
        return "AGENTS.md"
    return "none"


if __name__ == "__main__":
    # self-check: root 检测 + 收集逻辑
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "proj"
        sub = root / "pkg" / "sub"
        sub.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "AGENTS.md").write_text("# root rules\nuse python", encoding="utf-8")
        (sub / "AGENTS.md").write_text("# sub rules\nuse strict typing", encoding="utf-8")
        # case 1: 从 sub 调, 应该拿到 root + sub 两个
        text = load_project_context(sub)
        assert "root rules" in text and "sub rules" in text, f"收集失败: {text!r}"
        assert text.find("root rules") < text.find("sub rules"), "顺序错: root 应在前"
        assert context_source(sub) == "AGENTS.md"
        assert _find_workspace_root(sub) == root.resolve()
        # case 2: 有 .huginn.md 时完全覆盖
        (root / ".huginn.md").write_text("# explicit override", encoding="utf-8")
        text2 = load_project_context(sub)
        assert text2.strip() == "# explicit override", f".huginn.md 应覆盖: {text2!r}"
        assert context_source(sub) == ".huginn.md"
        # case 3: 无任何 context 文件
        empty = Path(td) / "empty"
        empty.mkdir()
        assert load_project_context(empty) == ""
        assert context_source(empty) == "none"
        print("self-check OK: root detection + AGENTS.md collection + .huginn.md override")
