"""Path guards shared by filesystem-backed tools.

The model supplies ``working_dir`` as free text.  A path written with
full-width punctuation ("C﹕＼Users＼...") or with its separators eaten
("CUserswanzhDesktop...") carries no real separator, so ``Path()`` sees a
single relative component and ``mkdir`` happily materialises a junk
directory next to the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# 全角冒号/斜杠/反斜杠/小冒号: 模型在中文语境里经常这样写 Windows 路径
_FULLWIDTH_SEPARATORS = ("\uff1a", "\uff0f", "\uff3c", "\ufe55")


def resolve_work_dir(raw: str | None) -> tuple[Path | None, str | None]:
    """Return ``(resolved_path, error)``; ``error`` is set when ``raw`` is unusable.

    ponytail: 只校验 + 归一化, 不自动建目录.
      天花板: 调用方若指望传一个不存在的目录让工具顺手创建, 会拿到错误.
      升级路径: 需要时给调用方加显式 create=True, 由它在自己的边界内建目录.
      现在就安全: 现有调用方 (orchestrator/bench runner/测试) 传的都是已存在的
      workspace, 建目录本来也该发生在 workspace 所有者那一层.
    """
    if not raw:
        return Path.cwd(), None

    if any(ch in raw for ch in _FULLWIDTH_SEPARATORS) or any(ord(ch) < 32 for ch in raw):
        return None, (
            f"working_dir contains full-width separators or control characters: {raw!r}. "
            "Use a plain path with ASCII ':' and '\\'."
        )

    path = Path(os.path.normpath(raw))
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        return None, (
            f"working_dir does not exist: {path}. Use an existing workspace directory; "
            "create new directories via file_write_tool or from inside the code."
        )
    if not path.is_dir():
        return None, f"working_dir is not a directory: {path}"
    return path, None
