"""跨模块共用的小工具 — 只收已存在 3+ 处重复的实现.

ponytail: 不主动加新功能, 发现重复 ≥3 处才下沉到这里.
当前成员:
- now_iso: 5 处重复 (plan_store/dynamic_workflow/background/credential_store/side_conversation)
- hash_text: 2 处重复 (loop_detector/adaptive_parser)
- atomic_write_json: 2 处重复 (task_metrics/checkpoint)
- chunk_text: 2 处重复 (codebase/knowledge.store)

注: huginn/utils/ 是 PEP 420 命名空间包 (无 __init__.py), 本模块是其中一个子模块.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """UTC ISO 8601 时间戳."""
    return datetime.now(UTC).isoformat()


def hash_text(text: str, length: int = 16) -> str:
    """sha256 前N位, 给文本做指纹用 (相似度去重/缓存键)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def atomic_write_json(
    path: str | Path, data: Any, *, indent: int | None = None
) -> None:
    """原子写 JSON: tmp 文件 + rename, 防中途崩溃留半截.

    indent=None 写紧凑单行; 传 indent=2 写多行 (跟 kg/graph.py save() 一致).
    path 接受 str | Path, data 可以是任意 JSON 可序列化对象 (dict/list/dataclass 等).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, default=str, indent=indent))
        os.replace(tmp, str(p))
    except OSError:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def chunk_text(text: str, size: int = 2000, overlap: int = 200) -> list[str]:
    """简单滑动窗口分块 (按字符).

    size=2000/overlap=200 是通用默认; 调用方有特定需求 (如 KB 的 800/100)
    显式传参覆盖. 跟 knowledge.store / codebase 原内联实现等价.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks
