"""跨模块共用的小工具 — 只收已存在 3+ 处重复的实现.

ponytail: 不主动加新功能, 发现重复 ≥3 处才下沉到这里.
当前成员:
- now_iso: 5 处重复 (plan_store/dynamic_workflow/background/credential_store/side_conversation)
- hash_text: 2 处重复 (loop_detector/adaptive_parser)
- atomic_write_json: 2 处重复 (task_metrics/checkpoint)

注: huginn/utils/ 是 PEP 420 命名空间包 (无 __init__.py), 本模块是其中一个子模块.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    """UTC ISO 8601 时间戳."""
    return datetime.now(timezone.utc).isoformat()


def hash_text(text: str, length: int = 16) -> str:
    """sha256 前N位, 给文本做指纹用 (相似度去重/缓存键)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def atomic_write_json(path: Path, payload: dict) -> None:
    """原子写 JSON: tmp 文件 + rename, 防中途崩溃留半截.

    跟 kg/graph.py save() 同范式, 但那个有 RLock + indent, 不适合合并.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str))
        os.replace(tmp, str(path))
    except OSError:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
