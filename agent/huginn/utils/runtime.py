"""Runtime Home —— 统一管理 Huginn 运行时目录结构.

所有持久化数据 (audit log, memory, research log, credentials, campaigns,
provenance, checkpoints) 都落在同一个 runtime home 下, 避免散落在各处.

默认路径: $HUGINN_CACHE_DIR 或 ~/.huginn
测试时可以通过环境变量切换到临时目录.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["get_runtime_home", "ensure_runtime_dirs", "RUNTIME_LAYOUT"]


# 运行时目录结构: key=子目录名, value=用途说明
RUNTIME_LAYOUT: dict[str, str] = {
    "": "runtime root",
    "audit.jsonl": "append-only audit log (hash chain + HMAC)",
    "memory.db": "long-term memory (SQLite)",
    "anomalies.db": "PRT anomaly log",
    "campaigns.sqlite": "campaign loop state",
    "research_log.sqlite": "structured research log (conjecture/verification/obstacle)",
    "credentials.json": "encrypted credential store",
    "provenance.jsonl": "provenance trail",
    "checkpoints.db": "session checkpoints",
    "goals.json": "goal scheduler state",
    "plans.json": "plan queue",
    "rag/": "RAG vector store (ChromaDB)",
    "personas/": "persona files",
    "knowledge/": "domain knowledge base",
}


def get_runtime_home() -> Path:
    """返回 runtime home 目录路径, 不创建.

    优先取 $HUGINN_CACHE_DIR, 没有就退回 ~/.huginn.
    """
    base = os.environ.get("HUGINN_CACHE_DIR")
    if base:
        return Path(base)
    return Path.home() / ".huginn"


def ensure_runtime_dirs() -> Path:
    """确保 runtime home 及关键子目录存在, 返回根路径.

    只创建目录, 不创建文件 (文件由各模块自己写).
    幂等, 多次调用没副作用.
    """
    home = get_runtime_home()
    home.mkdir(parents=True, exist_ok=True)
    # 只创建需要提前存在的子目录
    for subdir in ("rag", "personas", "knowledge"):
        (home / subdir).mkdir(parents=True, exist_ok=True)
    return home
