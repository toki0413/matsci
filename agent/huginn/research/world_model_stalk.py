"""最新世界模型预测的轻量注册表 — 供 PMK `world_model` stalk 读取.

物理工作区在每次执行后写入；PMK 读取时拿到带 worldview/domain/contract 标签的
结构化快照。这是进程内、按 (tenant, session) 隔离的 last-value 表，不是历史账。
"""
from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

_lock = threading.RLock()
_latest: dict[tuple[str, str], dict[str, Any]] = {}


def record_world_model_stalk(
    *,
    world_model_id: str,
    worldview: str,
    domain: str,
    contract_version: str,
    predicted: dict[str, float] | None = None,
    observed: dict[str, float] | None = None,
    prediction_error: float | None = None,
    epoch: int = 0,
    tenant_id: str = "default",
    session_id: str = "default",
) -> dict[str, Any]:
    record = {
        "world_model_id": str(world_model_id),
        "worldview": str(worldview),
        "domain": str(domain),
        "contract_version": str(contract_version),
        "predicted": dict(predicted or {}),
        "observed": dict(observed or {}),
        "prediction_error": prediction_error,
        "epoch": int(epoch),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    with _lock:
        _latest[(tenant_id, session_id)] = record
    return dict(record)


def get_world_model_stalk(
    tenant_id: str = "default",
    session_id: str = "default",
) -> dict[str, Any] | None:
    with _lock:
        record = _latest.get((tenant_id, session_id))
        return dict(record) if record else None


def clear_world_model_stalk(
    tenant_id: str = "default",
    session_id: str = "default",
) -> None:
    with _lock:
        _latest.pop((tenant_id, session_id), None)


__all__ = [
    "clear_world_model_stalk",
    "get_world_model_stalk",
    "record_world_model_stalk",
]
