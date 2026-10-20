"""Stress tests need a running server — mark them as integration."""
from pathlib import Path

import pytest

_STRESS_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items):
    # 本 hook 是 session 级: 不按目录过滤会把整场收集的 item 全标 integration
    # (全量套件被污染, `-m "not integration"` 直接选 0 个), 只对 tests/stress/ 下的 item 打标.
    for item in items:
        if _STRESS_DIR in Path(str(item.path)).resolve().parents:
            item.add_marker(pytest.mark.integration)
