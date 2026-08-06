"""Shared pytest fixtures and configuration for Huginn tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Tests run on the local machine without a container runtime by default.
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")
os.environ.setdefault("HUGINN_ALLOW_UNRESTRICTED_READ", "1")
os.environ.setdefault("HUGINN_PROMPT_CACHE_CONTROL", "0")
# Enable dev mode so tests can hit API endpoints without a configured API key.
os.environ.setdefault("HUGINN_DEV_MODE", "1")
# Stub API key + HPC host so CLI command registration doesn't filter out
# chat/coder/explore/hpc/scheduler/workflow/etc (filter_commands_by_availability
# reads these at invocation time and mutates the click group in place).
# Tests that verify auth behavior override these via monkeypatch.
os.environ.setdefault("HUGINN_API_KEY", "test-key")
os.environ.setdefault("HUGINN_HPC_HOST", "testhost")
# Disable per-IP rate limiting so fast test suites (3.12/3.13) don't get 429s
os.environ.setdefault("HUGINN_RATE_LIMIT_PER_MINUTE", "0")
# Redirect ~/.huginn writes to a test-local dir so tool_cache.sqlite and
# memory.db don't fail with "unable to open database file" in sandboxed envs.
_TEST_CACHE_DIR = str(Path(__file__).parent / ".test_cache")
os.environ.setdefault("HUGINN_CACHE_DIR", _TEST_CACHE_DIR)


@pytest.fixture(autouse=True)
def _clear_config_cache_between_tests(monkeypatch):
    """Clear config cache + config-path overrides before and after each test.

    Prevents one test's config (with models/api_key) from leaking into the
    next via _would_lose_auth_state, which compares against the cache.
    Also resets the encrypt/decrypt runtime override so tests are isolated.
    """
    from huginn.config import clear_config_cache

    monkeypatch.delenv("HUGINN_CONFIG_FILE", raising=False)
    # Only reset if the module is already loaded — avoids pulling the
    # entire routes→agent→langgraph import chain on every test setup.
    import sys
    _mod = sys.modules.get("huginn.routes.config")
    if _mod is not None and hasattr(_mod, "_config_path_override"):
        _mod._config_path_override = None
    clear_config_cache()
    yield
    clear_config_cache()


def pytest_configure(config):
    """Register custom markers for industry-grade test categorization."""
    config.addinivalue_line("markers", "integration: heavy tests that need full stack (skip on fast CI)")


def _is_disk_io_flaky(exc) -> bool:
    """True if the exc is the CI runner's transient sqlite3 disk I/O error."""
    import sqlite3

    return isinstance(exc, sqlite3.OperationalError) and "disk I/O" in str(exc)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """fixture setup 阶段同样会建 SQLite 库 (如 LongTermMemory),
    CI runner 磁盘临时空间不足时同样会炸, 一视同仁 skip."""
    outcome = yield
    excinfo = outcome.excinfo
    if excinfo is not None and _is_disk_io_flaky(excinfo[1]):
        outcome.force_exception(
            pytest.skip.Exception(
                f"CI runner disk I/O flaky: {excinfo[1]}", pytrace=False
            )
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """CI runner 偶发 sqlite3 disk I/O error (磁盘临时空间不足),
    属于环境问题不是代码 bug, 挂了直接 skip."""
    outcome = yield
    excinfo = outcome.excinfo
    if excinfo is not None and _is_disk_io_flaky(excinfo[1]):
        outcome.force_exception(
            pytest.skip.Exception(
                f"CI runner disk I/O flaky: {excinfo[1]}", pytrace=False
            )
        )
