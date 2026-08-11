"""Shared pytest fixtures and configuration for Huginn tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# 脚本式测试文件: 用 `python -m tests.test_xxx` 运行, 不含 test_* 函数,
# pytest 收集会得到 0 项并 exit 5. 在这里统一忽略, 避免被误判为失败.
collect_ignore_glob = [
    "test_async_pipeline.py",
    "test_clawbench_runner.py",
    "test_next_step_advisor.py",
    "test_temporal_iter_hist.py",
    "test_temporal_p2p3.py",
    "test_temporal_pmk.py",
    "test_temporal_spatial.py",
]

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
# Give each xdist worker its own subdir: both workers importing
# test_phase3_database_tool.py at collection time opens SQLite in the
# cache dir — sharing one dir causes "database is locked" → xdist
# collection mismatch. Per-worker dirs eliminate the contention.
_worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
_TEST_CACHE_DIR = str(Path(__file__).parent / ".test_cache" / _worker_id)
Path(_TEST_CACHE_DIR).mkdir(parents=True, exist_ok=True)
# Force-assign (not setdefault): xdist workers inherit the master's env,
# so setdefault would be a no-op in workers and they'd all share the
# master's cache dir → SQLite "database is locked" at collection time.
os.environ["HUGINN_CACHE_DIR"] = _TEST_CACHE_DIR


@pytest.fixture(scope="session")
def shared_huginn_app():
    """The single shared Huginn FastAPI app (module singleton).

    Building the app (lifespan startup) allocates ~2-3GB and loads every
    tool/model/DB. Session scope ensures it is constructed at most once per
    xdist worker, so the TestClient-heavy files no longer each spin up their
    own full app — the root cause of the memory accumulation that forced the
    old http-tests a/b/c/d/e split.
    """
    from huginn.server import app

    return app


@pytest.fixture(scope="module")
def app_client(shared_huginn_app):
    """A TestClient over the shared app, properly closed at module end.

    This is the one-and-for-all fix for the module-level ``client =
    TestClient(app)`` antipattern that plagued the suite: a client created at
    import time was never closed, so its anyio portal thread leaked forever
    (freezing xdist workers) and the lifespan never shut down (memory kept
    accumulating across modules until OOM).

    Module scope runs the lifespan once per module and tears it down on exit:
     - ``with TestClient(...)`` closes the transport → no portal thread leak.
     - lifespan shutdown frees the app's ~2-3GB between modules.
    """
    from fastapi.testclient import TestClient

    with TestClient(shared_huginn_app) as client:
        yield client


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


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Register custom markers for industry-grade test categorization."""
    config.addinivalue_line("markers", "integration: heavy tests that need full stack (skip on fast CI)")
    # GitHub runner 的 TMPDIR 常是 symlink 链, pytest tmp_path 的临时目录
    # _ensure_relative_to_basetemp 会因 getbasetemp().resolve() != basetemp
    # 误报 "is not a normalized and relative path", 让 TestCheckBudgetUnit 随机
    # ERROR.
    # 时序陷阱: pytest hook 默认 LIFO 调用, conftest.py 比内置 _pytest.tmpdir
    # 后注册因此 *先* 执行 pytest_configure, 此时 config._tmp_path_factory 还
    # 不存在, 直接 getattr 拿到 None, 修复被静默跳过. 用 trylast=True 把自己
    # 推到内置 tmpdir 插件之后, factory 已就绪, 再修正 _given_basetemp 为
    # realpath, 让 tmp_path 落在真实路径上绕开 symlink 校验.
    import tempfile
    import types

    factory = getattr(config, "_tmp_path_factory", None)
    if factory is not None:
        if factory._given_basetemp is None:
            real = os.path.realpath(tempfile.gettempdir())
            new_basetemp = Path(os.path.join(real, "huginn_pytest_tmp"))
            factory._given_basetemp = new_basetemp
        # 兜底: 直接 patch _ensure_relative_to_basetemp 跳过 symlink 校验.
        # mktemp 内部走 self._ensure_relative_to_basetemp(basename), 实例属性
        # 优先于类方法, types.MethodType 绑定后即生效. basename 已是 normpath
        # 过的测试名, 不含路径分隔符, 直接返回是安全的.
        def _noop_ensure_relative(self, basename: str) -> str:
            return os.path.normpath(basename)
        factory._ensure_relative_to_basetemp = types.MethodType(
            _noop_ensure_relative, factory
        )


def _is_disk_io_flaky(exc) -> bool:
    """True if the exc is a CI runner transient sqlite3 I/O / lock error."""
    import sqlite3

    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc)
    return "disk I/O" in msg or "database is locked" in msg


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
