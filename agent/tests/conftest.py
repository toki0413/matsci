"""Shared pytest fixtures and configuration for Huginn tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Tests run on the local machine without a container runtime by default.
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")
os.environ.setdefault("HUGINN_PROMPT_CACHE_CONTROL", "0")
# Enable dev mode so tests can hit API endpoints without a configured API key.
os.environ.setdefault("HUGINN_DEV_MODE", "1")
# Stub API key + HPC host so CLI command registration doesn't filter out
# chat/coder/explore/hpc/scheduler/workflow/etc (filter_commands_by_availability
# reads these at invocation time and mutates the click group in place).
# Tests that verify auth behavior override these via monkeypatch.
os.environ.setdefault("HUGINN_API_KEY", "test-key")
os.environ.setdefault("HUGINN_HPC_HOST", "testhost")
# Redirect ~/.huginn writes to a test-local dir so tool_cache.sqlite and
# memory.db don't fail with "unable to open database file" in sandboxed envs.
_TEST_CACHE_DIR = str(Path(__file__).parent / ".test_cache")
os.environ.setdefault("HUGINN_CACHE_DIR", _TEST_CACHE_DIR)


@pytest.fixture(autouse=True)
def _clear_config_cache_between_tests():
    """Clear the module-level config cache before and after each test.

    Prevents one test's config (with models/api_key) from leaking into the
    next via _would_lose_auth_state, which compares against the cache.
    """
    from huginn.config import clear_config_cache

    clear_config_cache()
    yield
    clear_config_cache()
