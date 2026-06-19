"""Shared pytest fixtures and configuration for Huginn tests."""

from __future__ import annotations

import os

# Tests run on the local machine without a container runtime by default.
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")
os.environ.setdefault("HUGINN_PROMPT_CACHE_CONTROL", "0")
