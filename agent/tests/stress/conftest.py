"""Stress tests need a running server — mark them as integration."""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.integration)
