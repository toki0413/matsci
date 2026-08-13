"""Test for tool registration spec validation.

This test validates that the validate_tool_specs() helper works correctly:
- It raises ImportError/AttributeError when the spec is wrong
- It passes when the spec is correct
- It skips non-huginn (third-party) modules when their deps are missing
"""

import pytest

from huginn.tools import validate_tool_specs


def test_validate_tool_specs_pass_correct_spec():
    """Test that correctly-specified core/optional tools pass validation.

    Because we import from both core and optional lists, if any internal
    huginn spec is broken (typo'd module/class name), this test will fail.
    This is exactly what validate_tool_specs was designed for.

    Note: Third-party optional tools may be skipped if their dependencies
    are not available, but that's expected — only huginn-internal path
    typos will raise.
    """
    # if validate_tool_specs completes without raising, all huginn-internal
    # tool specs are correctly typed. That's the test.
    result = validate_tool_specs()
    # we get at least some resolved specs from core tools
    assert len(result) > 0
    # each entry has the expected format: "module.className:ClassName"
    for entry in result:
        assert ":" in entry
        parts = entry.split(":")
        assert len(parts) == 2
        assert parts[1] == parts[0].split(".")[-1]


def test_validate_tool_specs_raises_on_bad_module():
    """Test that a typo'd module path raises ImportError immediately."""
    import importlib
    from unittest import mock

    # mock the lists with a bad spec
    bad_spec = [("huginn.tools.nonexistent_module", "NonexistentTool")]
    with (
        mock.patch("huginn.tools._CORE_MODULES", []),
        mock.patch("huginn.tools._OPTIONAL_MODULES", bad_spec),
    ):
        with pytest.raises(ImportError):
            validate_tool_specs()


def test_validate_tool_specs_raises_on_bad_class():
    """Test that a correct module but wrong class name raises AttributeError."""
    from unittest import mock

    # mock with a correct module but wrong class name
    # huginn.tools.base exists and has HuginnTool, ask for NonExistentTool
    bad_spec = [("huginn.tools.base", "NonExistentTool")]
    with (
        mock.patch("huginn.tools._CORE_MODULES", []),
        mock.patch("huginn.tools._OPTIONAL_MODULES", bad_spec),
    ):
        with pytest.raises(AttributeError):
            validate_tool_specs()
