"""T-BCSE-07: Landlock confinement module tests.

``huginn.security/__init__`` imports ``math_eval`` which pulls in numpy (not
installed in the test env), so importing the package fails here. These tests
load ``landlock.py`` standalone via importlib to exercise the pure logic and
the graceful-degradation contract without a real Landlock kernel.
"""

from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path

import pytest


def _load_landlock():
    path = Path(__file__).resolve().parent.parent / "huginn" / "security" / "landlock.py"
    spec = importlib.util.spec_from_file_location("_landlock_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ll():
    return _load_landlock()


def test_mask_hierarchy(ll):
    assert ll.fs_mask_for_abi(1) == (
        ll.LL_FS_EXECUTE | ll.LL_FS_WRITE_FILE | ll.LL_FS_READ_FILE | ll.LL_FS_READ_DIR
    )
    # ABI masks grow monotonically.
    assert (
        ll.fs_mask_for_abi(4)
        >= ll.fs_mask_for_abi(3)
        >= ll.fs_mask_for_abi(2)
        >= ll.fs_mask_for_abi(1)
    )


def test_read_side_never_allows_writes(ll):
    assert ll.READ_SIDE & ll.LL_FS_WRITE_FILE == 0
    assert ll.READ_SIDE & ll.LL_FS_REMOVE_DIR == 0


def test_ctypes_struct_layout(ll):
    # packed landlock_path_beneath_attr = u64 (8) + u32 (4) = 12 bytes.
    assert ctypes.sizeof(ll._PathBeneath) == 12
    # landlock_ruleset_attr = u64 handled_access_fs = 8 bytes.
    assert ctypes.sizeof(ll._RulesetAttr) == 8


def test_make_preexec_fn_degrades_when_unavailable(ll, monkeypatch):
    # No Landlock kernel → preexec_fn is None (graceful, not fatal).
    monkeypatch.setattr(ll, "probe_abi", lambda: 0)
    assert ll.make_preexec_fn(["/usr/lib"], ["/tmp"]) is None
    # required=True is fail-closed.
    with pytest.raises(ll.LandlockUnavailableError):
        ll.make_preexec_fn(["/usr/lib"], ["/tmp"], required=True)


def test_make_preexec_fn_produced_when_available(ll, monkeypatch):
    monkeypatch.setattr(ll, "probe_abi", lambda: 3)
    pre = ll.make_preexec_fn(["/usr/lib", "/usr/lib64"], ["/tmp", "/work"])
    assert pre is not None and callable(pre)


def test_unsupported_platform_returns_none(ll, monkeypatch):
    monkeypatch.setattr(ll, "os", _FakePosixArch("other-arch"))
    assert ll.probe_abi() == 0


class _FakePosixArch:
    """Stand-in for the os module that reports a non-x86/arm machine."""

    name = "posix"

    def __init__(self, machine: str) -> None:
        self._machine = machine

    def uname(self):
        return _FakeUname(self._machine)


class _FakeUname:
    def __init__(self, machine: str) -> None:
        self.machine = machine