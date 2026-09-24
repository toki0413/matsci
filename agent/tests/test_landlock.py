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
    # ABI >= 4 adds handled_access_net (u64) → 16 bytes.
    assert ctypes.sizeof(ll._RulesetAttrNet) == 16


# ── P1: 网络隔离 (ABI >= 4) ────────────────────────────────────────

def test_net_mask_values(ll):
    assert ll.NET_SIDE == ll.LL_NET_BIND_TCP | ll.LL_NET_CONNECT_TCP
    assert ll.NET_MIN_ABI == 4


def test_ruleset_masks_net_isolation(ll):
    # net-only: FS 完全不处理 (0), 只处理网络位 — 不锁文件系统.
    assert ll._ruleset_masks(4, False, True) == (0, ll.NET_SIDE)
    # FS + net.
    assert ll._ruleset_masks(4, True, True) == (ll.FS_MASK_ABI4, ll.NET_SIDE)
    # ABI 3 不支持网络 → 只做 FS.
    assert ll._ruleset_masks(3, True, True) == (ll.FS_MASK_ABI3, 0)
    # 都不开 → 空.
    assert ll._ruleset_masks(4, False, False) == (0, 0)


class _FakeLibc:
    """捕获 syscall 参数, 不真调内核."""

    def __init__(self, ruleset_fd: int = 7) -> None:
        self._fd = ruleset_fd
        self.calls: list[tuple] = []
        self.prctl_calls: list[tuple] = []

    def syscall(self, num, *args):
        self.calls.append((num, args))
        return self._fd if num == 444 else 0

    def prctl(self, *args):
        self.prctl_calls.append(args)
        return 0


def _patch_syscalls(ll, monkeypatch, abi: int):
    fake = _FakeLibc()
    monkeypatch.setattr(ll, "probe_abi", lambda: abi)
    monkeypatch.setattr(ll, "_libc", lambda: fake)
    monkeypatch.setattr(ll, "_syscalls", lambda: (444, 445, 446))
    monkeypatch.setattr(ll.os, "close", lambda fd: None)
    return fake


def test_restrict_net_only_passes_net_struct(ll, monkeypatch):
    fake = _patch_syscalls(ll, monkeypatch, abi=4)

    ll.restrict([], [], net_isolate=True)

    create = [c for c in fake.calls if c[0] == 444]
    assert create, fake.calls
    # create_ruleset 的第 2 个参数是 sizeof(attr) → 16 (net 变体).
    assert create[0][1][1] == 16
    assert fake.prctl_calls and fake.prctl_calls[0][0] == ll.PR_SET_NO_NEW_PRIVS
    # 无 path 规则 → 不调 add_rule (445).
    assert all(c[0] != 445 for c in fake.calls)


def test_restrict_fs_only_passes_fs_struct(ll, monkeypatch, tmp_path):
    fake = _patch_syscalls(ll, monkeypatch, abi=4)

    ll.restrict([str(tmp_path)], [], net_isolate=False)

    create = [c for c in fake.calls if c[0] == 444]
    assert create[0][1][1] == 8, "未请求网络隔离时用 8 字节 FS 变体"
    assert any(c[0] == 445 for c in fake.calls), "ro path 应产生 add_rule"


def test_restrict_nothing_to_confine_raises(ll, monkeypatch):
    monkeypatch.setattr(ll, "probe_abi", lambda: 4)
    with pytest.raises(ll.LandlockUnavailableError):
        ll.restrict([], [], net_isolate=False)


def test_restrict_net_unsupported_abi_raises(ll, monkeypatch):
    # ABI 3 + 只请求网络 → 无可处理位 → 报错 (调用方据此降级).
    monkeypatch.setattr(ll, "probe_abi", lambda: 3)
    with pytest.raises(ll.LandlockUnavailableError):
        ll.restrict([], [], net_isolate=True)


# ── P1: 本地资源限制 (setrlimit) ───────────────────────────────────

def test_apply_rlimits_sets_each(ll, monkeypatch):
    import resource as _res

    seen: list[tuple] = []
    monkeypatch.setattr(_res, "setrlimit", lambda r, lim: seen.append((r, lim)))
    ll._apply_rlimits({_res.RLIMIT_CPU: (10, 15), _res.RLIMIT_FSIZE: (1024, 1024)})
    assert (_res.RLIMIT_CPU, (10, 15)) in seen
    assert (_res.RLIMIT_FSIZE, (1024, 1024)) in seen


def test_apply_rlimits_best_effort(ll, monkeypatch):
    import resource as _res

    seen: list[tuple] = []

    def flaky(r, lim):
        if r == _res.RLIMIT_CPU:
            raise OSError("unsupported")
        seen.append((r, lim))

    monkeypatch.setattr(_res, "setrlimit", flaky)
    ll._apply_rlimits({_res.RLIMIT_CPU: (1, 2), _res.RLIMIT_FSIZE: (10, 10)})
    assert (_res.RLIMIT_FSIZE, (10, 10)) in seen, "单个失败不阻塞其余"


def test_apply_rlimits_noop_when_empty(ll, monkeypatch):
    import resource as _res

    called: list[int] = []
    monkeypatch.setattr(_res, "setrlimit", lambda r, lim: called.append(r))
    ll._apply_rlimits({})
    assert called == []


def test_make_preexec_fn_rlimit_only_without_landlock(ll, monkeypatch):
    import resource as _res

    monkeypatch.setattr(ll, "probe_abi", lambda: 0)
    seen: list[tuple] = []
    monkeypatch.setattr(_res, "setrlimit", lambda r, lim: seen.append((r, lim)))

    pre = ll.make_preexec_fn([], [], rlimits={_res.RLIMIT_CPU: (5, 6)})
    assert pre is not None
    pre()
    assert (_res.RLIMIT_CPU, (5, 6)) in seen


def test_make_preexec_fn_none_when_nothing_to_do(ll, monkeypatch):
    monkeypatch.setattr(ll, "probe_abi", lambda: 0)
    assert ll.make_preexec_fn([], []) is None


def test_make_preexec_fn_net_only_fails_closed_when_required(ll, monkeypatch):
    monkeypatch.setattr(ll, "probe_abi", lambda: 0)
    with pytest.raises(ll.LandlockUnavailableError):
        ll.make_preexec_fn([], [], net_isolate=True, required=True)
    # 非 required → None (优雅降级).
    assert ll.make_preexec_fn([], [], net_isolate=True) is None


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
