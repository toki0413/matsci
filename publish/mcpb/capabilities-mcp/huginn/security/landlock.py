"""Landlock confinement for Linux subprocess sandboxing (T-BCSE-07).

Landlock is a Linux kernel LSM (since 5.13) that lets an unprivileged process
restrict its own filesystem access. Huginn uses it to harden the otherwise
soft subprocess sandbox: instead of only whitelisting executables, a confined
child can only *read* a small set of ``ro`` paths and *read/write* a small set
of ``rw`` dirs — everything else is denied by the kernel.

Modeled on DeepSeek Harness's ``landlock-run`` (C): negotiate the kernel ABI
via ``landlock_create_ruleset(..., -1)``, build a ruleset with the handled FS
mask for that ABI, add path rules, then ``prctl(PR_SET_NO_NEW_PRIVS)`` +
``landlock_restrict_self``. Rules are inherited across ``execve``.

Two hardening axes beyond the FS path rules:
- **Network isolation** (``net_isolate=True``, ABI >= 4 / Linux 6.7+): handling
  ``handled_access_net`` while adding no net rules denies all TCP bind/connect
  for the confined child. FS handling is *not* engaged when no path rules are
  supplied, so net-only confinement does not lock the filesystem.
- **Local resource limits** (``rlimits={RLIMIT_*: (soft, hard)}``): applied in
  the forked child (never in the parent), so there is no cross-thread race.

Graceful degradation: if the kernel has no Landlock (older Linux) or we're not
on Linux, ``make_preexec_fn`` returns ``None`` (or an rlimit-only preexec) and
the caller falls back to the existing whitelist sandbox — never breaking
current behavior.

The syscall/ctypes layer is isolated behind small functions so it can be
monkeypatched in tests (no real kernel needed).
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Landlock FS access rights (linux/landlock.h) ───────────────────
LL_FS_EXECUTE = 1 << 0
LL_FS_WRITE_FILE = 1 << 1
LL_FS_READ_FILE = 1 << 2
LL_FS_READ_DIR = 1 << 3
LL_FS_REMOVE_DIR = 1 << 4
LL_FS_REMOVE_FILE = 1 << 5
LL_FS_MAKE_CHAR = 1 << 6
LL_FS_MAKE_DIR = 1 << 7
LL_FS_MAKE_REG = 1 << 8
LL_FS_MAKE_SOCK = 1 << 9
LL_FS_MAKE_FIFO = 1 << 10
LL_FS_MAKE_BLOCK = 1 << 11
LL_FS_MAKE_SYM = 1 << 12
LL_FS_REFER = 1 << 13
LL_FS_TRUNCATE = 1 << 14
LL_FS_IOCTL_DEV = 1 << 15

# Read-only side for `ro` paths: execute + read, no writes.
READ_SIDE = LL_FS_EXECUTE | LL_FS_READ_FILE | LL_FS_READ_DIR

# ── Landlock network access rights (ABI >= 4, Linux 6.7+) ─────────
# Set in ``handled_access_net``; since Landlock net rules are *allow* rules,
# handling these bits and adding no net rules denies all TCP bind/connect.
LL_NET_BIND_TCP = 1 << 0
LL_NET_CONNECT_TCP = 1 << 1
NET_SIDE = LL_NET_BIND_TCP | LL_NET_CONNECT_TCP

# ABI that first supports network access control.
NET_MIN_ABI = 4

# Handled mask grows with ABI version (kernel feature additions).
FS_MASK_ABI1 = LL_FS_EXECUTE | LL_FS_WRITE_FILE | LL_FS_READ_FILE | LL_FS_READ_DIR
FS_MASK_ABI2 = (
    FS_MASK_ABI1
    | LL_FS_MAKE_CHAR | LL_FS_MAKE_DIR | LL_FS_MAKE_REG | LL_FS_MAKE_SOCK
    | LL_FS_MAKE_FIFO | LL_FS_MAKE_BLOCK | LL_FS_MAKE_SYM
)
FS_MASK_ABI3 = FS_MASK_ABI2 | LL_FS_REFER | LL_FS_TRUNCATE
FS_MASK_ABI4 = FS_MASK_ABI3 | LL_FS_IOCTL_DEV
FS_MASK_MAX = FS_MASK_ABI4


def fs_mask_for_abi(abi: int) -> int:
    """Handled-access mask for a given kernel ABI version."""
    if abi >= 4:
        return FS_MASK_ABI4
    if abi == 3:
        return FS_MASK_ABI3
    if abi == 2:
        return FS_MASK_ABI2
    return FS_MASK_ABI1


def _ruleset_masks(abi: int, has_fs: bool, net_isolate: bool) -> tuple[int, int]:
    """Return ``(handled_access_fs, handled_access_net)`` for a ruleset.

    ``has_fs=False`` yields ``handled_access_fs=0`` — the kernel then leaves the
    filesystem completely unconfined, which is what a net-only (or rlimit-only)
    confinement needs. ``net_isolate=True`` only engages on ABI >= 4; older
    kernels silently get net=0 (the caller degrades to FS-only).
    """
    fs = fs_mask_for_abi(abi) if has_fs else 0
    net = NET_SIDE if (net_isolate and abi >= NET_MIN_ABI) else 0
    return fs, net


# ── Syscall numbers (linux/arch/{x86_64,aarch64}/...) ──────────────
# landlock_* syscalls are 444/445/446 on both x86_64 and aarch64.
_LANDLOCK_SYSCALLS: dict[str, tuple[int, int, int]] = {
    "x86_64": (444, 445, 446),
    "aarch64": (444, 445, 446),
}

# landlock_create_ruleset flag: pass -1 as flags to query the ABI version.
LANDLOCK_CREATE_RULESET_VERSION = -1
# landlock_add_rule rule_type == LANDLOCK_RULE_PATH_BENEATH
LANDLOCK_RULE_PATH_BENEATH = 1
# prctl option
PR_SET_NO_NEW_PRIVS = 38


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _RulesetAttrNet(ctypes.Structure):
    # ABI >= 4 adds ``handled_access_net`` (u64) after the FS field. Kernels
    # older than ABI 4 reject a ruleset struct larger than they know (E2BIG),
    # so this variant is only passed when ABI >= 4.
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
    ]


class _PathBeneath(ctypes.Structure):
    # Packed MSVC-compatible layout (u64 + u32 = 12 bytes, no padding), matching
    # the kernel's `__attribute__((packed))` landlock_path_beneath_attr.
    _layout_ = "ms"
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class LandlockUnavailableError(Exception):
    """Landlock is not supported on this kernel/platform."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _syscalls() -> tuple[int, int, int] | None:
    """(create_ruleset, add_rule, restrict_self) for the current arch, or None."""
    machine = os.uname().machine
    return _LANDLOCK_SYSCALLS.get(machine)


def _libc() -> Any:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def probe_abi() -> int:
    """Return the kernel Landlock ABI version, or 0 if unsupported.

    ``landlock_create_ruleset(NULL, 0, -1)`` returns the ABI version on
    kernels that support Landlock, and -1 (errno ENOSYS/ENOMSG) otherwise.
    """
    nums = _syscalls()
    if os.name != "posix" or nums is None:
        return 0
    try:
        libc = _libc()
        ret = libc.syscall(nums[0], None, 0, LANDLOCK_CREATE_RULESET_VERSION)
        return int(ret) if ret > 0 else 0
    except Exception:
        return 0


def _add_path_rule(ruleset_fd: int, path: str, access: int, abi: int) -> None:
    """Add a path-beneath rule for ``path`` with ``access`` (masked by ABI)."""
    libc = _libc()
    nums = _syscalls()
    assert nums is not None
    parent_fd = os.open(path.encode(), os.O_PATH | os.O_CLOEXEC)
    try:
        attr = _PathBeneath(allowed_access=access & fs_mask_for_abi(abi), parent_fd=parent_fd)
        ret = libc.syscall(nums[1], ruleset_fd, LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(attr), 0)
        if ret != 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
    finally:
        os.close(parent_fd)


def restrict(
    ro_paths: list[str],
    rw_paths: list[str],
    *,
    net_isolate: bool = False,
) -> None:
    """Apply Landlock confinement to the *current* process.

    Meant to run inside ``preexec_fn`` (in the forked child, before execve).
    Only paths reachable via ``ro_paths`` (read) and ``rw_paths`` (read/write)
    remain accessible; everything else is denied by the kernel. Rules survive
    ``execve``.

    ``net_isolate=True`` additionally denies all TCP bind/connect (ABI >= 4).
    When both path lists are empty the filesystem is left unconfined — that is
    the intended shape for a net-only confinement.
    """
    abi = probe_abi()
    if abi <= 0:
        raise LandlockUnavailableError("Landlock not supported on this kernel")
    nums = _syscalls()
    assert nums is not None
    libc = _libc()

    has_fs = bool(ro_paths or rw_paths)
    handled_fs, handled_net = _ruleset_masks(abi, has_fs, net_isolate)
    if handled_fs == 0 and handled_net == 0:
        raise LandlockUnavailableError(
            "nothing to confine (no path rules; network isolation needs ABI >= 4)"
        )

    # ABI >= 4 kernels expect the larger ruleset struct; older kernels reject a
    # struct bigger than they know (E2BIG), so only pass the net variant when
    # the net mask is actually handled.
    attr: _RulesetAttr | _RulesetAttrNet
    if handled_net:
        attr = _RulesetAttrNet(
            handled_access_fs=handled_fs, handled_access_net=handled_net
        )
    else:
        attr = _RulesetAttr(handled_access_fs=handled_fs)
    ruleset_fd = int(libc.syscall(nums[0], ctypes.byref(attr), ctypes.sizeof(attr), 0))
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")

    try:
        if handled_fs:
            for p in ro_paths:
                _add_path_rule(ruleset_fd, p, READ_SIDE, abi)
            for p in rw_paths:
                _add_path_rule(ruleset_fd, p, handled_fs, abi)

        # no_new_privs, then restrict self. Both are required.
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
        if libc.syscall(nums[2], ruleset_fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset_fd)


def _apply_rlimits(rlimits: dict[int, tuple[int, int]]) -> None:
    """Apply ``{resource.RLIMIT_*: (soft, hard)}`` to the current process.

    Runs in the forked child (inside ``preexec_fn``), so the parent's limits are
    never touched — no cross-thread race. Best-effort: one unsupported resource
    is logged and skipped rather than aborting the whole confinement.
    """
    if not rlimits:
        return
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX platforms
        return
    for res, limits in rlimits.items():
        try:
            resource.setrlimit(res, limits)
        except (ValueError, OSError):
            logger.debug("setrlimit(%s, %s) failed (non-fatal)", res, limits, exc_info=True)


def make_preexec_fn(
    ro_paths: list[str | Path],
    rw_paths: list[str | Path],
    *,
    required: bool = False,
    net_isolate: bool = False,
    rlimits: dict[int, tuple[int, int]] | None = None,
) -> Callable[[], None] | None:
    """Build a ``preexec_fn`` for ``subprocess.run`` confining the child.

    The returned callable applies ``rlimits`` (if any) and then the Landlock
    ruleset (FS paths and/or network isolation). Returns ``None`` when there is
    nothing to apply and ``required=False`` — the caller then falls back to the
    existing sandbox (graceful degradation). When ``required=True`` and nothing
    can be applied, raises ``LandlockUnavailableError``.
    """
    ro = [str(p) for p in ro_paths]
    rw = [str(p) for p in rw_paths]
    has_fs = bool(ro or rw)
    has_rl = bool(rlimits)

    abi = probe_abi()
    net_ok = bool(net_isolate) and abi >= NET_MIN_ABI
    want_ll = has_fs or net_ok

    if not want_ll and not has_rl:
        if required:
            raise LandlockUnavailableError(
                "nothing to confine (no paths, no network isolation, no rlimits)"
            )
        return None

    if want_ll and abi <= 0:
        if required:
            raise LandlockUnavailableError("Landlock not supported on this kernel")
        # No Landlock: keep the rlimits if requested, otherwise degrade fully.
        if not has_rl:
            return None

        def _rlimit_only() -> None:
            _apply_rlimits(rlimits or {})

        return _rlimit_only

    def _confine() -> None:
        if has_rl:
            _apply_rlimits(rlimits or {})
        if want_ll:
            restrict(ro, rw, net_isolate=net_ok)

    return _confine


def _selfcheck() -> None:
    print("Running landlock selfcheck...")
    # Masks grow monotonically with ABI.
    assert fs_mask_for_abi(4) >= fs_mask_for_abi(3) >= fs_mask_for_abi(2) >= fs_mask_for_abi(1)
    assert fs_mask_for_abi(1) == (LL_FS_EXECUTE | LL_FS_WRITE_FILE | LL_FS_READ_FILE | LL_FS_READ_DIR)
    assert READ_SIDE & LL_FS_WRITE_FILE == 0, "read side must not allow writes"
    print("  [OK] mask hierarchy + read-side isolation")
    # Network masks are independent of the FS masks.
    assert NET_SIDE == LL_NET_BIND_TCP | LL_NET_CONNECT_TCP
    assert _ruleset_masks(4, False, True) == (0, NET_SIDE), "net-only: FS unconfined"
    assert _ruleset_masks(4, True, True) == (FS_MASK_ABI4, NET_SIDE)
    assert _ruleset_masks(3, True, True) == (FS_MASK_ABI3, 0), "net needs ABI >= 4"
    print("  [OK] network isolation masks (ABI >= 4)")
    # Struct layout (packed path_beneath = 12 bytes).
    assert ctypes.sizeof(_PathBeneath) == 12, ctypes.sizeof(_PathBeneath)
    assert ctypes.sizeof(_RulesetAttr) == 8
    assert ctypes.sizeof(_RulesetAttrNet) == 16
    print("  [OK] ctypes struct layout")
    # On a non-Linux / no-Landlock kernel, preexec_fn degrades to None (not fatal).
    pre = make_preexec_fn(["/usr/lib"], ["/tmp"], required=False)
    if pre is None:
        print("  [OK] graceful degradation (no Landlock on this kernel)")
    else:
        print("  [OK] Landlock available; preexec_fn produced")
    print("landlock selfcheck passed.")


if __name__ == "__main__":
    _selfcheck()
