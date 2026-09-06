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

Graceful degradation: if the kernel has no Landlock (older Linux) or we're not
on Linux, ``make_preexec_fn`` returns ``None`` and the caller falls back to the
existing whitelist sandbox — never breaking current behavior.

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


def restrict(ro_paths: list[str], rw_paths: list[str]) -> None:
    """Apply Landlock confinement to the *current* process.

    Meant to run inside ``preexec_fn`` (in the forked child, before execve).
    Only paths reachable via ``ro_paths`` (read) and ``rw_paths`` (read/write)
    remain accessible; everything else is denied by the kernel. Rules survive
    ``execve``.
    """
    abi = probe_abi()
    if abi <= 0:
        raise LandlockUnavailableError("Landlock not supported on this kernel")
    nums = _syscalls()
    assert nums is not None
    libc = _libc()

    handled = fs_mask_for_abi(abi)
    attr = _RulesetAttr(handled_access_fs=handled)
    ruleset_fd = int(libc.syscall(nums[0], ctypes.byref(attr), ctypes.sizeof(attr), 0))
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")

    try:
        for p in ro_paths:
            _add_path_rule(ruleset_fd, p, READ_SIDE, abi)
        for p in rw_paths:
            _add_path_rule(ruleset_fd, p, handled, abi)

        # no_new_privs, then restrict self. Both are required.
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
        if libc.syscall(nums[2], ruleset_fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset_fd)


def make_preexec_fn(
    ro_paths: list[str | Path],
    rw_paths: list[str | Path],
    *,
    required: bool = False,
) -> Callable[[], None] | None:
    """Build a ``preexec_fn`` for ``subprocess.run`` confining the child.

    Returns ``None`` when Landlock is unavailable and ``required=False`` — the
    caller should then fall back to the existing sandbox (graceful degradation).
    When ``required=True`` and unavailable, raises ``LandlockUnavailable``.
    """
    if probe_abi() <= 0:
        if required:
            raise LandlockUnavailableError("Landlock not supported on this kernel")
        return None
    ro = [str(p) for p in ro_paths]
    rw = [str(p) for p in rw_paths]

    def _confine() -> None:
        restrict(ro, rw)

    return _confine


def _selfcheck() -> None:
    print("Running landlock selfcheck...")
    # Masks grow monotonically with ABI.
    assert fs_mask_for_abi(4) >= fs_mask_for_abi(3) >= fs_mask_for_abi(2) >= fs_mask_for_abi(1)
    assert fs_mask_for_abi(1) == (LL_FS_EXECUTE | LL_FS_WRITE_FILE | LL_FS_READ_FILE | LL_FS_READ_DIR)
    assert READ_SIDE & LL_FS_WRITE_FILE == 0, "read side must not allow writes"
    print("  [OK] mask hierarchy + read-side isolation")
    # Struct layout (packed path_beneath = 12 bytes).
    assert ctypes.sizeof(_PathBeneath) == 12, ctypes.sizeof(_PathBeneath)
    assert ctypes.sizeof(_RulesetAttr) == 8
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
