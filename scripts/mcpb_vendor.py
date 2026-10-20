"""``publish/mcpb`` 打包镜像的同步 / 审计工具.

``publish/mcpb/{capabilities,workflows}-mcp/`` 各自内嵌一份 ``huginn`` 源码副本
(自包含 MCPB bundle 需要 —— ``mcp_entry.py`` 直接 ``from huginn.cli...``). 但仓库
**没有任何构建脚本, CI 也不打 bundle**, 这份镜像是**手工维护**的 —— 于是必然漂移.
实测 (当前 HEAD): 每个镜像相对 ``agent/huginn`` 缺 71 个文件、多 3 个陈旧自检脚本、
263 个共有文件内容不同. 同一份"受治理的源码"出现了两个真值, 且镜像不受任何
治理测试覆盖.

本工具是这份镜像的**唯一同步入口**:
    python scripts/mcpb_vendor.py --check    # 报告漂移 (有漂移则非零退出)
    python scripts/mcpb_vendor.py --write    # 用 agent/huginn 覆盖镜像 (收敛)

门禁: ``agent/tests/test_publish_mirror_drift.py`` 以 only-shrink 预算调用
:func:`drift_report`, 保证漂移**只减不增**. 收敛后请下调其中的预算常量.

注意: 覆盖镜像是**发布动作** (bundle 版本号 / 发布内容), 需发布者确认, 故 ``--write``
不默认执行.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "agent" / "huginn"
BUNDLES = ("capabilities-mcp", "workflows-mcp")
_SKIP_DIRS = {"__pycache__"}
_SKIP_SUFFIX = {".pyc", ".pyo"}


def _hash_tree(root: Path) -> dict[str, str]:
    """``{相对路径: md5}``, 跳过 ``__pycache__`` / 字节码."""
    out: dict[str, str] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix in _SKIP_SUFFIX:
            continue
        out[p.relative_to(root).as_posix()] = hashlib.md5(p.read_bytes()).hexdigest()
    return out


@dataclass
class DriftReport:
    """一个 bundle 镜像相对源树的漂移."""

    bundle: str
    missing: list[str] = field(default_factory=list)   # 源里有, 镜像缺
    extra: list[str] = field(default_factory=list)      # 镜像里有, 源里无
    differing: list[str] = field(default_factory=list)  # 两边都有但内容不同

    @property
    def total(self) -> int:
        return len(self.missing) + len(self.extra) + len(self.differing)


def drift_report(bundle: str, src: Path = SRC) -> DriftReport:
    """计算 ``publish/mcpb/<bundle>/huginn`` 相对 ``src`` 的漂移."""
    mirror = REPO_ROOT / "publish" / "mcpb" / bundle / "huginn"
    if not mirror.is_dir():
        raise FileNotFoundError(f"镜像目录不存在: {mirror}")
    a, b = _hash_tree(src), _hash_tree(mirror)
    shared = set(a) & set(b)
    return DriftReport(
        bundle=bundle,
        missing=sorted(set(a) - set(b)),
        extra=sorted(set(b) - set(a)),
        differing=sorted(r for r in shared if a[r] != b[r]),
    )


def sync(bundle: str, src: Path = SRC, dry_run: bool = True) -> DriftReport:
    """用 ``src`` 覆盖镜像. ``dry_run=True`` 时只返回将要发生的漂移."""
    rep = drift_report(bundle, src)
    if dry_run:
        return rep
    mirror = REPO_ROOT / "publish" / "mcpb" / bundle / "huginn"
    for rel in rep.extra:
        (mirror / rel).unlink()
    (mirror).mkdir(parents=True, exist_ok=True)
    for rel in rep.missing + rep.differing:
        dst = mirror / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dst)
    # 清掉因删除而空掉的目录.
    for d in sorted((p for p in mirror.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    return rep


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="publish/mcpb 镜像同步/审计")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="只报告漂移 (有漂移则非零退出)")
    g.add_argument("--write", action="store_true", help="用 agent/huginn 覆盖镜像")
    args = ap.parse_args(argv)

    rc = 0
    for bundle in BUNDLES:
        rep = sync(bundle, dry_run=not args.write)
        verb = "将同步" if args.write else "漂移"
        print(
            f"[{bundle}] {verb}: 缺 {len(rep.missing)} / 多 {len(rep.extra)} "
            f"/ 内容不同 {len(rep.differing)} (共 {rep.total})"
        )
        for rel in (rep.extra + rep.missing + rep.differing)[:10]:
            print(f"    {rel}")
        if rep.total and not args.write:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(_main())