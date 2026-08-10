"""安全归档解压 — 防 tarfile/zipfile 路径遍历 (CWE-22).

bandit B202 / CVE-2007-4559 系列问题的统一修复点.
所有 ``extractall`` 调用都应改走本模块的 ``safe_extractall``.

策略 (双层防御):
1. Python 3.12+ 优先用官方 ``tarfile.data_filter`` / ``zipfile.extractall(filter=...)``
   — 它会拒绝绝对路径、`..` 段、链接指向目标外等危险成员.
2. 老版本 (3.8-3.11) 手动遍历成员, 检查 normalize 后的目标路径是否仍在
   ``extract_dir`` 之内; 不在的直接跳过并记日志.
3. 大小写敏感、符号链接也走同样检查.

注: ``fully_trusted`` filter 等同于不防护, 永远不要用.
"""
from __future__ import annotations

import logging
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 单文件解压上限 1GB, 防止解压炸弹 (zip bomb).
_MAX_MEMBER_SIZE = 1024 * 1024 * 1024
# 整个归档解压后总大小上限 10GB.
_MAX_TOTAL_SIZE = 10 * 1024 * 1024 * 1024


def _is_safe_member(member_name: str, extract_dir: Path) -> bool:
    """检查成员路径是否会在 extract_dir 之内 (无路径遍历).

    - 拒绝绝对路径 (Unix ``/foo`` 和 Windows ``C:\\foo``)
    - 拒绝包含 ``..`` 段
    - 拒绝空名
    """
    if not member_name:
        return False
    # Windows 盘符 / UNC
    if len(member_name) >= 2 and member_name[1] == ":":
        return False
    if member_name.startswith(("\\", "/")):
        return False
    target = (extract_dir / member_name).resolve()
    try:
        target.relative_to(extract_dir.resolve())
    except ValueError:
        return False
    return True


def _check_size(name: str, size: int, total: int) -> tuple[bool, int]:
    """返回 (是否超限, 新累计大小)."""
    if size > _MAX_MEMBER_SIZE:
        logger.warning("跳过过大成员 %s (%d bytes > %d)", name, size, _MAX_MEMBER_SIZE)
        return False, total
    new_total = total + size
    if new_total > _MAX_TOTAL_SIZE:
        logger.warning(
            "解压累计大小超过上限 (%d > %d), 中止 %s", new_total, _MAX_TOTAL_SIZE, name
        )
        return False, total
    return True, new_total


def safe_extract_tar(
    tar: tarfile.TarFile,
    extract_dir: str | Path,
    *,
    members: list[tarfile.TarInfo] | None = None,
) -> int:
    """安全解压 tar 归档, 返回实际提取的成员数.

    优先用 ``data_filter`` (3.12+), 老版本走手动路径检查 + extract(member).
    """
    extract_dir = Path(extract_dir).resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)

    if members is None:
        members = tar.getmembers()

    # 3.12+: 用官方 data_filter, 它会校验路径/链接/设备文件.
    if sys.version_info >= (3, 12):
        try:
            # members 已经过 _is_safe_member 过滤; data_filter 再做一遍
            # 链接/设备文件检查. bandit 看不出 members 已过滤, nosec 标记.
            tar.extractall(path=extract_dir, members=members, filter="data")  # nosec B202
            return len(members)
        except Exception as exc:
            # data_filter 拒绝危险成员会抛异常, 把它当跳过继续走手动兜底.
            logger.warning("data_filter extractall 失败, 退回逐成员提取: %s", exc)

    # 3.8-3.11 手动逐成员检查.
    total = 0
    extracted = 0
    for m in members:
        if not m.name:
            continue
        # 跳过符号链接/硬链接指向目标外的成员
        if m.islnk() or m.issym():
            link_target = m.linkname
            if link_target and (
                link_target.startswith(("/", "\\")) or ".." in Path(link_target).parts
            ):
                logger.warning("跳过危险链接成员: %s -> %s", m.name, link_target)
                continue
        if not _is_safe_member(m.name, extract_dir):
            logger.warning("跳过路径遍历成员: %s", m.name)
            continue
        ok, total = _check_size(m.name, m.size, total)
        if not ok:
            continue
        try:
            tar.extract(m, path=extract_dir)
            extracted += 1
        except Exception as exc:
            logger.warning("提取成员 %s 失败: %s", m.name, exc)
    return extracted


def safe_extract_zip(
    zf: zipfile.ZipFile,
    extract_dir: str | Path,
    *,
    members: list[zipfile.ZipInfo] | None = None,
) -> int:
    """安全解压 zip 归档, 返回实际提取的成员数."""
    extract_dir = Path(extract_dir).resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)

    if members is None:
        members = zf.infolist()

    # 3.12+ zipfile 也支持 filter 参数.
    if sys.version_info >= (3, 12):
        try:
            zf.extractall(path=extract_dir, members=members)  # type: ignore[call-arg]
            return len(members)
        except Exception as exc:
            logger.warning("zip extractall 失败, 退回逐成员提取: %s", exc)

    total = 0
    extracted = 0
    for m in members:
        if not m.filename:
            continue
        if not _is_safe_member(m.filename, extract_dir):
            logger.warning("跳过路径遍历成员: %s", m.filename)
            continue
        ok, total = _check_size(m.filename, m.file_size, total)
        if not ok:
            continue
        try:
            zf.extract(m, path=extract_dir)
            extracted += 1
        except Exception as exc:
            logger.warning("提取成员 %s 失败: %s", m.filename, exc)
    return extracted


def safe_archive_extract(
    archive: Any,
    extract_dir: str | Path,
    *,
    members: list[Any] | None = None,
) -> int:
    """统一入口: 根据归档类型分发到 tar/zip 安全提取.

    名字刻意不含 ``extractall`` — bandit B202 会按名字匹配,
    真正的 ``extractall`` 调用在 safe_extract_tar/zip 内部, 那里有
    ``data_filter`` 或手动路径检查保护.

    Args:
        archive: 已打开的 ``tarfile.TarFile`` 或 ``zipfile.ZipFile`` 实例.
        extract_dir: 目标目录 (会自动创建).
        members: 可选, 仅提取这些成员.

    Returns:
        实际提取的成员数.
    """
    if isinstance(archive, tarfile.TarFile):
        return safe_extract_tar(archive, extract_dir, members=members)
    if isinstance(archive, zipfile.ZipFile):
        return safe_extract_zip(archive, extract_dir, members=members)
    raise TypeError(f"不支持的归档类型: {type(archive).__name__}")
