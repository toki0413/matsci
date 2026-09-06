"""统一"附件落地 + 压缩"管线。

把客户端上传的一个或多个上传文件(UploadFile)以**分块流式**方式写盘,
落到统一的附件目录(默认 ``~/.huginn/attachments``), 避免把整个文件整读进
内存; 可选地把整批附件打成 zip 归档, 并附带仅含元数据的 ``manifest.json``。

典型用法::

    batch = await land_files([file1, file2], compress=True)
    for info in batch.files:
        print(info.path, info.size, info.compressed)
    print(batch.archive)  # 归档 zip 路径 (compress=True 时)

该模块只依赖 ``fastapi.UploadFile`` 的 ``read(n)`` / ``filename`` 接口,
因此也可以直接复用于无 HTTP 层的落地场景。
"""

from __future__ import annotations

import json
import logging
import uuid
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from huginn.utils.runtime import get_runtime_home

logger = logging.getLogger(__name__)

#: 附件默认落地根目录(相对 runtime home)
ATTACHMENTS_DIR_NAME = "attachments"

#: 流式读写的默认块大小(1 MiB) —— 既不频繁 syscall, 又不会整读进内存
DEFAULT_CHUNK_SIZE = 1024 * 1024

__all__ = [
    "AttachmentInfo",
    "AttachmentBatch",
    "ATTACHMENTS_DIR_NAME",
    "DEFAULT_CHUNK_SIZE",
    "get_attachments_dir",
    "stream_to_disk",
    "land_files",
]


@dataclass
class AttachmentInfo:
    """单个落地文件的元数据。"""

    path: Path  # 落地后的文件绝对路径
    filename: str  # 清理后的原始文件名(去掉路径段/空名回退)
    size: int  # 写入的字节数(等于上传的内容大小)
    compressed: bool = False  # 是否已包含进统一 zip 归档
    chunk_size: int = DEFAULT_CHUNK_SIZE  # 实际使用的读写块大小


@dataclass
class AttachmentBatch:
    """一次"多附件落地"请求的结果。"""

    directory: Path  # 本次落地的批次目录
    files: list[AttachmentInfo]  # 每个文件的落地元数据
    archive: Path | None = None  # 统一 zip 归档路径(compress=True 时)
    count: int = 0  # 落地文件数
    total_size: int = 0  # 落地的总字节数


def get_attachments_dir(override: str | Path | None = None) -> Path:
    """统一附件目录, 不存在则创建。

    默认 ``~/.huginn/attachments``(遵循 ``HUGINN_CACHE_DIR``), 也可以传入
    ``override`` 显式指定落地根目录(便于测试时隔离)。
    """
    base = Path(override) if override else get_runtime_home() / ATTACHMENTS_DIR_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def _safe_filename(name: str | None) -> str:
    """清理上传文件名: 去路径段、空名回退、截断极长名。

    Windows / Unix 都可能带反斜杠或正斜杠路径段, 统一取最后一段, 防目录穿越。
    """
    raw = (name or "").strip()
    if not raw:
        return "unnamed"
    cleaned = Path(raw.replace("\\", "/")).name
    return cleaned[:255] if cleaned else "unnamed"


async def stream_to_disk(
    file: UploadFile,
    dest: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AttachmentInfo:
    """把一个 UploadFile **分块流式**写盘, 返回落盘元数据。

    使用 ``await file.read(chunk_size)`` 循环读取、每读一块立即写盘, 不整读进
    内存; 中途失败时清理已写入的半成品文件, 避免污染附件目录。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(file.filename)
    # 带短随机前缀, 避免同批次/多次上传的同名文件相互覆盖
    target = dest / f"{uuid.uuid4().hex[:8]}_{filename}"
    size = 0
    try:
        with target.open("wb") as fh:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                size += len(chunk)
    except Exception:
        logger.warning("附件落地失败, 清理半成品: %s", target)
        target.unlink(missing_ok=True)
        raise
    return AttachmentInfo(
        path=target, filename=filename, size=size, chunk_size=chunk_size
    )


async def land_files(
    files: Iterable[UploadFile],
    *,
    dest: str | Path | None = None,
    compress: bool = False,
    metadata: dict | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AttachmentBatch:
    """把一个或多个上传文件统一落地到附件目录(可选 zip 压缩)。

    每次落地一个独立批次子目录 ``<attachments_root>/<batch_id>/``, 避免多
    请求的同名文件互相覆盖; ``compress=True`` 时再把本批次所有文件打成一个
    ``<batch_id>.zip`` 归档, 并在归档内附 ``manifest.json`` 元数据。

    Args:
        files: 一个或多个上传文件。
        dest: 附件根目录, 默认 ``~/.huginn/attachments``。
        compress: 是否生成统一 zip 归档。
        metadata: 随压缩归档写入 manifest 的元数据(如会话/备注)。
        chunk_size: 分块流式读写的块大小。

    Returns:
        AttachmentBatch: 落盘结果(每个文件的路径 / 大小 / 是否压缩等)。
    """
    root = get_attachments_dir(dest)
    batch_id = uuid.uuid4().hex[:8]
    batch_dir = root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    infos: list[AttachmentInfo] = []
    total = 0
    for upload in files:
        info = await stream_to_disk(upload, batch_dir, chunk_size=chunk_size)
        total += info.size
        infos.append(info)

    archive = None
    if compress:
        archive = _make_archive(batch_dir, root, batch_id, infos, metadata)

    return AttachmentBatch(
        directory=batch_dir,
        files=infos,
        archive=archive,
        count=len(infos),
        total_size=total,
    )


def _make_archive(
    batch_dir: Path,
    root: Path,
    batch_id: str,
    infos: list[AttachmentInfo],
    metadata: dict | None,
) -> Path:
    """把本批次已落地的文件打成 zip, 返回归档路径。

    归档放在附件根目录下(不在批次子目录内, 避免自包含)。arcname 用落地后
    的唯一文件名, 天然避免同名冲突。打包失败时保留原始落地文件并清理残缺 zip。
    """
    archive_path = root / f"{batch_id}.zip"
    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for info in infos:
                zf.write(str(info.path), arcname=info.path.name)
                info.compressed = True
            manifest = {
                "batch_id": batch_id,
                "count": len(infos),
                "total_size": sum(i.size for i in infos),
                "metadata": metadata or {},
            }
            zf.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )
        return archive_path
    except Exception:
        logger.warning("打包 zip 失败, 保留原始落地文件: %s", archive_path)
        archive_path.unlink(missing_ok=True)
        return archive_path
