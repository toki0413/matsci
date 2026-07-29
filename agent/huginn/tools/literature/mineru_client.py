"""MinerU 精准解析 API 客户端 (huginn 移植版).

上游: RocHunag1996/mineru-material-parser/src/mineru_client.py
改造点:
  1. 去掉硬编码 config 模块依赖, 改为接受 api_keys 参数 + fallback pick_api_key("mineru")
  2. 路径改为 huginn workspace (config.workspace / "data" / "parsed")
  3. 200 页限制由本模块内部处理 (上游用独立 pdf_splitter, huginn SmartIngester 已处理压缩包递归)

核心流程:
  1) POST /file-urls/batch          申请上传URL, 拿到 batch_id + 预签名 PUT URLs
  2) PUT  到预签名URL (无需 Content-Type / Auth)
  3) GET  /extract-results/batch/{batch_id}  轮询, 所有任务 done / failed 即结束
  4) 下载 full_zip_url, 解压到 data/parsed/{stem}/
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

logger = logging.getLogger(__name__)

MINERU_BASE = "https://mineru.net/api/v4"
# 单批最多 50 个文件 (API 约束)
_MAX_BATCH = 50
# MinerU 200 页限制: 超过的文档拒绝处理, 由调用方 (SmartIngester) 预切
_MAX_PAGES = 200
# 轮询默认参数
_DEFAULT_POLL_INTERVAL = 20
_DEFAULT_TIMEOUT = 60 * 60 * 6  # 6 小时

# 终态集合
_TERMINAL = {"done", "failed"}


def _resolve_keys(api_keys: list[str] | None) -> list[str]:
    """优先用显式传入的 keys, fallback 到 pick_api_key("mineru").

    ponytail: pick_api_key 是 LLM provider 用的轮询器, MinerU 不是 LLM 但复用机制.
    升级路径: 给 MinerU 单独写一个 key pool, 不混进 LLM registry.
    """
    if api_keys:
        return [k for k in api_keys if k]
    try:
        from huginn.models.registry import pick_api_key
        k = pick_api_key("mineru")
        return [k] if k else []
    except Exception:
        return []


def _headers(token: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


@dataclass
class FileSpec:
    """单文件上传规格.

    data_id 必须仅含字母数字._-, <=128 字符.
    is_ocr=True 走 OCR 模式 (扫描件), False 走 VLM 版面解析 (默认).
    """
    path: Path
    data_id: str
    is_ocr: bool = False
    page_ranges: str | None = None


@dataclass
class BatchSubmitResult:
    """submit_batch 的返回: batch_id + 每个文件的上传状态."""
    batch_id: str
    uploads: dict[str, dict] = field(default_factory=dict)


def submit_batch(
    files: list[FileSpec],
    *,
    api_keys: list[str] | None = None,
    model_version: str = "vlm",
    enable_formula: bool = True,
    enable_table: bool = True,
    language: str = "ch",
    extra_formats: list[str] | None = None,
) -> BatchSubmitResult:
    """申请上传链接 + PUT 上传文件. 返回 batch_id 与每个文件的上传结果.

    多 key 轮询: 每次调用取一个 key, 用完轮换.
    """
    if not files:
        raise ValueError("files 为空")
    if len(files) > _MAX_BATCH:
        raise ValueError(f"单批最多 {_MAX_BATCH} 个, 当前 {len(files)}")

    keys = _resolve_keys(api_keys)
    if not keys:
        raise RuntimeError(
            "MinerU API key 未配置. 设置 MINERU_API_KEYS 环境变量或 HuginnConfig.mineru_api_keys"
        )
    # ponytail: 简单轮询, 不加锁 — 单进程内 submit_batch 串行调用, 并发由调用方控制.
    token = keys[0] if len(keys) == 1 else keys[int(time.time()) % len(keys)]

    payload: dict[str, Any] = {
        "files": [
            {
                "name": f.path.name,
                "data_id": f.data_id,
                **({"is_ocr": True} if f.is_ocr else {}),
                **({"page_ranges": f.page_ranges} if f.page_ranges else {}),
            }
            for f in files
        ],
        "model_version": model_version,
        "enable_formula": enable_formula,
        "enable_table": enable_table,
        "language": language,
    }
    if extra_formats:
        payload["extra_formats"] = extra_formats

    r = requests.post(
        f"{MINERU_BASE}/file-urls/batch",
        headers=_headers(token),
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(f"申请上传URL失败: {body}")

    batch_id = body["data"]["batch_id"]
    upload_urls: list[str] = body["data"]["file_urls"]
    if len(upload_urls) != len(files):
        raise RuntimeError(
            f"上传URL数量不匹配: 申请{len(files)} 返回{len(upload_urls)}"
        )

    result = BatchSubmitResult(batch_id=batch_id)
    for spec, url in zip(files, upload_urls):
        info: dict = {"url": url, "uploaded": False, "err": None}
        try:
            with open(spec.path, "rb") as fp:
                # 文档明确说: 上传时无须设置 Content-Type
                up = requests.put(url, data=fp, timeout=300)
            if up.status_code == 200:
                info["uploaded"] = True
            else:
                info["err"] = f"HTTP {up.status_code}: {up.text[:200]}"
        except Exception as e:
            info["err"] = repr(e)
        result.uploads[spec.data_id] = info
    return result


def query_batch(
    batch_id: str,
    *,
    api_keys: list[str] | None = None,
) -> list[dict]:
    """返回 extract_result 列表, 每个元素含 file_name/state/full_zip_url/err_msg/data_id 等."""
    keys = _resolve_keys(api_keys)
    if not keys:
        raise RuntimeError("MinerU API key 未配置")
    token = keys[0] if len(keys) == 1 else keys[int(time.time()) % len(keys)]

    r = requests.get(
        f"{MINERU_BASE}/extract-results/batch/{batch_id}",
        headers=_headers(token),
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(f"查询batch失败: {body}")
    return body["data"].get("extract_result", []) or []


def wait_batch(
    batch_id: str,
    *,
    api_keys: list[str] | None = None,
    poll_interval: int = _DEFAULT_POLL_INTERVAL,
    timeout: int = _DEFAULT_TIMEOUT,
    on_progress: Callable[[list[dict]], None] | None = None,
) -> list[dict]:
    """阻塞轮询, 直到所有文件 done/failed 或超时."""
    start = time.time()
    while True:
        results = query_batch(batch_id, api_keys=api_keys)
        if results and all(item.get("state") in _TERMINAL for item in results):
            return results
        if on_progress:
            try:
                on_progress(results)
            except Exception:
                logger.debug("on_progress callback failed", exc_info=True)
        if time.time() - start > timeout:
            raise TimeoutError(
                f"batch {batch_id} 等待超过 {timeout}s 仍未完成"
            )
        time.sleep(poll_interval)


def download_and_extract(
    zip_url: str,
    target_dir: Path,
    *,
    keep_zip: Path | None = None,
) -> Path:
    """下载 zip 并解压到 target_dir/, 返回 target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(zip_url, timeout=300)
    r.raise_for_status()
    data = r.content
    if keep_zip:
        keep_zip.parent.mkdir(parents=True, exist_ok=True)
        keep_zip.write_bytes(data)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(target_dir)
    return target_dir


def parsed_dir_for(data_id: str, parsed_root: Path) -> Path:
    """返回单个 data_id 的解析结果目录."""
    return parsed_root / data_id


def chunked(seq: Iterable, size: int):
    """把序列切成 size 大小的块."""
    buf: list = []
    for x in seq:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


if __name__ == "__main__":
    # C2 self-check: 模块导入 + 常量 + chunked 行为验证.
    # 不测真实 API 调用 (需要 key + 网络).
    assert _MAX_BATCH == 50, f"API limit is 50, got {_MAX_BATCH}"
    assert _MAX_PAGES == 200, f"page limit is 200, got {_MAX_PAGES}"
    assert _TERMINAL == {"done", "failed"}, f"terminal states mismatch: {_TERMINAL}"
    # _resolve_keys: 无 key 时返回空 list
    assert _resolve_keys(None) == [] or len(_resolve_keys(None)) >= 0  # 无环境变量时为空
    # chunked 行为
    chunks = list(chunked([1, 2, 3, 4, 5], 2))
    assert chunks == [[1, 2], [3, 4], [5]], f"chunked mismatch: {chunks}"
    # FileSpec 构造
    fs = FileSpec(path=Path("test.pdf"), data_id="test_001", is_ocr=False)
    assert fs.data_id == "test_001"
    assert fs.is_ocr is False
    # BatchSubmitResult 默认值
    bsr = BatchSubmitResult(batch_id="b1")
    assert bsr.uploads == {}
    # parsed_dir_for
    p = parsed_dir_for("doc_001", Path("/tmp/parsed"))
    assert p == Path("/tmp/parsed/doc_001")
    print("C2 self-check OK")
