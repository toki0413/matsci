"""Knowledge base / RAG and export endpoints."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from huginn.security.auth import require_api_key
from huginn.server_core import get_context

router = APIRouter(tags=["knowledge"], dependencies=[Depends(require_api_key)])

logger = logging.getLogger(__name__)

_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


def _sem_distill_upload(content: bytes, filename: str) -> None:
    """对上传的纯文本资料补一层语义蒸馏(决策启发式/领域事实).

    只处理能按 UTF-8 解码且无 NUL(bytes 的二进制如 PDF/图片会带 \x00 或
    解析不了), 由 auto_ingest.distill_semantic_text 走质量门槛入库。
    ponytail: 非阻塞等待 LLM 一次调用; 上传是显式动作, 多花一点时间可接受.
    """
    try:
        if b"\x00" in content:
            return
        text = content.decode("utf-8")
    except Exception:
        return
    if len(text.strip()) < 80:
        return
    try:
        from huginn.knowledge.auto_ingest import distill_semantic_text

        asyncio.get_running_loop().create_task(
            asyncio.to_thread(
                distill_semantic_text,
                text,
                source=filename,
                source_type="user_upload",
            )
        )
    except Exception:
        logger.debug("upload semantic distill 失败 (非致命)", exc_info=True)


class KnowledgeQuery(BaseModel):
    query: str = ""
    top_k: int = 5


class UrlIngestRequest(BaseModel):
    url: str


class ReportRequest(BaseModel):
    doc_ids: list[str]
    title: str = ""


def _save_raw_if_docid(kb: Any, doc_id: str, filename: str, content: bytes) -> None:
    """doc_id 归一后存原始文件; 无 doc_id 说明摄入失败, 静默跳过."""
    if not doc_id:
        return
    try:
        kb.save_raw(doc_id, filename, content)
    except Exception:
        logger.debug("save_raw failed (non-fatal)", exc_info=True)


@router.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a document to the private knowledge base."""
    kb = get_context().kb
    if kb is None:
        return {"error": "Knowledge base is not available"}
    try:
        content = await file.read()
        if len(content) > _MAX_UPLOAD_BYTES:
            return {
                "success": False,
                "error": f"File too large ({len(content)} bytes, max {_MAX_UPLOAD_BYTES})",
            }

        # 优先走 SmartIngester: 按文件类型自动选解析方式 (图片 OCR + CV
        # 分析, PDF 文本/OCR + 内嵌图片, CSV/JSON 摘要). 缺依赖时退回原逻辑.
        filename = file.filename or "unnamed"
        smart_result = None
        try:
            from huginn.knowledge.smart_ingest import build_smart_ingester

            ingester = build_smart_ingester(get_context().kb)
            if ingester is not None:
                smart_result = await ingester.ingest(filename, content)
        except Exception:
            logger.debug("SmartIngester 不可用, 退回原生 add_document", exc_info=True)

        if smart_result is not None:
            _sem_distill_upload(content, filename)
            # SmartIngester 用 add_text 入库不落原始文件, 这里补存一份供"原文"预览
            _save_raw_if_docid(kb, smart_result.get("doc_id", ""), filename, content)
            return {"success": True, "document": smart_result}

        # 退路: KB 原生摄入
        result = get_context().kb.add_document(filename, content)
        _sem_distill_upload(content, filename)
        _save_raw_if_docid(kb, result.get("doc_id", ""), filename, content)
        return {"success": True, "document": result}
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}


@router.get("/knowledge")
async def list_knowledge() -> dict[str, Any]:
    """List documents in the knowledge base."""
    if get_context().kb is None:
        return {"documents": [], "available": False}
    try:
        return {
            "documents": get_context().kb.list_documents(),
            "count": get_context().kb.count(),
            "available": True,
        }
    except Exception as e:
        return {"documents": [], "error": str(e), "available": False}


@router.post("/knowledge/ingest-url")
async def ingest_url(req: UrlIngestRequest) -> dict[str, Any]:
    """Fetch a web page and add it to the knowledge base."""
    if get_context().kb is None:
        return {"success": False, "error": "Knowledge base is not available"}

    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        return {"success": False, "error": "URL must start with http:// or https://"}

    # SSRF 防护: 复用 web_search_tool 的 _is_ssrf_blocked, 拒绝内网/loopback.
    # 之前这里直接 urlopen, 可被诱导请求 169.254.169.254 等元数据端点.
    from huginn.tools.web_search_tool import _is_ssrf_blocked

    blocked, reason = _is_ssrf_blocked(url)
    if blocked:
        return {"success": False, "error": f"URL blocked by SSRF protection: {reason}"}

    try:
        import re
        import urllib.request

        req_obj = urllib.request.Request(url, headers={"User-Agent": "Huginn/1.0"})
        # 关闭重定向跟随: 30x 到内网是已知 SSRF 绕过手段.
        # urllib 的 HTTPRedirectHandler 默认跟随, 这里装一个 no-op 替代.
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise urllib.error.HTTPError(newurl, code, msg, headers, fp)

        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(req_obj, timeout=30) as resp:
            raw = resp.read(5 * 1024 * 1024)  # 5 MB cap
            charset = resp.headers.get_content_charset() or "utf-8"
            html = raw.decode(charset, errors="replace")

        # crude HTML → text: strip tags, collapse whitespace
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S | re.I)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 50:
            return {"success": False, "error": "Page content too short after extraction"}

        # use URL-derived filename
        from urllib.parse import urlparse
        domain = urlparse(url).netloc or "web"
        filename = f"{domain}_{hash(url) % 100000}.txt"

        result = get_context().kb.add_document(filename, text.encode("utf-8"))
        return {"success": True, "document": result, "source_url": url}
    except Exception as e:
        logger.error("URL ingest failed", exc_info=True)
        return {"success": False, "error": str(e)}


@router.get("/knowledge/{doc_id}/chunks")
async def get_document_chunks(doc_id: str) -> dict[str, Any]:
    """Return a document's chunks in order (full-text/fragment preview)."""
    kb = get_context().kb
    if kb is None:
        return {"chunks": [], "error": "Knowledge base is not available"}
    try:
        return {"chunks": kb.get_document_chunks(doc_id)}
    except Exception as e:
        logger.error("get_document_chunks failed", exc_info=True)
        return {"chunks": [], "error": str(e)}


@router.delete("/knowledge/{doc_id}")
async def delete_knowledge(doc_id: str) -> dict[str, Any]:
    """Remove a document from the knowledge base."""
    if get_context().kb is None:
        return {"success": False, "error": "Knowledge base is not available"}
    try:
        deleted = get_context().kb.delete_document(doc_id)
        return {"success": True, "deleted": deleted}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/knowledge/image")
async def get_knowledge_image(path: str) -> Any:
    """Serve a knowledge-base image (currently only PDF visual-compression pages).

    Path must resolve under ``~/.huginn/compressed_pages/`` — we resolve and
    ``samefile``-check to defeat traversal before reading. 之前 image_ref 只
    塞进 agent prompt 文本, 前端拿不到. 现在前端按 image_url 直接 fetch.
    """
    # ponytail: 仅 compressed_pages/ 下的图能服务, 不开放任意路径读取
    try:
        from huginn.utils.runtime import get_runtime_home
        base = (get_runtime_home() / "compressed_pages").resolve()
    except Exception:
        base = (get_runtime_home() / "compressed_pages").resolve()

    try:
        target = Path(path).resolve()
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="invalid path") from None

    # 路径穿越防护: 解析后必须在 base 之下
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=403, detail="path outside compressed_pages") from None

    if not target.is_file():
        raise HTTPException(status_code=404, detail="image not found")

    return FileResponse(str(target))


@router.get("/knowledge/{doc_id}/raw")
async def get_document_raw(doc_id: str) -> Any:
    """下载/查看上传的原始文件 (docs_dir 里存的字节).

    doc_id 直接用 glob 拼, 天然防穿越 (只匹配 {doc_id}_*). 无原始文件
    (蒸馏/自动沉淀类没有上游文件) 时返回 404 + 可读提示.
    """
    kb = get_context().kb
    if kb is None:
        raise HTTPException(status_code=404, detail="知识库不可用")
    path = kb.get_raw(doc_id)
    if path is None:
        raise HTTPException(status_code=404, detail="该文档无原始文件（蒸馏/自动沉淀无上游文件）")
    return FileResponse(str(path), filename=path.name)


@router.get("/knowledge/{doc_id}/images")
async def get_document_images(doc_id: str) -> dict[str, Any]:
    """返回文档相关的可渲染图片列表 (压缩页图), 供前端图片画廊.

    压缩页 chunk 带 metadata.image_ref(相对 compressed_pages) + parent_doc_id
    指向主文档, 这里按 parent_doc_id 收拢成 [{url, page, caption}].
    """
    kb = get_context().kb
    if kb is None:
        return {"images": [], "error": "Knowledge base is not available"}
    try:
        data = kb.collection.get(
            where={"parent_doc_id": doc_id}, include=["metadatas", "documents"]
        )
        images: list[dict[str, Any]] = []
        for i, _ in enumerate(data.get("ids") or []):
            meta = (data.get("metadatas") or [])[i] or {}
            ref = meta.get("image_ref")
            if not ref:
                continue
            images.append({
                "url": f"/knowledge/image?path={str(ref)}",  # 相对 compressed_pages 根
                "page": meta.get("page"),
                "caption": ((data.get("documents") or [])[i] or "")[:400],
            })
        return {"images": images}
    except Exception as e:
        logger.debug("get_document_images failed", exc_info=True)
        return {"images": [], "error": str(e)}


@router.post("/knowledge/report")
async def generate_kb_report(req: ReportRequest) -> dict[str, Any]:
    """把选定文档的文本+数据摘要+图表说明自动汇总成一份 Markdown 报告.

    用 lead agent 一次性合成(不强跑工具循环), 生成结论不污染聊天历史
    (独立 thread_id). doc_ids 为空/无文本时返回可读错误, 不空跑 LLM.
    """
    kb = get_context().kb
    if kb is None:
        return {"success": False, "error": "Knowledge base is not available"}
    if not req.doc_ids:
        return {"success": False, "error": "doc_ids 不能为空"}

    # 收拢各文档文本 + 图片说明, 供报告引用 (限制总量避免上下文爆炸)
    sections: list[str] = []
    total_chars = 0
    cap = 60_000
    for doc_id in req.doc_ids:
        try:
            chunks = kb.get_document_chunks(doc_id)
        except Exception:
            continue
        if not chunks:
            continue
        fname = ""
        text_parts: list[str] = []
        for c in chunks:
            fname = c.get("metadata", {}).get("filename", "") or fname
            seg = c.get("text", "").strip()
            if not seg:
                continue
            if total_chars + len(seg) > cap:
                text_parts.append("…(篇幅截断)")
                break
            text_parts.append(seg)
            total_chars += len(seg)
        body = "\n\n".join(text_parts)
        if body:
            sections.append(f"## 来源文档：{fname}\n\n{body}")
    if not sections:
        return {"success": False, "error": "所选文档没有可汇总的文本内容"}

    title = req.title.strip() or "知识库综合报告"
    prompt = (
        f"请基于下面给出的多份资料，撰写一份结构化的 Markdown 综合报告《{title}》。\n"
        "要求：分『核心结论 / 分项数据与证据 / 图表与图像说明 / 关联与矛盾点 / 建议下一步』"
        "五个小节；提炼关键数据和数值，标注来自哪份资料；不要编造资料里没有的数据；"
        "只输出 Markdown，不要对话寒暄。\n\n资料如下：\n\n"
        + "\n\n".join(sections)
    )

    agent = getattr(get_context(), "agent", None)
    if agent is None:
        return {"success": False, "error": "Lead agent 未就绪，无法生成报告"}
    try:
        import asyncio as _aio

        agent._permission_config.auto_approve_all = True
        # invoke 内部用 asyncio.run, 放线程池跑; 独立 thread_id 避免污染默认对话
        state = await _aio.wait_for(
            _aio.to_thread(agent.invoke, prompt, thread_id="kb_report"),
            timeout=240,
        )
        messages = state.get("messages", [])
        content = ""
        if messages and hasattr(messages[-1], "content"):
            content = messages[-1].content
        if not content:
            return {"success": False, "error": "报告生成返回为空（可能超时）"}
        return {"success": True, "report": content, "title": title}
    except Exception as e:
        logger.warning("kb report generation failed: %s", e)
        return {"success": False, "error": f"报告生成失败: {e}"}


@router.get("/export")
async def export_data(
    source: str = "audit",
    fmt: str = "json",
) -> Any:
    """Export Huginn records as a downloadable file."""
    # 白名单校验, 防止路径穿越 (source 直接拼进文件名)
    ALLOWED_SOURCES = {"audit", "remote_jobs", "knowledge", "checkpoints", "provenance", "trajectories"}
    if source not in ALLOWED_SOURCES:
        raise HTTPException(status_code=400, detail=f"Invalid source: {source}")
    # defense-in-depth: 剥掉可能的路径分隔符
    source = Path(source).name

    from huginn.export_manager import ExportManager

    manager = ExportManager(get_context().config.workspace)
    suffix = "md" if fmt == "markdown" else fmt
    output = Path(tempfile.gettempdir()) / f"huginn_export_{source}.{suffix}"
    result = manager.export(source=source, output_path=output, fmt=fmt)
    return FileResponse(
        result.output_path,
        filename=result.output_path.name,
        media_type="application/octet-stream",
    )


@router.post("/knowledge/query")
async def query_knowledge(params: KnowledgeQuery) -> dict[str, Any]:
    """Query the knowledge base and return relevant chunks."""
    if get_context().kb is None:
        return {"chunks": [], "error": "Knowledge base is not available"}
    try:
        chunks = get_context().kb.query(params.query, top_k=params.top_k)
        return {"chunks": chunks}
    except Exception as e:
        return {"chunks": [], "error": str(e)}
