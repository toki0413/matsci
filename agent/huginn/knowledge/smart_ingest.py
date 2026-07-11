"""智能摄入 — 按文件类型路由到最合适的解析方式.

上传文件进来后, 不是无脑塞进 KB, 而是先看类型:
- 图片: OCR 抠文字 + (有 image_analysis_tool 时) 自动跑 SEM/图表分析
- PDF: pymupdf 抠文字, 文字太少 (扫描件) 就 OCR, 顺带把内嵌图片拎出来逐个分析
- CSV/JSON: 解析 + 生成数据摘要 (列名/行数/类型/统计), 摘要和原文都入库
- 其他 (TXT/DOCX/MD): 直接提取文本

分析失败不影响 OCR 文本入库, 走 try/except 包裹.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 扩展名分类, 用小写匹配
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
_DATA_SUFFIXES = {".csv", ".json", ".jsonl", ".tsv"}
_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log"}
# 压缩包格式 — 解压后递归摄入每个文件
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".gz", ".7z", ".rar"}
_STRUCTURE_SUFFIXES = {".cif", ".poscar", ".contcar", ".vasp"}

# PDF 文本量低于这个阈值认为是扫描件, 走 OCR
_PDF_OCR_THRESHOLD = 100


class SmartIngester:
    """根据文件类型自动选择解析路径的摄入器."""

    def __init__(
        self,
        kb: Any,
        image_analysis_tool: Any | None = None,
        vision_available: bool = False,
    ) -> None:
        self.kb = kb
        self.image_analysis_tool = image_analysis_tool
        self.vision_available = vision_available

    # ── 入口: 按类型路由 ──────────────────────────────────────────

    async def ingest(self, filename: str, content: bytes) -> dict[str, Any]:
        """智能摄入入口, 根据文件扩展名路由到对应解析器.
        支持压缩包: zip/tar.gz 等会解压后递归摄入每个文件.
        支持结构文件: CIF/POSCAR 走 StructureChunker 结构化分块.
        """
        suffix = Path(filename).suffix.lower()
        # tar.gz / tar.bz2 双扩展名特殊处理
        lower_name = filename.lower()
        if lower_name.endswith((".tar.gz", ".tar.bz2")):
            suffix = "." + ".".join(lower_name.rsplit(".", 2)[-2:])
        try:
            if suffix in _ARCHIVE_SUFFIXES or lower_name.endswith((".tar.gz", ".tar.bz2")):
                return await self.ingest_archive(filename, content)
            if suffix in _IMAGE_SUFFIXES:
                return await self.ingest_image(filename, content)
            if suffix == ".pdf":
                return await self.ingest_pdf(filename, content)
            if suffix in _DATA_SUFFIXES:
                return await self.ingest_data(filename, content)
            if suffix in _STRUCTURE_SUFFIXES:
                return await self.ingest_structure(filename, content)
            return await self.ingest_text(filename, content)
        except Exception as exc:
            logger.warning("smart ingest '%s' 失败, 退回普通摄入: %s", filename, exc)
            # 兜底: 直接走 KB 原生 add_document, 别让上传整个挂掉
            try:
                result = self.kb.add_document(filename, content)
                result["smart_ingest"] = False
                result["fallback_reason"] = str(exc)
                return result
            except Exception:
                return {"doc_id": "", "smart_ingest": False, "error": str(exc)}

    # ── 压缩包: 解压后递归摄入 ─────────────────────────────────────

    async def ingest_archive(self, filename: str, content: bytes) -> dict[str, Any]:
        """解压压缩包, 对每个文件递归调用 ingest.
        支持 zip, tar.gz, tar.bz2, tar. 7z/rar 在有对应工具时也支持.
        这是 loop-of-loops: archive → files → each file is its own ingest loop.
        """
        import tempfile
        import zipfile
        import tarfile
        import shutil

        results: list[dict[str, Any]] = []
        total_files = 0
        total_chunks = 0

        with tempfile.TemporaryDirectory(prefix="huginn_archive_") as tmpdir:
            tmp_path = Path(tmpdir)
            archive_file = tmp_path / filename
            archive_file.write_bytes(content)

            extract_dir = tmp_path / "extracted"
            extract_dir.mkdir()

            lower = filename.lower()
            try:
                if lower.endswith(".zip"):
                    with zipfile.ZipFile(archive_file, "r") as zf:
                        zf.extractall(extract_dir)
                elif lower.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar")):
                    mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else \
                           "r:bz2" if lower.endswith((".tar.bz2", ".tbz2")) else "r:"
                    with tarfile.open(archive_file, mode) as tf:
                        tf.extractall(extract_dir)
                elif lower.endswith(".gz"):
                    import gzip
                    out_name = Path(filename).stem  # remove .gz
                    with gzip.open(archive_file, "rb") as gz:
                        (extract_dir / out_name).write_bytes(gz.read())
                elif lower.endswith(".7z"):
                    try:
                        import py7zr
                        with py7zr.SevenZipFile(archive_file, "r") as sz:
                            sz.extractall(extract_dir)
                    except ImportError:
                        return {"doc_id": "", "smart_ingest": False,
                                "error": "7z files require py7zr: pip install py7zr"}
                elif lower.endswith(".rar"):
                    try:
                        import rarfile
                        with rarfile.RarFile(archive_file) as rf:
                            rf.extractall(extract_dir)
                    except ImportError:
                        return {"doc_id": "", "smart_ingest": False,
                                "error": "rar files require rarfile: pip install rarfile"}
                else:
                    return {"doc_id": "", "smart_ingest": False,
                            "error": f"unsupported archive format: {filename}"}
            except Exception as exc:
                return {"doc_id": "", "smart_ingest": False,
                        "error": f"archive extraction failed: {exc}"}

            # 递归摄入每个文件 — 这就是内层 loop
            for inner_file in sorted(extract_dir.rglob("*")):
                if not inner_file.is_file():
                    continue
                # 跳过隐藏文件和 __MACOSX 等
                if any(part.startswith(".") or part == "__MACOSX" for part in inner_file.parts):
                    continue
                try:
                    inner_content = inner_file.read_bytes()
                    # 限制单文件 100MB 防止解压炸弹
                    if len(inner_content) > 100 * 1024 * 1024:
                        logger.warning("skip oversized file in archive: %s (%d bytes)",
                                       inner_file.name, len(inner_content))
                        continue
                    inner_result = await self.ingest(inner_file.name, inner_content)
                    results.append(inner_result)
                    total_files += 1
                    total_chunks += inner_result.get("n_chunks", 0)
                except Exception as exc:
                    logger.warning("failed to ingest %s from archive: %s",
                                   inner_file.name, exc)
                    results.append({"filename": inner_file.name, "error": str(exc)})

        return {
            "doc_id": f"archive:{filename}",
            "smart_ingest": True,
            "archive": True,
            "n_files": total_files,
            "n_chunks": total_chunks,
            "files": [r.get("filename", r.get("doc_id", "")) for r in results],
            "errors": [r for r in results if "error" in r],
        }

    # ── 结构文件: CIF/POSCAR 结构化分块 ──────────────────────────────

    async def ingest_structure(self, filename: str, content: bytes) -> dict[str, Any]:
        """CIF/POSCAR 结构文件: 提取结构化元数据 + 文本入库.
        激活了 chunker.py 里死掉的 StructureChunker — 它能解析空间群、
        晶格参数、原子坐标, 生成结构化摘要. 这比直接 UTF-8 入库信息密度高."""
        try:
            from huginn.knowledge.chunker import StructureChunker
            text = content.decode("utf-8", errors="ignore")
            chunks = StructureChunker.chunk(text, filename=filename)
            if not chunks:
                # StructureChunker 无法解析, 退回普通文本摄入
                return await self.ingest_text(filename, content)

            # 把结构化摘要 + 原始文本都入库
            all_texts = []
            for chunk in chunks:
                all_texts.append(chunk.text if hasattr(chunk, "text") else str(chunk))

            combined = f"# Structure: {filename}\n\n"
            combined += "\n\n---\n\n".join(all_texts)
            result = self.kb.add_document(filename, combined.encode("utf-8"))
            result["smart_ingest"] = True
            result["structure_aware"] = True
            result["n_chunks"] = len(all_texts)
            return result
        except ImportError:
            # StructureChunker 不可用, 退回普通文本
            return await self.ingest_text(filename, content)
        except Exception as exc:
            logger.warning("structure ingest failed for %s: %s", filename, exc)
            return await self.ingest_text(filename, content)

    # ── 图片 ──────────────────────────────────────────────────────

    async def ingest_image(self, filename: str, content: bytes) -> dict[str, Any]:
        """OCR 提取文本 + (可选) 视觉分析, 一起入库."""
        from huginn.knowledge.ocr_loader import extract_text_with_ocr

        ocr_text = extract_text_with_ocr(filename, content)
        cv_analysis: dict[str, Any] | None = None

        if self.image_analysis_tool is not None:
            cv_analysis = await self._run_cv_analysis(filename, content)

        # 把 OCR 文本和分析结果拼成一份 markdown 入库
        parts: list[str] = []
        if ocr_text.strip():
            parts.append(f"# OCR 文本 ({filename})\n\n{ocr_text}")
        if cv_analysis:
            parts.append(self._format_cv_analysis(cv_analysis))

        combined = "\n\n".join(parts)
        info = self.kb.add_text(
            combined or "(空)",
            filename=f"{filename}.ocr.md",
            metadata={"source": "smart_ingest", "file_type": "image"},
        )

        return {
            "doc_id": info.get("doc_id", ""),
            "ocr_text_length": len(ocr_text),
            "cv_analysis": cv_analysis,
            "smart_ingest": True,
        }

    async def _run_cv_analysis(
        self, filename: str, content: bytes
    ) -> dict[str, Any] | None:
        """保存临时文件, 调 image_analysis_tool 做材料科学分析.

        根据文件名/内容猜图片类型 (SEM/TEM/图表), 选对应 action.
        失败返回 None, 不影响 OCR 文本入库.
        """
        if self.image_analysis_tool is None:
            return None

        # 落临时文件给工具读 (image_analysis_tool 要 image_path)
        suffix = Path(filename).suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(content)
            tmp.close()
            return await self._dispatch_cv_actions(tmp.name, filename)
        except Exception as exc:
            logger.debug("CV 分析失败 (非致命): %s", exc)
            return None
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    async def _dispatch_cv_actions(
        self, image_path: str, original_name: str
    ) -> dict[str, Any]:
        """根据图片类型猜选调对应 action, 哪个能跑就跑哪个.

        策略: 文件名命中 sem/tem 关键词就跑对应显微分析; 命中 plot/xrd
        走 plot_extract; 拿不准就两个都试 (plot_extract + sem), 失败的吞掉.
        """
        name_lower = original_name.lower()
        results: dict[str, Any] = {}

        # 文件名关键词决定优先 action
        actions: list[tuple[str, dict[str, Any]]] = []
        if any(kw in name_lower for kw in ("sem", "fesem", "sem_photo")):
            actions.append(("sem_analysis", {}))
        elif any(kw in name_lower for kw in ("tem", "hrtem", "stem")):
            actions.append(("tem_lattice", {}))
        elif any(kw in name_lower for kw in ("xrd", "plot", "curve", "chart", "spectr")):
            actions.append(("plot_extract", {}))
        else:
            # 拿不准: 图表提取和 SEM 各试一次, 谁成功用谁
            actions.append(("plot_extract", {}))
            actions.append(("sem_analysis", {}))

        for action, params in actions:
            key = action
            if key in results:
                continue
            try:
                res = await self._call_image_tool(image_path, action, params)
                if res is not None:
                    results[key] = res
            except Exception as exc:
                logger.debug("action %s 失败 (跳过): %s", action, exc)

        return results or None

    async def _call_image_tool(
        self, image_path: str, action: str, parameters: dict[str, Any]
    ) -> dict[str, Any] | None:
        """封装一次 image_analysis_tool 调用, 返回 data dict 或 None."""
        tool = self.image_analysis_tool
        args = {
            "image_path": image_path,
            "action": action,
            "parameters": parameters,
        }
        context = self._make_tool_context()
        # HuginnTool.call 是 async 的
        result = await tool.call(args, context)
        if result is None:
            return None
        success = getattr(result, "success", False)
        if not success:
            return None
        data = getattr(result, "data", None)
        return data if isinstance(data, dict) else None

    def _make_tool_context(self) -> Any:
        """建一个最小 ToolContext 给 image_analysis_tool 用."""
        from huginn.types import ToolContext

        return ToolContext(
            session_id="smart_ingest",
            workspace=tempfile.gettempdir(),
        )

    @staticmethod
    def _format_cv_analysis(analysis: dict[str, Any]) -> str:
        """把 CV 分析结果格式化成 markdown 片段入库."""
        import json

        lines = ["# 图像分析结果"]
        for action, data in analysis.items():
            lines.append(f"\n## {action}\n")
            try:
                lines.append("```json")
                lines.append(json.dumps(data, ensure_ascii=False, indent=2, default=str))
                lines.append("```")
            except Exception:
                lines.append(str(data))
        return "\n".join(lines)

    # ── PDF ───────────────────────────────────────────────────────

    async def ingest_pdf(self, filename: str, content: bytes) -> dict[str, Any]:
        """pymupdf 抠文字, 太少就 OCR; 内嵌图片逐个 ingest_image."""
        try:
            import fitz  # pymupdf
        except ImportError as exc:
            raise RuntimeError(
                "PDF 摄入需要 pymupdf. Install: pip install pymupdf"
            ) from exc

        text_parts: list[str] = []
        images_analyzed = 0
        doc = None
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)

            # 文本太少 -> 扫描件, 走 OCR
            if len("\n".join(text_parts).strip()) < _PDF_OCR_THRESHOLD:
                from huginn.knowledge.ocr_loader import extract_text_with_ocr

                ocr_text = extract_text_with_ocr(filename, content)
                if ocr_text.strip():
                    text_parts.insert(0, f"[OCR]\n{ocr_text}")

            # 提取内嵌图片, 每张跑一次图像摄入 (OCR + CV 分析)
            if self.image_analysis_tool is not None:
                images_analyzed = await self._extract_and_ingest_pdf_images(doc)
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

        full_text = "\n\n".join(p for p in text_parts if p and p.strip())
        info = self.kb.add_text(
            full_text or "(空 PDF)",
            filename=f"{filename}.txt",
            metadata={"source": "smart_ingest", "file_type": "pdf"},
        )

        return {
            "doc_id": info.get("doc_id", ""),
            "text_length": len(full_text),
            "images_analyzed": images_analyzed,
            "smart_ingest": True,
        }

    async def _extract_and_ingest_pdf_images(self, doc: Any) -> int:
        """把 PDF 内嵌图片抠出来逐个 ingest_image, 返回处理数量."""
        count = 0
        for page_index in range(len(doc)):
            try:
                page = doc[page_index]
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    try:
                        pix = doc.extract_image(xref)
                    except Exception:
                        continue
                    img_bytes = pix.get("image")
                    if not img_bytes:
                        continue
                    ext = pix.get("ext", "png")
                    img_name = f"pdf_page{page_index}_{xref}.{ext}"
                    try:
                        await self.ingest_image(img_name, img_bytes)
                        count += 1
                    except Exception as exc:
                        logger.debug("PDF 内嵌图片摄入失败 (跳过): %s", exc)
            except Exception as exc:
                logger.debug("PDF 页 %d 图片提取失败: %s", page_index, exc)
        return count

    # ── 数据文件 (CSV/JSON) ────────────────────────────────────────

    async def ingest_data(self, filename: str, content: bytes) -> dict[str, Any]:
        """解析 CSV/JSON, 生成摘要, 摘要 + 原始数据一起入库."""
        suffix = Path(filename).suffix.lower()
        text = content.decode("utf-8", errors="ignore")

        rows = 0
        columns: list[str] = []
        dtypes: dict[str, str] = {}
        stats: dict[str, Any] = {}
        summary_lines: list[str] = []

        if suffix == ".csv" or suffix == ".tsv":
            delimiter = "\t" if suffix == ".tsv" else ","
            rows, columns, dtypes, stats, summary_lines = self._summarize_csv(
                text, delimiter
            )
        elif suffix in (".json", ".jsonl"):
            rows, columns, dtypes, stats, summary_lines = self._summarize_json(
                text, suffix == ".jsonl"
            )

        # 摘要 + 原始数据拼一起入库
        combined = "# 数据摘要\n\n" + "\n".join(summary_lines)
        combined += f"\n\n# 原始数据 ({filename})\n\n```\n{text[:50000]}\n```"
        info = self.kb.add_text(
            combined,
            filename=f"{filename}.summary.md",
            metadata={"source": "smart_ingest", "file_type": "data"},
        )

        return {
            "doc_id": info.get("doc_id", ""),
            "rows": rows,
            "columns": len(columns),
            "column_names": columns,
            "dtypes": dtypes,
            "stats": stats,
            "smart_ingest": True,
        }

    @staticmethod
    def _summarize_csv(
        text: str, delimiter: str
    ) -> tuple[int, list[str], dict[str, str], dict[str, Any], list[str]]:
        """用 csv 模块统计 CSV 结构, 返回 (行数, 列名, 类型, 统计, 摘要行)."""
        import csv

        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows_list = list(reader)
        if not rows_list:
            return 0, [], {}, {}, ["(空文件)"]

        header = rows_list[0]
        data_rows = rows_list[1:]
        columns = header

        # 推断每列类型: 扫前 50 行, 看 int/float/str
        dtypes: dict[str, str] = {}
        for col_idx, col_name in enumerate(header):
            dtypes[col_name] = SmartIngester._infer_column_type(
                data_rows, col_idx, limit=50
            )

        # 数值列做简单统计 (min/max/mean)
        stats: dict[str, Any] = {}
        for col_idx, col_name in enumerate(header):
            if dtypes[col_name] != "number":
                continue
            values: list[float] = []
            for row in data_rows[:200]:
                if col_idx < len(row):
                    try:
                        values.append(float(row[col_idx]))
                    except (ValueError, TypeError):
                        continue
            if values:
                stats[col_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "count": len(values),
                }

        summary = [
            f"- 文件类型: CSV",
            f"- 行数 (不含表头): {len(data_rows)}",
            f"- 列数: {len(columns)}",
            f"- 列名: {', '.join(columns)}",
            f"- 列类型: {dtypes}",
        ]
        if stats:
            summary.append("- 数值统计:")
            for col, s in stats.items():
                summary.append(
                    f"  - {col}: min={s['min']:.4g}, max={s['max']:.4g}, "
                    f"mean={s['mean']:.4g} (n={s['count']})"
                )
        return len(data_rows), columns, dtypes, stats, summary

    @staticmethod
    def _summarize_json(
        text: str, is_jsonl: bool
    ) -> tuple[int, list[str], dict[str, str], dict[str, Any], list[str]]:
        """解析 JSON / JSONL, 返回结构摘要."""
        import json

        rows = 0
        columns: list[str] = []
        dtypes: dict[str, str] = {}
        stats: dict[str, Any] = {}

        if is_jsonl:
            records: list[dict] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    continue
            rows = len(records)
            if records:
                # 合并所有 key 作为列
                key_set: dict[str, None] = {}
                for rec in records:
                    for k in rec:
                        key_set.setdefault(k, None)
                columns = list(key_set.keys())
                # 取第一条记录推断类型
                for col in columns:
                    val = records[0].get(col)
                    dtypes[col] = SmartIngester._python_type_name(val)
        else:
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return 0, [], {}, {}, ["(无效 JSON)"]
            if isinstance(obj, list):
                rows = len(obj)
                if obj and isinstance(obj[0], dict):
                    columns = list(obj[0].keys())
                    for col in columns:
                        dtypes[col] = SmartIngester._python_type_name(obj[0].get(col))
            elif isinstance(obj, dict):
                rows = 1
                columns = list(obj.keys())
                for col in columns:
                    dtypes[col] = SmartIngester._python_type_name(obj.get(col))
            else:
                rows = 1
                columns = ["value"]
                dtypes["value"] = SmartIngester._python_type_name(obj)

        summary = [
            f"- 文件类型: {'JSONL' if is_jsonl else 'JSON'}",
            f"- 记录数: {rows}",
            f"- 字段: {', '.join(columns) if columns else '(无)'}",
            f"- 字段类型: {dtypes}",
        ]
        return rows, columns, dtypes, stats, summary

    @staticmethod
    def _infer_column_type(
        rows: list[list[str]], col_idx: int, limit: int = 50
    ) -> str:
        """扫前 limit 行, 判断一列是 number 还是 string."""
        seen_num = 0
        seen_total = 0
        for row in rows[:limit]:
            if col_idx >= len(row):
                continue
            val = row[col_idx].strip()
            if not val:
                continue
            seen_total += 1
            try:
                float(val)
                seen_num += 1
            except ValueError:
                pass
        if seen_total == 0:
            return "empty"
        return "number" if seen_num / seen_total > 0.8 else "string"

    @staticmethod
    def _python_type_name(val: Any) -> str:
        """Python 值 -> 类型名, 给摘要用."""
        if isinstance(val, bool):
            return "bool"
        if isinstance(val, (int, float)):
            return "number"
        if isinstance(val, str):
            return "string"
        if isinstance(val, list):
            return "list"
        if isinstance(val, dict):
            return "object"
        if val is None:
            return "null"
        return type(val).__name__

    # ── 纯文本 / 其他 ─────────────────────────────────────────────

    async def ingest_text(self, filename: str, content: bytes) -> dict[str, Any]:
        """TXT/MD/DOCX 等直接提取文本入库 (走 KB 原生 add_document)."""
        result = self.kb.add_document(filename, content)
        result["smart_ingest"] = True
        result["file_type"] = "text"
        return result


def build_smart_ingester(kb: Any | None) -> SmartIngester | None:
    """从 server context 里凑齐依赖, 建一个 SmartIngester.

    KB 缺失直接返回 None (调用方退回原逻辑). image_analysis_tool 从
    ToolRegistry 取, 取不到就 None (图片只走 OCR 不走 CV 分析).
    """
    if kb is None:
        return None

    image_tool = None
    vision_available = False
    try:
        from huginn.tools.registry import ToolRegistry

        image_tool = ToolRegistry.get("image_analysis_tool")
    except Exception:
        logger.debug("image_analysis_tool 不可用, smart ingest 只走 OCR")

    try:
        from huginn.models.registry import get_model_capabilities

        # ponytail: 这里拿不到具体 model_name, 保守认为 vision 不可用,
        # 让 SmartIngester 只依赖 image_analysis_tool. 升级路径: 传入当前 model.
        vision_available = False
    except Exception:
        vision_available = False

    return SmartIngester(
        kb=kb,
        image_analysis_tool=image_tool,
        vision_available=vision_available,
    )
