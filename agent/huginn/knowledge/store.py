"""Local RAG knowledge base with ChromaDB and sentence-transformers."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huginn.utils.cache import TimedLRUCache
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "all-MiniLM-L6-v2"
SEED_DIR = Path(__file__).parent / "seed"


class _EmbeddingModel:
    """Wrapper that prefers ChromaDB's cached ONNX embedder, falling back to sentence-transformers.

    The underlying embedding models are loaded once and reused across
    ``KnowledgeBase`` instances to avoid repeated initialization overhead.
    Computed embeddings are also cached by content hash so identical documents
    do not need to be re-encoded.
    """

    _ef: Any | None = None
    _st: Any | None = None
    _use_chroma: bool = False
    _initialized: bool = False
    _embedding_cache: TimedLRUCache[np.ndarray] = TimedLRUCache(
        max_size=1024, ttl=3600.0
    )

    def __init__(self) -> None:
        if not _EmbeddingModel._initialized:
            try:
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

                ef = DefaultEmbeddingFunction()
                _ = ef(["test"])
                _EmbeddingModel._ef = ef
                _EmbeddingModel._use_chroma = True
            except Exception:
                _EmbeddingModel._use_chroma = False
            _EmbeddingModel._initialized = True

    def encode(self, texts: list[str], cache_key: str | None = None) -> "np.ndarray":
        if cache_key:
            cached = _EmbeddingModel._embedding_cache.get(cache_key)
            if cached is not None:
                return cached

        import numpy as np

        if _EmbeddingModel._use_chroma and _EmbeddingModel._ef is not None:
            vectors = _EmbeddingModel._ef(texts)
            result = np.asarray(vectors, dtype=np.float32)
        else:
            if _EmbeddingModel._st is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    raise RuntimeError(
                        "Embedding requires sentence-transformers or chromadb's default embedder. "
                        "Install: pip install sentence-transformers"
                    ) from e
                _EmbeddingModel._st = SentenceTransformer(EMBED_MODEL)
            result = _EmbeddingModel._st.encode(texts)

        if cache_key:
            _EmbeddingModel._embedding_cache.set(cache_key, result)
        return result


def _chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Simple sliding-window chunking by character."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _extract_text(filename: str, content: bytes) -> str:
    """Extract plain text from supported file types.

    Images and scanned PDFs fall back to OCR when normal extraction is empty.
    """
    from huginn.knowledge.ocr_loader import extract_text_with_ocr, is_image_file

    lower = filename.lower()
    if is_image_file(filename):
        return extract_text_with_ocr(filename, content)

    if lower.endswith(".pdf"):
        try:
            import fitz  # pymupdf
        except ImportError as e:
            raise RuntimeError(
                "PDF support requires pymupdf. Install: pip install pymupdf"
            ) from e
        doc = fitz.open(stream=content, filetype="pdf")
        parts = []
        for page in doc:
            parts.append(page.get_text())
        text = "\n".join(parts)
        # Fall back to OCR for scanned/image-based PDFs.
        if not text.strip():
            ocr_text = extract_text_with_ocr(filename, content)
            if ocr_text.strip():
                return ocr_text
        return text

    if lower.endswith((".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml")):
        return content.decode("utf-8", errors="ignore")

    # Best-effort for anything else
    return content.decode("utf-8", errors="ignore")


# ── 材料科学领域标签树 ────────────────────────────────────────────────
# Easy Dataset 启发: 一级领域 + 二级子领域, 关键词匹配自动打标

DOMAIN_TAG_TREE: dict[str, list[str]] = {
    "合金": ["高温合金", "轻合金", "高熵合金"],
    "半导体": ["宽禁带半导体", "热电半导体", "有机半导体"],
    "催化": ["电催化", "光催化", "热催化"],
    "能源材料": ["电池材料", "储氢材料", "超级电容器"],
    "生物材料": ["生物陶瓷", "生物聚合物", "生物活性涂层"],
    "机械工程": ["粉末冶金", "塑性加工", "增材制造", "热处理", "机械设计"],
}

# 一级 / 二级标签对应的关键词, 命中即打标. 不追求全, 覆盖常见场景够用
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    # 合金
    "合金": ["合金", "alloy", "superalloy", "金属间化合物", "intermetallic"],
    "高温合金": ["高温合金", "superalloy", "nickel-based", "镍基", "蠕变"],
    "轻合金": ["轻合金", "magnesium alloy", "镁合金", "titanium alloy", "钛合金", "铝合金", "aluminum alloy"],
    "高熵合金": ["高熵合金", "high-entropy", "HEA", "multi-principal"],
    # 半导体
    "半导体": ["半导体", "semiconductor", "带隙", "band gap", "载流子", "carrier", "doping", "掺杂"],
    "宽禁带半导体": ["宽禁带", "wide bandgap", "GaN", "SiC", "ZnO", "wide-gap"],
    "热电半导体": ["热电", "thermoelectric", "Seebeck", "塞贝克", "ZT"],
    "有机半导体": ["有机半导体", "organic semiconductor", "OFET", "OPV", "导电聚合物"],
    # 催化
    "催化": ["催化", "catalys", "催化活性", "活性位点", "active site", "吸附能", "adsorption"],
    "电催化": ["电催化", "electrocatalys", "ORR", "OER", "HER", "析氢", "析氧"],
    "光催化": ["光催化", "photocatalys", "光降解", "光分解水"],
    "热催化": ["热催化", "thermocatalys", "费托", "Fischer-Tropsch", "甲烷化"],
    # 能源材料
    "能源材料": ["能源材料", "energy storage", "电池", "battery", "capacitor", "储能"],
    "电池材料": ["电池", "battery", "正极", "cathode", "负极", "anode", "电解质", "electrolyte", "锂离子", "Li-ion"],
    "储氢材料": ["储氢", "hydrogen storage", "metal hydride", "金属氢化物"],
    "超级电容器": ["超级电容", "supercapacitor", "双电层电容", "EDLC"],
    # 生物材料
    "生物材料": ["生物材料", "biomaterial", "biocompatib", "生物相容", "植入"],
    "生物陶瓷": ["生物陶瓷", "bioceramic", "羟基磷灰石", "hydroxyapatite", "HAP"],
    "生物聚合物": ["生物聚合物", "biopolymer", "聚乳酸", "PLA", "壳聚糖", "chitosan"],
    "生物活性涂层": ["生物活性涂层", "bioactive coating", "表面改性", "生物涂层"],
    # 机械工程 — 域级关键词要覆盖各子领域常见词, 否则只提子领域不提"机械工程"的文本打不上域标签
    "机械工程": [
        "机械工程", "mechanical engineering", "机械设计", "mechanical design",
        "应力分析", "stress analysis", "制造工艺", "粉末冶金", "powder metallurgy",
        "塑性加工", "增材制造", "additive manufacturing", "热处理", "heat treatment",
        "疲劳", "fatigue", "轧制", "rolling", "烧结", "sintering", "锻造", "forging",
        "淬火", "quenching", "3D打印", "3D printing", "磨损", "wear",
    ],
    "粉末冶金": ["粉末冶金", "powder metallurgy", "压制", "die compaction", "烧结", "sintering", "HIP", "热等静压", "CIP", "冷等静压"],
    "塑性加工": ["塑性加工", "rolling", "轧制", "extrusion", "挤压", "forging", "锻造", "drawing", "拉拔", "板料", "sheet metal"],
    "增材制造": ["增材制造", "additive manufacturing", "3D printing", "3D打印", "SLM", "选择性激光熔化", "SLS", "选择性激光烧结", "FDM", "熔融沉积", "送粉", "scan path", "扫描路径"],
    "热处理": ["热处理", "heat treatment", "退火", "annealing", "淬火", "quenching", "回火", "tempering", "炉", "furnace"],
    "机械设计": ["应力分析", "stress analysis", "疲劳", "fatigue", "断裂", "fracture", "磨损", "wear", "摩擦学", "tribology"],
}


def auto_tag(text: str) -> dict[str, Any]:
    """关键词匹配给文档自动打领域标签.

    返回 {"domain_tags": [...], "sub_domain_tags": [...]}.
    domain_tags 是一级标签, sub_domain_tags 是二级标签.
    匹配不到任何标签时 domain_tags 为空列表.
    """
    if not text:
        return {"domain_tags": [], "sub_domain_tags": []}
    text_lower = text.lower()
    domain_tags: list[str] = []
    sub_tags: list[str] = []

    for domain, sub_domains in DOMAIN_TAG_TREE.items():
        kws = _DOMAIN_KEYWORDS.get(domain, [])
        if any(kw.lower() in text_lower for kw in kws):
            domain_tags.append(domain)
            for sub in sub_domains:
                sub_kws = _DOMAIN_KEYWORDS.get(sub, [])
                if any(kw.lower() in text_lower for kw in sub_kws):
                    sub_tags.append(sub)

    return {"domain_tags": domain_tags, "sub_domain_tags": sub_tags}


# ── 章节感知分块 ──────────────────────────────────────────────────────
# Markdown 按 ## / ### 分块, PDF 按章节标记分块, 纯文本退回固定长度

_MD_HEADING_RE = re.compile(r'^(#{2,3})\s+(.+)$', re.MULTILINE)
# PDF 章节常见模式: "Chapter 1", "第3章", "1. Introduction", "2.1 Methods"
_PDF_CHAPTER_RE = re.compile(
    r'^(?:Chapter\s+\d+|第[一二三四五六七八九十百\d]+章'
    r'|\d{1,2}\.\s+[A-Z]\w+|\d{1,2}\.\d{1,2}\s+\w+)',
    re.MULTILINE,
)


def _section_aware_chunk(
    text: str, filename: str = ""
) -> list[tuple[str, dict[str, Any]]]:
    """根据文件类型做章节感知分块.

    返回 [(chunk_text, metadata), ...], metadata 包含 section 和 chunk_type.
    Markdown 按 ## / ### 标题切; PDF 按章节标记切; 其他退回固定长度.
    """
    lower = filename.lower()

    if lower.endswith((".md", ".markdown")):
        return _chunk_markdown_sections(text)

    if lower.endswith(".pdf"):
        return _chunk_pdf_sections(text)

    # 纯文本 / 代码 / 配置文件: 固定长度分块 (原有行为)
    return [
        (c, {"section": "", "chunk_type": "fixed"})
        for c in _chunk_text(text)
    ]


def _chunk_markdown_sections(
    text: str,
) -> list[tuple[str, dict[str, Any]]]:
    """按 ## 和 ### 标题分块, 保留标题层级路径."""
    headings = list(_MD_HEADING_RE.finditer(text))

    if not headings:
        # 没有标题结构, 退回固定长度
        return [
            (c, {"section": "", "chunk_type": "fixed"})
            for c in _chunk_text(text)
        ]

    chunks: list[tuple[str, dict[str, Any]]] = []

    # 第一个标题之前的内容也存一块 (通常是标题 / 摘要)
    if headings[0].start() > 0:
        preamble = text[: headings[0].start()].strip()
        if preamble:
            chunks.append((preamble, {"section": "", "chunk_type": "preamble"}))

    section_path: list[str] = []

    for i, match in enumerate(headings):
        level = len(match.group(1))  # 2 → ##, 3 → ###
        title = match.group(2).strip()

        # 维护标题层级: level 2 是顶层, level 3 是子层
        while len(section_path) >= level - 1:
            section_path.pop()
        section_path.append(title)

        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        content = text[start:end].strip()

        if not content:
            continue

        section_str = " > ".join(section_path)

        if len(content) > CHUNK_SIZE:
            for sub in _chunk_text(content):
                chunks.append((sub, {
                    "section": section_str,
                    "chunk_type": "section",
                    "heading_level": level,
                }))
        else:
            chunks.append((content, {
                "section": section_str,
                "chunk_type": "section",
                "heading_level": level,
            }))

    if not chunks:
        return [(c, {"section": "", "chunk_type": "fixed"})
                for c in _chunk_text(text)]
    return chunks


def _chunk_pdf_sections(
    text: str,
) -> list[tuple[str, dict[str, Any]]]:
    """PDF 按章节标记分块. 找不到章节结构就退回固定长度."""
    matches = list(_PDF_CHAPTER_RE.finditer(text))

    if not matches:
        return [
            (c, {"section": "", "chunk_type": "fixed"})
            for c in _chunk_text(text)
        ]

    chunks: list[tuple[str, dict[str, Any]]] = []

    # 章节前的内容
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            chunks.append((pre, {"section": "", "chunk_type": "preamble"}))

    for i, match in enumerate(matches):
        title = match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if not content:
            continue

        if len(content) > CHUNK_SIZE:
            for sub in _chunk_text(content):
                chunks.append((sub, {
                    "section": title,
                    "chunk_type": "section",
                }))
        else:
            chunks.append((content, {
                "section": title,
                "chunk_type": "section",
            }))

    if not chunks:
        return [(c, {"section": "", "chunk_type": "fixed"})
                for c in _chunk_text(text)]
    return chunks


class KnowledgeBase:
    """A local vector knowledge base backed by ChromaDB."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs_dir = self.root / "docs"
        self.docs_dir.mkdir(exist_ok=True)

        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError(
                "Knowledge base requires chromadb. Install: pip install chromadb"
            ) from e

        self.client = chromadb.PersistentClient(path=str(self.root / "chroma"))
        self.collection = self.client.get_or_create_collection("huginn_kb")
        self._model: Any | None = None
        self._query_cache: TimedLRUCache[list[dict[str, Any]]] = TimedLRUCache(
            max_size=256, ttl=60.0
        )
        # 检索命中计数: 用于智能 KB 淘汰评分 (频率维度)
        self._hit_counts: dict[str, int] = {}
        # 语义缓存: gptcache 可选, 没装就退回上面的 TimedLRUCache
        self._semantic_cache: Any | None = None
        try:
            from gptcache import Cache
            from gptcache.embedding import Onnx
            from gptcache.manager.factory import manager_factory
            from gptcache.processor.pre import get_prompt
            from gptcache.similarity_evaluation import SearchDistanceEvaluation

            onnx = Onnx(model_name="all-MiniLM-L6-v2")
            data_manager = manager_factory(
                "sqlite,faiss",
                data_dir=str(self.root / "semantic_cache"),
                vector_params={"dimension": onnx.dimension},
            )
            self._semantic_cache = Cache()
            self._semantic_cache.init(
                pre_embedding_func=get_prompt,
                data_manager=data_manager,
                # max_distance=0.4 对应 cosine similarity 约 0.92
                # (L2 距离 d, sim = 1 - d^2/2, d=0.4 -> sim=0.92)
                similarity_evaluation=SearchDistanceEvaluation(max_distance=0.4),
                embedding_func=onnx.to_embeddings,
            )
        except Exception:
            # gptcache 没装或初始化失败 (模型下载/faiss 缺失等), 优雅降级
            self._semantic_cache = None

        # Feedback tracker for confidence-based reranking (lazy init)
        self._feedback_tracker: Any | None = None
        try:
            from huginn.rag.feedback import RetrievalFeedbackTracker
            self._feedback_tracker = RetrievalFeedbackTracker()
        except Exception:
            pass

    @property
    def model(self) -> _EmbeddingModel:
        if self._model is None:
            self._model = _EmbeddingModel()
        return self._model

    def _flush_semantic_cache(self) -> None:
        """知识库变更时清掉语义缓存, 避免返回过期结果."""
        if self._semantic_cache is not None:
            try:
                self._semantic_cache.flush()
            except Exception:
                logger.debug("flush failed", exc_info=True)

    def add_document(self, filename: str, content: bytes) -> dict[str, Any]:
        """Ingest a document, chunk it, and store embeddings.

        Markdown / PDF 文档按章节结构分块, 每块带 section 元数据.
        纯文本退回固定长度分块. 同时自动打领域标签写入 metadata.
        """
        doc_id = uuid.uuid4().hex[:12]
        text = _extract_text(filename, content)
        if not text.strip():
            raise ValueError("No text could be extracted from the file")

        sectioned = _section_aware_chunk(text, filename)
        chunks = [c for c, _ in sectioned]
        if not chunks:
            raise ValueError("Document is empty after chunking")

        # 全文做一次关键词匹配, 给文档打领域标签
        tags = auto_tag(text)
        domain_str = json.dumps(tags["domain_tags"], ensure_ascii=False) if tags["domain_tags"] else ""
        sub_domain_str = json.dumps(tags["sub_domain_tags"], ensure_ascii=False) if tags["sub_domain_tags"] else ""
        # primary domain 存成简单字符串, 方便 ChromaDB where 过滤
        primary_domain = tags["domain_tags"][0] if tags["domain_tags"] else "未分类"

        chunk_hash = hashlib.sha256("".join(chunks).encode()).hexdigest()
        embeddings = self.model.encode(chunks, cache_key=chunk_hash).tolist()
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        metadatas = []
        for i, (chunk, section_meta) in enumerate(sectioned):
            meta: dict[str, Any] = {"doc_id": doc_id, "filename": filename, "chunk": i}
            meta.update(section_meta)
            meta["domain"] = primary_domain
            if domain_str:
                meta["domain_tags"] = domain_str
            if sub_domain_str:
                meta["sub_domain_tags"] = sub_domain_str
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self._query_cache.clear()
        self._flush_semantic_cache()

        safe_name = f"{doc_id}_{Path(filename).name}"
        doc_path = self.docs_dir / safe_name
        doc_path.write_bytes(content)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "domain_tags": tags["domain_tags"],
            "sub_domain_tags": tags["sub_domain_tags"],
        }

    def add_text(
        self,
        text: str,
        filename: str = "auto_sediment",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingest raw text directly into the knowledge base.

        Unlike ``add_document`` which takes a file, this method accepts
        pre-extracted text — used by auto-sedimentation and distilled
        knowledge ingestion pipelines.

        章节感知分块 + 自动领域标签, 跟 add_document 一致.
        Returns a dict with doc_id and chunk count.
        """
        if not text or not text.strip():
            return {"doc_id": "", "chunks": 0}

        doc_id = uuid.uuid4().hex[:12]
        sectioned = _section_aware_chunk(text, filename)
        chunks = [c for c, _ in sectioned]
        if not chunks:
            return {"doc_id": doc_id, "chunks": 0}

        # 自动打领域标签, 调用方 metadata 里有 domain 就用调用方的
        tags = auto_tag(text)
        domain_str = json.dumps(tags["domain_tags"], ensure_ascii=False) if tags["domain_tags"] else ""
        sub_domain_str = json.dumps(tags["sub_domain_tags"], ensure_ascii=False) if tags["sub_domain_tags"] else ""
        primary_domain = tags["domain_tags"][0] if tags["domain_tags"] else "未分类"

        chunk_hash = hashlib.sha256("".join(chunks).encode()).hexdigest()
        embeddings = self.model.encode(chunks, cache_key=chunk_hash).tolist()
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        metadatas = []
        for i, (chunk, section_meta) in enumerate(sectioned):
            meta: dict[str, Any] = {"doc_id": doc_id, "filename": filename}
            meta.update(section_meta)
            meta["domain"] = primary_domain
            if domain_str:
                meta["domain_tags"] = domain_str
            if sub_domain_str:
                meta["sub_domain_tags"] = sub_domain_str

            # 调用方传入的 metadata 覆盖自动生成的 (domain 除外, 让调用方也能指定)
            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (list, dict)):
                        meta[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        meta[k] = str(v) if v is not None else ""

            meta["chunk"] = i
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self._query_cache.clear()
        self._flush_semantic_cache()

        return {
            "doc_id": doc_id,
            "chunks": len(chunks),
            "domain_tags": tags["domain_tags"],
            "sub_domain_tags": tags["sub_domain_tags"],
        }

    def list_documents(self) -> list[dict[str, Any]]:
        """Return unique documents stored in the collection."""
        data = self.collection.get(include=["metadatas"])
        docs: dict[str, dict[str, Any]] = {}
        for meta in data.get("metadatas") or []:
            doc_id = meta.get("doc_id")
            if not doc_id or doc_id in docs:
                continue
            docs[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("filename", "unknown"),
            }
        return sorted(docs.values(), key=lambda d: d["filename"])

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and its chunks from the knowledge base."""
        data = self.collection.get(where={"doc_id": doc_id}, include=[])
        ids = data.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
            self._query_cache.clear()
            self._flush_semantic_cache()
        for path in self.docs_dir.glob(f"{doc_id}_*"):
            path.unlink(missing_ok=True)
        return len(ids) > 0

    def cleanup_old_documents(
        self, max_docs: int = 200, prefix: str = "autoloop_iter_"
    ) -> int:
        """智能清理旧文档, 基于**信息价值三维评分**决定保留哪些.

        评分 = retrieval_frequency × information_density × recency
        - retrieval_frequency: 被检索命中次数 (需要 _hit_counts 追踪)
        - information_density: structure_aware/visual_primitives chunk 分数更高
        - recency: 时间衰减 (autoloop 迭代文档衰减快, 用户上传文档衰减慢)

        策略:
        1. autoloop_iter_ 文档: 只保留最近 50 轮 + 高分文档
        2. 总文档数超上限: 按信息价值评分排序, 淘汰低分文档
        3. 返回删除的文档数.
        """
        try:
            all_docs = self.list_documents()
            if len(all_docs) <= max_docs:
                return 0

            deleted = 0

            # 1. autoloop 迭代文档: 保留最近 50 轮
            auto_docs = [d for d in all_docs if d.get("filename", "").startswith(prefix)]
            if len(auto_docs) > 50:
                auto_docs.sort(key=lambda d: d.get("filename", ""))
                for d in auto_docs[:-50]:
                    if self.delete_document(d["doc_id"]):
                        deleted += 1
                all_docs = [d for d in all_docs if not d.get("filename", "").startswith(prefix) or d in auto_docs[-50:]]

            # 2. 总文档数仍超上限: 按信息价值评分淘汰
            excess = len(all_docs) - deleted - max_docs
            if excess <= 0:
                if deleted:
                    logger.info("KB cleanup: removed %d old iterations", deleted)
                return deleted

            # 为每个文档计算信息价值评分
            scored = []
            for d in all_docs:
                score = self._score_document_value(d)
                scored.append((score, d))

            # 按评分升序, 淘汰最低分的
            scored.sort(key=lambda x: x[0])
            for score, d in scored[:excess]:
                if self.delete_document(d["doc_id"]):
                    deleted += 1

            if deleted:
                logger.info("KB cleanup: removed %d low-value documents (max=%d)", deleted, max_docs)
            return deleted
        except Exception as exc:
            logger.warning("KB cleanup failed: %s", exc)
            return 0

    def _score_document_value(self, doc: dict[str, Any]) -> float:
        """计算文档的信息价值评分 — Generative Agents 加权模型.

        score = α·recency + β·importance + γ·relevance
        α=0.2, β=0.3, γ=0.5 (relevance 权重最高: 被用过的知识最有价值)

        0.0 = 无价值 (可删除), 1.0 = 高价值 (必须保留).
        """
        import time as _time

        # α: recency — 指数衰减, 半衰期 7 天 (研究周期)
        ts = doc.get("timestamp") or doc.get("created_at") or 0
        if ts:
            age_s = _time.time() - float(ts)
            half_life_s = 7 * 86400  # 7 days
            recency = 0.5 ** (age_s / half_life_s)
        else:
            recency = 0.5  # 没时间戳的给中间分

        # β: importance — 信息密度 (结构文件 > PDF > 普通文本 > 迭代摘要)
        filename = doc.get("filename", "")
        if any(ext in filename.lower() for ext in (".cif", ".poscar", ".contcar")):
            importance = 0.9
        elif filename.endswith(".pdf"):
            importance = 0.7
        elif filename.startswith("autoloop_iter_"):
            importance = 0.3
        else:
            importance = 0.5

        # γ: relevance — 检索频率 (命中次数)
        hit_count = getattr(self, "_hit_counts", {}).get(doc.get("doc_id", ""), 0)
        relevance = min(hit_count / 5.0, 1.0)

        score = 0.2 * recency + 0.3 * importance + 0.5 * relevance

        # 永不删除: 用户手动上传且被检索过
        if hit_count > 0 and not filename.startswith("autoloop_iter_"):
            score = max(score, 0.8)

        return score

    def query_with_dedup(
        self, text: str, top_k: int = 5, domain: str | None = None,
        similarity_threshold: float = 0.85
    ) -> list[dict[str, Any]]:
        """带去重的检索: 相似度 > threshold 的 chunk 只保留第一个.
        解决分块重叠导致的近似重复段落问题.
        """
        # 多捞 2x 候选, 去重后截断到 top_k
        raw = self.query(text, top_k=top_k * 2, domain=domain)
        if not raw or len(raw) <= 1:
            return raw

        # 简单文本 Jaccard 相似度去重
        def keywords(t: str) -> set[str]:
            import re
            words = re.findall(r'[a-zA-Z_]\w{2,}', t.lower())
            return set(words[:50])  # 只看前 50 词, 够快

        kept: list[dict] = []
        for chunk in raw:
            text_i = chunk.get("text", "")
            kw_i = keywords(text_i)
            is_dup = False
            for kept_chunk in kept:
                kw_j = keywords(kept_chunk.get("text", ""))
                if kw_i and kw_j:
                    jaccard = len(kw_i & kw_j) / len(kw_i | kw_j)
                    if jaccard > similarity_threshold:
                        is_dup = True
                        break
            if not is_dup:
                kept.append(chunk)
            if len(kept) >= top_k:
                break

        return kept

    def query(
        self, text: str, top_k: int = 5, domain: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve top-k relevant chunks for a query.

        domain 不为 None 时只返回该领域的文档块 (按 metadata.domain 过滤).
        """
        if not text.strip():
            return []

        # 语义缓存优先: 相近的 query 直接复用历史结果 (不过滤 domain 的时候才走)
        if domain is None and self._semantic_cache is not None:
            try:
                hit = self._semantic_cache.get(prompt=text.strip())
                if hit is not None:
                    return hit
            except Exception:
                logger.debug("get failed", exc_info=True)  # 缓存查询出错不影响正常流程

        cache_key = (text.strip(), top_k, domain)
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        embedding = self.model.encode([text]).tolist()
        where_filter = {"domain": domain} if domain else None
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, max(1, self.collection.count())),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i, doc_id in enumerate(results.get("ids", [[]])[0]):
            chunks.append(
                {
                    "chunk_id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        # 递归激活: 从首轮结果的 domain_tags 提取标签, 再查一轮关联 chunks.
        # SillyTavern World Info 的递归激活模式 — chunk A 的内容触发 chunk B.
        # 这里用 metadata 里的 domain_tags 做二次检索, 补充语义没命中但同域的块.
        if chunks and len(chunks) < top_k + 3:
            activated = self._recursive_activate(chunks, text, top_k)
            if activated:
                # 去重后合并, 不超过 top_k + 2 (多给 2 条关联结果)
                seen_ids = {c["chunk_id"] for c in chunks}
                for c in activated:
                    if c["chunk_id"] not in seen_ids:
                        chunks.append(c)
                        seen_ids.add(c["chunk_id"])
                        if len(chunks) >= top_k + 2:
                            break

        self._query_cache.set(cache_key, chunks)

        # Apply feedback-based reranking if tracker is available
        if self._feedback_tracker is not None:
            try:
                chunks = self._feedback_tracker.adjust_search_results(chunks)
            except Exception:
                pass  # reranking is best-effort

        # Feynman note 优先 + importance 加权 reranking
        # Generative Agents: relevance(=1-distance) × importance(hit_count)
        _hits = getattr(self, "_hit_counts", {})
        chunks.sort(
            key=lambda c: (
                0 if c.get("metadata", {}).get("filename", "").startswith("feynman_") else 1,
                # distance 越小越好 (更相似), hit_count 越大越好 (更常用)
                # 合并为: distance - 0.1 * log(1 + hit_count)
                c.get("distance", 1.0) - 0.1 * __import__("math").log1p(
                    _hits.get(c.get("metadata", {}).get("doc_id", ""), 0)
                ),
            )
        )

        # 回写语义缓存, 下次相似 query 能命中
        if self._semantic_cache is not None:
            try:
                self._semantic_cache.put(prompt=text.strip(), data=chunks)
            except Exception:
                logger.debug("put failed", exc_info=True)

        # 记录检索命中: 用于智能 KB 淘汰评分
        for c in chunks:
            did = c.get("metadata", {}).get("doc_id", "")
            if did:
                self._hit_counts[did] = self._hit_counts.get(did, 0) + 1

        return chunks

    def _recursive_activate(
        self, chunks: list[dict[str, Any]], original_query: str, top_k: int
    ) -> list[dict[str, Any]]:
        """从首轮结果的 tags 做二次检索, 补充同域关联块.

        SillyTavern World Info 递归激活的简化版: 不做多跳 (避免无界扩散),
        只做一轮 tag-based 扩展. 提取首轮结果的 domain_tags, 用 OR 条件查
        同标签的块, 排除已命中的.
        """
        # 收集首轮结果的所有 domain_tags
        all_tags: set[str] = set()
        for c in chunks:
            meta = c.get("metadata", {})
            raw_tags = meta.get("domain_tags", "[]")
            if isinstance(raw_tags, str):
                try:
                    import json
                    tags = json.loads(raw_tags)
                    if isinstance(tags, list):
                        all_tags.update(tags)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(raw_tags, list):
                all_tags.update(raw_tags)

        if not all_tags:
            return []

        # 用 tags 做 metadata 过滤的二次查询. ChromaDB 的 $in 操作符
        # 需要 domain_tags 是 list 类型, 但我们存的是 JSON string —
        # 所以改用纯语义查询 + 结果过滤的方式, 不依赖 where 子句.
        # ponytail: O(n) scan over 2x candidates, ok for <10K chunks;
        # switch to ChromaDB where-filter if KB grows past 50K.
        try:
            embedding = self.model.encode([original_query]).tolist()
            results = self.collection.query(
                query_embeddings=embedding,
                n_results=min(top_k * 2, max(1, self.collection.count())),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        seen_ids = {c["chunk_id"] for c in chunks}
        activated: list[dict[str, Any]] = []
        for i, doc_id in enumerate(results.get("ids", [[]])[0]):
            if doc_id in seen_ids:
                continue
            meta = results["metadatas"][0][i]
            raw_tags = meta.get("domain_tags", "[]")
            chunk_tags: set[str] = set()
            if isinstance(raw_tags, str):
                try:
                    import json
                    t = json.loads(raw_tags)
                    if isinstance(t, list):
                        chunk_tags.update(t)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(raw_tags, list):
                chunk_tags.update(raw_tags)
            # 有交集就算命中
            if chunk_tags & all_tags:
                activated.append(
                    {
                        "chunk_id": doc_id,
                        "text": results["documents"][0][i],
                        "metadata": meta,
                        "distance": results["distances"][0][i],
                    }
                )
                if len(activated) >= 2:
                    break
        return activated

    def count(self) -> int:
        return self.collection.count()


def seed_knowledge_base(kb: KnowledgeBase, force: bool = False) -> dict[str, Any]:
    """Ingest built-in seed reference documents into a knowledge base.

    Seeds are loaded from ``huginn/knowledge/seed/*.md``. Each seed is
    identified by ``seed:<sha256>`` so that unchanged files are skipped on
    subsequent runs. Passing ``force=True`` removes all existing seed entries
    and re-ingests them.
    """
    if not SEED_DIR.is_dir():
        return {"added": 0, "skipped": 0, "failed": 0}

    seed_files = sorted(SEED_DIR.glob("*.md"))
    existing_seed_ids = {
        doc["doc_id"]
        for doc in kb.list_documents()
        if doc["doc_id"].startswith("seed:")
    }

    if force:
        for doc_id in list(existing_seed_ids):
            kb.delete_document(doc_id)
        existing_seed_ids = set()

    added = skipped = failed = 0
    for path in seed_files:
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        doc_id = f"seed:{digest}"
        if doc_id in existing_seed_ids:
            skipped += 1
            continue

        try:
            text = _extract_text(path.name, content)
            sectioned = _section_aware_chunk(text, path.name)
            chunks = [c for c, _ in sectioned]
            if not chunks:
                skipped += 1
                continue

            tags = auto_tag(text)
            primary_domain = tags["domain_tags"][0] if tags["domain_tags"] else "未分类"
            domain_str = json.dumps(tags["domain_tags"], ensure_ascii=False) if tags["domain_tags"] else ""

            embeddings = kb.model.encode(chunks, cache_key=digest).tolist()
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = []
            for i, (chunk, section_meta) in enumerate(sectioned):
                meta: dict[str, Any] = {
                    "doc_id": doc_id,
                    "filename": path.name,
                    "chunk": i,
                    "seed": True,
                }
                meta.update(section_meta)
                meta["domain"] = primary_domain
                if domain_str:
                    meta["domain_tags"] = domain_str
                metadatas.append(meta)
            kb.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            added += 1
        except Exception:
            failed += 1

    return {"added": added, "skipped": skipped, "failed": failed}


_knowledge_base: KnowledgeBase | None = None
_kb_lock = __import__("threading").Lock()


def get_knowledge_base(workspace: str = ".") -> KnowledgeBase:
    """Thread-safe singleton accessor for the knowledge base."""
    global _knowledge_base
    if _knowledge_base is None:
        with _kb_lock:
            if _knowledge_base is None:  # double-checked locking
                _knowledge_base = KnowledgeBase(Path(workspace) / ".huginn_kb")
                seed_knowledge_base(_knowledge_base, force=False)
    return _knowledge_base
