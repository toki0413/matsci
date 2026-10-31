"""统一导出 / 导入 / 压缩 / 分享系统.

把 Huginn 智能体的全部数据（长期记忆、知识库、知识图谱、会话、配置）
打包成一个压缩归档，方便备份、迁移和分享。支持 zip / tar.gz / json 三种
格式，导入时可以选择合并（merge）而非覆盖。

典型用法::

    mgr = ExportShareManager(Path("/path/to/workspace"))
    info = mgr.export_all("/tmp/huginn_backup.zip", format="zip")
    # ... 后续恢复 ...
    result = mgr.import_all("/tmp/huginn_backup.zip", merge=True)
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sqlite3
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from huginn.utils.archive_safety import safe_archive_extract
from huginn.utils.runtime import HUGINN_DIR_NAME, get_runtime_home

logger = logging.getLogger(__name__)

# 归档格式版本，后续升级结构时可以据此做兼容处理
ARCHIVE_VERSION = "1.0"

# 默认包含的全部数据类型
ALL_COMPONENTS: list[str] = ["memory", "knowledge", "graph", "sessions", "config"]

# 单个 chunk 文件最多写入这么多条记录，避免内存爆掉
_CHUNK_BATCH = 5000

# SQLite 备份时每次读写的块大小
_SQLITE_COPY_CHUNK = 64 * 1024


@dataclass
class ExportStats:
    """导出过程中的统计信息，最终写进 manifest.json."""

    memory_entries: int = 0
    memory_md_exists: bool = False
    knowledge_documents: int = 0
    knowledge_chunks: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    sessions: int = 0
    config_exported: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_entries": self.memory_entries,
            "memory_md_exists": self.memory_md_exists,
            "knowledge_documents": self.knowledge_documents,
            "knowledge_chunks": self.knowledge_chunks,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "sessions": self.sessions,
            "config_exported": self.config_exported,
            "errors": self.errors,
        }


def _merge_sessions_into_threads(sessions: list, merge: bool) -> None:
    """把导入的会话写进 server_core 的线程表 (merge 时跳过已存在 id)."""
    from huginn import server_core

    with server_core._state_lock:
        for s in sessions:
            tid = s.get("id")
            if tid and (merge or tid not in server_core._threads):
                server_core._threads[tid] = s


def _chunk_records(
    ids: list,
    documents: list,
    metadatas: list,
    embeddings: list,
) -> list[dict[str, Any]]:
    """把 ChromaDB get() 的平行数组拼成 chunk 记录列表."""
    records: list[dict[str, Any]] = []
    for idx, cid in enumerate(ids):
        records.append(
            {
                "id": cid,
                "document": documents[idx] if idx < len(documents) else "",
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "embedding": embeddings[idx] if idx < len(embeddings) else None,
            }
        )
    return records


def _kb_import_batched(kb: Any, records: Any) -> int:
    """按 _CHUNK_BATCH 分批把记录 upsert 进 ChromaDB, 返回导入条数."""
    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_metas: list[dict] = []
    batch_embs: list[list[float]] = []
    imported = 0

    for record in records:
        batch_ids.append(record["id"])
        batch_docs.append(record.get("document", ""))
        batch_metas.append(record.get("metadata", {}))
        emb = record.get("embedding")
        if emb is not None:
            batch_embs.append(emb)

        if len(batch_ids) >= _CHUNK_BATCH:
            ExportShareManager._kb_upsert(
                kb, batch_ids, batch_docs, batch_metas, batch_embs
            )
            imported += len(batch_ids)
            batch_ids, batch_docs, batch_metas, batch_embs = [], [], [], []

    if batch_ids:
        ExportShareManager._kb_upsert(
            kb, batch_ids, batch_docs, batch_metas, batch_embs
        )
        imported += len(batch_ids)
    return imported


def _component_zip_bytes(
    write_fn: Any,
    subdir: str,
) -> bytes:
    """把单个组件目录打成内存 zip, 返回字节."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        write_fn(tmp_dir / subdir, ExportStats())
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in (tmp_dir / subdir).rglob("*"):
                if fp.is_file():
                    zf.write(str(fp), fp.relative_to(tmp_dir / subdir))
        return buf.getvalue()


# 归档后缀 → 导入方法名 (未命中走文件头嗅探)
_IMPORT_EXT_HANDLERS: tuple[tuple[tuple[str, ...], str], ...] = (
    ((".json",), "_import_from_json"),
    ((".zip",), "_import_zip"),
    ((".gz", ".tar.gz"), "_import_targz"),
)


def _import_handler(ext: str, name: str) -> str:
    """按归档后缀选择导入方法名."""
    for suffixes, handler in _IMPORT_EXT_HANDLERS:
        if ext in suffixes or (".tar.gz" in suffixes and name.endswith(".tar.gz")):
            return handler
    return "_import_by_magic"


class ExportShareManager:
    """统一管理 Huginn 数据的导出、导入、压缩和分享."""

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace).resolve()

    # ── 组件获取（懒加载，缺失时降级）──────────────────────────────

    def _get_longterm(self) -> Any | None:
        """获取长期记忆实例，不可用时返回 None."""
        try:
            from huginn.memory.longterm import LongTermMemory

            return LongTermMemory()
        except Exception as e:
            logger.warning("无法初始化长期记忆: %s", e)
            return None

    def _get_kb(self) -> Any | None:
        """获取知识库实例，不可用时返回 None."""
        try:
            from huginn.knowledge.store import KnowledgeBase

            return KnowledgeBase(self.workspace / ".huginn_kb")
        except Exception as e:
            logger.warning("无法初始化知识库: %s", e)
            return None

    def _get_kg(self) -> Any | None:
        """获取知识图谱实例，不可用时返回 None."""
        try:
            from huginn.kg.graph import ProjectKnowledgeGraph

            return ProjectKnowledgeGraph(self.workspace / HUGINN_DIR_NAME)
        except Exception as e:
            logger.warning("无法初始化知识图谱: %s", e)
            return None

    def _get_config_dict(self) -> dict[str, Any] | None:
        """获取当前配置快照（密钥会脱敏）."""
        try:
            from huginn.config import HuginnConfig

            cfg = HuginnConfig.from_env()
            return cfg.to_dict(mask_key=True)
        except Exception as e:
            logger.warning("无法读取配置: %s", e)
            return None

    def _get_sessions(self) -> list[dict[str, Any]]:
        """获取活跃会话/线程的元数据."""
        sessions: list[dict[str, Any]] = []
        try:
            from huginn import server_core

            with server_core._state_lock:
                for _tid, meta in server_core._threads.items():
                    sessions.append(dict(meta))
        except Exception as e:
            logger.warning("无法读取会话列表: %s", e)
        return sessions

    # ── 各组件导出 ──────────────────────────────────────────────────

    def _export_memory_to_dir(self, target_dir: Path, stats: ExportStats) -> None:
        """把长期记忆导出到指定目录."""
        target_dir.mkdir(parents=True, exist_ok=True)
        longterm = self._get_longterm()

        if longterm is None:
            # 退而求其次：直接拷贝 SQLite 文件
            self._copy_sqlite_memory(target_dir, stats)
            return

        # 1) JSON 格式的记忆条目（可移植、不依赖 SQLite）
        mem_json = target_dir / "memories.json"
        try:
            longterm.export(str(mem_json))
            # 统计条目数
            with open(mem_json, encoding="utf-8") as f:
                data = json.load(f)
            stats.memory_entries = len(data) if isinstance(data, list) else 0
        except Exception as e:
            stats.errors.append(f"memory json export: {e}")

        # 2) SQLite 数据库文件副本（保留完整索引/FTS）
        self._copy_sqlite_memory(target_dir, stats)

        # 3) MEMORY.md 文件（如果存在）
        memory_md = self.workspace / "MEMORY.md"
        if memory_md.exists():
            shutil.copy2(str(memory_md), str(target_dir / "MEMORY.md"))
            stats.memory_md_exists = True

        # 4) 主题记忆目录（如果存在）
        memory_dir = self.workspace / HUGINN_DIR_NAME / "memory"
        if memory_dir.is_dir():
            dest = target_dir / "memory_topics"
            self._copy_tree(memory_dir, dest)

    def _copy_sqlite_memory(self, target_dir: Path, stats: ExportStats) -> None:
        """用 SQLite 备份 API 安全拷贝数据库文件."""
        db_path = self._find_sqlite_db()
        if db_path is None:
            return
        dst = target_dir / "memory.db"
        try:
            # 在线备份，不阻塞其他连接
            src_conn = sqlite3.connect(str(db_path))
            dst_conn = sqlite3.connect(str(dst))
            with dst_conn:
                src_conn.backup(dst_conn)
            src_conn.close()
            dst_conn.close()
        except Exception as e:
            # 降级到直接文件拷贝
            try:
                shutil.copy2(str(db_path), str(dst))
            except Exception:
                stats.errors.append(f"sqlite copy: {e}")

    def _find_sqlite_db(self) -> Path | None:
        """找到长期记忆的 SQLite 数据库文件路径."""
        cache_dir = os.environ.get("HUGINN_CACHE_DIR")
        candidates = [
            Path(cache_dir) / "memory.db" if cache_dir else None,
            get_runtime_home() / "memory.db",
            self.workspace / HUGINN_DIR_NAME / "memory.db",
        ]
        for c in candidates:
            if c and c.exists():
                return c
        return None

    def _ekd_documents(self, kb: Any, target_dir: Path, stats: ExportStats) -> None:
        """把知识库文档列表写到 documents.json."""
        try:
            docs = kb.list_documents()
            (target_dir / "documents.json").write_text(
                json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            stats.knowledge_documents = len(docs)
        except Exception as e:
            stats.errors.append(f"knowledge docs: {e}")

    def _ekd_chunks(self, kb: Any, target_dir: Path, stats: ExportStats) -> None:
        """把全部 chunks + embeddings 分批写到 chunks.jsonl."""
        chunks_path = target_dir / "chunks.jsonl"
        total_chunks = 0
        try:
            with open(chunks_path, "w", encoding="utf-8") as f:
                data = kb.collection.get(
                    include=["documents", "metadatas", "embeddings"]
                )
                ids = data.get("ids") or []
                for i in range(0, len(ids), _CHUNK_BATCH):
                    batch = ids[i : i + _CHUNK_BATCH]
                    records = _chunk_records(
                        batch,
                        (data.get("documents") or [])[i : i + _CHUNK_BATCH],
                        (data.get("metadatas") or [])[i : i + _CHUNK_BATCH],
                        (data.get("embeddings") or [])[i : i + _CHUNK_BATCH],
                    )
                    for record in records:
                        f.write(
                            json.dumps(record, ensure_ascii=False, default=str) + "\n"
                        )
                    total_chunks += len(batch)
            stats.knowledge_chunks = total_chunks
        except Exception as e:
            stats.errors.append(f"knowledge chunks: {e}")

    def _ekd_raw_docs(self, target_dir: Path) -> None:
        """拷贝知识库原始文档文件."""
        kb_docs = self.workspace / ".huginn_kb" / "docs"
        if kb_docs.is_dir():
            self._copy_tree(kb_docs, target_dir / "docs")

    def _export_knowledge_to_dir(self, target_dir: Path, stats: ExportStats) -> None:
        """把知识库导出到指定目录."""
        target_dir.mkdir(parents=True, exist_ok=True)
        kb = self._get_kb()

        if kb is None:
            # 没有 ChromaDB，直接拷贝目录
            kb_dir = self.workspace / ".huginn_kb"
            if kb_dir.is_dir():
                self._copy_tree(kb_dir, target_dir / "kb_raw")
            return

        self._ekd_documents(kb, target_dir, stats)
        self._ekd_chunks(kb, target_dir, stats)
        self._ekd_raw_docs(target_dir)

    def _export_graph_to_dir(self, target_dir: Path, stats: ExportStats) -> None:
        """把知识图谱导出到指定目录."""
        target_dir.mkdir(parents=True, exist_ok=True)
        kg = self._get_kg()

        if kg is None:
            # 直接拷贝文件
            kg_file = self.workspace / HUGINN_DIR_NAME / "project_kg.json"
            if kg_file.exists():
                shutil.copy2(str(kg_file), str(target_dir / "project_kg.json"))
                try:
                    data = json.loads(kg_file.read_text(encoding="utf-8"))
                    stats.graph_nodes = len(data.get("nodes", []))
                    stats.graph_edges = len(data.get("links", []))
                except Exception:
                    logger.debug("loads failed", exc_info=True)
            return

        try:
            graph_data = kg.export(fmt="json")
            (target_dir / "project_kg.json").write_text(
                json.dumps(graph_data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            stats.graph_nodes = len(graph_data.get("nodes", []))
            stats.graph_edges = len(graph_data.get("links", []))
        except Exception as e:
            stats.errors.append(f"graph export: {e}")

    def _export_config_to_dir(self, target_dir: Path, stats: ExportStats) -> None:
        """把配置导出到指定目录."""
        target_dir.mkdir(parents=True, exist_ok=True)
        cfg_dict = self._get_config_dict()
        if cfg_dict is not None:
            (target_dir / "config.json").write_text(
                json.dumps(cfg_dict, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            stats.config_exported = True

    def _export_sessions_to_dir(self, target_dir: Path, stats: ExportStats) -> None:
        """把会话/线程元数据导出到指定目录."""
        target_dir.mkdir(parents=True, exist_ok=True)
        sessions = self._get_sessions()
        stats.sessions = len(sessions)
        (target_dir / "threads.json").write_text(
            json.dumps(sessions, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    # ── 压缩归档 ────────────────────────────────────────────────────

    def _build_export_dir(
        self,
        tmp_dir: Path,
        include: list[str] | None,
    ) -> tuple[Path, ExportStats]:
        """在临时目录里搭建好完整的导出目录结构，返回路径和统计信息."""
        include = include or list(ALL_COMPONENTS)
        stats = ExportStats()
        export_root = tmp_dir / "huginn_export"

        if "memory" in include:
            self._export_memory_to_dir(export_root / "memory", stats)
        if "knowledge" in include:
            self._export_knowledge_to_dir(export_root / "knowledge", stats)
        if "graph" in include:
            self._export_graph_to_dir(export_root / "graph", stats)
        if "sessions" in include:
            self._export_sessions_to_dir(export_root / "sessions", stats)
        if "config" in include:
            self._export_config_to_dir(export_root / "config", stats)

        # 写入 manifest
        manifest = {
            "version": ARCHIVE_VERSION,
            "created_at": datetime.now().isoformat(),
            "included": include,
            "stats": stats.to_dict(),
        }
        (export_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return export_root, stats

    def _write_zip(self, src_dir: Path, output_path: Path) -> None:
        """把目录递归打包成 zip."""
        with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(src_dir.rglob("*")):
                if file_path.is_file():
                    arcname = str(file_path.relative_to(src_dir.parent))
                    zf.write(str(file_path), arcname)

    def _write_targz(self, src_dir: Path, output_path: Path) -> None:
        """把目录递归打包成 tar.gz."""
        with tarfile.open(str(output_path), "w:gz") as tf:
            tf.add(str(src_dir), arcname="huginn_export")

    def _wjs_memory(self, payload: dict[str, Any], stats: ExportStats) -> None:
        """JSON 单文件导出: 长期记忆."""
        longterm = self._get_longterm()
        if longterm is not None:
            try:
                # 导到临时文件再读回来，LongTermMemory 只支持文件路径导出
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as tf:
                    tf_path = tf.name
                longterm.export(tf_path)
                with open(tf_path, encoding="utf-8") as f:
                    mem_data = json.load(f)
                os.unlink(tf_path)
                payload["memory"] = {"entries": mem_data}
                stats.memory_entries = (
                    len(mem_data) if isinstance(mem_data, list) else 0
                )
            except Exception as e:
                stats.errors.append(f"memory json: {e}")
        memory_md = self.workspace / "MEMORY.md"
        if memory_md.exists():
            payload["memory_md"] = memory_md.read_text(encoding="utf-8")
            stats.memory_md_exists = True

    def _wjs_knowledge(self, payload: dict[str, Any], stats: ExportStats) -> None:
        """JSON 单文件导出: 知识库."""
        kb = self._get_kb()
        if kb is None:
            return
        try:
            docs = kb.list_documents()
            data = kb.collection.get(include=["documents", "metadatas", "embeddings"])
            chunks = _chunk_records(
                data.get("ids") or [],
                data.get("documents") or [],
                data.get("metadatas") or [],
                data.get("embeddings") or [],
            )
            payload["knowledge"] = {"documents": docs, "chunks": chunks}
            stats.knowledge_documents = len(docs)
            stats.knowledge_chunks = len(chunks)
        except Exception as e:
            stats.errors.append(f"knowledge json: {e}")

    def _wjs_graph(self, payload: dict[str, Any], stats: ExportStats) -> None:
        """JSON 单文件导出: 知识图谱."""
        kg = self._get_kg()
        if kg is None:
            return
        try:
            graph_data = kg.export(fmt="json")
            payload["graph"] = graph_data
            stats.graph_nodes = len(graph_data.get("nodes", []))
            stats.graph_edges = len(graph_data.get("links", []))
        except Exception as e:
            stats.errors.append(f"graph json: {e}")

    def _wjs_sessions(self, payload: dict[str, Any], stats: ExportStats) -> None:
        """JSON 单文件导出: 会话."""
        sessions = self._get_sessions()
        payload["sessions"] = sessions
        stats.sessions = len(sessions)

    def _wjs_config(self, payload: dict[str, Any], stats: ExportStats) -> None:
        """JSON 单文件导出: 配置."""
        cfg_dict = self._get_config_dict()
        if cfg_dict is not None:
            payload["config"] = cfg_dict
            stats.config_exported = True

    def _write_json_single(
        self,
        include: list[str] | None,
        output_path: Path,
    ) -> ExportStats:
        """把所有数据写进一个 JSON 文件（不含二进制原始文件）."""
        include = include or list(ALL_COMPONENTS)
        stats = ExportStats()
        payload: dict[str, Any] = {
            "version": ARCHIVE_VERSION,
            "created_at": datetime.now().isoformat(),
            "included": include,
        }

        exporters = {
            "memory": self._wjs_memory,
            "knowledge": self._wjs_knowledge,
            "graph": self._wjs_graph,
            "sessions": self._wjs_sessions,
            "config": self._wjs_config,
        }
        for component, export in exporters.items():
            if component in include:
                export(payload, stats)

        payload["stats"] = stats.to_dict()
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return stats

    # ── 公开 API ────────────────────────────────────────────────────

    def export_all(
        self,
        output_path: str,
        format: str = "zip",
        include: list[str] | None = None,
    ) -> dict:
        """导出全部数据到压缩归档.

        Args:
            output_path: 归档保存路径
            format: "zip" / "tar.gz" / "json"
            include: 要导出的组件列表，None 表示全部

        Returns:
            {"path": str, "size_mb": float, "items": dict}
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fmt = format.lower().strip()
        if fmt not in ("zip", "tar.gz", "json"):
            raise ValueError(f"不支持的格式: {format}，可选 zip / tar.gz / json")

        if fmt == "json":
            stats = self._write_json_single(include, out)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                export_root, stats = self._build_export_dir(tmp_dir, include)
                if fmt == "zip":
                    self._write_zip(export_root, out)
                else:
                    self._write_targz(export_root, out)

        size_mb = round(out.stat().st_size / (1024 * 1024), 2)
        return {
            "path": str(out),
            "size_mb": size_mb,
            "items": stats.to_dict(),
        }

    def _import_zip(self, src: Path, merge: bool, result: dict) -> None:
        """从 zip 归档导入."""
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(str(src), "r") as zf:
                safe_archive_extract(zf, tmp)
            self._import_from_dir(Path(tmp), merge, result)

    def _import_targz(self, src: Path, merge: bool, result: dict) -> None:
        """从 tar.gz 归档导入."""
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(str(src), "r:gz") as tf:
                safe_archive_extract(tf, tmp)
            self._import_from_dir(Path(tmp), merge, result)

    def _import_by_magic(self, src: Path, merge: bool, result: dict) -> None:
        """后缀未知时按文件头判断归档格式."""
        with open(src, "rb") as f:
            magic = f.read(4)
        if magic[:2] == b"PK":
            self._import_zip(src, merge, result)
        else:
            result["errors"].append(f"无法识别的归档格式: {src.name}")

    def import_all(self, archive_path: str, merge: bool = True) -> dict:
        """从归档导入数据，可选择合并而非覆盖.

        Args:
            archive_path: 归档文件路径
            merge: True=合并（增量），False=覆盖

        Returns:
            {"imported": dict, "merged": bool, "errors": list}
        """
        src = Path(archive_path)
        if not src.exists():
            raise FileNotFoundError(f"归档文件不存在: {archive_path}")

        result: dict[str, Any] = {"imported": {}, "merged": merge, "errors": []}
        ext = "".join(src.suffixes).lower()

        try:
            handler = _import_handler(ext, src.name)
            getattr(self, handler)(src, merge, result)
        except Exception as e:
            result["errors"].append(f"导入失败: {e}")
            logger.error("导入归档失败", exc_info=True)

        return result

    # ── 导入实现 ────────────────────────────────────────────────────

    def _find_export_root(self, base: Path) -> Path | None:
        """在解压目录里找到 huginn_export 根目录."""
        # 直接命中
        manifest = base / "huginn_export" / "manifest.json"
        if manifest.exists():
            return base / "huginn_export"
        # 有可能解压时多套了一层
        for child in base.iterdir():
            if (child / "manifest.json").exists():
                return child
            if (child / "huginn_export" / "manifest.json").exists():
                return child / "huginn_export"
        return None

    def _import_from_dir(self, base: Path, merge: bool, result: dict) -> None:
        """从解压后的目录结构导入."""
        root = self._find_export_root(base)
        if root is None:
            result["errors"].append(
                "归档中找不到 manifest.json，可能不是 Huginn 导出文件"
            )
            return

        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            result["errors"].append(f"无法读取 manifest: {e}")
            return

        included = manifest.get("included", ALL_COMPONENTS)

        if "memory" in included:
            self._import_memory(root / "memory", merge, result)
        if "knowledge" in included:
            self._import_knowledge(root / "knowledge", merge, result)
        if "graph" in included:
            self._import_graph(root / "graph", merge, result)
        if "sessions" in included:
            self._import_sessions(root / "sessions", merge, result)
        if "config" in included:
            self._import_config(root / "config", merge, result)

    @staticmethod
    def _im_json_merge(mem_json: Path, longterm: Any, result: dict) -> None:
        """优先用 JSON 合并（增量导入）."""
        try:
            result["imported"]["memory_entries"] = longterm.import_(str(mem_json))
        except Exception as e:
            result["errors"].append(f"记忆 JSON 导入失败: {e}")

    @staticmethod
    def _im_clear_and_import(mem_json: Path, longterm: Any, result: dict) -> None:
        """不合并时，先清空再导入."""
        try:
            with longterm._connect() as conn:
                conn.execute("DELETE FROM memories")
                conn.commit()
            # G35: bulk clear 绕过 FTS5, 重建索引 (import_ 会重新灌 FTS5,
            # 但先清掉孤儿索引行防止 import_ 失败时留下脏状态)
            longterm._rebuild_fts_index()
            imported = longterm.import_(str(mem_json)) if mem_json.exists() else 0
            result["imported"]["memory_entries"] = imported
        except Exception as e:
            result["errors"].append(f"记忆覆盖导入失败: {e}")

    def _im_memory_md(self, mem_dir: Path, result: dict) -> None:
        """恢复 MEMORY.md."""
        memory_md = mem_dir / "MEMORY.md"
        if not memory_md.exists():
            return
        try:
            shutil.copy2(str(memory_md), str(self.workspace / "MEMORY.md"))
            result["imported"]["memory_md"] = True
        except Exception as e:
            result["errors"].append(f"MEMORY.md 恢复失败: {e}")

    def _im_topics(self, mem_dir: Path, merge: bool, result: dict) -> None:
        """恢复主题记忆目录."""
        topics_src = mem_dir / "memory_topics"
        if not topics_src.is_dir():
            return
        topics_dst = self.workspace / HUGINN_DIR_NAME / "memory"
        try:
            if not merge and topics_dst.is_dir():
                shutil.rmtree(str(topics_dst))
            self._copy_tree(topics_src, topics_dst)
            result["imported"]["memory_topics"] = True
        except Exception as e:
            result["errors"].append(f"主题记忆恢复失败: {e}")

    def _import_memory(self, mem_dir: Path, merge: bool, result: dict) -> None:
        """导入长期记忆."""
        if not mem_dir.is_dir():
            return
        longterm = self._get_longterm()
        mem_json = mem_dir / "memories.json"

        if mem_json.exists() and longterm is not None:
            self._im_json_merge(mem_json, longterm, result)
        if not merge and longterm is not None:
            self._im_clear_and_import(mem_json, longterm, result)

        self._im_memory_md(mem_dir, result)
        self._im_topics(mem_dir, merge, result)

    @staticmethod
    def _ik_clear(kb: Any, result: dict, *, record_error: bool = True) -> None:
        """不合并时先清空知识库."""
        try:
            all_data = kb.collection.get(include=[])
            ids = all_data.get("ids") or []
            if ids:
                kb.collection.delete(ids=ids)
        except Exception as e:
            if record_error:
                result["errors"].append(f"知识库清空失败: {e}")
            else:
                logger.debug("get failed", exc_info=True)

    @staticmethod
    def _ik_chunks(kb: Any, chunks_path: Path, result: dict) -> None:
        """从 chunks.jsonl 逐行导入."""
        try:
            with open(chunks_path, encoding="utf-8") as f:
                records = (json.loads(line) for line in f if line.strip())
                result["imported"]["knowledge_chunks"] = _kb_import_batched(kb, records)
        except Exception as e:
            result["errors"].append(f"知识库 chunks 导入失败: {e}")

    def _ik_docs(self, kb_dir: Path, merge: bool, result: dict) -> None:
        """恢复原始文档文件."""
        docs_src = kb_dir / "docs"
        if not docs_src.is_dir():
            return
        docs_dst = self.workspace / ".huginn_kb" / "docs"
        try:
            if not merge and docs_dst.is_dir():
                shutil.rmtree(str(docs_dst))
            self._copy_tree(docs_src, docs_dst)
            result["imported"]["knowledge_docs"] = True
        except Exception as e:
            result["errors"].append(f"文档文件恢复失败: {e}")

    def _import_knowledge(self, kb_dir: Path, merge: bool, result: dict) -> None:
        """导入知识库."""
        if not kb_dir.is_dir():
            return
        kb = self._get_kb()
        if kb is None:
            result["errors"].append("知识库不可用，无法导入")
            return

        if not merge:
            self._ik_clear(kb, result)

        chunks_path = kb_dir / "chunks.jsonl"
        if chunks_path.exists():
            self._ik_chunks(kb, chunks_path, result)

        self._ik_docs(kb_dir, merge, result)

    @staticmethod
    def _kb_upsert(
        kb: Any,
        ids: list[str],
        docs: list[str],
        metas: list[dict],
        embs: list[list[float]],
    ) -> None:
        """批量 upsert 到 ChromaDB，有 embeddings 就带上."""
        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": docs,
            "metadatas": metas,
        }
        if embs and len(embs) == len(ids):
            kwargs["embeddings"] = embs
        kb.collection.upsert(**kwargs)

    def _write_kg_file(self, data: dict[str, Any]) -> None:
        """把图谱数据直接写到 project_kg.json."""
        dst = self.workspace / HUGINN_DIR_NAME / "project_kg.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _ig_merge(self, kg: Any, data: dict[str, Any], result: dict) -> None:
        """合并：加载当前图谱，逐个添加节点和边."""
        try:
            nodes_added = 0
            edges_added = 0
            for node in data.get("nodes", []):
                kg.add_entity(
                    label=node.get("label", node.get("id", "unknown")),
                    entity_type=node.get("type", "Unknown"),
                    source=node.get("source", "import"),
                    confidence=node.get("confidence", 0.5),
                )
                nodes_added += 1
            for edge in data.get("links", []):
                kg.add_relation(
                    src_id=edge.get("source", ""),
                    relation=edge.get("relation", "related_to"),
                    dst_id=edge.get("target", ""),
                    source=edge.get("source_attr", "import"),
                    confidence=edge.get("confidence", 0.5),
                )
                edges_added += 1
            kg.save()
            result["imported"]["graph_nodes"] = nodes_added
            result["imported"]["graph_edges"] = edges_added
        except Exception as e:
            result["errors"].append(f"知识图谱合并失败: {e}")

    def _import_graph(self, graph_dir: Path, merge: bool, result: dict) -> None:
        """导入知识图谱."""
        if not graph_dir.is_dir():
            return
        kg_file = graph_dir / "project_kg.json"
        if not kg_file.exists():
            return

        try:
            data = json.loads(kg_file.read_text(encoding="utf-8"))
        except Exception as e:
            result["errors"].append(f"知识图谱读取失败: {e}")
            return

        if not merge:
            self._write_kg_file(data)
            result["imported"]["graph"] = "overwritten"
            return

        kg = self._get_kg()
        if kg is None:
            # 退而求其次，直接写文件
            self._write_kg_file(data)
            result["imported"]["graph"] = "written"
            return

        self._ig_merge(kg, data, result)

    def _import_sessions(self, sess_dir: Path, merge: bool, result: dict) -> None:
        """导入会话元数据."""
        threads_file = sess_dir / "threads.json"
        if not threads_file.exists():
            return
        try:
            sessions = json.loads(threads_file.read_text(encoding="utf-8"))
            # 会话是临时状态，合并时直接加入内存
            try:
                _merge_sessions_into_threads(sessions, merge)
                result["imported"]["sessions"] = len(sessions)
            except Exception:
                # server_core 不可用，只记录数量
                result["imported"]["sessions_count"] = len(sessions)
        except Exception as e:
            result["errors"].append(f"会话导入失败: {e}")

    def _import_config(self, cfg_dir: Path, merge: bool, result: dict) -> None:
        """导入配置（只读展示，不自动覆盖运行时配置）."""
        cfg_file = cfg_dir / "config.json"
        if not cfg_file.exists():
            return
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            result["imported"]["config"] = data
            # 配置不自动写入运行时，避免误覆盖密钥等敏感信息
            result["imported"]["config_note"] = (
                "配置已加载但未自动写入运行时，请手动确认"
            )
        except Exception as e:
            result["errors"].append(f"配置导入失败: {e}")

    def _ijf_memory(self, payload: dict[str, Any], merge: bool, result: dict) -> None:
        """从 JSON payload 导入记忆."""
        longterm = self._get_longterm()
        if longterm is not None:
            entries = payload["memory"].get("entries", [])
            if not merge:
                try:
                    with longterm._connect() as conn:
                        conn.execute("DELETE FROM memories")
                        conn.commit()
                    # G35: bulk clear 绕过 FTS5, 重建索引
                    longterm._rebuild_fts_index()
                except Exception:
                    logger.debug("connect failed", exc_info=True)
            # 写到临时文件再用 import_ 方法
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as tf:
                json.dump(entries, tf, ensure_ascii=False)
                tf_path = tf.name
            try:
                result["imported"]["memory_entries"] = longterm.import_(tf_path)
            except Exception as e:
                result["errors"].append(f"记忆导入失败: {e}")
            finally:
                os.unlink(tf_path)

        if "memory_md" in payload:
            dst = self.workspace / "MEMORY.md"
            dst.write_text(payload["memory_md"], encoding="utf-8")
            result["imported"]["memory_md"] = True

    def _ijf_knowledge(
        self, payload: dict[str, Any], merge: bool, result: dict
    ) -> None:
        """从 JSON payload 导入知识库."""
        kb = self._get_kb()
        if kb is None:
            return
        if not merge:
            self._ik_clear(kb, result, record_error=False)
        chunks = payload["knowledge"].get("chunks", [])
        records = (
            {
                "id": chunk.get("id", ""),
                "document": chunk.get("document", ""),
                "metadata": chunk.get("metadata", {}),
                "embedding": chunk.get("embedding"),
            }
            for chunk in chunks
        )
        result["imported"]["knowledge_chunks"] = _kb_import_batched(kb, records)

    def _ijf_graph(self, payload: dict[str, Any], merge: bool, result: dict) -> None:
        """从 JSON payload 导入知识图谱."""
        graph_data = payload["graph"]
        kg = self._get_kg()
        if kg is not None and merge:
            try:
                for node in graph_data.get("nodes", []):
                    kg.add_entity(
                        label=node.get("label", "unknown"),
                        entity_type=node.get("type", "Unknown"),
                        source=node.get("source", "import"),
                        confidence=node.get("confidence", 0.5),
                    )
                for edge in graph_data.get("links", []):
                    kg.add_relation(
                        src_id=edge.get("source", ""),
                        relation=edge.get("relation", "related_to"),
                        dst_id=edge.get("target", ""),
                        source=edge.get("source_attr", "import"),
                        confidence=edge.get("confidence", 0.5),
                    )
                kg.save()
                result["imported"]["graph"] = "merged"
            except Exception as e:
                result["errors"].append(f"图谱合并失败: {e}")
        else:
            self._write_kg_file(graph_data)
            result["imported"]["graph"] = "written"

    @staticmethod
    def _ijf_sessions(payload: dict[str, Any], result: dict) -> None:
        """从 JSON payload 导入会话."""
        try:
            from huginn import server_core

            with server_core._state_lock:
                for s in payload["sessions"]:
                    tid = s.get("id")
                    if tid:
                        server_core._threads[tid] = s
            result["imported"]["sessions"] = len(payload["sessions"])
        except Exception:
            result["imported"]["sessions_count"] = len(payload.get("sessions", []))

    @staticmethod
    def _ijf_config(payload: dict[str, Any], result: dict) -> None:
        """从 JSON payload 导入配置（只读展示）."""
        result["imported"]["config"] = payload["config"]
        result["imported"]["config_note"] = "配置已加载但未自动写入运行时，请手动确认"

    def _import_from_json(self, json_path: Path, merge: bool, result: dict) -> None:
        """从单个 JSON 文件导入（对应 export_all format=json）."""
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            result["errors"].append(f"JSON 读取失败: {e}")
            return

        included = payload.get("included", ALL_COMPONENTS)

        importers = {
            "memory": self._ijf_memory,
            "knowledge": self._ijf_knowledge,
            "graph": self._ijf_graph,
            "sessions": self._ijf_sessions,
            "config": self._ijf_config,
        }
        for component, importer in importers.items():
            if component in included and component in payload:
                if component in ("memory", "knowledge", "graph"):
                    importer(payload, merge, result)
                else:
                    importer(payload, result)

    # ── 单组件导出（返回 bytes）────────────────────────────────────

    def export_memory(self, format: str = "json") -> bytes:
        """只导出长期记忆，返回 bytes.

        包含 SQLite 数据库转储 + MEMORY.md 文件内容。
        """
        if format.lower() == "json":
            longterm = self._get_longterm()
            payload: dict[str, Any] = {}
            if longterm is not None:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as tf:
                    longterm.export(tf.name)
                try:
                    with open(tf.name, encoding="utf-8") as f:
                        payload["entries"] = json.load(f)
                finally:
                    os.unlink(tf.name)
            memory_md = self.workspace / "MEMORY.md"
            if memory_md.exists():
                payload["memory_md"] = memory_md.read_text(encoding="utf-8")
            return json.dumps(
                payload, indent=2, ensure_ascii=False, default=str
            ).encode("utf-8")
        elif format.lower() == "zip":
            return _component_zip_bytes(self._export_memory_to_dir, "memory")
        else:
            raise ValueError(f"不支持的格式: {format}，可选 json / zip")

    def _ek_json_payload(self) -> dict[str, Any]:
        """把知识库拼成单文件 JSON payload."""
        kb = self._get_kb()
        payload: dict[str, Any] = {"documents": [], "chunks": []}
        if kb is None:
            return payload
        try:
            payload["documents"] = kb.list_documents()
            data = kb.collection.get(include=["documents", "metadatas", "embeddings"])
            payload["chunks"] = _chunk_records(
                data.get("ids") or [],
                data.get("documents") or [],
                data.get("metadatas") or [],
                data.get("embeddings") or [],
            )
        except Exception as e:
            payload["error"] = str(e)
        return payload

    def export_knowledge(self, format: str = "json") -> bytes:
        """只导出知识库，返回 bytes.

        包含 ChromaDB collection 数据 + 文档文件。
        """
        if format.lower() == "json":
            payload = self._ek_json_payload()
            return json.dumps(
                payload, indent=2, ensure_ascii=False, default=str
            ).encode("utf-8")
        elif format.lower() == "zip":
            return _component_zip_bytes(self._export_knowledge_to_dir, "knowledge")
        else:
            raise ValueError(f"不支持的格式: {format}，可选 json / zip")

    # ── 状态查询 ────────────────────────────────────────────────────

    def get_export_status(self) -> dict[str, Any]:
        """检查当前有哪些数据可以导出."""
        status: dict[str, Any] = {}

        # 长期记忆
        db_path = self._find_sqlite_db()
        longterm = self._get_longterm()
        status["memory"] = {
            "available": longterm is not None or db_path is not None,
            "db_path": str(db_path) if db_path else None,
            "memory_md": (self.workspace / "MEMORY.md").exists(),
            "entry_count": len(longterm.list_all(limit=999999)) if longterm else 0,
        }

        # 知识库
        kb = self._get_kb()
        kb_dir = self.workspace / ".huginn_kb"
        status["knowledge"] = {
            "available": kb is not None or kb_dir.is_dir(),
            "doc_count": kb.count() if kb else 0,
            "kb_dir": str(kb_dir) if kb_dir.exists() else None,
        }

        # 知识图谱
        kg_file = self.workspace / HUGINN_DIR_NAME / "project_kg.json"
        kg = self._get_kg()
        status["graph"] = {
            "available": kg is not None or kg_file.exists(),
            "node_count": kg.stats()["nodes"] if kg else 0,
            "edge_count": kg.stats()["edges"] if kg else 0,
        }

        # 会话
        sessions = self._get_sessions()
        status["sessions"] = {
            "available": len(sessions) > 0,
            "count": len(sessions),
        }

        # 配置
        cfg = self._get_config_dict()
        status["config"] = {
            "available": cfg is not None,
        }

        return status

    # ── 工具方法 ────────────────────────────────────────────────────

    @staticmethod
    def _copy_tree(src: Path, dst: Path) -> None:
        """递归拷贝目录，已存在时合并."""
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            s = item
            d = dst / item.name
            if s.is_dir():
                ExportShareManager._copy_tree(s, d)
            else:
                shutil.copy2(str(s), str(d))
