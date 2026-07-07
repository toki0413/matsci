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
from typing import Any, Sequence

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

            return ProjectKnowledgeGraph(self.workspace / ".huginn")
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
                for tid, meta in server_core._threads.items():
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
        memory_dir = self.workspace / ".huginn" / "memory"
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
            Path.home() / ".huginn" / "memory.db",
            self.workspace / ".huginn" / "memory.db",
        ]
        for c in candidates:
            if c and c.exists():
                return c
        return None

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

        # 1) 文档列表
        try:
            docs = kb.list_documents()
            (target_dir / "documents.json").write_text(
                json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            stats.knowledge_documents = len(docs)
        except Exception as e:
            stats.errors.append(f"knowledge docs: {e}")

        # 2) 全部 chunks + embeddings（分批写出，防止内存炸）
        chunks_path = target_dir / "chunks.jsonl"
        total_chunks = 0
        try:
            with open(chunks_path, "w", encoding="utf-8") as f:
                data = kb.collection.get(
                    include=["documents", "metadatas", "embeddings"]
                )
                ids = data.get("ids") or []
                documents = data.get("documents") or []
                metadatas = data.get("metadatas") or []
                embeddings = data.get("embeddings") or []

                for i in range(0, len(ids), _CHUNK_BATCH):
                    batch = ids[i : i + _CHUNK_BATCH]
                    for j, cid in enumerate(batch):
                        idx = i + j
                        record = {
                            "id": cid,
                            "document": documents[idx] if idx < len(documents) else "",
                            "metadata": metadatas[idx] if idx < len(metadatas) else {},
                            "embedding": embeddings[idx]
                            if idx < len(embeddings)
                            else None,
                        }
                        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    total_chunks += len(batch)
            stats.knowledge_chunks = total_chunks
        except Exception as e:
            stats.errors.append(f"knowledge chunks: {e}")

        # 3) 原始文档文件
        docs_dir = target_dir / "docs"
        kb_docs = self.workspace / ".huginn_kb" / "docs"
        if kb_docs.is_dir():
            self._copy_tree(kb_docs, docs_dir)

    def _export_graph_to_dir(self, target_dir: Path, stats: ExportStats) -> None:
        """把知识图谱导出到指定目录."""
        target_dir.mkdir(parents=True, exist_ok=True)
        kg = self._get_kg()

        if kg is None:
            # 直接拷贝文件
            kg_file = self.workspace / ".huginn" / "project_kg.json"
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

        if "memory" in include:
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
                    stats.memory_entries = len(mem_data) if isinstance(mem_data, list) else 0
                except Exception as e:
                    stats.errors.append(f"memory json: {e}")
            memory_md = self.workspace / "MEMORY.md"
            if memory_md.exists():
                payload["memory_md"] = memory_md.read_text(encoding="utf-8")
                stats.memory_md_exists = True

        if "knowledge" in include:
            kb = self._get_kb()
            if kb is not None:
                try:
                    docs = kb.list_documents()
                    data = kb.collection.get(
                        include=["documents", "metadatas", "embeddings"]
                    )
                    chunks = []
                    ids = data.get("ids") or []
                    documents = data.get("documents") or []
                    metadatas = data.get("metadatas") or []
                    embeddings = data.get("embeddings") or []
                    for idx, cid in enumerate(ids):
                        chunks.append({
                            "id": cid,
                            "document": documents[idx] if idx < len(documents) else "",
                            "metadata": metadatas[idx] if idx < len(metadatas) else {},
                            "embedding": embeddings[idx] if idx < len(embeddings) else None,
                        })
                    payload["knowledge"] = {"documents": docs, "chunks": chunks}
                    stats.knowledge_documents = len(docs)
                    stats.knowledge_chunks = len(chunks)
                except Exception as e:
                    stats.errors.append(f"knowledge json: {e}")

        if "graph" in include:
            kg = self._get_kg()
            if kg is not None:
                try:
                    graph_data = kg.export(fmt="json")
                    payload["graph"] = graph_data
                    stats.graph_nodes = len(graph_data.get("nodes", []))
                    stats.graph_edges = len(graph_data.get("links", []))
                except Exception as e:
                    stats.errors.append(f"graph json: {e}")

        if "sessions" in include:
            sessions = self._get_sessions()
            payload["sessions"] = sessions
            stats.sessions = len(sessions)

        if "config" in include:
            cfg_dict = self._get_config_dict()
            if cfg_dict is not None:
                payload["config"] = cfg_dict
                stats.config_exported = True

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
            if ext == ".json":
                self._import_from_json(src, merge, result)
            elif ext in (".zip",):
                with tempfile.TemporaryDirectory() as tmp:
                    with zipfile.ZipFile(str(src), "r") as zf:
                        zf.extractall(tmp)
                    self._import_from_dir(Path(tmp), merge, result)
            elif ext in (".gz",) or src.name.endswith(".tar.gz"):
                with tempfile.TemporaryDirectory() as tmp:
                    with tarfile.open(str(src), "r:gz") as tf:
                        tf.extractall(tmp)
                    self._import_from_dir(Path(tmp), merge, result)
            else:
                # 尝试根据文件头判断
                with open(src, "rb") as f:
                    magic = f.read(4)
                if magic[:2] == b"PK":
                    with tempfile.TemporaryDirectory() as tmp:
                        with zipfile.ZipFile(str(src), "r") as zf:
                            zf.extractall(tmp)
                        self._import_from_dir(Path(tmp), merge, result)
                else:
                    result["errors"].append(f"无法识别的归档格式: {src.name}")
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
            result["errors"].append("归档中找不到 manifest.json，可能不是 Huginn 导出文件")
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

    def _import_memory(self, mem_dir: Path, merge: bool, result: dict) -> None:
        """导入长期记忆."""
        if not mem_dir.is_dir():
            return
        imported = 0
        longterm = self._get_longterm()

        # 优先用 JSON 合并（增量导入）
        mem_json = mem_dir / "memories.json"
        if mem_json.exists() and longterm is not None:
            try:
                imported = longterm.import_(str(mem_json))
                result["imported"]["memory_entries"] = imported
            except Exception as e:
                result["errors"].append(f"记忆 JSON 导入失败: {e}")

        # 不合并时，先清空再导入
        if not merge and longterm is not None:
            try:
                with longterm._connect() as conn:
                    conn.execute("DELETE FROM memories")
                    conn.commit()
                imported = longterm.import_(str(mem_json)) if mem_json.exists() else 0
                result["imported"]["memory_entries"] = imported
            except Exception as e:
                result["errors"].append(f"记忆覆盖导入失败: {e}")

        # 恢复 MEMORY.md
        memory_md = mem_dir / "MEMORY.md"
        if memory_md.exists():
            dst = self.workspace / "MEMORY.md"
            try:
                shutil.copy2(str(memory_md), str(dst))
                result["imported"]["memory_md"] = True
            except Exception as e:
                result["errors"].append(f"MEMORY.md 恢复失败: {e}")

        # 恢复主题记忆目录
        topics_src = mem_dir / "memory_topics"
        if topics_src.is_dir():
            topics_dst = self.workspace / ".huginn" / "memory"
            try:
                if not merge and topics_dst.is_dir():
                    shutil.rmtree(str(topics_dst))
                self._copy_tree(topics_src, topics_dst)
                result["imported"]["memory_topics"] = True
            except Exception as e:
                result["errors"].append(f"主题记忆恢复失败: {e}")

    def _import_knowledge(self, kb_dir: Path, merge: bool, result: dict) -> None:
        """导入知识库."""
        if not kb_dir.is_dir():
            return
        kb = self._get_kb()
        if kb is None:
            result["errors"].append("知识库不可用，无法导入")
            return

        # 不合并时先清空
        if not merge:
            try:
                all_data = kb.collection.get(include=[])
                ids = all_data.get("ids") or []
                if ids:
                    kb.collection.delete(ids=ids)
            except Exception as e:
                result["errors"].append(f"知识库清空失败: {e}")

        # 从 chunks.jsonl 逐行导入
        chunks_path = kb_dir / "chunks.jsonl"
        if chunks_path.exists():
            try:
                batch_ids: list[str] = []
                batch_docs: list[str] = []
                batch_metas: list[dict] = []
                batch_embs: list[list[float]] = []
                imported = 0

                with open(chunks_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        batch_ids.append(record["id"])
                        batch_docs.append(record.get("document", ""))
                        batch_metas.append(record.get("metadata", {}))
                        emb = record.get("embedding")
                        if emb is not None:
                            batch_embs.append(emb)

                        if len(batch_ids) >= _CHUNK_BATCH:
                            self._kb_upsert(
                                kb, batch_ids, batch_docs, batch_metas, batch_embs
                            )
                            imported += len(batch_ids)
                            batch_ids, batch_docs, batch_metas, batch_embs = (
                                [],
                                [],
                                [],
                                [],
                            )

                # 写入剩余
                if batch_ids:
                    self._kb_upsert(kb, batch_ids, batch_docs, batch_metas, batch_embs)
                    imported += len(batch_ids)

                result["imported"]["knowledge_chunks"] = imported
            except Exception as e:
                result["errors"].append(f"知识库 chunks 导入失败: {e}")

        # 恢复原始文档文件
        docs_src = kb_dir / "docs"
        if docs_src.is_dir():
            docs_dst = self.workspace / ".huginn_kb" / "docs"
            try:
                if not merge and docs_dst.is_dir():
                    shutil.rmtree(str(docs_dst))
                self._copy_tree(docs_src, docs_dst)
                result["imported"]["knowledge_docs"] = True
            except Exception as e:
                result["errors"].append(f"文档文件恢复失败: {e}")

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
            # 直接覆盖文件
            dst = self.workspace / ".huginn" / "project_kg.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            result["imported"]["graph"] = "overwritten"
        else:
            # 合并：加载当前图谱，逐个添加节点和边
            kg = self._get_kg()
            if kg is None:
                # 退而求其次，直接写文件
                dst = self.workspace / ".huginn" / "project_kg.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                result["imported"]["graph"] = "written"
            else:
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

    def _import_sessions(self, sess_dir: Path, merge: bool, result: dict) -> None:
        """导入会话元数据."""
        threads_file = sess_dir / "threads.json"
        if not threads_file.exists():
            return
        try:
            sessions = json.loads(threads_file.read_text(encoding="utf-8"))
            # 会话是临时状态，合并时直接加入内存
            try:
                from huginn import server_core

                with server_core._state_lock:
                    for s in sessions:
                        tid = s.get("id")
                        if tid:
                            if not merge and tid in server_core._threads:
                                continue
                            server_core._threads[tid] = s
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
            result["imported"]["config_note"] = "配置已加载但未自动写入运行时，请手动确认"
        except Exception as e:
            result["errors"].append(f"配置导入失败: {e}")

    def _import_from_json(self, json_path: Path, merge: bool, result: dict) -> None:
        """从单个 JSON 文件导入（对应 export_all format=json）."""
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            result["errors"].append(f"JSON 读取失败: {e}")
            return

        included = payload.get("included", ALL_COMPONENTS)

        # 记忆
        if "memory" in included and "memory" in payload:
            longterm = self._get_longterm()
            if longterm is not None:
                entries = payload["memory"].get("entries", [])
                if not merge:
                    try:
                        with longterm._connect() as conn:
                            conn.execute("DELETE FROM memories")
                            conn.commit()
                    except Exception:
                        logger.debug("connect failed", exc_info=True)
                # 写到临时文件再用 import_ 方法
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as tf:
                    json.dump(entries, tf, ensure_ascii=False)
                    tf_path = tf.name
                try:
                    count = longterm.import_(tf_path)
                    result["imported"]["memory_entries"] = count
                except Exception as e:
                    result["errors"].append(f"记忆导入失败: {e}")
                finally:
                    os.unlink(tf_path)

            if "memory_md" in payload:
                dst = self.workspace / "MEMORY.md"
                dst.write_text(payload["memory_md"], encoding="utf-8")
                result["imported"]["memory_md"] = True

        # 知识库
        if "knowledge" in included and "knowledge" in payload:
            kb = self._get_kb()
            if kb is not None:
                if not merge:
                    try:
                        all_data = kb.collection.get(include=[])
                        ids = all_data.get("ids") or []
                        if ids:
                            kb.collection.delete(ids=ids)
                    except Exception:
                        logger.debug("get failed", exc_info=True)
                chunks = payload["knowledge"].get("chunks", [])
                batch_ids: list[str] = []
                batch_docs: list[str] = []
                batch_metas: list[dict] = []
                batch_embs: list[list[float]] = []
                imported = 0
                for chunk in chunks:
                    batch_ids.append(chunk.get("id", ""))
                    batch_docs.append(chunk.get("document", ""))
                    batch_metas.append(chunk.get("metadata", {}))
                    emb = chunk.get("embedding")
                    if emb is not None:
                        batch_embs.append(emb)
                    if len(batch_ids) >= _CHUNK_BATCH:
                        self._kb_upsert(kb, batch_ids, batch_docs, batch_metas, batch_embs)
                        imported += len(batch_ids)
                        batch_ids, batch_docs, batch_metas, batch_embs = [], [], [], []
                if batch_ids:
                    self._kb_upsert(kb, batch_ids, batch_docs, batch_metas, batch_embs)
                    imported += len(batch_ids)
                result["imported"]["knowledge_chunks"] = imported

        # 知识图谱
        if "graph" in included and "graph" in payload:
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
                dst = self.workspace / ".huginn" / "project_kg.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(
                    json.dumps(graph_data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                result["imported"]["graph"] = "written"

        # 会话
        if "sessions" in included and "sessions" in payload:
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

        # 配置
        if "config" in included and "config" in payload:
            result["imported"]["config"] = payload["config"]
            result["imported"]["config_note"] = "配置已加载但未自动写入运行时，请手动确认"

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
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                stats = ExportStats()
                self._export_memory_to_dir(tmp_dir / "memory", stats)
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fp in (tmp_dir / "memory").rglob("*"):
                        if fp.is_file():
                            zf.write(str(fp), fp.relative_to(tmp_dir / "memory"))
                return buf.getvalue()
        else:
            raise ValueError(f"不支持的格式: {format}，可选 json / zip")

    def export_knowledge(self, format: str = "json") -> bytes:
        """只导出知识库，返回 bytes.

        包含 ChromaDB collection 数据 + 文档文件。
        """
        if format.lower() == "json":
            kb = self._get_kb()
            payload: dict[str, Any] = {"documents": [], "chunks": []}
            if kb is not None:
                try:
                    payload["documents"] = kb.list_documents()
                    data = kb.collection.get(
                        include=["documents", "metadatas", "embeddings"]
                    )
                    ids = data.get("ids") or []
                    documents = data.get("documents") or []
                    metadatas = data.get("metadatas") or []
                    embeddings = data.get("embeddings") or []
                    for idx, cid in enumerate(ids):
                        payload["chunks"].append({
                            "id": cid,
                            "document": documents[idx] if idx < len(documents) else "",
                            "metadata": metadatas[idx] if idx < len(metadatas) else {},
                            "embedding": embeddings[idx] if idx < len(embeddings) else None,
                        })
                except Exception as e:
                    payload["error"] = str(e)
            return json.dumps(
                payload, indent=2, ensure_ascii=False, default=str
            ).encode("utf-8")
        elif format.lower() == "zip":
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                stats = ExportStats()
                self._export_knowledge_to_dir(tmp_dir / "knowledge", stats)
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fp in (tmp_dir / "knowledge").rglob("*"):
                        if fp.is_file():
                            zf.write(str(fp), fp.relative_to(tmp_dir / "knowledge"))
                return buf.getvalue()
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
        kg_file = self.workspace / ".huginn" / "project_kg.json"
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
