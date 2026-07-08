"""步骤级文件快照与回滚.

灵感来自 OpenCode/OpenScience 的 snapshot/ 机制:
  - track()    : 工具执行前给工作区文件拍照 (记 sha256)
  - patch()    : 工具执行后比对哈希, 产出变化列表
  - revert()   : 把文件回滚到执行前状态
  - unrevert() : 撤销回滚, 恢复到执行后状态

和 huginn 已有的 turn 级 trajectory 日志互补: trajectory 记的是"对话",
这里记的是"文件系统", 给 agent 一个真正的 undo 能力.

存储约定 (跟 checkpointer / speculator_history 一样放全局 ~/.huginn/):
  ~/.huginn/snapshots/snapshots.jsonl        — 只追加的快照元数据日志
  ~/.huginn/snapshots/{step_id}/files/       — 执行前文件内容备份 (revert 用)
  ~/.huginn/snapshots/{step_id}/patches.json — patch() 产出的变化列表
  ~/.huginn/snapshots/{step_id}/reverted.json — 当前是否已回滚
  ~/.huginn/snapshots/{step_id}/revert_point/ — revert() 时存的执行后状态 (unrevert 用)
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ── 存储位置 ─────────────────────────────────────────────────────

_SNAPSHOT_ROOT = Path.home() / ".huginn" / "snapshots"
# 日志文件路径跟着实例 root 走 (见 _log_file), 不用模块常量 ——
# 这样非单例实例 (测试用独立 root) 也能正确隔离存储.

# 默认盯的材料科学文件后缀 (大小写不敏感). VASP 那几个没后缀的裸文件名
# 也一并认上 —— 这是 matsci agent, POSCAR/INCAR 几乎天天见.
# ponytail: 默认列表写死, 新增后缀要改这里; 比全盘扫描省 IO.
_DEFAULT_WATCH_PATTERNS: tuple[str, ...] = (
    "*.py", "*.cif", "*.poscar", "*.json", "*.yaml", "*.yml",
    "*.toml", "*.out", "*.dat", "*.csv",
    # VASP 裸文件名, 当字面量匹配
    "POSCAR", "CONTCAR", "INCAR", "KPOINTS", "OUTCAR", "CHGCAR", "WAVECAR",
)

# os.walk 下钻时跳过的目录: 缓存 / 版本库 / huginn 自己的输出.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".huginn", ".huginn_kb",
    "node_modules", ".venv", "venv", "env", ".tox", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", ".eggs", ".cache",
})

# 单文件备份上限: 超过就不拷内容 (太大, 回滚也不现实). 哈希照记, patch 照报.
_MAX_BACKUP_BYTES = 5 * 1024 * 1024  # 5 MiB

_PREVIEW_LEN = 500   # 内容预览截断长度, 跟 spec 对齐
_MAX_SNAPSHOTS = 100  # FIFO 上限


# ── 小工具 ───────────────────────────────────────────────────────


def _hash_file(path: Path) -> str | None:
    """流式算文件 sha256. 读不出来 (权限/损坏) 返回 None."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _safe_backup_name(rel_path: str) -> str:
    """把相对路径映射成安全的备份文件名.

    用 sha256 短摘要, 避免斜杠 / 特殊字符 / 平台差异.
    是确定性的, 所以恢复时不用存 manifest 反查, 直接按 rel 重算即可.
    """
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]


def _preview(content: bytes) -> str:
    """取前 _PREVIEW_LEN 字符做预览. 二进制/乱码用 replace 兜底."""
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = str(content)
    return text[:_PREVIEW_LEN]


def _preview_file(path: Path) -> str:
    """读文件头部做预览. 多读几字节防止 utf-8 多字节被截."""
    try:
        with path.open("rb") as f:
            return _preview(f.read(_PREVIEW_LEN * 4))
    except OSError:
        return ""


def _read_backup_preview(files_dir: Path, rel: str) -> str:
    bname = _safe_backup_name(rel)
    try:
        with (files_dir / bname).open("rb") as f:
            return _preview(f.read(_PREVIEW_LEN * 4))
    except OSError:
        return ""


def _write_json(path: Path, obj: object) -> None:
    """原子写 JSON: 先写 .tmp 再 replace, 防中途崩了留半截文件."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=str)
    tmp.replace(path)


def _read_json(path: Path, default: object) -> object:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _make_step_id(tool_name: str) -> str:
    """tool_name + 高分辨率时间 + 随机盐, 哈希成 16 位短 id. 单调唯一."""
    raw = f"{tool_name}:{time.time_ns()}:{os.urandom(4).hex()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _match_any(name_lower: str, patterns_lower: tuple[str, ...]) -> bool:
    """名字是否匹配任一 pattern. 支持 glob (* ? [) 和字面量名."""
    for p in patterns_lower:
        if any(ch in p for ch in ("*", "?", "[")):
            if fnmatch.fnmatch(name_lower, p):
                return True
        elif name_lower == p:
            return True
    return False


def _walk_watch(
    ws: Path, patterns: tuple[str, ...]
) -> Iterator[tuple[str, Path]]:
    """遍历 ws 下匹配 patterns 的文件, yield (相对路径, 绝对路径).

    相对路径用 posix 风格 (正斜杠), 跨平台一致, 也方便存 JSON.
    就地裁剪 dirnames 阻止 os.walk 钻进 _SKIP_DIRS.
    """
    patterns_lower = tuple(p.lower() for p in patterns)
    for dirpath, dirnames, filenames in os.walk(ws):
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in _SKIP_DIRS and not d.endswith(".egg-info")
        ]
        for fn in filenames:
            if _match_any(fn.lower(), patterns_lower):
                full = Path(dirpath) / fn
                rel = full.relative_to(ws).as_posix()
                yield rel, full


# ── 数据类 ───────────────────────────────────────────────────────


@dataclass
class FilePatch:
    """单个文件在快照前后的变化."""

    file_path: str                  # 相对工作区的路径
    change_type: str                # "created" | "modified" | "deleted"
    old_hash: str | None
    new_hash: str | None
    old_content_preview: str        # 执行前内容前 500 字符
    new_content_preview: str        # 执行后内容前 500 字符


@dataclass
class FileSnapshot:
    """一次工具执行步骤的文件系统快照."""

    step_id: str
    tool_name: str
    timestamp: float
    files: dict[str, str] = field(default_factory=dict)   # rel_path -> sha256 (执行前)
    patches: list[FilePatch] = field(default_factory=list)
    reverted: bool = False
    workspace: str = ""             # track 时的工作区, 回滚要回这里
    watch_patterns: list[str] = field(default_factory=list)


def _snapshot_from_record(rec: dict) -> FileSnapshot:
    return FileSnapshot(
        step_id=rec["step_id"],
        tool_name=rec.get("tool_name", ""),
        timestamp=rec.get("timestamp", 0.0),
        files=rec.get("files", {}),
        workspace=rec.get("workspace", ""),
        watch_patterns=rec.get("watch_patterns", []),
    )


# ── 管理器 ───────────────────────────────────────────────────────


class SnapshotManager:
    """管理步骤级文件快照, 支持 revert/unrevert. 单例.

    SnapshotManager() 拿全局单例 (默认存 ~/.huginn/snapshots).
    测试想隔离存储时传 root=, 拿一个独立实例, 不影响单例.
    """

    _instance: "SnapshotManager | None" = None

    def __new__(cls, root: Path | None = None) -> "SnapshotManager":
        if root is not None:
            # 显式 root: 每次新建, 给测试用, 不碰单例.
            inst = super().__new__(cls)
            inst._init(root)
            return inst
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(_SNAPSHOT_ROOT)
        return cls._instance

    def _init(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    # ---- 路径助手 ----

    def _snap_dir(self, step_id: str) -> Path:
        return self._root / step_id

    def _log_file(self) -> Path:
        # jsonl 跟着实例 root, 单例默认就是 ~/.huginn/snapshots/snapshots.jsonl
        return self._root / "snapshots.jsonl"

    def _files_dir(self, step_id: str) -> Path:
        return self._snap_dir(step_id) / "files"

    def _patches_file(self, step_id: str) -> Path:
        return self._snap_dir(step_id) / "patches.json"

    def _reverted_file(self, step_id: str) -> Path:
        return self._snap_dir(step_id) / "reverted.json"

    def _revert_dir(self, step_id: str) -> Path:
        return self._snap_dir(step_id) / "revert_point"

    # ---- 拍照 ----

    def track(
        self,
        tool_name: str,
        workspace: Path,
        watch_patterns: list[str] | None = None,
    ) -> str:
        """工具执行前对工作区拍照, 返回 step_id.

        记下所有匹配文件的 sha256, 并把内容备份到 {step_id}/files/.
        备份是 revert 的前提 —— patch 只报变化, 但回滚要的是旧内容,
        所以这里必须把执行前内容存下来 (只盯受 watch_patterns 限制的小集合,
        不是全盘拷). ponytail: 大于 5MiB 的文件跳过备份, 回滚时也跳过.
        文件备份放到后台线程跑 (Windows AV 扫描 copyfile 会卡几十秒),
        哈希计算留主线程 (revert 要用, 不能延迟).
        """
        ws = Path(workspace).resolve()
        patterns = tuple(watch_patterns) if watch_patterns else _DEFAULT_WATCH_PATTERNS
        step_id = _make_step_id(tool_name)

        snap = FileSnapshot(
            step_id=step_id,
            tool_name=tool_name,
            timestamp=time.time(),
            workspace=str(ws),
            watch_patterns=list(patterns),
        )

        files_dir = self._files_dir(step_id)
        files_dir.mkdir(parents=True, exist_ok=True)

        # 先算哈希 (主线程, 快), 收集需要备份的文件对
        to_backup: list[tuple[Path, str]] = []
        for rel, fpath in _walk_watch(ws, patterns):
            digest = _hash_file(fpath)
            if digest is None:
                continue
            snap.files[rel] = digest
            try:
                size = fpath.stat().st_size
            except OSError:
                size = 0
            if size > _MAX_BACKUP_BYTES:
                continue
            to_backup.append((fpath, _safe_backup_name(rel)))

        # 备份丢后台线程, 不阻塞 track() 返回
        def _do_backup():
            for src, bname in to_backup:
                try:
                    shutil.copyfile(src, files_dir / bname)
                except OSError as exc:
                    logger.debug("snapshot backup skip %s: %s", src, exc)

        t = threading.Thread(target=_do_backup, name=f"snap-backup-{step_id}", daemon=True)
        t.start()
        # 记住线程, patch() 里可以 join 确保备份完成再做 diff
        self._pending_backups: dict[str, threading.Thread] = getattr(self, "_pending_backups", {})
        self._pending_backups[step_id] = t

        self._append_log(snap)
        self._enforce_cap()
        logger.debug(
            "snapshot track %s (%s): %d files, %d queued for backup",
            step_id, tool_name, len(snap.files), len(to_backup),
        )
        return step_id

    # ---- 比对 ----

    def patch(self, step_id: str, workspace: Path) -> list[FilePatch]:
        """工具执行后比对当前状态与快照, 返回变化列表.

        只对真正变化的文件产 FilePatch (created/modified/deleted),
        没变的不会进列表 —— 这就是 "diff by hash, not full content copy".
        """
        # 确保后台备份线程已完成, 否则旧文件可能还没落盘
        pending = getattr(self, "_pending_backups", {}).pop(step_id, None)
        if pending is not None:
            pending.join(timeout=30.0)

        snap = self._load(step_id)
        if snap is None:
            logger.warning("patch: snapshot %s not found", step_id)
            return []

        ws = Path(workspace).resolve()
        # 以快照记录的工作区为准; 若调用方给的 workspace 不同, 用快照里的.
        if snap.workspace:
            ws = Path(snap.workspace)
        files_dir = self._files_dir(step_id)
        patterns = tuple(snap.watch_patterns) or _DEFAULT_WATCH_PATTERNS

        current: dict[str, str] = {}
        current_paths: dict[str, Path] = {}
        for rel, fpath in _walk_watch(ws, patterns):
            digest = _hash_file(fpath)
            if digest is None:
                continue
            current[rel] = digest
            current_paths[rel] = fpath

        patches: list[FilePatch] = []

        # 修改 / 删除: 快照里存在过的文件
        for rel, old_hash in snap.files.items():
            new_hash = current.get(rel)
            if new_hash is None:
                patches.append(FilePatch(
                    file_path=rel, change_type="deleted",
                    old_hash=old_hash, new_hash=None,
                    old_content_preview=_read_backup_preview(files_dir, rel),
                    new_content_preview="",
                ))
            elif new_hash != old_hash:
                patches.append(FilePatch(
                    file_path=rel, change_type="modified",
                    old_hash=old_hash, new_hash=new_hash,
                    old_content_preview=_read_backup_preview(files_dir, rel),
                    new_content_preview=_preview_file(current_paths[rel]),
                ))

        # 新建: 当前有, 快照里没有
        for rel, new_hash in current.items():
            if rel not in snap.files:
                patches.append(FilePatch(
                    file_path=rel, change_type="created",
                    old_hash=None, new_hash=new_hash,
                    old_content_preview="",
                    new_content_preview=_preview_file(current_paths[rel]),
                ))

        snap.patches = patches
        self._write_patches(step_id, patches)
        logger.debug("snapshot patch %s: %d changes", step_id, len(patches))
        return patches

    # ---- 回滚 ----

    def revert(self, step_id: str, workspace: Path) -> list[str]:
        """把文件回滚到 step_id 执行前. 返回受影响的相对路径列表.

        先把当前(执行后)状态存成 revert_point (含 present/deleted 标记),
        供 unrevert 完整还原 —— 否则 deleted 的文件 unrevert 时没法重新删掉.
        """
        snap = self._load(step_id)
        if snap is None:
            logger.warning("revert: snapshot %s not found", step_id)
            return []
        if self._is_reverted(step_id):
            logger.warning("revert: snapshot %s already reverted", step_id)
            return []

        ws = Path(snap.workspace or workspace).resolve()
        files_dir = self._files_dir(step_id)
        revert_dir = self._revert_dir(step_id)
        revert_dir.mkdir(parents=True, exist_ok=True)

        # 1) 存执行后状态: 相关文件 = 快照里有的 + patch 里新建的
        relevant: set[str] = set(snap.files.keys())
        for p in snap.patches:
            relevant.add(p.file_path)
        manifest: dict[str, dict] = {}
        for rel in relevant:
            fpath = ws / rel
            if fpath.exists() and fpath.is_file():
                try:
                    shutil.copyfile(fpath, revert_dir / _safe_backup_name(rel))
                    manifest[rel] = {"state": "present"}
                except OSError as exc:
                    logger.debug("revert_point skip %s: %s", rel, exc)
                    manifest[rel] = {"state": "deleted"}
            else:
                manifest[rel] = {"state": "deleted"}
        _write_json(revert_dir / "revert_manifest.json", manifest)

        # 2) 恢复执行前内容
        restored: list[str] = []
        for rel in snap.files:
            backup = files_dir / _safe_backup_name(rel)
            target = ws / rel
            if backup.exists():
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(backup, target)
                    restored.append(rel)
                except OSError as exc:
                    logger.warning("revert restore %s failed: %s", rel, exc)
            else:
                logger.info("revert skip %s: no content backup", rel)

        # 3) 删掉执行前不存在的文件 (patch 里 created 的)
        for p in snap.patches:
            if p.change_type == "created":
                target = ws / p.file_path
                try:
                    target.unlink()
                    restored.append(p.file_path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning("revert delete %s failed: %s", p.file_path, exc)

        self._mark_reverted(step_id, True)
        logger.info("snapshot revert %s: %d paths", step_id, len(restored))
        return restored

    def unrevert(self, step_id: str, workspace: Path) -> list[str]:
        """撤销回滚, 恢复到执行后状态. 返回受影响的相对路径列表."""
        snap = self._load(step_id)
        if snap is None:
            logger.warning("unrevert: snapshot %s not found", step_id)
            return []
        if not self._is_reverted(step_id):
            logger.warning("unrevert: snapshot %s not reverted", step_id)
            return []

        ws = Path(snap.workspace or workspace).resolve()
        revert_dir = self._revert_dir(step_id)
        manifest = _read_json(revert_dir / "revert_manifest.json", {}) or {}
        if not isinstance(manifest, dict):
            manifest = {}

        restored: list[str] = []
        for rel, info in manifest.items():
            if not isinstance(info, dict):
                continue
            target = ws / rel
            if info.get("state") == "present":
                src = revert_dir / _safe_backup_name(rel)
                if src.exists():
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, target)
                        restored.append(rel)
                    except OSError as exc:
                        logger.warning("unrevert restore %s failed: %s", rel, exc)
            else:
                # 执行后本就不存在, 再删掉
                try:
                    target.unlink()
                    restored.append(rel)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning("unrevert delete %s failed: %s", rel, exc)

        self._mark_reverted(step_id, False)
        logger.info("snapshot unrevert %s: %d paths", step_id, len(restored))
        return restored

    # ---- 历史 ----

    def get_history(self, tool_name: str | None = None) -> list[FileSnapshot]:
        """读快照历史, 可按 tool_name 过滤. 按时间升序."""
        snaps: list[FileSnapshot] = []
        for rec in self._load_all():
            if tool_name and rec.get("tool_name") != tool_name:
                continue
            snap = _snapshot_from_record(rec)
            snap.patches = self._load_patches(snap.step_id)
            snap.reverted = self._is_reverted(snap.step_id)
            snaps.append(snap)
        snaps.sort(key=lambda s: s.timestamp)
        return snaps

    # ---- 持久化内部方法 ----

    def _append_log(self, snap: FileSnapshot) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        # jsonl 只存 track 时的元数据; patches / reverted 各存各的文件,
        # 这样日志保持只追加, 不用在 patch/revert 时重写历史行.
        rec = {
            "step_id": snap.step_id,
            "tool_name": snap.tool_name,
            "timestamp": snap.timestamp,
            "files": snap.files,
            "workspace": snap.workspace,
            "watch_patterns": snap.watch_patterns,
        }
        with self._log_file().open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _load_all(self) -> list[dict]:
        log = self._log_file()
        if not log.exists():
            return []
        out: list[dict] = []
        with log.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        out.append(rec)
                except json.JSONDecodeError:
                    continue
        return out

    def _load(self, step_id: str) -> FileSnapshot | None:
        for rec in self._load_all():
            if rec.get("step_id") == step_id:
                snap = _snapshot_from_record(rec)
                snap.patches = self._load_patches(step_id)
                snap.reverted = self._is_reverted(step_id)
                return snap
        return None

    def _write_patches(self, step_id: str, patches: list[FilePatch]) -> None:
        rec = [
            {
                "file_path": p.file_path,
                "change_type": p.change_type,
                "old_hash": p.old_hash,
                "new_hash": p.new_hash,
                "old_content_preview": p.old_content_preview,
                "new_content_preview": p.new_content_preview,
            }
            for p in patches
        ]
        _write_json(self._patches_file(step_id), rec)

    def _load_patches(self, step_id: str) -> list[FilePatch]:
        rec = _read_json(self._patches_file(step_id), [])
        if not isinstance(rec, list):
            return []
        out: list[FilePatch] = []
        for item in rec:
            if not isinstance(item, dict):
                continue
            out.append(FilePatch(
                file_path=item.get("file_path", ""),
                change_type=item.get("change_type", ""),
                old_hash=item.get("old_hash"),
                new_hash=item.get("new_hash"),
                old_content_preview=item.get("old_content_preview", ""),
                new_content_preview=item.get("new_content_preview", ""),
            ))
        return out

    def _is_reverted(self, step_id: str) -> bool:
        rec = _read_json(self._reverted_file(step_id), {})
        if isinstance(rec, dict):
            return bool(rec.get("reverted"))
        return False

    def _mark_reverted(self, step_id: str, reverted: bool) -> None:
        _write_json(self._reverted_file(step_id), {"reverted": reverted})

    def _enforce_cap(self) -> None:
        """FIFO: 超过 _MAX_SNAPSHOTS 就删最老的, 目录和日志一起清."""
        recs = self._load_all()
        if len(recs) <= _MAX_SNAPSHOTS:
            return
        recs.sort(key=lambda r: r.get("timestamp", 0.0))
        keep = recs[len(recs) - _MAX_SNAPSHOTS:]
        keep_ids = {r["step_id"] for r in keep}
        for r in recs:
            if r["step_id"] not in keep_ids:
                # 目录删除放后台 daemon 线程: 这台机器上 AV 扫刚写的文件会让
                # shutil.rmtree 单步卡几十秒, 同步删会把 track() 拖死.
                # 日志压缩 (重写 jsonl) 仍同步做, 那部分只写文件很快.
                # ponytail: 留下少量孤儿目录由后续清理; 升级路径换 SQLite + 定时 GC.
                d = self._snap_dir(r["step_id"])
                threading.Thread(
                    target=shutil.rmtree, args=(d,), kwargs={"ignore_errors": True},
                    daemon=True,
                ).start()
        # ponytail: 只追加日志的代价是 cap 满后必须重写一次. 升级路径换 SQLite.
        try:
            log = self._log_file()
            tmp = log.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in keep:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(log)
        except OSError as exc:
            logger.warning("snapshot log compaction failed: %s", exc)
