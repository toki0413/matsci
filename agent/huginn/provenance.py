"""计算 provenance 快照.

physics AI 可复现性的底线: 每次计算结果带上种子/版本/参数快照,
事后能追溯某个结果是哪个版本的工具、什么参数、什么软件环境跑出来的.

典型用法::

    from huginn.provenance import capture

    snap = capture("vasp_tool", {"action": "relax", "encut": 520}, random_seed=42)
    result["provenance"] = snap.to_dict()

如果在意落盘, 用 save() 写到 .huginn/provenance/ 目录下, 之后 list_snapshots()
能列出来回放.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceSnapshot:
    """单次计算的可复现性元数据.

    input_hash / output_hash 只取 sha256 前 12 位, 够识别又不冗长.
    """

    timestamp: str  # ISO 8601, UTC
    tool_name: str
    tool_version: str
    input_params: dict[str, Any]
    random_seed: int | None = None
    software_versions: dict[str, str] = field(default_factory=dict)
    input_hash: str = ""
    output_hash: str | None = None
    workspace: str = ""
    user: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """转 dict, 方便塞进 ToolResult.data."""
        return asdict(self)

    def to_json(self) -> str:
        """序列化成 JSON 字符串, 保证可读."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProvenanceSnapshot":
        """从 dict 重建快照. 多余字段忽略, 缺字段用默认值."""
        # 只挑我们认识的字段, 别让外部塞的脏数据把构造器搞挂
        known = {
            "timestamp",
            "tool_name",
            "tool_version",
            "input_params",
            "random_seed",
            "software_versions",
            "input_hash",
            "output_hash",
            "workspace",
            "user",
        }
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# 软件版本收集
# ---------------------------------------------------------------------------


# 想跟踪的包列表, 能 import 就查 __version__, 查不到就跳过.
# 不抛错, 缺包不影响 provenance.
_TRACKED_PACKAGES = (
    "pymatgen",
    "numpy",
    "scipy",
    "ase",
    "matplotlib",
    "torch",
    "tensorflow",
    "jax",
    "spglib",
    "mp_api",
)


def _collect_software_versions() -> dict[str, str]:
    """收集关键科学计算库的版本, 缺的跳过."""
    versions: dict[str, str] = {}

    # python 本身也带上, 有时候 3.10 vs 3.11 差别不小
    versions["python"] = sys.version.split()[0]

    import importlib
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    for pkg in _TRACKED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
        except ImportError:
            continue

        # 三级回退: __version__ 属性 -> version 模块/属性 -> importlib.metadata
        ver: str | None = getattr(mod, "__version__", None)
        if ver is None:
            maybe = getattr(mod, "version", None)
            if hasattr(maybe, "__version__"):
                ver = maybe.__version__
            elif isinstance(maybe, str):
                ver = maybe

        if ver is None or ver == "":
            # 兜底: 直接查包元数据, 多数 pip 装的包都有
            try:
                ver = _pkg_version(pkg)
            except PackageNotFoundError:
                continue  # 装了能 import 但没元数据, 算了跳过

        if ver:
            versions[pkg] = str(ver)

    return versions


# ---------------------------------------------------------------------------
# hash 辅助
# ---------------------------------------------------------------------------


def _short_sha256(text: str) -> str:
    """sha256 前 12 位, 用来识别输入/输出."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _hash_input_params(params: dict[str, Any]) -> str:
    """对输入参数做 hash. 用 sort_keys 保证顺序稳定."""
    # default=str 兜底非 JSON 对象 (Path / 数据类等), 不让 dumps 挂
    serialized = json.dumps(params, sort_keys=True, default=str)
    return _short_sha256(serialized)


def _hash_output(output: Any) -> str:
    """对输出做 hash. output 可能不可 JSON 序列化, 用 str() 兜底."""
    try:
        serialized = json.dumps(output, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # 连 default=str 都救不了 (比如带循环引用), 直接转字符串
        serialized = str(output)
    return _short_sha256(serialized)


# ---------------------------------------------------------------------------
# 捕获入口
# ---------------------------------------------------------------------------


def _get_user() -> str:
    """拿当前用户名. USER / USERNAME / USERPROFILE 都试一遍."""
    for var in ("USER", "USERNAME", "USERPROFILE"):
        val = os.environ.get(var)
        if val:
            # USERPROFILE 在 Windows 上是路径, 只取最后一段当用户名
            return Path(val).name if "\\" in val or "/" in val else val
    return "unknown"


def _get_workspace() -> str:
    """当前工作目录, 绝对路径."""
    try:
        return str(Path.cwd())
    except Exception:
        # 极端情况 (cwd 被删了之类), 别让 provenance 挂
        return ""


def _get_tool_version(tool_name: str) -> str:
    """拿工具版本.

    优先从 huginn.tools.<tool_name> 模块读 __version__, 拿不到就退回
    huginn 整体版本, 再不行就标 'unknown'.
    """
    try:
        import huginn

        huginn_ver = getattr(huginn, "__version__", "unknown")
    except Exception:
        huginn_ver = "unknown"

    # 工具自己定义 __version__ 的情况不多, 但留个口子
    try:
        import importlib

        mod = importlib.import_module(f"huginn.tools.{tool_name}")
        tool_ver = getattr(mod, "__version__", None)
        if tool_ver:
            return str(tool_ver)
    except Exception:
        pass

    return huginn_ver


def capture(
    tool_name: str,
    input_params: dict[str, Any],
    random_seed: int | None = None,
    output: Any = None,
) -> ProvenanceSnapshot:
    """捕获一次计算的 provenance 快照.

    自动收集 timestamp / software_versions / hashes / workspace / user.
    output 不传则 output_hash 留 None, 后续有结果再补.

    Args:
        tool_name: 工具名 (跟 HuginnTool.name 对齐, 如 'vasp_tool').
        input_params: 输入参数字典, 会做 JSON 序列化后 hash.
        random_seed: 随机种子, 没用到就 None.
        output: 计算输出, 用来算 output_hash. 不可序列化也行, str() 兜底.

    Returns:
        ProvenanceSnapshot
    """
    # flag 关掉时返回最小空 snapshot, 不收集版本/hash
    try:
        from huginn.feature_flags import FeatureFlags
        if not FeatureFlags.shared().is_enabled("provenance"):
            return ProvenanceSnapshot(
                timestamp="",
                tool_name=tool_name,
                tool_version="",
                input_params={},
            )
    except Exception:
        # flag 层挂了不能带挂业务, 继续走原逻辑
        pass

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    software_versions = _collect_software_versions()
    input_hash = _hash_input_params(input_params)
    output_hash = _hash_output(output) if output is not None else None

    return ProvenanceSnapshot(
        timestamp=timestamp,
        tool_name=tool_name,
        tool_version=_get_tool_version(tool_name),
        input_params=dict(input_params),  # 拷贝一份, 别让外部后续改了影响快照
        random_seed=random_seed,
        software_versions=software_versions,
        input_hash=input_hash,
        output_hash=output_hash,
        workspace=_get_workspace(),
        user=_get_user(),
    )


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------


def save(snapshot: ProvenanceSnapshot, path: str | Path) -> None:
    """把快照写成 JSON 文件. 一般放 .huginn/provenance/{tool}_{ts}.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.to_json(), encoding="utf-8")


def load(path: str | Path) -> ProvenanceSnapshot:
    """从 JSON 文件读快照."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProvenanceSnapshot.from_dict(data)


def list_snapshots(dir: str | Path) -> list[Path]:
    """列出目录下的所有 provenance 快照文件 (按修改时间倒序)."""
    dir = Path(dir)
    if not dir.exists():
        return []
    files = sorted(
        dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files


# ---------------------------------------------------------------------------
# 默认快照目录: <workspace>/.huginn/provenance/
# ---------------------------------------------------------------------------


def default_snapshot_dir(workspace: str | Path | None = None) -> Path:
    """返回默认快照目录. workspace 不传就用 cwd."""
    base = Path(workspace) if workspace else Path.cwd()
    return base / ".huginn" / "provenance"


# ---------------------------------------------------------------------------
# M4 (W4): per-run provenance records + JSONL 持久化 + FAIR crate 导出
#
# ProvenanceSnapshot 是单次 tool-call 级别的; ProvenanceRecord 是 run 级别,
# 把一次 autoloop / campaign 跑的所有 snapshot 串成 tool_chain, 加上 run 级
# 的 inputs/outputs/DOIs/时间戳, 落到 provenance.jsonl (append-only).
# audit.jsonl 是安全审计 (谁在什么时候碰了什么), provenance.jsonl 是科研
# 溯源 (这个结果是怎么一步步算出来的), 两者用途不同不混用.
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceRecord:
    """一次研究 run 的完整溯源记录.

    tool_chain 里每个 ProvenanceSnapshot 对应一次 tool call, 按调用顺序排.
    inputs 是 run 级输入 (文件路径 + hash, 全局参数), outputs 是 run 级产出
    (结果文件 + 关键数值). dois 是引用或产出的 DOI 列表.
    """

    run_id: str
    objective: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    tool_chain: list[dict[str, Any]] = field(default_factory=list)
    timestamps: dict[str, str] = field(default_factory=dict)
    dois: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def add_snapshot(self, snap: ProvenanceSnapshot) -> None:
        """把一次 tool-call 快照追加到 tool_chain."""
        self.tool_chain.append(snap.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "tool_chain": list(self.tool_chain),
            "timestamps": dict(self.timestamps),
            "dois": list(self.dois),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProvenanceRecord":
        known = {
            "run_id", "objective", "inputs", "outputs",
            "tool_chain", "timestamps", "dois", "tags",
        }
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)


def _hash_file(path: str | Path) -> str:
    """算文件 sha256 前 12 位. 文件不存在返回空串."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def capture_run_inputs(files: list[str | Path] | None = None,
                        params: dict[str, Any] | None = None) -> dict[str, Any]:
    """收集 run 级输入: 文件 hash + 参数快照."""
    inputs: dict[str, Any] = {"params": dict(params or {})}
    file_hashes: dict[str, str] = {}
    for f in (files or []):
        fp = Path(f)
        file_hashes[fp.name] = _hash_file(fp)
    inputs["files"] = file_hashes
    return inputs


class ProvenanceLogger:
    """append-only JSONL 持久化, 落到 $HUGINN_CACHE_DIR/provenance.jsonl.

    跟 audit.jsonl 分开: audit 记安全事件 (谁碰了什么), provenance 记科研
    溯源 (结果怎么算出来的). 测试可注入临时 path 隔离.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else self._default_path()

    @staticmethod
    def _default_path() -> Path:
        base = os.environ.get("HUGINN_CACHE_DIR")
        if base:
            return Path(base) / "provenance.jsonl"
        return Path(".huginn") / "provenance.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def log(self, record: ProvenanceRecord) -> None:
        """追加一条记录. 目录不存在自动建."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False, default=str)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read_all(self) -> list[ProvenanceRecord]:
        """读全部记录. 文件不存在或损坏返回空列表."""
        if not self._path.exists():
            return []
        records: list[ProvenanceRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ProvenanceRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        return records

    def read_run(self, run_id: str) -> list[ProvenanceRecord]:
        """读单个 run 的全部记录 (一个 run 可能分多条记)."""
        return [r for r in self.read_all() if r.run_id == run_id]


def export_crate(record: ProvenanceRecord) -> dict[str, Any]:
    """把一条 ProvenanceRecord 导成 ROCrate 兼容的 JSON dict.

    ROCrate (Research Object Crate) 是 FAIR 数据打包规范, 核心是 @context +
    @graph. 这里产出最小可用的 crate: 一个 root entity (本次 run) + 每个
    tool_chain 节点作为一个 entity, inputs/outputs 作为 file entity.
    DOMAP / FAIR-Ware 这类工具能直接吃这个结构.
    """
    # 预计算每个 tool-call 的 @id, root instrument 引用和 tool entity 共用一套.
    # 同名工具多次调用用 _1 / _2 区分, 保证 @id 唯一且引用一致.
    seen_tools: dict[str, int] = {}
    tool_ids: list[str] = []
    for snap in record.tool_chain:
        tname = snap.get("tool_name", "unknown")
        count = seen_tools.get(tname, 0)
        seen_tools[tname] = count + 1
        suffix = f"_{count}" if count > 0 else ""
        tool_ids.append(f"tool:{tname}{suffix}")

    graph: list[dict[str, Any]] = []
    # root: 本次 run
    graph.append({
        "@id": f"run:{record.run_id}",
        "@type": ["CreateAction", "ResearchAction"],
        "name": record.objective or f"Run {record.run_id}",
        "startTime": record.timestamps.get("start", ""),
        "endTime": record.timestamps.get("end", ""),
        "instrument": [{"@id": tid} for tid in tool_ids],
        "object": [
            {"@id": f"input:{k}"}
            for k in (record.inputs.get("files", {}).keys()
                      if isinstance(record.inputs, dict) else [])
        ],
        "result": [
            {"@id": f"output:{k}"}
            for k in (record.outputs.keys()
                      if isinstance(record.outputs, dict) else [])
        ],
    })
    # 每个 tool-call 一个 entity, @id 与 root instrument 引用一致
    for i, snap in enumerate(record.tool_chain):
        tname = snap.get("tool_name", "unknown")
        graph.append({
            "@id": tool_ids[i],
            "@type": "SoftwareApplication",
            "name": tname,
            "softwareVersion": snap.get("tool_version", "unknown"),
            "url": f"huginn://tools/{tname}",
            "position": i,
        })
    # input file entities
    files = record.inputs.get("files", {}) if isinstance(record.inputs, dict) else {}
    for fname, fhash in files.items():
        graph.append({
            "@id": f"input:{fname}",
            "@type": "File",
            "name": fname,
            "sha256": fhash,
        })
    # output entities
    for oname, ovalue in record.outputs.items() if isinstance(record.outputs, dict) else []:
        graph.append({
            "@id": f"output:{oname}",
            "@type": "PropertyValue",
            "name": oname,
            "value": ovalue,
        })
    # DOI entities
    for doi in record.dois:
        graph.append({
            "@id": doi,
            "@type": "ScholarlyArticle",
            "identifier": doi,
        })
    return {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": graph,
    }
