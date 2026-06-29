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
