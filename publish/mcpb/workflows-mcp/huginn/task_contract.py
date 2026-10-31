"""机器可读的任务契约 (``contract.json``).

契约由任务下发方在 workspace 根目录提供, schema ``huginn.task-contract/v1``。
它**取代**从 objective 散文里正则抽取交付物/键名/假设标签的做法: 同一份契约
字节 → 同一判定结果, 与题面措辞无关。加载即校验, 缺字段/坏类型一律抛
:class:`ContractError`, 不做猜测。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "CONTRACT_FILENAME",
    "CONTRACT_SCHEMA",
    "ContractError",
    "HypothesisSpec",
    "LoopLogSpec",
    "RequiredKey",
    "StaticSpec",
    "TaskContract",
    "check_required_keys",
    "load_contract",
]

CONTRACT_SCHEMA = "huginn.task-contract/v1"
CONTRACT_FILENAME = "contract.json"

_TOP_LEVEL = frozenset(
    {"schema", "required_files", "required_keys", "hypotheses", "loop_log", "static"}
)
_REQUIRED_TOP_LEVEL = ("schema", "required_files", "required_keys", "hypotheses", "loop_log", "static")
_KEY_TYPES = frozenset({"number", "integer", "string", "boolean", "object", "array", "any"})
_KEY_ENTRY_FIELDS = frozenset({"key", "type", "file"})
_DEFAULT_KEY_FILE = "results.json"


class ContractError(ValueError):
    """``contract.json`` 缺失、格式错误或字段不自洽."""


@dataclass(frozen=True)
class RequiredKey:
    """``results.json`` (或 ``file`` 指定文件) 里必须存在的键及其类型."""

    key: str
    type: str = "any"
    file: str = _DEFAULT_KEY_FILE


@dataclass(frozen=True)
class HypothesisSpec:
    """假设标签集合; ``prediction_only`` 是其中尚无推翻条件、只能给预测的项."""

    labels: tuple[str, ...] = ()
    prediction_only: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class LoopLogSpec:
    """循环台账的行契约: 路径、最少行数、每行必须含的字段."""

    path: str
    lines: int = 0
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class StaticSpec:
    """脚本源码层面的静态约束."""

    forbidden_imports: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(frozen=True)
class TaskContract:
    """一份已校验的契约 + 其源字节指纹."""

    schema: str
    required_files: tuple[str, ...]
    required_keys: tuple[RequiredKey, ...]
    hypotheses: HypothesisSpec | None
    loop_log: LoopLogSpec | None
    static: StaticSpec | None
    sha256: str
    source: str

    @property
    def keys_by_file(self) -> dict[str, tuple[RequiredKey, ...]]:
        """按目标文件分组 (门逐文件校验时用)."""
        out: dict[str, list[RequiredKey]] = {}
        for spec in self.required_keys:
            out.setdefault(spec.file, []).append(spec)
        return {k: tuple(v) for k, v in out.items()}


def _fail(msg: str) -> None:
    raise ContractError(msg)


def _rel_path(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field_name} 必须是非空字符串")
    p = value.strip()
    if p.startswith("/") or p.startswith(".."):
        _fail(f"{field_name} 必须是工作区内的相对路径, 收到 {p!r}")
    return p


def _str_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        _fail(f"{field_name} 必须是字符串数组")
    return tuple(value)


def _parse_required_keys(raw: Any) -> tuple[RequiredKey, ...]:
    if not isinstance(raw, list):
        _fail("required_keys 必须是对象数组")
    out: list[RequiredKey] = []
    for i, entry in enumerate(raw):
        where = f"required_keys[{i}]"
        if not isinstance(entry, dict):
            _fail(f"{where} 必须是对象")
        unknown = set(entry) - _KEY_ENTRY_FIELDS
        if unknown:
            _fail(f"{where} 含未知字段 {sorted(unknown)}")
        key = entry.get("key")
        if not isinstance(key, str) or not key.strip():
            _fail(f"{where}.key 必须是非空字符串")
        typ = entry.get("type", "any")
        if typ not in _KEY_TYPES:
            _fail(f"{where}.type 必须是 {sorted(_KEY_TYPES)} 之一, 收到 {typ!r}")
        target = _rel_path(entry.get("file", _DEFAULT_KEY_FILE), f"{where}.file")
        out.append(RequiredKey(key=key.strip(), type=typ, file=target))
    return tuple(out)


def _parse_hypotheses(raw: Any) -> HypothesisSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _fail("hypotheses 必须是对象或 null")
    unknown = set(raw) - {"labels", "prediction_only"}
    if unknown:
        _fail(f"hypotheses 含未知字段 {sorted(unknown)}")
    if "labels" not in raw:
        _fail("hypotheses.labels 缺失")
    labels = _str_list(raw["labels"], "hypotheses.labels")
    only = frozenset(_str_list(raw.get("prediction_only", []), "hypotheses.prediction_only"))
    extra = only - set(labels)
    if extra:
        _fail(f"hypotheses.prediction_only 必须是 labels 的子集, 多出 {sorted(extra)}")
    return HypothesisSpec(labels=labels, prediction_only=only)


def _parse_loop_log(raw: Any) -> LoopLogSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _fail("loop_log 必须是对象或 null")
    unknown = set(raw) - {"path", "lines", "fields"}
    if unknown:
        _fail(f"loop_log 含未知字段 {sorted(unknown)}")
    if "path" not in raw:
        _fail("loop_log.path 缺失")
    path = _rel_path(raw["path"], "loop_log.path")
    lines = raw.get("lines", 0)
    if isinstance(lines, bool) or not isinstance(lines, int) or lines < 0:
        _fail("loop_log.lines 必须是非负整数")
    fields = _str_list(raw.get("fields", []), "loop_log.fields")
    return LoopLogSpec(path=path, lines=lines, fields=fields)


def _parse_static(raw: Any) -> StaticSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _fail("static 必须是对象或 null")
    unknown = set(raw) - {"forbidden_imports", "seed"}
    if unknown:
        _fail(f"static 含未知字段 {sorted(unknown)}")
    imports = _str_list(raw.get("forbidden_imports", []), "static.forbidden_imports")
    seed = raw.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        _fail("static.seed 必须是整数或 null")
    return StaticSpec(forbidden_imports=imports, seed=seed)


def load_contract(path: str | Path) -> TaskContract:
    """读取并校验 ``contract.json``; sha256 是对**原始字节**的指纹."""
    p = Path(path)
    if not p.is_file():
        _fail(f"契约文件不存在: {p}")
    raw = p.read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"契约不是合法 UTF-8 JSON: {exc}")
    if not isinstance(doc, dict):
        _fail("契约顶层必须是对象")

    missing = [f for f in _REQUIRED_TOP_LEVEL if f not in doc]
    if missing:
        _fail(f"契约缺字段: {', '.join(missing)}")
    unknown = set(doc) - _TOP_LEVEL
    if unknown:
        _fail(f"契约含未知字段: {sorted(unknown)}")
    if doc["schema"] != CONTRACT_SCHEMA:
        _fail(f"schema 必须是 {CONTRACT_SCHEMA!r}, 收到 {doc['schema']!r}")

    files = tuple(
        _rel_path(x, f"required_files[{i}]")
        for i, x in enumerate(_require_list(doc["required_files"], "required_files"))
    )
    return TaskContract(
        schema=doc["schema"],
        required_files=files,
        required_keys=_parse_required_keys(doc["required_keys"]),
        hypotheses=_parse_hypotheses(doc["hypotheses"]),
        loop_log=_parse_loop_log(doc["loop_log"]),
        static=_parse_static(doc["static"]),
        sha256=hashlib.sha256(raw).hexdigest(),
        source=str(p),
    )


def _require_list(value: Any, field_name: str) -> list:
    if not isinstance(value, list):
        _fail(f"{field_name} 必须是数组")
    return value


_CHECKERS: dict[str, Any] = {
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "any": lambda v: v is not None,
}


def check_required_keys(flat: Mapping[str, Any], keys: tuple[RequiredKey, ...]) -> list[str]:
    """按契约校验键存在性与类型; 返回问题清单 (空 = 通过).

    ``flat`` 是调用方已摊平一层的 payload (与审计口径一致)。
    """
    problems: list[str] = []
    for spec in keys:
        if spec.key not in flat:
            problems.append(f"{spec.file}:{spec.key} missing")
            continue
        value = flat[spec.key]
        if not _CHECKERS[spec.type](value):
            problems.append(
                f"{spec.file}:{spec.key} 必须是 {spec.type}, 收到 {type(value).__name__}"
            )
    return problems
