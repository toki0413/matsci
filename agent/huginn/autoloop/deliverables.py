"""确定性交付物清单 + 验收门.

清单只从 objective 文本解析, agent **不能改写** —— 这是独立于模型的可判定锚点:
执行对象 (objective + 声明交付物) 与验收标准都不再由模型自己生成.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Deliverable", "parse_manifest", "check_manifest"]

_ARTIFACT_RE = re.compile(
    r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:md|json|jsonl|csv|tsv|txt|png|jpg|jpeg|pdf|xlsx|parquet|npz|npy|py)\b"
)
_SECTION_MARKERS = (
    "【交付物】",
    "交付物：",
    "交付物:",
    "必须交付",
    "产出文件",
    "deliverables:",
    "deliverables：",
)
_SOURCE_PREFIXES = ("agent/", "src/", "huginn/", "tests/")
_MIN_BYTES = 32


@dataclass(frozen=True)
class Deliverable:
    path: str
    min_bytes: int = _MIN_BYTES


def _section_text(objective: str) -> str:
    """取「交付物」段文本; 无该段返回空串 (调用方退回全文扫描)."""
    low = objective.lower()
    for marker in _SECTION_MARKERS:
        idx = low.find(marker.lower())
        if idx < 0:
            continue
        tail = objective[idx + len(marker):]
        stop = len(tail)
        for sep in ("\n\n", "【", "\n#"):
            j = tail.find(sep)
            if 0 <= j < stop:
                stop = j
        return tail[:stop]
    return ""


def parse_manifest(objective: str) -> list[Deliverable]:
    """objective → 去重后的交付物清单 (无声明时返回 []，表示不适用).

    `.py` 只在**显式交付物段**里才算验收物: 散文里的 `["python3","run.py"]` 是执行
    示例, 若当交付物会永久判 missing (ws31 实测的同类误判)。
    """
    if not isinstance(objective, str) or not objective.strip():
        return []
    section = _section_text(objective)
    text = section or objective
    seen: set[str] = set()
    out: list[Deliverable] = []
    # 用 finditer 而不是 findall: 需要知道匹配起点, 才能判定它前面是不是 "/".
    # 绝对路径 "/workspace/.../design.md" 的匹配会**跳过前导斜杠**从 "workspace"
    # 开始, findall 拿不到那个 "/", startswith("/") 守卫永远不生效 → 绝对路径被
    # 当成 workspace 下的相对路径去验收, 必然判 missing (ws31 实测).
    for m in _ARTIFACT_RE.finditer(text):
        raw = m.group(0)
        if m.start() > 0 and text[m.start() - 1] == "/":
            continue  # 绝对路径: 引用材料不是交付物
        if not section and raw.endswith(".py"):
            continue  # 无交付物段时, 散文里的脚本名只是执行示例
        path = raw.strip().strip("'\"`，,。；;：:")
        if not path or path.startswith(("/", "..", "./")):
            continue
        if any(path.startswith(p) for p in _SOURCE_PREFIXES):
            continue
        if path in seen:
            continue
        seen.add(path)
        out.append(Deliverable(path))
    return out


def _json_has_substance(payload: Any) -> bool:
    """结构化交付物必须含至少一个非空叶子值.

    `{}` / `[]` / `{"results": []}` 都能解析、也能凑够 32 字节, 但没有任何可核对
    的数值 —— 存在性判据会把它判绿, 即假绿. 这里按递归内容判「有/无实质」.
    """
    if isinstance(payload, dict):
        return any(_json_has_substance(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_json_has_substance(v) for v in payload)
    if isinstance(payload, str):
        return bool(payload.strip())
    return payload is not None


def check_manifest(workspace: Any, manifest: list[Deliverable]) -> dict[str, Any]:
    """三关验收: 存在 / 非空 / 结构化文件可解析 (确定性, 不由 LLM 判定).

    「非空」既指零字节, 也指 JSON 虽可解析但没有实质内容 (空壳) —— 两者都判红.
    """
    root = Path(workspace)
    missing: list[str] = []
    empty: list[str] = []
    invalid: list[str] = []
    for d in manifest:
        p = root / d.path
        if not p.is_file():
            missing.append(d.path)
            continue
        try:
            size = p.stat().st_size
        except OSError:
            missing.append(d.path)
            continue
        if size == 0:
            empty.append(d.path)
            continue
        if p.suffix == ".json":
            # 零字节 → 空; 有内容但解析失败 → 无效; 可解析但无实质 → 空壳(判空).
            try:
                payload = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                # 读失败或 JSONDecodeError(⊂ValueError) → 声明交付物不可验收.
                invalid.append(d.path)
                continue
            if not isinstance(payload, (dict, list)) or not _json_has_substance(payload):
                empty.append(d.path)
                continue
        if size < d.min_bytes:
            empty.append(d.path)
    return {
        "passed": not (missing or empty or invalid),
        "declared": [d.path for d in manifest],
        "missing": missing,
        "empty": empty,
        "invalid": invalid,
        "checked": len(manifest),
    }
