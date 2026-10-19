"""Lean 供应链完整性守卫 — 离线, 只读, 无副作用.

``docs/threat_model.md`` §6 红/蓝覆盖矩阵中 "Lean package supply chain" 一行
此前红蓝双缺口 (仅靠 ``lake-manifest.json`` 提交纪律). 本模块把两条纪律落成
可执行的不变量:

1. **不健全证明标记** (unsound proof marker): ``sorry`` / ``admit`` / ``axiom``
   让 Lean 类型检查器接受一个"证明", 但定理实际未被证明 —— 类型检查通过
   ≠ 命题成立. 一个被污染的证明可借此伪装成已证. §4.3 事故响应本就要求
   "audit all recent proof additions for suspicious tactics (e.g. sorry, axiom)",
   这里把它变成机器可判定的检查.
2. **依赖固定** (dependency pinning): ``lake-manifest.json`` 声明的每个上游
   依赖必须钉到不可变的 commit sha; 浮动 ref (``master`` / ``HEAD`` / 短串)
   让上游可换实现, 是供应链注入入口. 缺失或非 sha 的 ``rev`` 一律判为未固定.

纯离线: 只做字符串/AST 无关的源码扫描与 manifest 结构校验, 不调用 ``lake``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Lean 的不健全占位符: 让 type-checker 通过但证明留空.
# 与 conjecture_library 的 Python proof_script sorry 扫描 (sympy 替代) 对应,
# 但这里扫的是真实 Lean 源码.
UNSOUND_MARKERS: tuple[str, ...] = ("sorry", "admit", "axiom")

# 依赖 rev 必须是不可变 commit sha: 全 hex 且长度 >= 40 (sha-1) / 64 (sha-256).
_REV_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_MIN_REV_LEN = 40

# 浮动 ref 的字面量 (即使看起来像 tag/分支名也不算不可变).
_FLOATING_REFS = frozenset(
    {"master", "main", "head", "develop", "trunk", "latest", "stable"}
)

_BLOCK_COMMENT_RE = re.compile(r"/-.*?-/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")


def _strip_comments(source: str) -> str:
    """去掉 Lean 行注释 ``--`` 与块注释 ``/- ... -/``.

    注释里的 ``sorry`` 只是文字说明, 不是不健全证明, 不能误报.
    """
    without_block = _BLOCK_COMMENT_RE.sub(" ", source)
    return _LINE_COMMENT_RE.sub(" ", without_block)


def scan_unsound_tactics(source: str) -> list[str]:
    """扫描 Lean 源码里的不健全证明标记, 返回命中的 marker 列表 (去重, 保序).

    只匹配整词 (``sorry`` 命中, ``sorryish`` 不命中), 且忽略注释中的出现.
    """
    text = _strip_comments(source)
    hits: list[str] = []
    for marker in UNSOUND_MARKERS:
        if re.search(rf"\b{re.escape(marker)}\b", text):
            hits.append(marker)
    return hits


def _rev_is_pinned(rev: object) -> bool:
    """rev 是否为不可变 commit sha (全 hex, 长度达标, 非浮动名)."""
    if not isinstance(rev, str):
        return False
    rev = rev.strip()
    if len(rev) < _MIN_REV_LEN:
        return False
    if not _REV_HEX_RE.match(rev):
        return False
    return rev.lower() not in _FLOATING_REFS


def check_dependency_pins(packages: object) -> list[str]:
    """校验 manifest ``packages`` 列表里每个依赖都被钉到不可变 rev.

    返回未固定项的说明列表 (空 = 全部已固定). packages 非 list 时判为无效.
    """
    if not isinstance(packages, list):
        return [f"packages 字段不是列表 (实际 {type(packages).__name__})"]

    findings: list[str] = []
    for i, pkg in enumerate(packages):
        if not isinstance(pkg, dict):
            findings.append(f"packages[{i}] 不是对象")
            continue
        name = pkg.get("name") or pkg.get("url") or f"packages[{i}]"
        rev = pkg.get("rev")
        if not _rev_is_pinned(rev):
            findings.append(
                f"依赖 '{name}' 未钉到不可变 commit sha (rev={rev!r})",
            )
    return findings


def audit_manifest(manifest_path: str | Path) -> list[str]:
    """读 lake-manifest.json, 返回未固定依赖说明列表 (空 = 通过).

    文件缺失或非法 JSON → 视为一项未固定 (无法审计 ≠ 安全).
    """
    path = Path(manifest_path)
    if not path.exists():
        return [f"manifest 缺失: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return [f"manifest 不可解析: {path}: {e}"]
    if not isinstance(data, dict):
        return [f"manifest 顶层不是对象: {path}"]
    return check_dependency_pins(data.get("packages", []))


def audit_project(project_path: str | Path) -> dict:
    """审计一个 Lean 项目的完整性.

    返回:
      {
        "unsound": [{"file": str, "markers": [str]}],   # 含不健全标记的源文件
        "unpinned": [str],                               # 未固定依赖说明
        "ok": bool,
      }
    """
    root = Path(project_path)
    unsound: list[dict] = []
    for src in sorted(root.rglob("*.lean")):
        try:
            text = src.read_text(encoding="utf-8")
        except OSError:
            continue
        markers = scan_unsound_tactics(text)
        if markers:
            unsound.append({"file": str(src.relative_to(root)), "markers": markers})

    unpinned = audit_manifest(root / "lake-manifest.json")
    return {"unsound": unsound, "unpinned": unpinned, "ok": not unsound and not unpinned}


__all__ = [
    "UNSOUND_MARKERS",
    "scan_unsound_tactics",
    "check_dependency_pins",
    "audit_manifest",
    "audit_project",
]
