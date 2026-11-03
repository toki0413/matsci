"""研究产物目录迁移层: 新名写入 + 旧名兼容读取.

历史产物在 ``research_outputs/shusheng_<name>/``, 现统一写入
``research_outputs/ai4s_<name>/``:

- ``output_dir(name)``: 返回新目录 (写入目标; 自动创建);
- ``read_entry(name, relative)``: 读路径优先新目录; 新目录没有则回落旧目录,
  并把**文件**一次性复制进新目录 (旧目录原件保留, 不改历史证据);
- 目录型条目 (如 ``taste_kb`` 的嵌入式库) 只回落读取, 不复制 —— 拷贝数据库
  风险高于收益, 由调用方决定何时重建.

本层只管路径, 不替调用方做科学判断.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "research_outputs"
_LEGACY_PREFIX = "shusheng_"
_CURRENT_PREFIX = "ai4s_"


def _base(root: Path | str | None) -> Path:
    return Path(root) if root is not None else OUTPUTS


def output_dir(name: str, root: Path | str | None = None) -> Path:
    """新目录 (写入目标); 不存在则创建."""
    path = _base(root) / f"{_CURRENT_PREFIX}{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def legacy_dir(name: str, root: Path | str | None = None) -> Path:
    """旧目录 (只读兼容); 不保证存在."""
    return _base(root) / f"{_LEGACY_PREFIX}{name}"


def read_entry(
    name: str, relative: str, root: Path | str | None = None
) -> Path:
    """读取条目: 新目录优先, 旧目录兜底.

    - 新目录命中 → 直接返回;
    - 旧目录是**文件** → 复制到新目录后返回新路径 (一次性迁移, 旧件保留);
    - 旧目录是**目录** → 返回旧目录 (只读兼容, 不复制);
    - 两处都没有 → 返回新目录下的目标路径 (调用方可据此写入).
    """
    new = _base(root) / f"{_CURRENT_PREFIX}{name}"
    old = legacy_dir(name, root)
    target = new / relative
    if target.exists():
        return target
    source = old / relative
    if source.exists():
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return target
        return source
    return target
