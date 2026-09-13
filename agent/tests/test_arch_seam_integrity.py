"""共享缝防腐蚀边界门禁（发现②治理 · invariant-lock 回归测试）.

锁两条不变量：
  1. `huginn/core_types.py`(最深共享内核) 不得 import 任何 `huginn.*` —— 内核隔离。
  2. `huginn/config.py`(根配置) 不得反向依赖业务/应用层包 —— 依赖方向只允许
     业务 → 基建, 禁止基建 → 业务。

性质：与既有 test_arch_* 同为"现行为已满足、锁住不许退化"的表征/不变量门禁，
零运行时改动；未来若向这两个共享缝偷塞业务代码/反向 import，本测试变红。
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# 基建包(获准被 config 依赖的自研模块); config 依赖它们 OK。
_ALLOWED_INFRA = {
    "huginn.crypto",
    "huginn.checkpointer",
    "huginn.models",
    "huginn.feature_flags",
    "huginn.config_integrity",
    "huginn.utils",
}

# 业务/应用层包: config 不得 import 它们(反向依赖=防线被破)。
_BUSINESS_LAYERS = {
    "huginn.tools",
    "huginn.autoloop",
    "huginn.agent",
    "huginn.agents",
    "huginn.metacog",
    "huginn.workflows",
    "huginn.knowledge",
    "huginn.causal",
    "huginn.memory",
    "huginn.evolution",
    "huginn.perception",
    "huginn.bench",
    "huginn.academic",
    "huginn.execution",
    "huginn.exploration",
    "huginn.coder",
}


def _huginn_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "huginn" or alias.name.startswith("huginn."):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and (
            node.module == "huginn" or node.module.startswith("huginn.")
        ):
            hits.append(node.module)
    return sorted(set(hits))


def test_core_types_kernel_has_no_business_imports() -> None:
    path = REPO_ROOT / "agent" / "huginn" / "core_types.py"
    assert path.is_file(), "core_types.py not found"
    hits = _huginn_imports(path)
    assert not hits, (
        "core_types(最深共享内核) 不得 import 任何 huginn.* —— 反向依赖会把共享内核"
        "绑到业务层/制造循环。请把该 import 移到非 kernel 模块。违规: %s" % hits
    )


def test_config_does_not_import_application_layers() -> None:
    path = REPO_ROOT / "agent" / "huginn" / "config.py"
    assert path.is_file(), "config.py not found"
    hits = [m for m in _huginn_imports(path) if not m.startswith(tuple(_ALLOWED_INFRA))]
    bad = sorted(h for h in hits if h.startswith(tuple(_BUSINESS_LAYERS)))
    assert not bad, (
        "config(根配置) 反向依赖业务/应用层 —— 依赖方向颠倒(基建→业务=防线被破)。"
        "config 只应依赖基建(_ALLOWED_INFRA)。违规: %s" % bad
    )