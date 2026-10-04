"""publish/mcpb 打包镜像的漂移棘轮 (only-shrink).

``publish/mcpb/{capabilities,workflows}-mcp/`` 各自内嵌一份 ``huginn`` 源码副本.
仓库无构建脚本、CI 也不打 bundle —— 镜像**手工维护**, 曾严重漂移 (每个镜像:
缺 71 / 多 3 / 内容不同 263). 已跑 ``scripts/mcpb_vendor.py --write`` 收敛至 0,
现为**零漂移**硬门禁.

本测试把漂移**冻结为只减不增的预算**: 任何让镜像更落后的改动都会变红, 逼作者
要么跑 ``scripts/mcpb_vendor.py --write`` 收敛, 要么显式上调预算 (可见的发布决策).

同步入口: ``python scripts/mcpb_vendor.py --write``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mcpb_vendor  # noqa: E402

# only-shrink 预算 (已跑 scripts/mcpb_vendor.py --write 收敛至 0). 不得上调.
MAX_MISSING = 0     # 源里有、镜像缺
MAX_EXTRA = 0       # 镜像里有、源里无 (陈旧自检脚本)
MAX_DIFFERING = 0   # 两边都有但内容不同


def test_publish_mirror_drift_only_shrinks():
    """不变量: 打包镜像相对 agent/huginn 的漂移只减不增."""
    for bundle in mcpb_vendor.BUNDLES:
        rep = mcpb_vendor.drift_report(bundle)
        assert len(rep.missing) <= MAX_MISSING, (
            f"[{bundle}] 镜像缺失文件从 {MAX_MISSING} 涨到 {len(rep.missing)} —— "
            f"agent/huginn 改了但镜像没跟. 跑 scripts/mcpb_vendor.py --write 收敛. "
            f"示例: {rep.missing[:8]}"
        )
        assert len(rep.extra) <= MAX_EXTRA, (
            f"[{bundle}] 镜像多出的陈旧文件从 {MAX_EXTRA} 涨到 {len(rep.extra)}. "
            f"示例: {rep.extra[:8]}"
        )
        assert len(rep.differing) <= MAX_DIFFERING, (
            f"[{bundle}] 镜像与源内容不同的文件从 {MAX_DIFFERING} 涨到 "
            f"{len(rep.differing)} —— 镜像在继续腐烂. 示例: {rep.differing[:8]}"
        )


def test_hash_tree_primitive(tmp_path: Path):
    """自检: _hash_tree 计数正确, 且内容变化能被检出 (防门禁自己失效)."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00")
    h = mcpb_vendor._hash_tree(tmp_path)
    assert set(h) == {"a.py", "sub/b.py"}, f"应跳过 __pycache__: {sorted(h)}"

    before = h["a.py"]
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert mcpb_vendor._hash_tree(tmp_path)["a.py"] != before, "内容变化未被检出"