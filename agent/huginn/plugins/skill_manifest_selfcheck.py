"""Skill manifest 机制自检 — 验证 generate_manifest + diff_manifest.

最小可运行检查: 构造临时 skill 目录, 生成 manifest, 篡改一个文件,
确认 diff 正确报告 changed. 不依赖外部网络.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def _selfcheck() -> int:
    from huginn.plugins.skill_loader import generate_manifest, diff_manifest

    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp)

        # 造两个 SKILL.md
        (skill_dir / "alpha").mkdir()
        (skill_dir / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\nversion: 0.1.0\ndescription: test alpha\n---\nbody\n",
            encoding="utf-8",
        )
        (skill_dir / "beta").mkdir()
        (skill_dir / "beta" / "SKILL.md").write_text(
            "---\nname: beta\nversion: 0.2.0\ndescription: test beta\n---\nbody\n",
            encoding="utf-8",
        )

        # 1. 生成 manifest
        local = generate_manifest(skill_dir)
        assert "skills" in local
        assert set(local["skills"].keys()) == {"alpha", "beta"}, local["skills"].keys()
        assert local["skills"]["alpha"]["version"] == "0.1.0"
        assert len(local["skills"]["alpha"]["sha256"]) == 64  # SHA-256 hex

        # 2. 无变化 → diff 为空
        assert diff_manifest(local, local) == []

        # 3. 篡改 alpha → diff 报告 changed
        remote = json.loads(json.dumps(local))  # deep copy
        remote["skills"]["alpha"]["sha256"] = "0" * 64
        diffs = diff_manifest(local, remote)
        assert len(diffs) == 1
        assert diffs[0]["name"] == "alpha"
        assert diffs[0]["reason"] == "changed"

        # 4. 远端新增 gamma → diff 报告 new
        remote["skills"]["gamma"] = {"version": "0.3.0", "sha256": "f" * 64, "path": "gamma/SKILL.md"}
        diffs = diff_manifest(local, remote)
        assert len(diffs) == 2
        reasons = {d["name"]: d["reason"] for d in diffs}
        assert reasons == {"alpha": "changed", "gamma": "new"}

        # 5. 本地有但远端没有 → 不报 (本地自定义)
        remote_partial = {"skills": {"alpha": local["skills"]["alpha"]}}
        diffs = diff_manifest(local, remote_partial)
        assert diffs == []  # beta 在本地有但远端没有, 不算更新

    print("skill_manifest_selfcheck: 5/5 passed")
    return 0


if __name__ == "__main__":
    sys.exit(_selfcheck())
