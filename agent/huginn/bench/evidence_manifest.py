"""Evidence manifest — SHA-256 审计清单, 让 reproducibility 可验证.

v6 G55: 扫 workspace/outputs/ 算每个文件的 sha256, 写 manifest.json.
provenance_id 关联到 audit_log 的 forward hash chain (已在 audit_log.py 实现).

不引入新依赖 — 只用 hashlib + pathlib + json.
跑法:
    python -m huginn.bench.evidence_manifest <workspace>
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _sha256_of_file(path: Path, chunk: int = 65536) -> str:
    """算文件 sha256, 大文件分块读避免 OOM."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except OSError as e:
        logger.debug("sha256 failed for %s: %s", path, e)
        return ""


def generate_evidence_manifest(workspace: str | Path) -> dict:
    """扫 workspace/outputs/ 算 sha256, 返回 manifest dict.

    返回结构:
        {
          "workspace": "<abs path>",
          "generated_at": "<ISO8601 UTC>",
          "files": [
            {"path": "<rel>", "sha256": "<hex>", "size": int,
             "provenance_id": "<from .huginn/provenance.jsonl if matchable>"}
          ],
          "manifest_sha256": "<sha256 of files[] sorted by path>"
        }

    ponytail: 不写文件, 只返回 dict; 调用方决定落盘到哪.
    ceiling: 单线程扫, 大 outputs/ 目录慢; 升级路径: 并行 sha256.
    """
    ws = Path(workspace).resolve()
    out_dir = ws / "outputs"
    files: list[dict] = []
    if out_dir.exists() and out_dir.is_dir():
        for p in sorted(out_dir.rglob("*")):
            if not p.is_file():
                continue
            sha = _sha256_of_file(p)
            if not sha:
                continue
            files.append({
                "path": str(p.relative_to(ws)),
                "sha256": sha,
                "size": p.stat().st_size,
                "provenance_id": "",  # 调用方可从 audit_log 反查回填
            })

    # manifest 整体 sha256 — 防 files 列表本身被篡改
    files_for_hash = json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8")
    manifest_sha = hashlib.sha256(files_for_hash).hexdigest()

    return {
        "workspace": str(ws),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
        "manifest_sha256": manifest_sha,
    }


# ── self-check ─────────────────────────────────────────────────
# ponytail: 非平凡逻辑留 runnable check. 验证空目录 + 有文件两种场景.

def _selfcheck() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # 空目录 — files=[], manifest_sha256 仍可算
        m1 = generate_evidence_manifest(tmp)
        assert m1["files"] == []
        assert len(m1["manifest_sha256"]) == 64

        # 造一个 outputs/ 文件
        out = Path(tmp) / "outputs"
        out.mkdir()
        (out / "a.txt").write_text("hello", encoding="utf-8")
        (out / "b.csv").write_text("x,y\n1,2\n", encoding="utf-8")
        m2 = generate_evidence_manifest(tmp)
        assert len(m2["files"]) == 2
        # sha256 长度 64
        assert all(len(f["sha256"]) == 64 for f in m2["files"])
        # 同输入同输出 (确定性)
        m3 = generate_evidence_manifest(tmp)
        assert m2["manifest_sha256"] == m3["manifest_sha256"]
        # a.txt 的 sha256 已知
        assert m2["files"][0]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    print("evidence_manifest selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
