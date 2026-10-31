"""文档断言: 首行标记 + 生成字节比对 + env 键交叉检查.

`docs/*.md` 的治理缺口是「文档写了但没人验证」. 本脚本给四个目标文档建立
可执行断言 (spec C):

1. 首行必须是三类标记之一:
   - ``<!-- generated: <cmd> -->`` — 生成物, 测试重跑 ``<cmd>`` 并做字节比对;
   - ``<!-- manual: true -->``    — 人工维护, 只做 env 键交叉检查;
   - 无标记 → FAIL (要么补标记, 要么删文件).
2. 文档中出现的 ``HUGINN_*`` 必须已在 :mod:`huginn.env_schema` 的 ``SCHEMA``
   登记 (拦截"文档承诺了不存在的配置键").
3. ``manual`` 文档同样过 env 键交叉检查.

用法::

    python scripts/doc_assertions.py            # 断言四个目标文档
    python scripts/doc_assertions.py --regen    # 额外重跑 generated 命令并比对字节
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

GENERATED_RE = re.compile(r"^<!-- generated: (?P<cmd>.+) -->$")
MANUAL_RE = re.compile(r"^<!-- manual: true -->$")
ENV_KEY_RE = re.compile(r"\bHUGINN_[A-Z0-9_]+")

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 本批次覆盖的目标文档 (其余按批次扩展).
TARGET_DOCS: tuple[str, ...] = (
    "docs/env-contract.md",
    "docs/feature-flags-contract.md",
    "docs/mece-audit.md",
    "docs/INDEX.md",
)


def classify(text: str) -> tuple[str, str]:
    """按首行分类: ``(generated|manual|unmarked, command)``."""
    first = text.splitlines()[0] if text else ""
    match = GENERATED_RE.match(first)
    if match:
        return "generated", match.group("cmd").strip()
    if MANUAL_RE.match(first):
        return "manual", ""
    return "unmarked", ""


def check_markers(docs: list[Path]) -> list[str]:
    """每个文档首行必须是 ``generated`` 或 ``manual``; 无标记即 FAIL."""
    errors: list[str] = []
    for doc in docs:
        kind, _ = classify(doc.read_text(encoding="utf-8"))
        if kind == "unmarked":
            errors.append(
                f"{doc}: 首行缺少标记; 需补 "
                "`<!-- generated: <cmd> -->` 或 `<!-- manual: true -->`, 或删除该文件"
            )
    return errors


def env_keys(text: str) -> set[str]:
    """提取文档里的 ``HUGINN_*`` 键名 (排除 ``HUGINN_`` / 占位符尾部)."""
    return {k for k in ENV_KEY_RE.findall(text) if not k.endswith("_")}


def check_env_keys(docs: list[Path], schema: dict[str, object] | None = None) -> list[str]:
    """文档里出现的 ``HUGINN_*`` 必须全部存在于 env_schema.SCHEMA."""
    if schema is None:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from huginn.env_schema import SCHEMA

        schema = SCHEMA
    errors: list[str] = []
    for doc in docs:
        missing = sorted(env_keys(doc.read_text(encoding="utf-8")) - set(schema))
        if missing:
            errors.append(
                f"{doc}: 引用了 env_schema 未登记的键: {', '.join(missing)}"
            )
    return errors


def compare_generated(doc: Path, rendered: bytes) -> list[str]:
    """把重新生成的结果与落盘字节逐字节比对."""
    if doc.read_bytes() != rendered:
        return [f"{doc}: 生成结果与落盘内容字节不一致 (文档已过期或被人手改)"]
    return []


def regenerate(doc: Path, command: str, cwd: Path = REPO_ROOT) -> bytes:
    """按首行命令重跑到临时文件, 返回生成字节."""
    argv = shlex.split(command)
    if not argv:
        raise ValueError(f"{doc}: generated 标记缺少命令")
    if argv[0] == "python":
        argv[0] = sys.executable
    # 把命令里的 --out <doc> 目标改到临时文件, 不覆盖仓内文档.
    if "--out" not in argv:
        raise ValueError(f"{doc}: generated 命令必须带 `--out <path>`")
    out_index = argv.index("--out") + 1
    if out_index >= len(argv):
        raise ValueError(f"{doc}: `--out` 后缺少路径")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_out = Path(tmp) / doc.name
        argv[out_index] = str(tmp_out)
        subprocess.run(
            argv,
            cwd=str(cwd),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return tmp_out.read_bytes()


def check_docs(docs: list[Path], regen: bool = False) -> list[str]:
    """执行全量断言, 返回错误列表 (空 = 全绿)."""
    errors = check_markers(docs)
    errors.extend(check_env_keys(docs))
    if regen:
        for doc in docs:
            kind, command = classify(doc.read_text(encoding="utf-8"))
            if kind != "generated":
                continue
            try:
                rendered = regenerate(doc, command)
            except (ValueError, subprocess.CalledProcessError) as exc:
                errors.append(f"{doc}: 重跑生成命令失败: {exc}")
                continue
            errors.extend(compare_generated(doc, rendered))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regen",
        action="store_true",
        help="重跑 generated 文档的生成命令并做字节比对 (较慢)",
    )
    args = parser.parse_args(argv)
    docs = [REPO_ROOT / name for name in TARGET_DOCS]
    missing = [str(d) for d in docs if not d.is_file()]
    if missing:
        print("目标文档缺失: " + ", ".join(missing), file=sys.stderr)
        return 1
    errors = check_docs(docs, regen=args.regen)
    for err in errors:
        print(err, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
