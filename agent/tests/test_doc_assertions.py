"""文档断言守护 (结构完整性修复 C 轨 T9–T10).

四个目标文档必须:
  - 首行带 ``<!-- generated: <cmd> -->`` 或 ``<!-- manual: true -->``;
  - 文档里出现的 ``HUGINN_*`` 全部存在于 ``huginn.env_schema.SCHEMA``;
  - generated 文档的落盘内容与重跑生成命令的输出字节一致.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import doc_assertions as da  # noqa: E402

_TARGETS = [_REPO / name for name in da.TARGET_DOCS]


def test_all_target_docs_have_marker():
    assert da.check_markers(_TARGETS) == []


def test_generated_docs_declare_their_command():
    generated = {
        path.name: da.classify(path.read_text(encoding="utf-8"))
        for path in _TARGETS
    }
    assert generated["env-contract.md"][0] == "generated"
    assert generated["feature-flags-contract.md"][0] == "generated"
    assert generated["mece-audit.md"][0] == "generated"
    assert generated["INDEX.md"][0] == "manual"


def test_env_keys_exist_in_schema():
    assert da.check_env_keys(_TARGETS) == []


def test_fabricated_env_key_fails(tmp_path):
    fake = tmp_path / "fake.md"
    fake.write_text(
        "<!-- manual: true -->\n\n# fake\n\nHUGINN_FAKE_KEY_NOT_IN_SCHEMA\n",
        encoding="utf-8",
    )
    errors = da.check_env_keys([fake])
    assert errors and "HUGINN_FAKE_KEY_NOT_IN_SCHEMA" in errors[0]


def test_unmarked_doc_fails(tmp_path):
    bare = tmp_path / "bare.md"
    bare.write_text("# no marker\n", encoding="utf-8")
    errors = da.check_markers([bare])
    assert errors and "缺少标记" in errors[0]


def test_generated_byte_comparison_detects_tamper(tmp_path):
    source = _REPO / "docs/mece-audit.md"
    tampered = tmp_path / "mece-audit.md"
    tampered.write_bytes(source.read_bytes() + b"\n")
    errors = da.compare_generated(tampered, source.read_bytes())
    assert errors and "字节不一致" in errors[0]


def test_env_contract_doc_is_current():
    doc = _REPO / "docs/env-contract.md"
    kind, command = da.classify(doc.read_text(encoding="utf-8"))
    assert kind == "generated"
    assert da.compare_generated(doc, da.regenerate(doc, command)) == []


def test_feature_flags_doc_is_current():
    doc = _REPO / "docs/feature-flags-contract.md"
    kind, command = da.classify(doc.read_text(encoding="utf-8"))
    assert kind == "generated"
    assert da.compare_generated(doc, da.regenerate(doc, command)) == []


def test_mece_audit_doc_is_current():
    doc = _REPO / "docs/mece-audit.md"
    kind, command = da.classify(doc.read_text(encoding="utf-8"))
    assert kind == "generated"
    assert da.compare_generated(doc, da.regenerate(doc, command)) == []
