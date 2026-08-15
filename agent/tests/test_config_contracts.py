"""契约文档漂移自检.

对 config_audit 的每个契约模式, 重新构建并渲染, 断言与已提交的 docs/*-contract.md
完全一致. 一旦代码改了 (新增 env / flag / 工具 / 事件 / mode ...), 而契约文档没
重新生成, 本测试即失败, 提醒开发者跑 config_audit 再生成. 这把契约文档从"一次性
产物"变成"可持续承诺".
"""

from __future__ import annotations

import pathlib

from huginn.cli import config_audit as ca

_REPO = pathlib.Path(__file__).resolve().parents[1]
_DOCS = _REPO / "docs"


def _render_flag(flag: str) -> str:
    builder, renderer, _docname = ca.CONTRACT_MODES[flag]
    return renderer(builder())


def _render_env() -> str:
    return ca.render_markdown(ca.build_inventory())


def test_all_contract_docs_are_not_drifted():
    """每个契约模式重建渲染 == 提交的文档, 防文档漂移."""
    cases = {ca.ENV_CONTRACT_NAME: _render_env}
    for flag, (_b, _r, docname) in ca.CONTRACT_MODES.items():
        cases[docname] = lambda f=flag: _render_flag(f)

    failures = []
    for docname, render in cases.items():
        committed = (_DOCS / docname).read_text(encoding="utf-8")
        fresh = render()
        if committed != fresh:
            failures.append(docname)

    assert not failures, (
        "以下契约文档与代码漂移, 请用 `python -m huginn.cli.config_audit "
        "--<mode> --out docs/<name>.md` 重新生成: "
        + ", ".join(failures)
    )


def test_contract_renders_are_deterministic():
    """同一次构建渲染两次结果一致, 保证可再生成 (无顺序漂移)."""
    for flag in ca.CONTRACT_MODES:
        assert _render_flag(flag) == _render_flag(flag), f"{flag} 契约渲染不确定"


def test_contract_docs_referenced_in_index():
    """INDEX.md 应登记每个契约文档, 防止文档不被发现."""
    index = (_DOCS / "INDEX.md").read_text(encoding="utf-8")
    all_docs = [ca.ENV_CONTRACT_NAME] + [v[2] for v in ca.CONTRACT_MODES.values()]
    for docname in all_docs:
        assert docname in index, f"INDEX.md 未登记 {docname}"