"""锁定 P1-2 文档化规模: journal_db 期刊数量 + 文献源覆盖.

这些断言把 README/模块 docstring 中声明的"14 种期刊 + 生命科学源"
变成可执行的回归检查, 防止后续重构悄悄改变规模.
"""

from __future__ import annotations


def test_journal_db_scale_documented_14() -> None:
    """journal_db 应恰好注册 14 种期刊 (中英文 key 各一份)."""
    from huginn.academic.journal_db import JOURNAL_DATABASE, list_journals

    # list_journals 按 id 去重, 得到真实期刊数
    journals = list_journals()
    assert len(journals) == 14, (
        f"journal_db 规模漂移: 期望 14, 实际 {len(journals)}. "
        "若新增/删除期刊, 请同步更新模块 docstring 中的 '14 种主流期刊'."
    )
    # 每本期刊至少注册一个可查 key (中英文名相同者会合并, 故 >= 14)
    assert len(JOURNAL_DATABASE) >= 14


def test_journal_db_required_journals_present() -> None:
    """关键代表期刊必须存在, 覆盖综合/物理/化学/材料/中文. """
    from huginn.academic.journal_db import get_journal

    for name in [
        "nature",
        "science",
        "physical review letters",
        "jacs",
        "advanced materials",
        "acta materialia",
        "物理学报",
        "化学学报",
        "金属学报",
        "无机材料学报",
        "硅酸盐学报",
        "中国科学",
    ]:
        assert get_journal(name) is not None, f"期刊缺失: {name}"


def test_life_science_sources_registered() -> None:
    """文献层应具备生命科学源: PubMed + Europe PMC + DOAJ. """
    import huginn.tools.literature.search_sources as ss

    for fn in [
        "_search_pubmed",
        "_search_europepmc",
        "_search_doaj",
    ]:
        assert hasattr(ss, fn), f"生命科学源缺失: {fn}"


def test_general_and_materials_sources_registered() -> None:
    """通用/材料/晶体/引文源也应注册. """
    import huginn.tools.literature.search_sources as ss

    for fn in [
        "_search_arxiv",
        "_search_s2",
        "_search_crossref",
        "_search_openalex",
        "_search_core",
        "_search_cod",
        "_search_materials_cloud",
        "_search_nomad",
        "_search_materials_project",
        "_opencitations_references",
        "_opencitations_citations",
    ]:
        assert hasattr(ss, fn), f"文献源缺失: {fn}"
