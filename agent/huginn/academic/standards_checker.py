"""学术规范检查器 —— 对照期刊规范逐项检查稿件.

StandardsChecker 读取 JournalSpec 中定义的限制值, 对标题/摘要/正文/参考文献/
图表/关键词做精确比对, 返回 CheckResult 列表. 也可按期刊格式生成参考文献条目
和投稿检查清单.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from huginn.academic.journal_db import (
    JOURNAL_DATABASE,
    JournalSpec,
    get_journal,
    get_reference_format,
)


@dataclass
class CheckResult:
    """单次检查的结果."""

    check_name: str           # 检查项名称
    passed: bool              # 是否通过
    message: str              # 结果描述
    suggestion: str = ""      # 修改建议
    severity: str = "error"   # error | warning | info


def _count_words(text: str) -> int:
    """统计英文词数: 按空白分割."""
    return len(text.split())


def _count_chars(text: str) -> int:
    """统计字符数 (不含首尾空白)."""
    return len(text.strip())


def _count_chinese_chars(text: str) -> int:
    """统计中文字符数 (CJK 统一表意区)."""
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


class StandardsChecker:
    """对照期刊规范检查稿件各部分."""

    def __init__(self) -> None:
        self._fmt_builders: dict[str, callable] = {
            "nature": self._fmt_nature,
            "science": self._fmt_science,
            "prl": self._fmt_prl,
            "acs": self._fmt_acs,
            "wiley": self._fmt_wiley,
            "elsevier": self._fmt_elsevier,
            "aps_cn": self._fmt_aps_cn,
            "acs_cn": self._fmt_acs_cn,
        }

    # ── 单项检查 ──────────────────────────────────────────────

    def check_title(self, title: str, journal: str) -> CheckResult:
        """检查标题长度."""
        spec = get_journal(journal)
        if spec is None:
            return CheckResult(
                "title", False,
                f"未找到期刊 '{journal}'",
                "请确认期刊名称拼写正确",
            )

        # 中文期刊用中文字符数, 英文期刊用字符数
        if spec.language.startswith("zh"):
            limit = spec.title_zh_max_chars or spec.title_max_chars
            if limit is None:
                return CheckResult("title", True, "该期刊未定义标题长度限制", severity="info")
            actual = _count_chinese_chars(title)
            label = "中文字符"
        else:
            limit = spec.title_max_chars
            if limit is None:
                return CheckResult("title", True, "该期刊未定义标题长度限制", severity="info")
            actual = _count_chars(title)
            label = "字符"

        if actual <= limit:
            return CheckResult(
                "title", True,
                f"标题长度 {actual} {label}, 上限 {limit}",
            )
        return CheckResult(
            "title", False,
            f"标题超长: {actual} {label}, 上限 {limit}",
            f"删减 {actual - limit} {label}",
        )

    def check_abstract(
        self, abstract: str, journal: str, lang: str = "en"
    ) -> CheckResult:
        """检查摘要长度. lang="en" 按词数, lang="zh" 按字符数."""
        spec = get_journal(journal)
        if spec is None:
            return CheckResult(
                "abstract", False,
                f"未找到期刊 '{journal}'",
                "请确认期刊名称",
            )

        if lang == "zh":
            limit = spec.abstract_zh_max_chars or spec.abstract_max_chars
            if limit is None:
                return CheckResult("abstract", True, "该期刊未定义中文摘要限制", severity="info")
            actual = _count_chinese_chars(abstract)
            unit = "字符"
        else:
            limit = spec.abstract_en_max_words or spec.abstract_max_words
            if limit is None:
                return CheckResult("abstract", True, "该期刊未定义英文摘要限制", severity="info")
            actual = _count_words(abstract)
            unit = "词"

        if actual <= limit:
            return CheckResult(
                "abstract", True,
                f"摘要长度 {actual} {unit}, 上限 {limit}",
            )
        return CheckResult(
            "abstract", False,
            f"摘要超长: {actual} {unit}, 上限 {limit}",
            f"精简至 {limit} {unit}以内",
        )

    def check_word_count(
        self, text: str, journal: str, section: str = "body"
    ) -> CheckResult:
        """检查正文字数. section 可选 "body" / "methods"."""
        spec = get_journal(journal)
        if spec is None:
            return CheckResult(
                f"word_count:{section}", False,
                f"未找到期刊 '{journal}'",
                "请确认期刊名称",
            )

        if section == "methods":
            limit = spec.methods_max_words
        else:
            limit = spec.body_max_words

        if limit is None:
            return CheckResult(
                f"word_count:{section}", True,
                "该期刊未定义此部分的字数限制",
                severity="info",
            )

        actual = _count_words(text)
        if actual <= limit:
            return CheckResult(
                f"word_count:{section}", True,
                f"{section} 字数 {actual}, 上限 {limit}",
            )
        return CheckResult(
            f"word_count:{section}", False,
            f"{section} 超字: {actual} 词, 上限 {limit}",
            f"删减 {actual - limit} 词",
        )

    def check_references(self, refs: list[str], journal: str) -> CheckResult:
        """检查参考文献数量."""
        spec = get_journal(journal)
        if spec is None:
            return CheckResult(
                "references", False,
                f"未找到期刊 '{journal}'",
                "请确认期刊名称",
            )

        count = len(refs)
        issues: list[str] = []

        if spec.references_max and count > spec.references_max:
            issues.append(f"参考文献 {count} 条, 超过上限 {spec.references_max}")

        if spec.references_min and count < spec.references_min:
            issues.append(f"参考文献 {count} 条, 少于下限 {spec.references_min}")

        if not issues:
            limits = []
            if spec.references_max:
                limits.append(f"上限 {spec.references_max}")
            if spec.references_min:
                limits.append(f"下限 {spec.references_min}")
            range_str = ", ".join(limits) if limits else "无数量限制"
            return CheckResult(
                "references", True,
                f"参考文献 {count} 条 ({range_str})",
            )

        return CheckResult(
            "references", False,
            "; ".join(issues),
            "增减参考文献至规定范围内",
        )

    def check_figures(self, figure_count: int, journal: str) -> CheckResult:
        """检查图表数量."""
        spec = get_journal(journal)
        if spec is None:
            return CheckResult(
                "figures", False,
                f"未找到期刊 '{journal}'",
                "请确认期刊名称",
            )

        limit = spec.figures_max or spec.display_items_max
        if limit is None:
            return CheckResult(
                "figures", True,
                "该期刊未定义图表数量限制",
                severity="info",
            )

        if figure_count <= limit:
            return CheckResult(
                "figures", True,
                f"图表 {figure_count} 个, 上限 {limit}",
            )
        return CheckResult(
            "figures", False,
            f"图表过多: {figure_count} 个, 上限 {limit}",
            f"删减至 {limit} 个以内",
        )

    def check_keywords(self, keywords: list[str], journal: str) -> CheckResult:
        """检查关键词数量."""
        spec = get_journal(journal)
        if spec is None:
            return CheckResult(
                "keywords", False,
                f"未找到期刊 '{journal}'",
                "请确认期刊名称",
            )

        if spec.keywords_max is None:
            return CheckResult(
                "keywords", True,
                "该期刊未定义关键词数量限制",
                severity="info",
            )

        count = len(keywords)
        if count <= spec.keywords_max:
            return CheckResult(
                "keywords", True,
                f"关键词 {count} 个, 上限 {spec.keywords_max}",
            )
        return CheckResult(
            "keywords", False,
            f"关键词过多: {count} 个, 上限 {spec.keywords_max}",
            f"删减至 {spec.keywords_max} 个",
        )

    # ── 综合检查 ──────────────────────────────────────────────

    def check_compliance(
        self, manuscript: dict, journal: str
    ) -> list[CheckResult]:
        """综合检查稿件各部分, 返回所有检查结果.

        manuscript 支持的 key:
          title, abstract, abstract_lang, body, methods,
          references (list), figure_count, keywords (list)
        """
        results: list[CheckResult] = []
        spec = get_journal(journal)
        if spec is None:
            results.append(CheckResult(
                "compliance", False,
                f"未找到期刊 '{journal}'",
                "请确认期刊名称",
            ))
            return results

        if "title" in manuscript and manuscript["title"]:
            results.append(self.check_title(manuscript["title"], journal))

        if "abstract" in manuscript and manuscript["abstract"]:
            lang = manuscript.get("abstract_lang", "en")
            results.append(self.check_abstract(
                manuscript["abstract"], journal, lang
            ))

        if "body" in manuscript and manuscript["body"]:
            results.append(self.check_word_count(
                manuscript["body"], journal, "body"
            ))

        if "methods" in manuscript and manuscript["methods"]:
            results.append(self.check_word_count(
                manuscript["methods"], journal, "methods"
            ))

        if "references" in manuscript and manuscript["references"] is not None:
            results.append(self.check_references(
                manuscript["references"], journal
            ))

        if "figure_count" in manuscript and manuscript["figure_count"] is not None:
            results.append(self.check_figures(
                manuscript["figure_count"], journal
            ))

        if "keywords" in manuscript and manuscript["keywords"]:
            results.append(self.check_keywords(
                manuscript["keywords"], journal
            ))

        # 声明类要求检查
        results.extend(self._check_declarations(manuscript, spec))

        return results

    def _check_declarations(
        self, manuscript: dict, spec: JournalSpec
    ) -> list[CheckResult]:
        """检查各类声明是否存在."""
        checks: list[tuple[bool, str, str]] = [
            (spec.requires_data_availability, "data_availability", "数据可用性声明"),
            (spec.requires_code_availability, "code_availability", "代码可用性声明"),
            (spec.requires_reporting_summary, "reporting_summary", "报告摘要"),
            (spec.requires_orcid, "orcid", "ORCID"),
            (spec.requires_bilingual, "bilingual", "中英文双语内容"),
            (spec.requires_copyright_agreement, "copyright_agreement", "版权协议"),
            (spec.requires_classification_number, "classification_number", "中图分类号"),
            (spec.requires_unit_proof, "unit_proof", "单位证明信"),
            (spec.requires_originality, "originality", "原创性声明"),
            (spec.requires_ai_declaration, "ai_declaration", "AI使用声明"),
        ]

        results: list[CheckResult] = []
        for required, key, label in checks:
            if not required:
                continue
            if not manuscript.get(key):
                results.append(CheckResult(
                    key, False,
                    f"缺少{label}",
                    f"请补充{label}",
                    severity="warning",
                ))
            else:
                results.append(CheckResult(
                    key, True,
                    f"已提供{label}",
                    severity="info",
                ))
        return results

    # ── 参考文献格式化 ────────────────────────────────────────

    def format_reference(self, ref_data: dict, journal: str) -> str:
        """按期刊格式生成单条参考文献.

        ref_data 支持的 key:
          authors (list[str]), title, journal, year, volume,
          pages, doi
        """
        spec = get_journal(journal)
        if spec is None:
            return f"[未找到期刊 '{journal}']"

        fmt_key = spec.reference_format or ""
        builder = self._fmt_builders.get(fmt_key)
        if builder is None:
            # 未知格式, 返回原始信息拼接
            return self._fmt_generic(ref_data)
        return builder(ref_data)

    def _fmt_authors_nature(self, authors: list[str]) -> str:
        """Nature 作者格式: 姓在前名缩写, 超5人用 et al."""
        if len(authors) > 5:
            first = authors[0]
            return f"{first} et al."
        if len(authors) == 1:
            return authors[0]
        if len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        return ", ".join(authors[:-1]) + f" & {authors[-1]}"

    def _fmt_nature(self, d: dict) -> str:
        authors = self._fmt_authors_nature(d.get("authors", []))
        parts = [authors]
        if d.get("title"):
            parts.append(d["title"] + ".")
        if d.get("journal"):
            parts.append(d["journal"])
        vol = d.get("volume", "")
        pages = d.get("pages", "")
        year = d.get("year", "")
        vol_pages = f"**{vol}**, {pages}" if vol else pages
        if year:
            vol_pages += f" ({year})"
        parts.append(vol_pages)
        return " ".join(parts) + "."

    def _fmt_science(self, d: dict) -> str:
        authors = d.get("authors", [])
        if len(authors) > 10:
            author_str = ", ".join(authors[:10]) + ", et al."
        else:
            author_str = ", ".join(authors)
        parts = [author_str + ","]
        if d.get("journal"):
            parts.append(d["journal"])
        vol = d.get("volume", "")
        pages = d.get("pages", "")
        year = d.get("year", "")
        vp = f"**{vol}**, {pages}" if vol else pages
        if year:
            vp += f" ({year})"
        parts.append(vp)
        return " ".join(parts) + "."

    def _fmt_prl(self, d: dict) -> str:
        authors = d.get("authors", [])
        if len(authors) > 5:
            author_str = ", ".join(authors[:5]) + ", et al."
        elif len(authors) == 2:
            author_str = f"{authors[0]} and {authors[1]}"
        else:
            author_str = ", ".join(authors)
        parts = [author_str + ","]
        if d.get("journal"):
            parts.append(d["journal"])
        vol = d.get("volume", "")
        pages = d.get("pages", "")
        year = d.get("year", "")
        vp = f"**{vol}**, {pages}" if vol else pages
        if year:
            vp += f" ({year})"
        parts.append(vp)
        return " ".join(parts) + "."

    def _fmt_acs(self, d: dict) -> str:
        authors = d.get("authors", [])
        author_str = "; ".join(authors)
        parts = []
        if author_str:
            parts.append(author_str)
        if d.get("title"):
            parts.append(d["title"] + ".")
        if d.get("journal"):
            parts.append(d["journal"])
        year = d.get("year", "")
        vol = d.get("volume", "")
        pages = d.get("pages", "")
        yvp = ""
        if year:
            yvp += f"**{year}**"
        if vol:
            yvp += f", *{vol}*"
        if pages:
            yvp += f", {pages}"
        if yvp:
            parts.append(yvp)
        return ". ".join(parts) + "."

    def _fmt_wiley(self, d: dict) -> str:
        authors = d.get("authors", [])
        author_str = ", ".join(authors)
        parts = []
        if author_str:
            parts.append(author_str + ",")
        if d.get("journal"):
            parts.append(d["journal"])
        year = d.get("year", "")
        vol = d.get("volume", "")
        pages = d.get("pages", "")
        yvp = f"**{year}**" if year else ""
        if vol:
            yvp += f", {vol}"
        if pages:
            yvp += f", {pages}"
        if yvp:
            parts.append(yvp)
        return " ".join(parts) + "."

    def _fmt_elsevier(self, d: dict) -> str:
        authors = d.get("authors", [])
        author_str = ", ".join(authors)
        parts = []
        if author_str:
            parts.append(author_str + ",")
        if d.get("title"):
            parts.append(d["title"] + ",")
        if d.get("journal"):
            parts.append(d["journal"])
        vol = d.get("volume", "")
        pages = d.get("pages", "")
        year = d.get("year", "")
        vp = f"**{vol}**" if vol else ""
        if year:
            vp += f" ({year})"
        if pages:
            vp += f" {pages}"
        if vp:
            parts.append(vp)
        return " ".join(parts) + "."

    def _fmt_aps_cn(self, d: dict) -> str:
        """中文期刊 APS 变体: 作者. 年份. 刊名 卷号 起始页码."""
        authors = d.get("authors", [])
        author_str = ", ".join(authors)
        parts = []
        if author_str:
            parts.append(author_str + ".")
        if d.get("year"):
            parts.append(str(d["year"]) + ".")
        if d.get("journal"):
            parts.append(d["journal"])
        vol = d.get("volume", "")
        pages = d.get("pages", "")
        if vol:
            parts.append(str(vol))
        if pages:
            # 取起始页码
            start = pages.split("-")[0] if "-" in pages else pages
            parts.append(f": {start}")
        return " ".join(parts) + "."

    def _fmt_acs_cn(self, d: dict) -> str:
        """中文期刊 ACS 变体: 作者. 标题. 刊名, 年份, 卷号, 起止页码."""
        authors = d.get("authors", [])
        author_str = ", ".join(authors)
        parts = []
        if author_str:
            parts.append(author_str + ".")
        if d.get("title"):
            parts.append(d["title"] + ".")
        if d.get("journal"):
            parts.append(d["journal"] + ",")
        comps = []
        if d.get("year"):
            comps.append(str(d["year"]))
        if d.get("volume"):
            comps.append(str(d["volume"]))
        if d.get("pages"):
            comps.append(str(d["pages"]))
        if comps:
            parts.append(", ".join(comps))
        return " ".join(parts) + "."

    def _fmt_generic(self, d: dict) -> str:
        """未知格式的兜底: 按常见字段顺序拼接."""
        authors = d.get("authors", [])
        parts = []
        if authors:
            parts.append(", ".join(authors) + ".")
        if d.get("title"):
            parts.append(d["title"] + ".")
        if d.get("journal"):
            parts.append(d["journal"] + ",")
        comps = []
        if d.get("year"):
            comps.append(str(d["year"]))
        if d.get("volume"):
            comps.append(str(d["volume"]))
        if d.get("pages"):
            comps.append(str(d["pages"]))
        if comps:
            parts.append(", ".join(comps))
        if d.get("doi"):
            parts.append(f"DOI: {d['doi']}")
        return " ".join(parts)

    # ── 投稿检查清单 ──────────────────────────────────────────

    def generate_submission_checklist(self, journal: str) -> list[str]:
        """生成投稿检查清单.

        优先使用 JournalSpec.submission_checklist; 没有则根据字段自动生成.
        """
        spec = get_journal(journal)
        if spec is None:
            return [f"未找到期刊 '{journal}'"]

        checklist = list(spec.submission_checklist)

        # 没有预定义清单时根据字段生成
        if not checklist:
            checklist = self._auto_checklist(spec)

        # 追加通用项
        if spec.ai_policy:
            checklist.append(f"AI政策: {spec.ai_policy}")
        if spec.ai_image_policy:
            checklist.append(f"图片政策: {spec.ai_image_policy}")
        if spec.plagiarism_check:
            checklist.append(f"查重要求: {spec.plagiarism_check}")
        if spec.file_size_max_mb:
            checklist.append(f"文件大小: 不超过 {spec.file_size_max_mb}MB")

        return checklist

    def _auto_checklist(self, spec: JournalSpec) -> list[str]:
        """根据 JournalSpec 字段自动生成基础检查清单."""
        items: list[str] = []
        if spec.title_max_chars:
            items.append(f"标题不超过 {spec.title_max_chars} 字符")
        if spec.abstract_max_words:
            items.append(f"摘要不超过 {spec.abstract_max_words} 词")
        if spec.body_max_words:
            items.append(f"正文不超过 {spec.body_max_words} 词")
        if spec.methods_max_words:
            items.append(f"Methods 不超过 {spec.methods_max_words} 词")
        if spec.display_items_max:
            items.append(f"图表不超过 {spec.display_items_max} 个")
        if spec.figures_max:
            items.append(f"图不超过 {spec.figures_max} 个")
        if spec.references_max:
            items.append(f"参考文献不超过 {spec.references_max} 条")
        if spec.keywords_max:
            items.append(f"关键词不超过 {spec.keywords_max} 个")
        if spec.requires_toc_graphic:
            items.append(f"提供TOC图形{f' ({spec.toc_size})' if spec.toc_size else ''}")
        if spec.requires_graphical_abstract:
            items.append("提供图文摘要")
        if spec.requires_bilingual:
            items.append("中英文标题/摘要/关键词齐全")
        if spec.requires_data_availability:
            items.append("提供数据可用性声明")
        if spec.requires_code_availability:
            items.append("提供代码可用性声明")
        if spec.requires_orcid:
            items.append("全部作者提供ORCID")
        if spec.requires_reporting_summary:
            items.append("提供报告摘要 (Reporting Summary)")
        if spec.requires_revtex:
            items.append("使用REVTeX模板排版")
        if spec.requires_copyright_agreement:
            items.append("签署版权转让协议")
        if spec.requires_classification_number:
            items.append("标注中图分类号")
        if spec.requires_unit_proof:
            items.append("附单位证明信")
        if spec.requires_originality:
            items.append("签署原创性声明")
        if spec.requires_ai_declaration:
            items.append("签署AI使用声明")
        if spec.requires_ccdc:
            items.append("晶体数据提交CCDC")
        if spec.requires_all_authors:
            items.append("列出全部作者信息")
        if spec.new_compound_characterization:
            items.append(
                "新化合物表征: " + "/".join(spec.new_compound_characterization)
            )
        return items
