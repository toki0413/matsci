"""学术期刊规范数据库 —— 14 种主流期刊的投稿规范结构化描述.

每本期刊用一个 JournalSpec 封装标题/摘要/正文/参考文献/图表/字体等全部限制,
供 StandardsChecker 和 PaperTool 调用. 数据来自各期刊官方投稿指南.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JournalSpec:
    """单本期刊的完整投稿规范."""

    # ── 基本信息 ──
    name: str                       # 期刊名 (中文或常用名)
    name_en: str                    # 英文名
    publisher: str                  # 出版方
    impact_factor: str              # 影响因子 (约值)
    language: str                   # 语言代码, 如 "en" / "zh-CN"
    article_types: list[str]        # 接收的文章类型
    field: str                      # 学科领域

    # ── 标题限制 ──
    title_max_chars: int | None = None       # 标题最大字符数
    title_zh_max_chars: int | None = None    # 中文标题最大字符数
    title_en_max_words: int | None = None    # 英文标题最大词数

    # ── 摘要限制 ──
    abstract_max_words: int | None = None    # 摘要最大词数 (英文)
    abstract_max_chars: int | None = None    # 摘要最大字符数
    abstract_zh_max_chars: int | None = None # 中文摘要最大字符数
    abstract_en_max_words: int | None = None # 英文摘要最大词数

    # ── 正文限制 ──
    body_max_words: int | None = None        # 正文最大词数
    methods_max_words: int | None = None     # Methods 部分最大词数
    pages_max: int | None = None             # 最大页数
    max_double_spaced_pages: int | None = None  # 双倍行距最大页数

    # ── 图表限制 ──
    display_items_max: int | None = None     # 图表总数上限
    figures_max: int | None = None           # 图数量上限
    figure_width_single: str | None = None   # 单栏图宽度
    figure_width_double: str | None = None   # 双栏图宽度
    figure_min_font_pt: int | None = None    # 图内最小字号
    figure_max_font_pt: int | None = None    # 图内最大字号
    figure_panel_label: str | None = None    # 分图标签格式
    figure_dpi: int | None = None            # 图分辨率 DPI
    figure_color_mode: str | None = None     # 色彩模式

    # ── 参考文献 ──
    references_max: int | None = None        # 参考文献最大数
    references_min: int | None = None        # 参考文献最小数
    reference_format: str | None = None      # 参考文献格式标识

    # ── 关键词 ──
    keywords_max: int | None = None          # 关键词最大数

    # ── 作者 ──
    authors_max: int | None = None           # 作者最大数
    requires_all_authors: bool = False          # 是否需列出全部作者
    requires_orcid: bool = False                # 是否要求 ORCID

    # ── 格式要求 ──
    font: str | None = None                  # 正文字体
    line_spacing: str | None = None          # 行距
    requires_revtex: bool = False               # 是否要求 REVTeX
    requires_toc_graphic: bool = False          # 是否要求 TOC 图
    toc_size: str | None = None              # TOC 图尺寸
    requires_graphical_abstract: bool = False   # 是否要求图文摘要
    requires_bilingual: bool = False            # 是否要求中英文双语
    requires_data_availability: bool = False    # 数据可用性声明
    requires_code_availability: bool = False    # 代码可用性声明
    requires_reporting_summary: bool = False    # 报告摘要
    requires_copyright_agreement: bool = False  # 版权协议
    requires_classification_number: bool = False  # 分类号
    requires_unit_proof: bool = False           # 单位证明信
    requires_originality: bool = False          # 原创性声明
    requires_ai_declaration: bool = False       # AI 使用声明
    requires_ccdc: bool = False                 # CCDC 晶体数据

    # ── AI / 预印本政策 ──
    allows_preprints: bool = False              # 是否允许预印本
    ai_policy: str | None = None             # AI 使用政策
    ai_image_policy: str | None = None       # AI 图片政策

    # ── 审稿 ──
    review_criteria: list[str] = field(default_factory=list)      # 审稿标准
    review_max: list[str] = field(default_factory=list)           # 各类型字数上限
    revision_deadline_days: int | None = None  # 修回期限 (天)

    # ── 特殊要求 ──
    special_requirements: list[str] = field(default_factory=list)
    special: list[str] = field(default_factory=list)
    submission_checklist: list[str] = field(default_factory=list)
    new_compound_characterization: list[str] = field(default_factory=list)  # 新化合物表征要求

    # ── 查重 / 文件 ──
    plagiarism_check: str | None = None      # 查重要求
    file_size_max_mb: int | None = None      # 文件大小上限

    # ── 系列 (中国科学) ──
    has_series: list[str] | None = None


# ──────────────────────────────────────────────────────────────────────
# 参考文献格式说明
# ──────────────────────────────────────────────────────────────────────

REFERENCE_FORMATS: dict[str, str] = {
    "nature": (
        "Nature 格式: 正文用上标编号引用; 列表按正文顺序排列. "
        "作者姓在前名缩写 (Smith, J. A.), 超过5位作者用 et al.; "
        "包含文章标题; 期刊名斜体缩写; 卷号加粗; 起止页码. "
        "示例: Smith, J. A. & Jones, B. C. Title of article. "
        "J. Am. Chem. Soc. **145**, 1234-1245 (2023)."
    ),
    "science": (
        "Science 格式: 正文用上标编号; 列表按引用顺序. "
        "作者 姓 名缩写, 最多10位后用 et al.; 期刊名缩写斜体; "
        "卷号加粗; 页码. 示例: J. A. Smith, B. C. Jones, "
        "Science **380**, 1234 (2023)."
    ),
    "prl": (
        "PRL (APS) 格式: REVTeX 模板; 正文用方括号编号 [1]. "
        "作者 名缩写 姓, 用逗号分隔; 期刊名缩写斜体; 卷号加粗. "
        "示例: J. A. Smith and B. C. Jones, Phys. Rev. Lett. "
        "**131**, 123401 (2023)."
    ),
    "acs": (
        "ACS 格式: 正文用上标编号; 列表按引用顺序. "
        "作者 名缩写 姓; 期刊名缩写斜体; 年份加粗; 卷号斜体; 页码. "
        "示例: Smith, J. A.; Jones, B. C. Title. "
        "J. Am. Chem. Soc. **2023**, *145*, 1234-1245."
    ),
    "wiley": (
        "Wiley 格式: 正文用方括号编号 [1]; 列表按引用顺序. "
        "作者 名缩写 姓, 用逗号分隔; 期刊名斜体; 年份加粗; 卷号; 页码. "
        "示例: J. A. Smith, B. C. Jones, Adv. Mater. **2023**, 35, 2301234."
    ),
    "elsevier": (
        "Elsevier 格式: 正文用方括号编号; 列表按引用顺序. "
        "作者 名缩写 姓; 期刊名缩写斜体; 卷号加粗; 页码; 年份. "
        "示例: Smith J.A., Jones B.C., Title, Acta Mater. "
        "**245** (2023) 123456."
    ),
    "aps_cn": (
        "中文期刊 APS 变体: 作者. 年份. 刊名 卷号 起始页码. "
        "示例: 张三, 李四. 2023. 物理学报 72: 123401."
    ),
    "acs_cn": (
        "中文期刊 ACS 变体: 作者. 文章标题. 刊名, 年份, 卷号, 起止页码. "
        "示例: 张三, 李四. 文章标题. 化学学报, 2023, 81: 123-130."
    ),
}


# ──────────────────────────────────────────────────────────────────────
# 期刊数据库
# ──────────────────────────────────────────────────────────────────────

JOURNAL_DATABASE: dict[str, JournalSpec] = {}


def _register(spec: JournalSpec) -> JournalSpec:
    """注册一本期刊到数据库, 同时用中英文名做 key."""
    JOURNAL_DATABASE[spec.name.lower()] = spec
    JOURNAL_DATABASE[spec.name_en.lower()] = spec
    return spec


# ── 1. Nature ──
_register(JournalSpec(
    name="Nature",
    name_en="Nature",
    publisher="Springer Nature",
    impact_factor="约50.5",
    language="en",
    article_types=["Article", "Letter", "Matter", "News & Views", "Review"],
    field="综合 / 自然科学",
    title_max_chars=75,
    abstract_max_words=200,
    body_max_words=2500,
    methods_max_words=3000,
    display_items_max=4,
    references_max=50,
    reference_format="nature",
    keywords_max= None,
    figure_width_single="89mm",
    figure_width_double="183mm",
    figure_min_font_pt=5,
    figure_max_font_pt=7,
    figure_panel_label="8pt bold, a,b,c",
    figure_dpi=300,
    figure_color_mode="RGB",
    font="Times New Roman 12pt",
    line_spacing="double",
    requires_data_availability=True,
    requires_code_availability=True,
    requires_reporting_summary=True,
    requires_orcid=True,
    allows_preprints=True,
    ai_policy="LLM不满足作者资格, 使用须在Methods声明",
    special_requirements=[
        "摘要段面向非专业读者",
        "化合物用粗体数字编号",
        "晶体数据须CCDC+CheckCIF",
        "太阳能电池/激光须专用报告表",
    ],
    submission_checklist=[
        "正文不超过2500词 (物理科学), Methods不超过3000词",
        "摘要不超过200词, 面向非专业读者",
        "图表总数不超过4个",
        "参考文献不超过50条",
        "提供数据可用性声明",
        "提供代码可用性声明 (如有代码)",
        "生命科学论文附Reporting Summary",
        "全部作者提供ORCID",
        "晶体数据提交CCDC并附CheckCIF报告",
        "太阳能电池论文附专用效率报告表",
        "如使用LLM, 在Methods中声明用途",
    ],
    review_criteria=["原创性", "跨学科重要性", "科学严谨性"],
))


# ── 2. Science ──
_register(JournalSpec(
    name="Science",
    name_en="Science",
    publisher="AAAS",
    impact_factor="约47.7",
    language="en",
    article_types=["Research Article", "Report", "Review", "Letter"],
    field="综合 / 自然科学",
    title_max_chars=120,
    abstract_max_words=125,
    body_max_words=4500,
    display_items_max=4,
    references_max=60,
    reference_format="science",
    figure_dpi=300,
    font="Times New Roman 12pt",
    line_spacing="double",
    requires_data_availability=True,
    allows_preprints=True,
    ai_policy="LLM不可列为作者, 使用须在致谢或Methods声明",
    special_requirements=[
        "摘要为单段, 不超过125词",
        "正文最多4500词含参考文献注解",
        "Research Article需附Research Article Summary",
    ],
    submission_checklist=[
        "标题不超过120字符",
        "摘要不超过125词",
        "正文不超过4500词",
        "图表不超过4个",
        "参考文献不超过60条",
        "提供数据可用性声明",
        "如使用LLM, 在致谢或Methods中声明",
    ],
    review_criteria=["原创性", "科学影响力", "方法论严谨性"],
))


# ── 3. Physical Review Letters (PRL) ──
_register(JournalSpec(
    name="Physical Review Letters",
    name_en="Physical Review Letters",
    publisher="American Physical Society",
    impact_factor="约8.6",
    language="en",
    article_types=["Letter"],
    field="物理学",
    title_max_chars=None,
    abstract_max_words=600,
    body_max_words=3750,
    pages_max=4,
    references_max=30,
    reference_format="prl",
    requires_revtex=True,
    allows_preprints=True,
    ai_policy="LLM不可列为作者, 使用须声明",
    special_requirements=[
        "必须使用REVTeX模板",
        "正文不超过4页含图表",
        "超长论文可投Physical Review D/E",
    ],
    submission_checklist=[
        "使用REVTeX模板排版",
        "正文不超过3750词 (约4页)",
        "图表嵌入正文, 总计不超过4页",
        "参考文献不超过30条",
    ],
    review_criteria=["原创性", "物理重要性", "正确性"],
))


# ── 4. JACS (Journal of the American Chemical Society) ──
_register(JournalSpec(
    name="JACS",
    name_en="Journal of the American Chemical Society",
    publisher="American Chemical Society",
    impact_factor="约14.5",
    language="en",
    article_types=["Article", "Communication", "Perspective"],
    field="化学",
    title_max_chars=None,
    abstract_max_words=250,
    body_max_words=None,
    references_max=50,
    reference_format="acs",
    figure_dpi=300,
    requires_toc_graphic=True,
    toc_size="8.25cm x 4.45cm",
    allows_preprints=True,
    ai_policy="LLM不可列为作者, 使用须在致谢声明",
    special_requirements=[
        "须提供TOC图形 (8.25cm x 4.45cm)",
        "摘要不超过250词",
        "新化合物须提供完整表征数据",
    ],
    submission_checklist=[
        "摘要不超过250词",
        "提供TOC图形 (8.25cm x 4.45cm, 300dpi)",
        "参考文献使用ACS格式",
        "新化合物附完整NMR/MS/IR表征",
        "晶体数据提交CCDC",
    ],
    review_criteria=["原创性", "化学重要性", "数据完整性"],
))


# ── 5. Advanced Materials ──
_register(JournalSpec(
    name="Advanced Materials",
    name_en="Advanced Materials",
    publisher="Wiley-VCH",
    impact_factor="约27.4",
    language="en",
    article_types=["Communication", "Full Paper", "Review", "Progress Report"],
    field="材料科学",
    title_max_chars=None,
    abstract_max_words=200,
    body_max_words=3000,
    display_items_max=5,
    references_max=50,
    reference_format="wiley",
    figure_dpi=300,
    requires_all_authors=True,
    allows_preprints=True,
    ai_policy="LLM不可列为作者, 使用须声明",
    special_requirements=[
        "Communications正文不超过3000词",
        "须列出全部作者信息",
        "鼓励提供TOC图形",
    ],
    submission_checklist=[
        "正文不超过3000词 (Communications)",
        "列出全部作者姓名及单位",
        "图表分辨率300dpi",
        "参考文献使用Wiley格式",
    ],
    review_criteria=["原创性", "材料科学重要性", "跨学科影响"],
))


# ── 6. Acta Materialia ──
_register(JournalSpec(
    name="Acta Materialia",
    name_en="Acta Materialia",
    publisher="Elsevier",
    impact_factor="约8.3",
    language="en",
    article_types=["Full Paper", "Review", "Short Communication"],
    field="材料科学",
    title_max_chars=None,
    abstract_max_words=200,
    max_double_spaced_pages=25,
    references_max=60,
    reference_format="elsevier",
    requires_graphical_abstract=True,
    allows_preprints=True,
    ai_policy="LLM不可列为作者, 使用须声明",
    special_requirements=[
        "双倍行距正文不超过25页",
        "须提供图文摘要 (Graphical Abstract)",
        "摘要不超过200词",
    ],
    submission_checklist=[
        "摘要不超过200词",
        "双倍行距正文不超过25页",
        "提供图文摘要 (531x1328px, TIFF/EPS)",
        "参考文献使用Elsevier格式",
    ],
    review_criteria=["原创性", "材料科学贡献", "实验严谨性"],
))


# ── 7. Nano Letters ──
_register(JournalSpec(
    name="Nano Letters",
    name_en="Nano Letters",
    publisher="American Chemical Society",
    impact_factor="约9.6",
    language="en",
    article_types=["Letter", "Communication"],
    field="纳米科学",
    title_max_chars=None,
    abstract_max_words=250,
    body_max_words=None,
    references_max=30,
    reference_format="acs",
    figure_dpi=300,
    requires_toc_graphic=True,
    toc_size="8.25cm x 4.45cm",
    allows_preprints=True,
    ai_policy="LLM不可列为作者, 使用须在致谢声明",
    special_requirements=[
        "须提供TOC图形",
        "摘要不超过250词",
        "正文不超过4页 (Letter)",
    ],
    submission_checklist=[
        "摘要不超过250词",
        "提供TOC图形 (8.25cm x 4.45cm)",
        "正文不超过4页 (Letter格式)",
        "参考文献使用ACS格式, 不超过30条",
    ],
    review_criteria=["原创性", "纳米科学重要性", "时效性"],
))


# ── 8. ACS Nano ──
_register(JournalSpec(
    name="ACS Nano",
    name_en="ACS Nano",
    publisher="American Chemical Society",
    impact_factor="约15.8",
    language="en",
    article_types=["Article", "Communication", "Review", "Perspective"],
    field="纳米科学",
    title_max_chars=None,
    abstract_max_words=250,
    body_max_words=None,
    references_max=50,
    reference_format="acs",
    figure_dpi=300,
    requires_toc_graphic=True,
    toc_size="8.25cm x 4.45cm",
    allows_preprints=True,
    ai_policy="LLM不可列为作者, 使用须在致谢声明",
    special_requirements=[
        "须提供TOC图形",
        "摘要不超过250词",
        "新合成纳米材料须提供完整表征",
    ],
    submission_checklist=[
        "摘要不超过250词",
        "提供TOC图形 (8.25cm x 4.45cm)",
        "参考文献使用ACS格式",
        "纳米材料附TEM/SEM/XRD/粒度等表征",
    ],
    review_criteria=["原创性", "纳米科学影响", "表征完整性"],
))


# ── 9. 物理学报 ──
_register(JournalSpec(
    name="物理学报",
    name_en="Acta Physica Sinica",
    publisher="中国物理学会",
    impact_factor="约1.5",
    language="zh-CN",
    article_types=["研究论文", "综述", "快报"],
    field="物理学",
    title_max_chars=30,
    abstract_zh_max_chars=300,
    abstract_en_max_words=500,
    keywords_max=5,
    references_min=20,
    reference_format="aps_cn",
    requires_bilingual=True,
    requires_copyright_agreement=True,
    special=[
        "须附详细英文摘要约一个版面",
        "图题表题中英文对照",
        "三线表",
        "AIGC不得列为作者",
    ],
    special_requirements=[
        "须附详细英文摘要约一个版面",
        "图题表题中英文对照",
        "三线表",
        "AIGC不得列为作者",
    ],
    submission_checklist=[
        "中文标题不超过30字",
        "中文摘要不超过300字",
        "英文摘要约500词, 约一个版面",
        "关键词3-5个, 中英文对照",
        "参考文献不少于20条",
        "图题表题中英文对照, 使用三线表",
        "签署版权转让协议",
    ],
    review_criteria=["原创性", "物理意义", "中文表达规范"],
))


# ── 10. 化学学报 ──
_register(JournalSpec(
    name="化学学报",
    name_en="Acta Chimica Sinica",
    publisher="中国化学会",
    impact_factor="约1.2",
    language="zh-CN",
    article_types=["研究论文", "研究简报", "综述", "进展评述"],
    field="化学",
    title_zh_max_chars=20,
    title_en_max_words=10,
    abstract_zh_max_chars=300,
    abstract_en_max_words=500,
    keywords_max=8,
    reference_format="acs_cn",
    requires_graphical_abstract=True,
    requires_ccdc=True,
    new_compound_characterization=["1H NMR", "13C NMR", "IR", "MS", "元素分析或HRMS"],
    special=[
        "中文标题不超过20字, 英文标题不超过10词",
        "须提供图文摘要",
        "新化合物须提供1H NMR/13C NMR/IR/MS/元素分析或HRMS",
        "晶体数据须提交CCDC",
    ],
    special_requirements=[
        "中文标题不超过20字, 英文标题不超过10词",
        "须提供图文摘要",
        "新化合物须提供1H NMR/13C NMR/IR/MS/元素分析或HRMS",
        "晶体数据须提交CCDC",
    ],
    submission_checklist=[
        "中文标题不超过20字, 英文标题不超过10词",
        "中文摘要不超过300字, 英文摘要不超过500词",
        "关键词不超过8个, 中英文对照",
        "提供图文摘要",
        "新化合物附1H NMR/13C NMR/IR/MS/元素分析或HRMS",
        "晶体数据提交CCDC",
        "参考文献使用中文ACS格式",
    ],
    review_criteria=["原创性", "化学意义", "表征完整性"],
))


# ── 11. 金属学报 ──
_register(JournalSpec(
    name="金属学报",
    name_en="Acta Metallurgica Sinica",
    publisher="中国金属学会",
    impact_factor="约0.8",
    language="zh-CN",
    article_types=["研究论文", "综述", "研究简报"],
    field="金属材料",
    authors_max=6,
    abstract_zh_max_chars=300,
    abstract_en_max_words=400,
    keywords_max=8,
    reference_format="acs_cn",
    requires_bilingual=True,
    revision_deadline_days=30,
    ai_image_policy="严禁AI工具制作/修改投稿图片",
    special=[
        "作者不超过6位",
        "严禁AI工具制作/修改投稿图片",
        "修回期限30天",
    ],
    special_requirements=[
        "作者不超过6位",
        "严禁AI工具制作/修改投稿图片",
        "修回期限30天",
    ],
    submission_checklist=[
        "作者不超过6位",
        "中文摘要不超过300字, 英文摘要不超过400词",
        "关键词不超过8个, 中英文对照",
        "投稿图片严禁使用AI工具制作或修改",
        "修回稿须在30天内提交",
    ],
    review_criteria=["原创性", "金属材料贡献", "实验严谨性"],
))


# ── 12. 无机材料学报 ──
_register(JournalSpec(
    name="无机材料学报",
    name_en="Journal of Inorganic Materials",
    publisher="中国科学院上海硅酸盐研究所",
    impact_factor="约0.7",
    language="zh-CN",
    article_types=["研究论文", "综述", "研究简报"],
    field="无机材料",
    title_max_chars=20,
    abstract_max_chars=250,
    keywords_max=4,
    authors_max=6,
    review_max=["综述≤8000字", "论文≤5000字", "简报≤3000字"],
    requires_classification_number=True,
    requires_unit_proof=True,
    special=[
        "综述正文不超过8000字, 论文不超过5000字, 简报不超过3000字",
        "须标注中图分类号",
        "须附单位证明信",
    ],
    special_requirements=[
        "综述正文不超过8000字, 论文不超过5000字, 简报不超过3000字",
        "须标注中图分类号",
        "须附单位证明信",
    ],
    submission_checklist=[
        "标题不超过20字",
        "摘要不超过250字",
        "关键词不超过4个",
        "作者不超过6位",
        "标注中图分类号",
        "附单位证明信",
        "正文字数: 综述≤8000, 论文≤5000, 简报≤3000",
    ],
    review_criteria=["原创性", "材料科学贡献", "规范性"],
))


# ── 13. 硅酸盐学报 ──
_register(JournalSpec(
    name="硅酸盐学报",
    name_en="Journal of the Chinese Ceramic Society",
    publisher="中国硅酸盐学会",
    impact_factor="约0.6",
    language="zh-CN/zh-EN",
    article_types=["研究论文", "综述", "快报"],
    field="硅酸盐 / 无机非金属",
    title_zh_max_chars=20,
    abstract_max_chars=220,
    keywords_max=8,
    figures_max=6,
    plagiarism_check="整段重复率>12%或总重复率>25%直接退稿",
    file_size_max_mb=15,
    special=[
        "整段重复率>12%或总重复率>25%直接退稿",
        "投稿文件不超过15MB",
        "图数量不超过6个",
    ],
    special_requirements=[
        "整段重复率>12%或总重复率>25%直接退稿",
        "投稿文件不超过15MB",
        "图数量不超过6个",
    ],
    submission_checklist=[
        "中文标题不超过20字",
        "摘要不超过220字",
        "关键词不超过8个",
        "图数量不超过6个",
        "投稿文件不超过15MB",
        "查重: 整段重复率≤12%且总重复率≤25%",
    ],
    review_criteria=["原创性", "学术规范", "查重合格"],
))


# ── 14. 中国科学 ──
_register(JournalSpec(
    name="中国科学",
    name_en="Science China",
    publisher="中国科学院 / 科学出版社",
    impact_factor="约1.5",
    language="zh-CN/zh-EN",
    article_types=["研究论文", "综述", "快报"],
    field="综合 / 自然科学",
    has_series=["化学", "物理学力学天文学", "材料科学", "技术科学"],
    requires_originality=True,
    requires_ai_declaration=True,
    special=[
        "分辑: 化学/物理学力学天文学/材料科学/技术科学",
        "须签署原创性声明",
        "须签署AI使用声明",
    ],
    special_requirements=[
        "分辑: 化学/物理学力学天文学/材料科学/技术科学",
        "须签署原创性声明",
        "须签署AI使用声明",
    ],
    submission_checklist=[
        "选择正确的分辑投稿",
        "签署原创性声明",
        "签署AI使用声明 (如使用AI工具须如实声明)",
        "中英文标题/摘要/关键词齐全",
    ],
    review_criteria=["原创性", "科学重要性", "规范性"],
))


# ──────────────────────────────────────────────────────────────────────
# 查询函数
# ──────────────────────────────────────────────────────────────────────


def get_journal(name: str) -> JournalSpec | None:
    """按名称查找期刊, 支持中英文, 模糊匹配.

    先精确匹配 (不区分大小写), 再尝试子串匹配.
    """
    key = name.strip().lower()
    if key in JOURNAL_DATABASE:
        return JOURNAL_DATABASE[key]

    # 子串模糊匹配: 输入可能是缩写或部分名称
    for db_key, spec in JOURNAL_DATABASE.items():
        if key in db_key or db_key in key:
            return spec

    # 常见缩写映射
    aliases: dict[str, str] = {
        "prl": "physical review letters",
        "nature": "nature",
        "science": "science",
        "jacs": "jacs",
        "adv mater": "advanced materials",
        "advanced materials": "advanced materials",
        "acta mater": "acta materialia",
        "nano lett": "nano letters",
        "acs nano": "acs nano",
    }
    if key in aliases:
        return JOURNAL_DATABASE.get(aliases[key])

    return None


def list_journals(field: str | None = None) -> list[JournalSpec]:
    """列出所有期刊, 可按学科领域过滤."""
    seen: set[int] = set()
    result: list[JournalSpec] = []
    for spec in JOURNAL_DATABASE.values():
        if id(spec) in seen:
            continue
        if field is not None and field.lower() not in spec.field.lower():
            continue
        seen.add(id(spec))
        result.append(spec)
    return result


def search_journals(query: str) -> list[JournalSpec]:
    """搜索期刊: 在名称、英文名、出版方、领域中匹配查询词."""
    q = query.strip().lower()
    seen: set[int] = set()
    result: list[JournalSpec] = []
    for spec in JOURNAL_DATABASE.values():
        if id(spec) in seen:
            continue
        haystack = " ".join([
            spec.name, spec.name_en, spec.publisher, spec.field,
        ]).lower()
        if q in haystack:
            seen.add(id(spec))
            result.append(spec)
    return result


def get_reference_format(name: str) -> str:
    """获取期刊的参考文献格式说明.

    先查期刊的 reference_format 字段, 再到 REFERENCE_FORMATS 查说明文本.
    """
    spec = get_journal(name)
    fmt_key = spec.reference_format if spec else None
    if fmt_key and fmt_key in REFERENCE_FORMATS:
        return REFERENCE_FORMATS[fmt_key]
    return f"未找到期刊 '{name}' 的参考文献格式, 请手动查阅投稿指南."
