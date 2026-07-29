"""材料文献 Schema 定义 (6 块 + 11 类性能).

上游: RocHunag1996/mineru-material-parser/赛题说明/Schema.json

设计:
  - SCHEMA_JSON: 上游 Schema.json 原样常量, 喂 LLM 做 structured output
  - PropertyCategory: 11 类性能枚举 (PHYSICAL/MECHANICAL/.../FUNCTIONAL)
  - 6 个 dataclass: Material/Structure/Properties/Synthesis/Application/Metadata
  - MaterialSchema: 顶层容器, 提供 from_llm_output 反序列化 LLM JSON 输出

ponytail: dataclass 字段用 snake_case (Python 风格), 但 SCHEMA_JSON 保留上游
PascalCase + 带空格 key 不变 — LLM 看到的就是 SCHEMA_JSON 原样, 零映射.
from_llm_output 用 Schema.json key 反查 dataclass 字段, 单点映射.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class PropertyCategory(str, Enum):
    """11 类性能一级类别. 继承 str 让枚举值直接 JSON 友好."""

    PHYSICAL = "PHYSICAL"
    MECHANICAL = "MECHANICAL"
    THERMAL = "THERMAL"
    ELECTRICAL = "ELECTRICAL"
    MAGNETIC = "MAGNETIC"
    OPTICAL = "OPTICAL"
    CHEMICAL = "CHEMICAL"
    SURFACE = "SURFACE"
    PROCESS = "PROCESS"
    ENVIRONMENTAL = "ENVIRONMENTAL"
    FUNCTIONAL = "FUNCTIONAL"


# ── 6 块 dataclass ───────────────────────────────────────────────────────────


@dataclass
class Material:
    """材料身份: CAS / 通用名 / 化学名 / 类别."""

    cas_rn: str | None = None
    generic_name_full: str | None = None
    generic_name_abbreviations: list[str] = field(default_factory=list)
    chemical_name: str | None = None
    material_type: str | None = None
    material_subtype: str | None = None


@dataclass
class Structure:
    """分子/拓扑/晶体/孔结构."""

    molecular_formula: str | None = None
    structural_formula: str | None = None
    smiles_expression: str | None = None
    basic_units: str | None = None
    molecular_weight: str | None = None
    topological_structure: str | None = None
    stereosequence_structure: str | None = None
    crystallinity: str | None = None
    crystalline_structure: str | None = None
    amorphous_structure: str | None = None
    pore_structure: str | None = None
    other_structure: str | None = None


@dataclass
class Property:
    """单条性能记录. category 走 PropertyCategory 枚举校验."""

    property_category: str | None = None
    property_name: str | None = None
    characterization: str | None = None
    equipment: str | None = None
    condition: str | None = None
    value: str | None = None
    unit: str | None = None
    original_text: str | None = None

    def validate_category(self) -> bool:
        """category 在 11 类枚举内才算合法. 不合法时上层保留原值但打 warning."""
        if not self.property_category:
            return False
        try:
            PropertyCategory(self.property_category)
            return True
        except ValueError:
            return False


@dataclass
class RawMaterial:
    name: str | None = None
    manufacturer: str | None = None


@dataclass
class SynthesisParameter:
    name: str | None = None
    value: str | None = None
    unit: str | None = None
    original_text: str | None = None


@dataclass
class PostTreatment:
    name: str | None = None
    process: str | None = None
    result_effect: str | None = None
    purity: str | None = None


@dataclass
class Synthesis:
    """合成工艺: 方法/方程/原料/催化剂/溶剂/参数/设备/后处理."""

    process: str | None = None
    reaction_equation: str | None = None
    raw_materials: list[RawMaterial] = field(default_factory=list)
    catalyst: str | None = None
    solvent: str | None = None
    parameters: list[SynthesisParameter] = field(default_factory=list)
    equipment: str | None = None
    post_treatment: PostTreatment | None = None


@dataclass
class Application:
    application_field: str | None = None
    application_description: str | None = None


@dataclass
class Author:
    name: str | None = None
    organizations: list[str] = field(default_factory=list)
    author_type: str | None = None  # 通讯/第一/第X作者


@dataclass
class Metadata:
    source: str | None = None
    uid: str | None = None  # DOI / 专利号
    year: str | None = None
    title: str | None = None
    author_organization: list[Author] = field(default_factory=list)


@dataclass
class MaterialSchema:
    """顶层容器, 对应上游 Schema.json 6 块."""

    material: Material = field(default_factory=Material)
    structure: Structure = field(default_factory=Structure)
    properties: list[Property] = field(default_factory=list)
    synthesis: Synthesis = field(default_factory=Synthesis)
    application: Application = field(default_factory=Application)
    metadata: Metadata = field(default_factory=Metadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def invalid_property_categories(self) -> list[str]:
        """返回未通过枚举校验的 category 原值, 供溯源校验打 warning."""
        return [p.property_category for p in self.properties if p.property_category and not p.validate_category()]


# ── SCHEMA_JSON: 上游 Schema.json 原样, 喂 LLM ────────────────────────────────
# ponytail: 不重新设计 Schema, 直接复用上游. LLM 输出按这个 key 结构,
# from_llm_output 再反序列化到 dataclass. 升级路径: 改这里就改这里.

SCHEMA_JSON: dict[str, Any] = {
    "Material": {
        "CAS RN": "美国化学文摘社为每种化学物质分配的唯一数字识别号码, 如 58-08-2.",
        "Generic Name": {
            "full name": "材料的通用名称或商品或俗称",
            "Abbreviation": ["别名/简称/缩写"],
        },
        "Chemical Name": "可唯一对应化学结构的名称 (IUPAC > 通用化学名 > 行业缩写)",
        "Material_type": "材料类别 (如高分子聚合物/小分子有机物等)",
        "Material_subtype": "材料细分类型 (如光刻胶/聚合物电解质/粘结剂/其他)",
    },
    "Structure": {
        "Molecular_Formula": "分子式 (各原子种类和数目), 如 C8H10N4O2",
        "Structural Formula": "结构式或结构简式, 显示原子连接方式",
        "SMILES Expression": "SMILES 结构规范描述",
        "Basic Units": "分子骨架/取代基/官能团组成 (小分子) 或重复结构单元 (高分子)",
        "Molecular Weight": "相对分子质量 (小分子) 或 Mn/Mw/PDI/聚合度 (高分子)",
        "Topological Structure": "分子链连接方式和空间拓扑 (支化度/交联点密度/接枝率/嵌段比)",
        "Stereosequence Structure": "单体空间排列和连接顺序 (立构规整度/序列分布/顺反异构)",
        "Crystallinity": "晶态区域占比",
        "Crystalline Structure": "晶胞参数/晶面间距/晶粒尺寸/取向度/堆积密度",
        "Amorphous Structure": "无定形区域近邻分子排布 (分子间距/配位数/链段取向度)",
        "Pore Structure": "孔隙率/孔径分布/孔容/BET 比表面积",
        "other structure": "极性/手性/芳香性/形貌/界面结构等",
    },
    "Properties": [
        {
            "Property_category": "性能一级类别: PHYSICAL/MECHANICAL/THERMAL/ELECTRICAL/MAGNETIC/OPTICAL/CHEMICAL/SURFACE/PROCESS/ENVIRONMENTAL/FUNCTIONAL",
            "Property_name": "性能标准化标识 (如 density/tensile_strength/thermal_conductivity 等)",
            "Characterization": "测试方法 (NMR/FTIR/GPC/XRD/DSC-TGA/SEM-TEM/AFM/EIS/拉伸试验机/光刻量测等)",
            "Equipment": "测试设备/装置",
            "Condition": "测量或成立的实验/环境条件 (室温/800C/氩气/真空/100Hz 等)",
            "Value": "性能数值, 不含单位, 保留原文格式",
            "Unit": "计量单位, 保留原文写法 (W/m·K, MPa 等)",
            "Original text": "原文性能名称, 便于溯源",
        }
    ],
    "Synthesis": {
        "Synthesis_process": "制备工艺方法名称",
        "Reaction equation": "反应物/主产物/副产物/反应条件",
        "Raw_materials": [
            {"Raw_materials name": "原材料名称", "Raw_materials manufacturer": "原材料制造商"}
        ],
        "Catalyst": "催化剂",
        "Solvent": "溶剂",
        "Parameters": [
            {
                "Parameter_name": "工艺参数类型 (温度/时间/压力/浓度/电压/气氛等)",
                "Value": "参数数值或范围, 不含单位",
                "Unit": "参数计量单位",
                "Original text": "原文描述",
            }
        ],
        "Equipment": "使用设备 (管式炉/球磨机/反应釜/CVD 设备等)",
        "post_treatment": {
            "post_treatment name": "后处理工艺名称",
            "post_treatment process": "后处理工艺过程",
            "result_effect": "工艺结果 (致密度/孔隙率/导电性/纳米结构等)",
            "Purity": "目标成分占总量的百分比",
        },
    },
    "Application": {
        "Application_field": "应用领域 (能源/电子/医疗等)",
        "Application_description": "应用描述",
    },
    "Metadata": {
        "Source": "数据来源 (论文/专利等)",
        "UID": "文档唯一 ID (DOI/专利号等)",
        "year": "年份",
        "Title": "文档标题",
        "author_organization": [
            {
                "Author": "作者",
                "Organization": ["机构名称"],
                "Author Type": "作者类型 (通讯/第一/第X作者)",
            }
        ],
    },
}


# ── LLM 输出反序列化 ─────────────────────────────────────────────────────────
# ponytail: 上游 Schema.json 用 PascalCase + 带空格 key, dataclass 用 snake_case.
# 这里是唯一映射点. 改 Schema 时同步改这里.

def _opt(d: dict | None, *keys: str) -> str | None:
    """从 dict 里按多个候选 key 取值, 取第一个非空. 容忍 key 大小写/空格差异."""
    if not d:
        return None
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return None


def _block(d: dict, *names: str) -> dict | None:
    """从顶层 dict 取 6 块之一, 同时容忍 PascalCase / snake_case / 大小写."""
    for n in names:
        v = d.get(n)
        if v:
            return v
    # 大小写不敏感 fallback
    lower_map = {k.lower(): v for k, v in d.items()}
    for n in names:
        v = lower_map.get(n.lower())
        if v:
            return v
    return None


def _parse_material(d: dict | None) -> Material:
    if not d:
        return Material()
    gn = d.get("Generic Name") or {}
    if isinstance(gn, str):
        gn_full, abbr = gn, []
    else:
        gn_full = gn.get("full name")
        abbr = gn.get("Abbreviation") or []
    return Material(
        cas_rn=_opt(d, "CAS RN", "cas_rn"),
        generic_name_full=gn_full,
        generic_name_abbreviations=list(abbr) if isinstance(abbr, list) else [],
        chemical_name=_opt(d, "Chemical Name", "chemical_name"),
        material_type=_opt(d, "Material_type", "material_type"),
        material_subtype=_opt(d, "Material_subtype", "material_subtype"),
    )


def _parse_structure(d: dict | None) -> Structure:
    if not d:
        return Structure()
    return Structure(
        molecular_formula=_opt(d, "Molecular_Formula", "molecular_formula"),
        structural_formula=_opt(d, "Structural Formula", "structural_formula"),
        smiles_expression=_opt(d, "SMILES Expression", "smiles_expression"),
        basic_units=_opt(d, "Basic Units", "basic_units"),
        molecular_weight=_opt(d, "Molecular Weight", "molecular_weight"),
        topological_structure=_opt(d, "Topological Structure", "topological_structure"),
        stereosequence_structure=_opt(d, "Stereosequence Structure", "stereosequence_structure"),
        crystallinity=_opt(d, "Crystallinity", "crystallinity"),
        crystalline_structure=_opt(d, "Crystalline Structure", "crystalline_structure"),
        amorphous_structure=_opt(d, "Amorphous Structure", "amorphous_structure"),
        pore_structure=_opt(d, "Pore Structure", "pore_structure"),
        other_structure=_opt(d, "other structure", "other_structure"),
    )


def _parse_property(d: dict | None) -> Property:
    if not d:
        return Property()
    return Property(
        property_category=_opt(d, "Property_category", "property_category"),
        property_name=_opt(d, "Property_name", "property_name"),
        characterization=_opt(d, "Characterization", "characterization"),
        equipment=_opt(d, "Equipment", "equipment"),
        condition=_opt(d, "Condition", "condition"),
        value=_opt(d, "Value", "value"),
        unit=_opt(d, "Unit", "unit"),
        original_text=_opt(d, "Original text", "original_text"),
    )


def _parse_synthesis(d: dict | None) -> Synthesis:
    if not d:
        return Synthesis()
    raw_list = d.get("Raw_materials") or []
    raws = [
        RawMaterial(
            name=_opt(r, "Raw_materials name", "name"),
            manufacturer=_opt(r, "Raw_materials manufacturer", "manufacturer"),
        )
        for r in raw_list if isinstance(r, dict)
    ]
    param_list = d.get("Parameters") or []
    params = [
        SynthesisParameter(
            name=_opt(p, "Parameter_name", "name"),
            value=_opt(p, "Value", "value"),
            unit=_opt(p, "Unit", "unit"),
            original_text=_opt(p, "Original text", "original_text"),
        )
        for p in param_list if isinstance(p, dict)
    ]
    pt = d.get("post_treatment") or {}
    post = PostTreatment(
        name=_opt(pt, "post_treatment name", "name"),
        process=_opt(pt, "post_treatment process", "process"),
        result_effect=_opt(pt, "result_effect"),
        purity=_opt(pt, "Purity", "purity"),
    ) if isinstance(pt, dict) and pt else None
    return Synthesis(
        process=_opt(d, "Synthesis_process", "process"),
        reaction_equation=_opt(d, "Reaction equation", "reaction_equation"),
        raw_materials=raws,
        catalyst=_opt(d, "Catalyst", "catalyst"),
        solvent=_opt(d, "Solvent", "solvent"),
        parameters=params,
        equipment=_opt(d, "Equipment", "equipment"),
        post_treatment=post,
    )


def _parse_application(d: dict | None) -> Application:
    if not d:
        return Application()
    return Application(
        application_field=_opt(d, "Application_field", "application_field"),
        application_description=_opt(d, "Application_description", "application_description"),
    )


def _parse_metadata(d: dict | None) -> Metadata:
    if not d:
        return Metadata()
    auth_list = d.get("author_organization") or []
    authors = [
        Author(
            name=_opt(a, "Author", "name"),
            organizations=list(a.get("Organization") or []) if isinstance(a, dict) else [],
            author_type=_opt(a, "Author Type", "author_type"),
        )
        for a in auth_list if isinstance(a, dict)
    ]
    return Metadata(
        source=_opt(d, "Source", "source"),
        uid=_opt(d, "UID", "uid"),
        year=_opt(d, "year"),
        title=_opt(d, "Title", "title"),
        author_organization=authors,
    )


def from_llm_output(data: dict | None) -> MaterialSchema:
    """从 LLM 输出的 JSON dict 反序列化为 MaterialSchema.

    容忍 None / 空 dict / 部分 key 缺失. 上游 Schema.json key 与 snake_case 都接受.
    """
    if not data:
        return MaterialSchema()
    props_raw = _block(data, "Properties", "properties") or []
    props = [_parse_property(p) for p in props_raw if isinstance(p, dict)]
    return MaterialSchema(
        material=_parse_material(_block(data, "Material", "material")),
        structure=_parse_structure(_block(data, "Structure", "structure")),
        properties=props,
        synthesis=_parse_synthesis(_block(data, "Synthesis", "synthesis")),
        application=_parse_application(_block(data, "Application", "application")),
        metadata=_parse_metadata(_block(data, "Metadata", "metadata")),
    )


if __name__ == "__main__":
    # C3 self-check: 枚举完整性 + SCHEMA_JSON 6 块 + 反序列化往返.
    assert len(PropertyCategory) == 11, f"expected 11 categories, got {len(PropertyCategory)}"
    for cat in PropertyCategory:
        assert cat.value == cat.name, f"enum value must equal name: {cat}"

    # SCHEMA_JSON 6 块齐全
    expected_blocks = {"Material", "Structure", "Properties", "Synthesis", "Application", "Metadata"}
    assert set(SCHEMA_JSON.keys()) == expected_blocks, f"schema blocks mismatch: {set(SCHEMA_JSON.keys())}"

    # 反序列化: 空 dict -> 默认值
    s0 = from_llm_output(None)
    assert s0.material.cas_rn is None
    assert s0.properties == []

    # 反序列化: 完整 LLM 输出 (PascalCase + 带空格 key)
    sample = {
        "Material": {
            "CAS RN": "58-08-2",
            "Generic Name": {"full name": "Caffeine", "Abbreviation": ["咖啡因"]},
            "Chemical Name": "1,3,7-Trimethylxanthine",
            "Material_type": "小分子有机物",
        },
        "Structure": {"Molecular_Formula": "C8H10N4O2", "SMILES Expression": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"},
        "Properties": [
            {"Property_category": "THERMAL", "Property_name": "melting_point", "Value": "235", "Unit": "℃"},
            {"Property_category": "INVALID_CAT", "Property_name": "x"},
        ],
        "Synthesis": {
            "Synthesis_process": "萃取",
            "Raw_materials": [{"Raw_materials name": "咖啡豆"}],
            "Parameters": [{"Parameter_name": "温度", "Value": "100", "Unit": "℃"}],
            "post_treatment": {"post_treatment name": "重结晶", "Purity": "99%"},
        },
        "Application": {"Application_field": "食品/医药"},
        "Metadata": {"DOI": "10.1xxx", "Title": "Caffeine review", "author_organization": [{"Author": "Zhang", "Organization": ["MIT"]}]},
    }
    s = from_llm_output(sample)
    assert s.material.cas_rn == "58-08-2"
    assert s.material.generic_name_full == "Caffeine"
    assert s.material.generic_name_abbreviations == ["咖啡因"]
    assert s.structure.molecular_formula == "C8H10N4O2"
    assert len(s.properties) == 2
    assert s.properties[0].property_category == "THERMAL"
    assert s.properties[0].validate_category() is True
    assert s.properties[1].validate_category() is False  # INVALID_CAT 不在 11 类
    assert s.synthesis.process == "萃取"
    assert s.synthesis.raw_materials[0].name == "咖啡豆"
    assert s.synthesis.parameters[0].unit == "℃"
    assert s.synthesis.post_treatment is not None
    assert s.synthesis.post_treatment.purity == "99%"
    assert s.metadata.title == "Caffeine review"
    # to_dict 往返
    d = s.to_dict()
    assert d["material"]["cas_rn"] == "58-08-2"
    # invalid_property_categories 收集非法 category
    assert s.invalid_property_categories() == ["INVALID_CAT"]
    # snake_case key 也容忍
    s2 = from_llm_output({"material": {"cas_rn": "1", "chemical_name": "X"}})
    assert s2.material.cas_rn == "1"
    assert s2.material.chemical_name == "X"
    print("C3 self-check OK")
