"""跨文献实体聚合 (huginn 移植版).

上游: RocHunag1996/mineru-material-parser/src/aggregator.py

改造点:
  1. 上游 `from config import DATA_DIR` → 接受 output_dir 参数 (调用方传 workspace)
  2. 输出格式对齐 huginn kg/graph.py 的 add_entity / add_relation API:
     - entities.json: [{label, entity_type, aliases, cas, chemical_name, docs, properties_count}, ...]
     - properties.tsv: doc_id \t entity_label \t category \t name \t value \t unit \t condition
     - relationships.json: [{source, relation, target, doc_id, confidence}, ...]
  3. Union-Find 用 stdlib dict 实现, 不引外部包

三级匹配 (与上游一致):
  Pass 1  精确主键: CAS > Chemical Name > Generic full name
  Pass 2  别名交叉: A 主名出现在 B 别名集合 → 合并
  Pass 3  模糊匹配: token Jaccard >= 0.75
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 模糊匹配阈值 (上游默认 0.75)
_FUZZY_THRESHOLD = 0.75


# ── 工具函数 (上游 _norm_name / _entity_key / _collect_all_names) ─────────────

def _norm_name(s: str | None) -> str:
    """基础归一化: 小写 + 去括号 + 合并空白."""
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    # 去所有括号 (中英文, 方/圆/花)
    s = re.sub(r"[\uff08\uff09()\u3010\u3011\u300a\u300b\[\]{}]", "", s)
    return s


def _entity_key(material: dict | None) -> str:
    """实体主键优先级: CAS > Chemical Name > Generic full name. 空返回 ""."""
    if not material:
        return ""
    cas = (material.get("CAS RN") or material.get("cas_rn") or "").strip()
    if cas:
        return f"CAS:{cas}"
    chem = _norm_name(material.get("Chemical Name") or material.get("chemical_name"))
    if chem:
        return f"CHEM:{chem}"
    gn = material.get("Generic Name") or material.get("generic_name_full")
    if isinstance(gn, dict):
        full = _norm_name(gn.get("full name"))
        if full:
            return f"GEN:{full}"
    elif isinstance(gn, str) and gn:
        return f"GEN:{_norm_name(gn)}"
    return ""


def _collect_all_names(material: dict | None) -> set[str]:
    """提取实体所有名称变体 (归一化后去重): full name + abbreviations + chemical + CAS."""
    names: set[str] = set()
    if not material:
        return names
    gn = material.get("Generic Name") or {}
    if isinstance(gn, str):
        n = _norm_name(gn)
        if n:
            names.add(n)
    elif isinstance(gn, dict):
        full = _norm_name(gn.get("full name"))
        if full:
            names.add(full)
        for ab in gn.get("Abbreviation") or gn.get("abbreviations") or []:
            n = _norm_name(ab)
            if n:
                names.add(n)
    chem = _norm_name(material.get("Chemical Name") or material.get("chemical_name"))
    if chem:
        names.add(chem)
    cas = (material.get("CAS RN") or material.get("cas_rn") or "").strip()
    if cas:
        names.add(f"cas:{cas.lower()}")
    return names


def _tokenize(s: str) -> set[str]:
    """简单分词: 按非字母数字分割, 小写."""
    return {t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Union-Find ────────────────────────────────────────────────────────────────

@dataclass
class _EntityRecord:
    """聚合中的实体记录 (一个 Union-Find 集合的根)."""

    canonical_label: str = ""
    aliases: set[str] = field(default_factory=set)
    cas: str | None = None
    chemical_name: str | None = None
    material_type: str | None = None
    docs: set[str] = field(default_factory=set)  # doc_id 集合
    properties: list[dict] = field(default_factory=list)
    synthesis: list[dict] = field(default_factory=list)
    applications: list[dict] = field(default_factory=list)


class UnionFind:
    """简单 Union-Find, 用 dict 存 parent."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            return x
        # path compression
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> str:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        # 简单策略: 把 rb 挂到 ra 下 (不做 rank, N 量级 < 1000)
        self._parent[rb] = ra
        return ra

    def groups(self) -> dict[str, list[str]]:
        """返回 {root: [members]}."""
        out: dict[str, list[str]] = {}
        for x in self._parent:
            root = self.find(x)
            out.setdefault(root, []).append(x)
        return out


# ── 聚合主流程 ────────────────────────────────────────────────────────────────

def _extract_doc_records(doc_id: str, schema_dict: dict) -> list[_EntityRecord]:
    """从一个文档的 schema_dict 抽出实体记录.

    上游是单文档单材料, 这里也按单材料处理 (一个 doc → 一个 _EntityRecord).
    升级路径: 一篇文档多材料时, 拆成多个 record.
    """
    material = schema_dict.get("Material") or schema_dict.get("material") or {}
    if not material:
        return []
    rec = _EntityRecord()
    gn = material.get("Generic Name") or {}
    if isinstance(gn, dict):
        full = gn.get("full name") or material.get("generic_name_full")
        if full:
            rec.canonical_label = full
    elif isinstance(gn, str) and gn:
        rec.canonical_label = gn
    if not rec.canonical_label:
        chem = material.get("Chemical Name") or material.get("chemical_name")
        if chem:
            rec.canonical_label = chem
        else:
            rec.canonical_label = doc_id  # 兜底: 用 doc_id 作 label

    rec.aliases = _collect_all_names(material)
    rec.cas = (material.get("CAS RN") or material.get("cas_rn") or "").strip() or None
    rec.chemical_name = material.get("Chemical Name") or material.get("chemical_name")
    rec.material_type = material.get("Material_type") or material.get("material_type")
    rec.docs = {doc_id}

    # properties / synthesis / application 挂到实体上 (扁平化, 带 doc_id 溯源)
    props = schema_dict.get("Properties") or schema_dict.get("properties") or []
    if isinstance(props, list):
        for p in props:
            if isinstance(p, dict):
                rec.properties.append({**p, "_doc_id": doc_id})

    syn = schema_dict.get("Synthesis") or schema_dict.get("synthesis") or {}
    if isinstance(syn, dict):
        rec.synthesis.append({**syn, "_doc_id": doc_id})

    app = schema_dict.get("Application") or schema_dict.get("application") or {}
    if isinstance(app, dict):
        rec.applications.append({**app, "_doc_id": doc_id})

    return [rec]


def _match_records(records: list[_EntityRecord]) -> dict[int, list[int]]:
    """三级匹配, 返回 {group_root_idx: [member_idx, ...]}.

    用 record 在 list 中的 index 作 UF 节点 id.
    """
    n = len(records)
    if n == 0:
        return {}
    uf = UnionFind()
    for i in range(n):
        uf.find(str(i))  # 初始化

    # Pass 1: 精确主键 (CAS > Chemical Name > Generic full name)
    key_to_idx: dict[str, int] = {}
    for i, rec in enumerate(records):
        # 用 _entity_key 逻辑
        key_parts: list[str] = []
        if rec.cas:
            key_parts.append(f"CAS:{rec.cas}")
        if rec.chemical_name:
            key_parts.append(f"CHEM:{_norm_name(rec.chemical_name)}")
        if rec.canonical_label:
            key_parts.append(f"GEN:{_norm_name(rec.canonical_label)}")
        for k in key_parts:
            if k in key_to_idx:
                uf.union(str(key_to_idx[k]), str(i))
            else:
                key_to_idx[k] = i

    # Pass 2: 别名交叉
    alias_to_idx: dict[str, int] = {}
    for i, rec in enumerate(records):
        for alias in rec.aliases:
            if alias in alias_to_idx and alias_to_idx[alias] != i:
                uf.union(str(alias_to_idx[alias]), str(i))
            else:
                alias_to_idx[alias] = i

    # Pass 3: 模糊匹配 (token Jaccard)
    # ponytail: O(n²) — n 量级 < 1000 时够用. 升级路径: n > 1000 时换 blocking + LSH.
    token_sets = [_tokenize(rec.canonical_label + " " + " ".join(rec.aliases)) for rec in records]
    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(str(i)) == uf.find(str(j)):
                continue
            if _jaccard(token_sets[i], token_sets[j]) >= _FUZZY_THRESHOLD:
                uf.union(str(i), str(j))

    # 输出 groups
    raw_groups = uf.groups()
    out: dict[int, list[int]] = {}
    for root_str, members in raw_groups.items():
        out[int(root_str)] = [int(m) for m in members]
    return out


def _merge_group(root_idx: int, members: list[int], records: list[_EntityRecord]) -> _EntityRecord:
    """合并一组 record 成单个 _EntityRecord (root 作 canonical)."""
    root = records[root_idx]
    merged = _EntityRecord(
        canonical_label=root.canonical_label,
        aliases=set(root.aliases),
        cas=root.cas,
        chemical_name=root.chemical_name,
        material_type=root.material_type,
        docs=set(root.docs),
        properties=list(root.properties),
        synthesis=list(root.synthesis),
        applications=list(root.applications),
    )
    for m in members:
        if m == root_idx:
            continue
        rec = records[m]
        merged.aliases |= rec.aliases
        if not merged.cas and rec.cas:
            merged.cas = rec.cas
        if not merged.chemical_name and rec.chemical_name:
            merged.chemical_name = rec.chemical_name
        if not merged.material_type and rec.material_type:
            merged.material_type = rec.material_type
        merged.docs |= rec.docs
        merged.properties.extend(rec.properties)
        merged.synthesis.extend(rec.synthesis)
        merged.applications.extend(rec.applications)
    return merged


# ── 输出格式 ──────────────────────────────────────────────────────────────────

def _to_entities_json(entities: list[_EntityRecord]) -> list[dict]:
    """转成 kg.add_entity 友好的格式."""
    out = []
    for e in entities:
        out.append({
            "label": e.canonical_label,
            "entity_type": "material",
            "aliases": sorted(a for a in e.aliases if a),
            "cas": e.cas,
            "chemical_name": e.chemical_name,
            "material_type": e.material_type,
            "docs": sorted(e.docs),
            "properties_count": len(e.properties),
        })
    return out


def _to_properties_tsv(entities: list[_EntityRecord]) -> str:
    """扁平化性能数据, TSV (制表符分隔). 带 doc_id 溯源."""
    header = "doc_id\tentity_label\tcategory\tname\tvalue\tunit\tcondition\toriginal_text"
    lines = [header]
    for e in entities:
        for p in e.properties:
            doc_id = p.get("_doc_id", "")
            cat = p.get("Property_category") or p.get("property_category") or ""
            name = p.get("Property_name") or p.get("property_name") or ""
            val = p.get("Value") or p.get("value") or ""
            unit = p.get("Unit") or p.get("unit") or ""
            cond = p.get("Condition") or p.get("condition") or ""
            ot = p.get("Original text") or p.get("original_text") or ""
            # TSV 里 tab/换行替换成空格, 防止破坏列对齐
            def _clean(s: str) -> str:
                return str(s).replace("\t", " ").replace("\n", " ").strip()
            lines.append("\t".join([
                _clean(doc_id), _clean(e.canonical_label), _clean(cat),
                _clean(name), _clean(val), _clean(unit), _clean(cond), _clean(ot),
            ]))
    return "\n".join(lines)


def _to_relationships_json(entities: list[_EntityRecord]) -> list[dict]:
    """材料 → 应用领域 / 材料 → 性能 的关系. 对齐 kg.add_relation.

    ponytail: 上游还有 material-to-material 关系 (合成产物-前驱体), 这里简化为
    材料-应用 + 材料-性能 两类. 升级路径: 加 Synthesis.Raw_materials 关系.
    """
    out = []
    for e in entities:
        if not e.canonical_label:
            continue
        # 材料 → 应用领域
        for app in e.applications:
            field = app.get("Application_field") or app.get("application_field")
            if field:
                out.append({
                    "source": e.canonical_label,
                    "relation": "applied_in",
                    "target": field,
                    "doc_id": ",".join(sorted(e.docs)),
                    "confidence": 0.8,
                })
        # 材料 → 性能 (按 category 聚合, 避免关系爆炸)
        cats: dict[str, set[str]] = {}
        for p in e.properties:
            cat = p.get("Property_category") or p.get("property_category")
            name = p.get("Property_name") or p.get("property_name")
            if cat and name:
                cats.setdefault(cat, set()).add(name)
        for cat, names in cats.items():
            for name in names:
                out.append({
                    "source": e.canonical_label,
                    "relation": f"has_property:{cat.lower()}",
                    "target": name,
                    "doc_id": ",".join(sorted(e.docs)),
                    "confidence": 0.7,
                })
    return out


# ── 顶层入口 ─────────────────────────────────────────────────────────────────

def aggregate(
    extracted: dict[str, dict],
    *,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """跨文献聚合.

    Args:
        extracted: {doc_id: schema_dict} — 每个文献的 MaterialSchema.to_dict() 输出
        output_dir: 写入目录. None 时不落盘, 只返回结果.

    Returns:
        {"entities": [...], "properties_tsv": str, "relationships": [...], "stats": {...}}
    """
    # 1. 抽 record
    records: list[_EntityRecord] = []
    for doc_id, schema_dict in extracted.items():
        records.extend(_extract_doc_records(doc_id, schema_dict))

    if not records:
        logger.warning("aggregate: 无可聚合 record")
        result = {"entities": [], "properties_tsv": "doc_id\tentity_label\tcategory\tname\tvalue\tunit\tcondition\toriginal_text",
                  "relationships": [], "stats": {"docs": 0, "entities": 0, "properties": 0}}
        if output_dir is not None:
            _write_outputs(Path(output_dir), result)
        return result

    # 2. 三级匹配
    groups = _match_records(records)

    # 3. 合并每组
    merged_entities: list[_EntityRecord] = []
    for root_idx, members in groups.items():
        merged_entities.append(_merge_group(root_idx, members, records))

    # 4. 输出
    result = {
        "entities": _to_entities_json(merged_entities),
        "properties_tsv": _to_properties_tsv(merged_entities),
        "relationships": _to_relationships_json(merged_entities),
        "stats": {
            "docs": len(extracted),
            "raw_records": len(records),
            "merged_entities": len(merged_entities),
            "properties": sum(len(e.properties) for e in merged_entities),
            "relationships": len(_to_relationships_json(merged_entities)),
        },
    }
    if output_dir is not None:
        _write_outputs(Path(output_dir), result)
    return result


def _write_outputs(output_dir: Path, result: dict) -> None:
    """落盘: entities.json / properties.tsv / relationships.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "entities.json").write_text(
        json.dumps(result["entities"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "properties.tsv").write_text(
        result["properties_tsv"],
        encoding="utf-8",
    )
    (output_dir / "relationships.json").write_text(
        json.dumps(result["relationships"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "aggregate 写入 %s: %d entities, %d relationships",
        output_dir, len(result["entities"]), len(result["relationships"]),
    )


# ── KG 集成: 把聚合结果喂给 huginn ProjectKnowledgeGraph ──────────────────────

def ingest_into_kg(kg: Any, result: dict) -> dict[str, int]:
    """把 aggregate() 的结果喂给 huginn kg/graph.py 实例.

    kg 需实现 add_entity(label, entity_type, ...) 和 add_relation(src, rel, dst, ...).
    返回 {entities_added, relations_added} 统计.

    ponytail: 不另建图存储 — 复用 huginn ProjectKnowledgeGraph. 调用方传 kg 实例.
    """
    stats = {"entities_added": 0, "relations_added": 0}
    label_to_id: dict[str, str] = {}

    for ent in result.get("entities") or []:
        label = ent.get("label")
        if not label:
            continue
        try:
            eid = kg.add_entity(
                label=label,
                entity_type=ent.get("entity_type", "material"),
                source="mineru_aggregator",
                confidence=0.85,
                cas=ent.get("cas") or "",
                chemical_name=ent.get("chemical_name") or "",
                material_type=ent.get("material_type") or "",
                docs=",".join(ent.get("docs") or []),
                aliases="|".join(ent.get("aliases") or []),
            )
            label_to_id[label] = eid
            stats["entities_added"] += 1
        except Exception as e:
            logger.warning("kg.add_entity(%s) failed: %s", label, e)

    for rel in result.get("relationships") or []:
        src = rel.get("source")
        dst = rel.get("target")
        if not src or not dst:
            continue
        # 节点不存在时先建 (relation 的 target 可能是新实体如应用领域)
        if src not in label_to_id:
            try:
                label_to_id[src] = kg.add_entity(src, "material", source="mineru_aggregator", confidence=0.7)
            except Exception:
                label_to_id[src] = ""
        if dst not in label_to_id:
            try:
                label_to_id[dst] = kg.add_entity(dst, "topic", source="mineru_aggregator", confidence=0.7)
            except Exception:
                label_to_id[dst] = ""
        src_id = label_to_id.get(src, "")
        dst_id = label_to_id.get(dst, "")
        if not src_id or not dst_id:
            continue
        try:
            kg.add_relation(
                src_id, rel.get("relation", "related_to"), dst_id,
                source="mineru_aggregator",
                confidence=rel.get("confidence", 0.7),
                doc_id=rel.get("doc_id", ""),
            )
            stats["relations_added"] += 1
        except Exception as e:
            logger.warning("kg.add_relation(%s -%s-> %s) failed: %s", src, rel.get("relation"), dst, e)

    return stats


if __name__ == "__main__":
    # C5 self-check: 三级匹配 + 输出格式 + KG 集成.
    import tempfile

    # 1. _norm_name / _entity_key / _collect_all_names
    assert _norm_name("PVDF (Polyvinylidene Fluoride)") == "pvdf polyvinylidene fluoride"
    assert _norm_name("  Hello  World  ") == "hello world"
    assert _norm_name(None) == ""
    assert _norm_name("") == ""

    m1 = {"CAS RN": "58-08-2", "Chemical Name": "Caffeine",
           "Generic Name": {"full name": "Caffeine", "Abbreviation": ["咖啡因"]}}
    assert _entity_key(m1) == "CAS:58-08-2"
    m2 = {"Chemical Name": "PVDF", "Generic Name": {"full name": "Polyvinylidene Fluoride"}}
    assert _entity_key(m2) == "CHEM:pvdf"
    m3 = {"Generic Name": {"full name": "PEO"}}
    assert _entity_key(m3) == "GEN:peo"
    assert _entity_key(None) == ""
    assert _entity_key({}) == ""

    names = _collect_all_names(m1)
    assert "caffeine" in names
    assert "咖啡因" in names
    assert "cas:58-08-2" in names

    # 2. UnionFind
    uf = UnionFind()
    assert uf.find("a") == "a"
    uf.union("a", "b")
    assert uf.find("a") == uf.find("b")
    uf.union("c", "d")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("d")
    groups = uf.groups()
    assert len(groups) == 1

    # 3. 三级匹配: 同一材料不同表达应合并
    extracted = {
        "doc1": {
            "Material": {"CAS RN": "58-08-2", "Chemical Name": "Caffeine",
                          "Generic Name": {"full name": "Caffeine", "Abbreviation": ["咖啡因"]}},
            "Properties": [{"Property_category": "THERMAL", "Property_name": "melting_point",
                             "Value": "235", "Unit": "℃", "Original text": "mp 235"}],
            "Application": {"Application_field": "食品"},
        },
        "doc2": {
            # 同一材料, 不同 doc, CAS 一样 → Pass 1 精确合并
            "Material": {"CAS RN": "58-08-2", "Chemical Name": "Caffeine",
                          "Generic Name": {"full name": "Caffeine"}},
            "Properties": [{"Property_category": "THERMAL", "Property_name": "density",
                             "Value": "1.23", "Unit": "g/cm³"}],
            "Application": {"Application_field": "医药"},
        },
        "doc3": {
            # 不同材料, 别名交叉 → Pass 2 合并
            "Material": {"Generic Name": {"full name": "PVDF", "Abbreviation": ["聚偏氟乙烯"]}},
            "Properties": [],
        },
        "doc4": {
            # 同 PVDF, 但用全称, 别名包含 "PVDF" → Pass 2 合并
            "Material": {"Generic Name": {"full name": "Polyvinylidene Fluoride",
                                           "Abbreviation": ["PVDF"]}},
        },
        "doc5": {
            # 完全无关材料
            "Material": {"Generic Name": {"full name": "PEO"}},
        },
    }
    with tempfile.TemporaryDirectory() as td:
        result = aggregate(extracted, output_dir=Path(td))
        # 应该聚成 3 个实体: Caffeine / PVDF / PEO
        assert result["stats"]["merged_entities"] == 3, f"expected 3, got {result['stats']['merged_entities']}"
        labels = [e["label"] for e in result["entities"]]
        assert "Caffeine" in labels
        # PVDF 或 Polyvinylidene Fluoride 之一作 canonical
        assert any("PVDF" in l or "Polyvinylidene" in l for l in labels)
        assert "PEO" in labels
        # Caffeine 跨 doc1+doc2, 应有 2 个 doc, 2 条 properties
        caffeine = next(e for e in result["entities"] if e["label"] == "Caffeine")
        assert set(caffeine["docs"]) == {"doc1", "doc2"}
        assert caffeine["properties_count"] == 2
        # 输出文件存在
        assert (Path(td) / "entities.json").exists()
        assert (Path(td) / "properties.tsv").exists()
        assert (Path(td) / "relationships.json").exists()
        # properties.tsv 至少 4 行 (header + 2 个 caffeine 性能 + 1 个 PVDF 性能... 等)
        tsv = (Path(td) / "properties.tsv").read_text(encoding="utf-8")
        assert "doc1" in tsv and "doc2" in tsv
        assert "melting_point" in tsv and "density" in tsv
        # relationships.json 包含 applied_in + has_property 关系
        rels = json.loads((Path(td) / "relationships.json").read_text(encoding="utf-8"))
        rel_kinds = {r["relation"] for r in rels}
        assert "applied_in" in rel_kinds
        assert any(r.startswith("has_property:") for r in rel_kinds)

    # 4. ingest_into_kg: mock KG 验证 API 对齐
    class MockKG:
        def __init__(self):
            self.entities = []
            self.relations = []
        def add_entity(self, label, entity_type, **kw):
            eid = f"node_{len(self.entities)}"
            self.entities.append({"id": eid, "label": label, "type": entity_type, **kw})
            return eid
        def add_relation(self, src, rel, dst, **kw):
            self.relations.append({"src": src, "rel": rel, "dst": dst, **kw})

    mock = MockKG()
    stats = ingest_into_kg(mock, result)
    assert stats["entities_added"] >= 3  # 至少 3 材料 + 可能的应用领域 topic
    assert stats["relations_added"] >= 2  # 至少 2 条 applied_in 关系
    # 实体类型对齐: material / topic
    types = {e["type"] for e in mock.entities}
    assert "material" in types
    # 关系字段对齐 kg.add_relation(src, rel, dst, source, confidence, doc_id)
    for r in mock.relations:
        assert "src" in r and "rel" in r and "dst" in r
        assert "source" in r and "confidence" in r and "doc_id" in r

    # 5. 空 extracted 兜底
    empty_result = aggregate({})
    assert empty_result["entities"] == []
    assert empty_result["stats"]["entities"] == 0

    print("C5 self-check OK")
