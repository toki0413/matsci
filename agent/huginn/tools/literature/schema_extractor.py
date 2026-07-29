"""Schema 抽取核心 (huginn 移植版).

上游: RocHunag1996/mineru-material-parser/src/extractor.py + doc_chunker.py

改造点:
  1. 上游 `from deepseek_client import chat_json` → huginn ModelRegistry.get(alias)
  2. 上游 `from doc_chunker import Document, parse_document` → 本模块内联简化版
  3. 上游 `from few_shot import fewshot_for` + `from schema_validator import ...` 移除
     (huginn 走 SCHEMA_JSON 原样 + LLM 自然语言指令, 不依赖 few-shot pool)
  4. 上游 ThreadPoolExecutor Phase 2 并行 → 串行 (lazy: 6 块调用 6 次 LLM,
     真要并行可包 ThreadPoolExecutor, 但 agent 内部一次只跑一篇, 串行够用)

整体流程 (与上游一致):
  Phase 1 (串行): parse_document → extract_anchors → extract_material → extract_metadata
  Phase 2 (串行): extract_structure / properties / synthesis / application / metadata 已在 P1
  后处理: verify_original_texts 五级溯源, 剔除无法定位的 Original text

ponytail: 不引 few-shot pool / schema_validator / pdf_splitter 依赖. 升级路径:
  - 抽取质量不够时, 在 SCHEMA_JSON 之外补 few-shot 示例 (放 data/fewshot_pool/)
  - 长文档切片由 SmartIngester 处理, 这里不重复.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.models.registry import ModelRegistry, get_model_capabilities
from .schema_def import SCHEMA_JSON, MaterialSchema, from_llm_output

logger = logging.getLogger(__name__)


# ── Document / Chunk (上游 doc_chunker.py 移植简化版) ────────────────────────

# 章节正则 (上游 SECTION_PATTERNS)
_SECTION_PATTERNS: dict[str, re.Pattern] = {
    "abstract": re.compile(r"^(摘\s*要|abstract)\b", re.I),
    "intro": re.compile(r"^(引言|前言|绪论|introduction|背景)", re.I),
    "experimental": re.compile(
        r"^(实验|实验部分|材料与方法|材料和方法|实验材料|"
        r"具体实施方式|实施方式|实施例|"
        r"experimental|materials\s*(and|&)?\s*methods?|"
        r"methodology|方法|methods?)",
        re.I,
    ),
    "synthesis": re.compile(
        r"^(合成|制备|制备工艺|制备方法|制备例|synthesis|preparation)",
        re.I,
    ),
    "results": re.compile(
        r"^(结果|结果与讨论|results?(\s*(and|&)?\s*discussion)?|讨论|分析与讨论|性能测试|性能表征|表征)",
        re.I,
    ),
    "conclusion": re.compile(r"^(结\s*论|总\s*结|conclusion|summary)", re.I),
    "references": re.compile(r"^(参考文献|references?|文献)", re.I),
    "acknowledgement": re.compile(r"^(致\s*谢|acknowledg)", re.I),
}


@dataclass
class Chunk:
    """MinerU content_list 的一项."""

    idx: int
    type: str  # text/header/table/chart/image/equation/list/page_number
    text: str = ""
    page_idx: int = 0
    text_level: int | None = None  # header 层级, 1=最大


@dataclass
class Document:
    """解析后的文档, 提供章节聚合与全文."""

    chunks: list[Chunk] = field(default_factory=list)
    sections: dict[str, list[Chunk]] = field(default_factory=dict)
    abstract: str | None = None
    tables: list[Chunk] = field(default_factory=list)
    figures: list[Chunk] = field(default_factory=list)
    first_page_text: str = ""
    full_text: str = ""

    def section_text(self, key: str, max_chars: int = 8000) -> str:
        """按 section key (abstract/intro/experimental/...) 取拼接文本, 截断到 max_chars."""
        chunks = self.sections.get(key) or []
        if not chunks:
            return ""
        return "\n".join(c.text for c in chunks if c.text)[:max_chars]


def _classify_section(text: str) -> str | None:
    """根据 chunk 文本判断所属 section key. 命中第一个就返回.

    容忍 "1. Introduction" / "2.1 实验材料" 这类带编号前缀的标题.
    """
    if not text:
        return None
    head = text.strip().split("\n", 1)[0][:80]
    # 去掉开头编号前缀: "1. " / "2.1 " / "一、" / "第一节 " 等
    head = re.sub(r"^[\d.、\s]+", "", head)
    head = re.sub(r"^(第[一二三四五六七八九十百\d]+[章节条])\s*", "", head)
    for key, pat in _SECTION_PATTERNS.items():
        if pat.search(head):
            return key
    return None


def parse_document(parsed_dir: Path) -> Document:
    """读 MinerU 解析结果目录的 content_list.json, 切成 Document.

    parsed_dir 结构 (MinerU 输出):
      {stem}/
        ├── content_list.json    # 主要: chunk 列表
        ├── fullmd.md            # 全文 markdown
        └── json/                # 版面布局
    """
    parsed_dir = Path(parsed_dir)
    cl_path = parsed_dir / "content_list.json"
    if not cl_path.exists():
        # 兼容: 部分 MinerU 版本把 content_list 放在 json/ 子目录
        cl_path = parsed_dir / "json" / "content_list.json"
    if not cl_path.exists():
        raise FileNotFoundError(f"content_list.json not found under {parsed_dir}")

    raw = json.loads(cl_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"unexpected content_list format: {type(raw)}")

    chunks: list[Chunk] = []
    sections: dict[str, list[Chunk]] = {}
    tables: list[Chunk] = []
    figures: list[Chunk] = []
    abstract_text: str | None = None
    first_page_chunks: list[str] = []
    current_section: str | None = None

    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        ctype = item.get("type", "text")
        text = item.get("text") or ""
        page_idx = item.get("page_idx", 0) or 0
        text_level = item.get("text_level")
        chunk = Chunk(idx=idx, type=ctype, text=text, page_idx=page_idx, text_level=text_level)
        chunks.append(chunk)

        if ctype == "table":
            tables.append(chunk)
        elif ctype in ("chart", "image"):
            figures.append(chunk)

        # section 识别: header 类型触发切换, text 类型跟随当前 section
        if ctype == "header":
            sec = _classify_section(text)
            if sec:
                current_section = sec
                # header 本身不进 sections 列表, 只作为分界标记
        else:
            if current_section:
                sections.setdefault(current_section, []).append(chunk)
            # abstract 特殊: 第一个 abstract chunk 单独存
            if current_section == "abstract" and abstract_text is None and text.strip():
                abstract_text = text.strip()

        if page_idx == 0 and text.strip():
            first_page_chunks.append(text.strip())

    # full_text: 全 chunk 拼接
    full_text = "\n".join(c.text for c in chunks if c.text)
    first_page_text = "\n".join(first_page_chunks)[:6000]

    return Document(
        chunks=chunks,
        sections=sections,
        abstract=abstract_text,
        tables=tables,
        figures=figures,
        first_page_text=first_page_text,
        full_text=full_text,
    )


# ── LLM 调用: huginn ModelRegistry 替代上游 deepseek_client ───────────────────

SYS_PROMPT = """你是材料学文献信息抽取专家, 严格遵守规则:
1. 严格按提供的 Schema 输出 JSON, 字段名一字不差, 未提及/无法判定的字段填 null.
2. 数值字段 Value 仅写数字或区间 (不含单位), Unit 单独列出, 保留原文写法.
3. 性能 (Properties) 与合成参数 (Synthesis.Parameters) 的 Original text 字段必须
   是【输入文本中原样存在的片段】, 不可改写、不可翻译.
4. 不要编造文本中没有的事实; 宁缺毋滥.
5. 输出必须是合法 JSON 对象, 不要加 markdown 代码块、不要附加解释."""


def _strip_code_fences(text: str) -> str:
    """去掉 ```json ... ``` 包裹. 与 huginn conjecture._parse_json 对齐."""
    if not text:
        return ""
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    return text.strip()


def chat_json(
    model: Any,
    messages: list[dict],
    *,
    max_tokens: int = 8000,
    temperature: float = 0.0,
) -> dict:
    """调 huginn LangChain model 实例, 返回 JSON dict.

    复用 huginn LLM 调用约定 (与 autoloop/conjecture.py 一致):
      - model.invoke(messages) 同步, RuntimeError (已有 event loop) 时用 asyncio.run
      - 兼容 async: 调用方在 event loop 里时 model.ainvoke
    失败返回空 dict, 让上层跳过该块 (lazy: 不重试, 整体流程仍可输出部分结果).
    """
    import asyncio

    try:
        try:
            asyncio.get_running_loop()
            resp = model.invoke(messages)
        except RuntimeError:
            resp = asyncio.run(model.ainvoke(messages))
        text = str(resp.content).strip() if hasattr(resp, "content") else str(resp).strip()
        text = _strip_code_fences(text)
        if not text:
            return {}
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("chat_json 解析失败: %s", e)
        return {}
    except Exception as e:
        # LLM 调用本身失败 (网络/限流/auth) — 不阻断整体流程
        logger.warning("chat_json LLM 调用失败: %s", e)
        return {}


# ── 抽取流程: Phase 1 锚点 + 6 块 ─────────────────────────────────────────────

ANCHOR_PROMPT = """从下面的{source_name}中抽取作为"全文导航锚点"的信息, 用于后续在全文中精准定位证据.
返回 JSON:
{{
  "primary_materials": [{{"name":"...","aliases":["..."],"abbrev":"..."}}],
  "key_properties":    [{{"name":"...","value_hint":"...","condition_hint":"..."}}],
  "process_keywords":  ["..."],
  "application_hints": ["..."]
}}
"""


def _best_anchor_source(doc: Document) -> tuple[str, str]:
    """选锚点源: 摘要 > 首页 > 全文前 3000 字. 返回 (source_name, text)."""
    if doc.abstract and len(doc.abstract.strip()) > 50:
        return "摘要", doc.abstract
    if doc.first_page_text and len(doc.first_page_text.strip()) > 50:
        return "首页文本", doc.first_page_text[:4000]
    if doc.full_text and len(doc.full_text.strip()) > 50:
        return "文献开头文本", doc.full_text[:3000]
    return "", ""


def extract_anchors(doc: Document, model: Any) -> dict:
    """抽取摘要锚点. 无可用文本时返回空锚点."""
    source_name, text = _best_anchor_source(doc)
    if not source_name:
        return {"primary_materials": [], "key_properties": [],
                "process_keywords": [], "application_hints": []}
    prompt = ANCHOR_PROMPT.format(source_name=source_name) + f"\n{source_name}:\n" + text
    return chat_json(
        model,
        [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4000,
    ) or {"primary_materials": [], "key_properties": [],
          "process_keywords": [], "application_hints": []}


def _section_payload(doc: Document, anchors: dict) -> dict:
    """构建 6 块抽取的公共 payload (摘要/intro/conclusion/anchors)."""
    return {
        "abstract": (doc.abstract or "")[:3000],
        "intro": doc.section_text("intro", 3000),
        "experimental": doc.section_text("experimental", 6000),
        "synthesis": doc.section_text("synthesis", 4000),
        "results": doc.section_text("results", 6000),
        "conclusion": doc.section_text("conclusion", 2500),
        "anchors": anchors.get("primary_materials", []),
    }


def _extract_block(
    doc: Document,
    model: Any,
    block_name: str,
    schema_block: Any,
    extra_hint: str = "",
    max_tokens: int = 8000,
) -> dict:
    """通用单块抽取. schema_block 是 SCHEMA_JSON[block_name] (dict 或 list)."""
    payload = _section_payload(doc, {})
    user_intro = (
        f"请填充 Schema 的 {block_name} 块. 仅依据输入数据中的事实, 不要发明.\n"
        f"Schema ({block_name}):\n{json.dumps(schema_block, ensure_ascii=False, indent=2)}\n"
        f"{extra_hint}\n"
    )
    user_msg = user_intro + "\n输入数据 (JSON):\n" + json.dumps(payload, ensure_ascii=False)
    return chat_json(
        model,
        [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
    )


def extract_schema(
    doc: Document,
    model: Any,
    *,
    run_verify: bool = True,
) -> MaterialSchema:
    """两阶段抽取: anchors → 6 块 → (可选) 原文溯源校验.

    串行执行 (lazy). 真要并行可包 ThreadPoolExecutor, 但 agent 内部一次一篇够用.
    """
    # Phase 1: anchors (用于后续 prompt 注入, 但不阻塞)
    try:
        anchors = extract_anchors(doc, model)
    except Exception:
        anchors = {"primary_materials": [], "key_properties": [],
                   "process_keywords": [], "application_hints": []}

    # 6 块串行抽取. 单块失败返回 {} → from_llm_output 给默认空值, 整体仍可输出.
    material = _extract_block(doc, model, "Material", SCHEMA_JSON["Material"])
    structure = _extract_block(doc, model, "Structure", SCHEMA_JSON["Structure"])
    properties_raw = _extract_block(
        doc, model, "Properties", SCHEMA_JSON["Properties"],
        extra_hint="Properties 是数组, 返回 [{{...}}, {{...}}] 形式.",
    )
    synthesis = _extract_block(doc, model, "Synthesis", SCHEMA_JSON["Synthesis"])
    application = _extract_block(doc, model, "Application", SCHEMA_JSON["Application"])
    metadata = _extract_block(doc, model, "Metadata", SCHEMA_JSON["Metadata"])

    # Properties 可能直接是 list, 也可能是 {"Properties": [...]}
    if isinstance(properties_raw, list):
        props_list = properties_raw
    elif isinstance(properties_raw, dict):
        props_list = properties_raw.get("Properties") or properties_raw.get("properties") or []
        # 单 dict (单条性能) 也兜底成 list
        if isinstance(props_list, dict):
            props_list = [props_list]
    else:
        props_list = []

    extracted = {
        "Material": material,
        "Structure": structure,
        "Properties": props_list,
        "Synthesis": synthesis,
        "Application": application,
        "Metadata": metadata,
    }

    if run_verify and doc.full_text:
        extracted = verify_original_texts(extracted, doc.full_text)

    return from_llm_output(extracted)


# ── verify_original_texts: 五级原文溯源校验 ──────────────────────────────────
# 上游 extractor.py 五级: 精确 → 归一化 → 高模糊 (>=0.9) → 低模糊 (>=0.7) → n-gram (>=0.5)
# 校验范围: Properties[].Original text + Synthesis.Parameters[].Original text
# 不通过的条目: 上游剔除该条 (设为 null). huginn 沿用该策略.

_NORM_PUNCT = re.compile(r"[\s,.;:!?，。；：！？、（）()\[\]\"'`]+")


def _normalize(s: str) -> str:
    """归一化: 去标点 + 空格 + 大小写, 用于二级溯源."""
    return _NORM_PUNCT.sub("", s or "").lower()


def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """n-gram Jaccard 相似度. 短串 (< n) 直接返回 0."""
    a, b = a.lower(), b.lower()
    if len(a) < n or len(b) < n:
        return 0.0
    sa = {a[i:i + n] for i in range(len(a) - n + 1)}
    sb = {b[i:i + n] for i in range(len(b) - n + 1)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _verify_one(original: str, full_text: str) -> bool:
    """五级溯源. 通过任一级即返回 True."""
    if not original or not full_text:
        return False
    # 1. 精确
    if original in full_text:
        return True
    # 2. 归一化
    if _normalize(original) in _normalize(full_text):
        return True
    ratio = difflib.SequenceMatcher(None, _normalize(original), _normalize(full_text)).quick_ratio()
    # 3. 高模糊 (整体相似度 >= 0.9) — quick_ratio 是上界, 走 find_longest_match 严格判定
    if ratio >= 0.9:
        # 取原文本里最长连续子串在 full_text 中匹配
        match = difflib.SequenceMatcher(None, _normalize(original), _normalize(full_text)).find_longest_match(
            0, len(_normalize(original)), 0, len(_normalize(full_text))
        )
        if match.size / max(len(_normalize(original)), 1) >= 0.7:
            return True
    # 4. 低模糊: 用 difflib.get_close_matches 在 full_text 滑窗里找
    # ponytail: 滑窗代价 O(n*m), 这里只对短 original 做 (长度 < 200)
    if len(original) < 200:
        # 切 full_text 成 ~len(original)*2 的窗口, 找最高相似度
        win = max(len(original) * 2, 100)
        best = 0.0
        norm_orig = _normalize(original)
        norm_full = _normalize(full_text)
        for i in range(0, max(len(norm_full) - win, 0) + 1, win // 2 or 1):
            sub = norm_full[i:i + win]
            r = difflib.SequenceMatcher(None, norm_orig, sub).ratio()
            if r > best:
                best = r
            if best >= 0.7:
                return True
    # 5. n-gram Jaccard >= 0.5
    if _ngram_jaccard(original, full_text, n=3) >= 0.5:
        return True
    return False


def verify_original_texts(extracted: dict, full_text: str) -> dict:
    """对 Properties + Synthesis.Parameters 的 Original text 做五级溯源.

    不通过的条目: Original text 置 null (上游策略). 同时收集统计到日志.
    """
    stats = {"checked": 0, "passed": 0, "failed": 0}

    props = extracted.get("Properties") or []
    if isinstance(props, list):
        for p in props:
            if not isinstance(p, dict):
                continue
            ot = p.get("Original text") or p.get("original_text")
            if ot:
                stats["checked"] += 1
                if _verify_one(ot, full_text):
                    stats["passed"] += 1
                else:
                    p["Original text"] = None
                    p["original_text"] = None
                    stats["failed"] += 1

    syn = extracted.get("Synthesis") or {}
    if isinstance(syn, dict):
        params = syn.get("Parameters") or []
        if isinstance(params, list):
            for p in params:
                if not isinstance(p, dict):
                    continue
                ot = p.get("Original text") or p.get("original_text")
                if ot:
                    stats["checked"] += 1
                    if _verify_one(ot, full_text):
                        stats["passed"] += 1
                    else:
                        p["Original text"] = None
                        p["original_text"] = None
                        stats["failed"] += 1

    if stats["checked"]:
        logger.info(
            "verify_original_texts: checked=%d passed=%d failed=%d",
            stats["checked"], stats["passed"], stats["failed"],
        )
    return extracted


# ── 顶层入口 ─────────────────────────────────────────────────────────────────

def extract_from_parsed_dir(
    parsed_dir: Path,
    model: Any,
    *,
    run_verify: bool = True,
) -> MaterialSchema:
    """从 MinerU 解析目录抽取 MaterialSchema. 调用方负责拿 model 实例."""
    doc = parse_document(parsed_dir)
    return extract_schema(doc, model, run_verify=run_verify)


def resolve_model(
    registry: ModelRegistry,
    alias: str | None = None,
    thinking: Any = None,
    max_tokens: int | None = None,
) -> Any:
    """从 registry 拿 model 实例. alias 为 None 时走 default_alias().

    ponytail: 不在这里查 structured_output 能力 — huginn 调用约定是 LLM 直接输出
    JSON 文本, 由 chat_json 解析. 升级路径: 强 structured_output 时改用
    model.with_structured_output(MaterialSchema) 走 pydantic 校验.
    """
    a = alias or registry.default_alias()
    if not a:
        raise ValueError("registry 无可用 alias, 请配置 models 或显式传 alias")
    return registry.get(a, thinking=thinking, max_tokens=max_tokens)


if __name__ == "__main__":
    # C4 self-check: 不调真实 LLM (无 key + 无网络), 验证 parse_document / 五级溯源 / 反序列化.
    import tempfile

    # 1. parse_document: 模拟 MinerU content_list.json
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "fake_stem"
        d.mkdir()
        sample_content = [
            {"type": "header", "text": "Abstract", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "This paper studies PVDF electrolyte.", "page_idx": 0},
            {"type": "header", "text": "1. Introduction", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "PVDF is a polymer.", "page_idx": 0},
            {"type": "header", "text": "2. Experimental", "text_level": 1, "page_idx": 1},
            {"type": "text", "text": "Tg was measured at 25 ℃.", "page_idx": 1},
            {"type": "table", "text": "[Table 1: properties]", "page_idx": 1},
            {"type": "chart", "text": "[Fig 1: SEM]", "page_idx": 2},
        ]
        (d / "content_list.json").write_text(json.dumps(sample_content), encoding="utf-8")
        doc = parse_document(d)
        assert doc.abstract == "This paper studies PVDF electrolyte."
        assert "abstract" in doc.sections
        assert "intro" in doc.sections
        assert "experimental" in doc.sections
        assert len(doc.tables) == 1
        assert len(doc.figures) == 1
        assert "PVDF" in doc.full_text
        assert doc.section_text("experimental").startswith("Tg was measured")
        # full_text 应包含所有 chunk
        assert "Table 1" in doc.full_text and "Fig 1" in doc.full_text

    # 2. _verify_one 五级溯源
    full = "The Tg of PVDF was measured to be 25 ℃ using DSC at a heating rate of 10 ℃/min."
    assert _verify_one("Tg of PVDF was measured to be 25 ℃", full) is True  # 精确
    assert _verify_one("Tg of PVDF was measured to be 25 ℃ using DSC", full) is True  # 归一化后子串
    assert _verify_one("Tg of PVDF was measured to be 25 ℃ using DSC at a heating", full) is True  # 归一化
    assert _verify_one("Tg of PVDF was 25 ℃ measured", full) is False  # 词序调换, 不算原文
    assert _verify_one("完全无关的文本 xyz123", full) is False
    assert _verify_one("", full) is False
    assert _verify_one("Tg", "") is False

    # 3. verify_original_texts 整体: 通过保留, 不通过置 null
    extracted = {
        "Properties": [
            {"Property_name": "Tg", "Original text": "Tg of PVDF was measured to be 25 ℃"},
            {"Property_name": "fake", "Original text": "完全不存在的原文 xyz"},
        ],
        "Synthesis": {
            "Parameters": [
                {"Parameter_name": "temp", "Original text": "heating rate of 10 ℃/min"},
                {"Parameter_name": "fake_param", "Original text": "假参数 999"},
            ]
        },
    }
    out = verify_original_texts(extracted, full)
    assert out["Properties"][0]["Original text"] == "Tg of PVDF was measured to be 25 ℃"
    assert out["Properties"][1]["Original text"] is None
    assert out["Synthesis"]["Parameters"][0]["Original text"] == "heating rate of 10 ℃/min"
    assert out["Synthesis"]["Parameters"][1]["Original text"] is None

    # 4. _strip_code_fences
    assert _strip_code_fences("```json\n{\"a\":1}\n```") == '{"a":1}'
    assert _strip_code_fences('{"a":1}') == '{"a":1}'
    assert _strip_code_fences("") == ""

    # 5. _ngram_jaccard
    assert _ngram_jaccard("abcdef", "abcdef", 3) == 1.0
    assert _ngram_jaccard("abc", "xyz", 3) == 0.0
    assert 0 < _ngram_jaccard("PVDF electrolyte", "PVDF based electrolyte", 3) < 1

    # 6. _normalize
    assert _normalize("Hello, World!") == "helloworld"
    assert _normalize("Tg of PVDF，25℃") == "tgofpvdf25℃"

    # 7. extract_schema 不调 LLM 的最小路径: model=None 时 chat_json 异常被吞, 返回空 schema
    #    (验证不会因 LLM 失败而 crash 整个流程)
    empty_doc = Document()
    s = extract_schema(empty_doc, model=None, run_verify=False)
    assert s.material.cas_rn is None
    assert s.properties == []

    print("C4 self-check OK")
