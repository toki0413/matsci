"""LiteratureTool 主体: 7 个 action 分发 + LLM 综述 + 文献基准对比.

search/summarize/benchmark_lookup/fetch_pdf/citations/ingest_to_rag/crawl_web.
HTTP 层在 _http, 搜索源在 search_sources, PDF 抓取在 pdf_fetch,
爬虫与订阅源认证在 crawl_web.
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools._llm import invoke_llm_text
from huginn.tools.base import HuginnTool

# ───────────────────────── search 增强 (P0-1 / P1-4) ─────────────────────────
from huginn.utils.cache import TimedLRUCache

from ._http import (
    _DISABLED_HINT,
    _disabled,
    _http_get_bytes,
    _http_get_json,
    logger,
)
from .crawl_web import (
    _PROVIDERS,
    _auth_login,
    _auth_logout,
    _list_sessions,
    crawl_direct,
    crawl_search_engine,
)
from .pdf_fetch import (
    _scihub_enabled,
    _scihub_pdf_url,
    europepmc_pdf,
    openalex_oa_url,
    split_sections,
    unpaywall_pdf,
)
from .search_sources import (
    _apply_filters,
    _dedup,
    _enrich_with_crossref,
    _norm_title,
    _opencitations_citations,
    _opencitations_references,
    _rerank,
    _search_arxiv,
    _search_cod,
    _search_core,
    _search_crossref,
    _search_datacite,
    _search_doaj,
    _search_europepmc,
    _search_materials_cloud,
    _search_materials_project,
    _search_nomad,
    _search_openaire,
    _search_openalex,
    _search_pubmed,
    _search_s2,
    _search_zenodo,
)

# P1-4: 检索结果 TTL 缓存 — 同一 query 命中直接返回, 不再重打 15 路 API.
# 有界 + 带 TTL, 避免重复限流也避免无限增长 (与 checkpointer 容量封顶同一纪律).
_SEARCH_CACHE: TimedLRUCache[ToolResult] = TimedLRUCache(max_size=128, ttl=3600.0)

# P0-1: 子查询只打"核心学术源" — 面向"侧面/子主题"召回, 材料数据库 (COD/NOMAD/DataCite)
# 和数据集对分面 query 收益低, 不打, 控制 API 调用量. 每个子查询结果数减半.
_SUBQUERY_SOURCES: dict[str, Any] = {
    "arxiv": _search_arxiv,
    "s2": _search_s2,
    "crossref": _search_crossref,
    "openalex": _search_openalex,
    "pubmed": _search_pubmed,
}

# 15 源统一分发表 (显式 if-blocks 的浓缩, 行为一致)
_SOURCE_FNS: dict[str, Any] = {
    "arxiv": _search_arxiv,
    "s2": _search_s2,
    "crossref": _search_crossref,
    "openalex": _search_openalex,
    "pubmed": _search_pubmed,
    "doaj": _search_doaj,
    "core": _search_core,
    "europepmc": _search_europepmc,
    "zenodo": _search_zenodo,
    "openaire": _search_openaire,
    "cod": _search_cod,
    "materials_cloud": _search_materials_cloud,
    "nomad": _search_nomad,
    "datacite": _search_datacite,
    "materials_project": _search_materials_project,
}

_EXPAND_SYSTEM_PROMPT = (
    "你是文献检索助手. 把用户的复杂研究问题拆成 2-3 个互补的子查询, "
    "每个子查询聚焦一个侧面/子主题, 尽量用具体名词而非修饰语. "
    "子查询之间不要互相包含 (去掉重叠).\n"
    '只输出 JSON: {"subqueries": ["子查询1", "子查询2", ...]}\n'
    '若问题已经足够具体单一, 输出 {"subqueries": []}.'
)


async def _expand_query(model: Any, query: str, max_n: int = 3) -> list[str]:
    """LLM 把 query 拆成子查询 (Q 端增强). 失败/空结果返回 [], 不阻塞搜索."""
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        messages = [
            SystemMessage(content=_EXPAND_SYSTEM_PROMPT),
            HumanMessage(content=f"研究问题: {query}"),
        ]
        if hasattr(model, "ainvoke"):
            response = await model.ainvoke(messages)
        else:
            response = await asyncio.to_thread(model.invoke, messages)
        content = response.content if hasattr(response, "content") else str(response)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        parsed = LiteratureTool._parse_json(content)
        subs = parsed.get("subqueries", []) if parsed else []
    except Exception as exc:
        logger.debug("query expansion failed, search without subqueries: %s", exc)
        return []

    out: list[str] = []
    norm_orig = _norm_title(query)
    for s in subs[:max_n]:
        s = (s or "").strip()
        if not s or _norm_title(s) == norm_orig:
            continue
        if s.lower() not in {o.lower() for o in out}:
            out.append(s)
    return out


def _search_cache_key(args: LiteratureInput, subqueries: tuple[str, ...]) -> tuple:
    """检索缓存 key: 覆盖 query/源/年份/条数 + 是否扩展 + 子查询列表.

    不含 args.action — summarize/ingest/benchmark 内部的隐式 search 与显式
    search 共享同一缓存 (语义一致, 减少重复 15 路).
    """
    return (
        args.query,
        tuple(args.sources),
        args.year_from,
        args.year_to,
        args.max_results,
        args.min_citations,
        args.oa_only,
        args.expand_query,
        subqueries,
    )


# ───────────────────────── helper: 跨源数值一致性标注 (L2a) ─────────────────────────


def _avc_unit_status(
    unit: str,
    items: list[tuple[int, float]],
    consistent_rel: float,
    moderate_rel: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    """一个单位组的稳健统计 + 每个值的标签; 返回 (status, counts)."""
    vals = sorted(v for _, v in items)
    n = len(vals)
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    span = max(abs(median), 1.0)
    counts = {"single_source": 0, "consistent": 0, "moderate": 0, "conflicting": 0}
    status: dict[str, Any] = {
        "unit": None if unit == "_none" else unit,
        "n_sources": n,
        "median": round(median, 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
        "values": [],
    }
    for _, v in items:
        rel = abs(v - median) / span
        if n == 1:
            label = "single_source"
        elif rel <= consistent_rel:
            label = "consistent"
        elif rel <= moderate_rel:
            label = "moderate"
        else:
            label = "conflicting"
        counts[label] += 1
        status["values"].append(
            {
                "rel_dev": round(rel, 4),
                "label": label,
                "value": v,
            }
        )
    return status, counts


def _avc_overall(
    n_reported: int, n_units: int, counts: dict[str, int]
) -> dict[str, Any]:
    """全局一致性判定 (consensus / mixed / conflicting_values)."""
    cross_verifiable = counts["consistent"] + counts["moderate"] + counts["conflicting"]
    agreement = counts["consistent"] + counts["moderate"]
    overall = {
        "n_sources": n_reported,
        "n_units": n_units,
        "agreement_ratio": round(agreement / cross_verifiable, 4)
        if cross_verifiable
        else 0.0,
        "verdict": "insufficient_data",
        "counts": dict(counts),
    }
    if n_reported < 2:
        overall["verdict"] = "insufficient_data"
    elif cross_verifiable and overall["agreement_ratio"] >= 0.75:
        overall["verdict"] = "consensus"
    elif cross_verifiable and overall["agreement_ratio"] >= 0.5:
        overall["verdict"] = "mixed"
    else:
        overall["verdict"] = "conflicting_values"
    return overall


def _annotate_value_consistency(
    reported: list[dict[str, Any]],
    *,
    consistent_rel: float = 0.05,
    moderate_rel: float = 0.20,
) -> dict[str, Any]:
    """跨源数值校验 — 给 benchmark_lookup 报的每个值打一致性标签.

    同体系同性质, 不同文献报的值应互相印证. 这里按单位分组, 用中位数作
    稳健中心 (对离群值不敏感), 算每个值的相对偏差 |v-median|/max(|median|,1),
    按阈值打标签:

      - single_source: 该单位只有一个来源, 无法交叉验证
      - consistent:  相对偏差 ≤ consistent_rel (5%)  — 跨源一致
      - moderate:    ≤ moderate_rel (20%)            — 大致相符, 可重复性一般
      - conflicting: > moderate_rel                  — 显著分歧, 需人工排查

    返回 {"by_unit": [...], "overall": {...}}. 纯函数, 不打网络.
    """
    # 按 (unit) 分组, 跳过无数值的
    groups: dict[str, list[tuple[int, float]]] = {}
    for idx, v in enumerate(reported):
        unit = str(v.get("unit") or "").strip() or "_none"
        groups.setdefault(unit, []).append((idx, v["value"]))

    by_unit: list[dict[str, Any]] = []
    totals = {"single_source": 0, "consistent": 0, "moderate": 0, "conflicting": 0}
    for unit, items in groups.items():
        status, counts = _avc_unit_status(unit, items, consistent_rel, moderate_rel)
        for label, count in counts.items():
            totals[label] += count
        by_unit.append(status)

    by_unit.sort(key=lambda u: u.get("unit") or "")
    return {
        "by_unit": by_unit,
        "overall": _avc_overall(len(reported), len(groups), totals),
    }


def _infer_known_conditions(reported: list[dict[str, Any]]) -> dict[str, str]:
    """从 reported 聚合"已定义"的自由度, 作缺度追问的 known 条件.

    原则 (与 condition_normalize 一致): 只把"明确出现"的自由度当作已知.
      为避免个别来源的声明掩盖真实的缺度, 采用**过半锁定**:
      - method_family: 超过 60% 来源明确同一方法族才锁定
      - functional  : 仅当全表过半集中在同一种明确泛函 (PBE+HSE 并存 → 不锁定,
                      恰好让 functional 维持缺度 → 触发 HSE vs PBE 补全)
      - temperature: 仅当过半来源落在同一明确温度档才锁定 (少数来源声明室温,
                     不掩盖其余 DFT 组的温度缺度)

    纯函数, 零网络/零 LLM.
    """
    from .condition_normalize import feature_vector

    n_total = max(1, len(reported))
    thr = 0.6
    fams: dict[str, int] = {}
    funcs: dict[str, int] = {}
    temps: dict[str, int] = {}
    for r in reported:
        f = feature_vector(
            str(r.get("method") or ""),
            str(r.get("note") or ""),
            str(r.get("unit") or ""),
        )
        if f["method_family"] != "unknown":
            fams[f["method_family"]] = fams.get(f["method_family"], 0) + 1
        if f["functional"]:
            funcs[f["functional"]] = funcs.get(f["functional"], 0) + 1
        if f["temperature"] != "unknown":
            temps[f["temperature"]] = temps.get(f["temperature"], 0) + 1

    known: dict[str, str] = {}
    if fams:
        fam = max(fams, key=fams.get)
        if fams[fam] / n_total >= thr:
            known["method_family"] = fam
    if len(funcs) == 1:
        func = next(iter(funcs))
        if funcs[func] / n_total >= thr:
            known["functional"] = func
    if len(temps) == 1:
        temp = next(iter(temps))
        if temps[temp] / n_total >= thr:
            known["temperature"] = temp
    return known


# ───────────────────────── LLM prompts ─────────────────────────


_SUMMARY_SYSTEM_PROMPT = """你是材料科学文献综述专家. 基于给定的 N 篇论文标题和摘要, 写一段结构化综述.

输出格式 (Markdown, 不要加代码块标记):

## 关键发现
- 发现1 [1]
- 发现2 [2][3]
...

## 领域共识
(研究者普遍接受的结论)

## 主要分歧
(不同研究矛盾的地方, 标引用编号)

## 数值汇总
| 体系 | 性质 | 数值 | 方法 | 来源 |
|------|------|------|------|------|
| ... | ... | ... | ... | [1] |

## 研究空白
(摘要里没覆盖到的方向, 值得做的下一步)

引用编号 [1][2]... 对应输入论文的顺序. 只引用给定的论文, 不要编造.
如果论文太少或摘要太短不足以综述, 直接说明, 不要硬凑."""


_BENCHMARK_SYSTEM_PROMPT = """你是材料科学数据提取专家. 从给定的论文列表里, 抽出关于指定体系和性质的具体报道值.

每篇论文如果报了具体数值, 提取:
- value: 数值 (float, 不要带单位字符串)
- unit: 单位 (如 "eV", "epsilon", "GPa", "K"; 没明确单位给空字符串)
- method: 计算/实验方法 (如 "DFT-PBE", "MD", "basin-hopping", "experiment"; 没说给空)
- paper_idx: 论文编号 (从 1 开始, 对应输入顺序)
- note: 一句话备注 (可选, 如 "T=0K", "PBE+U")

严格要求:
1. 只抽明确给出的数值, 不要从公式或趋势里推断.
2. 同一篇报多个值的, 每个值一条.
3. 没报数值的论文直接跳过.
4. 数值保留原文精度, 不要四舍五入.

输出严格 JSON, 不要 markdown 代码块, 不要解释:
{"values": [{"value": -44.33, "unit": "epsilon", "method": "basin-hopping", "paper_idx": 1, "note": ""}, ...]}
如果没有任何论文报数值, 返回 {"values": []}"""


# ───────────────────────── multi_review 透镜 (nuwa 启发) ─────────────────────────
# 6 路并行透镜, 每路独立 LLM 调用, 独立产出 findings. 失败透镜降级为空, 不阻塞其他.
# ponytail: 透镜数固定 6, 不做动态扩展. 升级: 根据 query 类型自适应选透镜.

_LENS_PROMPTS: dict[str, str] = {
    "methodology": """你是方法论审查员. 从给定论文里抽出方法论相关的关键论断 (claim).
关注: 使用的方法/实验设计/统计处理/控制变量/样本量是否充分.
每条 claim 标注: 哪些论文 (paper_idx) 支持它, 你的置信度 (high/medium/low).
输出 JSON: {"lens":"methodology","findings":[{"claim":"...","paper_idx":[1,3],"confidence":"medium"}],"summary":"一句话总结方法论整体情况"}""",
    "contributions": """你是贡献提取员. 从给定论文里抽出核心贡献相关的关键论断 (claim).
关注: 声称的新发现/改进幅度/理论突破/工程价值. 区分"作者声称"和"证据支持".
每条 claim 标注: 哪些论文 (paper_idx) 支持它, 你的置信度 (high/medium/low).
输出 JSON: {"lens":"contributions","findings":[{"claim":"...","paper_idx":[1],"confidence":"high"}],"summary":"一句话总结贡献整体情况"}""",
    "limitations": """你是局限性探测员. 从给定论文里抽出局限性和未解决问题相关的关键论断 (claim).
关注: 作者自承的局限/未测试的假设/外推风险/泛化边界/缺失的对照.
每条 claim 标注: 哪些论文 (paper_idx) 提到它, 你的置信度 (high/medium/low).
输出 JSON: {"lens":"limitations","findings":[{"claim":"...","paper_idx":[2,4],"confidence":"high"}],"summary":"一句话总结局限性整体情况"}""",
    "reproduction": """你是可复现性评估员. 从给定论文里抽出可复现性相关的关键论断 (claim).
关注: 是否给完整参数/数据是否公开/代码是否公开/随机种子/硬件依赖/复现成本.
每条 claim 标注: 哪些论文 (paper_idx) 涉及它, 你的置信度 (high/medium/low).
输出 JSON: {"lens":"reproduction","findings":[{"claim":"...","paper_idx":[1,2],"confidence":"medium"}],"summary":"一句话总结可复现性整体情况"}""",
    "citation_context": """你是引用语境分析员. 从给定论文里抽出引用定位相关的关键论断 (claim).
关注: 本文相对前人工作的定位/争议点/学派归属/与主流的异同.
每条 claim 标注: 哪些论文 (paper_idx) 体现它, 你的置信度 (high/medium/low).
输出 JSON: {"lens":"citation_context","findings":[{"claim":"...","paper_idx":[1,3],"confidence":"medium"}],"summary":"一句话总结引用语境整体情况"}""",
    "temporal": """你是时间脉络定位员. 从给定论文里抽出时间演化相关的关键论断 (claim).
关注: 领域的发展轨迹/转折点/当前热点/未来方向/方法迭代代际.
每条 claim 标注: 哪些论文 (paper_idx) 体现它, 你的置信度 (high/medium/low).
输出 JSON: {"lens":"temporal","findings":[{"claim":"...","paper_idx":[1,2,3],"confidence":"medium"}],"summary":"一句话总结时间脉络整体情况"}""",
}

_DEFAULT_LENSES: list[str] = [
    "methodology",
    "contributions",
    "limitations",
    "reproduction",
    "citation_context",
    "temporal",
]


# ───────────────────────── Input schema ─────────────────────────


class LiteratureInput(BaseModel):
    action: Literal[
        "search",
        "summarize",
        "benchmark_lookup",
        "fetch_pdf",
        "citations",
        "ingest_to_rag",
        "crawl_web",
        "citation_graph",
        "extract_figures",
        "multi_review",
    ] = Field(
        ...,
        description="search/summarize/benchmark_lookup (第一期) + "
        "fetch_pdf/citations/ingest_to_rag (第二期) + "
        "crawl_web (第四期, 爬虫补无API的源) + "
        "citation_graph (BFS 引文图) + "
        "extract_figures (从PDF提图调image_analysis) + "
        "multi_review (N 路并行透镜 + 三重验证, nuwa/cangjie 启发)",
    )
    query: str = Field(default="", description="搜索/综述 query")
    max_results: int = Field(
        default=10, ge=1, le=50, description="每个源最多取几条 (多路并发后去重)"
    )
    expand_query: bool = Field(
        default=True,
        description="search 前用 LLM 把 query 拆成 2-3 个子查询 (打核心学术源) 扩大召回. "
        "失败自动降级为不扩展. False=只用原始 query.",
    )
    sources: list[str] = Field(
        default_factory=lambda: [
            "arxiv",
            "s2",
            "crossref",
            "openalex",
            "pubmed",
            "doaj",
            "core",
            "europepmc",
            "zenodo",
            "openaire",
            "cod",
            "materials_cloud",
            "nomad",
            "datacite",
            "materials_project",
        ],
        description="搜索源, 默认十五路全开. 可减到 ['arxiv'] 单源. "
        "学术文献: arxiv/s2/crossref/openalex/pubmed/doaj/core/"
        "europepmc/zenodo/openaire; "
        "材料数据库: cod/materials_cloud/nomad/materials_project; "
        "数据集: datacite. "
        "materials_project 需设 HUGINN_MP_API_KEY",
    )
    year_from: int | None = Field(default=None, description="年份下限 (含)")
    year_to: int | None = Field(default=None, description="年份上限 (含)")
    min_citations: int | None = Field(
        default=None,
        ge=0,
        description="引用数下限. 搜到的结果里引用数已知且 < 该值的会被过滤掉. "
        "未提供引用数的源 (arXiv/CORE 等) 不受影响. 默认不过滤.",
    )
    oa_only: bool = Field(
        default=False,
        description="True 时只返回开放获取 (可下载全文) 的论文. "
        "各源的 OA 信号统一归一: is_oa/open_access/oa_url/download_url. 默认关.",
    )

    # summarize 专用: 可以直接喂 papers 跳过 search
    papers: list[dict[str, Any]] | None = Field(
        default=None,
        description="summarize/ingest_to_rag 时可直接传 paper 列表 (从上一次 search 拿)",
    )
    focus: str | None = Field(
        default=None,
        description="summarize 时的重点, 如 'doping efficiency' 或 'low-temp stability'",
    )

    # benchmark_lookup 专用
    system: str | None = Field(
        default=None, description="体系名, 如 'LJ_13 cluster' 或 'GaN wurtzite'"
    )
    property: str | None = Field(
        default=None, description="性质名, 如 'minimum energy' 或 'band gap'"
    )

    # 第二期: fetch_pdf / citations 单篇论文输入
    paper: dict[str, Any] | None = Field(
        default=None,
        description="fetch_pdf/citations 时传单篇 paper dict (从 search 拿), 含 url/doi 等",
    )
    arxiv_id: str | None = Field(
        default=None,
        description="直接给 arxiv id (如 '1911.08365' 或 'cond-mat/9909087')",
    )
    doi: str | None = Field(
        default=None,
        description="直接给 DOI (fetch_pdf 走 Unpaywall 找 OA 版; citations 走 S2 lookup)",
    )
    url: str | None = Field(
        default=None, description="直接给 PDF URL (fetch_pdf 直接下载)"
    )

    # fetch_pdf 专用
    max_chars: int = Field(
        default=50000,
        ge=1000,
        le=500000,
        description="全文截断长度, 默认 50k 字 (~12k token). 太长撑爆 LLM context",
    )

    # citations 专用
    direction: Literal["forward", "backward", "both"] = Field(
        default="both",
        description="forward=谁引了这篇; backward=这篇引了谁; both=都要. "
        "citations/citation_graph 共用. citation_graph 默认 both (双向 snowballing).",
    )
    max_citations: int = Field(
        default=20, ge=1, le=100, description="每个方向最多取几条引用"
    )

    # citation_graph 专用: BFS 深度 + 节点上限
    max_depth: int = Field(
        default=2,
        ge=1,
        le=3,
        description="citation_graph BFS 深度 (1=只取直接引用, 2/3=多跳)",
    )
    max_nodes: int = Field(
        default=50,
        ge=5,
        le=200,
        description="citation_graph 最多收集多少个节点, 防止跑飞",
    )

    # 第四期: crawl_web 专用
    engine: Literal["google_scholar", "google_patents", "duckduckgo", "direct"] = Field(
        default="direct",
        description="crawl_web 搜索引擎: google_scholar/google_patents/duckduckgo "
        "(需配 query); direct=直接爬 url",
    )

    # 第五期: 订阅源认证 (高校非 OA 期刊)
    auth_action: Literal["login", "status", "logout"] | None = Field(
        default=None,
        description="crawl_web 认证动作: login=弹浏览器手动登录存 profile, "
        "status=列出已存 session, logout=删 profile",
    )
    provider: str | None = Field(
        default=None,
        description="订阅源 provider: cnki/wanfang/cqvip/elsevier/springer/ieee/"
        "wiley/acs/rsc/nature/wos/tandfonline. 配 auth_action 用",
    )

    # multi_review 专用: N 路并行透镜 (nuwa 启发) + 三重验证 (cangjie 启发)
    lenses: list[str] | None = Field(
        default=None,
        description="multi_review 透镜列表. None=默认 6 透镜全开. "
        "可选: methodology/contributions/limitations/reproduction/"
        "citation_context/temporal. 可子集.",
    )
    verify_claims: bool = Field(
        default=True,
        description="multi_review 是否做三重验证 (V1 跨域复现/V2 生成力/V3 排他性). "
        "False=只跑透镜不验证, 适合快速扫描.",
    )

    model_config = {"protected_namespaces": ()}


# ───────────────────────── Tool ─────────────────────────


@dataclass
class _CitationGraphState:
    """Mutable BFS state for ``_do_citation_graph``."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, str]]
    visited: set[str]
    errors: list[str]
    source_mode: str = "s2"  # 默认用 S2, 限速后切 "opencitations"
    depth_reached: int = 0


class LiteratureTool(HuginnTool):
    """学术文献+材料数据库调研工具. 15 路并发搜索 (arXiv/S2/CrossRef/OpenAlex/PubMed/
    DOAJ/CORE/EuropePMC/Zenodo/OpenAIRE/COD/MaterialsCloud/NOMAD/DataCite/MaterialsProject),
    LLM 综述, 文献基准对比, OA PDF 全文抓取, 引用网络查询, RAG 入库."""

    name = "literature_tool"
    category = "search"
    description = (
        "Search 15 sources in parallel: literature (arXiv/S2/CrossRef/OpenAlex/PubMed/"
        "DOAJ/CORE/EuropePMC/Zenodo/OpenAIRE), materials databases (COD/Materials Cloud/"
        "NOMAD 19M+ entries/Materials Project), and datasets (DataCite). "
        "Generate multi-paper summaries with citations, look up literature-reported "
        "values for a given system+property (complements validate_tool's built-in benchmarks), "
        "fetch OA PDF full text (multi-source: OpenAlex/Unpaywall/Europe PMC/arXiv), "
        "query citation networks, and ingest papers into local RAG. "
        "Use this BEFORE running calculations to find known values, or AFTER to compare."
    )
    input_schema = LiteratureInput
    read_only = True

    async def call(self, args: LiteratureInput, context: ToolContext) -> ToolResult:
        # 兼容 dict 入参, 让 hypothesis_generator_tool._invoke_tool 那种传 dict 的路径也能用
        if isinstance(args, dict):
            args = LiteratureInput(**args)
        if _disabled():
            return ToolResult(
                data={"error": "literature_tool disabled", "hint": _DISABLED_HINT},
                success=False,
                error=_DISABLED_HINT,
            )
        try:
            if args.action == "search":
                return await self._do_search(args, context)
            if args.action == "summarize":
                return await self._do_summarize(args, context)
            if args.action == "benchmark_lookup":
                return await self._do_benchmark_lookup(args, context)
            if args.action == "fetch_pdf":
                return await self._do_fetch_pdf(args)
            if args.action == "citations":
                return await self._do_citations(args)
            if args.action == "citation_graph":
                return await self._do_citation_graph(args)
            if args.action == "ingest_to_rag":
                return await self._do_ingest_to_rag(args, context)
            if args.action == "crawl_web":
                return await self._do_crawl_web(args, context)
            if args.action == "extract_figures":
                return await self._do_extract_figures(args, context)
            if args.action == "multi_review":
                return await self._do_multi_review(args, context)
            return ToolResult(
                data=None, success=False, error=f"unknown action: {args.action}"
            )
        except Exception as exc:
            logger.exception("literature_tool %s failed", args.action)
            return ToolResult(data=None, success=False, error=str(exc))

    # ── search ──────────────────────────────────────────────

    @staticmethod
    def _search_tasks(
        args: LiteratureInput, query: str, subqueries: list[str]
    ) -> list[tuple[str, asyncio.Task]]:
        """按 sources 建并发任务; 子查询只打核心学术源, 结果数减半."""
        tasks: list[tuple[str, asyncio.Task]] = []
        for src in args.sources:
            fn = _SOURCE_FNS.get(src)
            if fn is None:
                continue
            tasks.append(
                (
                    src,
                    asyncio.create_task(
                        fn(query, args.max_results, args.year_from, args.year_to)
                    ),
                )
            )

        # 子查询只打核心学术源, 结果数减半 — 扩召回而不把 API 调用量翻倍
        if subqueries:
            sub_max = max(2, args.max_results // 2)
            for sq in subqueries:
                for src, fn in _SUBQUERY_SOURCES.items():
                    if src not in args.sources:
                        continue
                    tasks.append(
                        (
                            f"{src}@{sq[:24]}",
                            asyncio.create_task(
                                fn(sq, sub_max, args.year_from, args.year_to)
                            ),
                        )
                    )
        return tasks

    @staticmethod
    async def _collect_search_results(
        tasks: list[tuple[str, asyncio.Task]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """多路并发, 任一失败不阻塞其他."""
        results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)
        all_papers: list[dict[str, Any]] = []
        source_status: dict[str, Any] = {}
        for (src, _), res in zip(tasks, results):
            if isinstance(res, Exception):
                source_status[src] = {"ok": False, "error": str(res)[:200]}
                logger.warning("source %s failed: %s", src, res)
            else:
                source_status[src] = {"ok": True, "count": len(res)}
                all_papers.extend(res)
        return all_papers, source_status

    @staticmethod
    def _search_result(
        query: str,
        subqueries: list[str],
        ranked: list[dict[str, Any]],
        tasks: list[tuple[str, asyncio.Task]],
        source_status: dict[str, Any],
    ) -> ToolResult:
        """把检索结果包装成 ToolResult."""
        return ToolResult(
            data={
                "action": "search",
                "query": query,
                "subqueries": subqueries,
                "total": len(ranked),
                "papers": ranked,
                "sources_tried": [s for s, _ in tasks],
                "source_status": source_status,
            },
            success=True,
        )

    async def _search_subqueries(
        self,
        args: LiteratureInput,
        context: ToolContext | None,
        query: str,
    ) -> list[str]:
        """P0-1: LLM 把 query 拆成子查询 (Q 端增强).

        需要 context 取 model, 失败降级为不扩展 — 搜索绝不能被增强逻辑阻塞.
        """
        if not (args.expand_query and context is not None):
            return []
        try:
            model = self._get_model(context)
        except Exception as exc:
            logger.debug("query expansion skipped (no model): %s", exc)
            return []
        if model is None:
            return []
        return await _expand_query(model, query)

    async def _do_search(
        self, args: LiteratureInput, context: ToolContext | None = None
    ) -> ToolResult:
        # 离线/CI 环境下不得发起真实网络检索. `call()` 已在入口拦一次, 但本函数
        # 也被 _do_summarize / benchmark_lookup 的缺度补全 (_do_complement_retrieval)
        # 等内部路径直接调用 (experimental 的 run_real_body 便绕过 `call()`),
        # 故在 15 源扇出前再判一次, 杜绝"声明离线却静默联网".
        if _disabled():
            return ToolResult(
                data={"error": "literature_tool disabled", "hint": _DISABLED_HINT},
                success=False,
                error=_DISABLED_HINT,
            )
        query = (args.query or "").strip()
        if not query:
            return ToolResult(
                data=None, success=False, error="query is required for search"
            )

        subqueries = await self._search_subqueries(args, context, query)

        # P1-4: TTL 缓存. key 含子查询, 不同扩展不同条目.
        cache_key = _search_cache_key(args, tuple(subqueries))
        cached = _SEARCH_CACHE.get(cache_key)
        if cached is not None:
            return cached

        tasks = self._search_tasks(args, query, subqueries)
        if not tasks:
            return ToolResult(
                data=None,
                success=False,
                error=f"no valid sources in {args.sources}",
            )

        all_papers, source_status = await self._collect_search_results(tasks)

        deduped = _dedup(all_papers)
        # L1b: 引用数下限 + OA 优先过滤 (跨源归一)
        filtered = _apply_filters(
            deduped,
            min_citations=args.min_citations,
            oa_only=args.oa_only,
        )
        # P0-2: query 相关度重排 (替代纯 citation 排序), OA 在同等相关度下靠前
        ranked = _rerank(query, filtered)[: args.max_results * 2]

        result = self._search_result(query, subqueries, ranked, tasks, source_status)
        _SEARCH_CACHE.set(cache_key, result)
        return result

    # ── summarize ───────────────────────────────────────────

    @staticmethod
    def _summary_paper_block(papers: list[dict[str, Any]]) -> str:
        """构造论文清单给 LLM — P0-3: 优先喂全文 (fetch_pdf 拿到的), 退回 abstract."""
        paper_block_parts: list[str] = []
        for i, p in enumerate(papers, 1):
            authors_short = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors_short += " et al."
            year = p.get("year") or ""
            venue = p.get("venue") or ""
            body = (p.get("full_text") or p.get("abstract") or "").strip()
            label = "Full text" if p.get("full_text") else "Abstract"
            if len(body) > 3000:
                body = body[:3000] + "..."
            paper_block_parts.append(
                f"[{i}] {p.get('title', '')}\n"
                f"  Authors: {authors_short}\n"
                f"  Year: {year}  Venue: {venue}  DOI: {p.get('doi') or '-'}\n"
                f"  {label}: {body}"
            )
        return "\n\n".join(paper_block_parts)

    @staticmethod
    def _summary_result(
        args: LiteratureInput,
        content: str,
        bibtex: str,
        papers: list[dict[str, Any]],
    ) -> ToolResult:
        """把综述结果包装成 ToolResult."""
        return ToolResult(
            data={
                "action": "summarize",
                "query": args.query or "",
                "focus": args.focus,
                "n_papers": len(papers),
                "summary_markdown": content,
                "bibtex": bibtex,
                "papers": [
                    {
                        "idx": i + 1,
                        "title": p.get("title", ""),
                        "doi": p.get("doi"),
                        "year": p.get("year"),
                    }
                    for i, p in enumerate(papers)
                ],
            },
            success=True,
        )

    async def _do_summarize(
        self, args: LiteratureInput, context: ToolContext
    ) -> ToolResult:
        papers = args.papers
        if not papers and args.query:
            search_res = await self._do_search(args, context)
            if not search_res.success:
                return search_res
            papers = (search_res.data or {}).get("papers", [])
        if not papers or len(papers) == 0:
            return ToolResult(
                data=None,
                success=False,
                error="no papers to summarize (provide papers or a query)",
            )

        # 太多就截断, 喂 LLM 的 context 不能爆
        papers = papers[:15]
        focus_line = f"\n\n综述重点 (如果给的话): {args.focus}" if args.focus else ""

        user_prompt = (
            f"研究 query: {args.query or '(未指定)'}{focus_line}\n\n"
            f"论文列表 ({len(papers)} 篇):\n\n{self._summary_paper_block(papers)}\n\n"
            "请基于以上论文写结构化综述."
        )

        try:
            model = self._get_model(context)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"LLM 初始化失败: {exc}",
            )

        content = await invoke_llm_text(model, _SUMMARY_SYSTEM_PROMPT, user_prompt)

        # 顺带生成 bibtex, 方便用户直接拿去用
        bibtex = self._to_bibtex(papers)

        return self._summary_result(args, content, bibtex, papers)

    # ── benchmark_lookup ────────────────────────────────────

    async def _do_benchmark_lookup(
        self, args: LiteratureInput, context: ToolContext
    ) -> ToolResult:
        err = _bl_validate(args)
        if err is not None:
            return err

        # search query 优先用用户给的, 没给就拼 system+property
        query = args.query or f"{args.system} {args.property}"
        papers, err = await self._bl_collect_papers(args, query)
        if err is not None:
            return err
        if not papers:
            return _bl_empty_result(args, query)

        model, err = self._mr_get_model(context)
        if err is not None:
            return err

        reported = await self._bl_extract_reported(papers, args, model)
        consensus, spread = _bl_consensus(reported)
        consistency = _annotate_value_consistency(reported)
        compat, completion = await self._bl_completion(args, reported, model)
        kb_written = _bl_write_kb(args, reported)
        return _bl_build_result(
            args,
            query,
            papers,
            reported,
            consensus,
            spread,
            consistency,
            compat,
            completion,
            kb_written,
        )

    async def _bl_collect_papers(self, args: LiteratureInput, query: str):
        """拿 papers: 直接传了就用传的, 否则 search. 返回 (papers, error)."""
        # 如果直接传了 papers (可能带 full_text), 就用传的, 不重新 search.
        # 典型场景: 先 fetch_pdf 拿全文, 再喂给 benchmark_lookup 抽精确值
        if args.papers:
            return args.papers, None
        search_res = await self._do_search(
            LiteratureInput(
                action="search",
                query=query,
                max_results=max(args.max_results, 15),
                sources=args.sources,
                year_from=args.year_from,
                year_to=args.year_to,
            )
        )
        if not search_res.success:
            return None, search_res
        return (search_res.data or {}).get("papers", []), None

    async def _bl_extract_reported(self, papers, args: LiteratureInput, model):
        """抽取报道值 + 对象级取证."""
        # 抽取报道值 — 复用 _llm_extract_reported (缺度补全检索的二次抽取也走它)
        reported = await self._llm_extract_reported(
            papers,
            args.system,
            args.property,
            model,
        )
        # 对象级取证 (#1): 给每条值绑定来源论文的强引用证据标签 (fid + sha256 快照),
        # 让 "这条值来自哪段原文" 随结果流动、可独立核实, 而非事后声明.
        try:
            from .completion_evidence import attach_evidence

            reported = attach_evidence(reported, papers)
        except Exception:
            logger.debug("attach_evidence skipped (non-fatal)", exc_info=True)
        return reported

    @staticmethod
    def _bl_completion_entry(
        compat: dict[str, Any],
        known: dict[str, str],
        missing_dims: list[Any],
        summarize_missing: Any,
        followup: dict[str, Any],
    ) -> dict[str, Any]:
        """组装缺度追问诊断字典."""
        return {
            "missing_dims": missing_dims,
            "missing_summaries": summarize_missing(missing_dims),
            "known_conditions": known,
            "local": compat["local"],
            "global_compat": compat["global_compat"],
            "flat_confounding": compat["flat_confounding"],
            "overall_verdict": compat["overall_verdict"],
            "followup": followup,
        }

    async def _bl_completion(self, args: LiteratureInput, reported, model):
        """缺度追问 (对接1) + 实际补全检索 (对接2). 返回 (compat, completion)."""
        compat = None
        completion = None
        if not reported:
            return compat, completion

        from huginn.experimental.local_global_compat import run_compat_experiment

        from .query_completion import build_followup_input, summarize_missing

        try:
            known = _infer_known_conditions(reported)
            compat = run_compat_experiment(reported)
            missing_dims = compat.get("missing_dims", [])
            unit = (reported[0].get("unit") or "") if reported else ""
            followup = build_followup_input(
                args.system,
                args.property,
                unit,
                missing_dims,
                method_family=known.get("method_family"),
                functional=known.get("functional"),
                temperature=known.get("temperature"),
            )
            completion = self._bl_completion_entry(
                compat, known, missing_dims, summarize_missing, followup
            )
            # 对接2: 只对"应发起且关键"的补全查询跑实际检索
            if followup.get("fills"):
                completion[
                    "complement_retrieval"
                ] = await self._do_complement_retrieval(
                    reported,
                    compat,
                    followup,
                    args.system,
                    args.property,
                    model,
                    args,
                )
        except Exception as exc:
            logger.warning("benchmark_lookup 缺度追问失败: %s", exc)
        return compat, completion

    # ── 缺度追问: 抽取 + 实际补全检索 (对接1/对接2) ─────────────

    def _decision_ledger(self) -> Any:
        """返回缺度豁免决策档 (可注入/可 mock).

        默认写到系统临时目录下的 decisions.jsonl, 不绑定全局单例, 便于测试
        与隔离; 子类/测试可覆写此方法注入自定义路径或 mock.
        """
        import tempfile
        from pathlib import Path

        from .completion_evidence import DecisionLedger

        return DecisionLedger(
            str(getattr(self, "_decision_log_path", "") or "")
            or str(Path(tempfile.gettempdir()) / "huginn_completion_decisions.jsonl")
        )

    @staticmethod
    def _extract_paper_block(papers: list[dict[str, Any]]) -> str:
        """抽值用的论文块 (最多 10 篇, 正文截 4000 字)."""
        paper_block_parts: list[str] = []
        for i, p in enumerate(papers[:10], 1):
            text = (p.get("full_text") or p.get("abstract") or "").strip()
            # full_text 可能几万字, 截到 4000 字 (~1000 token) 够 LLM 抽数值
            if len(text) > 4000:
                text = text[:4000] + "..."
            label = "Full text" if p.get("full_text") else "Abstract"
            paper_block_parts.append(f"[{i}] {p.get('title', '')}\n  {label}: {text}")
        return "\n\n".join(paper_block_parts)

    @staticmethod
    def _map_extracted_values(
        raw_values: list[Any], papers: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """把 paper_idx 映射回真实 paper 信息, 丢弃不可解析的条目."""
        reported: list[dict[str, Any]] = []
        for v in raw_values:
            if not isinstance(v, dict):
                continue
            try:
                value = float(v.get("value"))
            except (TypeError, ValueError):
                continue
            idx = int(v.get("paper_idx", 0))
            if idx < 1 or idx > len(papers):
                continue
            paper = papers[idx - 1]
            reported.append(
                {
                    "value": value,
                    "unit": str(v.get("unit", "")),
                    "method": str(v.get("method", "")),
                    "note": str(v.get("note", "")),
                    "source_paper": paper.get("title", ""),
                    "doi": paper.get("doi"),
                    "year": paper.get("year"),
                    "venue": paper.get("venue", ""),
                }
            )
        return reported

    async def _llm_extract_reported(
        self,
        papers: list[dict[str, Any]],
        system: str,
        property: str,
        model: Any,
    ) -> list[dict[str, Any]]:
        """从论文列表抽报道值: 组装 paper_block → LLM → 映射回真实 paper.

        benchmark_lookup 与缺度补全检索共用同一管线, 保证两轮抽取一致.
        无文本论文 / LLM 失败 → 返回 [].
        """
        papers_with_text = [
            p for p in papers if p.get("full_text") or p.get("abstract")
        ]
        if not papers_with_text:
            return []
        user_prompt = (
            f"体系: {system}\n"
            f"性质: {property}\n\n"
            f"论文列表 ({len(papers_with_text[:10])} 篇, 含全文或 abstract):\n\n"
            f"{self._extract_paper_block(papers_with_text)}\n\n"
            f"请抽出关于 {system} 的 {property} 报道值."
        )
        try:
            content = await invoke_llm_text(
                model, _BENCHMARK_SYSTEM_PROMPT, user_prompt
            )
        except Exception as exc:
            logger.debug("benchmark value extraction failed: %s", exc)
            return []
        parsed = self._parse_json(content)
        raw_values = parsed.get("values", []) if parsed else []

        return self._map_extracted_values(raw_values, papers_with_text[:10])

    async def _comp_search(
        self, args: LiteratureInput, query: str
    ) -> tuple[list[dict[str, Any]], str | None]:
        """跑补全检索, 返回 (papers, error); 失败降级为诊断文本."""
        try:
            search_res = await self._do_search(
                LiteratureInput(
                    action="search",
                    query=query,
                    max_results=max(args.max_results, 10),
                    sources=args.sources,
                    year_from=args.year_from,
                    year_to=args.year_to,
                )
            )
        except Exception as exc:  # pragma: no cover - 网络层异常降级
            return [], f"补全检索失败: {exc}"
        if not search_res.success:
            return [], search_res.error or "补全检索失败"
        return (search_res.data or {}).get("papers", []), None

    @staticmethod
    def _comp_evidence(
        new_reported: list[dict[str, Any]], new_papers: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """补全轮同样做对象级取证 (#1): 给补新值绑定证据标签."""
        try:
            from .completion_evidence import attach_evidence

            return attach_evidence(new_reported, new_papers)
        except Exception:
            logger.debug(
                "complement attach_evidence skipped (non-fatal)", exc_info=True
            )
            return new_reported

    def _comp_gate(
        self,
        missing_after: list[Any],
        system: str,
        property: str,
        revised: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, bool | None]:
        """补后仍缺的关键自由度落 waiver 决策档 + 门禁判定.

        显式豁免 + 决策档 (#2) + 门禁不变量 (#3): 补后仍缺的关键自由度
        (urgency>=2, 影响整体互洽判定) 不再 silent 接受, 落一条 waiver
        决策档 (可复核), 再由 assess_gate 判能否放行.
        """
        if not missing_after:
            return [], None, None

        from .completion_evidence import assess_gate, build_waivers, critical_missing
        from .query_completion import _URGENCY

        blocking_dims = critical_missing(missing_after, _URGENCY)
        waivers = build_waivers(
            blocking_dims, reason_var=f"system={system},property={property}"
        )
        waived_dims = [w["dim"] for w in waivers]
        gate = assess_gate(
            blocking_dims, waived_dims, verdict=revised["overall_verdict"]
        )
        # 落决策档 (best-effort, 失败不阻塞主结果). DecisionLedger/路径均可
        # 由测试注入; 默认写到系统临时目录, 不绑定全局单例.
        decision_log_verified: bool | None = None
        try:
            ledger = self._decision_ledger()
            for w in waivers:
                ledger.append(w)
            decision_log_verified, decision_log_problems = ledger.verify()
            gate["decision_log_path"] = str(getattr(ledger, "_path", ""))
            gate["decision_log_verified"] = bool(decision_log_verified)
            if decision_log_problems:
                gate["decision_log_problems"] = decision_log_problems
        except Exception:
            logger.debug(
                "waiver decision ledger write skipped (non-fatal)", exc_info=True
            )
        return waivers, gate, decision_log_verified

    async def _comp_extract(
        self,
        new_papers: list[dict[str, Any]],
        system: str,
        property: str,
        model: Any,
    ) -> list[dict[str, Any]]:
        """对补全论文再抽一轮报道值, 并绑定对象级证据."""
        if not new_papers:
            return []
        reported = await self._llm_extract_reported(new_papers, system, property, model)
        return self._comp_evidence(reported, new_papers)

    @staticmethod
    def _comp_verdicts(
        compat: dict[str, Any],
        revised: dict[str, Any],
        revised_consistency: dict[str, Any],
        missing_after: list[Any],
        n_total: int,
    ) -> dict[str, Any]:
        """补前/补后的缺失自由度与判定对比."""
        return {
            "n_total_sources": n_total,
            "missing_dims_before": compat.get("missing_dims", []),
            "missing_dims_after": missing_after,
            "overall_verdict_before": compat["overall_verdict"],
            "overall_verdict_after": revised["overall_verdict"],
            "consistency_before": compat["flat_confounding"]["overall"]["verdict"],
            "consistency_after": revised_consistency["overall"]["verdict"],
        }

    @staticmethod
    def _comp_empty_result(query: str, n_new_papers: int) -> dict[str, Any]:
        """补全检索跑通但没抽到新值的返回."""
        return {
            "executed": True,
            "query": query,
            "n_new_papers": n_new_papers,
            "new_reported": [],
            "message": "补全检索没抽到新的报道值",
        }

    async def _comp_recheck(
        self, merged: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """合并两轮后重跑局部-整体 + 一致性."""
        from huginn.experimental.local_global_compat import run_compat_experiment

        return run_compat_experiment(merged), _annotate_value_consistency(merged)

    async def _do_complement_retrieval(
        self,
        reported: list[dict[str, Any]],
        compat: dict[str, Any],
        followup: dict[str, Any],
        system: str,
        property: str,
        model: Any,
        args: LiteratureInput,
    ) -> dict[str, Any]:
        """缺度追问的实际补全检索执行 (对接2).

        followup 里"应检索且关键"的补全查询丢给 _do_search, 拿回新论文再抽一轮,
        合并后重跑补后校验并报告变化; 任何失败降级返回诊断.
        """
        fills = followup.get("fills") or []
        query = fills[0]["query"] if fills else ""
        if not query:
            return {"executed": False, "reason": "无满足条件的补全查询"}
        new_papers, error = await self._comp_search(args, query)
        if error is not None:
            return {"executed": False, "reason": error}

        new_reported = await self._comp_extract(new_papers, system, property, model)
        if not new_reported:
            return self._comp_empty_result(query, len(new_papers))

        # 补后校验: 合并两轮, 重跑局部-整体 + 一致性
        merged = reported + new_reported
        revised, revised_consistency = await self._comp_recheck(merged)
        missing_after = revised.get("missing_dims", [])
        waivers, gate, decision_log_verified = self._comp_gate(
            missing_after, system, property, revised
        )

        return {
            "executed": True,
            "query": query,
            "n_new_papers": len(new_papers),
            "new_reported": new_reported,
            "waivers": waivers,
            "gate": gate,
            "decision_log_verified": decision_log_verified,
            "revised": revised,
            **self._comp_verdicts(
                compat, revised, revised_consistency, missing_after, len(merged)
            ),
        }

    # ── fetch_pdf ───────────────────────────────────────────

    @staticmethod
    def _pdf_error_result(tried: list[dict[str, str]]) -> ToolResult:
        """所有候选源失败时的诊断结果."""
        last_status = tried[-1]["status"] if tried else ""
        err_msg = f"所有 PDF 候选源都失败 (试了 {len(tried)} 个). 最后错误: {last_status[:200]}"
        if any(
            "10060" in t["status"] or "timeout" in t["status"].lower() for t in tried
        ):
            err_msg += " | 多个源超时, 试试配 HTTPS_PROXY 环境变量走代理"
        err_msg += f" | tried: {tried}"
        return ToolResult(data=None, success=False, error=err_msg)

    @staticmethod
    def _pdf_pages_text(pdf_bytes: bytes) -> tuple[list[str] | None, str | None]:
        """PyMuPDF 抽逐页正文; 返回 (pages, error)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None, "PyMuPDF (fitz) 未安装, 无法抽 PDF 正文. pip install pymupdf"
        try:
            # 用 with 保证异常路径也释放文件句柄
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                return [page.get_text() for page in doc], None
        except Exception as exc:
            return None, f"PDF 解析失败: {exc}"

    def _pdf_result(
        self,
        args: LiteratureInput,
        used_url: str,
        tried: list[dict[str, str]],
        pages_text: list[str],
        truncated: bool,
        sections: Any,
        full_text: str,
    ) -> ToolResult:
        """把抽好的正文 + 元数据包装成 ToolResult."""
        return ToolResult(
            data={
                "action": "fetch_pdf",
                "pdf_url": used_url,
                "n_pages": len(pages_text),
                "n_chars": len("\n\n".join(pages_text)),
                "truncated": truncated,
                "n_sections": len(sections),
                "sections": sections,
                "full_text": full_text,
                "candidates_tried": tried,
                # P0-3: 可直接喂回 summarize/ingest_to_rag 的 paper dict —
                # 打通"抓全文 → 综述/入库用全文"的链路
                "paper": {
                    "title": (args.paper or {}).get("title", ""),
                    "authors": (args.paper or {}).get("authors", []),
                    "year": (args.paper or {}).get("year"),
                    "venue": (args.paper or {}).get("venue", ""),
                    "doi": (args.paper or {}).get("doi") or args.doi,
                    "url": used_url,
                    "full_text": full_text,
                    "sections": sections,
                },
            },
            success=True,
        )

    async def _do_fetch_pdf(self, args: LiteratureInput) -> ToolResult:
        """下载 OA PDF + PyMuPDF 抽正文 + 分节. 解决 benchmark_lookup
        从 abstract 抽不到数值的问题 (精确值通常在论文正文表里).

        多源候选: OpenAlex oa_url → Unpaywall → Europe PMC → arxiv.org/pdf.
        arxiv.org/pdf 在国内常超时, 放最后兜底, 前面的 OA 源先试.
        """
        pdf_bytes, used_url, tried = await self._download_pdf_bytes(args)
        if pdf_bytes is None:
            if not tried:
                return ToolResult(
                    data=None,
                    success=False,
                    error="无法解析 PDF URL, 需要 arxiv_id/doi/url 或 paper dict (含 url/doi/oa_url)",
                )
            return self._pdf_error_result(tried)

        pages_text, error = self._pdf_pages_text(pdf_bytes)
        if pages_text is None:
            return ToolResult(data=None, success=False, error=error)

        full_text = "\n\n".join(pages_text)
        n_chars = len(full_text)
        sections = split_sections(full_text)
        truncated = False
        if n_chars > args.max_chars:
            full_text = full_text[: args.max_chars]
            truncated = True

        return self._pdf_result(
            args, used_url, tried, pages_text, truncated, sections, full_text
        )

    @staticmethod
    def _pdf_paper_candidates(
        args: LiteratureInput,
    ) -> tuple[list[str], str | None, str | None]:
        """从 paper dict 提取候选 URL / doi / arxiv_id."""
        candidates: list[str] = []
        doi: str | None = args.doi
        arxiv_id: str | None = args.arxiv_id
        if not args.paper:
            return candidates, doi, arxiv_id

        p = args.paper
        p_url = p.get("url", "") or ""
        p_doi = p.get("doi")
        if p_doi and not doi:
            doi = p_doi
        # paper dict 自带的 oa_url (OpenAlex 搜出来会有)
        oa_url = p.get("oa_url") or ""
        if oa_url and (oa_url.endswith(".pdf") or "arxiv.org" not in oa_url):
            candidates.append(oa_url)
        # CORE 直接给 download_url (全文 PDF 链接), 优先级高
        core_dl = p.get("download_url") or ""
        if core_dl and core_dl not in candidates:
            candidates.append(core_dl)
        # paper.url 本身就是 .pdf
        if p_url and p_url.endswith(".pdf") and p_url not in candidates:
            candidates.append(p_url)
        # 从 arxiv abs url 提 arxiv_id (放后面拼 arxiv pdf)
        if not arxiv_id and "arxiv.org/abs/" in p_url:
            arxiv_id = p_url.split("/abs/")[-1].strip("/")
        elif not arxiv_id and "arxiv.org/pdf/" in p_url:
            aid = p_url.split("/pdf/")[-1].strip("/")
            arxiv_id = aid[:-4] if aid.endswith(".pdf") else aid
        return candidates, doi, arxiv_id

    async def _pdf_lookup_candidates(
        self, candidates: list[str], doi: str | None, arxiv_id: str | None
    ) -> list[str]:
        """按 DOI/arxiv_id 补候选: OpenAlex → Unpaywall → Europe PMC → arxiv → Sci-Hub."""
        # OpenAlex 按 DOI 查 oa_url (OpenAlex 在国内一般可达)
        if doi and not any("openalex" in c for c in candidates):
            oa = await openalex_oa_url(doi)
            if oa:
                candidates.append(oa)

        # Unpaywall
        if doi:
            upw = await unpaywall_pdf(doi)
            if upw and upw not in candidates:
                candidates.append(upw)

        # Europe PMC (biomedical 为主, 但也覆盖一些材料/化学)
        if doi:
            epmc = await europepmc_pdf(doi)
            if epmc and epmc not in candidates:
                candidates.append(epmc)

        # arxiv.org/pdf 放最后 (国内常超时)
        if arxiv_id:
            aid = arxiv_id.strip().strip("/")
            aid = aid[:-4] if aid.endswith(".pdf") else aid
            arxiv_pdf = f"https://arxiv.org/pdf/{aid}.pdf"
            if arxiv_pdf not in candidates:
                candidates.append(arxiv_pdf)

        # Sci-Hub: 仅 HUGINN_ENABLE_SCIHUB=1 时启用, 最后兜底
        # 法律灰色地带: Sci-Hub 托管版权论文未获出版商授权, 用户需自行承担合规风险.
        if doi and _scihub_enabled():
            sh_url = await _scihub_pdf_url(doi)
            if sh_url and sh_url not in candidates:
                candidates.append(sh_url)
        return candidates

    @staticmethod
    def _dedupe_urls(candidates: list[str]) -> list[str]:
        """去重保序."""
        seen: set[str] = set()
        deduped: list[str] = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                deduped.append(c)
        return deduped

    async def _resolve_pdf_candidates(self, args: LiteratureInput) -> list[str]:
        """多源解析 PDF URL, 返回候选列表 (优先级高的在前).

        顺序考虑网络可达性 (国内环境):
          1. 直接给的 .pdf url / paper.url 是 .pdf
          2. paper.oa_url (OpenAlex 搜出来的直接给)
          3. OpenAlex 按 DOI 查 oa_url
          4. Unpaywall (DOI → OA PDF)
          5. Europe PMC (DOI → fullTextUrl)
          6. arxiv.org/pdf (arxiv abs url 或 arxiv_id) — 放最后, 国内常超时
          7. Sci-Hub (仅 HUGINN_ENABLE_SCIHUB=1 时启用, 最后兜底, 法律灰色地带)
        """
        candidates, doi, arxiv_id = self._pdf_paper_candidates(args)

        # 直接给的 url 最优先
        if args.url and args.url not in candidates:
            candidates.insert(0, args.url)

        candidates = await self._pdf_lookup_candidates(candidates, doi, arxiv_id)
        return self._dedupe_urls(candidates)

    async def _resolve_pdf_url(self, args: LiteratureInput) -> str | None:
        """单 URL 解析 (向后兼容). 返回第一个候选."""
        cands = await self._resolve_pdf_candidates(args)
        return cands[0] if cands else None

    async def _download_pdf_bytes(
        self, args: LiteratureInput
    ) -> tuple[bytes | None, str, list[dict[str, str]]]:
        """多源下载 PDF, 返回 (pdf_bytes, used_url, tried).

        pdf_bytes 为 None 表示全部候选源失败, tried 列表含各源失败原因.
        _do_fetch_pdf 和 _do_extract_figures 共用这段下载逻辑.
        """
        candidates = await self._resolve_pdf_candidates(args)
        if not candidates:
            return None, "", []

        tried: list[dict[str, str]] = []
        pdf_bytes = b""
        used_url = ""
        for cand in candidates:
            try:
                pdf_bytes = await _http_get_bytes(cand)
                # magic bytes 校验: %PDF 开头且够大, OpenAlex 有时返回 HTML 冒充 PDF
                if len(pdf_bytes) < 1000:
                    tried.append(
                        {"url": cand, "status": f"too small ({len(pdf_bytes)} bytes)"}
                    )
                    continue
                if not pdf_bytes[:5].startswith(b"%PDF"):
                    tried.append(
                        {"url": cand, "status": f"not PDF (got {pdf_bytes[:20]!r})"}
                    )
                    continue
                used_url = cand
                break
            except Exception as exc:
                err_str = str(exc)
                tried.append({"url": cand, "status": err_str[:120]})
                logger.info("fetch_pdf 候选 %s 失败: %s", cand[:60], exc)

        if not used_url:
            return None, "", tried

        return pdf_bytes, used_url, tried

    # ── citations ───────────────────────────────────────────

    async def _forward_citations(
        self,
        result: dict[str, Any],
        paper_id: str | None,
        doi: str | None,
        args: LiteratureInput,
    ) -> None:
        """前向: 谁引了这篇. S2 优先, 挂了回退 OpenCitations."""
        s2_failed = False
        if paper_id:
            try:
                pid_enc = urllib.parse.quote(paper_id, safe="")
                data = await _http_get_json(
                    f"https://api.semanticscholar.org/graph/v1/paper/{pid_enc}/citations"
                    f"?fields=title,year,authors,citationCount,externalIds&limit={args.max_citations}"
                )
                for c in data.get("data", []) or []:
                    cp = c.get("citingPaper", {}) or {}
                    result["forward_citations"].append(self._s2_paper_to_dict(cp))
                result["sources_used"].append("s2")
            except Exception as exc:
                s2_failed = True
                result["forward_error_s2"] = str(exc)[:200]
                logger.warning("S2 citations (forward) 失败: %s", exc)

        # S2 挂了就回退 OpenCitations
        if s2_failed and doi:
            try:
                oc_cites = await _opencitations_citations(doi, args.max_citations)
                if oc_cites:
                    # CrossRef 补元数据
                    oc_cites = await _enrich_with_crossref(oc_cites)
                    result["forward_citations"].extend(oc_cites)
                    result["sources_used"].append("opencitations")
                    result["forward_fallback"] = "opencitations"
            except Exception as exc:
                result["forward_error_oc"] = str(exc)[:200]
                logger.warning("OpenCitations citations (forward) 失败: %s", exc)

    async def _backward_references(
        self,
        result: dict[str, Any],
        paper_id: str | None,
        doi: str | None,
        args: LiteratureInput,
    ) -> None:
        """后向: 这篇引了谁. S2 优先, 挂了回退 OpenCitations."""
        backward_s2_failed = False
        if paper_id:
            try:
                pid_enc = urllib.parse.quote(paper_id, safe="")
                data = await _http_get_json(
                    f"https://api.semanticscholar.org/graph/v1/paper/{pid_enc}/references"
                    f"?fields=title,year,authors,citationCount,externalIds&limit={args.max_citations}"
                )
                for r in data.get("data", []) or []:
                    cp = r.get("citedPaper", {}) or {}
                    result["backward_references"].append(self._s2_paper_to_dict(cp))
                if "s2" not in result["sources_used"]:
                    result["sources_used"].append("s2")
            except Exception as exc:
                backward_s2_failed = True
                result["backward_error_s2"] = str(exc)[:200]
                logger.warning("S2 references (backward) 失败: %s", exc)

        # S2 挂了就回退 OpenCitations
        if backward_s2_failed and doi:
            try:
                oc_refs = await _opencitations_references(doi, args.max_citations)
                if oc_refs:
                    oc_refs = await _enrich_with_crossref(oc_refs)
                    result["backward_references"].extend(oc_refs)
                    if "opencitations" not in result["sources_used"]:
                        result["sources_used"].append("opencitations")
                    result["backward_fallback"] = "opencitations"
            except Exception as exc:
                result["backward_error_oc"] = str(exc)[:200]
                logger.warning("OpenCitations references (backward) 失败: %s", exc)

    async def _do_citations(self, args: LiteratureInput) -> ToolResult:
        """前/后向引用网络. 先试 S2, 被限速(429)就回退到 OpenCitations.

        forward=谁引了这篇; backward=这篇引了谁.
        S2 给完整元数据 (title/authors/citations), OpenCitations 只给 DOI 边,
        靠 CrossRef 补元数据. 两路都挂才报错.
        """
        paper_id = self._resolve_s2_paper_id(args)
        doi = args.doi
        if not doi and args.paper:
            doi = (args.paper or {}).get("doi")
        if not doi and paper_id and paper_id.startswith("DOI:"):
            doi = paper_id[4:]

        if not paper_id and not doi:
            return ToolResult(
                data=None,
                success=False,
                error="无法解析 paper_id, 需要 doi/arxiv_id 或 paper dict (含 doi/url)",
            )

        result: dict[str, Any] = {
            "action": "citations",
            "paper_id": paper_id or (f"DOI:{doi}" if doi else ""),
            "direction": args.direction,
            "forward_citations": [],
            "backward_references": [],
            "sources_used": [],
        }

        if args.direction in ("forward", "both"):
            await self._forward_citations(result, paper_id, doi, args)
        if args.direction in ("backward", "both"):
            await self._backward_references(result, paper_id, doi, args)

        result["forward_count"] = len(result["forward_citations"])
        result["backward_count"] = len(result["backward_references"])
        return ToolResult(data=result, success=True)

    @staticmethod
    def _resolve_s2_paper_id(args: LiteratureInput) -> str | None:
        """构造 S2 paper lookup key. DOI:xxx / arXiv:xxx 格式."""
        doi = args.doi
        arxiv_id = args.arxiv_id
        if args.paper:
            p = args.paper
            doi = doi or p.get("doi")
            url = p.get("url", "") or ""
            if not arxiv_id and "arxiv.org/abs/" in url:
                arxiv_id = url.split("/abs/")[-1].strip("/")
        if doi:
            return f"DOI:{doi}"
        if arxiv_id:
            return f"arXiv:{arxiv_id.strip().strip('/')}"
        return None

    @staticmethod
    def _s2_paper_to_dict(p: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": p.get("title", "") or "",
            "authors": [
                a.get("name", "") for a in (p.get("authors") or []) if a.get("name")
            ],
            "year": p.get("year"),
            "citations": p.get("citationCount"),
            "paperId": p.get("paperId"),
            "doi": (p.get("externalIds") or {}).get("DOI"),
        }

    # ── citation_graph ─────────────────────────────────────

    @staticmethod
    def _cg_seed(
        args: LiteratureInput, seed_id: str | None, doi: str | None
    ) -> dict[str, Any]:
        """种子节点 (depth=0)."""
        return {
            "paper_id": seed_id or (f"DOI:{doi}" if doi else ""),
            "title": (args.paper or {}).get("title", "") if args.paper else "",
            "year": (args.paper or {}).get("year") if args.paper else None,
            "doi": doi,
            "depth": 0,
        }

    async def _cg_bfs(
        self,
        st: _CitationGraphState,
        current_layer: list[str],
        current_dois: list[str],
        do_backward: bool,
        do_forward: bool,
        args: LiteratureInput,
    ) -> None:
        """逐层 BFS 扩展 (双向), 触顶 max_nodes / max_depth 就停."""
        for depth in range(1, args.max_depth + 1):
            if len(st.nodes) >= args.max_nodes or not current_layer:
                break
            next_layer: list[str] = []
            next_dois: list[str] = []

            for idx, parent_id in enumerate(current_layer):
                if len(st.nodes) >= args.max_nodes:
                    break
                parent_doi = current_dois[idx] if idx < len(current_dois) else None

                if do_backward:
                    await self._expand_backward(
                        st, parent_id, parent_doi, depth, args, next_layer, next_dois
                    )
                if do_forward:
                    await self._expand_forward(
                        st, parent_id, parent_doi, depth, args, next_layer, next_dois
                    )

            current_layer = next_layer
            current_dois = next_dois

    @staticmethod
    def _cg_result(
        seed_node: dict[str, Any],
        st: _CitationGraphState,
        direction: str,
        args: LiteratureInput,
    ) -> ToolResult:
        """打包引文图结果."""
        n_forward = sum(1 for e in st.edges if e.get("direction") == "forward")
        n_backward = sum(1 for e in st.edges if e.get("direction") == "backward")
        return ToolResult(
            data={
                "action": "citation_graph",
                "seed_paper_id": seed_node["paper_id"],
                "direction": direction,
                "depth_reached": st.depth_reached,
                "n_unique_papers": len(st.nodes),
                "n_edges": len(st.edges),
                "n_forward_edges": n_forward,
                "n_backward_edges": n_backward,
                "nodes": st.nodes,
                "edges": st.edges,
                "errors": st.errors,
                "truncated": len(st.nodes) >= args.max_nodes,
                "source_mode": st.source_mode,
            },
            success=True,
        )

    async def _do_citation_graph(self, args: LiteratureInput) -> ToolResult:
        """BFS 引文图: 从种子 paper 出发, 沿 backward references 往上挖.

        每层调 S2 references API, 收集 nodes + edges. S2 被限速(429)时
        自动切到 OpenCitations DOI 边 + CrossRef 元数据补全. 触顶 max_nodes
        或 max_depth 就停. 两路都挂也返回已收集的部分图, agent 能用部分结果.
        """
        seed_id = self._resolve_s2_paper_id(args)
        doi = args.doi
        if not doi and args.paper:
            doi = (args.paper or {}).get("doi")
        if not doi and seed_id and seed_id.startswith("DOI:"):
            doi = seed_id[4:]

        if not seed_id and not doi:
            return ToolResult(
                data=None,
                success=False,
                error="无法解析 paper_id, 需要 doi/arxiv_id 或 paper dict (含 doi/url)",
            )

        st = _CitationGraphState(nodes=[], edges=[], visited=set(), errors=[])

        # 种子节点
        seed_node = self._cg_seed(args, seed_id, doi)
        st.nodes.append(seed_node)
        st.visited.add(seed_node["paper_id"])

        # 双向 snowballing 开关: forward=沿被引(向上游), backward=沿被引(向下游)
        # 这里沿用 citations 的 direction 语义:
        #   forward = 谁引用了这篇 (citations 端点)
        #   backward = 这篇引用了谁 (references 端点)
        #   both = 双向扩展 (默认, snowballing 全图)
        direction = getattr(args, "direction", "both") or "both"
        do_backward = direction in ("backward", "both")
        do_forward = direction in ("forward", "both")

        current_dois: list[str] = [doi] if doi else []
        await self._cg_bfs(
            st, [seed_node["paper_id"]], current_dois, do_backward, do_forward, args
        )

        self._persist_citation_graph(st.nodes, st.edges)

        return self._cg_result(seed_node, st, direction, args)

    async def _expand_backward(
        self, st, parent_id, parent_doi, depth, args, next_layer, next_dois
    ) -> None:
        """backward: 这篇引用了谁 (references 端点)."""
        await self._expand_direction(
            st,
            parent_id,
            parent_doi,
            depth,
            args,
            next_layer,
            next_dois,
            direction="backward",
            s2_endpoint="references",
            wrapper_key="citedPaper",
        )

    async def _expand_forward(
        self, st, parent_id, parent_doi, depth, args, next_layer, next_dois
    ) -> None:
        """forward: 谁引用了这篇 (citations 端点)."""
        await self._expand_direction(
            st,
            parent_id,
            parent_doi,
            depth,
            args,
            next_layer,
            next_dois,
            direction="forward",
            s2_endpoint="citations",
            wrapper_key="citingPaper",
        )

    async def _expand_direction(
        self,
        st: _CitationGraphState,
        parent_id: str,
        parent_doi: str | None,
        depth: int,
        args: LiteratureInput,
        next_layer: list[str],
        next_dois: list[str],
        *,
        direction: str,
        s2_endpoint: str,
        wrapper_key: str,
    ) -> None:
        """单方向扩展: S2 端点优先, 限速时切 OpenCitations DOI 边."""
        await self._expand_s2(
            st,
            parent_id,
            depth,
            args,
            next_layer,
            next_dois,
            direction=direction,
            s2_endpoint=s2_endpoint,
            wrapper_key=wrapper_key,
        )
        await self._expand_oc(
            st,
            parent_id,
            parent_doi,
            depth,
            args,
            next_layer,
            next_dois,
            direction=direction,
        )

    async def _expand_s2(
        self,
        st: _CitationGraphState,
        parent_id: str,
        depth: int,
        args: LiteratureInput,
        next_layer: list[str],
        next_dois: list[str],
        *,
        direction: str,
        s2_endpoint: str,
        wrapper_key: str,
    ) -> None:
        """S2 端点扩展; 失败时 (仅 backward) 切 OpenCitations."""
        if st.source_mode != "s2" or not parent_id:
            return
        try:
            children = await self._s2_children(
                parent_id, s2_endpoint, wrapper_key, args.max_citations
            )
        except Exception as exc:
            st.errors.append(f"depth={depth} {parent_id} {direction} s2: {exc}"[:200])
            logger.warning(
                "S2 BFS depth=%d %s failed, switching to OpenCitations: %s",
                depth,
                direction,
                exc,
            )
            # 只有 backward 会把整体数据源切到 OpenCitations (与旧行为一致)
            if direction == "backward":
                st.source_mode = "opencitations"
            return
        self._append_children(
            st,
            children,
            parent_id,
            direction,
            depth,
            args,
            next_layer,
            next_dois,
            source="s2",
            dedupe_visited=False,
        )
        st.depth_reached = depth

    async def _expand_oc(
        self,
        st: _CitationGraphState,
        parent_id: str,
        parent_doi: str | None,
        depth: int,
        args: LiteratureInput,
        next_layer: list[str],
        next_dois: list[str],
        *,
        direction: str,
    ) -> None:
        """OpenCitations DOI 边扩展."""
        if st.source_mode != "opencitations" or not parent_doi:
            return
        try:
            children = await self._oc_children(
                direction, parent_doi, args.max_citations
            )
        except Exception as exc:
            st.errors.append(f"depth={depth} {parent_id} {direction} oc: {exc}"[:200])
            return
        if children is None:
            return
        self._append_children(
            st,
            children,
            parent_id,
            direction,
            depth,
            args,
            next_layer,
            next_dois,
            source="opencitations",
            dedupe_visited=True,
        )
        st.depth_reached = depth

    @staticmethod
    async def _s2_children(
        parent_id: str, endpoint: str, wrapper_key: str, limit: int
    ) -> list[dict[str, Any]]:
        """Fetch one S2 references/citations endpoint as normalized child dicts."""
        pid_enc = urllib.parse.quote(parent_id, safe="")
        data = await _http_get_json(
            f"https://api.semanticscholar.org/graph/v1/paper/{pid_enc}/{endpoint}"
            f"?fields=title,year,externalIds,paperId&limit={limit}"
        )
        children: list[dict[str, Any]] = []
        for item in data.get("data", []) or []:
            cp = item.get(wrapper_key, {}) or {}
            child_id = cp.get("paperId") or ""
            if not child_id:
                continue
            children.append(
                {
                    "child_id": child_id,
                    "title": cp.get("title", "") or "",
                    "year": cp.get("year"),
                    "doi": (cp.get("externalIds") or {}).get("DOI"),
                }
            )
        return children

    @staticmethod
    async def _oc_children(
        kind: str, parent_doi: str, limit: int
    ) -> list[dict[str, Any]] | None:
        """OpenCitations children (enriched via CrossRef). None = no refs at all."""
        if kind == "backward":
            refs = await _opencitations_references(parent_doi, limit)
        else:
            refs = await _opencitations_citations(parent_doi, limit)
        if not refs:
            return None
        children: list[dict[str, Any]] = []
        for ref in await _enrich_with_crossref(refs):
            child_doi = ref.get("doi", "")
            child_id = f"DOI:{child_doi}" if child_doi else ""
            if not child_id:
                continue
            children.append(
                {
                    "child_id": child_id,
                    "title": ref.get("title", ""),
                    "year": ref.get("year"),
                    "doi": child_doi,
                }
            )
        return children

    def _append_children(
        self,
        st,
        children,
        parent_id,
        direction,
        depth,
        args,
        next_layer,
        next_dois,
        *,
        source: str,
        dedupe_visited: bool,
    ) -> None:
        """Append edges + nodes for one fetched child batch."""
        for ch in children:
            child_id = ch["child_id"]
            if dedupe_visited and child_id in st.visited:
                continue
            if direction == "backward":
                st.edges.append(
                    {"from": parent_id, "to": child_id, "direction": "backward"}
                )
            else:
                st.edges.append(
                    {"from": child_id, "to": parent_id, "direction": "forward"}
                )
            added = self._add_node(
                st.nodes,
                st.visited,
                next_layer,
                next_dois,
                child_id=child_id,
                title=ch["title"],
                year=ch["year"],
                doi=ch["doi"],
                depth=depth,
                max_nodes=args.max_nodes,
                source=source,
            )
            if added is False:
                break

    @staticmethod
    def _persist_citation_graph(
        nodes: list[dict[str, Any]], edges: list[dict[str, str]]
    ) -> None:
        """引文图持久化到 KG — Zotero 启发: 引用关系应持久存储供后续查询.

        之前是 ephemeral 的, agent 每次都要重新 BFS. 现在写一次, 后续
        KG query 和 /graph 可视化都能看到.
        """
        try:
            from pathlib import Path

            from huginn.kg.entities import EntityType, Relation
            from huginn.kg.graph import ProjectKnowledgeGraph

            kg = ProjectKnowledgeGraph(Path("."))
            # pid -> node eid, 供 add_relation 用
            eid_map: dict[str, str] = {}
            for node in nodes:
                pid = node.get("paper_id", "")
                if not pid:
                    continue
                title = node.get("title", "") or pid
                eid = kg.add_entity(
                    title,
                    EntityType.LITERATURE,
                    source="citation_graph",
                    confidence=0.8 if node.get("depth", 99) <= 1 else 0.6,
                    paper_id=pid,
                    doi=node.get("doi", ""),
                    year=node.get("year"),
                    depth=node.get("depth", 0),
                )
                eid_map[pid] = eid
            for edge in edges:
                src = eid_map.get(edge.get("from", ""))
                dst = eid_map.get(edge.get("to", ""))
                if src and dst:
                    kg.add_relation(src, Relation.CITES, dst, source="citation_graph")
            kg.save()
        except Exception:
            logger.debug("citation graph kg.save() failed, best-effort", exc_info=True)

    @staticmethod
    def _add_node(
        nodes: list[dict[str, Any]],
        visited: set[str],
        next_layer: list[str],
        next_dois: list[str],
        *,
        child_id: str,
        title: str,
        year: Any,
        doi: Any,
        depth: int,
        max_nodes: int,
        source: str = "s2",
    ) -> bool | None:
        """向图谱追加一个节点 (去重 + 加入下一层). 返回 False 表示触顶应停."""
        if child_id in visited:
            return True
        if len(nodes) >= max_nodes:
            return False
        nodes.append(
            {
                "paper_id": child_id,
                "title": title,
                "year": year,
                "doi": doi,
                "depth": depth,
                "source": source,
            }
        )
        visited.add(child_id)
        next_layer.append(child_id)
        if doi:
            next_dois.append(doi)
        return True

    # ── ingest_to_rag ───────────────────────────────────────

    @staticmethod
    def _rag_doc_text(p: dict[str, Any]) -> tuple[str, str] | None:
        """拼入库文本 + doc_id; 无 title/abstract/full_text 返回 None."""
        title = p.get("title", "")
        authors = ", ".join((p.get("authors") or [])[:5])
        year = p.get("year") or ""
        venue = p.get("venue", "")
        doi = p.get("doi") or ""
        abstract = p.get("abstract", "") or ""
        full_text = p.get("full_text", "") or ""
        if not full_text and not abstract and not title:
            return None
        # P0-3: 有全文优先入库全文 (截到 3000 字, 保真度 > 摘要), 退回 abstract
        body = full_text if full_text else abstract
        if len(body) > 3000:
            body = body[:3000] + "..."
        body_label = "FullText" if full_text else "Abstract"
        doc_text = (
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Year: {year}\n"
            f"Venue: {venue}\n"
            f"DOI: {doi}\n"
            f"{body_label}: {body}"
        )
        return doc_text, doi or f"lit_{_norm_title(title)[:40]}"

    async def _rag_ingest_one(
        self, rag: Any, p: dict[str, Any], context: ToolContext
    ) -> tuple[str, dict[str, str] | None]:
        """ingest 单篇; 返回 (doc_id, failure). 跳过时 doc_id 为空."""
        from huginn.rag.rag_tool import RAGToolInput

        built = self._rag_doc_text(p)
        if built is None:
            return "", None
        doc_text, doc_id = built
        try:
            res = await rag.call(
                RAGToolInput(action="ingest", document=doc_text, doc_id=doc_id),
                context,
            )
            if res.success:
                return doc_id, None
            return doc_id, {"doc_id": doc_id, "error": (res.error or "")[:100]}
        except Exception as exc:
            return doc_id, {"doc_id": doc_id, "error": str(exc)[:100]}

    async def _do_ingest_to_rag(
        self, args: LiteratureInput, context: ToolContext
    ) -> ToolResult:
        """把搜到的论文 ingest 进 rag_tool, 下次同主题 search 时本地能搜到."""
        from huginn.tools.registry import ToolRegistry

        papers = args.papers
        if not papers and args.query:
            search_res = await self._do_search(args, context)
            if not search_res.success:
                return search_res
            papers = (search_res.data or {}).get("papers", [])
        if not papers:
            return ToolResult(
                data=None,
                success=False,
                error="no papers to ingest (provide papers or query)",
            )

        rag = ToolRegistry.get("rag_tool")
        if rag is None:
            return ToolResult(
                data=None,
                success=False,
                error="rag_tool 未注册, 无法 ingest. 需要先初始化 rag_tool",
            )

        ingested: list[str] = []
        failed: list[dict[str, str]] = []
        for p in papers:
            doc_id, failure = await self._rag_ingest_one(rag, p, context)
            if doc_id:
                ingested.append(doc_id)
            if failure is not None:
                failed.append(failure)

        return ToolResult(
            data={
                "action": "ingest_to_rag",
                "n_ingested": len(ingested),
                "n_failed": len(failed),
                "doc_ids": ingested,
                "failures": failed,
            },
            success=True,
        )

    # ── crawl_web ───────────────────────────────────────────

    @staticmethod
    def _crawl_auth_status() -> ToolResult:
        """auth_action=status: 列出所有已存 session."""
        sessions = _list_sessions()
        return ToolResult(
            data={
                "action": "crawl_web",
                "auth_action": "status",
                "n_sessions": len(sessions),
                "sessions": sessions,
                "supported_providers": list(_PROVIDERS.keys()),
                "note": "用 auth_action=login provider=<name> 登录新源. "
                "crawl_web engine=direct 会自动检测 URL 域名复用对应 session.",
            },
            success=True,
        )

    async def _crawl_auth_login(self, args: LiteratureInput) -> ToolResult:
        """auth_action=login: 弹非 headless 浏览器让用户手动登录订阅源."""
        if not args.provider:
            return ToolResult(
                data=None,
                success=False,
                error="auth_action=login 需要 provider 字段. 可选: "
                + ", ".join(_PROVIDERS.keys()),
            )
        try:
            res = await _auth_login(args.provider)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"auth login 失败: {exc}",
            )
        return ToolResult(
            data={
                "action": "crawl_web",
                "auth_action": "login",
                **res,
                "note": (
                    "profile 已存盘. 之后 crawl_web engine=direct url=<该源页面> "
                    "会自动 headless 复用此 profile. session 过期会提示重新 login."
                )
                if res["result"] == "success"
                else "登录未确认成功, profile 仍存盘可试. 换 result 字段看原因.",
            },
            success=True,
        )

    async def _crawl_auth_logout(self, args: LiteratureInput) -> ToolResult:
        """auth_action=logout: 删 provider 的 profile."""
        if not args.provider:
            return ToolResult(
                data=None,
                success=False,
                error="auth_action=logout 需要 provider 字段",
            )
        try:
            res = await _auth_logout(args.provider)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"auth logout 失败: {exc}",
            )
        return ToolResult(
            data={"action": "crawl_web", "auth_action": "logout", **res},
            success=True,
        )

    async def _do_crawl_web(
        self, args: LiteratureInput, context: ToolContext
    ) -> ToolResult:
        """通用网页爬取 + 搜索引擎桥接 + 订阅源认证.

        三种模式:
          - auth_action=login: 弹非 headless 浏览器让用户手动登录订阅源, 存 profile
          - auth_action=status: 列出所有已存 session
          - auth_action=logout: 删 provider 的 profile
          - engine=direct: 直接爬 args.url (有 session 自动复用 profile)
          - engine=google_scholar/google_patents/duckduckgo: 搜结果链接列表

        crawl4ai 不可用时降级到 urllib + 简单 HTML 抽正文.
        """
        # 认证动作优先
        if args.auth_action == "status":
            return self._crawl_auth_status()
        if args.auth_action == "login":
            return await self._crawl_auth_login(args)
        if args.auth_action == "logout":
            return await self._crawl_auth_logout(args)

        engine = args.engine
        url = (args.url or "").strip()
        query = (args.query or "").strip()

        # 模式判定
        if engine == "direct":
            if not url:
                return ToolResult(
                    data=None,
                    success=False,
                    error="crawl_web engine=direct 需要 url",
                )
            return await crawl_direct(url, args.max_results)
        # 搜索引擎模式
        if not query:
            return ToolResult(
                data=None,
                success=False,
                error=f"crawl_web engine={engine} 需要 query",
            )
        return await crawl_search_engine(engine, query, args.max_results, context)

    # ── extract_figures ────────────────────────────────────

    @staticmethod
    def _pdf_figures(
        pdf_bytes: bytes, tmp_dir: Any
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """PyMuPDF 抽嵌入图片到 tmp_dir; 返回 (extracted, error)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None, "PyMuPDF (fitz) 未安装, 无法提取 PDF 图片. pip install pymupdf"

        extracted: list[dict[str, Any]] = []
        # 用 with 保证异常路径也释放文件句柄
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    try:
                        base_image = doc.extract_image(xref)
                    except Exception as exc:
                        logger.warning("extract_image xref=%s 失败: %s", xref, exc)
                        continue
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    img_filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                    img_path = tmp_dir / img_filename
                    img_path.write_bytes(image_bytes)
                    extracted.append(
                        {
                            "page": page_num + 1,
                            "index": img_index + 1,
                            "path": str(img_path),
                            "ext": image_ext,
                            "size_bytes": len(image_bytes),
                        }
                    )
        return extracted, None

    async def _analyze_figures(
        self,
        extracted: list[dict[str, Any]],
        img_tool: Any,
        context: ToolContext,
    ) -> list[dict[str, Any]]:
        """逐张调 image_analysis_tool 做 plot_extract."""
        analyses: list[dict[str, Any]] = []
        for fig in extracted:
            try:
                result = await img_tool.call(
                    {
                        "image_path": fig["path"],
                        "action": "plot_extract",
                        "parameters": {},
                    },
                    context,
                )
                analyses.append(
                    {
                        "image": fig["path"],
                        "page": fig["page"],
                        "success": result.success,
                        "data": result.data if result.success else None,
                        "error": result.error if not result.success else None,
                    }
                )
            except Exception as exc:
                analyses.append(
                    {
                        "image": fig["path"],
                        "page": fig["page"],
                        "success": False,
                        "error": str(exc)[:200],
                    }
                )
        return analyses

    @staticmethod
    def _figures_result(
        used_url: str,
        extracted: list[dict[str, Any]],
        analyses: list[dict[str, Any]],
        tmp_dir: Any,
        message: str | None = None,
    ) -> ToolResult:
        """打包图片提取结果."""
        data: dict[str, Any] = {
            "action": "extract_figures",
            "pdf_url": used_url,
            "n_images": len(extracted),
            "images": extracted,
            "analyses": analyses,
        }
        if message is not None:
            data["message"] = message
        else:
            data["tmp_dir"] = str(tmp_dir)
        return ToolResult(data=data, success=True)

    async def _do_extract_figures(
        self, args: LiteratureInput, context: ToolContext
    ) -> ToolResult:
        """从论文 PDF 中提取所有嵌入图片, 逐张调 image_analysis_tool 做 plot_extract.

        流程: 多源下载 PDF -> PyMuPDF 抽 image xobjects -> 存临时文件 ->
        调 image_analysis_tool plot_extract 分析每张图.
        """
        pdf_bytes, used_url, tried = await self._download_pdf_bytes(args)
        if pdf_bytes is None:
            if not tried:
                return ToolResult(
                    data=None,
                    success=False,
                    error="无法解析 PDF URL, 需要 arxiv_id/doi/url 或 paper dict (含 url/doi/oa_url)",
                )
            last_status = tried[-1]["status"] if tried else ""
            return ToolResult(
                data=None,
                success=False,
                error=f"所有 PDF 候选源都失败 (试了 {len(tried)} 个). 最后错误: {last_status[:200]}",
            )

        import tempfile
        from pathlib import Path

        tmp_dir = Path(tempfile.mkdtemp(prefix="lit_figures_"))
        extracted, error = self._pdf_figures(pdf_bytes, tmp_dir)
        if extracted is None:
            return ToolResult(data=None, success=False, error=error)

        if not extracted:
            return self._figures_result(
                used_url, extracted, [], tmp_dir, message="PDF 中未找到嵌入图片"
            )

        from huginn.tools.registry import ToolRegistry

        img_tool = ToolRegistry.get("image_analysis_tool")
        if img_tool is None:
            logger.warning("image_analysis_tool 未注册, 跳过图片分析")
            return self._figures_result(used_url, extracted, [], tmp_dir)

        analyses = await self._analyze_figures(extracted, img_tool, context)
        return self._figures_result(used_url, extracted, analyses, tmp_dir)

    # ── multi_review (nuwa 6 路并行透镜 + cangjie 三重验证) ────

    async def _do_multi_review(
        self, args: LiteratureInput, context: ToolContext
    ) -> ToolResult:
        """N 路并行透镜 + 三重验证的深度综述.

        nuwa 启发: 6 个透镜并行调用 LLM, 失败透镜降级为空 findings 不阻塞.
        cangjie 启发: 每条 claim 三重验证 (跨域复现/生成力/排他性), 2-3 分 high.
        高阶网络视角: 调和分量 β₁>0 对应"研究孤岛".
        """
        papers, err = await self._mr_collect_papers(args, context)
        if err is not None:
            return err

        lens_names, err = self._mr_select_lenses(args)
        if err is not None:
            return err

        paper_block = self._build_paper_block(papers)
        model, err = self._mr_get_model(context)
        if err is not None:
            return err

        lenses_ok, lenses_failed = await self._mr_run_lenses(
            model,
            papers,
            args.query,
            lens_names,
            paper_block,
        )
        all_claims = self._mr_collect_claims(lenses_ok)
        verified_claims = self._mr_verify_claims(args, all_claims, len(papers))
        by_lens, high_conf_claims, lens_summaries = self._mr_summarize(
            verified_claims,
            lenses_ok,
        )
        claim_audit_result = self._audit_verified_claims(
            verified_claims, papers, workspace=getattr(context, "workspace", "") or ""
        )
        return self._mr_build_result(
            args,
            papers,
            lens_names,
            lenses_ok,
            lenses_failed,
            lens_summaries,
            by_lens,
            high_conf_claims,
            verified_claims,
            claim_audit_result,
        )

    async def _mr_collect_papers(self, args: LiteratureInput, context: ToolContext):
        """拿 papers: 优先用显式传入, 没有就 search. 返回 (papers, error)."""
        papers = args.papers
        if not papers and args.query:
            search_res = await self._do_search(args, context)
            if not search_res.success:
                return None, search_res
            papers = (search_res.data or {}).get("papers", [])
        if not papers:
            return None, ToolResult(
                data=None,
                success=False,
                error="no papers to review (provide papers or a query)",
            )
        # 截断, 避免每个透镜的 LLM context 爆掉
        return papers[:12], None

    def _mr_select_lenses(self, args: LiteratureInput):
        """过滤掉不认识的透镜名, 避免传垃圾进 _LENS_PROMPTS."""
        lens_names = args.lenses if args.lenses else _DEFAULT_LENSES
        lens_names = [ln for ln in lens_names if ln in _LENS_PROMPTS]
        if not lens_names:
            return None, ToolResult(
                data=None,
                success=False,
                error=f"no valid lenses in {args.lenses}, valid: {list(_LENS_PROMPTS.keys())}",
            )
        return lens_names, None

    def _mr_get_model(self, context: ToolContext):
        """初始化 LLM model. 返回 (model, error)."""
        try:
            return self._get_model(context), None
        except Exception as exc:
            return None, ToolResult(
                data=None,
                success=False,
                error=f"LLM 初始化失败: {exc}",
            )

    async def _mr_run_lenses(
        self,
        model,
        papers,
        query: str,
        lens_names: list[str],
        paper_block: str,
    ):
        """并行跑 N 个透镜. 返回 (lenses_ok, lenses_failed)."""

        async def _run_lens(lens_name: str) -> dict[str, Any] | None:
            system_prompt = _LENS_PROMPTS[lens_name]
            user_prompt = (
                f"研究 query: {query or '(未指定)'}\n\n"
                f"论文列表 ({len(papers)} 篇):\n\n{paper_block}\n\n"
                f"请用 '{lens_name}' 透镜分析, 按 JSON 格式输出."
            )
            try:
                content = await invoke_llm_text(model, system_prompt, user_prompt)
                parsed = self._parse_json(content)
                if not parsed or "findings" not in parsed:
                    logger.warning(
                        "lens %s 返回无效 JSON: %s", lens_name, content[:200]
                    )
                    return None
                parsed["lens"] = lens_name
                return parsed
            except Exception as exc:
                logger.warning("lens %s 失败 (降级为空): %s", lens_name, exc)
                return None

        lens_results = await asyncio.gather(*[_run_lens(ln) for ln in lens_names])
        lenses_ok: list[dict[str, Any]] = [r for r in lens_results if r is not None]
        lenses_failed: list[str] = [
            ln for ln, r in zip(lens_names, lens_results) if r is None
        ]
        return lenses_ok, lenses_failed

    def _mr_collect_claims(
        self, lenses_ok: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """收集所有 claims, 每条带 lens + paper_idx."""
        all_claims: list[dict[str, Any]] = []
        for lens_out in lenses_ok:
            for finding in lens_out.get("findings") or []:
                if not isinstance(finding, dict):
                    continue
                claim = (finding.get("claim") or "").strip()
                if not claim:
                    continue
                all_claims.append(
                    {
                        "claim": claim,
                        "lens": lens_out.get("lens", ""),
                        "paper_idx": finding.get("paper_idx", []) or [],
                        "llm_confidence": finding.get("confidence", "medium"),
                    }
                )
        return all_claims

    def _mr_verify_claims(
        self,
        args: LiteratureInput,
        all_claims: list[dict[str, Any]],
        n_papers: int,
    ) -> list[dict[str, Any]]:
        """三重验证 (cangjie V1/V2/V3); 不验证时用 LLM 自报置信度."""
        if not (args.verify_claims and all_claims):
            return [
                {**c, "final_confidence": c.get("llm_confidence", "medium")}
                for c in all_claims
            ]

        verified_claims: list[dict[str, Any]] = []
        for c in all_claims:
            v1, v2, v3 = self._triple_verify(c, n_papers)
            score = sum([v1, v2, v3])
            if score >= 2:
                final_conf = "high"
            elif score == 1:
                final_conf = "medium"
            else:
                final_conf = "low"
            verified_claims.append(
                {
                    **c,
                    "v1_cross_domain": v1,
                    "v2_generative": v2,
                    "v3_exclusive": v3,
                    "verification_score": score,
                    "final_confidence": final_conf,
                }
            )
        return verified_claims

    def _mr_summarize(
        self, verified_claims: list[dict[str, Any]], lenses_ok: list[dict[str, Any]]
    ):
        """按透镜分组 findings + 高置信 claims + 透镜 summary."""
        by_lens: dict[str, list[dict[str, Any]]] = {}
        for vc in verified_claims:
            by_lens.setdefault(vc.get("lens", ""), []).append(vc)

        # 高置信 claims 单独拎出来 (cangjie stress test: 只信通过 2/3 验证的)
        high_conf_claims = [
            vc for vc in verified_claims if vc.get("final_confidence") == "high"
        ]
        lens_summaries = {
            (lo.get("lens") or ""): (lo.get("summary") or "") for lo in lenses_ok
        }
        return by_lens, high_conf_claims, lens_summaries

    def _mr_build_result(
        self,
        args: LiteratureInput,
        papers: list[dict[str, Any]],
        lens_names: list[str],
        lenses_ok: list[dict[str, Any]],
        lenses_failed: list[str],
        lens_summaries: dict[str, str],
        by_lens: dict[str, list[dict[str, Any]]],
        high_conf_claims: list[dict[str, Any]],
        verified_claims: list[dict[str, Any]],
        claim_audit_result: dict[str, Any],
    ) -> ToolResult:
        """组装 multi_review ToolResult."""
        return ToolResult(
            data={
                "action": "multi_review",
                "query": args.query or "",
                "n_papers": len(papers),
                "n_lenses_requested": len(lens_names),
                "n_lenses_ok": len(lenses_ok),
                "lenses_failed": lenses_failed,
                "lens_summaries": lens_summaries,
                "findings_by_lens": by_lens,
                "high_confidence_claims": high_conf_claims,
                "all_claims": verified_claims,
                "n_claims_total": len(verified_claims),
                "n_claims_high": len(high_conf_claims),
                "verification_enabled": args.verify_claims,
                "claim_audit": claim_audit_result,
                "papers": [
                    {
                        "idx": i + 1,
                        "title": p.get("title", ""),
                        "doi": p.get("doi"),
                        "year": p.get("year"),
                    }
                    for i, p in enumerate(papers)
                ],
            },
            success=True,
        )

    @staticmethod
    def _triple_verify(claim: dict[str, Any], n_papers: int) -> tuple[bool, bool, bool]:
        """cangjie 三重验证: V1 跨域复现 / V2 生成力 / V3 排他性.

        V1: claim 出现在 ≥2 篇论文 → True (跨域复现)
        V2: claim 含可证伪预测结构 (如果/if/则/当/predict/应当/should) → True (生成力)
        V3: claim 不是常识 (出现在 <50% 论文) → True (排他性)

        ponytail: V2 用关键词匹配是粗启发式. 升级: LLM 判定可证伪性.
        ponytail: V3 的 50% 阈值未校准. 升级: 按领域动态调整.
        """
        paper_idx = claim.get("paper_idx") or []
        # V1: 跨域复现 — 出现在 ≥2 篇
        v1 = len(paper_idx) >= 2

        # V2: 生成力 — 含可证伪预测结构
        claim_text = (claim.get("claim") or "").lower()
        generative_markers = [
            "如果",
            "if ",
            "则",
            "then",
            "当",
            "when ",
            "predict",
            "应当",
            "should",
            "预期",
            "expect",
            "会导致",
            "leads to",
            "implies",
        ]
        v2 = any(m in claim_text for m in generative_markers)

        # V3: 排他性 — 不是常识 (出现在 <50% 论文)
        # ponytail: n_papers=0 时除零保护
        coverage = len(paper_idx) / n_papers if n_papers > 0 else 1.0
        v3 = coverage < 0.5

        return (v1, v2, v3)

    @staticmethod
    def _claim_findings(
        verified_claims: list[dict[str, Any]],
        papers: list[dict[str, Any]],
        conf_map: dict[str, float],
    ) -> list[dict[str, Any]]:
        """把 high/medium 置信 claim 转成 ClaimAuditor findings (low 丢弃)."""
        findings: list[dict[str, Any]] = []
        for vc in verified_claims:
            conf = vc.get("final_confidence", "medium")
            if conf not in ("high", "medium"):
                continue
            claim = (vc.get("claim") or "").strip()
            if not claim:
                continue
            # paper_idx 是 1-based (lens prompt 约定), 映射回论文 DOI/标题
            literature_ids: list[str] = []
            for i in vc.get("paper_idx") or []:
                if not isinstance(i, int) or not (1 <= i <= len(papers)):
                    continue
                p = papers[i - 1]
                literature_ids.append(p.get("doi") or p.get("title") or f"paper_{i}")
            findings.append(
                {
                    "claim": claim,
                    "premises": literature_ids,
                    "mode": "AND",  # "论文支持结论"是合取证据
                    "evidence_strength": conf_map.get(conf, 0.7),
                    "literature_id": literature_ids[0] if literature_ids else "",
                    "source": "literature_multi_review",
                }
            )
        return findings

    def _audit_verified_claims(
        self,
        verified_claims: list[dict[str, Any]],
        papers: list[dict[str, Any]],
        workspace: str = "",
    ) -> dict[str, Any]:
        """把 multi_review 的高/中置信结论登记进 claim 层 (best-effort).

        复用 ClaimAuditor.ingest_findings: 每条 verified claim 带 premise
        (论文引用) + evidence_strength (final_confidence 映射) + 来源 (论文
        DOI/标题). 只登记 final_confidence in {high, medium}, low 丢弃 (避免
        污染知识层). 失败 (缺依赖/workspace 无效) 返回 {"registered": 0,
        "error": ...}, 不 raise — multi_review 主结果不受影响.
        """
        try:
            from huginn.kg.claim_audit import ClaimAuditor
        except Exception as exc:
            return {"registered": 0, "error": f"claim_audit_unavailable: {exc}"}

        if not workspace:
            return {"registered": 0, "error": "no_workspace"}

        conf_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
        findings = self._claim_findings(verified_claims, papers, conf_map)

        if not findings:
            return {"registered": 0, "n_findings": 0}

        try:
            auditor = ClaimAuditor(workspace)
            return auditor.ingest_findings(findings, source_prefix="lit_review")
        except Exception as exc:
            return {"registered": 0, "error": f"audit_failed: {exc}"}

    @staticmethod
    def _build_paper_block(papers: list[dict[str, Any]]) -> str:
        """构造论文清单给 LLM, multi_review 和 summarize 共用格式."""
        parts: list[str] = []
        for i, p in enumerate(papers, 1):
            authors_short = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors_short += " et al."
            year = p.get("year") or ""
            venue = p.get("venue") or ""
            abstract = (p.get("abstract") or "").strip()
            if len(abstract) > 1500:
                abstract = abstract[:1500] + "..."
            parts.append(
                f"[{i}] {p.get('title', '')}\n"
                f"  Authors: {authors_short}\n"
                f"  Year: {year}  Venue: {venue}  DOI: {p.get('doi') or '-'}\n"
                f"  Abstract: {abstract}"
            )
        return "\n\n".join(parts)

    # ── helpers ─────────────────────────────────────────────

    def _get_model(self, context: ToolContext) -> Any:
        from huginn.llm import get_model

        config = getattr(context, "config", None)
        return get_model(config=config, temperature=0.2, max_tokens=6000)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        text = text.strip()
        # 容忍 ```json 代码块
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def _to_bibtex(papers: list[dict[str, Any]]) -> str:
        """把 paper 列表转成 BibTeX 字符串. cite key 用 firstauthor+year."""
        lines: list[str] = []
        for i, p in enumerate(papers, 1):
            authors = p.get("authors") or []
            first_author = authors[0].split()[-1] if authors else "unknown"
            year = p.get("year") or "nd"
            cite_key = f"{first_author.lower()}{year}"
            if len(papers) > 1:
                cite_key += f"_{i}"
            title = p.get("title", "").replace("{", "").replace("}", "")
            venue = p.get("venue", "")
            doi = p.get("doi") or ""
            url = p.get("url", "")
            lines.append(
                f"@article{{{cite_key},\n"
                f"  title = {{{title}}},\n"
                f"  author = {{{' and '.join(authors)}}},\n"
                f"  year = {{{year}}},\n"
                f"  journal = {{{venue}}},\n"
                + (f"  doi = {{{doi}}},\n" if doi else "")
                + (f"  url = {{{url}}},\n" if url else "")
                + "}\n"
            )
        return "\n".join(lines)

    def estimate_cost(self, args: LiteratureInput) -> dict[str, float] | None:
        if args.action == "search":
            return {"cpu_hours": 0.0, "walltime_hours": 0.01}
        if args.action == "fetch_pdf":
            # 多源候选轮询, 可能试 4-5 个 URL 才成
            return {"cpu_hours": 0.0, "walltime_hours": 0.08}
        if args.action == "citations":
            return {"cpu_hours": 0.0, "walltime_hours": 0.02}
        if args.action == "ingest_to_rag":
            return {"cpu_hours": 0.0, "walltime_hours": 0.05}  # embedding 调用
        if args.action == "crawl_web":
            if args.auth_action == "login":
                # 非 headless 等用户操作, 上限 5min
                return {"cpu_hours": 0.0, "walltime_hours": 0.3}
            if args.auth_action in ("status", "logout"):
                return {"cpu_hours": 0.0, "walltime_hours": 0.01}
            # crawl4ai 起 Playwright 浏览器, 比 API 慢得多
            return {"cpu_hours": 0.0, "walltime_hours": 0.15}
        if args.action == "extract_figures":
            # 下载 PDF + 提图 + 逐张调 image_analysis, 图多时偏慢
            return {"cpu_hours": 0.0, "walltime_hours": 0.2}
        if args.action == "multi_review":
            # N 路透镜并行 LLM 调用 (默认 6), 比单次 summarize 慢但并行
            n_lenses = len(args.lenses) if args.lenses else 6
            return {"cpu_hours": 0.0, "walltime_hours": 0.05 * n_lenses / 6}
        return {
            "cpu_hours": 0.0,
            "walltime_hours": 0.02,
        }  # summarize/benchmark_lookup 调 LLM


# ── benchmark_lookup 分阶段 helper ──────────────────────────────────


def _bl_validate(args) -> ToolResult | None:
    """校验 system/property 必填."""
    if not args.system or not args.property:
        return ToolResult(
            data=None,
            success=False,
            error="system and property are required for benchmark_lookup",
        )
    return None


def _bl_empty_result(args, query: str) -> ToolResult:
    """没搜到文献时的空结果."""
    return ToolResult(
        data={
            "action": "benchmark_lookup",
            "system": args.system,
            "property": args.property,
            "query": query,
            "reported_values": [],
            "consensus": None,
            "spread": None,
            "n_papers_searched": 0,
            "n_papers_with_values": 0,
            "kb_written": 0,
            "message": "没搜到相关文献",
        },
        success=True,
    )


def _bl_consensus(reported: list[dict[str, Any]]):
    """共识 (mean/median) + 离散度 (min/max/range)."""
    consensus = None
    spread = None
    if not reported:
        return consensus, spread
    vals = [r["value"] for r in reported]
    mean = sum(vals) / len(vals)
    consensus = {
        "mean": round(mean, 6),
        "median": round(sorted(vals)[len(vals) // 2], 6),
        "unit": reported[0]["unit"],
        "n_sources": len(reported),
    }
    if len(vals) >= 2:
        spread = {
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
            "range": round(max(vals) - min(vals), 6),
        }
    return consensus, spread


def _bl_write_kb(args, reported: list[dict[str, Any]]) -> int:
    """把抽到的文献报道值写回知识库. 返回写入条数."""
    kb_written = 0
    try:
        from huginn.knowledge.store import get_knowledge_base

        kb = get_knowledge_base()
        for rv in reported:
            doi = rv.get("doi") or ""
            title = rv.get("source_paper") or ""
            text = (
                f"{args.system} | {args.property} = {rv['value']} {rv['unit']}\n"
                f"method: {rv.get('method', '')}\n"
                f"source: {title}\n"
                f"doi: {doi}\n"
                f"year: {rv.get('year', '')}\n"
            )
            meta = {"doi": doi, "title": title, "source": "benchmark_lookup"}
            kb.add_text(text, filename="benchmark_lookup", metadata=meta)
            kb_written += 1
    except Exception as exc:
        logger.warning("benchmark_lookup KB 写回失败: %s", exc)
    return kb_written


def _bl_build_result(
    args,
    query: str,
    papers: list[dict[str, Any]],
    reported: list[dict[str, Any]],
    consensus,
    spread,
    consistency: dict[str, Any],
    compat,
    completion,
    kb_written: int,
) -> ToolResult:
    """组装 benchmark_lookup ToolResult."""
    return ToolResult(
        data={
            "action": "benchmark_lookup",
            "system": args.system,
            "property": args.property,
            "query": query,
            "reported_values": reported,
            "consensus": consensus,
            "spread": spread,
            "consistency": consistency,
            "compat": compat,
            "completion": completion,
            "n_papers_searched": len(papers),
            "n_papers_with_values": len(reported),
            "kb_written": kb_written,
        },
        success=True,
    )
