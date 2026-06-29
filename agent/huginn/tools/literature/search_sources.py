"""7 路学术搜索 + 去重 / 规范化.

arXiv / S2 / CrossRef / OpenAlex / PubMed / DOAJ / CORE 各自独立函数,
失败返回空列表不阻塞其他源. 去重以 DOI 优先, 无 DOI 走标题归一化.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from ._http import (
    _OPENER,
    _USER_AGENT,
    _http_get_json,
    _http_get_text,
    _timeout,
    logger,
)

# arXiv Atom feed 命名空间
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_DOI_NS = "{http://arxiv.org/schemas/atom}doi"


# ───────────────────────── 去重 / 规范化 ─────────────────────────


def _norm_title(title: str) -> str:
    """标题归一化做去重 key: 小写 + 去标点 + 压空格.

    arXiv/S2/CrossRef 同一篇论文标题可能差几个标点或大小写, 直接 string compare 会漏.
    """
    t = re.sub(r"[^\w\s]", " ", title.lower())
    return re.sub(r"\s+", " ", t).strip()


def _dedup(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """三路结果去重. DOI 优先, 没 DOI 用标题归一化.

    合并策略: 保留 abstract 最长的版本 (有的源 abstract 为空), citations 取大值,
    source 记录所有来源方便排查.
    """
    seen: dict[tuple, dict[str, Any]] = {}
    for p in papers:
        if p["doi"]:
            key = ("doi", p["doi"].lower())
        else:
            key = ("title", _norm_title(p["title"]))
            if not key[1]:
                continue
        if key not in seen:
            seen[key] = dict(p)
            seen[key]["_sources"] = [p["source"]]
        else:
            existing = seen[key]
            existing["_sources"] = list(set(existing.get("_sources", []) + [p["source"]]))
            # abstract 取更长的
            if len(p.get("abstract", "")) > len(existing.get("abstract", "")):
                existing["abstract"] = p["abstract"]
            # citations 取大值
            if (p.get("citations") or 0) > (existing.get("citations") or 0):
                existing["citations"] = p["citations"]
            # year/venue 缺了就补
            if not existing.get("year") and p.get("year"):
                existing["year"] = p["year"]
            if not existing.get("venue") and p.get("venue"):
                existing["venue"] = p["venue"]
            if not existing.get("doi") and p.get("doi"):
                existing["doi"] = p["doi"]
    out = list(seen.values())
    for p in out:
        p["sources"] = p.pop("_sources", [p.get("source", "")])
    return out


def _sort_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 citation 降序排, 有 abstract 的优先. 没引用数的算 0."""
    return sorted(
        papers,
        key=lambda p: (bool(p.get("abstract")), p.get("citations") or 0),
        reverse=True,
    )


# ───────────────────────── arXiv ─────────────────────────


async def _search_arxiv(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """arXiv API. 返回 Atom XML, 解析成 paper dict 列表."""
    q = urllib.parse.quote(query)
    url = (
        f"http://export.arxiv.org/api/query?search_query=all:{q}"
        f"&start=0&max_results={max_results}&sortBy=relevance"
    )
    try:
        xml_text = await _http_get_text(url)
    except Exception as exc:
        logger.warning("arXiv 请求失败: %s", exc)
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("arXiv XML 解析失败: %s", exc)
        return []

    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        title_el = entry.find("atom:title", _ARXIV_NS)
        summary_el = entry.find("atom:summary", _ARXIV_NS)
        id_el = entry.find("atom:id", _ARXIV_NS)
        published_el = entry.find("atom:published", _ARXIV_NS)
        if title_el is None or id_el is None:
            continue
        title = re.sub(r"\s+", " ", (title_el.text or "").strip())
        abstract = (summary_el.text or "").strip() if summary_el is not None else ""
        authors: list[str] = []
        for author in entry.findall("atom:author", _ARXIV_NS):
            name_el = author.find("atom:name", _ARXIV_NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())
        year: int | None = None
        if published_el is not None and published_el.text:
            m = re.match(r"(\d{4})", published_el.text)
            if m:
                year = int(m.group(1))
        # year 过滤放这儿, arXiv API 本身不支持 year range
        if year_from and year and year < year_from:
            continue
        if year_to and year and year > year_to:
            continue
        url_ = (id_el.text or "").strip()
        doi_el = entry.find(_ARXIV_DOI_NS)
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None
        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": "arXiv",
            "doi": doi,
            "abstract": abstract,
            "url": url_,
            "citations": None,  # arXiv 不给引用数
            "source": "arxiv",
        })
    return papers


# ───────────────────────── Semantic Scholar ─────────────────────────


async def _search_s2(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """Semantic Scholar Graph API. JSON. 覆盖全学科, 有引用数和 external DOI."""
    q = urllib.parse.quote(query)
    fields = "title,abstract,authors,year,venue,externalIds,citationCount"
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}"
        f"&limit={max_results}&fields={fields}"
    )
    try:
        data = await _http_get_json(url)
    except Exception as exc:
        logger.warning("Semantic Scholar 请求失败: %s", exc)
        return []

    papers: list[dict[str, Any]] = []
    for item in data.get("data", []) or []:
        year = item.get("year")
        if year_from and year and year < year_from:
            continue
        if year_to and year and year > year_to:
            continue
        ext = item.get("externalIds", {}) or {}
        doi = ext.get("DOI")
        arxiv_id = ext.get("ArXiv")
        url_ = item.get("url", "") or ""
        if not url_ and doi:
            url_ = f"https://doi.org/{doi}"
        elif not url_ and arxiv_id:
            url_ = f"https://arxiv.org/abs/{arxiv_id}"
        authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
        papers.append({
            "title": item.get("title", "") or "",
            "authors": authors,
            "year": year,
            "venue": item.get("venue", "") or "",
            "doi": doi,
            "abstract": item.get("abstract", "") or "",
            "url": url_,
            "citations": item.get("citationCount"),
            "source": "s2",
        })
    return papers


# ───────────────────────── CrossRef ─────────────────────────


async def _search_crossref(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """CrossRef Works API. JSON. DOI 权威源, 期刊文献全."""
    q = urllib.parse.quote(query)
    select = "DOI,title,author,issued,container-title,abstract,URL"
    url = (
        f"https://api.crossref.org/works?query={q}"
        f"&rows={max_results}&select={select}"
    )
    # CrossRef 支持 filter 参数做 year range
    filters = []
    if year_from:
        filters.append(f"from-pub-date:{year_from}-01-01")
    if year_to:
        filters.append(f"until-pub-date:{year_to}-12-31")
    if filters:
        url += "&filter=" + ",".join(filters)

    try:
        data = await _http_get_json(url)
    except Exception as exc:
        logger.warning("CrossRef 请求失败: %s", exc)
        return []

    papers: list[dict[str, Any]] = []
    items = (data.get("message") or {}).get("items", []) or []
    for item in items:
        title_list = item.get("title") or []
        title = title_list[0] if title_list else ""
        if not title:
            continue
        authors: list[str] = []
        for a in item.get("author", []) or []:
            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
            if name:
                authors.append(name)
        year: int | None = None
        issued = item.get("issued", {}) or {}
        date_parts = issued.get("date-parts") or [[]]
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
        venue_list = item.get("container-title") or []
        venue = venue_list[0] if venue_list else ""
        doi = item.get("DOI")
        abstract = item.get("abstract", "") or ""
        # CrossRef abstract 带 <jats:p> 之类的 XML 标签, 全剥掉
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()
        url_ = item.get("URL", "") or ""
        if not url_ and doi:
            url_ = f"https://doi.org/{doi}"
        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue,
            "doi": doi,
            "abstract": abstract,
            "url": url_,
            "citations": None,  # CrossRef 不给引用数
            "source": "crossref",
        })
    return papers


# ───────────────────────── OpenAlex ─────────────────────────


def _openalex_abstract_from_inverted_index(inv: dict[str, list[int]] | None) -> str:
    """OpenAlex 的 abstract 是倒排索引 (word -> [positions]), 重建回字符串."""
    if not inv:
        return ""
    # 算出最大位置, 定位数组长度
    max_pos = 0
    for positions in inv.values():
        for p in positions:
            if p > max_pos:
                max_pos = p
    words: list[str] = [""] * (max_pos + 1)
    for word, positions in inv.items():
        for p in positions:
            if 0 <= p <= max_pos:
                words[p] = word
    return " ".join(w for w in words if w)


async def _search_openalex(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """OpenAlex Works API. JSON. 覆盖全学科, 自带 cited_by_count 和 open_access.oa_url.

    免费, polite pool 靠 mailto 参数 (放 User-Agent 里也行).
    abstract 是倒排索引格式, 要重建.
    """
    q = urllib.parse.quote(query)
    # mailto 进 polite pool, 走更快的限流通道
    email = os.environ.get("HUGINN_UNPAYWALL_EMAIL", "user@example.com")
    fields = "id,doi,title,abstract_inverted_index,publication_year," \
             "cited_by_count,authorships,primary_location,open_access"
    url = (
        f"https://api.openalex.org/works?search={q}"
        f"&per-page={max_results}&select={fields}&mailto={email}"
    )
    # year range 走 filter 参数
    filters = []
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{year_to}-12-31")
    if filters:
        url += "&filter=" + ",".join(filters)

    try:
        data = await _http_get_json(url)
    except Exception as exc:
        logger.warning("OpenAlex 请求失败: %s", exc)
        return []

    papers: list[dict[str, Any]] = []
    for item in data.get("results", []) or []:
        title = item.get("title") or item.get("display_name") or ""
        if not title:
            continue
        year = item.get("publication_year")
        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in (item.get("authorships") or [])
            if (a.get("author") or {}).get("display_name")
        ]
        doi_raw = item.get("doi") or ""
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None
        venue = ""
        primary_loc = item.get("primary_location") or {}
        source = primary_loc.get("source") or {}
        venue = source.get("display_name") or ""
        abstract = _openalex_abstract_from_inverted_index(
            item.get("abstract_inverted_index")
        )
        oa = item.get("open_access") or {}
        oa_url = oa.get("oa_url") or ""
        # 优先用 oa_url, 没有就退回 doi.org
        url_ = oa_url
        if not url_ and doi:
            url_ = f"https://doi.org/{doi}"
        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue,
            "doi": doi,
            "abstract": abstract,
            "url": url_,
            "citations": item.get("cited_by_count"),
            "source": "openalex",
            "oa_url": oa_url,  # 留着, fetch_pdf 直接能用
            "is_oa": oa.get("is_oa"),
        })
    return papers


# ───────────────────────── PubMed (NCBI E-utilities) ─────────────────────────


async def _search_pubmed(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """PubMed via NCBI E-utilities. 两步: esearch 拿 PMID 列表, esummary 拿元数据.

    生物医学为主, 但也覆盖生物材料/纳米医学/电化学等材料相关方向.
    无 key, 限速 3 req/s. 返回 JSON.
    """
    q = urllib.parse.quote(query)
    # esearch: 搜 PMID
    esearch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={q}&retmax={max_results}&retmode=json"
    )
    # 加年份过滤
    if year_from or year_to:
        yf = year_from or 1900
        yt = year_to or 2100
        esearch_url += f'&mindate={yf}&maxdate={yt}&datetype=pdat'

    try:
        esearch_data = await _http_get_json(esearch_url)
    except Exception as exc:
        logger.warning("PubMed esearch 失败: %s", exc)
        return []

    pmids = (esearch_data.get("esearchresult") or {}).get("idlist", []) or []
    if not pmids:
        return []

    # esummary: 批量拿元数据
    pmid_str = ",".join(pmids)
    esummary_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pubmed&id={pmid_str}&retmode=json"
    )
    try:
        esummary_data = await _http_get_json(esummary_url)
    except Exception as exc:
        logger.warning("PubMed esummary 失败: %s", exc)
        return []

    papers: list[dict[str, Any]] = []
    result = esummary_data.get("result", {}) or {}
    for pmid in pmids:
        item = result.get(pmid, {}) or {}
        if not item.get("title"):
            continue
        title = item.get("title", "").strip()
        # 去掉 PMID 尾巴
        authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
        year: int | None = None
        pubdate = item.get("pubdate", "") or ""
        m = re.match(r"(\d{4})", pubdate)
        if m:
            year = int(m.group(1))
        venue = item.get("fulljournalname") or item.get("source", "") or ""
        doi = ""
        for aid in (item.get("articleids") or []):
            if aid.get("idtype") == "doi":
                doi = aid.get("value", "")
                break
        # PubMed esummary 不给 abstract, 得调 efetch; 这里先留空, fetch_pdf 可补
        url_ = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        if doi:
            url_ = f"https://doi.org/{doi}"
        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue,
            "doi": doi or None,
            "abstract": "",  # esummary 不给 abstract
            "url": url_,
            "citations": None,
            "source": "pubmed",
            "pmid": pmid,
        })
    return papers


# ───────────────────────── DOAJ ─────────────────────────


async def _search_doaj(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """DOAJ (Directory of Open Access Journals) API. 纯 OA 期刊, JSON.

    覆盖 2 万+ OA 期刊, 700 万+ 篇. 适合找 OA 全文.
    无 key. API: doaj.org/api/search/articles/{query}
    """
    q = urllib.parse.quote(query)
    page_size = min(max_results, 100)
    url = (
        f"https://doaj.org/api/search/articles/{q}"
        f"?page=1&pageSize={page_size}"
    )

    try:
        data = await _http_get_json(url)
    except Exception as exc:
        logger.warning("DOAJ 请求失败: %s", exc)
        return []

    papers: list[dict[str, Any]] = []
    for item in data.get("results", []) or []:
        bibjson = item.get("bibjson", {}) or {}
        title = bibjson.get("title", "") or ""
        if not title:
            continue
        authors = [
            a.get("name", "")
            for a in (bibjson.get("author") or [])
            if a.get("name")
        ]
        year: int | None = None
        m = re.match(r"(\d{4})", bibjson.get("year", "") or "")
        if m:
            year = int(m.group(1))
        if year_from and year and year < year_from:
            continue
        if year_to and year and year > year_to:
            continue
        journal = (bibjson.get("journal") or {})
        venue = journal.get("title", "") or ""
        doi = bibjson.get("doi")
        abstract = bibjson.get("abstract", "") or ""
        # DOAJ abstract 偶尔带 HTML 标签
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()
        links = bibjson.get("link") or []
        url_ = ""
        for link in links:
            if link.get("type") == "fulltext" or link.get("url"):
                url_ = link.get("url", "")
                break
        if not url_ and doi:
            url_ = f"https://doi.org/{doi}"
        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue,
            "doi": doi,
            "abstract": abstract,
            "url": url_,
            "citations": None,
            "source": "doaj",
        })
    return papers


# ───────────────────────── CORE ─────────────────────────


async def _search_core(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """CORE API v3. 全球最大 OA 语料, 含全文. JSON.

    无 key 可用 (限速: 5 单请求/10秒). 注册 key 更快.
    CORE 直接给 download_url, 是 fetch_pdf 的好补充 (不光元数据).
    """
    q = urllib.parse.quote(query)
    api_key = os.environ.get("HUGINN_CORE_API_KEY", "")
    # 无 key 也能用, 带 key 进 polite 路径
    headers_extra = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = (
        f"https://api.core.ac.uk/v3/search/works"
        f"?q={q}&limit={max_results}"
    )
    # year range
    if year_from:
        url += f'&year_created_min={year_from}-01-01'
    if year_to:
        url += f'&year_created_max={year_to}-12-31'

    def _fetch() -> dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **headers_extra})
        with _OPENER.open(req, timeout=_timeout()) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.warning("CORE 请求失败: %s", exc)
        return []

    papers: list[dict[str, Any]] = []
    for item in data.get("results", []) or []:
        title = item.get("title", "") or ""
        if not title:
            continue
        authors = [
            a.get("name", "")
            for a in (item.get("authors") or [])
            if isinstance(a, dict) and a.get("name")
        ]
        year: int | None = item.get("year_published")
        doi = item.get("doi")
        abstract = item.get("abstract", "") or ""
        # download_url 是 CORE 直接给的全文 PDF 链接, 留着给 fetch_pdf
        download_url = item.get("download_url") or ""
        source_urls = item.get("source_fulltext_urls") or []
        url_ = download_url or (source_urls[0] if source_urls else "")
        if not url_ and doi:
            url_ = f"https://doi.org/{doi}"
        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": item.get("publisher", "") or "",
            "doi": doi,
            "abstract": abstract,
            "url": url_,
            "citations": None,
            "source": "core",
            "download_url": download_url,  # fetch_pdf 直接能用
        })
    return papers
