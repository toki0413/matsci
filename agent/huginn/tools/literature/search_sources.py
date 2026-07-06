"""7 路学术搜索 + 去重 / 规范化.

arXiv / S2 / CrossRef / OpenAlex / PubMed / DOAJ / CORE 各自独立函数,
失败返回空列表不阻塞其他源. 去重以 DOI 优先, 无 DOI 走标题归一化.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import socket
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from huginn.security.external_breaker import CircuitOpenError, circuit_guard

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


def _is_safe_url(url: str) -> bool:
    """Block non-public URLs to prevent SSRF."""
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        try:
            ip = socket.getaddrinfo(hostname, None)[0][4][0]
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
                return False
        except (socket.gaierror, ValueError):
            return False
        return True
    except Exception:
        return False


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
        with circuit_guard("arxiv"):
            xml_text = await _http_get_text(url)
    except CircuitOpenError:
        logger.info("arXiv circuit open, skipping")
        return []
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
        with circuit_guard("semantic_scholar"):
            data = await _http_get_json(url)
    except CircuitOpenError:
        logger.info("Semantic Scholar circuit open, skipping")
        return []
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
        with circuit_guard("crossref"):
            data = await _http_get_json(url)
    except CircuitOpenError:
        logger.info("CrossRef circuit open, skipping")
        return []
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
        with circuit_guard("openalex"):
            data = await _http_get_json(url)
    except CircuitOpenError:
        logger.info("OpenAlex circuit open, skipping")
        return []
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
        with circuit_guard("pubmed"):
            esearch_data = await _http_get_json(esearch_url)
    except CircuitOpenError:
        logger.info("PubMed circuit open, skipping")
        return []
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
        with circuit_guard("pubmed"):
            esummary_data = await _http_get_json(esummary_url)
    except CircuitOpenError:
        logger.info("PubMed circuit open, skipping")
        return []
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
        with circuit_guard("doaj"):
            data = await _http_get_json(url)
    except CircuitOpenError:
        logger.info("DOAJ circuit open, skipping")
        return []
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
        if not _is_safe_url(url):
            raise RuntimeError("URL blocked by SSRF protection")
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **headers_extra})
        with _OPENER.open(req, timeout=_timeout()) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        with circuit_guard("core"):
            data = await asyncio.to_thread(_fetch)
    except CircuitOpenError:
        logger.info("CORE circuit open, skipping")
        return []
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


# ───────────────────────── Europe PMC ─────────────────────────


async def _search_europepmc(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """Europe PMC REST API. 33M+ publications, 10.2M full text.

    No API key required. Supports JSON output, citation counts,
    and open access filtering. Covers biomedical and materials science.
    """
    q = urllib.parse.quote(query)
    url = (
        f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query={q}&format=json&pageSize={min(max_results, 25)}"
        "&resultType=lite"
    )
    if year_from:
        url += f"&filter=FIRST_PDATE:[{year_from}-01-01 TO "
        if year_to:
            url += f"{year_to}-12-31]"
        else:
            url += "3000-12-31]"
    elif year_to:
        url += f"&filter=FIRST_PDATE:[1900-01-01 TO {year_to}-12-31]"

    try:
        with circuit_guard("europepmc"):
            data = await _http_get_json(url)
    except CircuitOpenError:
        logger.info("Europe PMC circuit open, skipping")
        return []
    except Exception as exc:
        logger.warning("Europe PMC 请求失败: %s", exc)
        return []

    papers: list[dict[str, Any]] = []
    for item in data.get("resultList", {}).get("result", []) or []:
        title = item.get("title", "") or ""
        if not title:
            continue
        authors: list[str] = []
        author_str = item.get("authorString", "") or ""
        if author_str:
            authors = [a.strip() for a in author_str.split(",") if a.strip()]
        year = item.get("pubYear")
        if year:
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None
        doi = item.get("doi") or None
        if doi:
            doi = doi.lower()
        abstract = ""
        pmid = item.get("pmid", "")
        pmcid = item.get("pmcid", "")
        url_ = item.get("journalInfo", {}).get("journal", {}).get("title", "")
        if doi:
            url_ = f"https://doi.org/{doi}"
        elif pmcid:
            url_ = f"https://europepmc.org/article/pmc/{pmcid}"
        elif pmid:
            url_ = f"https://europepmc.org/article/med/{pmid}"
        citations = item.get("citedByCount")
        if citations:
            try:
                citations = int(citations)
            except (ValueError, TypeError):
                citations = None
        is_oa = item.get("isOpenAccess") == "Y" or bool(pmcid)
        pdf_url = ""
        if pmcid:
            pdf_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": item.get("journalInfo", {}).get("journal", {}).get("title", "") or "",
            "doi": doi,
            "abstract": abstract,
            "url": url_,
            "citations": citations,
            "source": "europepmc",
            "open_access": is_oa,
            "download_url": pdf_url,
        })
    return papers


# ───────────────────────── Zenodo ─────────────────────────


async def _search_zenodo(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """Zenodo REST API. CERN's open data repository.

    No API key required for search. Returns datasets, software,
    publications, and figures. Many materials science datasets.
    Zenodo can be slow — we use a 45s timeout.
    """
    q = urllib.parse.quote(query)
    url = (
        f"https://zenodo.org/api/records"
        f"?q={q}&size={min(max_results, 25)}&sort=mostrecent"
        "&type=publication"
    )
    # year filtering via Zenodo's [date-range] syntax
    date_filter = ""
    if year_from:
        date_filter = f"[{year_from}-01-01 TO "
        date_filter += f"{year_to}-12-31]" if year_to else "2999-12-31]"
    elif year_to:
        date_filter = f"[1900-01-01 TO {year_to}-12-31]"
    if date_filter:
        url += f"&publication_date={date_filter}"

    def _fetch() -> dict[str, Any]:
        if not _is_safe_url(url):
            raise RuntimeError("URL blocked by SSRF protection")
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _OPENER.open(req, timeout=45) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        with circuit_guard("zenodo"):
            data = await asyncio.to_thread(_fetch)
    except CircuitOpenError:
        logger.info("Zenodo circuit open, skipping")
        return []
    except Exception as exc:
        logger.warning("Zenodo 请求失败: %s", exc)
        return []

    papers: list[dict[str, Any]] = []
    for item in data.get("hits", {}).get("hits", []) or []:
        meta = item.get("metadata", {})
        title = meta.get("title", "") or ""
        if not title:
            continue
        authors: list[str] = []
        for creator in meta.get("creators", []) or []:
            name = creator.get("name", "")
            if name:
                authors.append(name)
        year: int | None = None
        pub_date = meta.get("publication_date", "") or ""
        if pub_date:
            m = re.match(r"(\d{4})", pub_date)
            if m:
                year = int(m.group(1))
        doi = meta.get("doi") or None
        rec_id = item.get("id", "")
        url_ = f"https://zenodo.org/record/{rec_id}" if rec_id else ""
        if doi:
            url_ = f"https://doi.org/{doi}"
        abstract = meta.get("description", "") or ""
        # strip HTML tags from description
        if abstract:
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()[:500]
        upload_type = meta.get("upload_type", "publication")
        # find PDF download URL in files
        download_url = ""
        for f in item.get("files", []) or []:
            if f.get("type") == "pdf":
                download_url = f.get("links", {}).get("self", "")
                break
            if not download_url:
                download_url = f.get("links", {}).get("self", "")
        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": "Zenodo",
            "doi": doi,
            "abstract": abstract,
            "url": url_,
            "citations": None,
            "source": "zenodo",
            "open_access": meta.get("access_right") == "open",
            "download_url": download_url,
            "upload_type": upload_type,
        })
    return papers


# ───────────────────────── OpenAIRE ─────────────────────────


async def _search_openaire(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """OpenAIRE Search API. 150M+ EU open access publications.

    No API key required. Covers EU-funded research including
    Horizon 2020 / Horizon Europe materials science projects.
    """
    q = urllib.parse.quote(query)
    url = (
        f"https://api.openaire.eu/search/publications"
        f"?keywords={q}&size={min(max_results, 25)}&format=json"
    )
    if year_from:
        url += f"&fromDateAccepted={year_from}-01-01"
    if year_to:
        url += f"&untilDateAccepted={year_to}-12-31"

    def _fetch() -> dict[str, Any]:
        if not _is_safe_url(url):
            raise RuntimeError("URL blocked by SSRF protection")
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _OPENER.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        with circuit_guard("openaire"):
            data = await asyncio.to_thread(_fetch)
    except CircuitOpenError:
        logger.info("OpenAIRE circuit open, skipping")
        return []
    except Exception as exc:
        logger.warning("OpenAIRE 请求失败: %s", exc)
        return []

    papers: list[dict[str, Any]] = []
    # OpenAIRE wraps results under response -> results -> result
    results_root = data.get("response", {})
    results = results_root.get("results", {}).get("result", []) or []
    if isinstance(results, dict):
        results = [results]

    for wrapper in results:
        item = (wrapper.get("metadata", {})
                .get("oaf:entity", {})
                .get("oaf:result", {}))
        if not item:
            continue

        # Title — list of dicts with "$" key, pick the first non-empty
        title = ""
        title_field = item.get("title", [])
        if isinstance(title_field, list):
            for t in title_field:
                if isinstance(t, dict) and t.get("$"):
                    title = t["$"].strip()
                    break
        elif isinstance(title_field, dict) and title_field.get("$"):
            title = title_field["$"].strip()
        if not title:
            continue

        # Authors — under "creator" field
        authors: list[str] = []
        creators = item.get("creator", [])
        if isinstance(creators, list):
            for c in creators:
                if isinstance(c, dict) and c.get("$"):
                    authors.append(c["$"])
        elif isinstance(creators, dict) and creators.get("$"):
            authors.append(creators["$"])

        # Year
        year: int | None = None
        date_field = item.get("dateofacceptance", {})
        if isinstance(date_field, dict):
            date_str = date_field.get("$", "") or ""
            if date_str:
                m = re.match(r"(\d{4})", date_str)
                if m:
                    year = int(m.group(1))

        # DOI — under "pid" field (dict with @classid="doi")
        doi = None
        pid = item.get("pid", {})
        if isinstance(pid, dict):
            if pid.get("@classid") == "doi" and pid.get("$"):
                doi = pid["$"].lower()
        elif isinstance(pid, list):
            for p in pid:
                if isinstance(p, dict) and p.get("@classid") == "doi" and p.get("$"):
                    doi = p["$"].lower()
                    break

        # URL
        url_ = ""
        instances = item.get("instance", [])
        if isinstance(instances, list):
            for inst in instances:
                if isinstance(inst, dict):
                    webres = inst.get("webresource", {})
                    if isinstance(webres, dict):
                        url_ = webres.get("url", {}).get("$", "") or ""
                        if url_:
                            break
        elif isinstance(instances, dict):
            webres = instances.get("webresource", {})
            url_ = webres.get("url", {}).get("$", "") or ""
        if not url_ and doi:
            url_ = f"https://doi.org/{doi}"

        # Abstract
        abstract = ""
        desc = item.get("description", {})
        if isinstance(desc, dict) and desc.get("$"):
            abstract = desc["$"][:500]
        elif isinstance(desc, list):
            for d in desc:
                if isinstance(d, dict) and d.get("$"):
                    abstract = d["$"][:500]
                    break

        # Citation count — under "measure" list with @id="citationCount"
        citations = None
        measures = item.get("measure", [])
        if isinstance(measures, list):
            for m_entry in measures:
                if isinstance(m_entry, dict) and m_entry.get("@id") == "citationCount":
                    try:
                        citations = int(float(m_entry.get("@score", "0")))
                    except (ValueError, TypeError):
                        pass
                    break

        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": item.get("publisher", {}).get("$", "") or "OpenAIRE",
            "doi": doi,
            "abstract": abstract,
            "url": url_,
            "citations": citations,
            "source": "openaire",
            "open_access": True,
        })
    return papers


# ───────────────────────── Crystallography Open Database ─────────────────────────


async def _search_cod(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """Crystallography Open Database (COD) REST API.

    Free, no API key. 500k+ crystal structures with CIF files.
    Tries formula search first (COD's strongest feature), falls back
    to text/mineral search.
    """
    q = urllib.parse.quote(query)
    limit = min(max_results, 25)

    # Try formula search first — COD is primarily a structure database,
    # and formula queries are its most reliable interface.
    formula_url = (
        f"http://www.crystallography.net/cod/result"
        f"?format=json&count={limit}&formula={q}"
    )

    def _fetch(url: str) -> Any:
        if not _is_safe_url(url):
            raise RuntimeError("URL blocked by SSRF protection")
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _OPENER.open(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        with circuit_guard("cod"):
            data = await asyncio.to_thread(_fetch, formula_url)
    except CircuitOpenError:
        logger.info("COD circuit open, skipping")
        return []
    except Exception as exc:
        logger.warning("COD formula 请求失败: %s", exc)
        data = []

    structures: list = []
    if isinstance(data, dict):
        structures = data.get("structures", data.get("result", []))
    elif isinstance(data, list):
        structures = data

    # Fall back to text search when formula yields nothing.
    if not structures:
        text_url = (
            f"http://www.crystallography.net/cod/result"
            f"?format=json&count={limit}&text={q}"
        )
        try:
            with circuit_guard("cod"):
                data = await asyncio.to_thread(_fetch, text_url)
        except CircuitOpenError:
            logger.info("COD circuit open, skipping text search")
            data = []
        except Exception as exc:
            logger.warning("COD text 请求失败: %s", exc)
            data = []
        if isinstance(data, dict):
            structures = data.get("structures", data.get("result", []))
        elif isinstance(data, list):
            structures = data

    papers: list[dict[str, Any]] = []
    for item in structures:
        if not isinstance(item, dict):
            continue
        file_id = item.get("file", "") or ""
        formula = item.get("formula", "") or ""
        mineral = item.get("mineral", "") or ""
        title = mineral or formula or f"COD entry {file_id}"
        if not title:
            continue
        # Build URL to the CIF file
        if file_id:
            # COD stores files in directory hierarchy based on ID
            dir_path = file_id[:1] + "/" + file_id[1:3] + "/"
            url_ = f"http://www.crystallography.net/cod/{dir_path}{file_id}.cif"
        else:
            url_ = ""
        sg = item.get("sg", "") or ""
        a = item.get("a", "") or ""
        b = item.get("b", "") or ""
        c = item.get("c", "") or ""
        abstract_parts = []
        if formula:
            abstract_parts.append(f"Formula: {formula}")
        if sg:
            abstract_parts.append(f"Space group: {sg}")
        if a:
            abstract_parts.append(f"a={a} b={b} c={c}")
        abstract = "; ".join(abstract_parts)
        papers.append({
            "title": title,
            "authors": [],
            "year": None,
            "venue": "Crystallography Open Database",
            "doi": None,
            "abstract": abstract,
            "url": url_,
            "citations": None,
            "source": "cod",
            "open_access": True,
            "download_url": url_,
            "formula": formula,
            "space_group": sg,
        })
    return papers


# ───────────────────────── Materials Cloud ─────────────────────────


async def _search_materials_cloud(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """Materials Cloud API. EPFL's computational materials platform.

    Free, no API key. Contains curated datasets, workflows, and
    computational materials data (DFT, MD, ML potentials).
    Uses the Invenio REST API at archive.materialscloud.org.
    """
    q = urllib.parse.quote(query)
    # Materials Cloud runs Invenio; the REST API is under /api/records
    url = (
        f"https://archive.materialscloud.org/api/records"
        f"?q={q}&size={min(max_results, 25)}&sort=newest"
    )

    def _fetch() -> dict[str, Any]:
        if not _is_safe_url(url):
            raise RuntimeError("URL blocked by SSRF protection")
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _OPENER.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        with circuit_guard("materials_cloud"):
            data = await asyncio.to_thread(_fetch)
    except CircuitOpenError:
        logger.info("Materials Cloud circuit open, skipping")
        return []
    except Exception as exc:
        logger.warning("Materials Cloud 请求失败: %s", exc)
        return []

    papers: list[dict[str, Any]] = []
    for item in data.get("hits", {}).get("hits", []) or []:
        meta = item.get("metadata", {})
        title = meta.get("title", "") or ""
        if not title:
            continue
        # Invenio v3: creators live under person_or_org.name
        authors: list[str] = []
        for creator in meta.get("creators", []) or []:
            if not isinstance(creator, dict):
                continue
            porg = creator.get("person_or_org", {})
            name = (porg or {}).get("name", "") if isinstance(porg, dict) else ""
            if name:
                authors.append(name)
        year: int | None = None
        pub_date = meta.get("publication_date", "") or ""
        if pub_date:
            m = re.match(r"(\d{4})", pub_date)
            if m:
                year = int(m.group(1))
        # Invenio v3: DOI is under pids.doi.identifier
        doi = None
        pids = item.get("pids", {}) or {}
        if isinstance(pids, dict):
            doi_info = pids.get("doi", {})
            if isinstance(doi_info, dict):
                doi = doi_info.get("identifier", "") or None
        if not doi:
            doi = meta.get("doi") or None
        if doi:
            doi = doi.lower()
        rec_id = item.get("id", "")
        url_ = f"https://archive.materialscloud.org/record/{rec_id}" if rec_id else ""
        if doi:
            url_ = f"https://doi.org/{doi}"
        abstract = meta.get("description", "") or ""
        if abstract:
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()[:500]
        # Invenio v3: files is a dict (metadata), not a list of file objects.
        # Download links are under links.files or links.self.
        download_url = ""
        links = item.get("links", {}) or {}
        if isinstance(links, dict):
            download_url = links.get("files", "") or links.get("self", "")
        # access control moved to a separate dict in v3
        access = item.get("access", {}) or {}
        is_open = True
        if isinstance(access, dict):
            is_open = access.get("record", "") != "restricted"
        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": "Materials Cloud",
            "doi": doi,
            "abstract": abstract,
            "url": url_,
            "citations": None,
            "source": "materials_cloud",
            "open_access": is_open,
            "download_url": download_url,
        })
    return papers


# ───────────────────────── NOMAD ─────────────────────────


async def _search_nomad(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """NOMAD (Novel Materials Discovery) API. 19M+ computational entries.

    Free, no API key. Hosted by FAIRmat/MPSD. Covers DFT, MD, GW,
    ML potentials from VASP, Quantum ESPRESSO, FHI-aims, CASTEP, etc.
    Search via the /entries endpoint with ?search= parameter.
    """
    q = urllib.parse.quote(query)
    page_size = min(max_results, 25)
    url = (
        f"https://nomad-lab.eu/prod/v1/api/v1/entries"
        f"?search={q}&page_size={page_size}&owner=public"
    )

    def _fetch() -> dict[str, Any]:
        if not _is_safe_url(url):
            raise RuntimeError("URL blocked by SSRF protection")
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _OPENER.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        with circuit_guard("nomad"):
            data = await asyncio.to_thread(_fetch)
    except CircuitOpenError:
        logger.info("NOMAD circuit open, skipping")
        return []
    except Exception as exc:
        logger.warning("NOMAD 请求失败: %s", exc)
        return []

    total = (data.get("pagination") or {}).get("total", 0)
    logger.info("NOMAD search '%s': %d total entries, returning %d",
                query, total, len(data.get("data", [])))

    papers: list[dict[str, Any]] = []
    for entry in data.get("data", []) or []:
        results = entry.get("results", {}) or {}
        material = results.get("material", {}) or {}
        method = results.get("method", {}) or {}

        formula = material.get("chemical_formula_descriptive", "") or ""
        formula_reduced = material.get("chemical_formula_reduced", "") or ""
        elements = material.get("elements", []) or []
        material_id = material.get("material_id", "") or ""

        # method info
        workflow = method.get("workflow_name", "") or ""
        code = method.get("program_name", "") or ""

        entry_id = entry.get("entry_id", "") or ""
        upload_id = entry.get("upload_id", "") or ""

        title = f"{formula} ({workflow}, {code})" if formula else f"NOMAD entry {entry_id}"

        # Build URL to NOMAD entry
        url_ = f"https://nomad-lab.eu/prod/v1/gui/entry/id/{entry_id}" if entry_id else ""

        abstract_parts = []
        if formula:
            abstract_parts.append(f"Formula: {formula}")
        if elements:
            abstract_parts.append(f"Elements: {', '.join(elements)}")
        if workflow:
            abstract_parts.append(f"Method: {workflow}")
        if code:
            abstract_parts.append(f"Code: {code}")
        n_atoms = material.get("n_atoms")
        if n_atoms:
            abstract_parts.append(f"Atoms: {n_atoms}")
        space_group = (material.get("symmetry") or {}).get("space_group_number")
        if space_group:
            abstract_parts.append(f"Space group: {space_group}")
        abstract = "; ".join(abstract_parts)

        papers.append({
            "title": title,
            "authors": [],
            "year": None,
            "venue": "NOMAD",
            "doi": None,
            "abstract": abstract,
            "url": url_,
            "citations": None,
            "source": "nomad",
            "open_access": True,
            "formula": formula,
            "formula_reduced": formula_reduced,
            "elements": elements,
            "material_id": material_id,
            "entry_id": entry_id,
            "method": workflow,
            "code": code,
            "total_entries": total,
        })
    return papers


# ───────────────────────── DataCite ─────────────────────────


async def _search_datacite(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """DataCite REST API. DOI registry for research datasets.

    Free, no API key. Covers datasets, software, and other research
    outputs registered with DataCite. Great for finding raw data,
    computational results, and supplementary materials.
    """
    q = urllib.parse.quote(query)
    page_size = min(max_results, 25)
    url = (
        f"https://api.datacite.org/dois"
        f"?page[size]={page_size}&query={q}"
    )
    if year_from:
        url += f"&filter=publicationYear>={year_from}"
    if year_to:
        url += f"&filter=publicationYear<={year_to}"

    def _fetch() -> dict[str, Any]:
        if not _is_safe_url(url):
            raise RuntimeError("URL blocked by SSRF protection")
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _OPENER.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        with circuit_guard("datacite"):
            data = await asyncio.to_thread(_fetch)
    except CircuitOpenError:
        logger.info("DataCite circuit open, skipping")
        return []
    except Exception as exc:
        logger.warning("DataCite 请求失败: %s", exc)
        return []

    papers: list[dict[str, Any]] = []
    for item in data.get("data", []) or []:
        attrs = item.get("attributes", {}) or {}
        # titles is a list of {title: "...", titleType: "..."}
        title = ""
        titles = attrs.get("titles", []) or []
        for t in titles:
            if isinstance(t, dict) and t.get("title"):
                title = t["title"]
                break
        if not title:
            continue

        # creators: list of {name: "...", nameType: "Personal"/"Organizational"}
        authors: list[str] = []
        for c in attrs.get("creators", []) or []:
            if isinstance(c, dict) and c.get("name"):
                authors.append(c["name"])

        year = attrs.get("publicationYear")

        doi = attrs.get("doi", "") or ""
        if doi:
            doi = doi.lower()

        url_ = attrs.get("url", "") or ""
        if not url_ and doi:
            url_ = f"https://doi.org/{doi}"

        # descriptions: list of {description: "...", descriptionType: "Abstract"}
        abstract = ""
        for d in attrs.get("descriptions", []) or []:
            if isinstance(d, dict) and d.get("description"):
                abstract = d["description"][:500]
                break

        # subjects for keyword tagging
        subjects = []
        for s in attrs.get("subjects", []) or []:
            if isinstance(s, dict) and s.get("subject"):
                subjects.append(s["subject"])

        resource_type = ""
        types = attrs.get("types", {}) or {}
        if isinstance(types, dict):
            resource_type = types.get("resourceTypeGeneral", "") or ""

        papers.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": attrs.get("publisher", "") or "DataCite",
            "doi": doi,
            "abstract": abstract,
            "url": url_,
            "citations": None,
            "source": "datacite",
            "open_access": True,
            "resource_type": resource_type,
            "subjects": subjects[:10],
        })
    return papers


# ───────────────────────── Materials Project ─────────────────────────


async def _search_materials_project(
    query: str, max_results: int, year_from: int | None, year_to: int | None
) -> list[dict[str, Any]]:
    """Materials Project API. 150k+ computed materials with properties.

    Requires API key (free registration at materialsproject.org).
    Set HUGINN_MP_API_KEY env var. Without key, returns empty list.
    Properties include band gap, formation energy, elastic tensor, etc.
    """
    api_key = os.environ.get("HUGINN_MP_API_KEY", "")
    if not api_key:
        logger.info("Materials Project skipped: no HUGINN_MP_API_KEY set")
        return []

    q = urllib.parse.quote(query)
    limit = min(max_results, 25)
    url = (
        f"https://api.materialsproject.org/materials/summary/"
        f"?_limit={limit}&_api_key={api_key}"
    )
    # year filtering not supported by MP; skip silently

    def _fetch() -> Any:
        if not _is_safe_url(url):
            raise RuntimeError("URL blocked by SSRF protection")
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _OPENER.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        with circuit_guard("materials_project"):
            data = await asyncio.to_thread(_fetch)
    except CircuitOpenError:
        logger.info("Materials Project circuit open, skipping")
        return []
    except Exception as exc:
        logger.warning("Materials Project 请求失败: %s", exc)
        return []

    # MP returns a list of material summary dicts
    if isinstance(data, dict):
        materials = data.get("data", [])
    elif isinstance(data, list):
        materials = data
    else:
        materials = []

    papers: list[dict[str, Any]] = []
    for m in materials:
        if not isinstance(m, dict):
            continue
        formula = m.get("formula_pretty", "") or m.get("full_formula", "") or ""
        if not formula:
            continue
        material_id = m.get("material_id", "") or ""
        title = f"{formula} (MP-{material_id})" if material_id else formula

        url_ = f"https://nextgen.materialsproject.org/materials/{material_id}" if material_id else ""

        abstract_parts = [f"Formula: {formula}"]
        band_gap = m.get("band_gap")
        if band_gap is not None:
            abstract_parts.append(f"Band gap: {band_gap:.3f} eV")
        formation_energy = m.get("formation_energy_per_atom")
        if formation_energy is not None:
            abstract_parts.append(f"Formation energy: {formation_energy:.4f} eV/atom")
        energy_above_hull = m.get("energy_above_hull")
        if energy_above_hull is not None:
            abstract_parts.append(f"Above hull: {energy_above_hull:.4f} eV")
        density = m.get("density")
        if density is not None:
            abstract_parts.append(f"Density: {density:.2f} g/cm3")
        symmetry = m.get("symmetry", {}) or {}
        crystal_system = symmetry.get("crystal_system", "") or ""
        space_group = symmetry.get("number", "") or ""
        if crystal_system:
            abstract_parts.append(f"Crystal system: {crystal_system}")
        if space_group:
            abstract_parts.append(f"Space group: {space_group}")
        deprecated = m.get("deprecated", False)
        if deprecated:
            abstract_parts.append("[DEPRECATED]")
        abstract = "; ".join(abstract_parts)

        papers.append({
            "title": title,
            "authors": [],
            "year": None,
            "venue": "Materials Project",
            "doi": None,
            "abstract": abstract,
            "url": url_,
            "citations": None,
            "source": "materials_project",
            "open_access": True,
            "formula": formula,
            "material_id": material_id,
            "band_gap": band_gap,
            "formation_energy": formation_energy,
            "crystal_system": crystal_system,
        })
    return papers


# ───────────────────────── OpenCitations ─────────────────────────


async def _opencitations_references(
    doi: str, limit: int = 25
) -> list[dict[str, Any]]:
    """OpenCitations COCI API — backward references (papers this DOI cites).

    Free, no API key. Returns DOI-level citation edges without metadata.
    We enrich with CrossRef DOI lookup when possible.
    """
    doi_clean = doi.lower().strip()
    encoded = urllib.parse.quote(doi_clean, safe="")
    url = f"https://opencitations.net/index/coci/api/v1/references/{encoded}"

    def _fetch() -> Any:
        if not _is_safe_url(url):
            raise RuntimeError("URL blocked by SSRF protection")
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _OPENER.open(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        with circuit_guard("opencitations"):
            data = await asyncio.to_thread(_fetch)
    except CircuitOpenError:
        logger.info("OpenCitations circuit open, skipping references")
        return []
    except Exception as exc:
        logger.warning("OpenCitations references 请求失败: %s", exc)
        return []

    if not isinstance(data, list):
        return []

    refs: list[dict[str, Any]] = []
    for item in data[:limit]:
        if not isinstance(item, dict):
            continue
        cited_doi = (item.get("cited") or "").lower().strip()
        if not cited_doi:
            continue
        refs.append({
            "doi": cited_doi,
            "title": "",
            "authors": [],
            "year": None,
            "venue": "",
            "citations": None,
            "url": f"https://doi.org/{cited_doi}",
            "source": "opencitations",
            "creation": item.get("creation", ""),
            "timespan": item.get("timespan", ""),
        })
    return refs


async def _opencitations_citations(
    doi: str, limit: int = 25
) -> list[dict[str, Any]]:
    """OpenCitations COCI API — forward citations (papers that cite this DOI).

    Free, no API key. Returns DOI-level citation edges.
    """
    doi_clean = doi.lower().strip()
    encoded = urllib.parse.quote(doi_clean, safe="")
    url = f"https://opencitations.net/index/coci/api/v1/citations/{encoded}"

    def _fetch() -> Any:
        if not _is_safe_url(url):
            raise RuntimeError("URL blocked by SSRF protection")
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _OPENER.open(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        with circuit_guard("opencitations"):
            data = await asyncio.to_thread(_fetch)
    except CircuitOpenError:
        logger.info("OpenCitations circuit open, skipping citations")
        return []
    except Exception as exc:
        logger.warning("OpenCitations citations 请求失败: %s", exc)
        return []

    if not isinstance(data, list):
        return []

    cites: list[dict[str, Any]] = []
    for item in data[:limit]:
        if not isinstance(item, dict):
            continue
        citing_doi = (item.get("citing") or "").lower().strip()
        if not citing_doi:
            continue
        cites.append({
            "doi": citing_doi,
            "title": "",
            "authors": [],
            "year": None,
            "venue": "",
            "citations": None,
            "url": f"https://doi.org/{citing_doi}",
            "source": "opencitations",
            "creation": item.get("creation", ""),
            "timespan": item.get("timespan", ""),
        })
    return cites


async def _crossref_doi_lookup(doi: str) -> dict[str, Any] | None:
    """Fetch paper metadata from CrossRef by DOI. Used to enrich OpenCitations results."""
    doi_clean = doi.lower().strip()
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi_clean, safe='')}"
    try:
        with circuit_guard("crossref"):
            data = await _http_get_json(url)
    except CircuitOpenError:
        return None
    except Exception:
        return None

    item = (data.get("message") or {})
    if not item:
        return None
    title_list = item.get("title") or []
    title = title_list[0] if title_list else ""
    if not title:
        return None
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
    abstract = re.sub(r"<[^>]+>", "", (item.get("abstract") or "")).strip()
    return {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi_clean,
        "abstract": abstract,
        "url": item.get("URL", "") or f"https://doi.org/{doi_clean}",
        "citations": None,
        "source": "crossref",
    }


async def _enrich_with_crossref(
    papers: list[dict[str, Any]], batch_size: int = 5
) -> list[dict[str, Any]]:
    """Enrich a list of paper dicts (with DOIs but no titles) via CrossRef lookups.

    OpenCitations only returns DOI edges — no titles or authors. This function
    batches CrossRef /works/{doi} lookups to fill in metadata. Papers that
    already have titles are left alone. Papers where CrossRef fails keep their
    DOI-only stub.
    """
    to_enrich = [p for p in papers if p.get("doi") and not p.get("title")]
    if not to_enrich:
        return papers

    # Process in small batches to avoid hammering CrossRef
    for i in range(0, len(to_enrich), batch_size):
        batch = to_enrich[i:i + batch_size]
        tasks = [asyncio.create_task(_crossref_doi_lookup(p["doi"])) for p in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for p, res in zip(batch, results):
            if isinstance(res, dict) and res.get("title"):
                p["title"] = res["title"]
                p["authors"] = res.get("authors", [])
                p["year"] = res.get("year")
                p["venue"] = res.get("venue", "")
                p["abstract"] = res.get("abstract", "")
                p["url"] = res.get("url", p.get("url", ""))
    return papers
