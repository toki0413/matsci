"""自适应元素追踪 — 受 Scrapling adaptive scraping 启发.

网站改版后 CSS/XPath 选择器可能失效. 这个模块用 SQLite 存储元素的
结构指纹 (tag/text/attrs/position), 改版后用相似度评分重新定位.

使用场景:
  - 知识库自动入库: 记录数据所在元素的选择器 + 指纹, 下次抓取时自适应
  - 文献追踪: 期刊页面改版后自动找到同样的数据元素
  - browser_tool extract: 提取后存指纹, 下次自适应查找

不依赖 lxml, 用纯 Python + sqlite3, 兼容 Playwright/Selenium 输出.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlparse

from huginn.utils.common import hash_text

logger = logging.getLogger(__name__)

_DEFAULT_DB = os.path.expanduser("~/.huginn/element_fingerprints.db")
_lock = threading.Lock()


def _get_db_path() -> str:
    return os.environ.get("HUGINN_FP_DB", _DEFAULT_DB)


def _ensure_db(db_path: str) -> None:
    """建表, 幂等."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS element_fingerprints (
                id INTEGER PRIMARY KEY,
                domain TEXT NOT NULL,
                identifier TEXT NOT NULL,
                tag TEXT,
                text_hash TEXT,
                text_preview TEXT,
                attrs_hash TEXT,
                attrs_json TEXT,
                selector TEXT,
                xpath TEXT,
                position TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (domain, identifier)
            )
        """"")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fp_domain ON element_fingerprints(domain)"
        )
        conn.commit()
    finally:
        conn.close()


def _element_to_dict(element: dict[str, Any]) -> dict[str, Any]:
    """从元素信息提取指纹数据."""
    tag = (element.get("tag") or "").lower()
    text = (element.get("text") or "").strip()[:200]
    attrs = element.get("attributes") or element.get("attrib") or {}

    # 过滤掉动态属性 (nonce, data-reactid 等)
    stable_attrs = {
        k: v for k, v in attrs.items()
        if not k.startswith("data-react") and k not in ("nonce", "data-key")
    }

    attrs_str = str(sorted(stable_attrs.items()))
    return {
        "tag": tag,
        "text": text,
        "text_hash": hash_text(text) if text else "",
        "text_preview": text[:80],
        "attrs_hash": hash_text(attrs_str) if stable_attrs else "",
        "attrs_json": str(stable_attrs)[:500],
        "selector": element.get("selector", ""),
        "xpath": element.get("xpath", ""),
        "position": element.get("position", ""),
    }


def save_fingerprint(
    domain: str,
    identifier: str,
    element: dict[str, Any],
    db_path: str | None = None,
) -> None:
    """保存元素指纹. domain + identifier 唯一标识.

    Args:
        domain: 网站域名 (如 "nature.com")
        identifier: 元素标识 (如 "article_title" / "abstract_text")
        element: 元素信息 dict, 至少包含 tag/text/attributes
    """
    if not domain or not identifier:
        return

    path = db_path or _get_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fp = _element_to_dict(element)

    with _lock:
        _ensure_db(path)
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO element_fingerprints
                (domain, identifier, tag, text_hash, text_preview,
                 attrs_hash, attrs_json, selector, xpath, position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    domain, identifier,
                    fp["tag"], fp["text_hash"], fp["text_preview"],
                    fp["attrs_hash"], fp["attrs_json"],
                    fp["selector"], fp["xpath"], fp["position"],
                ),
            )
            conn.commit()
        except Exception:
            logger.debug("save_fingerprint failed", exc_info=True)
        finally:
            conn.close()


def retrieve_fingerprint(
    domain: str,
    identifier: str,
    db_path: str | None = None,
) -> dict[str, Any] | None:
    """取出之前保存的指纹."""
    path = db_path or _get_db_path()
    if not os.path.exists(path):
        return None

    with _lock:
        conn = sqlite3.connect(path)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM element_fingerprints WHERE domain=? AND identifier=?",
                (domain, identifier),
            ).fetchone()
            return dict(row) if row else None
        except Exception:
            return None
        finally:
            conn.close()


def find_best_match(
    domain: str,
    candidates: list[dict[str, Any]],
    identifier: str,
    db_path: str | None = None,
) -> tuple[dict[str, Any] | None, float]:
    """在候选元素中找与历史指纹最相似的.

    Args:
        domain: 网站域名
        candidates: 当前页面上的候选元素列表
        identifier: 之前保存的标识

    Returns:
        (best_match_element, similarity_score) 或 (None, 0.0)
    """
    stored = retrieve_fingerprint(domain, identifier, db_path)
    if stored is None or not candidates:
        return None, 0.0

    best_score = 0.0
    best_match: dict[str, Any] | None = None

    for candidate in candidates:
        candidate_fp = _element_to_dict(candidate)
        score = _similarity(stored, candidate_fp)
        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match, best_score


def _similarity(stored: dict[str, Any], candidate: dict[str, Any]) -> float:
    """计算两个指纹的相似度, 返回 0.0-1.0.

    权重: tag (20%) + attrs_hash (30%) + text (50%)
    """
    score = 0.0

    # tag 匹配
    if stored.get("tag") and stored["tag"] == candidate.get("tag"):
        score += 0.2

    # attrs hash 精确匹配
    if stored.get("attrs_hash") and stored["attrs_hash"] == candidate.get("attrs_hash"):
        score += 0.3
    elif stored.get("attrs_hash") and candidate.get("attrs_hash"):
        # 部分匹配: 比较 attrs_json 的相似度
        ratio = SequenceMatcher(
            None, stored.get("attrs_json", ""), candidate.get("attrs_json", "")
        ).ratio()
        score += 0.3 * ratio

    # text 相似度
    stored_text = stored.get("text_preview", "")
    candidate_text = candidate.get("text_preview", "")
    if stored_text and candidate_text:
        ratio = SequenceMatcher(None, stored_text, candidate_text).ratio()
        score += 0.5 * ratio
    elif stored_text == candidate_text:
        score += 0.5

    return min(score, 1.0)


def extract_domain(url: str) -> str:
    """从 URL 提取域名."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


# ── 便捷接口 ──────────────────────────────────────────

def save_from_extraction(
    url: str,
    identifier: str,
    elements: list[dict[str, Any]],
    db_path: str | None = None,
) -> None:
    """从 browser_tool / web_search_tool 的提取结果保存指纹.

    取第一个元素的指纹作为 identifier 的代表.
    """
    domain = extract_domain(url)
    if not domain or not elements:
        return
    save_fingerprint(domain, identifier, elements[0], db_path)


def adaptive_extract(
    url: str,
    identifier: str,
    candidates: list[dict[str, Any]],
    threshold: float = 0.4,
    db_path: str | None = None,
) -> dict[str, Any] | None:
    """自适应提取: 用历史指纹在候选元素中找最佳匹配.

    Args:
        url: 当前页面 URL
        identifier: 之前保存的元素标识
        candidates: 当前页面上的候选元素
        threshold: 最低相似度阈值

    Returns:
        最佳匹配的元素 dict (含 _similarity_score), 或 None
    """
    domain = extract_domain(url)
    if not domain:
        return None

    match, score = find_best_match(domain, candidates, identifier, db_path)
    if match is None or score < threshold:
        return None

    match["_similarity_score"] = round(score, 3)
    match["_matched_identifier"] = identifier
    return match
