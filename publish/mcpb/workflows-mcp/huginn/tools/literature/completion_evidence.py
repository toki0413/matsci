"""缺度追问的取证 + 决策档 + 门禁不变量 — 三条沉淀 (AICC / OpenAI4S 启发).

背景:
  补全检索目前只在内存里返回 "补了/补后缺度变了", 没有可复核的审计轨迹, 也
  没有 "补后仍缺怎么办" 的显式处置。本模块把三条原则钉进结构层:

  1) 对象级取证 (make_evidence / attach_evidence):
     给 reported / 补全结果绑定**强引用证据标签** (fid + source + 原文 sha256
     快照 + snippet)。让 "这条值来自哪段原文" 成为随结果流动、可独立核实的
     强引用, 而非事后声明。
     纪律 (OpenAI4S kernel/provenance.py): "provenance 错比缺失更糟, 因为它
     会被相信"。取证必须来自源, 拿不到原文就显式标记未取到, 不臆造。

  2) 显式豁免 + 决策档 (build_waivers / DecisionLedger):
     补后仍缺的关键自由度**不再 silent 接受**: 落一条 waiver 决策 (含 dim,
     决策 id, reason, references, before/after)。决策档是追加式 JSONL + 哈希
     链, 幂等 (同输入必同决策 id, 重跑不重复)。
     门禁 (AICC classify_claim/validate_gate): 豁免不是被吞掉的例外, 是另一种
     必须可追溯的正式状态, 每条豁免都附理由。

  3) 评审门禁不变量 (assess_gate):
     AICC validate_gate 核心不变量 "status: pass 禁带 blocking_issues"。映射
     到缺度追问: 关键缺度 (urgency>=2) 补后仍缺, 必须有一条 waiver 覆盖才能
     pass; 否则 needs_waiver。矛盾状态 (pass 但仍有未豁免的关键缺度) 不被允许。

设计约束:
  - 纯函数 + 可注入落盘路径, 零网络/零 LLM, 幂等 (同输入必同输出)。
  - 不绑定全局单例/事件总线; 落盘路径由调用方注入, 便于测试与隔离。
  - 与 condition_normalize/query_completion 一致: 不臆造, 缺就显式暴露。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_GENESIS_HASH = "0" * 64
_EVIDENCE_SNIPPET_CHARS = 256
_WAIVER_VERSION = 1

# 关键自由度判定阈值 (对齐 query_completion._URGENCY: >=2 影响整体互洽判定)
CRITICAL_URGENCY = 2


# ─────────────────────────── 1) 对象级取证 ───────────────────────────


def sha256_text(text: str | None) -> str:
    """原文 sha256 (空/None → 空串, 表示"未取到原文", 不伪造)."""
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def make_evidence(
    source_id: str,
    source_text: str | None,
    index: int = 1,
) -> dict[str, Any]:
    """为单条值构造强引用证据标签.

    Args:
        source_id: 稳定的来源标识 (DOI / 标题 / 论文 id)。
        source_text: 取自原文的文本 (全文/摘要片段)。
        index: 同一来源的第几条。

    Returns:
        {"fid", "source_id", "sha256", "snippet"}。sha256 为空串表示原文未取到
        (诚实标记, 不臆造)。
    """
    fid = f"{source_id}#v{index}" if source_id else f"_anonymous#{index}"
    raw = (source_text or "").strip()
    snippet = raw[:_EVIDENCE_SNIPPET_CHARS]
    return {
        "fid": fid,
        "source_id": source_id or "",
        "sha256": sha256_text(raw),
        "snippet": snippet,
    }


def attach_evidence(
    reported: list[dict[str, Any]],
    papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """给每条 reported 的数值绑定到来源论文的证据标签 (对象级取证).

    reported 条目含 source_paper(s) / doi / value; 从 papers 里按 doi/标题
    匹配回原文文本, 生成 evidence 附进该条 (浅拷贝 + evidence 键)。
    匹配不上或 papers 无原文 → evidence.sha256="" (未取到, 不伪造)。

    Returns: 新列表, 每条多一个 "evidence" 键 (可能是 None)。
    """
    # 建立 doi/标题 → 原文 的查找
    lookup: dict[str, str] = {}
    for p in papers:
        doi = (p.get("doi") or "").strip()
        text = p.get("full_text") or p.get("abstract") or ""
        if doi:
            lookup.setdefault(doi, str(text))
        title = (p.get("title") or "").strip()
        if title:
            lookup.setdefault(title, str(text))

    out: list[dict[str, Any]] = []
    index_counter: dict[str, int] = {}
    for r in reported:
        row = dict(r)
        source_id = str(r.get("doi") or r.get("source_paper") or "")
        # 按 doi 优先, 退回 source_paper 标题
        text = lookup.get(str(r.get("doi") or ""), "") or lookup.get(source_id, "")
        index_counter[source_id] = index_counter.get(source_id, 0) + 1
        row["evidence"] = make_evidence(source_id, text or None, index_counter[source_id])
        out.append(row)
    return out


# ─────────────────────── 2) 显式豁免 + 决策档 ───────────────────────


def _waiver_reason(dim: str, reason_var: str = "") -> str:
    base = {
        "temperature": "补全检索后仍缺温度自由度: 0K vs 室温的数值不可直接对账",
        "functional": "补全检索后仍缺泛函自由度: 无法区分 LDA/GGA vs HSE 系统性偏差",
        "method_family": "补全检索后仍缺方法族: 数值的物理含义未能跨源对齐",
    }
    reason = base.get(dim, f"补全检索后仍缺自由度: {dim}")
    if reason_var:
        reason = f"{reason} [{reason_var}]"
    return reason


def build_waivers(
    missing_after: list[str],
    *,
    reason_var: str = "",
) -> list[dict[str, Any]]:
    """为"补后仍缺"的自由度生成显式豁免决策记录 (幂等).

    每条豁免是独立决策: 含决策 id (由内容哈希生成, 同输入同 id)、维度、理由、
    引用依据 (references, 空列表占位)、状态 (accepted 表示"已记录这次豁免")。

    Returns: waiver 决策列表, 按维度字典序。
    """
    waivers: list[dict[str, Any]] = []
    for dim in sorted(set(missing_after)):
        body = {
            "kind": "waiver",
            "dim": dim,
            "decision": "waived",
            "version": _WAIVER_VERSION,
            "reason": _waiver_reason(dim, reason_var),
            "references": [],
        }
        # 决策 id 由内容决定 → 幂等; 前缀 w 便于与其它决策区分
        body["decision_id"] = "w-" + sha256_text(json.dumps(body, sort_keys=True))[:16]
        waivers.append(body)
    return waivers


def _canonical_json(payload: dict[str, Any]) -> bytes:
    """与追加/校验保持字节级一致的规范序列化."""
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


class DecisionLedger:
    """追加式决策档 (JSONL + sha256 哈希链), 幂等.

    每条记录带 `_prev_hash`（链尾前一条内容的哈希）与 `_hash`（本条脱链后的
    自哈希）——任何篡改都会断链。同 decision_id 的记录不会重复追加（幂等）。

    落盘路径由调用方注入; 不像 audit_log 那样依赖全局单例 + install, 便于测试。
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        """读档尾最后一条的 _hash 以续链; 空/损坏 → genesis (新链)."""
        if not self._path.exists():
            return _GENESIS_HASH
        last = _GENESIS_HASH
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                h = rec.get("_hash")
                if isinstance(h, str) and h:
                    last = h
        return last

    def _exists(self, decision_id: str) -> bool:
        if not self._path.exists():
            return False
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("decision_id") == decision_id:
                    return True
        return False

    def append(self, record: dict[str, Any]) -> str:
        """追加一条决策记录 (幂等). 返回 decision_id. 同 id 已存在则跳过."""
        decision_id = str(record.get("decision_id") or "")
        if decision_id and self._exists(decision_id):
            return decision_id
        prev = self._last_hash()
        body = dict(record)
        body["_prev_hash"] = prev
        body["_hash"] = sha256_text(_canonical_json(body).decode("utf-8"))
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n")
        return decision_id

    def verify(self) -> tuple[bool, list[str]]:
        """校验整条哈希链. 返回 (ok, problems). 空档 → (True, [])."""
        problems: list[str] = []
        expected_prev = _GENESIS_HASH
        seen_decision_ids: set[str] = set()
        if not self._path.exists():
            return True, []
        with self._path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    problems.append(f"line {lineno}: invalid JSON ({exc})")
                    continue
                if rec.get("_prev_hash") != expected_prev:
                    problems.append(f"line {lineno}: broken chain (prev hash mismatch)")
                body = {k: v for k, v in rec.items() if k != "_hash"}
                actual = sha256_text(_canonical_json(body).decode("utf-8"))
                if rec.get("_hash") != actual:
                    problems.append(f"line {lineno}: hash mismatch (record modified)")
                expected_prev = rec.get("_hash") or _GENESIS_HASH
                did = rec.get("decision_id")
                if isinstance(did, str) and did:
                    if did in seen_decision_ids:
                        problems.append(f"line {lineno}: duplicate decision_id {did}")
                    seen_decision_ids.add(did)
        return not problems, problems


# ─────────────────────────── 3) 门禁不变量 ───────────────────────────


def critical_missing(missing_dims: list[str], urgency_map: dict[str, int]) -> list[str]:
    """筛出"关键"缺度 (对整体互洽判定影响大, 对齐 query_completion._URGENCY).

    Args:
        missing_dims: 补后仍缺的自由度。
        urgency_map: {dim: urgency}; urgency >= CRITICAL_URGENCY 视为关键。
    """
    return sorted(
        d for d in missing_dims if urgency_map.get(d, 0) >= CRITICAL_URGENCY
    )


def assess_gate(
    blocking_dims: list[str],
    waived_dims: list[str],
    *,
    verdict: str = "",
) -> dict[str, Any]:
    """评审门禁不变量 (AICC: pass 禁带 blocking_issues).

    Args:
        blocking_dims: 补后仍缺的关键自由度 (未豁免的会用掉才可放行)。
        waived_dims: 已由 waiver 覆盖、可带豁免放行的自由度。
        verdict: 当前局部-整体判定的标签 (透传, 仅展示)。

    Returns:
        {"status", "issues", "blocking_dims", "waived_dims", "verdict"}
        status ∈ {"pass", "pass_with_waiver", "needs_waiver"}。
    """
    unmet = sorted(d for d in blocking_dims if d not in (waived_dims or []))
    if unmet:
        issues = [f"blocking missing freedimension '{d}' has no waiver" for d in unmet]
        return {
            "status": "needs_waiver",
            "issues": issues,
            "blocking_dims": unmet,
            "waived_dims": sorted(d for d in (waived_dims or [])),
            "verdict": verdict,
        }
    has_waiver = bool(waived_dims)
    return {
        "status": "pass_with_waiver" if has_waiver else "pass",
        "issues": [],
        "blocking_dims": [],
        "waived_dims": sorted(d for d in (waived_dims or [])),
        "verdict": verdict,
    }


__all__ = [
    "CRITICAL_URGENCY",
    "DecisionLedger",
    "assess_gate",
    "attach_evidence",
    "build_waivers",
    "critical_missing",
    "make_evidence",
    "sha256_text",
]
