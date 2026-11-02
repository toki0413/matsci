"""Diagnose why a curated row failed the verbatim quote check.

For every row: fetch the source once (cached on disk), then measure
  - number presence: does the reported Ea value appear in the source at all?
  - anchor coverage: longest contiguous run of 8-char anchors of the quote
    found in the source, as a fraction of the quote length.
Classification:
  exact  : whole quote fragment found verbatim
  partial: >= 60% anchor coverage (minor wording/glyph edits)
  absent : < 60% (paraphrase, wrong source, or text not retrievable)
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

RAW = Path("/workspace/research/shusheng_run/_supervisor/lit_raw")
CACHE = RAW / "cache"
sys.path.insert(0, str(RAW))

from verify import get, norm, pmcid_for_doi, strip_tags  # noqa: E402

ANCHOR = 8
THRESH_PARTIAL = 0.60


def fetch_cached(row: dict) -> tuple[str, str]:
    url = row["url"]
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    path = CACHE / f"{key}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8"), url
    if "pmc.ncbi.nlm.nih.gov" in url:
        text, src = strip_tags(get(url)), url
    else:
        pmcid = pmcid_for_doi(row["doi"])
        if pmcid:
            src = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            text = strip_tags(get(src))
        else:
            src, text = url, strip_tags(get(url))
    CACHE.mkdir(exist_ok=True)
    path.write_text(f"SRC {src}\n{text}", encoding="utf-8")
    return path.read_text(encoding="utf-8"), src


def anchor_coverage(part: str, hay: str) -> tuple[float, bool]:
    p = norm(part)
    if not p:
        return 1.0, True
    if p in hay:
        return 1.0, True
    n = len(p)
    hits = [p[i:i + ANCHOR] in hay for i in range(0, max(n - ANCHOR + 1, 1))]
    if not hits:
        return 0.0, False
    best = cur = 0
    for h in hits:
        cur = cur + 1 if h else 0
        best = max(best, cur)
    cov = min((best + ANCHOR - 1) / n, 1.0)
    return cov, False


def number_present(value: str, hay: str) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    forms = {f"{v:g}", f"{v:.1f}", f"{v:.2f}", f"{v:.3f}"}
    return any(f in hay for f in forms if f)


def main() -> None:
    rows = list(csv.DictReader((RAW / "ea_dataset.csv").open(encoding="utf-8")))
    out = []
    for i, row in enumerate(rows, start=2):
        rec: dict = {"row": i, "material": row["material"], "ea": row["Ea_kJ_per_mol"]}
        try:
            text, src = fetch_cached(row)
        except Exception as e:  # noqa: BLE001
            rec.update(status="fetch_error", detail=str(e))
            out.append(rec)
            print(f"{i:2d} FETCH {e}")
            continue
        hay = norm(text.split("\n", 1)[1] if text.startswith("SRC ") else text)
        parts = [p for p in row["quote"].split("\u2016") if p.strip()]
        covs = []
        exact = True
        for p in parts:
            cov, is_exact = anchor_coverage(p, hay)
            covs.append(cov)
            exact = exact and is_exact
        cov_min = min(covs) if covs else 1.0
        status = "exact" if exact else ("partial" if cov_min >= THRESH_PARTIAL else "absent")
        rec.update(status=status, cov=round(cov_min, 3), src=src,
                   num=number_present(row["Ea_kJ_per_mol"], hay))
        out.append(rec)
        print(f"{i:2d} {status:7s} cov={cov_min:4.2f} num={'Y' if rec['num'] else 'n'} "
              f"{row['material'][:40]}")
    (RAW / "diagnose.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for s in ("exact", "partial", "absent", "fetch_error"):
        print(f"{s}: {sum(1 for r in out if r['status'] == s)}")


if __name__ == "__main__":
    main()
