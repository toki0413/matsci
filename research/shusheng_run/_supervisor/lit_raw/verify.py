"""Verify every curated row's `quote` against the live source text.

- PMC pages: fetch HTML directly.
- Others (MDPI/RSC blocked): resolve DOI -> PMCID via Europe PMC, then fetch
  full-text XML.
Normalisation strips tags/whitespace and unifies dash/degree glyphs, so
transcription of layout variants does not cause false mismatches.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path("/workspace/research/shusheng_run/_supervisor/lit_raw")
UA = {"User-Agent": "Mozilla/5.0 (research verification)"}


def get(url: str, timeout: int = 40) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def strip_tags(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    return re.sub(r"(?s)<[^>]+>", " ", html)


def norm(s: str) -> str:
    s = strip_tags(s)
    s = (s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
          .replace("\u2019", "'").replace("\u00b0", "deg").replace("\u25e6", "deg"))
    s = s.lower()
    return re.sub(r"\s+", "", s)


def pmcid_for_doi(doi: str) -> str | None:
    q = urllib.parse.quote(f'DOI:"{doi}"')
    url = (f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
           f"?query={q}&format=json&resultType=core")
    try:
        data = json.loads(get(url))
    except Exception:
        return None
    for res in data.get("resultList", {}).get("result", []):
        if res.get("pmcid"):
            return res["pmcid"]
    return None


def source_text(row: dict) -> tuple[str, str]:
    url = row["url"]
    if "pmc.ncbi.nlm.nih.gov" in url:
        return url, strip_tags(get(url))
    pmcid = pmcid_for_doi(row["doi"])
    if pmcid:
        xml_url = (f"https://www.ebi.ac.uk/europepmc/webservices/rest/"
                   f"{pmcid}/fullTextXML")
        return xml_url, strip_tags(get(xml_url))
    return url, strip_tags(get(url))


def main() -> None:
    rows = list(csv.DictReader((RAW / "ea_dataset.csv").open(encoding="utf-8")))
    cache: dict[str, str] = {}
    results = []
    for i, row in enumerate(rows, start=2):
        try:
            src_url, text = source_text(row)
        except Exception as e:
            results.append({"row": i, "ok": None, "detail": f"FETCH_FAIL {e}"})
            print(f"row {i:2d} FETCH_FAIL {e}")
            continue
        cache[src_url] = text
        hay = norm(text)
        parts = [p for p in row["quote"].split("\u2016") if p.strip()]
        missing = [p[:70] for p in parts if norm(p) not in hay]
        ok = not missing
        results.append({"row": i, "ok": ok, "src": src_url,
                        "missing": missing})
        print(f"row {i:2d} {'OK ' if ok else 'MISS'} {row['material'][:38]:38s} "
              f"{'' if ok else missing}")
    (RAW / "verification.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    ok_n = sum(1 for r in results if r["ok"] is True)
    bad_n = sum(1 for r in results if r["ok"] is False)
    err_n = sum(1 for r in results if r["ok"] is None)
    print(f"\nsummary: verified={ok_n} mismatch={bad_n} fetch_error={err_n} "
          f"total={len(results)}")


if __name__ == "__main__":
    sys.exit(main())
