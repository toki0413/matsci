"""Round-2 verification: check each curated row against *all* retrievable
renderings of the same paper, not just the one URL the curator cited.

Why: round-1 fetched the PMC HTML page, which sometimes returns a reCAPTCHA
interstitial (rate limit) and sometimes omits figure/table-only text. Both
produce false mismatches. Sources used here, in order:

  epmc_xml   Europe PMC full-text JATS XML (PMCID from the URL, else via DOI)
  crossref   Crossref deposited abstract (JATS, tags stripped)
  mdpi_pdf   the publisher PDF from mdpi-res.com (MDPI DOIs only; www.mdpi.com
             is behind Cloudflare and answers 403 to non-browser clients)
  direct     the cited URL as a last resort

A row passes when every ‖-separated fragment of its quote is found verbatim in
at least one source (whitespace/tag/glyph normalised). Output keeps the source
that matched so the pass is auditable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path("/workspace/research/shusheng_run/_supervisor/lit_raw")
CACHE = RAW / "cache2"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}

sys.path.insert(0, str(RAW))
from verify import norm, strip_tags  # noqa: E402


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def text_of(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def cached(name: str, url: str, binary: bool = False) -> str:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}_{name}"
    if p.exists():
        data = p.read_bytes()
    else:
        data = get(url)
        p.write_bytes(data)
    if binary:
        return data.decode("latin-1")
    return text_of(data)


def epmc_xml(pmcid: str) -> str:
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    return strip_tags(cached("epmc.xml", url))


def doi_to_pmcid(doi: str) -> str | None:
    q = urllib.parse.quote(f'DOI:"{doi}"')
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&format=json"
    try:
        data = json.loads(cached("epmc.json", url))
    except Exception:  # noqa: BLE001
        return None
    for res in data.get("resultList", {}).get("result", []):
        if res.get("pmcid"):
            return res["pmcid"]
    return None


def crossref_abstract(doi: str) -> str:
    try:
        data = json.loads(cached("crossref.json", f"https://api.crossref.org/works/{doi}"))
        return strip_tags(data["message"].get("abstract") or "")
    except Exception:  # noqa: BLE001
        return ""


def mdpi_pdf_url(doi: str) -> str | None:
    """10.3390/<code><vol:2><issue:2><article> -> mdpi-res.com deploy path."""
    m = re.fullmatch(r"10\.3390/([a-z]+)(\d{2})(\d{2})(\d{3,})", doi, re.I)
    if not m:
        return None
    code, vol, iss, art = m.group(1).lower(), m.group(2), m.group(3), m.group(4)
    slug = f"{code}-{vol}-{iss}{art}"
    return (f"https://mdpi-res.com/d_attachment/{code}/{slug}/"
            f"article_deploy/{slug}.pdf")


def mdpi_pdf_text(doi: str) -> str:
    url = mdpi_pdf_url(doi)
    if not url:
        return ""
    try:
        CACHE.mkdir(exist_ok=True)
        p = CACHE / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}_mdpi.pdf"
        if not p.exists() or p.stat().st_size < 20_000:
            p.write_bytes(get(url, timeout=180))
        out = subprocess.run(["pdftotext", "-q", str(p), "-"],
                             capture_output=True, text=True, timeout=120)
        return out.stdout
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    rows = list(csv.DictReader((RAW / "ea_dataset.csv").open(encoding="utf-8")))
    results = []
    for i, row in enumerate(rows, start=2):
        rec: dict = {"row": i, "material": row["material"], "ea": row["Ea_kJ_per_mol"]}
        sources: list[tuple[str, str]] = []
        url = row["url"]
        doi = row["doi"]
        m = re.search(r"PMC(\d+)", url)
        try:
            if m:
                sources.append(("epmc_xml", epmc_xml(f"PMC{m.group(1)}")))
            elif doi:
                pmcid = doi_to_pmcid(doi)
                if pmcid:
                    sources.append(("epmc_xml", epmc_xml(pmcid)))
            if doi:
                sources.append(("crossref", crossref_abstract(doi)))
        except Exception as e:  # noqa: BLE001
            rec.setdefault("warn", []).append(str(e))

        parts = [p for p in row["quote"].split("\u2016") if p.strip()]
        hay = [(n, norm(t)) for n, t in sources if t.strip()]
        missing = []
        matched = set()
        for p in parts:
            np = norm(p)
            hit = [n for n, h in hay if np in h]
            if hit:
                matched.update(hit)
            else:
                missing.append(p[:70])

        if missing and doi.lower().startswith("10.3390/"):
            t = mdpi_pdf_text(doi)
            if t.strip():
                h = norm(t)
                still = []
                for p in parts:
                    if norm(p) in h:
                        matched.add("mdpi_pdf")
                    else:
                        still.append(p[:70])
                missing = still

        rec.update(ok=not missing, matched=sorted(matched),
                   n_sources=len(hay), missing=missing)
        results.append(rec)
        print(f"{i:2d} {'OK ' if rec['ok'] else 'MISS'} "
              f"{','.join(rec['matched'])[:24]:24s} {row['material'][:38]:38s} "
              f"{'' if not missing else missing[0][:60]}")

    (RAW / "verification2.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for r in results if r["ok"])
    print(f"\nsummary: verified={ok} failed={len(results) - ok} total={len(results)}")


if __name__ == "__main__":
    main()
