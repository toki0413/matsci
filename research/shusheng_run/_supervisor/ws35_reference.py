"""Supervisor's independent reference implementation for ws35.

Same pre-registered procedure the agent is told to run, implemented separately
so the agent's numbers can be audited against a fixed target. Writes
ws35_truth.json next to this file (outside the agent workspace).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

WS = Path("/workspace/research/shusheng_run/ws35")
DATA = WS / "data" / "ea_dataset.csv"
OUT = Path(__file__).with_name("ws35_truth.json")
SEED = 20261029
DROP = {"MgH2 + 5 wt% V"}


def num(s: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", s or "")
    return float(m.group()) if m else None


def load() -> list[dict]:
    with DATA.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r["direction"] != "desorption" or r["material"] in DROP:
            continue
        ea = num(r["Ea_kJ_per_mol"])
        if ea is None:
            continue
        r["ea"] = ea
        out.append(r)
    return out


def h1_deltas(rows: list[dict]) -> dict[str, float]:
    by_doi: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        by_doi.setdefault(r["doi"], {}).setdefault(r["material_class"], []).append(r["ea"])
    d = {}
    for doi, g in by_doi.items():
        if "pure_MgH2" in g and "catalyzed_MgH2" in g:
            d[doi] = float(np.mean(g["pure_MgH2"]) - np.mean(g["catalyzed_MgH2"]))
    return d


def main() -> None:
    rows = load()
    deltas = h1_deltas(rows)
    d = np.array([v for v in deltas.values() if v > 0])
    all_d = np.array(list(deltas.values()))

    # DL with equal v_i=1 -> equal weights, pooled = mean; tau2 = max(0, s2 - 1)
    pooled = float(d.mean())
    s2 = float(d.var(ddof=1))
    tau2 = max(0.0, s2 - 1.0)

    rng = np.random.default_rng(SEED)
    keys = list(deltas)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(keys), len(keys))
        vals = [deltas[keys[i]] for i in idx]
        pos = [v for v in vals if v > 0]
        boot.append(np.mean(pos) if pos else np.nan)
    boot = np.array(boot)
    ci_low, ci_high = float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))

    # H2: eta^2 on catalyzed rows, permutation at paper level
    cat = [r for r in rows if r["material_class"] == "catalyzed_MgH2"]
    y = np.array([r["ea"] for r in cat])
    fam = np.array([r["catalyst_family"] for r in cat])
    doi = np.array([r["doi"] for r in cat])

    def eta2(labels: np.ndarray) -> float:
        groups = {g: y[labels == g] for g in set(labels)}
        ss_b = sum(len(v) * (v.mean() - y.mean()) ** 2 for v in groups.values())
        ss_t = ((y - y.mean()) ** 2).sum()
        return float(ss_b / ss_t) if ss_t else 0.0

    e2 = eta2(fam)
    # papers are the randomisation unit; all catalyzed rows of a paper share a family
    paper_fam = {}
    for p, f in zip(doi, fam):
        paper_fam.setdefault(p, set()).add(f)
    assert all(len(v) == 1 for v in paper_fam.values()), "family varies within a paper"
    paper_fam = {p: list(v)[0] for p, v in paper_fam.items()}
    papers = list(paper_fam)
    perm = []
    for _ in range(10000):
        shuffled = dict(zip(papers, rng.permutation([paper_fam[p] for p in papers])))
        lab = np.array([shuffled[p] for p in doi])
        perm.append(eta2(lab))
    perm_p = float(np.mean(np.array(perm) >= e2))

    # H3: same doi + same material, two different methods
    pairs = []
    by_cell: dict[tuple[str, str], dict[str, list[float]]] = {}
    for r in rows:
        by_cell.setdefault((r["doi"], r["material"]), {}).setdefault(r["method"], []).append(r["ea"])
    for (doi_k, mat), g in by_cell.items():
        if "Kissinger" in g and "isothermal_Arrhenius" in g:
            pairs.append({"doi": doi_k, "material": mat,
                          "kissinger": float(np.mean(g["Kissinger"])),
                          "arrhenius": float(np.mean(g["isothermal_Arrhenius"])),
                          "diff": float(np.mean(g["Kissinger"]) - np.mean(g["isothermal_Arrhenius"]))})

    # H4: LOSO with additive one-hot model
    feats = ["material_class", "catalyst_family", "method"]
    def design(rs: list[dict], levels: dict[str, list[str]]) -> np.ndarray:
        cols = [np.ones(len(rs))]
        for f in feats:
            vals = {lv: np.array([1.0 if r[f] == lv else 0.0 for r in rs]) for lv in levels[f]}
            base = levels[f][0]
            for lv in levels[f][1:]:
                cols.append(vals[lv] - vals[base])
        return np.column_stack(cols)

    levels = {f: sorted({r[f] for r in rows}) for f in feats}
    y_all = np.array([r["ea"] for r in rows])
    dois = np.array([r["doi"] for r in rows])
    pred = np.full(len(rows), np.nan)
    for p in sorted(set(dois)):
        te = dois == p
        tr = ~te
        Xtr, Xte = design([r for r, t in zip(rows, tr) if t], levels), design([r for r, t in zip(rows, te) if t], levels)
        beta, *_ = np.linalg.lstsq(Xtr, y_all[tr], rcond=None)
        pred[te] = Xte @ beta
    resid = y_all - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    sd = float(np.std(resid, ddof=1))
    cover = float(np.mean(np.abs(resid) <= 1.96 * sd))
    base_pred = np.array([np.mean(y_all[dois != p]) for p in dois])
    base_rmse = float(np.sqrt(np.mean((y_all - base_pred) ** 2)))

    # H5: exact H4 design, query the empty other x isothermal cell
    X = design(rows, levels)
    beta, *_ = np.linalg.lstsq(X, y_all, rcond=None)
    q = {"material_class": "catalyzed_MgH2", "catalyst_family": "other",
         "method": "isothermal_Arrhenius"}
    xq = design([q], levels)
    h5 = float((xq @ beta)[0])

    truth = {
        "seed": SEED,
        "n_desorption_rows": len(rows),
        "n_papers": len({r["doi"] for r in rows}),
        "h1_pairs": deltas,
        "h1_n_papers": len(d),
        "h1_delta_ea": pooled,
        "h1_between_paper_var": s2,
        "h1_tau2": tau2,
        "h1_ci_low": ci_low,
        "h1_ci_high": ci_high,
        "h1_negative_pairs": [k for k, v in deltas.items() if v <= 0],
        "h2_eta2": e2,
        "h2_perm_p": perm_p,
        "h3_pairs": pairs,
        "h3_method_pairs_n": len(pairs),
        "h3_method_diff_kj": abs(pairs[0]["diff"]) if pairs else None,
        "h3_catalyst_median_delta": float(np.median(d)),
        "h4_rmse": rmse,
        "h4_mae": mae,
        "h4_resid_sd": sd,
        "h4_coverage": cover,
        "h4_baseline_rmse": base_rmse,
        "h5_pred": h5,
        "h5_pi_low": h5 - 1.96 * sd,
        "h5_pi_high": h5 + 1.96 * sd,
    }
    OUT.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in truth.items():
        if k not in ("h1_pairs", "h3_pairs"):
            print(f"{k:26s} {v}")


if __name__ == "__main__":
    main()
