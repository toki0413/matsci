"""ws35 元分析：催化是否降低 MgH2 脱氢表观活化能（固定协议，seed=20261029）。

只用 numpy / scipy + 标准库 csv；推断一律按 doi 整群（cluster）处理。
运行：python3 analysis.py（在工作区根目录，读 data/ea_dataset.csv）
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

SEED = 20261029
DATA = Path("data") / "ea_dataset.csv"
DROP_MATERIAL = {"MgH2 + 5 wt% V"}  # README 使用前必读 (2)：阶段归属不明，剔除
FEATS = ["material_class", "catalyst_family", "method"]


def num(s: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", s or "")
    return float(m.group()) if m else None


def load_rows() -> list[dict]:
    with DATA.open(encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    rows = []
    for r in raw:
        if r["direction"] != "desorption":       # absorption 6 行不得混入脱氢
            continue
        if r["material"] in DROP_MATERIAL:
            continue
        ea = num(r["Ea_kJ_per_mol"])
        if ea is None:
            continue
        r = dict(r)
        r["ea"] = ea
        rows.append(r)
    return rows


def h1_deltas(rows: list[dict]) -> dict[str, float]:
    """同文配对：同一 doi 同时有 pure_MgH2 与 catalyzed_MgH2 行时给一个 ΔEa。"""
    by_doi: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        by_doi.setdefault(r["doi"], {}).setdefault(r["material_class"], []).append(r["ea"])
    out = {}
    for doi, g in by_doi.items():
        if "pure_MgH2" in g and "catalyzed_MgH2" in g:
            out[doi] = float(np.mean(g["pure_MgH2"]) - np.mean(g["catalyzed_MgH2"]))
    return out


def eta_sq(y: np.ndarray, labels: np.ndarray) -> float:
    ss_t = float(((y - y.mean()) ** 2).sum())
    if not ss_t:
        return 0.0
    ss_b = sum(len(v) * (v.mean() - y.mean()) ** 2 for g in set(labels)
               for v in [y[labels == g]])
    return float(ss_b / ss_t)


def levels_of(rows: list[dict]) -> dict[str, list[str]]:
    return {f: sorted({r[f] for r in rows}) for f in FEATS}


def design(rs: list[dict], levels: dict[str, list[str]]) -> np.ndarray:
    """可加 one-hot 设计阵（每个因子丢掉基线水平，避免共线）。"""
    cols = [np.ones(len(rs))]
    for f in FEATS:
        vals = {lv: np.array([1.0 if r[f] == lv else 0.0 for r in rs]) for lv in levels[f]}
        base = levels[f][0]
        for lv in levels[f][1:]:
            cols.append(vals[lv] - vals[base])
    return np.column_stack(cols)


def loso(rows: list[dict], levels: dict[str, list[str]]) -> tuple[np.ndarray, float, float]:
    """留一论文交叉验证：预测值、RMSE、基线(训练集均值) RMSE。"""
    y = np.array([r["ea"] for r in rows])
    dois = np.array([r["doi"] for r in rows])
    pred = np.full(len(rows), np.nan)
    base = np.full(len(rows), np.nan)
    for p in sorted(set(dois)):
        te = dois == p
        tr = ~te
        if tr.sum() == 0:
            continue
        X = design(rows, levels)
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[te] = X[te] @ beta
        base[te] = y[tr].mean()
    return pred, float(np.sqrt(np.mean((y - pred) ** 2))), float(np.sqrt(np.mean((y - base) ** 2)))


def main() -> None:
    rows = load_rows()
    deltas = h1_deltas(rows)
    pos = np.array([v for v in deltas.values() if v > 0])
    d = np.array(list(deltas.values()))

    pooled = float(pos.mean())
    s2 = float(pos.var(ddof=1))                      # 论文间方差（DL 的矩估计输入）
    tau2 = max(0.0, s2 - 1.0)                        # v_i 取等权 1 时的 DL 矩估计

    # H1 95% CI：按论文整群重采样 1000 次（有放回抽论文，取全部论文级 ΔEa）
    rng = np.random.default_rng(SEED)
    keys = list(deltas)
    boot = np.empty(1000)
    for i in range(1000):
        idx = rng.integers(0, len(keys), len(keys))
        vals = [deltas[keys[j]] for j in idx]
        pos_i = [v for v in vals if v > 0]
        boot[i] = np.mean(pos_i) if pos_i else np.nan
    ci_low = float(np.nanpercentile(boot, 2.5))
    ci_high = float(np.nanpercentile(boot, 97.5))

    # H2：只在 catalyzed_MgH2 行上，族间方差占比 + 以论文为单位整体置换族标签
    cat = [r for r in rows if r["material_class"] == "catalyzed_MgH2"]
    y_cat = np.array([r["ea"] for r in cat])
    fam = np.array([r["catalyst_family"] for r in cat])
    doi_cat = np.array([r["doi"] for r in cat])
    eta2 = eta_sq(y_cat, fam)
    paper_fam: dict[str, str] = {}
    for p, f in zip(doi_cat, fam):
        paper_fam.setdefault(p, f)
    papers = list(paper_fam)
    perm = np.empty(10000)
    for i in range(10000):
        shuffled = dict(zip(papers, rng.permutation([paper_fam[p] for p in papers])))
        perm[i] = eta_sq(y_cat, np.array([shuffled[p] for p in doi_cat]))
    perm_p = float(np.mean(perm >= eta2))

    # H3：同一 doi + 同一 material 被两种方法各算一次的对子
    by_cell: dict[tuple[str, str], dict[str, list[float]]] = {}
    for r in rows:
        by_cell.setdefault((r["doi"], r["material"]), {}).setdefault(r["method"], []).append(r["ea"])
    pairs = []
    for (doi_k, mat), g in by_cell.items():
        if "Kissinger" in g and "isothermal_Arrhenius" in g:
            pairs.append({
                "doi": doi_k, "material": mat,
                "kissinger": float(np.mean(g["Kissinger"])),
                "arrhenius": float(np.mean(g["isothermal_Arrhenius"])),
                "diff": float(np.mean(g["Kissinger"]) - np.mean(g["isothermal_Arrhenius"])),
            })
    method_diff = abs(pairs[0]["diff"]) if pairs else float("nan")
    catalyst_median = float(np.median(pos))

    # H4：LOSO
    levels = levels_of(rows)
    pred, rmse, base_rmse = loso(rows, levels)
    y_all = np.array([r["ea"] for r in rows])
    resid = y_all - pred
    mae = float(np.mean(np.abs(resid)))
    resid_sd = float(np.std(resid, ddof=1))
    coverage = float(np.mean(np.abs(resid) <= 1.96 * resid_sd))

    # H5：用 H4 的全样本模型外推空白格 other × isothermal_Arrhenius
    X = design(rows, levels)
    beta, *_ = np.linalg.lstsq(X, y_all, rcond=None)
    q = {"material_class": "catalyzed_MgH2", "catalyst_family": "other",
         "method": "isothermal_Arrhenius"}
    yhat = float((design([q], levels) @ beta)[0])
    pi_low = yhat - 1.96 * resid_sd
    pi_high = yhat + 1.96 * resid_sd

    results = {
        "seed": SEED,
        "n_desorption_rows": len(rows),
        "n_papers": len({r["doi"] for r in rows}),
        "q1_n_papers": int(len(pos)),
        "q1_delta_ea": pooled,
        "q1_ci_low": ci_low,
        "q1_ci_high": ci_high,
        "q1_tau2": tau2,
        "q1_between_paper_var": s2,
        "q1_negative_pairs": {k: v for k, v in deltas.items() if v <= 0},
        "q1_pair_deltas": deltas,
        "q2_eta2": eta2,
        "q2_perm_p": perm_p,
        "q3_method_pairs_n": len(pairs),
        "q3_method_diff_kj": method_diff,
        "q3_catalyst_median_delta": catalyst_median,
        "q4_rmse": rmse,
        "q4_baseline_rmse": base_rmse,
        "q4_mae": mae,
        "q4_coverage": coverage,
        "q4_resid_sd": resid_sd,
        "q5_pred": yhat,
        "q5_pi_low": pi_low,
        "q5_pi_high": pi_high,
        "hypotheses": {
            "P1": "supported" if (pooled >= 25 and ci_low > 0) else "refuted",
            "P2": "supported" if (eta2 < 0.20 and perm_p > 0.05) else "refuted",
            "P3": "supported" if (len(pairs) > 0 and method_diff < catalyst_median) else "refuted",
            "P4": "supported" if (rmse < 30 and rmse <= 0.75 * base_rmse and 0.80 <= coverage <= 0.98) else "refuted",
            "P5": "prediction_only",
        },
    }
    Path("results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # 50 个循环 = 50 次整群 bootstrap：每次重跑 H1 合并估计与 H4 LOSO
    rng_loop = np.random.default_rng(SEED + 1)
    keys_pos = [k for k in keys if deltas[k] > 0]
    with Path("loop_log.jsonl").open("w", encoding="utf-8") as f:
        for loop_id in range(1, 51):
            idx = rng_loop.integers(0, len(keys), len(keys))
            sampled_dois = [keys[j] for j in idx]
            rows_b = [r for r in rows if r["doi"] in set(sampled_dois)]
            # 该次重采样下的 H1 合并估计（只取 ΔEa > 0 的论文）
            vals = [deltas[k] for k in sampled_dois if deltas.get(k, 0) > 0]
            d_loop = float(np.mean(vals)) if vals else float("nan")
            # 该次重采样内部的整群 bootstrap CI（200 次，只为给出逐轮区间）
            inner = np.empty(200)
            for i in range(200):
                jdx = rng_loop.integers(0, len(sampled_dois), len(sampled_dois))
                vs = [deltas[sampled_dois[j]] for j in jdx if deltas.get(sampled_dois[j], 0) > 0]
                inner[i] = np.mean(vs) if vs else np.nan
            lo = float(np.nanpercentile(inner, 2.5))
            hi = float(np.nanpercentile(inner, 97.5))
            _, rmse_b, _ = loso(rows_b, levels) if rows_b else (None, float("nan"), float("nan"))
            f.write(json.dumps({
                "loop_id": loop_id,
                "status": "ok" if np.isfinite(d_loop) else "empty_resample",
                "delta_ea": d_loop,
                "ci_low": lo,
                "ci_high": hi,
                "loso_rmse": rmse_b,
                "n_papers": len(set(sampled_dois)),
                "n_rows": len(rows_b),
            }, ensure_ascii=False) + "\n")

    print(json.dumps({k: v for k, v in results.items()
                      if not isinstance(v, dict) and not isinstance(v, list)},
                     indent=2, ensure_ascii=False))
    print("loop_log lines:", 50)


if __name__ == "__main__":
    main()
