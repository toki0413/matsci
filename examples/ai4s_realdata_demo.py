#!/usr/bin/env python3
"""Huginn × 书生 — 面向「真正的科学开放问题」的自然科学开放研究 demo.

与杜撰数据不同：数据来自**真实公开库**（经一次真实抓取后缓存，可复现）：
  - exoplanet  : NASA 系外行星档案（真实周期/质量/半径）
  - chembl     : ChEMBL 真实生物活性 pChEMBL + 分子 SMILES
问题没有预设答案；模型自主做符号/回归建模，对**留出未见的真实样本**做预测，
再由框架**回库对账**（verify_predictions 返回每个留出样本的预测 vs 真实值），
从而每个结论都能被真实数据验证/推翻（可证伪）。

「结论证伪门禁」沿用框架 huginn/validation/claim_grounding：报告里每个数值必须落在
工具执行轨迹（= 真实API值 + 真实拟合 + 真实对账）中，否则 needs_grounding 打回。

用法:
    export INTERNLM_API_KEY=<书生 token>
    python examples/ai4s_realdata_demo.py --problem exoplanet
    python examples/ai4s_realdata_demo.py --problem chembl
依赖: pip install requests openai
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

_BASE_URL = os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1")
_DEFAULT_MODEL = "intern-s2-preview"

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))   # 产品模块(huginn.*)
from huginn.research import grounding_verifier  # noqa: E402   # 声明门禁唯一实现
_verify = grounding_verifier()


# ─────────────────────────── 1) 真实数据抓取 + 缓存 ─────────────────────
def _http_get(url, params=None, timeout=40):
    import requests
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_exoplanet() -> list[dict]:
    cache = OUT / "real_exoplanet.json"
    if cache.exists():
        return json.loads(cache.read_text("utf-8"))
    # NASA Exoplanet Archive (public TAP): confirmed planets; filter in python
    q = ("select top 400 pl_name,pl_orbper,pl_rade,pl_bmassj from ps")
    rows = _http_get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                     params={"query": q, "format": "json"})
    # 只保留关键列齐全的行; 采样到合理规模
    clean = []
    for r in rows:
        name = r.get("pl_name")
        if not name or not r.get("pl_rade") or not r.get("pl_bmassj") or not r.get("pl_orbper"):
            continue
        clean.append({
            "name": name,
            "orbper_d": float(r["pl_orbper"]),
            "radius_re": float(r["pl_rade"]),
            "mass_mjup": float(r["pl_bmassj"]),
        })
    cache.write_text(json.dumps(clean, ensure_ascii=False), "utf-8")
    return clean


def _atoms_from_smiles(s: str) -> dict:
    """极简 SMILES 元素计数 (中括号原子也可). 序贯解析, 不处理分支括号的复杂位移."""
    import re
    counts: dict[str, int] = {}
    toks = re.findall(r"\[[^\]]+\]|[A-Z][a-z]?", s or "")
    for t in toks:
        el = t.strip("[]")
        if el and el[0].isupper():
            counts[el] = counts.get(el, 0) + 1
    return counts


_MASS = {"C": 12.011, "N": 14.007, "O": 15.999, "H": 1.008, "S": 32.06, "P": 30.974,
         "F": 18.998, "Cl": 35.45, "Br": 79.904, "I": 126.90, "Si": 28.085}


def chembl_descriptors(smiles: str) -> dict:
    """从 SMILES 计算的轻量描述子 (无 RDKit, 工程近似, 确定性)."""
    counts = _atoms_from_smiles(smiles)
    heavy = sum(v for el, v in counts.items() if el != "H")
    mw = sum(_MASS.get(el, 0) * v for el, v in counts.items() if el != "H")
    mw += _MASS["H"] * counts.get("H", 0)  # H 仅显式计入; 未显式 H 是近似
    # 环闭合数字 / 芳香标记 / 杂原子 — 轻量近似
    rings = smiles.count("1") + smiles.count("2")

    def _logp_hint():
        # 极简 logP 启发: 脂溶性≈ 重原子减氧氮
        o = counts.get("O", 0); n = counts.get("N", 0)
        return max(0.0, heavy - o - n) * 0.3
    return {"mw": mw, "heavy": heavy, "rings": rings, "logp": round(_logp_hint(), 3)}


def fetch_chembl(target: str = "CHEMBL1829") -> list[dict]:
    cache = OUT / f"real_chembl_{target}.json"
    if cache.exists():
        return json.loads(cache.read_text("utf-8"))
    acts = _http_get("https://www.ebi.ac.uk/chembl/api/data/activity.json",
                     params={"target_chembl_id": target,
                             "pchembl_value__isnull": "false", "limit": 120},
                     timeout=40).get("activities", [])
    uniq: dict[str, dict] = {}
    for a in acts:
        mol = a.get("molecule_chembl_id")
        pc = a.get("pchembl_value")
        if not mol or not pc:
            continue
        uniq.setdefault(mol, {"pchembl": max(float(pc), float(uniq[mol]["pchembl"]) if mol in uniq else 0)})
    mids = list(uniq.keys())
    rows = []
    import requests
    for i in range(0, len(mids), 40):
        chunk = mids[i:i + 40]
        mols = requests.get("https://www.ebi.ac.uk/chembl/api/data/molecule/",
                            params={"molecule_chembl_id__in": ",".join(chunk), "limit": len(chunk)},
                            headers={"Accept": "application/json"}, timeout=40).json().get("molecules", [])
        for m in mols:
            mid = m.get("molecule_chembl_id")
            smiles = (m.get("molecule_structures") or {}).get("canonical_smiles")
            if mid in uniq and smiles:
                rows.append({"name": mid, "smiles": smiles,
                             "pchembl": uniq[mid]["pchembl"],
                             **chembl_descriptors(smiles)})
    cache.write_text(json.dumps(rows, ensure_ascii=False), "utf-8")
    return rows


# ─────────────────────────── 2) 纯 Python 最小二乘 ──────────────────────
def _solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate([row[:] for row in A])]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col] / M[col][col]
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def ols(feature_rows: list[list[float]], y: list[float]) -> dict:
    n = len(y)
    k = len(feature_rows[0]) + 1 if feature_rows else 1
    X = [[1.0] + row for row in feature_rows]
    XtX = [[sum(X[i][r] * X[i][c] for i in range(n)) for c in range(k)] for r in range(k)]
    Xty = [sum(X[i][c] * y[i] for i in range(n)) for c in range(k)]
    beta = _solve(XtX, Xty)
    pred = [sum(beta[c] * X[i][c] for c in range(k)) for i in range(n)]
    rss = sum((y[i] - pred[i]) ** 2 for i in range(n))
    sst = sum((yy - sum(y) / n) ** 2 for yy in y)
    r2 = 1.0 - rss / max(sst, 1e-12)
    aic = n * math.log(max(rss / n, 1e-12)) + 2 * k
    loo = 0.0
    for i in range(n):
        Xo = X[:i] + X[i + 1:]; yo = y[:i] + y[i + 1:]
        try:
            b = _solve([[sum(Xo[t][r] * Xo[t][c] for t in range(n - 1)) for c in range(k)] for r in range(k)],
                       [sum(Xo[t][c] * yo[t] for t in range(n - 1)) for c in range(k)])
            ph = sum(b[c] * X[i][c] for c in range(k))
            loo += (y[i] - ph) ** 2
        except Exception:
            pass
    return {"beta": [round(v, 4) for v in beta], "r2": round(r2, 3),
            "aic": round(aic, 3), "rss": round(rss, 4), "n": n,
            "loocv_mse": round(loo / n, 4)}


# ─────────────────────────── 3) 问题定义 ────────────────────────────────
def _exo_features(r):
    return {"log_mass": math.log10(r["mass_mjup"]), "log_per": math.log10(r["orbper_d"]),
            "mass": r["mass_mjup"], "per": r["orbper_d"]}


def _chembl_features(r):
    return {"mw": r["mw"], "logp": r["logp"], "rings": r["rings"], "heavy": r["heavy"]}


PROBLEMS = {
    "exoplanet": {
        "title": "真实系外行星：周期-质量-半径经验律的自主发现与可证伪预测",
        "note": ("真实 NASA 系外行星档案。目标: 由行星的质量(木星质量)、轨道周期(天)等自主"
                 "构建 log(半径) 的经验规律, 并对留出的 3 颗真实行星预测半径, 回库对账."),
        "fetch": fetch_exoplanet,
        "features": _exo_features,
        "feature_keys": ["log_mass", "log_per", "mass", "per"],
        "target_key": "radius_re",
        "target_name": "log(radius/R_Earth)",
        "id": lambda r: r["name"],
        "n_holdout": 3,
        "desc": "每条记录含 name / orbper_d(天) / radius_re(R⊕) / mass_mjup(M_Jup)。",
    },
    "chembl": {
        "title": "真实 ChEMBL：分子描述子—活性(pChEMBL)经验律的自主发现与可证伪预测",
        "note": ("真实 ChEMBL 生物活性数据 (EGFR 靶点)。目标: 用 SMILES 算出的轻量描述子"
                 "(mw/logp/rings/heavy) 自主建模活性对数值 pChEMBL, 对留出的 3 个真实分子预测, 回库对账."),
        "fetch": lambda: fetch_chembl("CHEMBL1829"),
        "features": _chembl_features,
        "feature_keys": ["mw", "logp", "rings", "heavy"],
        "target_key": "pchembl",
        "target_name": "pChEMBL (活性)",
        "id": lambda r: r["name"],
        "n_holdout": 3,
        "desc": "每条记录含 name(CHEMBL id) / SMILES → 描述子 mw,logp,rings,heavy / pchembl。",
    },
}


# ─────────────────────────── 4) 工具与执行 ──────────────────────────────
def build_tools(problem: str):
    return [
        {"type": "function", "function": {"name": "load_data",
            "description": f"加载问题 '{problem}' 的真实数据(训练集)并说明可用特征与目标。",
            "parameters": {"type": "object", "properties": {"problem": {"type": "string", "enum": [problem]}}, "required": ["problem"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "fit_model",
            "description": "用选定的特征做真实最小二乘建模(可 log 变换目标), 返回系数/R²/AIC/LOOCV。",
            "parameters": {"type": "object", "properties": {
                "problem": {"type": "string", "enum": [problem]},
                "features": {"type": "array", "items": {"type": "string"}},
                "target": {"type": "string", "default": "raw", "enum": ["raw", "log"]}},
                "required": ["problem", "features"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "propose_law",
            "description": "给出你认为的显式经验律(基于已拟合系数), 作为可证伪假设。",
            "parameters": {"type": "object", "properties": {
                "law": {"type": "string"}, "note": {"type": "string"}},
                "required": ["law"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "predict_holdout",
            "description": f"用最近一次 fit_model 得到的最优模型, 对留出的 {PROBLEMS[problem]['n_holdout']} 个真实未知样本预测目标。",
            "parameters": {"type": "object", "properties": {"problem": {"type": "string", "enum": [problem]}}, "required": ["problem"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "verify_predictions",
            "description": "回库真值对账: 拉取留出样本的真实目标值, 与预测逐一比对, 返回误差。这是本研究的可证伪验证步骤。",
            "parameters": {"type": "object", "properties": {"problem": {"type": "string", "enum": [problem]}}, "required": ["problem"], "additionalProperties": False}}},
    ]


def build_goal(problem: str):
    p = PROBLEMS[problem]
    return (
        "你是 Huginn 科研智能体, 以 Intern-S2 身份对**真实开放问题**做原创研究(非材料, 通用自然科学).\n"
        f"问题: {p['title']}。\n{p['note']}\n{p['desc']}\n"
        "研究步骤由你自主决定, 建议: 用 load_data 看数据 → 用 fit_model 试几个特征组合与变换"
        "(可用目标 log 变换得幂律) → 用 propose_law 提出显式经验律 → 用 predict_holdout 对留出真实样本预测"
        "→ 用 verify_predictions 回库对账。\n"
        "门禁提醒: 报告里每个数值必须落在你实际调用工具返回的真实结果(真实API/真实拟合/真实对账)中, "
        "未落地主张会被拒绝。因此请务必真正调用上述工具获取数字。\n"
        "最终撰写完整开放研究报告(研究问题/数据与方法/结果分析/预测-对账/结论与局限/下一步), 只输出正文。"
    )


# ─────────────────────────── 5) 代理循环 ────────────────────────────────
def _pick(tc):
    import ast
    name = tc.function.name
    raw = tc.function.arguments if isinstance(tc.function.arguments, str) else tc.function.arguments
    if not isinstance(raw, str):
        return name, raw
    for c in (raw, raw.replace("'", '"')):
        for fn in (json.loads, ast.literal_eval):
            try:
                return name, fn(c)
            except Exception:
                continue
    return name, {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", required=True, choices=list(PROBLEMS))
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    key = os.environ.get("INTERNLM_API_KEY")
    if not key:
        print("error: INTERNLM_API_KEY not set", file=__import__("sys").stderr); return 2

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=args.base_url or _BASE_URL)
    verify = _verify
    prob = PROBLEMS[args.problem]

    # 抓取真实数据 + 确定性留出切分
    rows = prob["fetch"]()
    rng = random.Random(args.seed)
    ids = sorted(prob["id"](r) for r in rows)
    hold = set(rng.sample(ids, min(prob["n_holdout"], len(ids))))
    train = [r for r in rows if prob["id"](r) not in hold]
    holdrows = [r for r in rows if prob["id"](r) in hold]
    state = {"active_model": None}

    def _feat_vec(r, features):
        f = prob["features"](r)
        return [f[feat] for feat in features]

    def exec_tool(name, a):
        if name == "load_data":
            feats = prob["feature_keys"]
            return json.dumps({"problem": args.problem, "n_train": len(train), "n_holdout": len(holdrows),
                               "feature_keys": feats, "target": prob["target_name"],
                               "sample_features": _feat_vec(train[0], feats),
                               "description": prob["desc"]}, ensure_ascii=False)
        if name == "fit_model":
            feats = a.get("features") or [prob["feature_keys"][0]]
            tmode = a.get("target", "raw")
            F = [(_feat_vec(r, feats)) for r in train]
            y = [math.log(r[prob["target_key"]]) if tmode == "log" else r[prob["target_key"]] for r in train]
            res = ols(F, y)
            res["features"] = feats; res["target_transform"] = tmode
            state["active_model"] = {"features": feats, "target": tmode, "beta": res["beta"]}
            return json.dumps(res, ensure_ascii=False)
        if name == "propose_law":
            return json.dumps({"law": a.get("law", ""), "note": a.get("note", "")}, ensure_ascii=False)
        if name == "predict_holdout":
            if not state["active_model"]:
                return json.dumps({"error": "请先 fit_model"}, ensure_ascii=False)
            m = state["active_model"]; feats = m["features"]; tmode = m["target"]
            preds = []
            for r in holdrows:
                x = _feat_vec(r, feats)
                b = state["active_model"]["beta"]
                v = b[0] + sum(b[i + 1] * x[i] for i in range(len(x)))
                if tmode == "log":
                    v = math.exp(v)
                preds.append({"id": prob["id"](r), "pred": round(v, 4)})
            state["active_model"]["preds"] = preds
            return json.dumps({"predictions": preds}, ensure_ascii=False)
        if name == "verify_predictions":
            m = state.get("active_model") or {}
            preds = m.get("preds") or []
            byid = {p["id"]: p["pred"] for p in preds}
            out = []
            for r in holdrows:
                i = prob["id"](r); true = r[prob["target_key"]]
                p = byid.get(i)
                if p is None:
                    out.append({"id": i, "true": round(true, 4), "pred": None, "abs_err": None, "rel_err": None})
                else:
                    out.append({"id": i, "true": round(true, 4), "pred": p,
                                "abs_err": round(abs(p - true), 4),
                                "rel_err": round(abs(p - true) / max(abs(true), 1e-9), 3)})
            return json.dumps({"reconciliation": out, "note": "回库真值对账完成."}, ensure_ascii=False)
        raise AssertionError(name)

    messages = [{"role": "user", "content": build_goal(args.problem)}]
    trace: list[str] = []
    transcript: list[str] = []

    for _ in range(16):
        r = client.chat.completions.create(model=args.model, messages=messages, tools=build_tools(args.problem),
                                           tool_choice="auto", max_tokens=1200, temperature=0.2)
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            break
        for tc in calls:
            name, a = _pick(tc)
            result = exec_tool(name, a)
            print(f"[tool] {name} {tc.function.arguments}\n  -> {result}")
            trace.append(result); transcript.append(f"`{name}` {tc.function.arguments} → {result}")
            messages.append({"role": "assistant", "content": msg.content or "",
                            "tool_calls": [tc.model_dump() for tc in calls]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            break

    # 若模型跳过了回库对账, 在此强制补齐"可证伪验证"这一科学步骤(带工具交互)
    if not any(t.startswith("`verify_predictions") for t in transcript):
        messages.append({"role": "user", "content":
            "为满足可证伪要求，你尚未完成关键的「回库对账」：请依次调用 "
            "predict_holdout 然后 verify_predictions（如需先拟合可再 fit_model）。完成后再等我的指令写报告。"})
        for _ in range(8):
            r = client.chat.completions.create(model=args.model, messages=messages,
                                               tools=build_tools(args.problem), tool_choice="auto",
                                               max_tokens=900, temperature=0.2)
            msg = r.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                break
            for tc in calls[:1]:
                name, a = _pick(tc)
                result = exec_tool(name, a)
                print(f"[tool][补] {name} {tc.function.arguments}\n  -> {result}")
                trace.append(result); transcript.append(f"`{name}` {tc.function.arguments} → {result}")
                messages.append({"role": "assistant", "content": msg.content or "",
                                "tool_calls": [tc.model_dump() for tc in calls]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            if any(t.startswith("`verify_predictions") for t in transcript):
                break

    def _gen() -> str:
        _kw = {}
        try:
            if "intern-ai.org.cn" in str(client.base_url):
                _kw = {"extra_body": {"thinking_mode": False}}
        except Exception:  # noqa: BLE001
            _kw = {}
        return client.chat.completions.create(model=args.model, messages=messages,
                                              max_tokens=1800, temperature=0.2,
                                              **_kw).choices[0].message.content or ""

    final = ""; verdict, ungrounded = "needs_grounding", []
    did_verify = any(t.startswith("`verify_predictions") for t in transcript)
    for _ in range(3):
        final = _gen()
        g = verify(final, trace)
        print(f"\n[门禁] {g['verdict']} unsubstantiated={g['unsubstantiated']}")
        reasons = []
        if not did_verify:
            reasons.append("还差可证伪验证: 必须先 predict_holdout, 再用 verify_predictions 回库与真实值对账")
        if g["verdict"] != "pass":
            reasons.append(f"以下数值不在真实工具轨迹中、无法溯源: {g['unsubstantiated']}")
        if len(final.strip()) < 200:
            reasons.append("报告过于简短(空转嫌疑), 请按完整结构重写")
        if not reasons:
            verdict, ungrounded = "pass", []
            break
        ungrounded = g["unsubstantiated"]
        messages.append({"role": "user", "content":
            "未交付: " + "; ".join(reasons) + "。请补齐后按(研究问题/数据与方法/结果分析/预测-对账/结论与局限/下一步)重写完整报告。"})
        messages.append({"role": "assistant", "content": final})

    rep = OUT / f"ai4s_real_open_{args.problem}_report.md"
    header = (f"# 开放自然科学问题研究 — 书生 Intern-S2 × Huginn · 真实数据\n\n"
              f"> 问题: {prob['title']}\n> 数据来源: 真实公开库(一次抓取缓存在 out/, 可复现)\n"
              f"> **结论证伪门禁: {verdict}**（未落地主张: {ungrounded or '无'}）\n\n"
              f"## 一、工具执行轨迹（门禁证据，含真实值）\n\n")
    body = "\n".join(f"- {t}" for t in transcript)
    footer = "\n\n## 二、结论(开放)\n\n"
    rep.write_text(header + body + footer + final.strip() + "\n\n---\n*回库对账 + 结论证伪门禁: 每个数值均可复现。*\n", encoding="utf-8")
    print("\n报告:", rep.resolve())
    print("门禁:", verdict, ungrounded)
    print("\n---- 正文(前1200字) ----\n", final[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())