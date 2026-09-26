"""书生 × Huginn · 解空间刚性探针的**自主循环**.

定位: 本脚本不替书生下结论. 它只提供
  (a) 一个受预算约束的实验执行器 (调用 nn_rigidity_decoupling.py);
  (b) 把已有/新产的证据压缩成可读表格;
  (c) 每轮把"证据 + 预算 + 已跑过的配置"喂给书生, 让书生 **自己** 决定
      "再跑一组合什么实验" 还是 "可以结题了, 我的裁决是 ...".

循环: 书生决策 -> 本地执行 -> 证据回流 -> 书生再决策 ... 直到书生 conclude
或触发预算上限 (轮数/墙钟). 结束后把书生的最终裁决原样落盘.

用法:
    export INTERNLM_API_KEY=<书生 token>
    python examples/shusheng_rigidity_loop.py --rounds 4 --budget-s 5400

产物 (research_outputs/shusheng_nn_rigidity_probe/):
    loop_transcript.md   逐轮: 书生判断 + 它选的配置 + 该轮实测表 + 书生的理由
    loop_verdict.md      书生最终裁决 (原文)
    loop_state.json      机器可读的全过程状态
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except Exception:  # noqa: BLE001
    sys.exit("需要 pip install openai")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
EXP = EXAMPLES / "nn_rigidity_decoupling.py"
OUTDIR = EXAMPLES / "out" / "nn_rigidity_decoupling"
REPORT_DIR = ROOT / "research_outputs" / "shusheng_nn_rigidity_probe"

# ── 预算硬约束: 书生只能在框内选, 保证单轮几十分钟内可跑完 ──────────────────
ALLOWED_KINDS = ["poly", "osc", "hi", "hism", "fat"]
ALLOWED_WIDTHS = [4, 8, 16, 32]
ALLOWED_NS = [1, 2, 3, 4, 8]
ALLOWED_ADAM = [2000, 3000, 4000, 5000]
MAX_TASKS = 12
MAX_WIDTHS = 3
MAX_NS = 4
MAX_SEEDS = 2
JOBS = 3

_ANCHORS = {
    "poly": "u''=2, u*=x^2+0.3x-0.2. 解空间 {x^2+ax+b} 2 自由度; N>=2 个**相异**点约束 -> 唯一(刚性). 曲率小/forcing O(1), 优化最容易.",
    "osc": "u''=-(2pi)^2 cos(2pi x), u*=cos(2pi x)+0.3x-0.2. 同样 2 自由度 -> N>=2 刚性. 中等频率.",
    "hi": "u''=-(8pi)^2 cos(8pi x), u*=cos(8pi x)+0.3x-0.2. 同样 2 自由度 -> N>=2 刚性. 高频 **且 forcing 量纲巨大 (|f|~632)**, 已知会让优化崩溃.",
    "hism": "u''=-cos(8pi x), u*=cos(8pi x)/(8pi)^2+0.3x-0.2. 与 hi 同频同自由度, 但 forcing 归一为 O(1). 用于分离'数值尺度'与'高频表达'两个混淆.",
    "fat": "u''=2, u*=x^2+0.3x-0.2, 但 N 个点值约束**全部堆在 x=0.5** -> 约束秩恒为 1, 自由参数 a 永不消失 -> V_ho 恒 O(1), 任何 N/w 都不饱和. **真胖(非唯一)阴性对照**.",
}

_QUESTION = """神经网络的"泛化行为"能否作为 bootstrap 解空间刚性的探针?
若解空间一维(Veneziano 情形, 唯一性成立), 小模型见过有限样本后应能泛化到没见过的输入;
若解空间高维/连续(隧穿振幅), 同样的小模型应泛化失败. 于是问:
"唯一性是否成立"这个纯数学问题, 能否变成"多大容量的模型能在留出集上零违规"这个可测量量?"""

_CRITERIA = """实验器给出的确定性判据 (你据此判读, 不可改动定义):
- 观测量: 在**未参与训练**的 128 个留出中点上测 V_ho = 均方( (u''-f)^2 + (u-u*)^2 ).
- pass_frac: 该 (w,N) 下 V_tr < 1e-10 的种子占比. pass_frac=100% 才说明优化管道真的收敛,
  此时 V_ho 才可信; pass_frac=0 时该配置的 V_ho 无意义(优化没到位, 不是解的性质).
- N_c(w) = 最小的 N, 使得所有 N'>=N 都有 V_ho < 1e-8; 若无此 N 则 N_c=None.
- 拟合 N_c(w) ~ w^beta:
    beta≈0 (|beta|<0.15)                  -> H1: 饱和点是**解的性质**, 探针站得住;
    beta>0.3 且 R²>0.85                   -> H2: 饱和点是**模型记忆容量/归纳偏置**, 探针被驳回;
    介于两者                              -> 灰区;
    N_c 大量为 None (无饱和)              -> "无法判定" (第三类结果).

已知的混淆/红队意见 (你必须据此警惕):
1. 优化可达性: 优化不足会伪装成"解太胖"(假胖) -> 必须看 pass_frac.
2. 架构表达瓶颈/数值尺度: 高频或大 forcing 会让刚性问题的 V_tr 也下不去, 从而伪装成"胖".
   注意 'hi' 与 'hism' 的对照正是为分离这一点而设.
3. 阈值敏感: V_tr 门槛(1e-10)严于 V_ho 判据(1e-8), 出现"N_c(全部种子) 有限 而 N_c(合规种子)=None"
   这类翻转时, 结论取决于用哪套阈值.
4. "真胖"阴性对照 'fat' 用于确认判据在非唯一时确实给 N_c=None (即判据有灵敏度, 不是永远报 None).
5. 本机只有 3 核: 过参数化大区 (w>=256) 与 N>=16 暂不可及, 这是当前证据的硬边界."""


# ── 证据收集与压缩 ────────────────────────────────────────────────────────
def _lg(x) -> str:
    if x is None or (isinstance(x, float) and (x != x or x in (float("inf"), float("-inf")))):
        return "nan"
    if x <= 0:
        return "-inf"
    return f"{x:.1e}"


def load_evidence() -> list[dict]:
    """扫描所有 decoupling_*.json, 压成紧凑记录 (按 kind/tag 去重保留最新)."""
    recs = []
    for p in sorted(OUTDIR.glob("decoupling_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — 坏文件不阻断循环
            continue
        cfg = d.get("config", {})
        recs.append({
            "file": p.name,
            "kind": cfg.get("kind"),
            "tag": cfg.get("tag") or "",
            "widths": cfg.get("widths"),
            "ns": cfg.get("ns"),
            "seeds": cfg.get("seeds"),
            "adam_steps": cfg.get("adam_steps"),
            "rows": d.get("rows", []),
            "N_c": d.get("N_c", {}),
            "fit": d.get("fit", {}),
            "verdict": d.get("verdict", ""),
            "elapsed_s": d.get("elapsed_s"),
            "source": "已有" if "loop" not in (cfg.get("tag") or "") else "本轮",
        })
    return recs


def _evidence_sig(r: dict) -> tuple:
    return (r["kind"], tuple(r.get("widths") or []), tuple(r.get("ns") or []),
            r.get("seeds"), r.get("adam_steps"))


def render_evidence(recs: list[dict]) -> str:
    if not recs:
        return "(尚无任何实验证据)"
    out = []
    for r in recs:
        out.append(f"\n### {r['file']}  [锚点 {r['kind']}, 来源 {r['source']}]")
        out.append(f"锚点含义: {_ANCHORS.get(r['kind'], '?')}")
        out.append(f"网格: w={r['widths']} N={r['ns']} seeds={r['seeds']} adam={r['adam_steps']}")
        out.append("  w   N | pass | log10 V_tr | log10 V_ho(all)")
        for row in r["rows"]:
            out.append(f"  {row['w']:>3} {row['n']:>3} | {row['pass_frac']:>4.0%} | "
                       f"{_lg(row['v_tr_med']):>10} | {_lg(row['v_ho_med_all']):>10}")
        out.append(f"  N_c(全部种子): {r['N_c'].get('all')}")
        out.append(f"  N_c(仅合规种子): {r['N_c'].get('good')}")
        f = r["fit"] or {}
        for lab in ("全部种子", "仅合规种子"):
            if lab in f:
                out.append(f"  beta[{lab}]={f[lab].get('beta')} R2={f[lab].get('r2')} k={f[lab].get('k')}")
        out.append(f"  程序裁决: {r['verdict']}")
    return "\n".join(out)


def render_budget(recs: list[dict]) -> str:
    done = "\n".join(
        f"  - kind={r['kind']} widths={r['widths']} ns={r['ns']} seeds={r['seeds']} adam={r['adam_steps']}"
        for r in recs
    ) or "  (无)"
    return f"""本机 3 核, 单轮必须控制在 {MAX_TASKS} 个训练任务以内. 你只能选:
- kind ∈ {ALLOWED_KINDS}
- widths ⊆ {ALLOWED_WIDTHS}, 最多 {MAX_WIDTHS} 个
- ns ⊆ {ALLOWED_NS}, 最多 {MAX_NS} 个
- seeds ∈ 1..{MAX_SEEDS}
- adam_steps ∈ {ALLOWED_ADAM}
- 约束: len(widths)*len(ns)*seeds <= {MAX_TASKS}

**已跑过的配置 (签名 kind|widths|ns|seeds|adam, 不得重复)**:
{done}"""


# ── 书生调用 ──────────────────────────────────────────────────────────────
def _compat_kwargs(client: OpenAI) -> dict:
    base = str(getattr(client, "base_url", None) or "")
    if "intern-ai.org.cn" in base or "/intern" in base:
        return {"extra_body": {"thinking_mode": False}}
    return {}


def _ask(client: OpenAI, model: str, prompt: str, max_tokens: int, retries: int = 3) -> str:
    last = None
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
                timeout=600,
                **_compat_kwargs(client),
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 — 网络/限流重试
            last = exc
            print(f"      !! 书生调用失败({type(exc).__name__}): {exc}; 退避重试 {i + 1}/{retries}",
                  file=sys.stderr)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"书生 API 连续失败: {last}")


def _extract_json(text: str) -> dict | None:
    """从可能带 markdown 围栏/前后废话的回复里抠出第一个平衡的 JSON 对象."""
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        pass
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except Exception:  # noqa: BLE001
                    return None
    return None


# ── 决策校验/夹紧 ─────────────────────────────────────────────────────────
def _clamp_run(run: dict) -> tuple[dict | None, list[str]]:
    notes = []
    if not isinstance(run, dict):
        return None, ["run 字段缺失或非对象"]
    kind = run.get("kind")
    if kind not in ALLOWED_KINDS:
        return None, [f"kind={kind} 不在允许集合"]
    ws = [w for w in dict.fromkeys(run.get("widths") or []) if w in ALLOWED_WIDTHS]
    ns = [n for n in dict.fromkeys(run.get("ns") or []) if n in ALLOWED_NS]
    if len(ws) != len(run.get("widths") or []):
        notes.append(f"widths 中非法值已剔除 -> {ws}")
    if len(ns) != len(run.get("ns") or []):
        notes.append(f"ns 中非法值已剔除 -> {ns}")
    ws, ns = sorted(ws), sorted(ns)
    if len(ws) > MAX_WIDTHS:
        notes.append(f"widths 超上限, 截断为 {ws[:MAX_WIDTHS]}")
        ws = ws[:MAX_WIDTHS]
    if len(ns) > MAX_NS:
        notes.append(f"ns 超上限, 截断为 {ns[:MAX_NS]}")
        ns = ns[:MAX_NS]
    if not ws or not ns:
        return None, notes + ["widths/ns 为空"]
    seeds = int(run.get("seeds") or 1)
    if seeds < 1 or seeds > MAX_SEEDS:
        notes.append(f"seeds={seeds} 越界, 夹到 1..{MAX_SEEDS}")
        seeds = min(max(1, seeds), MAX_SEEDS)
    adam = int(run.get("adam_steps") or 5000)
    if adam not in ALLOWED_ADAM:
        near = min(ALLOWED_ADAM, key=lambda a: abs(a - adam))
        notes.append(f"adam_steps={adam} 不在允许集合, 取最近值 {near}")
        adam = near
    # 任务量夹紧: 超预算则先砍 ns 再砍 widths
    while len(ws) * len(ns) * seeds > MAX_TASKS and len(ns) > 1:
        ns = ns[:-1]
        notes.append(f"任务量超预算, 砍 ns -> {ns}")
    while len(ws) * len(ns) * seeds > MAX_TASKS and len(ws) > 1:
        ws = ws[:-1]
        notes.append(f"任务量仍超预算, 砍 widths -> {ws}")
    if len(ws) * len(ns) * seeds > MAX_TASKS:
        return None, notes + ["夹紧后仍超预算"]
    return {"kind": kind, "widths": ws, "ns": ns, "seeds": seeds, "adam_steps": adam}, notes


def run_experiment(cfg: dict, tag: str) -> dict | None:
    cmd = [sys.executable, str(EXP),
           "--kind", cfg["kind"],
           "--widths", *map(str, cfg["widths"]),
           "--ns", *map(str, cfg["ns"]),
           "--seeds", str(cfg["seeds"]),
           "--adam-steps", str(cfg["adam_steps"]),
           "--jobs", str(JOBS),
           "--tag", tag]
    print(f"      $ {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if p.returncode != 0:
        print(f"      !! 实验失败 rc={p.returncode}\n{p.stderr[-1500:]}", file=sys.stderr)
        return None
    out = OUTDIR / f"decoupling_{cfg['kind']}_{tag}.json"
    if not out.exists():
        print(f"      !! 未找到预期产物 {out}", file=sys.stderr)
        return None
    d = json.loads(out.read_text(encoding="utf-8"))
    d["_wall_s"] = time.time() - t0
    return d


def _rec_from_json(d: dict, tag: str, source: str) -> dict:
    cfg = d.get("config", {})
    return {
        "file": f"decoupling_{cfg.get('kind')}_{tag}.json",
        "kind": cfg.get("kind"), "tag": tag,
        "widths": cfg.get("widths"), "ns": cfg.get("ns"),
        "seeds": cfg.get("seeds"), "adam_steps": cfg.get("adam_steps"),
        "rows": d.get("rows", []), "N_c": d.get("N_c", {}), "fit": d.get("fit", {}),
        "verdict": d.get("verdict", ""), "elapsed_s": d.get("elapsed_s"),
        "source": source,
    }


# ── 主循环 ────────────────────────────────────────────────────────────────
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4, help="书生最多自主决策几轮")
    ap.add_argument("--budget-s", type=float, default=5400.0, help="循环墙钟预算(秒)")
    ap.add_argument("--model", default=os.environ.get("INTERNLM_MODEL", "intern-s2"))
    ap.add_argument("--base", default=os.environ.get("INTERNLM_BASE_URL",
                                                     "https://chat.intern-ai.org.cn/api/v1"))
    args = ap.parse_args()

    key = os.environ.get("INTERNLM_API_KEY")
    if not key:
        print("error: 未设置 INTERNLM_API_KEY", file=sys.stderr)
        return 2
    client = OpenAI(api_key=key, base_url=args.base)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    recs = load_evidence()
    sigs = {_evidence_sig(r) for r in recs}
    transcript = ["# 书生自主循环 · 逐轮记录", "",
                  f"> 模型: 书生 `{args.model}` @ `{args.base}`",
                  f"> 起跑时已有证据: {len(recs)} 份 JSON", ""]
    state = {"model": args.model, "rounds": [], "started": time.time()}
    t_start = time.time()
    verdict_text, conclusion = None, None
    last_decision = None
    note = ""

    for rd in range(1, args.rounds + 1):
        if time.time() - t_start > args.budget_s:
            note = "墙钟预算耗尽"
            break
        print(f"\n===== 第 {rd}/{args.rounds} 轮 · 请书生决策 =====", flush=True)

        prompt = f"""{_QUESTION}

{_CRITERIA}

【实验器: 锚点含义】
""" + "\n".join(f"- {k}: {v}" for k, v in _ANCHORS.items()) + f"""

【本轮你可用的预算与约束】
{render_budget(recs)}
{f'''【上一轮你的决定】
- action={last_decision.get('action')}  run={last_decision.get('run')}
- 你的理由: {last_decision.get('reason')}
- 执行/校验备注: {note}
''' if last_decision else ''}
【累计证据 (这是你唯一的经验来源)】
{render_evidence(recs)}

【你的任务】
你是**主研者**. 基于上面的证据自主决定下一步:
- 若还缺关键对照 (例如: 真胖对照 fat 是否给 N_c=None、hi 与 hism 的对照是否证明失败来自数值尺度、
  pass_frac 是否 100%、beta 是否真的恒为 0、灰区是否需要更细网格), 选 action="run" 并给出配置;
- 若证据已足以回答原始问题, 选 action="conclude".

**只输出一个 JSON 对象**, 不要任何解释性前后文. schema:
{{
  "thought": "你读证据后的判断, 中文, <=200字",
  "action": "run" 或 "conclude",
  "run": {{"kind": "...", "widths": [..], "ns": [..], "seeds": N, "adam_steps": N}},
  "reason": "这一步为什么能推进判别, 中文, <=120字",
  "conclude": {{"verdict": "接受/有条件接受/驳回", "conclusion": "你的最终结论, 中文", \
"answer_to_question": "对原始问题的一句话回答", "strongest_dissent": "最强反方理由", \
"remaining_uncertainty": "仍未排除的不确定性 / 需更大算力才能做的实验 (不要留空)"}}
}}
(action="run" 时 conclude 可省略; action="conclude" 时 run 可省略.)
"""
        raw = _ask(client, args.model, prompt, max_tokens=1600)
        dec = _extract_json(raw)
        if dec is None:
            note = "书生回复无法解析为 JSON, 本轮作废"
            transcript += [f"## 第 {rd} 轮 · 解析失败", "", "```", raw[:2000], "```", ""]
            print(f"      !! {note}", file=sys.stderr)
            state["rounds"].append({"round": rd, "error": "json_parse", "raw": raw[:4000]})
            continue

        action = dec.get("action")
        print(f"      书生判断: {dec.get('thought')}")
        print(f"      action={action} reason={dec.get('reason')}")
        transcript += [f"## 第 {rd} 轮 · 书生决策", "",
                       f"**判断**: {dec.get('thought')}", "",
                       f"**action**: `{action}`", "",
                       f"**理由**: {dec.get('reason')}", ""]

        if action == "conclude":
            conclusion = dec.get("conclude") or {}
            verdict_text = raw
            transcript += ["**书生宣布结题**, 裁决如下:", "",
                           f"- verdict: {conclusion.get('verdict')}",
                           f"- conclusion: {conclusion.get('conclusion')}",
                           f"- answer_to_question: {conclusion.get('answer_to_question')}",
                           f"- strongest_dissent: {conclusion.get('strongest_dissent')}", ""]
            state["rounds"].append({"round": rd, "decision": dec})
            break

        cfg, notes = _clamp_run(dec.get("run") or {})
        last_decision = dec
        if cfg is None:
            note = "配置被拒: " + "; ".join(notes)
            print(f"      !! {note}", file=sys.stderr)
            transcript += [f"**配置被拒**: {note}", ""]
            state["rounds"].append({"round": rd, "decision": dec, "rejected": notes})
            continue
        sig = (cfg["kind"], tuple(cfg["widths"]), tuple(cfg["ns"]), cfg["seeds"], cfg["adam_steps"])
        if sig in sigs:
            note = f"配置与已跑过的重复, 已拒绝: {sig}"
            print(f"      !! {note}", file=sys.stderr)
            transcript += [f"**配置重复, 已拒绝**: `{sig}`", ""]
            state["rounds"].append({"round": rd, "decision": dec, "rejected": ["duplicate"]})
            continue

        tag = f"loop{rd:02d}"
        transcript += [f"**执行配置**: `kind={cfg['kind']} widths={cfg['widths']} "
                       f"ns={cfg['ns']} seeds={cfg['seeds']} adam={cfg['adam_steps']}`", ""]
        if notes:
            transcript += [f"(校验备注: {'; '.join(notes)})", ""]
        d = run_experiment(cfg, tag)
        if d is None:
            note = "实验执行失败, 未产出结果"
            transcript += [f"**实验失败**: {note}", ""]
            state["rounds"].append({"round": rd, "decision": dec, "exec_error": True})
            continue
        rec = _rec_from_json(d, tag, "本轮")
        recs.append(rec)
        sigs.add(sig)
        note = "配置已按约束执行完毕"
        transcript += ["**本轮实测**:", "", "```", render_evidence([rec]).strip(), "```", ""]
        state["rounds"].append({"round": rd, "decision": dec, "config": cfg,
                                "record": {k: rec[k] for k in
                                           ("kind", "widths", "ns", "seeds", "adam_steps",
                                            "N_c", "fit", "verdict", "elapsed_s", "file")}})
        print(f"      -> 本轮完成: N_c={rec['N_c'].get('all')} 裁决={rec['verdict']}")
        (REPORT_DIR / "loop_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    # 未结题 -> 强制要一次最终裁决
    if conclusion is None:
        print("\n===== 预算结束 · 强制书生给出最终裁决 =====", flush=True)
        prompt = f"""{_QUESTION}

{_CRITERIA}

【全部证据 (这是你做最终裁决的依据)】
{render_evidence(recs)}

【情况】
本轮预算/轮数已用尽, 你必须现在结题. {note}
请**只输出一个 JSON 对象**:
{{"verdict": "接受/有条件接受/驳回", "conclusion": "最终结论, 中文", \
"answer_to_question": "对原始问题的一句话回答", "strongest_dissent": "最强反方理由", \
"remaining_uncertainty": "仍未排除的不确定性 / 需要更大算力才能做的实验"}}
"""
        raw = _ask(client, args.model, prompt, max_tokens=2200)
        verdict_text = raw
        conclusion = _extract_json(raw) or {"conclusion": raw}
        transcript += ["# 最终裁决 (预算结束, 书生被迫结题)", "", f"```json\n{raw}\n```", ""]

    (REPORT_DIR / "loop_transcript.md").write_text("\n".join(transcript), encoding="utf-8")
    md = ["# 书生自主循环 · 最终裁决", "",
          f"> 书生 `{args.model}` 在读入 {len(recs)} 份实验证据后作出, 未经本地改写.", "",
          "## 裁决 (原文)", "", "```", verdict_text or "(无)", "```", "",
          "## 结构化字段", "",
          f"- verdict: {conclusion.get('verdict')}",
          f"- conclusion: {conclusion.get('conclusion')}",
          f"- answer_to_question: {conclusion.get('answer_to_question')}",
          f"- strongest_dissent: {conclusion.get('strongest_dissent')}",
          f"- remaining_uncertainty: {conclusion.get('remaining_uncertainty')}", ""]
    (REPORT_DIR / "loop_verdict.md").write_text("\n".join(md), encoding="utf-8")
    state["finished"] = time.time()
    (REPORT_DIR / "loop_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 循环结束 ==")
    print(f"逐轮记录 -> {(REPORT_DIR / 'loop_transcript.md').resolve()}")
    print(f"最终裁决 -> {(REPORT_DIR / 'loop_verdict.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())