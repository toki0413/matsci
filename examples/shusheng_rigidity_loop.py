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
    python examples/shusheng_rigidity_loop.py --rounds 6 --budget-s 7200 --session s2

产物 (research_outputs/shusheng_nn_rigidity_probe/):
    loop_transcript<sfx>.md   逐轮: 书生判断 + 它选的配置 + 该轮实测表 + 书生的理由
    loop_verdict<sfx>.md      书生最终裁决 (原文)
    loop_state<sfx>.json      机器可读的全过程状态
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

# ── 预算: 软限制 + 撞顶自动续投 (对齐 Huginn 的 HUGINN_BUDGET_APPROVAL=auto 语义) ──
# 书生在框内自由选; 请求一旦冲破当前上限, 自动续投 ×RENEW_FACTOR, 最多 MAX_RENEWALS 次;
# 续投额度用尽后仍是硬刹车 (不无头烧钱). 成本按宽度加权: w=8->1, 32->2, 64->4, 128->8, 256->16.
ALLOWED_KINDS = ["poly", "osc", "hi", "hism", "fat"]
ALLOWED_WIDTHS = [4, 8, 16, 32, 64, 128, 256]
ALLOWED_NS = [1, 2, 3, 4, 6, 8, 16, 32]
ALLOWED_ADAM = [2000, 3000, 4000, 5000, 8000, 12000]
SOFT_TASKS = 96      # 软限制: 初始任务数上限
SOFT_COST = 128      # 软限制: 初始加权成本上限
RENEW_FACTOR = 1.5   # 撞顶后每次续投的放大系数 (与 Huginn 一致)
MAX_RENEWALS = 6     # 最多自动续投次数 (对齐 HUGINN_BUDGET_MAX_RENEWALS=6)
MAX_WIDTHS = 7
MAX_NS = 8
MAX_SEEDS = 5
MIN_NS = 3      # ns 不得被砍到 3 以下 (否则无法检验"不再回升", N_c 不可信)
MIN_WIDTHS = 2  # widths 不得被砍到 2 以下 (至少要能拟合一条斜率)
JOBS = 3


#: adam 档 -> 成本倍率. 步数拉满但宽度小的网格不能显得便宜 (实际更慢).
_ADAM_COST = {2000: 1, 3000: 1, 4000: 1, 5000: 1, 8000: 2, 12000: 3}


def _cost(w: int, adam: int = 5000) -> int:
    """单任务成本代理: 宽度与训练步数都计入.

    宽度 (w=8->1, 32->2, 64->4, 128->8) × adam 倍率 (>=8000 再 ×2, 12000 再 ×3).
    """
    return max(1, w // 16) * _ADAM_COST.get(adam, 1)


# 锚点: 只给**结构事实** (ODE / 解空间维数 / 约束点位置 / 残差量纲). 不下判断, 不预设结论.
_ANCHORS = {
    "poly": "ODE u''=2; 解空间 {x^2+ax+b}, 2 维; N 个点值约束取 [0,1] 上 N 个相异点; 残差量纲 |f|=2.",
    "osc": "ODE u''=-(2pi)^2 cos(2pi x); 解空间 {cos(2pi x)+ax+b}, 2 维; 约束点同上(相异); |f|max≈39.5.",
    "hi": "ODE u''=-(8pi)^2 cos(8pi x); 解空间 {cos(8pi x)+ax+b}, 2 维; 约束点同上(相异); |f|max≈632.",
    "hism": "ODE u''=-cos(8pi x); 解空间 {cos(8pi x)/(8pi)^2+ax+b}, 2 维; 约束点同上(相异); |f|max=1. 与 hi 同频, 仅 forcing 尺度不同.",
    "fat": "ODE u''=2; 解空间 {x^2+ax+b}, 2 维; 但 N 个点值约束**全部落在同一点 x=0.5**; |f|=2.",
}

_QUESTION = """神经网络的"泛化行为"能否作为 bootstrap 解空间刚性的探针?
若解空间一维(Veneziano 情形, 唯一性成立), 小模型见过有限样本后应能泛化到没见过的输入;
若解空间高维/连续(隧穿振幅), 同样的小模型应泛化失败. 于是问:
"唯一性是否成立"这个纯数学问题, 能否变成"多大容量的模型能在留出集上零违规"这个可测量量?"""

_CRITERIA = """实验器给出的确定性判据 (协议定义, 不可改动, 你据此判读):
- 观测量: 在**未参与训练**的 128 个留出中点上测 V_ho = 均方( (u''-f)^2 + (u-u*)^2 ).
- pass_frac: 该 (w,N) 下 V_tr < 1e-10 的种子占比. 它只反映优化管道是否收敛, 与解空间维数无关.
- N_c(w) = 最小的 N, 使得所有 N'>=N 都有 V_ho < 1e-8; 若无此 N 则 N_c=None.
- 拟合 N_c(w) ~ w^beta:
    |beta|<0.15                  -> H1: 饱和点是解的性质;
    beta>0.3 且 R²>0.85          -> H2: 饱和点是模型记忆容量/归纳偏置;
    介于两者                     -> 灰区;
    N_c 大量为 None              -> 无法判定 (第三类结果).

参考(非指令): 上游红队阶段是你自己此前的输出, 其中列过若干"会使结论翻转的混淆"以及一个最小决定性
实验设计. 是否仍成立、要不要把这些对照做掉、做到什么程度算够, 由你在循环里自行决定. 本脚本不替你
选实验, 也不替你下结论."""


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


def render_budget(recs: list[dict], caps: dict) -> str:
    done = "\n".join(
        f"  - kind={r['kind']} widths={r['widths']} ns={r['ns']} seeds={r['seeds']} adam={r['adam_steps']}"
        for r in recs
    ) or "  (无)"
    return f"""本机 3 核. 单轮资源上限 (双重约束, 必须同时满足):
- 训练任务数 len(widths)*len(ns)*seeds <= {caps['tasks']}
- 加权成本 sum_任务 max(1, w//16)*adam倍率 <= {caps['cost']}   (w=8记1/32记2/64记4/128记8/256记16; adam>=8000 再×2, 12000 再×3)
软限制初值 tasks={SOFT_TASKS} cost={SOFT_COST}; 请求撞顶会**自动续投** ×{RENEW_FACTOR} (最多 {MAX_RENEWALS} 次),
已用 {MAX_RENEWALS - caps['renewals_left']}/{MAX_RENEWALS}, 剩余 {caps['renewals_left']} 次; 续投用尽才是硬刹车.
=> 想要更大的网格可以直接提, 不必为省预算自我压缩; 真撞硬刹车时我会把额度反馈给你重规划.

你只能选:
- kind ∈ {ALLOWED_KINDS}
- widths ⊆ {ALLOWED_WIDTHS}, 最多 {MAX_WIDTHS} 个
- ns ⊆ {ALLOWED_NS}, 最多 {MAX_NS} 个
- seeds ∈ 1..{MAX_SEEDS}
- adam_steps ∈ {ALLOWED_ADAM}
越界会被自动削减: 依次砍 ns(不低于 {MIN_NS} 个) -> seeds(不低于 1) -> widths(不低于 {MIN_WIDTHS} 个);
若削到地板仍超预算, 该轮会被**拒绝**并把预算反馈给你重规划. 削减/拒绝都会在下一轮告知你.

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
def _clamp_run(run: dict, caps: dict) -> tuple[dict | None, list[str]]:
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

    def _tot(ws_, ns_, sd_):
        return len(ws_) * len(ns_) * sd_, sum(_cost(w, adam) for w in ws_) * len(ns_) * sd_

    # 撞顶自动续投 (软限制 -> 硬刹车之间, 对齐 Huginn auto 语义): 请求冲破当前上限就 ×3/2 续投,
    # 最多 MAX_RENEWALS 次, 额度跨轮持久. 目的是**不替书生削网格**——它想跑就让它跑.
    while True:
        t, c = _tot(ws, ns, seeds)
        if (t <= caps["tasks"] and c <= caps["cost"]) or caps["renewals_left"] <= 0:
            break
        caps["tasks"] = caps["tasks"] * 3 // 2
        caps["cost"] = caps["cost"] * 3 // 2
        caps["renewals_left"] -= 1
        notes.append(f"请求撞顶 -> 自动续投: tasks<={caps['tasks']} cost<={caps['cost']} "
                     f"(剩余续投 {caps['renewals_left']}/{MAX_RENEWALS})")

    def _over(ws_, ns_, sd_):
        t, c = _tot(ws_, ns_, sd_)
        return t > caps["tasks"] or c > caps["cost"]

    # 续投额度用尽后才回落: 依次削减 ns -> seeds -> widths, 保持科学有效性 (ns>=MIN_NS, widths>=MIN_WIDTHS);
    # 若削到地板仍超预算, 直接拒绝并把预算反馈给书生, 让它自己重规划 (不替它改设计).
    while _over(ws, ns, seeds) and len(ns) > MIN_NS:
        ns = ns[:-1]
        notes.append(f"续投额度用尽仍超上限, 砍 ns -> {ns}")
    while _over(ws, ns, seeds) and seeds > 1:
        seeds -= 1
        notes.append(f"续投额度用尽仍超上限, 砍 seeds -> {seeds}")
    while _over(ws, ns, seeds) and len(ws) > MIN_WIDTHS:
        ws = ws[:-1]
        notes.append(f"续投额度用尽仍超上限, 砍 widths -> {ws}")
    if _over(ws, ns, seeds):
        t, c = _tot(ws, ns, seeds)
        return None, notes + [
            f"请求超硬刹车且削到地板后仍不够 (tasks={t}>{caps['tasks']} 或 cost={c}>{caps['cost']}); "
            f"续投额度已用尽 ({MAX_RENEWALS}/{MAX_RENEWALS}); "
            f"请减小网格: 例如减少 widths 个数或把最大宽度从 256 降到 128/64, 再提交"]
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
    ap.add_argument("--rounds", type=int, default=10, help="书生最多自主决策几轮")
    ap.add_argument("--budget-s", type=float, default=21600.0, help="循环墙钟预算(秒)")
    ap.add_argument("--session", default="", help="产物文件名后缀, 避免覆盖历次循环记录")
    ap.add_argument("--model", default=os.environ.get("INTERNLM_MODEL", "intern-s2"))
    ap.add_argument("--base", default=os.environ.get("INTERNLM_BASE_URL",
                                                     "https://chat.intern-ai.org.cn/api/v1"))
    args = ap.parse_args()
    sfx = f"_{args.session}" if args.session else ""

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
    caps = {"tasks": SOFT_TASKS, "cost": SOFT_COST, "renewals_left": MAX_RENEWALS}
    state["caps"] = caps
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
{render_budget(recs, caps)}
{f'''【上一轮你的决定】
- action={last_decision.get('action')}  run={last_decision.get('run')}
- 你的理由: {last_decision.get('reason')}
- 执行/校验备注: {note}
''' if last_decision else ''}
【累计证据 (这是你唯一的经验来源)】
{render_evidence(recs)}

【你的任务】
你是**主研者**, 也是唯一决策者. 基于上面的证据自主决定下一步:
- action="run": 继续做实验, 给出你选的配置 (用哪个锚点、扫哪些宽度与约束数、要不要做对照, 由你定);
- action="conclude": 你认为证据已足以回答原始问题, 现在结题.

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

        cfg, notes = _clamp_run(dec.get("run") or {}, caps)
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

        tag = f"loop{rd:02d}{sfx}"
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
        (REPORT_DIR / f"loop_state{sfx}.json").write_text(
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

    (REPORT_DIR / f"loop_transcript{sfx}.md").write_text("\n".join(transcript), encoding="utf-8")
    md = ["# 书生自主循环 · 最终裁决", "",
          f"> 书生 `{args.model}` 在读入 {len(recs)} 份实验证据后作出, 未经本地改写.", "",
          "## 裁决 (原文)", "", "```", verdict_text or "(无)", "```", "",
          "## 结构化字段", "",
          f"- verdict: {conclusion.get('verdict')}",
          f"- conclusion: {conclusion.get('conclusion')}",
          f"- answer_to_question: {conclusion.get('answer_to_question')}",
          f"- strongest_dissent: {conclusion.get('strongest_dissent')}",
          f"- remaining_uncertainty: {conclusion.get('remaining_uncertainty')}", ""]
    (REPORT_DIR / f"loop_verdict{sfx}.md").write_text("\n".join(md), encoding="utf-8")
    state["finished"] = time.time()
    (REPORT_DIR / f"loop_state{sfx}.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 循环结束 ==")
    print(f"逐轮记录 -> {(REPORT_DIR / f'loop_transcript{sfx}.md').resolve()}")
    print(f"最终裁决 -> {(REPORT_DIR / f'loop_verdict{sfx}.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())