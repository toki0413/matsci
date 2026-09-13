"""research_imagination_real —— 真实 LLM 想象末态盆地探测(最后一击).

承接 research_imagination_basins.py(数值代理)与双路分形负结论. 这次用**真实 InternLM
(书生)模型**跑 imagine, 回答互补的可证伪问题:

  真实想象末态在数值/语义空间, 是否形成**多个自洽盆地(多吸引域机制)**?
  还是坍缩成单一稳定机制(单盆地)?

严格分形 β 为什么这里不测(诚实边界, 前序已声明):
  真实 LLM 想象是**离散采样文本**, 不是连续参数空间网格. P_cross(δ) 需要网格点,
  稀疏离散点无法定义 δ → β 不可测. 因此这一击不冒充"测分形维数", 只测"是否存在
  多个盆地 + 分离度" —— 这是数值代理的互补, 不重复同样测量.

方法:
  - parent 假设 = 真实科学断言(掺杂半导体电导率 σ=neμ), predictions 连续.
  - 三族(self-llic imagination.py 的 algebraic/topological/order) LLM 结构变换,
    每族重复 sampling(n=4, 温度高) → 一批"想象末态".
  - 数值轴: predictions 向量, 手写 KMeans inertia(1..3) 判是否存在多焦点.
  - 语义轴: ST 嵌入 description 文本, 同样 inertia + 均值距离.
  判读: inertia(1)/inertia(2)≈1 → 单盆地; <<1 → 需要多盆地; 且看 2-簇 分离度.

零训练/不更新权重; 纯推理生成末态. key 从 /tmp/intern_key 读, 命令不出现明文.
"""
from __future__ import annotations

import os
import sys
import json
import urllib.request
import numpy as np

# 让 import 用 /workspace/agent 下的 imagination
_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if "/workspace/agent" not in sys.path:
    sys.path.insert(0, "/workspace/agent")

KEY = os.environ.get("INTERN_KEY")
if not KEY:
    KEY = open("/tmp/intern_key").read().strip()
BASE = os.environ.get("INTERN_BASE", "https://chat.intern-ai.org.cn/api/v1")
MODEL = os.environ.get("INTERN_MODEL", "intern-s1")
MAX_TOKENS = int(os.environ.get("INTERN_MAX_TOKENS", "4000"))
TEMP = float(os.environ.get("INTERN_TEMP", "0.9"))
THINKING = os.environ.get("INTERN_THINKING", "1") == "1"


def call_chat(system: str, user: str) -> str:
    """调 Intern chat.completions(OpenAI 兼容, 走代理 env, 可开 thinking_mode).

    thinking_mode=true 时书生的推理进独立通道, content 留最终回答;
    因而 JSON 遵循率应显著提升(修掉此前 CoT 挤爆/截断问题)。
    返回 assistant content 全文(含可能的 thinking 前缀, 解析时容错).
    """
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMP,
    }
    if THINKING:
        body["thinking_mode"] = True
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        obj = json.loads(resp.read().decode())
    msg = obj["choices"][0]["message"]
    # content 可能是 str; 若含独立 thinking 字段则忽略之, 只用 content
    return msg.get("content") or msg.get("content_part") or ""


def _last_json(text: str) -> dict | None:
    """取文本中**最后一个**平衡 {...} 的字典(避开 thinking 叙事里首个 {})."""
    depth = 0
    start = -1
    last = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        last = json.loads(text[start:i + 1])
                    except Exception:  # noqa: BLE001
                        last = None
    return last


def _h_from_obj(obj: dict):
    """从解析出的 dict 构造 Hypothesis(复刻 imagination 的字段校验)."""
    from huginn.metacog.imagination import Hypothesis
    desc = obj.get("new_description")
    if not isinstance(desc, str) or not desc.strip():
        return None
    preds = obj.get("new_predictions")
    if not isinstance(preds, dict) or not preds:
        return None
    try:
        preds_clean = {str(k): float(v) for k, v in preds.items()}
    except (TypeError, ValueError):
        return None
    try:
        n_params = max(1, int(obj.get("new_n_params", 1)))
    except (TypeError, ValueError):
        n_params = 1
    return Hypothesis(h_id="h_imagine_real", description=desc.strip(),
                      predictions=preds_clean, n_params=n_params)


def main():
    from huginn.metacog.imagination import (
        _build_transform_prompt,
        _parse_transform_response,
        Hypothesis,
    )

    parent = Hypothesis(
        h_id="h0",
        description=(
            "Doped semiconductor: carrier density n, conductivity sigma = n e mu, "
            "mobility mu bounded by ionized impurity scattering"
        ),
        predictions={"conductivity": 1.0, "mobility": 100.0},
        n_params=2,
    )

    keys = ("conductivity", "mobility")
    records = []  # (transform, rep, predictions, description)
    transforms = ("algebraic", "topological", "order")
    REPS = int(os.environ.get("REPS", "3"))
    # 加固 JSON 约束, 抑制书生模型的自由推理(否则 JSON 被 CoT 挤到截断)
    _HARD_JSON = (
        "\nHARD RULES: Output ONLY a single, well-formed JSON object and nothing else. "
        "Do NOT add thinking, reasoning, analysis, markdown fences, or any text "
        "before or after the JSON. The first character must be '{' and the last must be '}'."
    )
    print("=" * 72)
    print(f"真实 imagine 末态采样  model={MODEL}  每族抽样={REPS}  温度={TEMP}")
    print("=" * 72)
    for tt in transforms:
        sys_t, usr_t = _build_transform_prompt(parent, tt)
        sys_t = sys_t[:-1] + _HARD_JSON  # 移除原句点, 附加硬约束
        for rep in range(REPS):
            text = call_chat(sys_t, usr_t)
            new_h = _parse_transform_response(text)
            if new_h is None:  # thinking 叙事里首个 {} 误取 → 退回取最后一个 JSON
                obj = _last_json(text)
                new_h = _h_from_obj(obj) if obj else None
            if new_h is None:
                print(f"[{tt:>9} rep={rep}] 解析失败 -> 跳过")
                continue
            preds = [new_h.predictions.get(k) for k in keys]
            records.append((tt, rep, preds, new_h.description))
            print(f"[{tt:>9} rep={rep}] preds={new_h.predictions}")
            print(f"    desc: {new_h.description[:120]}")

    if not records:
        print("无有效末态, 退出.")
        return

    print("\n── 按族分析: 单机制指令下多次采样是否自发分裂成多簇(真'多山谷') ──")
    # 跨族混合会被 prompt 分簇污染, 不作为判断; 每族独立看族内自发分裂.
    for tt in transforms:
        f_recs = [r for r in records if r[0] == tt]
        if not f_recs:
            print(f"\n[{tt}] 无样本")
            continue
        print(f"\n[{tt}] 样本={len(f_recs)}")
        # 数值轴
        P = np.asarray([[r[2][0], r[2][1]] for r in f_recs if all(x is not None for x in r[2])])
        if len(P) >= 3:
            inert = _kmeans_inertias(P, k_max=2, seed=7)
            _report_spontaneous(inert, len(P), "数值")
        # 语义轴
        try:
            from sentence_transformers import SentenceTransformer
            st = _get_st()
            vecs = st.encode([r[3] for r in f_recs], normalize_embeddings=True,
                             show_progress_bar=False)
            inert = _kmeans_inertias(np.asarray(vecs), k_max=2, seed=7)
            _report_spontaneous(inert, len(vecs), "语义")
        except Exception as e:  # noqa: BLE001
            print(f"    语义轴不可用: {e!r}")

    print("\n判读(诚实): 样本={} 每族, 离散采样非网格, 此为定性信号非严格分形; "
          "聚焦'单机制自发多源性'而非跨族. ".format(REPS))


def _get_st():
    import os as _os
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


def _report_spontaneous(inert, n, axis):
    i1, i2 = inert[1], inert[2]
    ratio = i2 / i1 if i1 else float("inf")
    print(f"    {axis}轴 n={n}  inertia1={i1:.5f}  inertia2={i2:.5f}  ratio(2/1)={ratio:.3f}")
    if ratio > 0.85:
        print(f"      → 单盆地: 该族指令下末态自发收敛到单一稳定模型, 无多山谷")
    else:
        print(f"      → 自发多盆地: 该族指令下末态分裂为多个自洽模型(2簇增益显著)")


def _kmeans_inertias(X, k_max=3, seed=7, iters=60):
    r = np.random.default_rng(seed)
    best = {}
    for k in range(1, k_max + 1):
        cent = X[r.permutation(len(X))[:k]].copy()
        for _ in range(iters):
            d = np.linalg.norm(X[:, None, :] - cent[None, :, :], axis=2)
            lab = np.argmin(d, axis=1)
            nlab = [np.where(lab == j)[0] for j in range(k)]
            newcent = np.stack([X[idxs].mean(axis=0) if len(idxs) else cent[j]
                                for j, idxs in enumerate(nlab)], axis=0)
            if np.allclose(newcent, cent, atol=1e-9):
                cent = newcent
                break
            cent = newcent
        lab = np.argmin(np.linalg.norm(X[:, None, :] - cent[None, :, :], axis=2), axis=1)
        best[k] = float(np.sum((X - cent[lab]) ** 2))
    return best


def _report_inertia(inertias, n, label):
    i1 = inertias[1]
    i2 = inertias[2]
    i3 = inertias.get(3, i1)
    ratio12 = i2 / i1 if i1 else float("inf")
    ratio23 = i3 / i2 if i2 else float("inf")
    print(f"  {label} n={n}  inertia k=1 {i1:.4f}  k=2 {i2:.4f}  k=3 {i3:.4f}")
    print(f"     ratio(2/1)={ratio12:.3f}   ratio(3/2)={ratio23:.3f}")
    if ratio12 > 0.90:
        print(f"    → 单盆地(1簇已够): 想象末态坍缩到单一稳定机制, 无多吸引域")
    elif ratio23 < 0.90:
        print(f"    → 多盆地(3簇仍有增益): 存在>2个自洽机制簇")
    else:
        print(f"    → 双盆地(2簇起决定增益): 存在2个自洽机制簇")


if __name__ == "__main__":
    main()