"""交互可解释性 —— 博弈交互分解 + 等效交互度量 (对齐 上海交大张拳石团队).

存在目的 (为什么一个科考/科算 agent 需要它):
  Huginn 在 AI4S 里常用"代理模型(surrogate)"替代昂贵仿真(搜→读→算→做)。治理上的
  难处不是"输出对不对"(那是 `law_model.reconcile` 数值对账管的), 而是"**推理结构**
  是否与真实物理定律一致"。一个代理完全可能:
    - 在训练分布上数值吻合 (reconcile 全绿);
    - 却依赖一条**虚假交互(shortcut/混淆)**——比如把 {质量, 刚度} 的组合交互当成
      {质量, 温度} 来预测。
  张拳石团队的主张: 把模型推断分解成一组"交互基元(输入特征子集的联合贡献)", 再比较
  两套交互结构是否**等效(network-level interaction equivalence)** —— 结构失调即检出
  捷径/混喏, 而非等到数值失配才暴露。

  本模块是把这套思想落成**可运行的治理原语** (不假装是完整 DNN 交互理论, 是其中
  最基础的 order-1/order-2 博弈交互+等效度量):
    - :func:`shapley_values`                主效应 (order-1 Shapley): 单特征贡献
    - :func:`pair_interactions`             Shapley 交互指数 (order-2): 特征对是否
                                             联合超出相加 (协同+/拮抗-)
    - :func:`interaction_primitives`        把主效应+交互合成统一"交互基元"表
    - :func:`salient_interactions`          取显著交互(超过容差) —— 交互结构的代表
    - :func:`equivalent_interaction`        两模型的显著交互结构是否一致 (等效度量)
    - :func:`surrogate_law_alignment`       代理 vs 定律的对齐审计 (治理入口)
    - :func:`interaction_trace`             基元序列化为 grounding 证据

  数学公式 (order-2 Shapley 交互指数, n=|features|):
    varphi_i   = Σ_{S⊆N∖{i}}      |S|!(n-|S|-1)!/n! · (v(S∪{i}) - v(S))
  varphi_ij  = Σ_{S⊆N∖{i,j}}    |S|!(n-|S|-2)!/(n-1)! · Δ_ij(S)
  Δ_ij(S)     = v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)   (离散二阶导)
  值函数 v(S) 把 S 内特征设为 instance 值、其余设为 baseline, 再喂给 predict。
  对 order-2 Δ 恰好只保留"联合非加性"贡献 —— 纯加性模型交互为 0, 乘积项(如 c·x1·x2)
  才会被非零捕获 (这正是 shortcut 探测的关键)。

  诚实边界: 精确交互需枚举子集, 适用于特征数小的科学代理(典型 3~8 个物理描述符)。
  n 较大时联指数枚举不适合 — 调用方应先用特征选择压到小特征集, 或改用抽样估计。
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from itertools import combinations
from typing import Any

# 特征组合在主/交互键里的区分前缀 (order-1/order-2), 便于统一"交互基元"键空间.
MAIN = "1"
PAIR = "2"


# ── 值函数构造 ───────────────────────────────────────────────────────


def _make_value(predict: Callable[[dict[str, float]], float],
                features: list[str],
                instance: dict[str, float],
                baseline: dict[str, float]) -> Callable[[int], float]:
    """把 (predict, features, instance, baseline) 编译成位掩码值函数 v(mask).

    mask 的第 i 位=1 表示 feature[i] 用 instance 值, =0 用 baseline 值。
    返回 v(mask)→float。predict 必须接受 {feat: value} 的完整输入 dict。
    """
    n = len(features)
    cache: dict[int, float] = {}

    def _v(mask: int) -> float:
        if mask in cache:
            return cache[mask]
        inputs = {
            features[i]: (instance[features[i]] if (mask >> i) & 1 else baseline[features[i]])
            for i in range(n)
        }
        val = float(predict(inputs))
        cache[mask] = val
        return val

    return _v


def shapley_values(predict: Callable[[dict[str, float]], float],
                   features: list[str],
                   instance: dict[str, float],
                   baseline: dict[str, float] | None = None) -> dict[str, float]:
    """order-1 Shapley 主效应: 每个特征的独立贡献 (其余特征 baseline)."""
    base = baseline or {f: 0.0 for f in features}
    n = len(features)
    if n == 0:
        return {}
    v = _make_value(predict, features, instance, base)
    fact = math.factorial
    phi: dict[str, float] = {}
    for i in range(n):
        acc = 0.0
        others = [j for j in range(n) if j != i]
        for r in range(n):
            for s in combinations(others, r):
                s_plus = s + (i,)
                S_mask = 0
                Sp_mask = 0
                for idx in s:
                    S_mask |= 1 << idx
                Sp_mask = S_mask | (1 << i)
                w = fact(r) * fact(n - r - 1) / fact(n)
                acc += w * (v(Sp_mask) - v(S_mask))
        phi[features[i]] = round(acc, 6)
    return phi


def pair_interactions(predict: Callable[[dict[str, float]], float],
                      features: list[str],
                      instance: dict[str, float],
                      baseline: dict[str, float] | None = None) -> dict[tuple[str, str], float]:
    """order-2 Shapley 交互指数: (i,j) 联合是否贡献超出相加 (协同+/拮抗-)."""
    base = baseline or {f: 0.0 for f in features}
    n = len(features)
    if n < 2:
        return {}
    v = _make_value(predict, features, instance, base)
    fact = math.factorial
    out: dict[tuple[str, str], float] = {}
    for i, j in combinations(range(n), 2):
        acc = 0.0
        others = [k for k in range(n) if k not in (i, j)]
        for r in range(len(others) + 1):
            for s in combinations(others, r):
                S = 0
                for k in s:
                    S |= 1 << k
                Si = S | (1 << i)
                Sj = S | (1 << j)
                Sij = Si | (1 << j)
                delta = v(Sij) - v(Si) - v(Sj) + v(S)
                w = fact(r) * fact(n - r - 2) / fact(n - 1)
                acc += w * delta
        out[(features[i], features[j])] = round(acc, 6)
    return out


def interaction_primitives(predict: Callable[[dict[str, float]], float],
                           features: list[str],
                           instance: dict[str, float],
                           baseline: dict[str, float] | None = None) -> dict[tuple[str, ...], float]:
    """统一"交互基元"表: 主效应(("1", feat)) + 交互(("2", f, g)), 键空间互斥."""
    prim: dict[tuple[str, ...], float] = {}
    for f, phi in shapley_values(predict, features, instance, baseline).items():
        prim[(MAIN, f)] = phi
    for (f, g), phi in pair_interactions(predict, features, instance, baseline).items():
        prim[(PAIR, f, g)] = phi
    return prim


def salient_interactions(prim: dict[tuple[str, ...], float],
                         tol: float = 1e-3) -> set[tuple[str, ...]]:
    """显著交互结构: |phi|>tol 的基元集合 (交互结构的代表形状)."""
    return {k for k, v in prim.items() if abs(v) > tol}


def equivalent_interaction(a: dict[tuple[str, ...], float],
                           b: dict[tuple[str, ...], float],
                           *,
                           tol: float = 1e-3) -> dict[str, Any]:
    """等效交互度量 (张拳石"两网络推理结构一致")。

    判定两套交互基元是否"推理等效": 显著基元集合一致 = 等效; 否则报出仅一方有的
    基元 (通常是代理引入的虚假交互), 便于定位 shortcut。

    Returns: {"equivalent": bool, "agreement": float, "a_only": [...], "b_only": [...]}
    """
    sa, sb = salient_interactions(a, tol), salient_interactions(b, tol)
    common = sa & sb
    union = sa | sb
    agreement = (len(common) / len(union)) if union else 1.0
    return {
        "equivalent": sa == sb,
        "agreement": round(agreement, 4),
        "a_only": sorted(sa - sb, key=str),
        "b_only": sorted(sb - sa, key=str),
        "salient_a": sorted(sa, key=str),
        "salient_b": sorted(sb, key=str),
        "tol": tol,
    }


def surrogate_law_alignment(*, surrogate: Callable[[dict[str, float]], float],
                            law: Callable[[dict[str, float]], float],
                            features: list[str],
                            instance: dict[str, float],
                            baseline: dict[str, float] | None = None,
                            tol: float = 1e-3) -> dict[str, Any]:
    """治理入口: 代理 vs 定律 的**结构对齐审计**.

    只要交互结构不一致(即使数值吻合), 就如实报 not-aligned, 并给出代理多出来/少掉的
    显著交互 —— 这正是"代理在训练分布上数值匹配, 却抓到 shortcut/混淆交互"的早期探测。

    Returns: {"aligned": bool, **equivalent_interaction(...), "instance": instance}
    """
    pi_s = interaction_primitives(surrogate, features, instance, baseline)
    pi_l = interaction_primitives(law, features, instance, baseline)
    # equivalent_interaction(a=surrogate, b=law): a_only=surrogate 独有, b_only=定律独有.
    eq = equivalent_interaction(pi_s, pi_l, tol=tol)
    return {
        "aligned": eq["equivalent"],
        **eq,
        "surrogate_only": eq["a_only"],   # 代理多出来的交互(通常是 shortcut/混淆)
        "law_only": eq["b_only"],         # 定律有而代理缺失的交互(代理漏了真实机制)
        "instance": dict(instance),
    }


def interaction_trace(prim: dict[tuple[str, ...], float]) -> str:
    """把交互基元序列化成可证伪工件, 供并进研究 trace / grounding 门禁核实."""
    rows = [
        {"order": k[0], "features": list(k[1:]), "interaction": v}
        for k, v in sorted(prim.items(), key=lambda kv: (kv[0][0], str(kv[0])))
    ]
    return json.dumps({"type": "interaction_primitives", "rows": rows}, ensure_ascii=False)


def build_alignment_outcome(surrogate: Callable[[dict[str, float]], float],
                            law: Callable[[dict[str, float]], float],
                            features: list[str],
                            instance: dict[str, float],
                            baseline: dict[str, float] | None = None,
                            tol: float = 1e-3,
                            ) -> Callable[[list[dict], str], dict]:
    """装饰成 `run_research_program(structural_audit=...)` 可用的**结构闸门**.

    返回一个 ``(survivors, goal) -> dict`` 回调, 供深研管线在代理结论进报告/决策前自动
    调用; 内部跑 :func:`surrogate_law_alignment`(代理 vs 定律的交互等效审计)。管线会把这个
    结果的 pass/surrogate_only/law_only 并入 trace、写进报告, 并在未通过时醒目标注
    shortcut 风险。survivors 仅为满足管线调用签名, 审计对象由本闭包外注入的
    surrogate/law/instance 决定。
    """
    def _gate(survivors: list[dict], goal: str) -> dict:
        audit = surrogate_law_alignment(
            surrogate=surrogate, law=law, features=features, instance=instance,
            baseline=baseline, tol=tol)
        return {
            "pass": audit["aligned"],
            "reason": ("交互结构一致, 代理与定律无结构偏差。" if audit["aligned"]
                       else "代理存在与定律不符的交互结构(疑似 shortcut/混淆), 未通过结构闸门。"),
            "surrogate_only": audit["surrogate_only"],
            "law_only": audit["law_only"],
            "goal": goal,
        }

    return _gate