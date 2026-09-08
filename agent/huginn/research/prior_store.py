"""A2 · 跨 run 稳定度先验 (cross-run settlement prior) —— 把一次 run 的分数高原
沉淀为可传递、可注入的预算先验.

背景: 同类目标域反复研究时, "这个领域通常在第几层进入分数高原"是**可复用**的调度
经验. 本模块把一次 run 的稳定度观察提取成纯 dict(extract_prior), 再由下一次 run
以保守方式注入(resolve_early_stop_args) —— 但它只是**预算参数先验**, 绝不参与、
更不替代任何真实实验.

A4(goal 归一化匹配): 先验携带 goal_slug(与 program._slug_goal 同语义), 注入时
若当前 goal 的 slug 与先验不一致 → 视为异域先验, 拒绝套用(行为=无先验) ——
防止把 A 领域的"高原层数"错套到 B 领域.

A5(先验时间衰减): prior["age"](距上次 run 的轮数)按半衰期=1 的指数衰减
(weight = 0.5 ** age) —— 越旧先验权重越低, 领域漂移后旧经验自然淡出.

A6(goal 模糊匹配): 超越 slug 字符串等价, 按 TF-IDF 余弦相似度(平滑 IDF, 纯
Python 确定性)分档:
  - exact(同 slug / 同文本) → 相似度 1.0, 全量注入;
  - related(相似度 >= 0.5, 领域语义聚类) → 相似度比例注入(半权起步);
  - foreign(相似度 < 0.5) → 拒绝套用(等同 A4 异域拒绝, 文本也可证伪);
  - unknown(先验无 goal 信息) → 向后兼容: 不惩罚, 只受 A5 衰减影响.

诚实红线(不可逾越):
  1. extract_prior 只读 out.consolidated(聚合头唯一出口), 不新增 out.* 字段;
  2. 先验只能把早停参数推向**更保守**方向(min_layers 单调不减), 永不因"上次稳定"
     而激进提前终止 —— 领域可能漂移, 先验不为历史背书;
  3. 异域先验(goal_slug 不匹配)一律拒绝套用 —— 先验只对**同域**生效;
  4. applicable=False / 无 slug 的字典可安全注入(等价无先验/未验证), 不抛错.
"""
from __future__ import annotations

import math
import re
from typing import Any


def goal_slug(goal: str, limit: int = 40) -> str:
    """goal 归一化 slug (与 program._slug_goal 同语义, 保证跨模块匹配口径一致).

    小写 + 非字母/数字/下划线 → "_", 去首尾 "_", 截断上限 —— "a4 match" → "a4_match".
    """
    s = re.sub(r"[^0-9a-z_]+", "_", (goal or "").lower()).strip("_")
    return s[:limit] or "final"


def extract_prior(out: Any) -> dict[str, Any]:
    """从 ResearchOutcome 提取一次 run 的稳定度先验 (纯函数, 无副作用).

    只消费 out.consolidated["early_stop"] —— 未启用早停/无观测 → applicable=False.
    返回说明:
      - applicable: 是否有可复用的高原观察(early_stopped 才为 True);
      - plateau: {layer_index, top, score, relative_change} 首次判稳的那一层;
      - goal: 目标原文; goal_slug: 归一化 slug(A4 跨 run 匹配凭据, 与 program 同口径).
    """
    con = getattr(out, "consolidated", None) or {}
    es = con.get("early_stop") or {}
    st = es.get("stability") or {}
    stopped = es.get("verdict") == "early_stopped"
    layer_index = es.get("stopped_after_layer")
    _goal = ((getattr(out, "plan_summary", None) or {}).get("goal")
             or (getattr(out, "harness", None) or {}).get("goal") or "")
    _base = {
        "source": "layered_settlement",
        "goal": _goal,
        "goal_slug": goal_slug(_goal),
    }
    if not stopped or layer_index is None or not st:
        _base.update({"applicable": False, "reason": "no_early_stopped"})
        return _base
    _base.update({
        "applicable": True,
        "plateau": {
            "layer_index": int(layer_index),
            "top": st.get("top"),
            "score": st.get("score"),
            "relative_change": st.get("relative_change"),
        },
    })
    return _base


# 轻量英文功能词停用表(冠词/介词/连词/代词等): 只剔纯语法词, 不碰领域实义词.
# 让"同一领域不同措辞"的共享实义词主导相似度(双文档语料下泛词 IDF 会失真,
# 不剔除反而抬高独有功能词权重, 稀释主题信号).
_STOPWORDS = frozenset(
    "a an the and or of for to in on with at by from into about as be by over under "
    "this that these those it its is are was were been being have has had do does did "
    "we you they them their our your my me him her he she i will would can could "
    "should may might must not no nor but if then than so such more most".split()
)


def _tfidf_cosine(a: str, b: str) -> float:
    """TF-IDF 余弦相似度(两文档语料, 平滑 IDF, 纯 Python 确定性).

    与 Jaccard 的区别: ① 剔除功能词停用表(and/of/the...), 共享实义词主导相似度;
    ② 词频加权 —— 重复出现的领域词贡献更高; ③ 连续值域(0..1), 而非离散重叠比.
    实现: 平滑 IDF = ln((N+1)/(df+1)) + 1 (N=2, scikit-learn 同款公式),
    词频 × IDF 后 L2 归一化求余弦 —— 同输入必同输出, 无外部依赖.
    """
    ta = [w for w in re.findall(r"[0-9a-z]+", (a or "").lower()) if w not in _STOPWORDS]
    tb = [w for w in re.findall(r"[0-9a-z]+", (b or "").lower()) if w not in _STOPWORDS]
    if not ta or not tb:
        return 0.0
    vocab = set(ta) | set(tb)
    df = {w: (w in ta) + (w in tb) for w in vocab}
    idf = {w: math.log(3.0 / (d + 1.0)) + 1.0 for w, d in df.items()}

    def _vec(tokens: list[str]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for w in tokens:
            tf[w] = tf.get(w, 0) + 1
        return {w: tf.get(w, 0) * idf[w] for w in vocab}

    va, vb = _vec(ta), _vec(tb)
    dot = sum(va[w] * vb[w] for w in vocab)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def goal_match_level(
    prior_goal: str,
    current_goal: str,
    prior_slug: str | None,
    current_slug: str | None,
) -> dict[str, Any]:
    """goal 匹配档位(纯函数, 确定性). A6: 超越 slug 等价, 支持领域语义聚类.

    返回 {"level", "overlap"}:
      - exact:   slug 双方存在且相等(或全文等价) → overlap=1.0;
      - related: TF-IDF 余弦相似度 >= 0.5(领域聚类) → overlap=该相似度;
      - foreign: TF-IDF 余弦相似度 < 0.5(异域) → overlap=该相似度;
      - unknown: 先验不含 goal 信息(无法判定) → overlap=None.
    """
    prior_goal = (prior_goal or "").strip()
    current_goal = (current_goal or "").strip()
    if prior_slug and current_slug and str(prior_slug) == str(current_slug):
        return {"level": "exact", "overlap": 1.0}
    if not prior_goal or not current_goal:
        return {"level": "unknown", "overlap": None}
    sim = _tfidf_cosine(prior_goal, current_goal)
    if sim >= 0.5:
        return {"level": "related", "overlap": sim}
    return {"level": "foreign", "overlap": sim}


def resolve_early_stop_args(
    prior: dict | None,
    *,
    default_min_layers: int = 2,
    default_margin: float = 0.02,
    current_goal_slug: str | None = None,
    current_goal: str | None = None,
) -> dict[str, Any]:
    """把先验映射为早停参数(纯函数, 确定性). A4: 异域先验拒绝套用.
    A5: 时间衰减(半衰期=1). A6: goal 模糊匹配分档注入.

    规则(全部可证伪):
      - prior 为空/不可用 → 默认参数, note="no_prior", goal_matched=None;
      - goal 判定为 foreign(异域, slug 与 TF-IDF 文本相似度均可证伪) → **拒绝套用**:
        返回默认参数, note="goal_mismatch", goal_matched=False(绝不张冠李戴);
      - 其余(exact/related/unknown 且 applicable) → min_layers =
        default + round(delta * weight), 钳制在 [default_min_layers, 4];
        delta = max(0, clamp(plateau.layer_index+1, 4) - default) —— 上次 N 层
        才稳定, 这次至少等 N 层才允许查稳定(**更保守**, 防领域漂移误停);
        weight = decay(0.5 ** age) × match(1.0 exact/unknown, sim related);
      - unknown(先验无 goal 信息) → 不因匹配惩罚, 只受 A5 衰减影响(向后兼容).
    """
    no_prior = not prior or not prior.get("applicable")
    prior_goal = (prior or {}).get("goal", "")
    prior_slug = (prior or {}).get("goal_slug")
    # A6: 匹配档位(exact/related/foreign/unknown)
    _match = goal_match_level(prior_goal, current_goal or "",
                              prior_slug, current_goal_slug)
    goal_level = _match["level"]
    goal_overlap = _match["overlap"]
    goal_matched = (goal_level in ("exact", "related"))
    # A5: 时间衰减(半衰期=1)
    prior_age = float((prior or {}).get("age", 0.0) or 0.0)
    decay_weight = 0.5 ** prior_age
    # 综合权重: 匹配档位权重(exact/unknown=1.0, related=overlap) × 时间衰减
    match_weight = 1.0 if goal_level in ("exact", "unknown") else goal_overlap
    weight = (decay_weight * match_weight) if match_weight is not None else decay_weight

    if no_prior:
        return {"min_layers": default_min_layers, "margin": default_margin,
                "source": "default", "note": "no_prior",
                "goal_matched": None, "goal_level": goal_level,
                "goal_overlap": goal_overlap, "decay_weight": decay_weight,
                "prior_age": prior_age, "weight": weight}
    if goal_level == "foreign":
        return {"min_layers": default_min_layers, "margin": default_margin,
                "source": "rejected", "note": "goal_mismatch", "goal_matched": False,
                "goal_level": goal_level, "goal_overlap": goal_overlap,
                "decay_weight": decay_weight, "prior_age": prior_age, "weight": weight}
    plateau = prior.get("plateau") or {}
    idx = plateau.get("layer_index")
    if idx is None:
        return {"min_layers": default_min_layers, "margin": default_margin,
                "source": "default", "note": "prior_without_plateau",
                "goal_matched": goal_matched, "goal_level": goal_level,
                "goal_overlap": goal_overlap, "decay_weight": decay_weight,
                "prior_age": prior_age, "weight": weight}
    candidate = min(int(idx) + 1, 4)                      # 钳制上限 4
    delta = max(0, candidate - default_min_layers)
    adjusted = default_min_layers + round(delta * weight)  # 权重合成的增量
    min_layers = max(default_min_layers, adjusted)         # 诚实红线: 永不低于默认
    return {"min_layers": min_layers, "margin": default_margin,
            "source": "cross_run_prior", "note": f"plateau_layer={idx}",
            "goal_matched": goal_matched, "goal_level": goal_level,
            "goal_overlap": goal_overlap, "decay_weight": decay_weight,
            "prior_age": prior_age, "weight": weight}