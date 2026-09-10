"""连续 Claim-Grounding 奖励 + Anti-Hacking 三件套.

把 `claim_grounding.py` 的**布尔门禁**升级为可喂 RL/偏好后训练的**连续奖励**，
并套上从 VibeWorlding-Gym 提炼的 anti-hacking 三件套，防止 agent 在训练中"刷分"：

  - 连续数字贴近度：`exp(-|ŷ-y|/(|y|+ε))`（Code-as-World GRPO 同款）或 MRA 式多档阈值。
  - grounding 溯源系数：主张数值能回溯到工具轨迹（含经真值推导）才给分，未落地即扣——这是
    把 claim_grounding 变成**稀疏但权威的 ORM 信号**，不是让 agent 背答案。
  - Anti-Hacking：
      1. strict-scope：越出"授权改动集合"→ 该轨奖励清零；
      2. 效率折扣：首次全对越早越好，按轮次打折；
      3. 空转惩罚：已解决还继续出数的轮次扣分。

设计原则（与 reward_design.md 双轨一致）：
  - 本模块是 **ORM（结果权威裁判）**，判"数值对不对"，不练能力；练能力走 PRM。
  - 全程可注入 `true_values`（ground-truth）也可只传 tool_trace 做无监督溯源分，
    使"离线 trace reward shaping"（reward_design §9 消融）能直接做。
  - 纯标准库，零网络/零 LLM/幂等，可独立单测。

关联实现：
  - `claim_grounding.verify_claims` 负责溯源判定（本模块直接复用其 verdict/数值分类）。
  - 折叠点：可把本模块输出并进现有 `reconcile_r_phys(base, world_reward)` 的 reward 侧。
"""
from __future__ import annotations

import importlib
import importlib.util
import math
from pathlib import Path
from typing import Any

# 直接复用 claim_grounding.verify_claims 作溯源判定。为防止 `huginn.validation`
# 包 __init__ 拉起 langchain 等重依赖 (纯离线/单测时可能未装), 这里用惰性绑定:
# 首次使用时才定位真实模块并读取 verify_claims, 失败则回退纯数值核对的兜底实现。
_verify_claims: Any = None


def _load_verify_claims():
    global _verify_claims
    if _verify_claims is not None:
        return _verify_claims
    try:
        from huginn.validation import claim_grounding as _cg
        _verify_claims = _cg.verify_claims
    except Exception:  # noqa: BLE001 — 包重依赖未装/初始化失败, 回退兜底
        # 兜底: 直接按文件加载 claim_grounding (纯标准库, 不触发包 __init__)。
        try:
            import pathlib
            _root = pathlib.Path(__file__).resolve().parent
            _spec = importlib.util.spec_from_file_location(
                "claim_grounding_embed", _root / "claim_grounding.py"
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _verify_claims = _mod.verify_claims
        except Exception:  # noqa: BLE001
            _verify_claims = _fallback_verify_claims
    return _verify_claims


def _fallback_verify_claims(text, trace, **_: Any) -> dict[str, Any]:
    """纯数值兜底: 报告数值能否近似匹配轨迹里任一数值. 仅在真实 claim_grounding
    import 失败时启用, 语义对齐(1/2位容差 + 小整数)以保持可证伪."""
    import re
    _tok = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", text or "")
    _ev = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", "\n".join(trace or []))
    ev_set = {round(float(x), 2) for x in _ev}
    matched, un = [], []
    for t in _tok:
        try:
            v = float(t)
        except ValueError:
            continue
        got = any(abs(round(v, 2) - e) <= 0.03 for e in ev_set)
        (matched if got else un).append(v)
    ok = not un
    return {"matched": matched, "derived": [], "unsubstantiated": un,
            "verdict": "pass" if ok else "needs_grounding",
            "note": "fallback grounding"}

# 数字贴近度的分母保护 (避免 y=0 除零); 与 Code-as-World 的 ε 同哲学.
_EPS = 1e-9
# MRA 式多档阈值: 相对误差小到满足某档记 1 分, 取均值 -> 连续而非 0/1 硬判.
_MRA_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def numeric_accuracy_reward(
    pred: float,
    truth: float,
    *,
    mode: str = "mra",
    eps: float = _EPS,
) -> float:
    """单数值的连续贴近度奖励, 在 [0, 1].

    mode:
      "mra"  -> MRA 式多档阈值均值 (鲁棒, 与 QuantiPhy/Code-as-World 社区可比).
      "exp"  -> exp(-|pred-truth|/max(|truth|, eps)) (Code-as-World GRPO reward 同款,
                相对误差越近 0 越接近 1, 误差源头敏感).
    用相对误差而非绝对误差: 不同物理量尺度差异巨大, 绝对误差不可跨量纲比较.
    """
    if not math.isfinite(pred) or not math.isfinite(truth):
        return 0.0
    denom = max(abs(truth), eps)
    rel = abs(pred - truth) / denom
    if truth == 0 and pred == 0:
        rel = 0.0
    if mode == "exp":
        return math.exp(-rel)
    # mra: 相对误差 < 1-threshold 记 1 分, 取均值
    return sum(1.0 for th in _MRA_THRESHOLDS if rel < (1.0 - th)) / len(_MRA_THRESHOLDS)


def grounding_source_reward(
    final_text: str,
    tool_trace: list[str],
    *,
    allow_derived: bool = True,
) -> dict[str, Any]:
    """基于溯源系数的连续接地奖励 (无 ground-truth 也能算).

    复用 claim_grounding: 把报告主张数值按"能否回溯到工具轨迹"给连续分。
      得分 = (matched + derived) / (matched + derived + unsubstantiated)。
    单独一个未落地主张 = 1/(n+1) 档, 多个未落地 => 分数显著压低 —— 这是对
    "背答案 / 脑内模拟数字" 的连续惩罚, 比布尔 needs_grounding 更平滑。

    Returns: {"grounding_score", "verdict", "matched", "derived", "unsubstantiated"}
    """
    res = _load_verify_claims()(
        final_text,
        tool_trace or [],
        include_all_numbers=False,
        allow_derived=allow_derived,
    )
    m = len(res["matched"])
    d = len(res["derived"])
    u = len(res["unsubstantiated"])
    n = m + d + u
    score = 1.0 if n == 0 else (m + d) / n
    return {
        "grounding_score": score,
        "verdict": res["verdict"],
        "matched": m,
        "derived": d,
        "unsubstantiated": u,
    }


def grounded_accuracy_reward(
    final_text: str,
    tool_trace: list[str],
    *,
    values: dict[float, float] | None = None,
    mode: str = "mra",
    grounding_weight: float = 0.5,
    allow_derived: bool = True,
) -> dict[str, Any]:
    """总奖励 = 溯源系数与数字贴近度的加权 —— "连续 claim_grounding reward" 主入口.

    - 若传入 `values` (主张数值 -> 真值映射, 来自工具基准/解析解), 对每个主张按
      numeric_accuracy_reward 给贴近分, 并在溯源过关的前提下加权, 得到即时 RL 信号。
    - 无 `values` 时退化为纯 grounding_source_reward (离线 accident 打分)。
    - grounding_weight 控制"必须溯源"的权重: 数字再贴近, 若未落地即打对折,
      惩罚"背答案" —— 这正对应 reward_design 对齐桥梁的反黑客项。

    Returns: {"reward", "grounding", "detail": {pred, truth, near, grounded, ok}[]}
    """
    g = grounding_source_reward(
        final_text, tool_trace, allow_derived=allow_derived
    )
    if not values:
        return {"reward": g["grounding_score"], "grounding": g["verdict"], "detail": []}

    # 用 verify_claims 的数值分类, 保证只对"真实主张"计价 (忽略轮次编号/小节号).
    # 注意: verify 返回的 matched/derived/unsubstantiated 已 round(3), 不能直接当键;
    # 用 extract_numeric_claims 的原始数值对账 values(它是主张数值->真值映射),
    # 再用 rounded 集合判断该主张是否溯源落地。
    # 取原始主张数值: 直接从真实 claim_grounding 模块读 extract_numeric_claims(惰性同源),
    # 避免裸 from ... import 触发包 __init__ 拉 langchain 重依赖。
    _spec = importlib.util.spec_from_file_location(
        "claim_grounding_cr", Path(__file__).resolve().parent / "claim_grounding.py"
    )
    _mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    res = _load_verify_claims()(final_text, tool_trace or [], allow_derived=allow_derived)
    raw_claims = _mod.extract_numeric_claims(final_text)
    rounded_matched = set(res["matched"]) or set()
    rounded_ok = set(res["matched"]) | set(res["derived"])
    claims: dict[float, bool] = {}
    for c in raw_claims:
        rc = round(c, 3)
        grounded = rc in rounded_ok or c in rounded_matched
        claims.setdefault(c, grounded)
    detail: list[dict[str, Any]] = []
    acc = 0.0
    n = 0
    for c, grounded in claims.items():
        truth = values.get(c)
        if truth is None:
            continue
        near = numeric_accuracy_reward(c, truth, mode=mode)
        # 溯源过关才全给分; 未落地打 (1-grounding_weight) 折, 惩罚背答案.
        weight = 1.0 if grounded else (1.0 - grounding_weight)
        acc += near * weight
        n += 1
        detail.append(
            {"pred": c, "truth": truth, "near": round(near, 4),
             "grounded": grounded, "ok": grounded}
        )
    accuracy = (acc / n) if n else 0.0
    # 最终奖励: 溯源系数与贴合精度各占一半 — 既不能靠刷溯源骗分, 也不能背数字逃过溯源.
    reward = 0.5 * g["grounding_score"] + 0.5 * accuracy
    return {"reward": reward, "grounding": g["verdict"], "detail": detail}


# ──────────────────── Anti-Hacking 三件套 ────────────────────


def strict_scope_reward(full_reward: float, authorized_ratio: float) -> float:
    """① strict-scope：越出授权改动集合直接清零.

    authorized_ratio: 本次改动落在"授权集合"内的比例 [0,1]。
    任何越界改动 (多删/多移/顺手加 —— VibeWorlding 识别到的典型 reward hacking)
    都把整轨奖励清零: `authorized_ratio == 1.0` 才全给, 否则 0。

    Returns: full_reward if authorized_ratio >= 1.0 - _EPS else 0.0
    """
    if authorized_ratio >= (1.0 - _EPS):
        return full_reward
    return 0.0


def efficiency_discount(reward: float, solved_at_turn: int, *, alpha: float = 0.3) -> float:
    """② 效率折扣：首次全对越早越好.

    把 reward 按"第一次全部正确所在的轮次"打折, 促进 agent 尽早收敛而非绕圈:
        reward * (1 - alpha * max(0, solved_at_turn - 1))
    首轮 (turn==1) 不打折; turn>=2 每多一轮扣 0.3。具体 alpha 可调。

    Returns: 折算后奖励 (>= 0)。
    """
    if solved_at_turn is None or solved_at_turn <= 1:
        return reward
    discount = max(0.0, 1.0 - alpha * (solved_at_turn - 1))
    return reward * discount


def idle_turn_penalty(reward: float, extra_turns: int, *, beta: float = 0.1) -> float:
    """③ 空转惩罚：已解决还继续出数的轮次扣分.

    extra_turns: 超过"达成即终止"的轮数 (0 = 未空转, 无扣分)。
    目的: 防止 agent 达成目标后仍反复调用工具刷字面 (VibeWorlding 的空转检测)。

    Returns: max(0, reward - beta * extra_turns)。
    """
    if extra_turns is None or extra_turns <= 0:
        return reward
    return max(0.0, reward - beta * (extra_turns if isinstance(extra_turns, int) else int(extra_turns)))


def anti_hacking_reward(
    base_reward: float,
    *,
    authorized_ratio: float = 1.0,
    solved_at_turn: int | None = None,
    extra_turns: int | None = None,
    alpha: float = 0.3,
    beta: float = 0.1,
) -> float:
    """Anti-Hacking 三件套的打包入口: 依序做 scope 清零 -> 效率折扣 -> 空转惩罚.

    顺序设计 (VibeWorlding 语义):
      1. 先做 strict-scope 硬关; 越界直接 0, 后面两项不再生效。
      2. 再效率折扣 (越早全对越高分)。
      3. 最后空转惩罚 (达成后多余轮数扣分)。
    """
    r = strict_scope_reward(base_reward, authorized_ratio)
    r = efficiency_discount(r, solved_at_turn, alpha=alpha)
    r = idle_turn_penalty(r, extra_turns, beta=beta)
    return r


def reconcile_r_phys(
    base: float,
    *,
    world_reward: float | None = None,
    world_weight: float = 0.5,
    authorized_ratio: float = 1.0,
) -> float:
    """与现有 `world_state.reconcile_r_phys` 同构的折叠点.

    把"世界预测命中奖励 (world_reward)"按权并入 base; 无 world_reward 时原样返回
    不退分 (与既有 workspace/ValidateTool 行为一致), 保证零回归兼容。
    """
    if world_reward is None:
        return base
    return (1.0 - world_weight) * base + world_weight * world_reward


__all__ = [
    "numeric_accuracy_reward",
    "grounding_source_reward",
    "grounded_accuracy_reward",
    "strict_scope_reward",
    "efficiency_discount",
    "idle_turn_penalty",
    "anti_hacking_reward",
    "reconcile_r_phys",
    "MRA_THRESHOLDS",
]