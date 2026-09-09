"""判断层 · 分级护栏 —— 模型无关的科学取省软提示器.

与「真伪层(硬门禁, claim_grounding)」严格分离:
  - 真伪层: 数值可证伪 → 任何模型都硬否决(幻觉/编造/引用不在轨迹).
  - 判断层: 科学取省(该不该补对照/量纲怎么读/要不要重复) → 只**软提示, 绝不否决**,
    且**按能力档位注水**. 档位决定注入多少提示, 与具体模型解耦 —— agent 不是为某个
    model 设计的, 而是"先信最强假设(档位0=默认零提示), 看到能力缺口才降级补提示".

档位语义 (strictness):
  0 (默认) : 判读层全关, 只保留真伪层硬门禁 —— 信任模型推理, CLI/聚合并无行为变化;
  1        : 最小提示 —— 仅针对 target 里显著出现的候选自变量, 软确认"是否做过对照",
             不替代模型判断;
  2        : 完整提示 —— 档位1 + 量纲/自定义指标语义复核 + 方向性结论建议重复注明.

关键不变量 (可证伪, 供测试):
  - 返回的提示**只追加**进成文 prompt, 不参与 verify(不构成硬门禁) → 不会锁死强模型;
  - strictness=0 恒返回 [] → 零行为变化(向后兼容锚);
  - 同一 (goal, trace, survivors) 下三档是**逐层包含**的: hints[2] ⊇ hints[1] ⊇ hints[0].
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

# 候选自变量词表(target 里的"该不该扫对照"候选)。用宽泛匹配, 软, 不硬判。
_VAR_TOKENS: list[tuple[str, str]] = [
    # (正则片段, 人类可读名)
    (r"(?:N|样本量|sample)", "样本量 N"),
    (r"eps|精度|threshold|截止", "精度/eps"),
    (r"约束|constraint|point.?value", "约束数量"),
    (r"(?:w|宽度|width)", "网络宽度 w"),
    (r"(?:steps|预算|budget|训练轮)", "优化器预算/steps"),
    (r"(?:seed|种子|重复|repeat|n_runs)", "随机重复(seeds)"),
]
# 自定义"过程量"怀疑词(出现即提示复核量纲语义, 软)。
_DIM_SUSPECT = re.compile(r"(?:log|ratio|conv|disc|score|index|effect|GAP|MSE|residual)", re.I)

# 不确定度信号: summary/objectives 里出现则视为"至少带一层重复/方差"。
_REPEAT_HINT = re.compile(r"(?:_k\d|k\s*[:=]\s*\d|sigmaH|variance|std|seed|repeat|runs|n_runs|\[.*,.*\])", re.I)


def _goal_vars(goal: str) -> list[str]:
    """从 target 里找出显著出现的候选自变量名(软候选, 供提示引用)."""
    names: list[str] = []
    for pat, name in _VAR_TOKENS:
        if re.search(pat, goal or "", re.I):
            names.append(name)
    return names


def _has_repeat_signal(trace: Iterable[str]) -> bool:
    """trace 里是否存在"重复/方差"信号 —— 可证伪: 全是单点 scalar 则 False."""
    for t in trace or []:
        if not isinstance(t, str):
            t = json.dumps(t, ensure_ascii=False)
        if _REPEAT_HINT.search(t):
            return True
    return False


def _has_dim_suspect(trace: Iterable[str]) -> bool:
    """trace 里是否存在待复核量纲的自定义过程量(软触发器)."""
    for t in trace or []:
        if not isinstance(t, str):
            t = json.dumps(t, ensure_ascii=False)
        if _DIM_SUSPECT.search(t):
            return True
    return False


def build_judgment_hints(
    goal: str,
    *,
    trace: Iterable[str] | None = None,
    survivors: Iterable[dict] | None = None,
    strictness: int = 0,
) -> list[str]:
    """生成判断层软提示 (模型无关, 只写进 prompt, 不参与硬门禁).

    strictness=0 → [] (默认信任模型推理, 零行为变化).

    Returns: 一段段人类可读的"自省提示", 供调用方追加进 LLM 成文 prompt.
    """
    strictness = int(strictness)
    if strictness <= 0:
        return []
    trace = list(trace or [])
    hints: list[str] = []

    # 档位 >=1: 候选自变量"对照确认"(软)。不硬判缺测, 只提醒模型: 若你断言该
    # 变量主导, 请说明是否做过它的扫描对照(而非只固定一个值)。
    if strictness >= 1:
        for name in _goal_vars(goal):
            hints.append(
                f"[判读-{name}] 若结论断言「{name}」主导饱和/刚性, 请说明你是否扫过 "
                f"{name} 的取值区间(而不仅是固定一个值); 若未扫, 把该结论标为「待 {name}-对照验证」。"
            )
        # 不确定度: 方向性结论若无可证伪的重复/方差信号, 提示注明或标初步。
        if trace and not _has_repeat_signal(trace):
            hints.append(
                "[判读-重复] 当前支持“方向性结论”的实验看不到重复/方差信号 "
                "(如 seeds/sigmaH/std/多值数组)。请对关键方向结论注明重复次数, "
                "或在无法重复时把结论明确标为「初步, 未量化不确定度」。"
            )

    # 档位 >=2: 量纲/自定义指标语义复核 + 结论-证据强度校准。
    if strictness >= 2:
        if _has_dim_suspect(trace):
            hints.append(
                "[判读-量纲] 轨迹里含自定义过程量(如 log/conv/disc/effect/GAP 等)。"
                "引用它们之前请说明其量纲语义与取值边界(如「该值大/小意味着什么」), "
                "不要把某个约定俗成之外的阈值当作客观异常判据。"
            )
        hints.append(
            "[判读-强度] 下结论前做强度校准: 单次测量支撑「相关性」, 多次重复/对照才支撑"
            "「因果性/主导性」。请确保结论强度不超过证据强度。"
        )
    return hints


def hint_block(
    goal: str,
    *,
    trace: Iterable[str] | None = None,
    survivors: Iterable[dict] | None = None,
    strictness: int = 0,
) -> str:
    """把提示列表渲染成一段可为 prompt 追加的文本(空则返回空串)."""
    hs = build_judgment_hints(goal, trace=trace, survivors=survivors, strictness=strictness)
    if not hs:
        return ""
    head = "【判断层·软提示】以下为科学取省层的自省提醒(非硬性要求, 不取代你的判断):\n"
    return head + "\n".join(f"- {h}" for h in hs)


# 兼容导入 (旧路径曾以模块名误引).
__all__ = ["build_judgment_hints", "hint_block"]