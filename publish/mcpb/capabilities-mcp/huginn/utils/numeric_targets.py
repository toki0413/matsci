"""从自由文本抽取 "metric = value" 目标值 (白名单 + 量级过滤).

DRY: ``agent.hint_coordinator`` 与 ``cli.rcb_utils`` 曾各复制一份完全相同的
``_METRIC_WHITELIST`` / ``_NUMERIC_PAIR_RE`` / ``extract_numeric_targets``.
两边都是 "读 LLM 文本里的量化目标" 的同一事实来源, 任一处的白名单调整都会让
hint 打分与 RCB 判分对同一段文本给出不同结论.
"""
from __future__ import annotations

import re

METRIC_WHITELIST = frozenset({
    "mae", "rmse", "mse", "r2", "r²", "r3", "accuracy",
    "precision", "recall", "f1", "auc", "pearson", "spearman",
    "loss", "error", "score", "bias",
})

# metric = value / metric: value / metric of value / metric ≈ value
NUMERIC_PAIR_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_²³]{1,15})\s*(?:=|:|of|≈|~|is)\s*"
    r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
)


def extract_numeric_targets(text: str) -> dict[str, float]:
    """从文本抓 'metric = value' 模式, 返回 {metric: value}.

    白名单 + 数值量级过滤, 避免误抓年份/版本号. 无文本返回空 dict.
    """
    if not text:
        return {}
    targets: dict[str, float] = {}
    for m in NUMERIC_PAIR_RE.finditer(text):
        name = m.group(1).lower()
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        if name not in METRIC_WHITELIST:
            continue
        if abs(val) > 1e6:
            continue
        targets[name] = val
    return targets
