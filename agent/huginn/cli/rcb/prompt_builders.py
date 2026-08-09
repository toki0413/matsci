"""rcb_runner 拆分: 旧 hint 拼接 prompt builder.

抽自 rcb_runner.py L426-471, 单一职责 = iter 0 / iter>0 prompt 旧拼法.
生产 else 分支 (HUGINN_HINT_COORDINATOR=0 兜底) 仍调用 _legacy_build_step2_prompt,
_legacy_build_iter_prompt 仅被 self-check 测试调用. 因生产路径仍可达, 暂不删除.

ponytail: 不引新依赖, 不改逻辑, 纯 import 抽取. 测试靠 rcb_runner self-check 路径.
"""
from __future__ import annotations

import warnings


def _legacy_build_step2_prompt(step2_prompt_base: str, scan_hint: str, fcm_hint: str) -> str:
    """iter 0 prompt 旧拼法 — base + scan_hint + fcm_hint 直接 concat."""
    # [legacy-debt] 旧 hint 拼接路径, 用 HintCoordinator 替代. 见模块顶部注释.
    warnings.warn(
        "_legacy_build_step2_prompt 是旧 hint 拼接路径, 已被 HintCoordinator 替代; "
        "后续版本将移除, 请改用 HintCoordinator.coordinate(iter_n=0, ...).",
        DeprecationWarning,
        stacklevel=2,
    )
    return step2_prompt_base + scan_hint + fcm_hint


def _legacy_build_iter_prompt(
    iter_prompt_base: str,
    compass: str | None,
    fcm_winner_reminder: str | None,
    kb_chunks_text: str | None,
    merge_hint: str,
    imagination_block: str | None,
    ctx_inject: str | None,
) -> str:
    """iter>0 prompt 旧拼法 — 按原 += 顺序 concat 各 hint 块.

    顺序: base → compass → fcm_winner → kb_chunks → merge_hint → imagination → ctx_inject.
    ponytail: 顺序固化, 不做动态优先级 — 跟原 rcb_runner 行为一致, 跑分对照才公平.
    """
    # [legacy-debt] 旧 hint 拼接路径, 用 HintCoordinator 替代. 见模块顶部注释.
    warnings.warn(
        "_legacy_build_iter_prompt 是旧 hint 拼接路径, 已被 HintCoordinator 替代; "
        "后续版本将移除, 请改用 HintCoordinator.coordinate(iter_n>0, ...).",
        DeprecationWarning,
        stacklevel=2,
    )
    p = iter_prompt_base
    if compass:
        p += "\n\n" + compass
    if fcm_winner_reminder:
        p += fcm_winner_reminder
    if kb_chunks_text:
        p += kb_chunks_text
    p += merge_hint
    if imagination_block:
        p += "\n\n" + imagination_block
    if ctx_inject:
        p += "\n\n" + ctx_inject
    return p
