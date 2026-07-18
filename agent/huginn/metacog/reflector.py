"""Reflector — 工具调用异常时的反思介入 (G69).

should_continue 检测到连续工具调用异常后, 把 ToolCallHealth 喂给 reflect(),
拿到 ReflectorAction 列表, format_reflector_text 注入到下一轮 prompt 让 agent 自检
配置/参数/换工具. 规则按严重度从高到低, 首个匹配即返回.

ponytail: 规则启发式, 不做根因 LLM 推理. 升级路径: 把 audit_log + last_step_evaluations
喂给 LLM 做模式分析, 输出更精细的 action_type + details. last_step_evaluations 和
audit_log_path 当前仅占位, 升级时喂给 LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ToolCallHealth 在 step_evaluator.py 里定义, 这里 duck typing 兼容, 不硬 import.
# ponytail: 不绑死类型, dataclass / dict 都能取值. 升级路径: 真 import 做类型检查.

_VALID_ACTION_TYPES = frozenset({
    "check_config", "check_params", "check_model",
    "switch_tool", "switch_model", "adjust_params",
})


@dataclass
class ReflectorAction:
    """反思动作."""

    action_type: str  # check_config / check_params / check_model / switch_tool / switch_model / adjust_params
    description: str
    details: dict = field(default_factory=dict)


def _health_attr(health: Any, name: str, default: Any = None) -> Any:
    """ToolCallHealth 兼容取值: dataclass / dict 都能取."""
    if isinstance(health, dict):
        return health.get(name, default)
    return getattr(health, name, default)


def reflect(
    tool_call_health: Any,
    last_step_evaluations: list | None = None,
    audit_log_path: Path | None = None,
) -> list[ReflectorAction]:
    """根据工具调用健康度产出反思动作.

    规则 (按严重度从高到低, 首个匹配即返回):
    - 全部异常 (success_rate==0 且 total_calls>0) → switch_tool + switch_model
    - success_rate < 0.5 → check_config + check_params
    - param_error_count > retry_count → check_params (LLM 生成参数格式错)
    - timeout_count > 2 → check_config (endpoint/超时设置)
    - 无异常 → []

    last_step_evaluations / audit_log_path 留给升级路径 (LLM 分析模式), 当前版本不用.

    ponytail: 规则启发式, 不做根因 LLM 推理. 升级路径: LLM 分析 audit_log 模式
    + last_step_evaluations 的 attempted/found 文本, 输出更精细的 details.
    """
    if tool_call_health is None:
        return []

    success_rate = float(_health_attr(tool_call_health, "success_rate", 1.0))
    total_calls = int(_health_attr(tool_call_health, "total_calls", 0))
    retry_count = int(_health_attr(tool_call_health, "retry_count", 0))
    timeout_count = int(_health_attr(tool_call_health, "timeout_count", 0))
    param_error_count = int(_health_attr(tool_call_health, "param_error_count", 0))

    # 1) 全部异常 → 换工具/换模型. success_rate==0 说明一次都没成, 工具或模型本身有问题
    if total_calls > 0 and success_rate == 0.0:
        return [
            ReflectorAction(
                action_type="switch_tool",
                description="所有工具调用均失败, 当前工具可能不适用, 建议换工具",
                details={"total_calls": total_calls, "success_rate": success_rate},
            ),
            ReflectorAction(
                action_type="switch_model",
                description="全失败也可能是底层模型问题, 建议换模型/endpoint 再试",
                details={"total_calls": total_calls},
            ),
        ]

    # 2) success_rate < 0.5 → 查配置 + 查参数. 一半都不到, 配置和参数格式都可能有坑
    if success_rate < 0.5:
        return [
            ReflectorAction(
                action_type="check_config",
                description="工具调用成功率偏低, 检查 endpoint/认证/超时等配置",
                details={"success_rate": success_rate, "total_calls": total_calls},
            ),
            ReflectorAction(
                action_type="check_params",
                description="成功率偏低也可能是参数格式问题, 检查最近调用的参数 schema",
                details={
                    "success_rate": success_rate,
                    "param_error_count": param_error_count,
                },
            ),
        ]

    # 3) 参数错误多于重试 → LLM 生成参数格式不对, 重点查 params
    if param_error_count > retry_count and param_error_count > 0:
        return [
            ReflectorAction(
                action_type="check_params",
                description="参数错误次数超过重试次数, 疑似 LLM 生成的参数格式不对",
                details={
                    "param_error_count": param_error_count,
                    "retry_count": retry_count,
                },
            ),
        ]

    # 4) 超时多 → 查配置 (endpoint 可用性 / 超时阈值)
    if timeout_count > 2:
        return [
            ReflectorAction(
                action_type="check_config",
                description="超时次数偏多, 检查 endpoint 可用性 / 超时阈值设置",
                details={"timeout_count": timeout_count},
            ),
        ]

    return []


def format_reflector_text(actions: list[ReflectorAction]) -> str:
    """格式化反思动作为 context 注入文本. 空 list 返回空串."""
    if not actions:
        return ""
    lines = ["[Reflector] 工具调用异常反思, 请对照自检:"]
    for i, a in enumerate(actions, 1):
        lines.append(f"{i}. [{a.action_type}] {a.description}")
        if a.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in a.details.items())
            lines.append(f"   详情: {detail_str}")
    return "\n".join(lines)


# === 自检 ===

if __name__ == "__main__":
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class _Health:
        # 和 step_evaluator.ToolCallHealth 同构, 不硬 import 避免路径问题
        success_rate: float = 1.0
        total_calls: int = 0
        retry_count: int = 0
        timeout_count: int = 0
        param_error_count: int = 0

    # 1) 正常健康度 → 空 list
    healthy = _Health(success_rate=1.0, total_calls=10)
    actions = reflect(healthy)
    assert actions == [], f"healthy → [], got {actions!r}"

    # 2) success_rate < 0.5 → check_config + check_params
    low_sr = _Health(success_rate=0.3, total_calls=10)
    actions = reflect(low_sr)
    assert len(actions) == 2, f"low sr → 2 actions, got {len(actions)}"
    types = {a.action_type for a in actions}
    assert types == {"check_config", "check_params"}, f"low sr types: {types}"

    # 3) 参数错误多 (success_rate >= 0.5) → 只返回 check_params
    param_err = _Health(
        success_rate=0.6, total_calls=10, param_error_count=3, retry_count=1)
    actions = reflect(param_err)
    assert len(actions) == 1, f"param err → 1 action, got {len(actions)}"
    assert actions[0].action_type == "check_params", \
        f"param err → check_params, got {actions[0].action_type}"

    # 4) 全异常 (success_rate==0) → switch_tool + switch_model
    all_fail = _Health(success_rate=0.0, total_calls=5)
    actions = reflect(all_fail)
    types = {a.action_type for a in actions}
    assert "switch_tool" in types, f"all fail → switch_tool, got {types}"
    assert "switch_model" in types, f"all fail → switch_model, got {types}"

    # 5) 超时多 (success_rate 正常) → check_config
    timeout_h = _Health(success_rate=0.8, total_calls=10, timeout_count=3)
    actions = reflect(timeout_h)
    assert len(actions) == 1, f"timeout → 1 action, got {len(actions)}"
    assert actions[0].action_type == "check_config", \
        f"timeout → check_config, got {actions[0].action_type}"

    # 6) None / 空 → []
    assert reflect(None) == [], "None → []"

    # 7) dict 形式也兼容 (duck typing)
    dict_h = {"success_rate": 0.2, "total_calls": 5}
    actions = reflect(dict_h)
    assert len(actions) == 2, f"dict health → 2 actions, got {len(actions)}"
    assert {a.action_type for a in actions} == {"check_config", "check_params"}

    # 8) format_reflector_text
    text = format_reflector_text(reflect(all_fail))
    assert "[Reflector]" in text, f"format missing header: {text!r}"
    assert "switch_tool" in text, f"format missing action: {text!r}"
    assert format_reflector_text([]) == "", "empty → ''"

    # 9) ReflectorAction 字段完整性
    a = actions[0]
    assert hasattr(a, "action_type") and hasattr(a, "description") and hasattr(a, "details")
    assert isinstance(a.details, dict)

    # 10) last_step_evaluations / audit_log_path 不影响当前规则 (升级路径用)
    actions_with_evals = reflect(healthy, last_step_evaluations=[1, 2, 3])
    assert actions_with_evals == [], "healthy + evals → [] (evals 暂未用)"
    actions_with_log = reflect(healthy, audit_log_path=Path("/tmp/audit.jsonl"))
    assert actions_with_log == [], "healthy + audit_log → [] (audit_log 暂未用)"

    print("all self-checks passed")
