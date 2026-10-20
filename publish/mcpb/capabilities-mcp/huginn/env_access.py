"""HUGINN_* 环境变量的类型化访问器 (配置面唯一受认可的读端).

配置面的**声明**在 :mod:`huginn.env_schema` (``SCHEMA``, 单一权威源); 本模块是
**读端**: 新代码读 ``HUGINN_*`` 一律经此, 不再裸 ``os.environ.get``. 收益有三:

- **未声明即报错**: 读一个不在 ``SCHEMA`` 里的变量抛 :class:`UndeclaredEnvVarError`,
  迫使新增配置先登记 (拦截配置面静默膨胀);
- **默认值单一来源**: 默认值只来自 ``SCHEMA``, 不在每个读点重复 (消除双真值);
- **类型统一**: int/float/bool/json 解析集中一处, 解析失败 fail-open 回落默认值;
- **值域约束**: 声明里的 ``choices`` / ``range`` 在此收口校验, 环境取值违反约束时
  告警并回落权威默认值 (把"非法值静默回落"变成"显式回落"), 见
  :func:`constraint_violation`.

用法::

    from huginn.env_access import env_bool, env_int, env_str

    timeout = env_int(name)            # 默认值取自 SCHEMA
    debug = env_bool(name, default=False)   # 显式覆盖默认值

``default`` 省略时用 ``SCHEMA`` 里的权威默认值; 显式传入则覆盖之 (临时用, 不应在
多处重复同一默认值 —— 那正是本模块要消灭的).
"""

from __future__ import annotations

import json as _json
import logging
import os
from typing import Any

from huginn.env_schema import SCHEMA, EnvSpec

logger = logging.getLogger(__name__)

_UNSET = object()

_TRUE = ("true", "1", "yes", "on")
_FALSE = ("false", "0", "no", "off", "")


class UndeclaredEnvVarError(KeyError):
    """读取了未在 :data:`huginn.env_schema.SCHEMA` 登记的 ``HUGINN_*`` 变量.

    这是配置面治理的硬闸: 新增环境变量必须先在 ``env_schema.SCHEMA`` 登记,
    再经本模块读取. 抛 KeyError 子类以便调用方按需捕获.
    """


class EnvConstraintError(ValueError):
    """环境变量取值违反声明约束 (``choices`` / ``range``).

    读端默认不抛此错 (fail-open: 告警 + 回落权威默认值, 与解析失败一致); 本异常
    供审计/测试经 :func:`check` 做**严格校验**时显式抛出.
    """


def constraint_violation(spec_obj: EnvSpec, value: Any) -> str | None:
    """按声明校验 ``value`` 是否落在值域约束内; 违反返回原因, 通过返回 ``None``.

    - ``choices``: str 型按大小写不敏感比较 (读点多将取值 lower 化); 数值型按
      声明类型解析后比较.
    - ``range``: 闭区间 ``(lo, hi)``; 任一端点为 ``None`` 表示该侧开放; 仅对
      int/float 型有意义 (非数值取值报违反).
    无约束声明恒返回 ``None``.
    """
    if spec_obj.choices is not None:
        if spec_obj.type == "str":
            norm = str(value).strip().lower()
            allowed = {c.strip().lower() for c in spec_obj.choices}
            if norm not in allowed:
                return f"取值 {value!r} 不在允许集合 {spec_obj.choices}"
        else:
            matched = False
            for c in spec_obj.choices:
                try:
                    if parse_value(spec_obj.type, c) == value:
                        matched = True
                        break
                except (ValueError, TypeError):
                    # 声明里的候选值无法按声明类型解析 —— 属声明不自洽, 由治理测试显形.
                    continue
            if not matched:
                return f"取值 {value!r} 不在允许集合 {spec_obj.choices}"
    if spec_obj.range is not None:
        lo, hi = spec_obj.range
        try:
            if lo is not None and value < lo:
                return f"取值 {value!r} 低于下界 {lo}"
            if hi is not None and value > hi:
                return f"取值 {value!r} 高于上界 {hi}"
        except TypeError:
            return f"取值 {value!r} 非数值, 无法做区间校验"
    return None


def check(name: str, value: Any) -> str | None:
    """按 :data:`huginn.env_schema.SCHEMA` 校验变量取值; 未登记抛
    :class:`UndeclaredEnvVarError`. 返回违反原因或 ``None`` (供审计/测试)."""
    return constraint_violation(spec(name), value)


def validate(name: str, value: Any) -> Any:
    """严格校验: 违反约束抛 :class:`EnvConstraintError`, 通过返回 ``value``.

    读端 (:func:`env_str` 等) 走 fail-open 回落默认值; 需要在**写端/审计**处
    显式拒绝非法取值时用本函数 (SPEC 缺口② 的"显式报错"路径).
    """
    reason = check(name, value)
    if reason is not None:
        raise EnvConstraintError(f"{name}: {reason}")
    return value


def spec(name: str) -> EnvSpec:
    """取变量的权威声明; 未登记抛 :class:`UndeclaredEnvVarError`."""
    try:
        return SCHEMA[name]
    except KeyError:
        raise UndeclaredEnvVarError(
            f"{name!r} 未在 huginn.env_schema.SCHEMA 登记; "
            f"新增配置请先登记再经 env_access 读取"
        ) from None


def is_declared(name: str) -> bool:
    """变量是否已在 ``SCHEMA`` 登记 (不抛错, 供工具/审计查询)."""
    return name in SCHEMA


def env_is_set(name: str) -> bool:
    """变量是否在环境里显式设置且非空 (用于区分"未设 → 用默认"与"显式设值").

    ``env_bool`` 等访问器只看最终布尔结果, 无法区分"显式设为 false"与"未设(回落
    默认)". 需要按"是否设置"分支时用本函数. 未在 ``SCHEMA`` 登记仍抛
    :class:`UndeclaredEnvVarError`.
    """
    spec(name)
    raw = os.environ.get(name)
    return raw is not None and str(raw).strip() != ""


def parse_value(typ: str, raw: Any) -> Any:
    """按声明类型解析字面量; 供访问器与治理测试共用同一解析口径."""
    if typ == "bool":
        return _as_bool(raw)
    if typ == "int":
        return int(str(raw).strip())
    if typ == "float":
        return float(str(raw).strip())
    if typ == "json":
        return _json.loads(str(raw))
    return str(raw)


def _as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return False  # 无法识别: 保守当关 (与 FeatureFlags._parse_env_value 一致)


def _raw(name: str, default: Any) -> tuple[EnvSpec, Any]:
    """解析声明 + effective default; 返回 (spec, default)."""
    s = spec(name)
    d = s.default if default is _UNSET else default
    return s, d


def _read(name: str, default: Any) -> tuple[EnvSpec, Any, bool]:
    """返回 (spec, raw_or_default, is_set)."""
    s, d = _raw(name, default)
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return s, d, False
    return s, raw, True


def _enforce(s: EnvSpec, name: str, value: Any, is_set: bool) -> Any:
    """值域约束收口: 仅校验**来自环境**的取值; 违反则告警并回落权威默认值.

    未设置 (用声明默认值) 或该变量无约束声明时不介入 —— 默认值自身满足约束由
    ``tests/test_env_surface_governance.py`` 硬判, 读端不重复校验.
    """
    if not is_set:
        return value
    reason = constraint_violation(s, value)
    if reason is None:
        return value
    logger.warning(
        "env %s=%r 违反约束 (%s), 回落权威默认 %r", name, value, reason, s.default
    )
    return parse_value(s.type, s.default)


def env_str(name: str, default: Any = _UNSET) -> str:
    """读字符串; 未设或空 → 默认值 (字符串化); 违反 choices 约束 → 回落默认值."""
    s, raw, is_set = _read(name, default)
    value = "" if raw is None else str(raw)
    return _enforce(s, name, value, is_set)


def env_int(name: str, default: Any = _UNSET) -> int:
    """读整数; 未设/解析失败 → 默认值 (再失败 → 0); 违反 range 约束 → 回落默认值."""
    s, raw, is_set = _read(name, default)
    return _enforce(s, name, _coerce(s, name, raw, int, 0), is_set)


def env_float(name: str, default: Any = _UNSET) -> float:
    """读浮点; 未设/解析失败 → 默认值 (再失败 → 0.0); 违反 range 约束 → 回落默认值."""
    s, raw, is_set = _read(name, default)
    return _enforce(s, name, _coerce(s, name, raw, float, 0.0), is_set)


def env_bool(name: str, default: Any = _UNSET) -> bool:
    """读布尔; true/1/yes/on → 真, false/0/no/off/空 → 假, 其他 → 默认值."""
    s, raw, is_set = _read(name, default)
    if is_set:
        r = str(raw).strip().lower()
        if r in _TRUE:
            return True
        if r in _FALSE:
            return False
        logger.warning("env %s=%r 非布尔, 回落默认 %r", name, raw, s.default)
    return _as_bool(s.default if default is _UNSET else default)


def env_json(name: str, default: Any = _UNSET) -> Any:
    """读 JSON; 未设/解析失败 → 默认值解析; 再失败 → None."""
    s, raw, is_set = _read(name, default)
    if isinstance(raw, dict | list):
        return raw
    if is_set:
        try:
            return _json.loads(str(raw))
        except (ValueError, TypeError):
            logger.warning("env %s=%r 非 JSON, 回落默认", name, raw)
    try:
        return parse_value(s.type, raw if raw is not None else s.default)
    except (ValueError, TypeError):
        return None


def env_default(name: str) -> Any:
    """返回变量的权威默认值 (按声明类型解析)."""
    s = spec(name)
    return parse_value(s.type, s.default)


def env_resolve(name: str) -> Any:
    """读变量的**生效值**, 支持 ``derived_from`` 跨变量派生.

    解析顺序:

    1. 本变量显式设置 → 用本变量 (仍按 ``choices`` / ``range`` 校验);
    2. 否则沿 ``derived_from`` 有序来源链, 取第一个**已显式设置**的值 (来源自身
       也声明了派生时继续向上找, 途中不去读取任何变量的默认值);
    3. 全未设置 → 回落本变量的权威默认值.

    来源值按**本变量**的声明类型解析; 解析失败或违反约束时告警并回落默认值
    (与 :func:`env_str` 等的 fail-open 口径一致, 不中断服务).
    """
    s = spec(name)
    hit = _raw_from_chain(name, set())
    if hit is None:
        return env_default(name)
    source, raw = hit
    try:
        value = parse_value(s.type, raw)
    except (ValueError, TypeError):
        logger.warning(
            "env %s 的派生来源 %s=%r 非 %s, 回落默认 %r",
            name, source, raw, s.type, s.default,
        )
        return env_default(name)
    # 值来自派生来源时同样受本变量约束管辖; 违反 → 回落权威默认值.
    return _enforce(s, name, value, True)


def _raw_from_chain(name: str, seen: set[str]) -> tuple[str, str] | None:
    """沿 ``derived_from`` 找第一个显式设置的原始值; 返回 ``(来源名, 原值)``.

    ``seen`` 防环 (声明面已由治理测试硬判无环, 这里是运行期兜底).
    """
    if name in seen:
        return None
    seen.add(name)
    raw = os.environ.get(name)
    if raw is not None and str(raw).strip() != "":
        return name, str(raw)
    for src in spec(name).derived_from or ():
        if not is_declared(src):
            # 未声明来源属声明面缺陷, 由治理测试显形; 运行期跳过以保持可用.
            continue
        hit = _raw_from_chain(src, seen)
        if hit is not None:
            return hit
    return None


def _coerce(spec_obj: EnvSpec, name: str, raw: Any, fn: Any, fallback: Any) -> Any:
    try:
        return fn(str(raw).strip())
    except (ValueError, TypeError):
        logger.warning("env %s=%r 非 %s, 回落默认 %r", name, raw, spec_obj.type, spec_obj.default)
        try:
            return fn(str(spec_obj.default).strip())
        except (ValueError, TypeError):
            return fallback
