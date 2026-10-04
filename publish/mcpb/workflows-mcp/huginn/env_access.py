"""HUGINN_* 环境变量的类型化访问器 (配置面唯一受认可的读端).

配置面的**声明**在 :mod:`huginn.env_schema` (``SCHEMA``, 单一权威源); 本模块是
**读端**: 新代码读 ``HUGINN_*`` 一律经此, 不再裸 ``os.environ.get``. 收益有三:

- **未声明即报错**: 读一个不在 ``SCHEMA`` 里的变量抛 :class:`UndeclaredEnvVar`,
  迫使新增配置先登记 (拦截配置面静默膨胀);
- **默认值单一来源**: 默认值只来自 ``SCHEMA``, 不在每个读点重复 (消除双真值);
- **类型统一**: int/float/bool/json 解析集中一处, 解析失败 fail-open 回落默认值.

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


class UndeclaredEnvVar(KeyError):
    """读取了未在 :data:`huginn.env_schema.SCHEMA` 登记的 ``HUGINN_*`` 变量.

    这是配置面治理的硬闸: 新增环境变量必须先在 ``env_schema.SCHEMA`` 登记,
    再经本模块读取. 抛 KeyError 子类以便调用方按需捕获.
    """


def spec(name: str) -> EnvSpec:
    """取变量的权威声明; 未登记抛 :class:`UndeclaredEnvVar`."""
    try:
        return SCHEMA[name]
    except KeyError:
        raise UndeclaredEnvVar(
            f"{name!r} 未在 huginn.env_schema.SCHEMA 登记; "
            f"新增配置请先登记再经 env_access 读取"
        ) from None


def is_declared(name: str) -> bool:
    """变量是否已在 ``SCHEMA`` 登记 (不抛错, 供工具/审计查询)."""
    return name in SCHEMA


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


def env_str(name: str, default: Any = _UNSET) -> str:
    """读字符串; 未设或空 → 默认值 (字符串化)."""
    _s, raw, _set = _read(name, default)
    return "" if raw is None else str(raw)


def env_int(name: str, default: Any = _UNSET) -> int:
    """读整数; 未设/解析失败 → 默认值 (再失败 → 0)."""
    s, raw, _set = _read(name, default)
    return _coerce(s, name, raw, int, 0)


def env_float(name: str, default: Any = _UNSET) -> float:
    """读浮点; 未设/解析失败 → 默认值 (再失败 → 0.0)."""
    s, raw, _set = _read(name, default)
    return _coerce(s, name, raw, float, 0.0)


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
    if isinstance(raw, (dict, list)):
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


def _coerce(spec_obj: EnvSpec, name: str, raw: Any, fn: Any, fallback: Any) -> Any:
    try:
        return fn(str(raw).strip())
    except (ValueError, TypeError):
        logger.warning("env %s=%r 非 %s, 回落默认 %r", name, raw, spec_obj.type, spec_obj.default)
        try:
            return fn(str(spec_obj.default).strip())
        except (ValueError, TypeError):
            return fallback