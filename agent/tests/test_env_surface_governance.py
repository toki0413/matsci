"""配置面治理测试: 单一权威声明 + 只减不增的裸读棘轮.

见 :mod:`huginn.env_schema` 的模块 docstring. 本测试把配置面从"无治理的散读"
变成"可持续承诺"的四条不变量:

1. 代码里出现的每个 ``HUGINN_*`` 必须已在 ``SCHEMA`` 声明 (拦截静默新增);
2. 每条 ``SCHEMA`` 声明必须仍被代码引用 (拦截死声明; 动态读的规范名/旧别名除外);
3. ``SCHEMA`` 默认值必须与代码里最常见默认值**语义一致** (拦截默认漂移);
4. 裸读站点数 ``<= RAW_READ_BASELINE``, 且已知默认值分散只减不增 (棘轮).

代码里新增 ``HUGINN_*`` 而没登记 → 测试红; 要改默认值 → 先改 ``SCHEMA``;
要收敛配置面 → 把读点迁到 :mod:`huginn.env_access` 并下调 ``RAW_READ_BASELINE``.
"""

from __future__ import annotations

import re

from huginn import env_access as ea
from huginn import env_schema as es
from huginn.cli import config_audit as ca
from huginn.feature_flags import FeatureFlags

# 与 env_schema 生成口径一致: 丢弃表达式默认值 (如 str(x)) 与裸常量名 (如 _DEFAULT_X),
# 只对字面量默认值做漂移比对, 避免推断误差造成假阳性.
_LITERAL = re.compile(r"^-?[\w.]*$")
_CONST_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _clean(raw: str) -> str:
    raw = (raw or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1]
    if not _LITERAL.match(raw) or _CONST_NAME.match(raw):
        return ""
    return raw


def _inventory() -> dict[str, dict]:
    return ca.build_inventory()


def _dynamic_names() -> set[str]:
    """经 ``FeatureFlags`` 动态读的变量名 —— 静态扫描抓不到, 不算死声明."""
    canon = {f"HUGINN_FEATURE_{name.upper()}" for name in FeatureFlags._DEFAULTS}
    return canon | set(FeatureFlags._ENV_ALIASES)


def test_every_referenced_env_is_declared():
    """不变量 1: 代码里出现的每个 HUGINN_* 必须已在 SCHEMA 登记."""
    missing = sorted(set(_inventory()) - set(es.SCHEMA))
    assert not missing, (
        "以下 HUGINN_* 在代码里被引用但未在 huginn/env_schema.py 的 SCHEMA 登记. "
        "新增配置请先在 SCHEMA 登记, 再经 huginn.env_access 读取: " + ", ".join(missing)
    )


def test_no_dead_schema_entries():
    """不变量 2: 每条声明必须仍被代码引用 (动态读名除外)."""
    referenced = set(_inventory()) | _dynamic_names()
    dead = sorted(set(es.SCHEMA) - referenced)
    assert not dead, (
        "以下 SCHEMA 声明在代码里已无引用 (死配置), 应删除声明与相关代码: "
        + ", ".join(dead)
    )


def test_feature_flags_declared_and_consistent():
    """不变量 2b: FeatureFlags 的每个 flag 都必须在 SCHEMA 登记, 且默认值一致.

    ``FeatureFlags._DEFAULTS`` 与 ``env_schema`` 是同一个开关的两处声明 (flag 名
    经 ``HUGINN_FEATURE_<NAME>`` 动态读, 静态扫描抓不到) —— 这正是双真值来源的
    高发区. 本测试把两处钉死: 新增 flag 未登记 → 红; 两处默认值不一致 → 红.
    """
    for name, default in FeatureFlags._DEFAULTS.items():
        canon = f"HUGINN_FEATURE_{name.upper()}"
        assert canon in es.SCHEMA, (
            f"FeatureFlags 新增了 flag {name!r} 但 SCHEMA 未登记 {canon}. "
            "请先在 huginn/env_schema.py 登记 (type=bool, 默认值与 _DEFAULTS 一致)."
        )
        spec = es.SCHEMA[canon]
        assert spec.type == "bool", f"{canon} 是功能开关, 类型应为 bool, 实为 {spec.type!r}"
        assert ea.parse_value(spec.type, spec.default) == bool(default), (
            f"{canon} 默认值漂移: SCHEMA={spec.default!r} vs "
            f"FeatureFlags._DEFAULTS[{name!r}]={default!r}"
        )


def test_feature_flag_aliases_declared():
    """不变量 2c: 旧开关变量名 (``_ENV_ALIASES``) 必须已登记, 否则读端会漏."""
    missing = sorted(a for a in FeatureFlags._ENV_ALIASES if a not in es.SCHEMA)
    assert not missing, (
        "以下 FeatureFlags 别名未在 SCHEMA 登记 (旧变量名仍在生效, 读端非法): "
        + ", ".join(missing)
    )


def test_schema_defaults_match_code():
    """不变量 3: SCHEMA 默认值必须与代码里最常见的默认值语义一致."""
    bad: list[tuple[str, str, str]] = []
    for name, item in _inventory().items():
        spec = es.SCHEMA.get(name)
        if spec is None:
            continue
        code_default = _clean(item.get("default", ""))
        if not code_default:
            continue
        try:
            if ea.parse_value(spec.type, code_default) != ea.parse_value(
                spec.type, spec.default
            ):
                bad.append((name, spec.default, code_default))
        except (ValueError, TypeError):
            # 类型与代码默认值不自洽 —— 同样视为漂移, 需人工核对.
            bad.append((name, spec.default, code_default))
    assert not bad, (
        "以下变量的 SCHEMA 默认值与代码默认值不一致 (默认值漂移). 请把代码默认值"
        "收敛到 SCHEMA, 或更新 SCHEMA: "
        + "; ".join(f"{n}: schema={s!r} code={c!r}" for n, s, c in bad)
    )


def test_default_dispersion_does_not_grow():
    """不变量 4a: 同一变量在多处的字面量默认值分散只减不增."""
    current: dict[str, set[str]] = {}
    for name, item in _inventory().items():
        vals = {_clean(r["default"]) for r in item["reads"] if _clean(r["default"])}
        if len(vals) > 1:
            current[name] = vals

    known = {k: set(v) for k, v in es.KNOWN_DEFAULT_DISPERSION.items()}
    new = sorted(set(current) - set(known))
    grown = {k: sorted(current[k] - known[k]) for k in current if current[k] - known[k]}
    assert not new, f"新增默认值分散 (双真值来源), 请先收敛读点再登记: {new}"
    assert not grown, f"已知分散出现新的取值, 请收敛: {grown}"


def test_raw_read_sites_ratchet():
    """不变量 4b: 裸 HUGINN_* 读点总数只减不增 (棘轮)."""
    total = sum(len(item["reads"]) for item in _inventory().values())
    assert total <= es.RAW_READ_BASELINE, (
        f"裸 HUGINN_* 读点 {total} 超过基线 {es.RAW_READ_BASELINE}. "
        "新增读点请用 huginn.env_access; 若已迁移读点, 请下调 "
        "env_schema.RAW_READ_BASELINE."
    )


def test_env_access_enforces_declaration_and_types():
    """访问器自检: 未登记抛错; 声明变量的类型/默认值按 SCHEMA 生效."""
    import pytest

    with pytest.raises(ea.UndeclaredEnvVarError):
        ea.env_str("HUGINN_DEFINITELY_NOT_DECLARED_XYZ")

    # 每个已登记变量的默认值都能按其声明类型解析 (无坏声明).
    for name, spec in es.SCHEMA.items():
        ea.parse_value(spec.type, spec.default) if spec.default != "" else None
        assert ea.is_declared(name)
