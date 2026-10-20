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
    """不变量 4b: **不经 env_access 的** HUGINN_* 读点总数只减不增 (棘轮).

    只算"裸读": 直接 ``os.environ.*`` 与 ad-hoc 包装 (``_env_int`` 等). 经
    ``huginn.env_access`` 规范访问器的读点标 ``canonical`` —— 它们是治理的**目标
    态**, 不计入棘轮 (否则"迁移读点到 env_access"反而推高计数, 与棘轮自相矛盾).
    """
    total = sum(
        1
        for item in _inventory().values()
        for r in item["reads"]
        if not r.get("canonical")
    )
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


# ── 缺口②: 值域约束 (choices / range) 自洽门 ──────────────────────────


def test_declared_constraints_are_self_consistent():
    """不变量 5: 约束声明的默认值必须自身满足该约束.

    ``choices`` / ``range`` 与默认值同为声明面; 声明一个默认值即越界的约束, 等于
    读端一启动就恒回落并刷告警. 本门把"约束 ↔ 默认值"钉死: 新增/改动约束后若默认
    值越界 → 红.
    """
    bad: list[tuple[str, str]] = []
    for name, spec in es.SCHEMA.items():
        if spec.choices is None and spec.range is None:
            continue
        default = ea.parse_value(spec.type, spec.default)
        reason = ea.constraint_violation(spec, default)
        if reason is not None:
            bad.append((name, reason))
    assert not bad, "以下变量的约束与其权威默认值不自洽 (默认值即越界): " + "; ".join(
        f"{n}: {r}" for n, r in bad
    )


def test_choice_candidates_parse_under_declared_type():
    """不变量 5b: choices 非空, 且每个候选都能按声明类型解析 (无坏候选)."""
    for name, spec in es.SCHEMA.items():
        if spec.choices is None:
            continue
        assert spec.choices, f"{name} 的 choices 为空元组 (无意义约束)"
        for cand in spec.choices:
            try:
                ea.parse_value(spec.type, cand)
            except (ValueError, TypeError) as exc:  # pragma: no cover - 断言路径
                raise AssertionError(
                    f"{name} 的 choices 候选 {cand!r} 无法按 {spec.type!r} 解析: {exc}"
                ) from exc


def test_range_declared_only_on_numeric_types():
    """不变量 5c: range 仅用于 int/float 型 (非数值型的区间约束无法比较)."""
    bad = sorted(
        name
        for name, spec in es.SCHEMA.items()
        if spec.range is not None and spec.type not in ("int", "float")
    )
    assert not bad, f"以下变量声明了 range 但类型非数值 (int/float): {bad}"


def test_strict_validate_rejects_out_of_domain_value():
    """严格校验出口: ``env_access.validate`` 对越界取值抛错、对合法取值放行.

    读端走 fail-open (告警 + 回落), 写端/审计用 ``validate`` **显式拒绝** —— 这是
    缺口② 的"显式报错"路径; 本门确保它真会红, 且约束面非空 (门不空转).
    """
    import pytest

    constrained = [
        n for n, s in es.SCHEMA.items() if s.choices is not None or s.range is not None
    ]
    assert constrained, "无任何变量声明值域约束 —— 缺口② 的约束面为空 (前向非空转)"

    name = "HUGINN_MODEL_TIER"
    assert ea.validate(name, "full") == "full"
    with pytest.raises(ea.EnvConstraintError):
        ea.validate(name, "definitely-not-a-tier")


# ── 缺口②: 跨变量派生 (derived_from) 自洽门 ────────────────────────────


def test_derived_from_sources_are_declared_and_sane():
    """不变量 6: derived_from 来源必须已声明, 且无自引用 / 无重复来源.

    派生来源是一个**变量名**, 若未在 ``SCHEMA`` 登记则运行期读不到 (静默失效);
    自引用与重复来源是声明笔误. 本门把这三类钉死.
    """
    bad: list[str] = []
    for name, spec in es.SCHEMA.items():
        if not spec.derived_from:
            continue
        for src in spec.derived_from:
            if src not in es.SCHEMA:
                bad.append(f"{name} ← {src} (来源未登记)")
        if name in spec.derived_from:
            bad.append(f"{name} 自引用自身")
        if len(set(spec.derived_from)) != len(spec.derived_from):
            bad.append(f"{name} 派生链含重复来源")
    assert not bad, "derived_from 声明不自洽: " + "; ".join(bad)


def test_derived_from_chain_is_acyclic():
    """不变量 6b: 派生图无环 (resolve 不会沿链死循环)."""
    def walk(node: str, stack: list[str]) -> None:
        for src in es.SCHEMA[node].derived_from or ():
            if src in stack:
                raise AssertionError("派生环: " + " -> ".join([*stack, src]))
            walk(src, [*stack, src])

    for name in es.SCHEMA:
        walk(name, [name])


def test_env_resolve_prefers_own_value_then_source_then_default(monkeypatch):
    """resolve 行为门: 自身设置 > 派生来源 > 权威默认; 且派生面非空 (门不空转)."""
    derived = [n for n, s in es.SCHEMA.items() if s.derived_from]
    assert derived, "无任何变量声明 derived_from —— 缺口② 的派生面为空 (前向非空转)"

    name = "HUGINN_JWT_SECRET"
    src = es.SCHEMA[name].derived_from[0]

    monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(src, raising=False)
    assert ea.env_resolve(name) == ea.env_default(name)

    monkeypatch.setenv(src, "from-source")
    assert ea.env_resolve(name) == "from-source"

    monkeypatch.setenv(name, "own-value")
    assert ea.env_resolve(name) == "own-value"
