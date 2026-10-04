"""One-off generator for huginn/env_schema.py (scratch; deleted after use)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/agent")

from huginn.cli import config_audit as ca  # noqa: E402
from huginn.feature_flags import FeatureFlags as FF  # noqa: E402

inv = ca.build_inventory()
aliases = set(FF._ENV_ALIASES)
canon = {f"HUGINN_FEATURE_{n.upper()}": v for n, v in FF._DEFAULTS.items()}

_LITERAL = re.compile(r"^-?[\w.]*$")
_CONST_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def clean_default(raw: str) -> str:
    raw = (raw or "").strip()
    if not _LITERAL.match(raw):
        return ""  # expression / call default -> unknown
    if _CONST_NAME.match(raw):
        return ""  # bare constant name (e.g. _DEFAULT_TIMEOUT_SEC), not a literal
    return raw


def norm_bool(raw: str) -> str:
    return "true" if (raw or "").strip().lower() in ("true", "1", "yes", "on") else "false"


def infer_type(name: str, default: str) -> str:
    if name in canon or name in aliases:
        return "bool"
    d = (default or "").strip().lower()
    if d in ("true", "false", "yes", "no", "on", "off"):
        return "bool"
    if re.fullmatch(r"-?\d+", d):
        return "int"
    if re.fullmatch(r"-?\d+\.\d*", d):
        return "float"
    if (default or "").lstrip().startswith(("{", "[")):
        return "json"
    return "str"


_DEPLOY = re.compile(
    r"(API_KEY|_KEY$|TOKEN|SECRET|PASSWORD|CREDENTIAL|_URL$|BASE_URL|ENDPOINT"
    r"|_HOST$|_PORT$|_PATH$|_DIR$|_HOME$|PROXY|CERT|WORKSPACE|_ROOT$"
    r"|MODEL|PROVIDER)"
)


def scope_of(name: str, typ: str) -> str:
    if typ == "bool":
        return "feature"
    if name in aliases:
        return "legacy"
    if _DEPLOY.search(name):
        return "deployment"
    if typ in ("int", "float"):
        return "tuning"
    return "internal"


entries: dict[str, tuple[str, str, str, str]] = {}
for name, item in inv.items():
    default = clean_default(item["default"])
    typ = infer_type(name, default)
    if typ == "bool":
        default = norm_bool(default)
    entries[name] = (typ, default, scope_of(name, typ), "live")

for cname, default in canon.items():
    entries[cname] = ("bool", "true" if default else "false", "feature", "live")

for alias, flag in FF._ENV_ALIASES.items():
    d = "true" if FF._DEFAULTS.get(flag, True) else "false"
    entries[alias] = ("bool", d, "legacy", "deprecated")

dispersion: dict[str, list[str]] = {}
for name, item in inv.items():
    ds = sorted({clean_default(r["default"]) for r in item["reads"] if clean_default(r["default"])})
    if len(ds) > 1:
        dispersion[name] = ds

raw_read_sites = sum(len(item["reads"]) for item in inv.values())

_SCOPE_ORDER = ["deployment", "feature", "tuning", "internal", "legacy"]

L: list[str] = []
A = L.append
A('"""HUGINN_* 环境变量的单一权威声明 (配置面的声明层).')
A("")
A("为什么需要: 配置面此前散在 145 个文件、385 个变量上 —— 每个裸读点各自带")
A("默认值, 无统一 schema, 无类型, 新变量可静默膨胀, 死变量无人回收. 本模块把")
A('"配置面"收成**一处声明**: 每个 `HUGINN_*` 在此登记一次 (类型 / 权威默认值 /')
A("作用域 / 状态), 读点改用 :mod:`huginn.env_access` 的类型化访问器取值.")
A("")
A("本文件是配置面的**单一权威源** (single source of truth): 默认值只在这里声明一")
A("次, 不再在每个读点重复. `tests/test_env_surface_governance.py` 强制以下不变量:")
A("")
A("1. 代码里出现的每个 `HUGINN_*` 必须已在此声明 (拦截静默新增);")
A("2. 每条声明必须仍被代码引用 (拦截死声明);")
A("3. 声明的默认值必须与代码里最常见的默认值一致 (拦截默认漂移);")
A("4. 裸 `os.environ` 读点数只减不增 (棘轮, 逼配置面收敛).")
A("")
A("字段:")
A("    type:    str|int|float|bool|json —— 值解析方式")
A("    default: 权威默认值 (字符串形式; 空串 = 无默认)")
A("    scope:   deployment(部署相关: 凭证/路径/模型) | feature(开关)")
A("             | tuning(数值调参) | internal(内部常量) | legacy(迁移残余)")
A("    status:  live | deprecated(旧变量名, 仅为向后兼容保留)")
A("")
A("改本文件后跑 `python -m huginn.cli.config_audit` 复核; 新增变量必须先登记,")
A("再在代码里经 env_access 读取 —— 不允许再出现未经声明的裸读.")
A('"""')
A("")
A("from __future__ import annotations")
A("")
A("from dataclasses import dataclass")
A("")
A("")
A("@dataclass(frozen=True)")
A("class EnvSpec:")
A('    """单个 ``HUGINN_*`` 变量的权威声明."""')
A("")
A("    type: str")
A("    default: str")
A("    scope: str")
A('    status: str = "live"')
A("")
A("")
A("# 单一权威声明表 (按作用域分组, 组内按名排序).")
A("SCHEMA: dict[str, EnvSpec] = {")
for scope in _SCOPE_ORDER:
    members = sorted(n for n, v in entries.items() if v[2] == scope)
    if not members:
        continue
    A(f"    # ---- {scope} ({len(members)}) " + "-" * max(0, 54 - len(scope)))
    for name in members:
        typ, default, _s, status = entries[name]
        suffix = "" if status == "live" else f', status="{status}"'
        A(f'    "{name}": EnvSpec("{typ}", "{default}", "{scope}"{suffix}),')
A("}")
A("")
A("")
A("# 裸读站点棘轮基线 = 当前代码里字面量 HUGINN_* 读点总数.")
A("# 治理测试断言实际值 <= 此数, 即只减不增; 迁移读点到 env_access 后应下调此数.")
A(f"RAW_READ_BASELINE = {raw_read_sites}")
A("")
A("")
A("# 已知默认值分散: 同一变量在代码多处字面量默认值不一致 (双真值来源).")
A("# 治理测试只允许此集合收窄, 不允许新增 —— 新增分散须先收敛读点再登记.")
A("KNOWN_DEFAULT_DISPERSION: dict[str, tuple[str, ...]] = {")
for name in sorted(dispersion):
    vals = ", ".join(f'"{v}"' for v in dispersion[name])
    A(f'    "{name}": ({vals}),')
A("}")
A("")

Path("/workspace/agent/huginn/env_schema.py").write_text("\n".join(L), encoding="utf-8")

from collections import Counter  # noqa: E402

print("wrote", len(entries), "entries; raw_read_sites =", raw_read_sites)
print("by scope:", dict(Counter(v[2] for v in entries.values())))
print("by type:", dict(Counter(v[0] for v in entries.values())))
print("dispersion:", dispersion)