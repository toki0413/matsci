"""符号解析辅助: 符号声明 + 表达式安全解析 + Einstein 指标 token 解析."""

from __future__ import annotations

import re

import sympy as sp

# sympify 内部走 eval, 拦掉危险模式做 defense-in-depth
# 注意: 不能用泛化的 "__" 匹配, 因为合法的内部变量名 (如 __u_prime__) 也含双下划线
_DANGEROUS_PATTERNS = (
    "__import__", "__builtins__", "__class__", "__subclasses__",
    "__globals__", "__dict__", "__getattribute__", "__bases__",
    "__mro__", "__loader__", "__spec__", "__code__",
    "import ", "exec(", "eval(", "open(", "os.", "sys.",
    "subprocess", "globals(", "locals(", "getattr(", "setattr(",
)


def parse_symbols(
    symbol_names: list[str], assumptions: dict[str, str]
) -> dict[str, sp.Symbol]:
    """按 assumptions 给每个名字建 SymPy Symbol."""
    sym_dict = {}
    for name in symbol_names:
        ass = {}
        if name in assumptions:
            a = assumptions[name]
            if a == "positive":
                ass["positive"] = True
            elif a == "real":
                ass["real"] = True
            elif a == "complex":
                ass["complex"] = True
            elif a == "nonnegative":
                ass["nonnegative"] = True
        sym_dict[name] = sp.Symbol(name, **ass)
    return sym_dict


def _validate_expression(expr: str) -> None:
    """拦掉可能触发代码执行的模式."""
    low = expr.lower()
    for pat in _DANGEROUS_PATTERNS:
        if pat.lower() in low:
            raise ValueError(f"表达式包含禁用序列 '{pat}'")


def safe_parse(expr_str: str, sym_dict: dict[str, sp.Symbol]) -> sp.Expr:
    """把字符串安全解析成 SymPy 表达式.

    先塞内置常数和函数, 再让用户符号覆盖它们 —— 否则用户的 "E" (杨氏模量)
    会被 sp.E (欧拉数) 吃掉.
    """
    _validate_expression(expr_str)
    local_dict = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
        "E": sp.E,
        "diff": sp.diff,
        "integrate": sp.integrate,
    }
    local_dict.update(sym_dict)
    return sp.sympify(expr_str, locals=local_dict)


def parse_einstein_token(token: str):
    """解析一个张量 token，返回 (name, upper, lower).

    支持 A_ij / A^i_j / A_{ij} / A^{ij} 这几种常见记号.
    """
    m = re.match(r"([A-Za-z][A-Za-z0-9]*)", token)
    if not m:
        return None
    name = m.group(1)
    rest = token[m.end():]

    upper: list[str] = []
    lower: list[str] = []
    i = 0
    while i < len(rest):
        c = rest[i]
        if c in "^_":
            pos = c
            i += 1
            # 花括号包住的多字符指标
            if i < len(rest) and rest[i] == "{":
                end = rest.index("}", i)
                chars = rest[i + 1:end].replace(" ", "")
                indices = list(chars)
                i = end + 1
            else:
                # 没有花括号就把后续连续的字母数字都当成指标序列
                # 每个字符是一个指标，遇到下一个 ^/_ 或结尾停止
                j = i
                while j < len(rest) and rest[j] not in "^_":
                    j += 1
                indices = list(rest[i:j])
                i = j
            if pos == "^":
                upper.extend(indices)
            else:
                lower.extend(indices)
        else:
            i += 1
    return name, upper, lower
