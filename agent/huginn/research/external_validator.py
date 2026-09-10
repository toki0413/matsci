"""外置契约验证器 (External Contract Validator).

背景(泛化诊断): 之前 `reuse_score` 的判据(`_objectives_extract`)刻意复刻了 harness 自身
的宽容路径 `_coerce_author_result` / `_check_run_schema` —— 判定器和被测产出**共享同一套
假设**(同一个宽松提取术 / 同一张别名表), 于是"检查器无法和它检查的东西分歧", 错误要到
真实边界才暴露。这里抽出**独立、零宽容**的严格判据, 不 import / 不复用 harness 内部,
只认理想契约 `{summary: dict, objectives:{k: 纯数值}}`。

用途:
  - 泛化主线的 **strict contract** 打分(跨域机制是否真对齐统一契约)。
  - held-out(扣出域)泛化的权威判据 —— 与宽容路径可分歧, 而非给它背书。

诚实边界(与宽容路径的关系):
  - 运行期 `code_lab.sandbox_run` 仍走宽容路径(为不浪费真实计算), 但**泛化度量**用本
    模块的严格判据判定 —— 域输出必须自带统一容器, harness 代偿几许, 严格判据如实记。
  - 纯标准库、幂等、零 LLM、零 np 依赖; 判据绝不引用 `_coerce_author_result` /
    `_check_run_schema` / `_alias_cfg` 任一实现, 从结构上保证能和产出"分歧"。
"""
from __future__ import annotations

from typing import Any


def _is_numeric_leaf(v: Any) -> bool:
    """纯数值叶: int/float, 排除 bool。用 (int, float) 而非 numbers.Real 以保持零依赖.*"""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def strict_objectives(res: Any) -> tuple[bool, str]:
    """零宽容契约判定: 通过返回 (True, ''), 否则 (False, 原因)。

    通过所需(每条硬性, 无兜底分支):
      1. res 为非空 dict;
      2. 有 `objectives` 且为非空 dict;
      3. objectives 中 ≥1 个纯数值叶(int/float、非 bool);
      4. 有 `summary` 且为 dict。
    裸数值 dict / 整数键逐族 dict / 缺容器 / 叶为字符串或列表 → 一律不过。
    这条判据不读 harness 的任何实现 —— 与 `_coerce_author_result` 能并互相分歧。
    """
    if not isinstance(res, dict):
        return False, "非 dict"
    obj = res.get("objectives")
    summary = res.get("summary")
    if not isinstance(obj, dict) or not obj:
        return False, "缺非空 objectives(dict)"
    if not isinstance(summary, dict):
        return False, "缺 summary(dict)"
    if not any(_is_numeric_leaf(v) for v in obj.values()):
        return False, "objectives 无纯数值叶"
    return True, ""


def validate_scientific_contract(objectives: dict, quantities: dict) -> tuple[bool, list[str]]:
    """域级科学契约校验(HEP 机器可读科学契约: 约定/有效域).

    消费域声明(compile_domain_guards 的 scientific_contract.quantities)里的
    量纲(unit) + 有效域(domain)。判定**独立于 harness**, 只查域声明的元数据:

      - 'objectives' 里**已声明**的量: 校验落在 [domain.min, domain.max](若声明);
      - 'objectives' 里**未声明**的量: 记为 coverage gap(有效域未知, 如实露), 不判成败
        —— 诚实暴露"契约没覆盖到它", 而不是擅自通过.

    返回 (ok, gaps): ok=False 当存在有效域违反; coverage gap 只在 gaps 里区分标注.
    """
    if not isinstance(quantities, dict):
        return True, []
    gaps: list[str] = []
    any_violation = False
    for key, val in (objectives or {}).items():
        meta = quantities.get(key)
        if meta is None:
            gaps.append(f"{key}:未声明(量纲/有效域未知)")
            continue
        dom = meta.get("domain")
        if not isinstance(dom, dict):
            continue
        lo = dom.get("min")
        hi = dom.get("max")
        if lo is not None and val < lo:
            gaps.append(f"{key}={val}<{lo}(违反有效域下界)")
            any_violation = True
        elif hi is not None and val > hi:
            gaps.append(f"{key}={val}>{hi}(违反有效域上界)")
            any_violation = True
    return (not any_violation, gaps)


__all__ = ["strict_objectives", "validate_scientific_contract"]