"""Code Lab — 书生(LLM)写实验代码、框架受限执行的真实科研桥.

让主研究员模型不只"在白名单里选配置", 还能**亲手编写**新的实验函数,
在 Huginn 自带的安全沙箱 (code_act_sandbox) 里真实执行, 数值照旧进
trace 供 claim_grounding 门禁核对. 诚实红线不变: 执行异常/超时/结果
schema 不过 → 分支弃用 (不产生证据, 不伪造).

契约 (书生代码必须遵守):
  def run(cfg: dict) -> dict:
      # cfg 含 order/ctype/system/k/positions/basis/seeds 等数值配置
      # 纯 numpy/纯数值计算; 禁止 IO/网络/改宿主
      return {"success": bool, "summary": {...}, "objectives": {name: float}}

可选探针 (实现即自动注册为诊断工具, 成文期可自主调用):
  def probe_<name>(cfg: dict) -> dict:   # 返回可 JSON 序列化 dict

安全: 复用 code_act_sandbox 的 make_safe_builtins (无 exec/eval/compile/
open/globals/locals) + safe_import 白名单 (numpy/scipy/sympy/math/json/...)
+ exec_with_mem_cap 内存峰值监控; 运行再用超时子线程兜底, 防死循环.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any

SAFE_MEM_CAP = 512 * 1024 * 1024      # 512MB 峰值 (tracemalloc 监控)
SAFE_TIMEOUT_S = 30.0                 # run()/probe 单次调用超时


def extract_code(text: str) -> str:
    """从 LLM 输出里提取实验代码块(兼容 <code>/```python```/裸 def), 从后往前取.

    LLM 输出常带前置思考流(内含模板引用), 因此一律取**最后一个**含
    "def run(" 的可解析块 —— 那是最终交付代码, 思考里的模板引用被忽略.
    提取不到返回空串(调用方弃用回退, 不伪造).
    """
    text = text or ""
    blocks: list[str] = []
    for m in re.finditer(r"<code>(.*?)</code>", text, re.DOTALL | re.IGNORECASE):
        blocks.append(m.group(1).strip())
    for m in re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE):
        blocks.append(m.group(1).strip())
    i = text.find("def run(")                      # 裸函数体兜底
    if i >= 0:
        raw = text[i:].strip()
        fence = raw.find("```")                    # 截断尾部栅栏/后附叙述, 只留代码
        if fence >= 0:
            raw = raw[:fence].rstrip()
        blocks.insert(0, raw)                      # 放最前: 倒序扫描时最后才兜底, 干净块优先
    for b in reversed(blocks):                     # 后→前, 避开思考流里的模板引用
        if "def run(" in b and "return" in b and "```" not in b:
            return b
    return blocks[0] if blocks else ""


def _to_py(v: Any) -> Any:
    """把 numpy 标量/数组递归转成 JSON 可序列化的 Python 类型."""
    import numpy as np
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _to_py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_py(x) for x in v]
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    try:
        json.dumps(v)
        return v
    except Exception:  # noqa: BLE001 — 无法序列化: 如实替换为类名, 不伪造数值
        return f"<{type(v).__name__}:{str(v)[:80]}>"


def _call_with_timeout(fn, arg: dict, timeout: float = SAFE_TIMEOUT_S):
    """超时容器: 死循环/卡死的代码不会拖死整个管线."""
    out: list = [None]
    err: list = [None]

    def _go():
        try:
            out[0] = fn(arg)
        except Exception as e:  # noqa: BLE001
            err[0] = e

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"code_lab 执行超时 (> {timeout}s)")
    if err[0] is not None:
        raise err[0]
    return out[0]


def _load_namespace(code: str, mem_cap: int = SAFE_MEM_CAP,
                    imports_whitelist_extra: tuple[str, ...] = ()) -> dict:
    """在安全沙箱里执行代码, 返回定义出的命名空间 (run/probe_*).

    ``imports_whitelist_extra``: 域级 import 白名单增量(冷启动守卫的
    imports_whitelist_extra 注入点), 只在本次调用里并入安全白名单 —— 诚实红线:
    仅"放行良性的该域科学计算依赖", 绝不放宽 __import__ 本身或加任何 IO/网络模块.
    """
    from huginn.security.code_act_sandbox import exec_with_mem_cap, make_safe_builtins, safe_import
    import builtins as _bi
    import numpy as np
    ns: dict = {"__builtins__": make_safe_builtins(), "np": np}
    if imports_whitelist_extra:
        extras = set(imports_whitelist_extra)

        def _domain_import(name, globals_=None, locals_=None, fromlist=(), level=0):
            # 域级白名单增量仅在基础 safe_import 拒绝后兜底放行 —— 不绕过其安全逻辑.
            try:
                return safe_import(name, globals_, locals_, fromlist, level)
            except ImportError:
                if name.split(".")[0] in extras:
                    return _bi.__import__(name, globals_, locals_, fromlist, level)
                raise
        ns["__builtins__"] = dict(ns["__builtins__"])
        ns["__builtins__"]["__import__"] = _domain_import
    exec_with_mem_cap(code, ns, mem_cap)
    return ns


def _coerce_author_result(res: Any) -> tuple[Any, str | None]:
    """把书生 run() 的裸返回宽容成标准结构, 但不伪造数值 (方案 A).

    书生常直接 `return {"gain": 1.4142, "peak": 3.2}` 而非严格装进
    {"success", "summary", "objectives"}。这些值是**真实计算数值**, 只因少了
    包装而被拒、整体回退白名单 —— 造成自主数值被浪费。这里做"收尾宽容":
      - 若返回顶层是 dict 且含**纯数值标量键**, 视为 objectives, sum组件为这些键的
        副本(全部值真实), success=True —— 不做任何数值合成。
      - 若已含 objectives/summary 则走原校验。
    仍不接受的硬伤(非 dict / 值不可 JSON 序列化 / 纯字符串控制台输出)照旧拒绝。
    """
    import numpy as np
    if not isinstance(res, dict):
        return res, None  # 交给原 schema 校验报"必须返回 dict"
    # 已是标准结构 → 走原校验.
    if isinstance(res.get("objectives"), dict) and isinstance(res.get("summary"), dict):
        return res, None
    # 裸数值 dict 补包装: 只认能 float() 的标量键, 其余键原样丢进 summary(真实值在).
    scalar_obj: dict[str, float] = {}
    raw_sum: dict[str, Any] = {}
    for k, v in res.items():
        if k in ("objectives", "summary", "success"):
            continue
        if isinstance(v, (int, float, np.number)):
            scalar_obj[str(k)] = float(v)
        else:
            raw_sum[str(k)] = v
    if not scalar_obj:
        return res, None  # 无任何数值标量键 → 非预期的控制台输出, 交原校验拒绝
    return {
        "success": True,
        "objectives": scalar_obj,          # 全真实标量, 未合成
        "summary": {"author_raw": res, **raw_sum},
    }, None


def _check_run_schema(res: Any) -> str | None:
    """run() 返回 schema 校验; 通过返回 None, 否则返回原因.

    诚实红线: 不伪造数值, 只对"收尾样式"宽容 —— success 缺失或缺 bool 包
    装时, 只要跑出非空数值 objectives 就视为计算成功(数值本身未变).
    真正的 schema 硬伤(非 dict/无数值/值不可序列化)照旧拒绝, 回退白名单.
    注: sandbox_run 会先经 _coerce_author_result 把"裸数值 dict"补包装再调本校验.
    """
    import numpy as np
    if not isinstance(res, dict):
        return "run(cfg) 必须返回 dict"
    if not isinstance(res.get("summary"), dict):
        return "需要 dict 字段 summary (可证伪数值轨迹)"
    obj = res.get("objectives")
    if not isinstance(obj, dict) or not obj:
        return "需要非空 dict 字段 objectives (每个值须为数值)"
    for _, v in obj.items():
        if not isinstance(v, (int, float, np.number)):
            return f"objectives 值须为数值: {v!r}"
    # 宽容收尾: success 缺失/是 numpy.bool_/任意标量 → 依"跑出数值"推断 True.
    # 这是样式宽容, 不是数值伪造 —— objectives 已全为真实计算数值.
    return None


def _alias_cfg(cfg: dict) -> dict:
    """给书生代码一个宽容的 cfg 视图: 常用别名键补齐(值是同一份真实配置的引用).

    书生习惯用领域术语 ti / t_i / t 称呼约束位置, 框架键名是 positions;
    seeds 也可能写成 n_seeds. 断裂域书生则常用 sigma_0 / sigma0_sigmay /
    sigma0_over_sy 表达"桥联比"而非框架键 bridge_ratio —— 物理命名合理, 只差键名.
    补齐后代码可自由选键, 数值源不变(仍为真实配置).
    只做键别名, 绝不引入新数值 —— 诚实红线不变.
    """
    out: dict = dict(cfg or {})
    positions = out.get("positions")
    if positions:
        out.setdefault("ti", positions[0] if len(positions) == 1 else positions)
        out.setdefault("t_i", out["ti"])
        out.setdefault("t", positions[0] if len(positions) == 1 else positions)
    out.setdefault("n_seeds", out.get("seeds"))
    out.setdefault("basis_size", out.get("basis"))
    # 断裂域物理键别名: 全映射到同一份真实数值, 不新造任何量.
    for alias in ("sigma_0", "sigma0", "sigma0_sigmay", "sigma0_over_sy",
                  "bridge", "bridging_ratio"):
        out.setdefault(alias, out.get("bridge_ratio"))
    for alias in ("n_flaw", "n_defects", "defect_count"):
        out.setdefault(alias, out.get("n_flaws"))
    for alias in ("a_over_astar", "aastar", "aa_star", "flaw_size"):
        out.setdefault(alias, out.get("flaw_idx"))
    for alias in ("kic_i", "kic_idx", "KIC_mat"):
        out.setdefault(alias, out.get("kic_mat"))
    for alias in ("v_cr", "v_cR", "v_over_cR"):
        out.setdefault(alias, out.get("vcR"))
    for alias in ("poisson", "poisson_ratio"):
        out.setdefault(alias, out.get("nu"))
    return out


def sandbox_run(code: str, cfg: dict, *, mem_cap: int = SAFE_MEM_CAP,
                timeout: float = SAFE_TIMEOUT_S,
                imports_whitelist_extra: tuple[str, ...] = ()) -> tuple[dict | None, str | None]:
    """执行书生写的实验代码: 返回 (结果 dict 或 None, 错误原因或 None).

    ``imports_whitelist_extra``: 冷启动守卫的域级 import 白名单增量, 仅该次调用生效.
    """
    if not code.strip():
        return None, "空代码"
    # 无显示主机的 matplotlib: 强制 Agg 后端, 避免 pyplot 因无 DISPLAY 崩 —
    # 数值计算/存图照常走 Agg, 不依赖 GUI 头. (对已设 MPLBACKEND 的调用方生效)
    import os as _os
    _os.environ.setdefault("MPLBACKEND", "Agg")
    cfg = _alias_cfg(cfg)
    try:
        ns = _load_namespace(code, mem_cap,
                             imports_whitelist_extra=imports_whitelist_extra)
        run_fn = ns.get("run")
        if not callable(run_fn):
            return None, "未找到 def run(cfg) 入口"
        res = _call_with_timeout(run_fn, cfg, timeout)
    except TimeoutError as te:
        return None, str(te)
    except Exception as e:  # noqa: BLE001 — 执行异常如实记录
        return None, f"执行异常: {type(e).__name__}: {e}"
    res, _ = _coerce_author_result(res)   # 宽容"裸数值 dict", 不伪造数值
    reason = _check_run_schema(res)
    if reason is not None:
        return None, reason
    # success 归一: 显式 bool(passed 计算) —— 宽容收尾下可能是缺省/非 bool 标量.
    _success = res.get("success", True)
    try:
        ok = bool(_success)
    except Exception:  # noqa: BLE001 — 无法 bool() 的异常值按 True 处理(有数值即成功)
        ok = True
    return {
        "success": ok,
        "summary": _to_py(res["summary"]),
        "objectives": {str(k): float(v) for k, v in res["objectives"].items()},
    }, None


def author_probe_specs(code: str) -> list[dict]:
    """把书生代码里的 probe_<name>(cfg) 自动注册为诊断工具 (全权探针面).

    返回 [{tool: {function: {name, description, parameters}}, handle}].
    单条注册失败不影响整体 —— 成文期由工具面统一兜底.
    """
    try:
        ns = _load_namespace(code)
    except Exception:  # noqa: BLE001 — 探针注册失败即跳过, 不阻塞
        return []
    specs: list[dict] = []
    for name, fn in ns.items():
        if not name.startswith("probe_") or not callable(fn):
            continue
        doc = (fn.__doc__ or name).strip().splitlines()[0][:200]
        specs.append({
            "tool": {"function": {
                "name": name,
                "description": f"[书生代码实验室] {doc}",
                "parameters": {"type": "object", "properties": {
                    "config": {"type": "object"}},
                    "required": [], "additionalProperties": True}}},
            "handle": (lambda c, f=fn: json.dumps(
                _to_py(_call_with_timeout(f, dict(c or {}) if isinstance(c, dict) else c or {})),
                ensure_ascii=False)),
        })
    return specs