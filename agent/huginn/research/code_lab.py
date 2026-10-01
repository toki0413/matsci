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

import importlib
import importlib.util
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SAFE_MEM_CAP = 512 * 1024 * 1024      # 512MB 峰值 (tracemalloc 监控)
SAFE_TIMEOUT_S = 30.0                 # run()/probe 单次调用超时

# 平台默认(命题无关)的 run(cfg) 骨架: 只固定输入/输出契约, 不含任何领域原语、
# 约束族或判据。领域脚手架属于**任务资产**, 经 Scaffold/load_scaffold 装配 ——
# 平台内核不替任何命题长出专用代码 (低熵红线: 通用骨架不为单个命题累积例外)。
DEFAULT_TEMPLATE = """\
import numpy as np


def run(cfg):
    # 你的实验入口: 读参数用 cfg.get('x', 默认值), 不要整体解包 cfg.
    seed = int(cfg.get('seed', 0))
    # ... 你的数值实验 ...
    return {"success": True,
            "summary": {"note": "可证伪的数值轨迹"},
            "objectives": {"score": 0.0}}   # 每个值须为数值
"""


@dataclass
class Scaffold:
    """任务脚手架: 某命题自带、经 load_scaffold 注入沙箱的实验资产.

    平台内核(本模块)只提供**注入点**与契约, 不内置任何领域原语. 命题把自己的
    模板(TEMPLATE)、提示(HINTS)、原语(PRIMITIVES)放进一个独立模块, autoloop/
    示例按需装配 —— 这样"谁定义原语"这件事本身是通用的, 平台不会为单个命题
    长出专用代码.
    """

    name: str = ""
    template: str = ""
    hints: str = ""
    primitives: dict[str, Any] = field(default_factory=dict)
    primitive_names: tuple[str, ...] = ()


def load_scaffold(source: Any) -> Scaffold | None:
    """把外部任务脚手架装配成 Scaffold. ``source`` 支持:

      - None / "" / Scaffold: 原样返回 (无脚手架 → 平台走命题无关默认);
      - "...py" 文件路径 或 "pkg.mod" 模块名: 加载模块, 读约定字段
        ``NAME`` / ``TEMPLATE`` / ``HINTS`` / ``PRIMITIVE_NAMES``, 并调用
        ``primitives()`` 取要注入沙箱的原语表.
    """
    if source is None or source == "":
        return None
    if isinstance(source, Scaffold):
        return source
    mod = source
    if isinstance(source, str | Path):
        p = Path(str(source))
        if p.suffix == ".py":
            spec = importlib.util.spec_from_file_location(
                "huginn_codelab_scaffold_" + p.stem, p)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载脚手架文件: {p}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            mod = importlib.import_module(str(source))
    prims = getattr(mod, "primitives", None)
    prims = dict(prims() if callable(prims) else (prims or {}))
    names = tuple(getattr(mod, "PRIMITIVE_NAMES", ()) or tuple(prims))
    return Scaffold(
        name=str(getattr(mod, "NAME", "") or ""),
        template=str(getattr(mod, "TEMPLATE", "") or ""),
        hints=str(getattr(mod, "HINTS", "") or ""),
        primitives=prims,
        primitive_names=names,
    )


def build_author_prompt(goal: str, *, scaffold: Scaffold | None = None,
                        guard_block: str = "",
                        repair_hint: str = "", prev_code: str = "",
                        focus: str = "") -> str:
    """构造"让书生在 Code Lab 亲手写一轮实验"的作者提示 (命题无关).

    单一出处: autoloop 的 execute 内建动作与 examples 自主循环都调本函数, 通用
    契约(纯 numpy、返回 objectives、不 assert)只维护一份. ``goal`` 由外部传入 ——
    只给方向, 不绑定任何具体命题. 领域相关的模板/提示由 ``scaffold``(任务资产)
    注入; 不给 scaffold 时就是一份无领域耦合的默认骨架.
    ``repair_hint`` 非空时把上一轮沙箱真实报错回灌, 让书生自己改对 (通用修 bug).
    ``focus`` 是**本轮可变**的聚焦文本(当前假设 + 本轮实验步骤). ``goal`` 常是恒定
    的研究目标, 若只喂 goal, 提示每轮逐字节相同 → 同一问题反复问、执行输出恒同、
    零新证据 (run47 实测 prompt_len 恒 5379). 带上 focus 后提示随迭代演进.
    """
    template = (scaffold.template if scaffold and scaffold.template
                else DEFAULT_TEMPLATE)
    builtin_ref = ("、".join(scaffold.primitive_names)
                   if scaffold and scaffold.primitive_names
                   else "沙箱内置的数值工具")
    # 超时是跟"代码写错"完全不同的一类失败: 报错里没有语法/形状线索, 只有"执行超时".
    # 若仍套用 NameError/形状 那套对症提示, 书生会继续产出同样重的扫描 → 反复超时
    # (run59/60 实测: 三组都被 900s 超时饿死, 无一产生执行证据). 故单独给一条
    # **可执行的降算力**指令 (命题无关, 只谈算力预算, 不碰科学判断).
    _timeout_like = ("超时" in repair_hint) or ("timeout" in repair_hint.lower())
    if not repair_hint:
        _repair_block = ""
    else:
        if _timeout_like:
            _repair_block = (
                "上一轮该代码在沙箱**执行超时**(算力预算耗尽), 请**大幅削减计算量**后重写:\n"
                "对症改(极常见): (a) 扫描组合数太多 → 砍掉 (kind × w × h × seeds) 的组合数, "
                "例如 w 只取 2-3 个值、h 只取 3-4 个宽度、seeds 降到 1; "
                "(b) 单次拟合太慢 → 降低 mlp_fit 的 maxiter; "
                "(c) 先跑最小可判的配置拿到证据, 再考虑扩大扫描。\n"
                "报错原文:\n" + repair_hint[:800] + "\n"
            )
        else:
            _repair_block = (
                "上一轮该代码在沙箱真实执行报错如下, 请据此改正后重写:\n"
                "对症改(极常见): (a) NameError/未定义名 → 只调用沙箱**内置**的 "
                + builtin_ref + ", 不要自己重写已有工具; "
                "(b) 形状不匹配(matmul/广播/concatenate) → 数组一律 .reshape(-1, 1) 对齐二维; "
                "(c) Generator 没有 randn → 用 rng.standard_normal(n); "
                "(d) assert/raise 中断执行 → 删掉断言直接 return 真实数值。\n"
                "报错原文:\n" + repair_hint[:800] + "\n"
            )
        if prev_code:
            _repair_block += (
                "你上一版失败的代码(请在其基础上做**最小改动**修正它, 保留其余已正确的部分, "
                "不要凭空重写):\n" + prev_code[:2500] + "\n"
            )
    return (
        "你是实验代码作者。请写一段 Python 实验脚本推进下面的研究目标。\n"
        "硬约束: 禁止 IO/网络/读写文件; 不要 try/except、不要 class; "
        "**不要用 assert / raise 判定实验成败**(会中断执行、拿不到任何证据): "
        "把测到的真实数值(含不理想的结果)全部放进 summary/objectives 里 return, 由上层裁决。"
        "单行 <= 88 字符; 每个 for/if/def 后紧跟缩进 4 空格; 结尾必须有 return。\n"
        "cfg 是 dict(可能只含 seed); 读参数请写 cfg.get('x', 默认值), 其余实验参数直接写在代码里。"
        "严禁把 cfg 整体解包成多个变量。\n"
        + (scaffold.hints if scaffold and scaffold.hints else "")
        + (("参考冷启动守卫(软提示):\n" + guard_block + "\n") if guard_block else "")
        + _repair_block
        + "**必须直接采用下面这份模板作为完整脚本骨架**: "
        + template +
        "\n只输出 <code>...</code> 内的**完整可用代码**(即模板 + 你的改动), 不要任何多余文字。\n\n研究目标:\n"
        + goal[:4000]
        + (("\n\n本轮聚焦(只推进下面这一条的具体实验, 不要重复整个研究目标):\n"
            + focus[:1200]) if focus else "")
    )


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
    if isinstance(v, list | tuple):
        return [_to_py(x) for x in v]
    if isinstance(v, int | float | bool | str) or v is None:
        return v
    try:
        json.dumps(v)
        return v
    except Exception:  # noqa: BLE001 — 无法序列化: 如实替换为类名, 不伪造数值
        return f"<{type(v).__name__}:{str(v)[:80]}>"


def strip_abort_statements(code: str) -> str:
    """删掉书生代码里的 assert 语句 (确定性去断言, 命题无关).

    契约要求"把真实数值(含不理想的结果)return 出来由上层裁决, 不许 assert/raise
    中断"; 但模型常无视 (反复出现 ``AssertionError: Anchor failed: ...`` 把整轮
    实验打断、拿不到任何证据). assert 只在失败时中断执行, 摘除它**不改变任何数值
    结果** —— 只是保证不理想的结果也能如实 return. 语法不合法则原样返回, 交沙箱
    如实报语法错误走自修复; 代码里没有 assert 时零开销直返 (不做无谓 AST 往返).
    """
    if not code or "assert" not in code:
        return code
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not any(isinstance(n, ast.Assert) for n in ast.walk(tree)):
        return code

    class _DropAssert(ast.NodeTransformer):
        def visit_Assert(self, node):  # noqa: N802 — ast 访问器命名
            return None

    tree = _DropAssert().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:  # noqa: BLE001 — 反解析失败即原样返回, 不阻塞
        return code


def strip_primitive_redefinitions(code: str,
                                  primitive_names: tuple[str, ...] = ()) -> str:
    """剥掉书生对沙箱注入原语(脚手架 PRIMITIVE_NAMES)的顶层重定义.

    书生若顶层重定义同名函数会**覆盖**注入的正确实现 (反复出现: 它想自己拿参数
    就重写一版, 且切片/形状写错 → 整轮跑不通). 这些原语的定义即契约, 重定义只
    可能引入 bug、不可能带来科学自由度, 故确定性剥除顶层重定义, 保留注入版本.
    只在**顶层**定义时剥除(嵌套闭包同名不影响模块级绑定); 不改变任何数值语义.
    无脚手架/无重定义时原样直返(零开销).
    """
    if not code or not primitive_names or not any(n in code for n in primitive_names):
        return code
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    kept = [
        n for n in tree.body
        if not (
            isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            and n.name in primitive_names
        )
    ]
    if len(kept) == len(tree.body):
        return code
    tree.body = kept
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:  # noqa: BLE001 — 反解析失败即原样返回, 不阻塞
        return code


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
                    imports_whitelist_extra: tuple[str, ...] = (),
                    scaffold: Scaffold | None = None) -> dict:
    """在安全沙箱里执行代码, 返回定义出的命名空间 (run/probe_*).

    ``imports_whitelist_extra``: 域级 import 白名单增量(冷启动守卫的
    imports_whitelist_extra 注入点), 只在本次调用里并入安全白名单 —— 诚实红线:
    仅"放行良性的该域科学计算依赖", 绝不放宽 __import__ 本身或加任何 IO/网络模块.
    ``scaffold``: 任务脚手架; 仅注入它声明的原语(命题无关内核不内置任何领域原语).
    """
    import builtins as _bi

    import numpy as np

    from huginn.security.code_act_sandbox import (
        exec_with_mem_cap,
        make_safe_builtins,
        safe_import,
    )
    code = strip_abort_statements(code)   # 强制"不 assert 中断"契约 (命题无关)
    if scaffold and scaffold.primitive_names:
        # 剥除对脚手架原语的顶层重定义(用注入版), 避免书生重写引入 bug.
        code = strip_primitive_redefinitions(code, scaffold.primitive_names)
    # 只注入平台基座(np) — 领域原语全部来自 scaffold, 内核不内置任何命题专用工具.
    ns: dict = {
        "__builtins__": make_safe_builtins(),
        "np": np,
    }
    if scaffold:
        ns.update(scaffold.primitives)
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
        if isinstance(v, int | float | np.number):
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
        if not isinstance(v, int | float | np.number):
            return f"objectives 值须为数值: {v!r}"
    # 宽容收尾: success 缺失/是 numpy.bool_/任意标量 → 依"跑出数值"推断 True.
    # 这是样式宽容, 不是数值伪造 —— objectives 已全为真实计算数值.
    return None


def _alias_cfg(cfg: dict, extra_aliases: dict | None = None) -> dict:
    """给书生代码一个宽容的 cfg 视图: 常用别名键补齐(值是同一份真实配置的引用).

    只保留**域无关**的别名(约束位置 positions→ti/t_i/t、seeds→n_seeds、
    basis→basis_size —— 它们是 huginn 自带配置域的通用键位, 与具体物理解耦).
    域级物理别名(如断裂域 sigma_0→bridge_ratio)不在通用 harness 里硬编码, 而由
    调用方经 `extra_aliases`(compile_domain_guards 的 cfg_aliases)注入 —— 域专用
    该显式待在域声明里, 不让共享骨架悄悄累积例外(低熵红线).
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
    # 域级别名: 全映射到同一份真实数值, 不新造任何量; 缺失的才补, 显式给的不覆盖.
    for alias, target in (extra_aliases or {}).items():
        out.setdefault(alias, out.get(target))
    return out


def sandbox_run(code: str, cfg: dict, *, mem_cap: int = SAFE_MEM_CAP,
                timeout: float = SAFE_TIMEOUT_S,
                imports_whitelist_extra: tuple[str, ...] = (),
                cfg_aliases: dict | None = None,
                scaffold: Scaffold | None = None) -> tuple[dict | None, str | None]:
    """执行书生写的实验代码: 返回 (结果 dict 或 None, 错误原因或 None).

    ``imports_whitelist_extra``: 冷启动守卫的域级 import 白名单增量, 仅该次调用生效.
    ``cfg_aliases``: 冷启动守卫的域级 cfg 键别名(compile_domain_guards 的 cfg_aliases),
    ``_alias_cfg`` 据此补齐别名, 域专用别名不进通用 harness.
    ``scaffold``: 任务脚手架; 其声明的原语注入沙箱(不给则只有 np 基座).
    """
    if not code.strip():
        return None, "空代码"
    # 无显示主机的 matplotlib: 强制 Agg 后端, 避免 pyplot 因无 DISPLAY 崩 —
    # 数值计算/存图照常走 Agg, 不依赖 GUI 头. (对已设 MPLBACKEND 的调用方生效)
    import os as _os
    _os.environ.setdefault("MPLBACKEND", "Agg")
    cfg = _alias_cfg(cfg, extra_aliases=cfg_aliases)
    try:
        ns = _load_namespace(code, mem_cap,
                             imports_whitelist_extra=imports_whitelist_extra,
                             scaffold=scaffold)
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


def author_probe_specs(code: str, scaffold: Scaffold | None = None) -> list[dict]:
    """把书生代码里的 probe_<name>(cfg) 自动注册为诊断工具 (全权探针面).

    返回 [{tool: {function: {name, description, parameters}}, handle}].
    单条注册失败不影响整体 —— 成文期由工具面统一兜底. ``scaffold`` 用于在注册
    探针时复现与执行侧一致的命名空间(注入同样的原语).
    """
    try:
        ns = _load_namespace(code, scaffold=scaffold)
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
