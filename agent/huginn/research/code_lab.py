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

# 书生写码的默认模板 + 提示词唯一出处: 让 autoloop 的 execute 内建动作与
# examples 的自主循环共用同一份契约, 避免两套 author 提示各自漂移.
# 这是一份**可直接跑通**的"容量扫描"骨架: 多起点 + L-BFGS-B + 每宽度最优
# 训练/留出最大误差 + objectives 表. 书生按自己的约束族改前几行即可, 不必重写
# 参数打包/优化器逻辑 (那正是反复出 bug 的地方). 用 float('inf') 初始化最优值,
# 不用 None —— 避免 "'<' not supported between int and NoneType".
AUTHOR_TEMPLATE = """\
import numpy as np


def family(kind, w, seed):
    # ===== 你只改这个函数: 定义"刚(rigid)/肥(fat)"各自对应的约束族 =====
    # kind: 族名('rigid' 或 'fat'); w: 训练约束点数; seed: 随机种子
    # 返回 dict {'X','y','Xv','yv'}: X/y 是 w 个训练约束点, Xv/yv 是留出约束点。
    # 下面是一个示例(低维正弦族), 请按你的解析 ground truth 替换。
    rng = np.random.default_rng(seed)
    X = np.linspace(0.0, 1.0, w).reshape(-1, 1)
    y = np.sin(np.pi * X)
    Xv = np.linspace(0.0, 1.0, 200).reshape(-1, 1)
    yv = np.sin(np.pi * Xv)
    return {'X': X, 'y': y, 'Xv': Xv, 'yv': yv}


def run(cfg):
    # 已由脚手架提供: 训练/多起点/宽度扫描/零违规判据/N_c 汇总都在 capacity_scan 里。
    # **不要改这个函数, 也不要把训练/优化代码写进来** —— 否则会重蹈手搓 numpy 的错。
    seed = int(cfg.get('seed', 0))
    return capacity_scan(family, seed=seed)
"""

#: 分工硬提示: 训练/扫描/判据/汇总已由沙箱内置 capacity_scan 提供, 书生只写
#: family(科学决策部分). 只讲"分工与契约", 不绑定任何具体命题.
FAMILY_CONTRACT_HINT = (
    "分工(极重要): 沙箱**已内置完整数值管线** capacity_scan(family, ...) —— 训练"
    "(两层 tanh 网络 + L-BFGS-B 多起点)/宽度扫描/零违规判据/N_c 汇总 都由它完成。\n"
    "**你只需实现 family(kind, w, seed) 这一个函数**(这是真正需要你决策的科学部分), "
    "并让 run(cfg) 原样 `return capacity_scan(family, seed=...)`。"
    "**绝对不要**把训练/优化/参数打包/前向写进代码 —— 那正是反复 shape bug 的根源, "
    "已被验证在本环境里必错。\n"
    "- family 返回 dict {'X','y','Xv','yv'}: X/y 是 w 个**训练**约束点, "
    "Xv/yv 是**留出**约束点(未参与训练); 只想给一组时可省略 Xv/yv(回落到 X/y)。\n"
    "- **留出集必须真的留出**: Xv 里的输入点不能与 X 重合(否则'留出误差'=训练误差, "
    "整个 N_c 判据失效, 脚手架会判 trend='invalid_heldout')。Xv 用不同网格/不同采样点。\n"
    "- kind 取 'rigid'/'fat' 两族, 由你定义其解析 ground truth: "
    "刚性 = 低维/唯一解的约束族; 肥 = 高维/连续族的约束族。用 kind 分支返回各自数据。\n"
)

#: 数组纪律硬提示: family 里建数组的常见坑. 只讲"怎么建数组", 不绑定命题.
ARRAY_SHAPE_HINT = (
    "数组纪律: X/y/Xv/yv 用二维 (N,1) —— 建完用 .reshape(-1, 1) 升维; "
    "np.linspace/np.array 直接用。\n"
    "- 随机数: rng = np.random.default_rng(seed); 用 rng.standard_normal(n)/rng.uniform(...)"
    "(Generator **没有** randn/rand/random_sample)。\n"
    "- 建数组只用 np.zeros/np.ones/np.full/np.eye/np.array/np.linspace。\n"
    "- **广播陷阱**: (N,) 与 (N,1) 相加会变成 (N,N)! 例如 "
    "`X[:,0] + rng.standard_normal((w,1))` 得到 (w,w) 而非 (w,1)。"
    "标签务必写成 (N,1): 先 `x = <一维数组>` 再 `x.reshape(-1, 1)`, 不要拿 (N,) 与 (N,1) 混算。"
    "脚手架会检查标签形状, 非 (N,1) 直接报错。\n"
    "- 不要在 family 里做任何矩阵乘/训练; 只负责返回约束点数据。\n"
)


def _as_2d(a):
    """把 1-D 输入变 (N,1); 已是 2-D 则原样 —— 形状对齐原语."""
    import numpy as np
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def mlp_fit(X, y, h, seeds=3, maxiter=20000, seed=0):
    """通用两层 tanh 函数拟合器: L-BFGS-B 多起点, 返回最小训练 MSE 的参数.

    命题无关的数值原语 —— 只封装"参数打包 / L-BFGS-B / 多起点 / 形状对齐"这类
    最容易写错的样板, 不编码任何科学假设(约束族/扫描/判据仍由调用方自定). 反复
    出现的 ``matmul: ... size 5 is different from 1`` 正是手搓这套打包/前向造成,
    故把它下沉为已验证原语.

    X:(N,d) 或 (N,), y:(N,1) 或 (N,); h=隐藏层宽度; seeds=起点数.
    返回 dict(params 列表, h, d, train_mse) —— 可 JSON 序列化, 交给 mlp_predict 前向.
    """
    import numpy as np
    from scipy.optimize import minimize

    X = _as_2d(X)
    y = _as_2d(y)
    h = int(h)
    d = int(X.shape[1])
    n = d * h + h + h + 1

    def unpack(p):
        i = 0
        W1 = p[i:i + d * h].reshape(d, h); i += d * h
        b1 = p[i:i + h]; i += h
        W2 = p[i:i + h].reshape(h, 1); i += h
        b2 = p[i:i + 1].reshape(1, 1)
        return W1, b1, W2, b2

    def loss(p):
        W1, b1, W2, b2 = unpack(p)
        out = np.tanh(X @ W1 + b1) @ W2 + b2
        return float(np.mean((out - y) ** 2))

    best_fun = float('inf')
    best_p = None
    for s in range(int(seeds)):
        rng = np.random.default_rng(int(seed) + 101 * s + 7 * h + d)
        p0 = rng.standard_normal(n) * 0.5
        res = minimize(loss, p0, method='L-BFGS-B',
                       options={'maxiter': int(maxiter)})
        if float(res.fun) < best_fun:
            best_fun = float(res.fun)
            best_p = res.x
    return {'params': best_p.tolist(), 'h': h, 'd': d, 'train_mse': best_fun}


def mlp_predict(model, X):
    """用 mlp_fit 的返回对 X 前向, 返回 (N,1) 预测 —— 形状与 mlp_fit 严格一致."""
    import numpy as np
    X = _as_2d(X)
    p = np.asarray(model['params'], dtype=float)
    h = int(model['h'])
    d = int(model['d'])
    i = 0
    W1 = p[i:i + d * h].reshape(d, h); i += d * h
    b1 = p[i:i + h]; i += h
    W2 = p[i:i + h].reshape(h, 1); i += h
    b2 = p[i:i + 1].reshape(1, 1)
    return np.tanh(X @ W1 + b1) @ W2 + b2


def _overlap_fraction(X, Xv):
    """Xv 中有多少比例的点与训练点 X 重合 (留出集有效性守卫).

    完全重合(返回 1.0)意味着"留出误差"就是训练误差, 零违规判据失效 —— 正是
    run25 伪结果的来源. 用四舍五入后的行元组做集合比对, 与维度无关.
    """
    import numpy as np

    X = _as_2d(X)
    Xv = _as_2d(Xv)
    if Xv.shape[0] == 0:
        return 0.0
    if X.shape[0] == 0:
        return 0.0
    rows = {tuple(np.round(r, 9)) for r in X}
    hit = sum(1 for r in Xv if tuple(np.round(r, 9)) in rows)
    return float(hit) / float(Xv.shape[0])


def _check_label_shapes(X, y, Xv, yv, kind, w):
    """标签形状守卫: y/yv 必须是 (N,1), 行数分别匹配 X/Xv.

    捕获最常见的**静默伪结果**来源 —— 广播 bug. 例如
    ``X[:, 0] + rng.standard_normal((w, 1)) * 0.0``: 左边 (w,) 与右边 (w,1)
    相加会广播成 (w,w) 而非 (w,1). 这种 y 与 (N,1) 预测再广播成 (N,N), 使
    "误差"退化成常数级伪值(如 ~0.5), 却**不抛任何异常**, 直接污染整张 N_c 表
    (run27 的 fat 臂 heldout≈0.51 恒不变, 正是此坑). 故在此显式抬高错误,
    让上层修复循环把精确形状问题回灌给作者, 而不是产出伪证据.
    """
    import numpy as np

    for nm, xx, yy in (("y", X, y), ("yv", Xv, yv)):
        if yy.ndim != 2 or yy.shape[1] != 1:
            raise ValueError(
                "%s w=%d: 标签 %s 形状应为 (N,1), 实为 %s —— 多半是广播 bug "
                "( (N,) 与 (N,1) 相加会变成 (N,N) ); 请用 .reshape(-1, 1) 逐列构造标签."
                % (kind, int(w), nm, tuple(yy.shape))
            )
        if yy.shape[0] != xx.shape[0]:
            raise ValueError(
                "%s w=%d: 标签 %s 行数 %d 与输入 %s 行数 %d 不一致."
                % (kind, int(w), nm, yy.shape[0],
                   "X" if nm == "y" else "Xv", xx.shape[0])
            )


def capacity_scan(family, kinds=("rigid", "fat"), ws=(5, 10, 20),
                  widths=(2, 4, 8, 16, 32, 64), seeds=3,
                  tr_tol=3e-4, ho_tol=1e-3, seed=0):
    """命题无关的容量扫描脚手架 (harness 侧已验证的数值管线).

    把"训练/多起点/宽度扫描/零违规判据/N_c 汇总"这些最容易手搓出错的样板下沉为
    已验证原语; **科学决策仍由调用方写 family 提供** —— 即"刚/肥"各自对应什么
    约束族(解析 ground truth), 由 family 决定. 这样书生只写他真正该决策的部分.

    family(kind, w, seed) -> dict 或 tuple:
        - dict  {'X','y','Xv','yv'}: 训练约束点 (w 个) 与留出约束点 (M 个);
        - tuple (X, y, Xv, yv) 或 (X, y): 缺省 Xv/yv 回落到 X/y.
        X/y 一维或二维均可(内部自动升维到 (N,1) 对齐).
    kinds: 要比较的族名(默认 'rigid'/'fat'); ws: 约束点数; widths: 隐藏层宽度;
    seeds: 每个 (kind,w,h) 的独立起点数; tr_tol/ho_tol: 训练/留出误差门限.
    tr_tol 只需"足够小"(默认 3e-4, 仅用于确认网络真的拟合上了, 避免优化器没收敛
    却被记成"可达"); ho_tol=1e-3 才是"零违规"的判据, 过严的 tr_tol 会把本来可达
    的宽度误判成 None(伪不可达), 故刻意留松.

    返回 {"success", "summary", "objectives"}:
        summary['rows'][kind_w{w}_h{h}] = {'train_err','heldout_err'}  (全为有限数)
        summary['Nc'][kind][w] = 最小零违规宽度 h 或 None(该 w 在扫描内不可达)
        summary['trend'][kind] = 'flat'|'increasing'|'decreasing'|'mixed'|'inconclusive'
                               |'invalid_heldout'(留出集与训练集重合, 结果无效)
        summary['anchor'] = 全局留出误差最小的 (kind,w,h) 锚点 —— 证明零违规可达
        summary['heldout_overlap'][kind_w{w}] = Xv 与 X 的重合比例 (0 才有效)
        summary['warnings'] = 留出集重叠等有效性告警文本
        objectives['neg_heldout_<kind>_w<w>_h<h>'] = -heldout_max_err (越大越好)
    不伪造: 达不到零违规的行如实报其有限留出误差, Nc 记 None, 绝不写 inf.
    """
    import numpy as np

    def _one(kind, w, s):
        out = family(kind, int(w), int(s))
        if isinstance(out, dict):
            X, y = out["X"], out["y"]
            Xv, yv = out.get("Xv", X), out.get("yv", y)
        elif isinstance(out, (tuple, list)) and len(out) >= 2:
            X, y = out[0], out[1]
            Xv = out[2] if len(out) > 2 else X
            yv = out[3] if len(out) > 3 else y
        else:
            raise ValueError("family 须返回 dict(X,y,Xv,yv) 或 (X,y[,Xv,yv])")
        X, y, Xv, yv = _as_2d(X), _as_2d(y), _as_2d(Xv), _as_2d(yv)
        _check_label_shapes(X, y, Xv, yv, kind, w)
        return X, y, Xv, yv

    summary = {"rows": {}, "Nc": {}, "trend": {}, "anchor": None,
               "heldout_overlap": {}, "warnings": []}
    objectives = {}
    for kind in kinds:
        nc: dict = {}
        _fully_overlapping = False
        for w in ws:
            datasets = [_one(kind, w, s) for s in range(int(seeds))]
            # 留出集有效性守卫: 若 Xv 与训练点 X 重合, "留出误差"其实是训练误差,
            # 零违规判据形同虚设 —— run25 的伪结果(heldout==train)正源于此.
            _ov = max(_overlap_fraction(X, Xv) for (X, y, Xv, yv) in datasets)
            if _ov > 0.0:
                summary["heldout_overlap"]["%s_w%d" % (kind, int(w))] = round(_ov, 4)
                summary["warnings"].append(
                    "留出集与训练集重叠%s: %s w=%d (留出误差无效)"
                    % ("(完全重合)" if _ov >= 1.0 else "", kind, int(w))
                )
                if _ov >= 1.0:
                    _fully_overlapping = True
            best_h = None
            for h in widths:
                ho_best = float("inf")
                tr_best = float("inf")
                for s, (X, y, Xv, yv) in enumerate(datasets):
                    m = mlp_fit(X, y, int(h), seeds=1,
                                seed=int(seed) + 1000 * int(w) + 10 * int(h) + s)
                    tr = float(np.max(np.abs(mlp_predict(m, X) - y)))
                    ho = float(np.max(np.abs(mlp_predict(m, Xv) - yv)))
                    if ho < ho_best:
                        ho_best, tr_best = ho, tr
                key = "%s_w%d_h%d" % (kind, int(w), int(h))
                summary["rows"][key] = {"train_err": tr_best, "heldout_err": ho_best}
                objectives["neg_heldout_" + key] = -ho_best
                if best_h is None and ho_best <= ho_tol and tr_best <= tr_tol:
                    best_h = int(h)
                if summary["anchor"] is None or ho_best < summary["anchor"]["heldout_err"]:
                    summary["anchor"] = {"kind": kind, "w": int(w), "h": int(h),
                                         "train_err": tr_best, "heldout_err": ho_best}
            nc[int(w)] = best_h
        summary["Nc"][kind] = nc
        vals = [v for v in nc.values() if v is not None]
        if _fully_overlapping:
            # 留出集 == 训练集: 所谓零违规只是插值训练点, 趋势无科学意义.
            summary["trend"][kind] = "invalid_heldout"
        elif len(vals) < 2:
            summary["trend"][kind] = "inconclusive"
        elif len(set(vals)) == 1:
            summary["trend"][kind] = "flat"
        elif all(b >= a for a, b in zip(vals, vals[1:])):
            summary["trend"][kind] = "increasing"
        elif all(b <= a for a, b in zip(vals, vals[1:])):
            summary["trend"][kind] = "decreasing"
        else:
            summary["trend"][kind] = "mixed"
    return {"success": True, "summary": summary, "objectives": objectives}


def build_author_prompt(goal: str, *, guard_block: str = "",
                        template: str = AUTHOR_TEMPLATE,
                        repair_hint: str = "", prev_code: str = "") -> str:
    """构造"让书生在 Code Lab 亲手写一轮实验"的作者提示 (命题无关).

    单一出处: autoloop 的 execute 内建动作与 examples 自主循环都调本函数, 契约
    (纯 numpy、<=30 行、返回 objectives) 只维护一份. ``goal`` 由外部传入 —— 只给
    方向, 不绑定任何具体命题; ``guard_block`` 是可选的冷启动守卫软提示.
    ``repair_hint`` 非空时把上一轮沙箱真实报错回灌, 让书生自己改对 (通用修 bug,
    不绑定任何命题).
    """
    return (
        "你是实验代码作者。请写一段 Python 实验脚本推进下面的研究目标。\n"
        "硬约束: 禁止 IO/网络/读写文件; 不要 try/except、不要 class; "
        "**不要用 assert / raise 判定实验成败**(会中断执行、拿不到任何证据): "
        "把测到的真实数值(含不理想的结果)全部放进 summary/objectives 里 return, 由上层裁决。"
        "单行 <= 88 字符; 每个 for/if/def 后紧跟缩进 4 空格; 结尾必须有 return。\n"
        "cfg 是 dict(可能只含 seed); 读参数请写 cfg.get('x', 默认值), 其余实验参数直接写在代码里。"
        "严禁把 cfg 整体解包成多个变量。\n"
        + FAMILY_CONTRACT_HINT
        + ARRAY_SHAPE_HINT
        + (("参考冷启动守卫(软提示):\n" + guard_block + "\n") if guard_block else "")
        + (("上一轮该代码在沙箱真实执行报错如下, 请据此改正后重写:\n"
           "对症改(极常见): (a) NameError/未定义名 → 只调用沙箱**内置**的 family/capacity_scan/"
           "mlp_fit/mlp_predict, 不要自己重写训练管线; "
           "(b) 形状不匹配(matmul/广播/concatenate) → family 返回的 X/y/Xv/yv 一律 .reshape(-1, 1); "
           "(c) Generator 没有 randn → 用 rng.standard_normal(n); "
           "(d) assert/raise 中断执行 → 删掉断言直接 return 真实数值。\n"
           "报错原文:\n" + repair_hint[:800] + "\n"
           + (("你上一版失败的代码(请在其基础上做**最小改动**修正它, 保留其余已正确的部分, "
               "不要凭空重写):\n" + prev_code[:2500] + "\n") if prev_code else "")
           if repair_hint else ""))
        + "**必须直接采用下面这份已跑通的模板作为完整脚本骨架**: "
        "你**只改 family(kind, w, seed) 这一个函数**(定义 rigid/fat 各自的约束族数据), "
        "run(cfg) 原样 `return capacity_scan(family, seed=...)` 不改。\n"
        + template +
        "\n只输出 <code>...</code> 内的**完整可用代码**(即模板 + 你对 family 的改动), 不要任何多余文字。\n\n研究目标:\n"
        + goal[:4000]
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


#: 沙箱注入的命题无关原语名. 书生若顶层重定义同名函数会**覆盖**注入的正确实现
#: (反复出现: 它想自己拿参数就重写一版, 且切片/形状写错 → 整轮跑不通). 这些
#: 原语的定义即契约, 重定义只可能引入 bug、不可能带来科学自由度(约束族/扫描/
#: 判据都在 run() 里, 与原语无关), 故确定性剥除顶层重定义, 保留注入版本.
BUILTIN_PRIMITIVE_NAMES = ("mlp_fit", "mlp_predict", "_as_2d", "capacity_scan")


def strip_primitive_redefinitions(code: str) -> str:
    """剥掉书生对沙箱内置原语(mlp_fit/mlp_predict/_as_2d)的顶层重定义.

    只在**顶层**定义时剥除(嵌套闭包同名不影响模块级绑定); 摘除的是"对已验证
    原语的重复实现", 不改变书生对约束族/扫描/判据的任何决策, 也不改任何数值
    语义. 语法不合法/无重定义时原样直返(零开销).
    """
    if not code or not any(n in code for n in BUILTIN_PRIMITIVE_NAMES):
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
            and n.name in BUILTIN_PRIMITIVE_NAMES
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
                    imports_whitelist_extra: tuple[str, ...] = ()) -> dict:
    """在安全沙箱里执行代码, 返回定义出的命名空间 (run/probe_*).

    ``imports_whitelist_extra``: 域级 import 白名单增量(冷启动守卫的
    imports_whitelist_extra 注入点), 只在本次调用里并入安全白名单 —— 诚实红线:
    仅"放行良性的该域科学计算依赖", 绝不放宽 __import__ 本身或加任何 IO/网络模块.
    """
    import builtins as _bi

    import numpy as np

    from huginn.security.code_act_sandbox import (
        exec_with_mem_cap,
        make_safe_builtins,
        safe_import,
    )
    code = strip_abort_statements(code)   # 强制"不 assert 中断"契约 (命题无关)
    code = strip_primitive_redefinitions(code)  # 剥除对内置原语的重定义(用注入版)
    # 注入命题无关的数值原语: 参数打包/L-BFGS-B/多起点/形状对齐已下沉, 书生
    # 直接 mlp_fit/mlp_predict 即可, 不必手搓这套样板(反复 shape bug 的根源).
    ns: dict = {
        "__builtins__": make_safe_builtins(),
        "np": np,
        "mlp_fit": mlp_fit,
        "mlp_predict": mlp_predict,
        "_as_2d": _as_2d,
        "capacity_scan": capacity_scan,
    }
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
                cfg_aliases: dict | None = None) -> tuple[dict | None, str | None]:
    """执行书生写的实验代码: 返回 (结果 dict 或 None, 错误原因或 None).

    ``imports_whitelist_extra``: 冷启动守卫的域级 import 白名单增量, 仅该次调用生效.
    ``cfg_aliases``: 冷启动守卫的域级 cfg 键别名(compile_domain_guards 的 cfg_aliases),
    ``_alias_cfg`` 据此补齐别名, 域专用别名不进通用 harness.
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
