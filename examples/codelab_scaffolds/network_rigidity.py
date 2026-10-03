"""任务脚手架: 神经网络容量扫描 (命题 = NN 泛化行为作为解空间刚性探针).

这是**任务资产**, 不属于 Huginn 平台内核。平台 (huginn.research.code_lab) 只提供
安全沙箱、输入/输出契约与脚手架注入点, 不内置任何领域原语。任何命题都可以照此
自带一份脚手架: 定义自己的 ``TEMPLATE`` / ``HINTS`` / ``PRIMITIVE_NAMES`` 与
``primitives()``, 由调用方经 ``load_scaffold`` 装配。

装配方式 (autoloop / examples):
    from huginn.research.code_lab import load_scaffold
    scaffold = load_scaffold("/abs/path/to/network_rigidity.py")
    build_author_prompt(goal, scaffold=scaffold)   # 提示自带 family 契约
    sandbox_run(code, cfg, scaffold=scaffold)      # 沙箱注入 mlp_fit/mlp_predict

分工 (v11 彻底解绑): 本文件**只提供命题无关的基础数值工具**(mlp_fit/mlp_predict/
poly_basis/_as_2d), 不再托管任何数值管线。**扫描参数、训练、零违规判据、N_c 汇总
全部由书生自己实现** —— 早期版本把"扫描/判据/聚合"下沉为 ``capacity_scan`` 原语,
结果科学决策被架空: 每轮 ``run`` 都返回同 36 个 objectives, 输入信息恒定, 循环
空转 (run47/48 实测)。解绑后, 书生的实验设计空间不再被提前解掉。
"""
from __future__ import annotations

import numpy as np

NAME = "network_rigidity"

#: 命题的**唯一硬性口径**(不可更改): 零违规 = 留出点最大绝对误差 <= 此阈值.
#: 命题允许"族/扫描/脚本由书生自定", 但把判据明确钉死为 1e-3 —— 早期版本把判据
#: 也留给书生每轮重写, 结果同一命题不同轮用不同比较(甚至拿训练误差当留出误差),
#: 报出 N_c=8 而留出误差实为 1.7e-3 > 1e-3 的自相矛盾数值 (run66 实测). 故把
#: 不可更改的口径下沉为注入原语, 与自由部分(族设计/扫描网格)解耦.
CRITERION_MAX_ERR = 1e-3

#: 极简骨架: 只固定 family 契约与 run(cfg) 的输入/输出形状, **不含任何扫描/
#: 判据/聚合实现** —— 那正是本轮实验需要书生自己设计的科学内容. 占位实现是
#: 一份"必须整体替换"的退化数据(两族无区分度), 用于逼迫书生写出真实约束族.
TEMPLATE = """\
import numpy as np


def family(kind, w, seed):
    # ===== 你只改这个函数: 定义"刚(rigid)/肥(fat)"各自对应的约束族 =====
    # kind: 族名('rigid' 或 'fat'); w: 训练约束点数; seed: 随机种子
    # 返回 dict {'X','y','Xv','yv'}: X/y 是 w 个训练约束点, Xv/yv 是留出约束点。
    #
    # 这是本题的**科学决策部分**, 脚手架**不提供任何现成族** —— 你必须自己推导
    # 两族的解析 ground truth:
    #   rigid(刚): 低维、由该族唯一确定; 少量样本应能定死它, 从而泛化到留出集
    #              (期望 N_c(w) 有限且小)。
    #   fat(肥):   高维/连续族; 有限样本钉不死, 小网络应泛化失败
    #              (期望 N_c(w) 在扫描宽度内不可达)。
    # 下面这段占位实现返回**无方差**的退化数据(两族一样, 没有区分度), 它不是
    # 示例答案, **必须整体替换**为你设计的真实族。
    rng = np.random.default_rng(seed)
    X = np.linspace(0.05, 0.95, w).reshape(-1, 1)
    Xv = np.linspace(0.02, 0.98, 200).reshape(-1, 1)
    y = np.zeros((w, 1))                        # TODO: 换成你的 rigid/fat 约束值
    yv = np.zeros((200, 1))                     # TODO: 留出集标签须由同一族生成
    return {'X': X, 'y': y, 'Xv': Xv, 'yv': yv}


def run(cfg):
    # ===== 你必须自己实现完整实验逻辑 (无现成管线可调) =====
    # 1. 选扫描参数 (约束点数 w、隐藏层宽度 h 的集合等);
    # 2. 对每个 (kind, w, h): 调 family 取数据, 用 mlp_fit(X, y, h, seeds=3)
    #    训练, 再用 mlp_predict(model, Xv) 评估留出误差;
    # 3. 判定"零违规"**必须**用注入原语 zero_violation(留出最大绝对误差)(口径固定
    #    1e-3, 不要自己写比较、不要拿训练误差冒充), 得到每个 (kind, w) 的最小
    #    零违规宽度 N_c (扫描内都达不到则记 None);
    # 4. 把数值结果组织成 summary, 并把可比较的数值放进 objectives。
    seed = int(cfg.get('seed', 0))
    # ... 你的实验代码 ...
    return {'success': True, 'summary': {'note': '待实现'},
            'objectives': {'score': 0.0}}
"""

#: 分工硬提示: 沙箱只给基础数值工具(训练/预测/形状/幂基), **扫描/判据/汇总由
#: 书生自己实现**. 只讲"契约与坑", 不绑定任何具体命题.
HINTS = (
    "分工(极重要): 沙箱**只内置基础数值工具**: mlp_fit(训练网络)/mlp_predict(预测)/"
    "poly_basis(多项式基)/_as_2d(形状对齐), 外加**唯一硬性口径**的判据原语 "
    "zero_violation(留出最大绝对误差 <= 1e-3 则 True)。**没有现成的扫描/汇总管线** —— "
    "你必须自己实现完整实验逻辑: 选扫描参数、训练网络、评估留出误差、判定零违规、"
    "求每个 (kind,w) 的最小零违规宽度 N_c, 并把数值结果放进 summary/objectives。\n"
    "- 零违规 = 留出约束点上的最大绝对误差 <= 1e-3 (口径固定, 不可更改)。判定**必须**"
    "调用注入原语 zero_violation(held_out_max_abs_err), **不要自己写比较** —— 自己写"
    "容易出现阈值写错/拿训练误差冒充留出误差, 使同一命题不同轮 N_c 互斥(不可采信)。\n"
    "- 刚性 = 扫描范围内存在较小的有限 N_c(且不随 w 增长); "
    "胖 = 扫描上限内任何宽度都达不到零违规(N_c 不可达)。判别是二值的。\n"
    "- 所有报告数值必须是有限数 (禁止 inf/nan/None)。\n"
    "- 训练用 mlp_fit(X, y, h, seeds=3)(两层 tanh 网络 + L-BFGS-B 多起点), "
    "预测用 mlp_predict(model, Xv); 返回的 model 可直接复用, 不要自己手搓前向。\n"
    "- family 返回 dict {'X','y','Xv','yv'}: X/y 是 w 个**训练**约束点, "
    "Xv/yv 是**留出**约束点(未参与训练); 只想给一组时可省略 Xv/yv(回落到 X/y)。\n"
    "- **留出集必须真的留出**: Xv 里的输入点不能与 X 重合, 否则'留出误差'=训练误差, "
    "N_c 判据失效。Xv 用不同网格/不同采样点。\n"
    "- kind 取 'rigid'/'fat' 两族, 由你定义其解析 ground truth: "
    "刚性 = 低维/唯一解的约束族; 肥 = 高维/连续族的约束族。用 kind 分支返回各自数据。\n"
    "- **只有最终的 X/y/Xv/yv 才需要是 (N,1)**(建完用 .reshape(-1, 1)); "
    "**幂次/系数这类中间量保持一维 (n,)**: 如 `coeffs = rng.standard_normal(n)` 后 "
    "`poly_basis(X, n) @ coeffs`。**不要把 np.arange(n)/coeffs 再 reshape 成 (n,1)** —— "
    "(w,1) 与 (n,1) 做幂/乘会广播失败(报 shapes (w,1)/(n,1)), 这是本环境反复踩的坑。\n"
    "- 多项式/幂基直接用原语 `poly_basis(X, n)`(返回 (N,n), 列是 1,x,...,x^(n-1)), "
    "再 `@ coeffs` 得 (N,1); **不要手搓** `x ** np.arange(n)` 之类的广播。\n"
    "- 目标量级必须 O(1): 构造完把 y 与 yv 除以**同一个**尺度常数"
    "(可在固定稠密网格上算 std, 与 w 无关更稳)再报误差; **不要各自归一**。\n"
)

#: 沙箱注入的原语名. 书生若顶层重定义同名函数会覆盖注入的正确实现, 故会确定性剥除.
#: zero_violation 是**唯一硬性口径**的实现(断言性质), 一并注入并禁止重写 ——
#: 保证 N_c 判据跨轮一致、不被书生改口径或误用训练误差.
PRIMITIVE_NAMES = ("mlp_fit", "mlp_predict", "_as_2d", "poly_basis",
                   "zero_violation")


def _as_2d(a):
    """把 1-D 输入变 (N,1); 已是 2-D 则原样 —— 形状对齐原语."""
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def poly_basis(X, n):
    """多项式幂基 [1, x, x^2, ..., x^(n-1)]: X 一维或 (N,d)(取第 0 列) → (N, n).

    命题无关的形状原语: 内部把 x 压成一维再外积, 从根上避免
    ``(N,1) ** (n,1)`` 的广播陷阱 (反复出现的 shapes (w,1)/(n,1) 报错)。
    """
    X = _as_2d(X)
    x = X[:, 0]                                # (N,)
    powers = np.arange(int(n))                 # (n,)
    return x[:, None] ** powers[None, :]       # (N,1) ** (1,n) → (N,n)


def zero_violation(max_abs_err, tol=CRITERION_MAX_ERR):
    """零违规判据原语: 留出点最大绝对误差 <= tol 即零违规 (命题硬性口径).

    **唯一硬性口径的单一实现** —— 命题只允许书生自定"族/扫描/脚本", 判据固定为
    1e-3. 判 N_c 时必须用本原语, 不要自己写比较: 早期把判据留给书生每轮重写,
    出现"拿训练误差当留出误差""阈值写成 1e-2/1e-1"等, 同一命题不同轮 N_c 互斥
    (run66 实测: 报 N_c=8 而留出误差 1.7e-3 > 1e-3 的自相矛盾). 本原语把该口径
    钉死, 使 N_c(w) 跨轮可比.

    max_abs_err: 留出约束点上的**最大绝对误差**(标量); tol: 阈值(默认 1e-3).
    返回 bool. 非有限值(inf/nan)一律不算零违规.
    """
    import math
    try:
        e = float(max_abs_err)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(e):
        return False
    return e <= float(tol)


def mlp_fit(X, y, h, seeds=3, maxiter=3000, maxfun=30000, seed=0):
    """通用两层 tanh 函数拟合器: L-BFGS-B(解析梯度) 多起点, 返回最小训练 MSE 的参数.

    命题无关的数值原语 —— 只封装"参数打包 / L-BFGS-B / 多起点 / 形状对齐 / 解析
    梯度"这类最容易写错的样板, 不编码任何科学假设(约束族/扫描/判据仍由调用方自定).

    X:(N,d) 或 (N,), y:(N,1) 或 (N,); h=隐藏层宽度; seeds=起点数.
    返回 dict(params 列表, h, d, train_mse, scaler) —— 可 JSON 序列化.
    """
    from scipy.optimize import minimize

    X = _as_2d(X)
    y = _as_2d(y)
    h = int(h)
    d = int(X.shape[1])
    N = int(X.shape[0])
    n = d * h + h + h + 1

    def unpack(p):
        i = 0
        W1 = p[i:i + d * h].reshape(d, h); i += d * h
        b1 = p[i:i + h]; i += h
        W2 = p[i:i + h].reshape(h, 1); i += h
        b2 = p[i:i + 1].reshape(1, 1)
        return W1, b1, W2, b2

    # 输入/输出标定 (缩放): 使 tanh 网络能逼近一般光滑函数, 并让 L-BFGS-B 不被
    # 目标绝对量级主导. 预测侧必须原样反转(mlp_predict 里同处理), 保证返回的
    # 误差是原始尺度上的真实误差.
    Xm, Xs = X.mean(0), X.std(0) + 1e-9
    Xn = (X - Xm) / Xs
    ym, ys = float(y.mean()), float(y.std()) + 1e-9
    yn = (y - ym) / ys

    def loss(p):
        W1, b1, W2, b2 = unpack(p)
        out = np.tanh(Xn @ W1 + b1) @ W2 + b2
        return float(np.mean((out - yn) ** 2))

    def grad(p):
        """解析梯度 (反向传播): 每次迭代 1 次求值, 替代有限差分的 (n+1) 次."""
        W1, b1, W2, b2 = unpack(p)
        H = np.tanh(Xn @ W1 + b1)                 # (N,h)
        g = 2.0 * (H @ W2 + b2 - yn) / N          # dL/dout (N,1)
        gW2 = H.T @ g                             # (h,1)
        gb2 = g.sum(0)                            # (1,)
        gA = (g @ W2.T) * (1.0 - H ** 2)          # dL/dA (N,h)
        gW1 = Xn.T @ gA                           # (d,h)
        gb1 = gA.sum(0)                           # (h,)
        o = np.empty(n)
        i = 0
        o[i:i + d * h] = gW1.reshape(-1); i += d * h
        o[i:i + h] = gb1; i += h
        o[i:i + h] = gW2.reshape(-1); i += h
        o[i:i + 1] = gb2
        return o

    best_fun = float('inf')
    best_p = None
    # 容差: 默认 gtol=1e-5 会让小网络在 ~1e-4 量级"假收敛"停机, 把本可达的刚性族
    # 误判成不可达(实测: 收紧到 1e-10 后 h=2 即可零违规). ftol/gtol 取"够紧但不极端"
    # —— 极端的 1e-15 只会让不可拟合的胖族跑满迭代, 白白拖垮整张扫描(超过沙箱超时).
    opts = {'maxiter': int(maxiter), 'ftol': 1e-14, 'gtol': 1e-10,
            'maxfun': int(maxfun)}
    for s in range(int(seeds)):
        rng = np.random.default_rng(int(seed) + 101 * s + 7 * h + d)
        p0 = rng.standard_normal(n) * 0.5
        res = minimize(loss, p0, jac=grad, method='L-BFGS-B', options=opts)
        if float(res.fun) < best_fun:
            best_fun = float(res.fun)
            best_p = res.x
    return {'params': best_p.tolist(), 'h': h, 'd': d, 'train_mse': best_fun,
            'scaler': {'Xm': Xm.tolist(), 'Xs': Xs.tolist(),
                       'ym': ym, 'ys': ys}}


def mlp_predict(model, X):
    """用 mlp_fit 的返回对 X 前向, 返回 (N,1) 预测 —— 形状与 mlp_fit 严格一致."""
    X = _as_2d(X)
    p = np.asarray(model['params'], dtype=float)
    h = int(model['h'])
    d = int(model['d'])
    sc = model.get('scaler')
    if sc is not None:   # 还原 mlp_fit 的输入/输出标定, 使误差落在原始尺度
        X = (X - np.asarray(sc['Xm'], dtype=float)) / np.asarray(sc['Xs'], dtype=float)
    i = 0
    W1 = p[i:i + d * h].reshape(d, h); i += d * h
    b1 = p[i:i + h]; i += h
    W2 = p[i:i + h].reshape(h, 1); i += h
    b2 = p[i:i + 1].reshape(1, 1)
    out = np.tanh(X @ W1 + b1) @ W2 + b2
    if sc is not None:
        out = out * float(sc['ys']) + float(sc['ym'])
    return out


def primitives() -> dict:
    """返回注入沙箱的原语表 (键 = 书生代码可直接调用的名字).

    v11 彻底解绑: 只注入**基础数值工具**(训练/预测/形状/幂基)与**唯一硬性口径**的
    判据原语 zero_violation; 不注入任何扫描/汇总管线 —— 那些是书生本轮要自己设计的
    科学内容. 判据不是"科学内容"而是命题钉死的口径, 故由脚手架提供, 保证跨轮一致.
    """
    return {
        "mlp_fit": mlp_fit,
        "mlp_predict": mlp_predict,
        "_as_2d": _as_2d,
        "poly_basis": poly_basis,
        "zero_violation": zero_violation,
    }