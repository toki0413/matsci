"""任务脚手架: 神经网络容量扫描 (命题 = NN 泛化行为作为解空间刚性探针).

这是**任务资产**, 不属于 Huginn 平台内核。平台 (huginn.research.code_lab) 只提供
安全沙箱、输入/输出契约与脚手架注入点, 不内置任何领域原语。任何命题都可以照此
自带一份脚手架: 定义自己的 ``TEMPLATE`` / ``HINTS`` / ``PRIMITIVE_NAMES`` 与
``primitives()``, 由调用方经 ``load_scaffold`` 装配。

装配方式 (autoloop / examples):
    from huginn.research.code_lab import load_scaffold
    scaffold = load_scaffold("/abs/path/to/network_rigidity.py")
    build_author_prompt(goal, scaffold=scaffold)   # 提示自带 family 契约
    sandbox_run(code, cfg, scaffold=scaffold)      # 沙箱注入 mlp_fit/capacity_scan

分工: "训练/多起点/宽度扫描/零违规判据/N_c 汇总"已由本文件的 ``capacity_scan``
下沉为已验证原语; 书生只写 ``family(kind, w, seed)`` 这一真正需要科学决策的
约束族函数。
"""
from __future__ import annotations

import numpy as np

NAME = "network_rigidity"

#: 可直接跑通的"容量扫描"骨架: 书生按自己的约束族改 family 即可, 不必重写
#: 参数打包/优化器逻辑 (那正是反复出 bug 的地方).
TEMPLATE = """\
import numpy as np


def family(kind, w, seed):
    # ===== 你只改这个函数: 定义"刚(rigid)/肥(fat)"各自对应的约束族 =====
    # kind: 族名('rigid' 或 'fat'); w: 训练约束点数; seed: 随机种子
    # 返回 dict {'X','y','Xv','yv'}: X/y 是 w 个训练约束点, Xv/yv 是留出约束点。
    #
    # 这是本题的**科学决策部分**, 脚手架**不提供任何现成族** —— 你必须自己推导
    # 两族的解析 ground truth, 再让 capacity_scan 的 N_c(w) 趋势去检验它:
    #   rigid(刚): 低维、由该族唯一确定; 少量样本应能定死它, 从而泛化到留出集
    #              (期望 N_c(w) 有限且小)。
    #   fat(肥):   高维/连续族; 有限样本钉不死, 小网络应泛化失败
    #              (期望 N_c(w) 在扫描宽度内不可达 = trend 'unreachable')。
    # 下面这段占位实现返回**无方差**的退化数据(两族一样, 没有区分度), 脚手架会
    # 直接判为无效族并回灌——**它不是示例答案, 必须整体替换**为你设计的真实族。
    rng = np.random.default_rng(seed)
    X = np.linspace(0.05, 0.95, w).reshape(-1, 1)
    Xv = np.linspace(0.02, 0.98, 200).reshape(-1, 1)
    y = np.zeros((w, 1))                        # TODO: 换成你的 rigid/fat 约束值
    yv = np.zeros((200, 1))                     # TODO: 留出集标签须由同一族生成
    return {'X': X, 'y': y, 'Xv': Xv, 'yv': yv}


def run(cfg):
    # 已由脚手架提供: 训练/多起点/宽度扫描/零违规判据/N_c 汇总都在 capacity_scan 里。
    # **不要改这个函数, 也不要把训练/优化代码写进来** —— 否则会重蹈手搓 numpy 的错。
    seed = int(cfg.get('seed', 0))
    return capacity_scan(family, seed=seed)
"""

#: 分工硬提示: 训练/扫描/判据/汇总已由沙箱内置 capacity_scan 提供, 书生只写
#: family(科学决策部分). 只讲"分工与契约", 不绑定任何具体命题.
HINTS = (
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
PRIMITIVE_NAMES = ("mlp_fit", "mlp_predict", "_as_2d", "poly_basis", "capacity_scan")


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


def _overlap_fraction(X, Xv):
    """Xv 中有多少比例的点与训练点 X 重合 (留出集有效性守卫).

    完全重合(返回 1.0)意味着"留出误差"就是训练误差, 零违规判据失效。用四舍五入
    后的行元组做集合比对, 与维度无关。
    """
    X = _as_2d(X)
    Xv = _as_2d(Xv)
    if Xv.shape[0] == 0 or X.shape[0] == 0:
        return 0.0
    rows = {tuple(np.round(r, 9)) for r in X}
    hit = sum(1 for r in Xv if tuple(np.round(r, 9)) in rows)
    return float(hit) / float(Xv.shape[0])


def _check_label_shapes(X, y, Xv, yv, kind, w):
    """标签形状守卫: y/yv 必须是 (N,1), 行数分别匹配 X/Xv.

    捕获最常见的**静默伪结果**来源 —— 广播 bug. 例如
    ``X[:, 0] + rng.standard_normal((w, 1)) * 0.0``: 左边 (w,) 与右边 (w,1)
    相加会广播成 (w,w) 而非 (w,1). 这种 y 与 (N,1) 预测再广播成 (N,N), 使
    "误差"退化成常数级伪值, 却不抛任何异常, 直接污染整张 N_c 表。故在此显式
    抬高错误, 让上层修复循环把精确形状问题回灌给作者, 而不是产出伪证据。
    """
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
                  widths=(2, 4, 8, 16, 32, 64), seeds=3, fit_starts=2,
                  tr_tol=3e-4, ho_tol=1e-3, seed=0):
    """命题无关的容量扫描脚手架 (harness 侧已验证的数值管线).

    把"训练/多起点/宽度扫描/零违规判据/N_c 汇总"这些最容易手搓出错的样板下沉为
    已验证原语; **科学决策仍由调用方写 family 提供** —— 即"刚/肥"各自对应什么
    约束族(解析 ground truth), 由 family 决定。

    family(kind, w, seed) -> dict 或 tuple:
        - dict  {'X','y','Xv','yv'}: 训练约束点 (w 个) 与留出约束点 (M 个);
        - tuple (X, y, Xv, yv) 或 (X, y): 缺省 Xv/yv 回落到 X/y.
        X/y 一维或二维均可(内部自动升维到 (N,1) 对齐).
    kinds: 要比较的族名(默认 'rigid'/'fat'); ws: 约束点数; widths: 隐藏层宽度;
    seeds: 每个 w 的**数据**随机重抽次数(取各次里最好的留出误差);
    fit_starts: 每个 (kind,w,h) 的**优化器**起点数(默认 3, 抗局部极小);
    tr_tol/ho_tol: 训练/留出误差门限.
    tr_tol 只需"足够小"(默认 3e-4, 仅用于确认网络真的拟合上了, 避免优化器没收敛
    却被记成"可达"); ho_tol=1e-3 才是"零违规"的判据。

    返回 {"success", "summary", "objectives"}:
        summary['rows'][kind_w{w}_h{h}] = {'train_err','heldout_err'}  (全为有限数)
        summary['Nc'][kind][w] = 最小零违规宽度 h 或 None(该 w 在扫描内不可达)
        summary['trend'][kind] = 'flat'|'increasing'|'decreasing'|'mixed'|'inconclusive'
                               |'unreachable'(扫描内无宽度零违规 = 胖的签名)
                               |'invalid_heldout'(留出集与训练集重合, 结果无效)
        summary['anchor'] = 全局留出误差最小的 (kind,w,h) 锚点
        summary['heldout_overlap'][kind_w{w}] = Xv 与 X 的重合比例 (0 才有效)
        summary['warnings'] = 留出集重叠等有效性告警文本
        objectives['neg_heldout_<kind>_w<w>_h<h>'] = -heldout_max_err (越大越好)
    不伪造: 达不到零违规的行如实报其有限留出误差, Nc 记 None, 绝不写 inf.
    """

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
        # 退化族守卫: 标签近常数(无方差) = 该族不含任何约束信息, 拟合"零违规"是
        # 平凡真, 会伪造出"刚性=可泛化"的假结论。占位实现(全 0 标签)正落在这里,
        # 抬错回灌作者, 逼其写出真正的约束族, 而不是产出伪证据(不伪造红线)。
        if float(np.std(y)) < 1e-9 and float(np.std(yv)) < 1e-9:
            raise ValueError(
                "%s w=%d: 标签无方差(近常数) —— 这不是有效约束族, 无法支撑"
                "'刚性/胖'的容量判别. 请把你的 family 实现为真正的解析约束族"
                "(rigid=低维唯一确定族, fat=高维连续族), 两族须有区分度."
                % (kind, int(w))
            )
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
            # 零违规判据形同虚设。
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
                    m = mlp_fit(X, y, int(h), seeds=int(fit_starts),
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
            summary["trend"][kind] = "invalid_heldout"
        elif not vals:
            # 扫描内任何宽度都达不到零违规 —— "胖"(解空间高维/连续族)的判别签名:
            # 有限样本装不下, N_c = ∞。与"inconclusive"(样本太少、测不出趋势)区分开。
            summary["trend"][kind] = "unreachable"
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


def primitives() -> dict:
    """返回注入沙箱的原语表 (键 = 书生代码可直接调用的名字)."""
    return {
        "mlp_fit": mlp_fit,
        "mlp_predict": mlp_predict,
        "_as_2d": _as_2d,
        "poly_basis": poly_basis,
        "capacity_scan": capacity_scan,
    }