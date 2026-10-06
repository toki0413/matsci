"""离线端到端: 用**书生会写的那种代码**走真实 sandbox_run 路径 (脚手架注入 +
zero_violation 判据 + 隔离子进程), 确认 rigid/fat 两族可区分 (无 LLM, 确定性).

判据: rigid 在小宽度即零违规 (N_c 有限且不随 w 增长); fat 在扫描上限内不可达.
若此脚本打印 rigid=[2,2,2] / fat=[None,None,None], 则"探针判别性"修复在真实执行
路径上成立, 与 _scratch/probe_discrim.py (直接调原语) 互为交叉验证.
"""
from huginn.research.code_lab import load_scaffold, sandbox_run

SCAFFOLD = "/workspace/examples/codelab_scaffolds/network_rigidity.py"

# 书生风格的实验代码 (确定性 ground truth: 训练/留出共用同一组参数)
CODE = r'''
import numpy as np


def family(kind, w, seed):
    rng = np.random.default_rng(seed)
    X = np.linspace(0.05, 0.95, w).reshape(-1, 1)
    Xv = np.linspace(0.02, 0.98, 200).reshape(-1, 1)
    x = X[:, 0]
    xv = Xv[:, 0]
    if kind == 'rigid':
        # 低维唯一解: 固定解析函数 sin(pi x), 少量样本即可定死
        y = np.sin(np.pi * x)[:, None]
        yv = np.sin(np.pi * xv)[:, None]
    else:
        # 高维连续族: 15 项多项式, 系数只抽一次, 训练/留出共用
        c = rng.standard_normal(15)
        y = poly_basis(X, 15) @ c[:, None]
        yv = poly_basis(Xv, 15) @ c[:, None]
    s = float(np.std(y)) + 1e-9
    return {'X': X, 'y': y / s, 'Xv': Xv, 'yv': yv / s}


def run(cfg):
    widths = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    seed = int(cfg.get('seed', 0))
    objs = {}
    rows = {}
    for kind in ('rigid', 'fat'):
        for w in (10, 20, 40):
            d = family(kind, w, seed)
            Nc = None
            err = 999.0
            for h in widths:
                m = mlp_fit(d['X'], d['y'], h, seeds=3, seed=0)
                err = float(np.max(np.abs(mlp_predict(m, d['Xv']) - d['yv'])))
                if zero_violation(err):
                    Nc = h
                    break
            rows['%s_w%d' % (kind, w)] = {'Nc': Nc, 'err': err}
            objs['%s_w%d_nc' % (kind, w)] = 999.0 if Nc is None else float(Nc)
    return {'success': True, 'summary': {'rows': rows}, 'objectives': objs}
'''


def main():
    sc = load_scaffold(SCAFFOLD)
    res, reason = sandbox_run(CODE, {"seed": 0}, timeout=120, scaffold=sc)
    assert res is not None, f"sandbox_run 失败: {reason}"
    rows = res["summary"]["rows"]
    rigid = [rows[f"rigid_w{w}"]["Nc"] for w in (10, 20, 40)]
    fat = [rows[f"fat_w{w}"]["Nc"] for w in (10, 20, 40)]
    print("rigid N_c =", rigid)
    print("fat   N_c =", fat)
    ok = all(n is not None for n in rigid) and all(n is None for n in fat)
    print("VERDICT:", "DISCRIMINATIVE (PASS)" if ok else "NOT discriminative (FAIL)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()