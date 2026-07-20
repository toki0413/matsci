"""周期检测 — 判 tool_call 序列卡顿.

治 spec 天花板 "_check_stuck 存全历史判卡顿, 长程任务历史膨胀".

数学动机: Floyd tortoise & hare 双指针 O(1) 空间检测 cycle.
  Floyd 要求序列足够长让 hare 追上 tortoise (≥ μ+2λ+1).
  对 _check_stuck 场景序列 ≤530 步, Floyd 对短 cycle 短序列不敏感.
  实际用周期检测暴力法: O(n²) 但 n≤530 完全可接受, 准确性优先.

  数论背景: 周期检测是 Pollard rho / Brent 的核心操作, 这里用其直觉
  (序列进入 cycle 即卡顿), 但实现务实. 升级路径: 序列超 1万步时换 Floyd.

序列元素 hash 到有限域 Z/pZ (p=2^61-1 Mersenne prime):
  - tool_name + 参数 hash → int → mod p
  - 单射性足够 (冲突率 ~1/p 可忽略)

不做 (YAGNI):
  - Brent 改进 — 暴力法已够快, 序列短
  - KMP 字符串匹配 — 引入复杂度, O(n²) 暴力对 n≤530 无感

天花板: 假设序列元素可 hash. 升级: 不可 hash 时退化为长度判等.
"""
from __future__ import annotations

from typing import Any

# Mersenne prime 2^61 - 1, 数论常用模数, 乘法群 Z/pZ 是有限域
_P = (1 << 61) - 1


def _hash_step(step: Any) -> int:
    """把 tool_call 步骤 hash 到 Z/pZ.

    step 可以是 str (tool_name) 或 dict (含 name + args).
    ponytail: 用 hash() + mod p 已够, 不引入 hashlib. 冲突率 ~1/p.
    """
    if isinstance(step, str):
        s = step
    elif isinstance(step, dict):
        s = step.get("name", "")
        args = step.get("args", {})
        if isinstance(args, dict):
            s += str(sorted(args.items()))
        else:
            s += str(args)
    else:
        s = str(step)
    return hash(s) % _P


def detect_cycle(
    sequence: list[Any],
    *,
    min_cycle_len: int = 2,
    min_repeats: int = 2,
) -> tuple[int, int] | None:
    """检测序列是否进入 cycle (周期性重复).

    返回 (mu, lam):
      mu  = cycle 前缀长度 (进入 cycle 前的步数)
      lam = cycle 周期长度
    返回 None: 序列无 cycle (或太短无法判定).

    min_cycle_len: cycle 长度 < 此值不算卡顿 (单步重复可能是正常重试).
    min_repeats: cycle 至少重复几次才算卡顿.

    算法: 对每个可能的 (mu, lam), 检查 h[mu:mu+lam] 是否重复 min_repeats 次.
    O(n²) 但 n≤530 完全可接受. 准确性优先于 O(1) 空间.
    """
    n = len(sequence)
    if n < 2:
        return None

    h = [_hash_step(s) for s in sequence]

    # 暴力: 枚举 lam (周期) 从 1 起, 对每个 lam 枚举 mu (起点), 检查重复.
    # 找到最小周期 lam_min. 如果 lam_min < min_cycle_len → 不算卡顿 (单步重复是正常重试).
    # 如果最小周期下 repeats < min_repeats → 不算卡顿.
    for lam in range(1, n // min_repeats + 1):
        for mu in range(n - lam * min_repeats + 1):
            pattern = h[mu:mu + lam]
            if all(
                h[mu + r * lam:mu + (r + 1) * lam] == pattern
                for r in range(1, min_repeats)
            ):
                # 找到合法 (mu, lam), 但 lam 太小不算卡顿
                if lam < min_cycle_len:
                    return None
                return (mu, lam)
    return None


def is_stuck(
    sequence: list[Any],
    *,
    min_cycle_len: int = 2,
    min_repeats: int = 2,
) -> bool:
    """便捷接口: 序列是否卡顿 (进入 cycle 且重复够)."""
    return detect_cycle(
        sequence,
        min_cycle_len=min_cycle_len,
        min_repeats=min_repeats,
    ) is not None


# ── selfcheck ──────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 经典 cycle: [a,b,c,a,b,c] → (mu=0, lam=3)
    seq = ["a", "b", "c", "a", "b", "c"]
    r = detect_cycle(seq)
    assert r == (0, 3), f"[a,b,c,a,b,c] 应 (0,3), got {r}"
    print(f"[ok] [a,b,c,a,b,c] → {r}")

    # 2. 有前缀的 cycle: [x,y,a,b,a,b,a,b] → mu=2, lam=2
    seq = ["x", "y", "a", "b", "a", "b", "a", "b"]
    r = detect_cycle(seq)
    assert r == (2, 2), f"got {r}"
    print(f"[ok] [x,y,a,b,a,b,a,b] → {r}")

    # 3. 无 cycle: [a,b,c,d] → None
    seq = ["a", "b", "c", "d"]
    r = detect_cycle(seq)
    assert r is None, f"[a,b,c,d] 应 None, got {r}"
    print(f"[ok] [a,b,c,d] → None")

    # 4. 序列未结束不判: [a,b,c,d,a,b] 长度 6, cycle 长度 4 但只重复 1.5 次
    #    min_repeats=2 → 不算卡顿
    seq = ["a", "b", "c", "d", "a", "b"]
    r = detect_cycle(seq, min_repeats=2)
    assert r is None, f"重复不够应 None, got {r}"
    print(f"[ok] [a,b,c,d,a,b] (重复<2) → None")

    # 5. dict 元素: 相同 name+args 视为同步
    seq = [
        {"name": "code_tool", "args": {"cmd": "train"}},
        {"name": "bash_tool", "args": {"cmd": "ls"}},
        {"name": "code_tool", "args": {"cmd": "train"}},
        {"name": "bash_tool", "args": {"cmd": "ls"}},
    ]
    r = detect_cycle(seq)
    assert r == (0, 2), f"dict cycle 应 (0,2), got {r}"
    print(f"[ok] dict [code,bash,code,bash] → {r}")

    # 6. min_cycle_len 过滤: [a,a,a,a] cycle len=1, 默认 min=2 → None
    seq = ["a", "a", "a", "a"]
    r = detect_cycle(seq)
    assert r is None, f"单步重复应被 min_cycle_len 过滤, got {r}"
    r = detect_cycle(seq, min_cycle_len=1)
    assert r == (0, 1), f"min_cycle_len=1 时应 (0,1), got {r}"
    print(f"[ok] [a,a,a,a] min_cycle_len=2 → None, =1 → (0,1)")

    # 7. is_stuck 便捷接口
    assert is_stuck(["a", "b", "a", "b"]) is True
    assert is_stuck(["a", "b", "c", "d"]) is False
    print("[ok] is_stuck 便捷接口正确")

    print("[cycle_detect] self-check OK (7/7)")
