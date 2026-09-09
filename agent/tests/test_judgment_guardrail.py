"""测试: 判断层·分级护栏 (judgment_guardrail).

核心不变量 (模型无关, 与具体 LLM 的强弱解耦):
  1. strictness=0 → 恒 [] (只真伪硬门禁, 零行为变化) —— agent 先信最强假设.
  2. 三档逐层包含: hints[2] ⊇ hints[1] ⊇ hints[0].
  3. 缺对照(候选自变量)/不确定度检测是可证伪的: 换 trace 的重复信号, 判定翻转.
  4. 纯函数、确定性: 同输入必同输出.
  5. 护栏只生成"软提示文本", 不参与任何可证伪硬判定(verify) —— 强模型不被锁.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 文件级加载, 绕过 huginn/__init__ 的重依赖副作用(同 program.grounding_verifier 做法).
_SRC = _ROOT / "huginn" / "research" / "judgment_guardrail.py"
_spec = importlib.util.spec_from_file_location("_judgment_guardrail", str(_SRC))
_jg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_jg)  # noqa: E402

build_judgment_hints = _jg.build_judgment_hints
hint_block = _jg.hint_block

GOAL = (
    "以神经网络容量为探针测 bootstrap 解空间刚性: 饱和判据应依赖基因 N(样本量) "
    "还是 eps(精度截止)? 约束数量如何决定解族维数?"
)

# 单点 scalar trace(无重复/方差信号) —— 应该触发"不确定度"式提示。
TRACE_SINGLE = [
    '{"exp": "X1", "eps": 1e-6, "cstar_rigid": 2.0}',
    '{"exp": "X2", "vho": 0.016, "conv": 1.78}',
]
# 带重复/方差信号的 trace —— "不确定度"式提示应消失、且"过程量量纲"式提示在档位2仍在。
TRACE_REPEAT = [
    '{"exp": "X3", "sigmaH_k0": 1.3294, "sigmaH_k2": 0.0}',
    '{"exp": "X1", "cstar_rigid": [2.0, 3.0, 4.0]}',
]


def _has(hints, *keys):
    joined = "\n".join(hints)
    return all(k in joined for k in keys)


def test_strictness_zero_is_empty():
    """档位0: 任意输入都返回 [] —— 默认信任模型, 零行为变化."""
    assert build_judgment_hints(GOAL, trace=TRACE_SINGLE, strictness=0) == []
    assert build_judgment_hints(GOAL, trace=[], strictness=0) == []


def test_tiered_inclusion():
    """三档逐层包含: 档位2 ⊇ 档位1 ⊇ 档位0(=空)."""
    h0 = build_judgment_hints(GOAL, trace=TRACE_SINGLE, strictness=0)
    h1 = build_judgment_hints(GOAL, trace=TRACE_SINGLE, strictness=1)
    h2 = build_judgment_hints(GOAL, trace=TRACE_SINGLE, strictness=2)
    assert set(h0) <= set(h1) <= set(h2)
    assert h1, "档位1 应至少产出候选自变量提示"
    assert len(h2) >= len(h1)


def test_goal_variables_detected():
    """档位1: target 里显著出现的候选自变量会被软确认 (N/eps/约束)."""
    h1 = build_judgment_hints(GOAL, trace=TRACE_SINGLE, strictness=1)
    assert _has(h1, "样本量 N", "判读-")       # N 在 goal 里
    assert _has(h1, "精度/eps")                # eps 在 goal 里


def test_uncertainty_signal_is_falsifiable():
    """可证伪: 单点 scalar trace 触发"重复/初步"提示; 带 sigmaH/数组则提示消失."""
    h_single = build_judgment_hints(GOAL, trace=TRACE_SINGLE, strictness=1)
    h_repeat = build_judgment_hints(GOAL, trace=TRACE_REPEAT, strictness=1)
    assert _has(h_single, "重复") and _has(h_single, "初步")
    assert not _has(h_repeat, "重复"), "带重复/方差信号的 trace 不应再提示'重复'"


def test_dimension_hint_only_at_tier2():
    """档位2 才出现量纲语义提示; 档位1 不出现."""
    h1 = build_judgment_hints(GOAL, trace=TRACE_SINGLE, strictness=1)
    h2 = build_judgment_hints(GOAL, trace=TRACE_SINGLE, strictness=2)
    assert not _has(h1, "量纲")
    assert _has(h2, "量纲") and _has(h2, "强度校准")


def test_deterministic_pure():
    """纯函数确定性: 同输入同输出."""
    a = build_judgment_hints(GOAL, trace=TRACE_SINGLE, strictness=2)
    b = build_judgment_hints(GOAL, trace=TRACE_SINGLE, strictness=2)
    assert a == b


def test_hint_block_empty_at_zero():
    """hint_block 档位0 → 空串(不会污染 prompt); 档位>0 → 非空."""
    assert hint_block(GOAL, trace=TRACE_SINGLE, strictness=0) == ""
    assert hint_block(GOAL, trace=TRACE_SINGLE, strictness=1) != ""


def test_hints_are_soft_not_hard():
    """护栏只产出"提示文本", 不注入可被硬判的断言(不含绝对结论/数值 lies)."""
    for s in (1, 2):
        for h in build_judgment_hints(GOAL, trace=TRACE_SINGLE, strictness=s):
            # 软提示必须以请求/提醒语气, 不做事实断言(不含"="的客观数值归属)
            assert not h.startswith("结论是"), f"软提示不应硬断言: {h}"