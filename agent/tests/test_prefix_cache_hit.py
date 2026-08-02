"""Byte-level common-prefix cache-hit derivation tests.

模拟 DeepSeek context cache 的命中行为: 用字节级公共前缀推导 cache hit
token, 断言 prefix 稳定化让多轮对话尾部命中率 > 90%.

参考 Reasonix cachehit_e2e_test.go 的 mockDeepSeek 设计, 但不起真 HTTP
server (ponytail: 不引新框架), 直接用一个类模拟端点接收 messages 推导
hit. 我们的 prefix 稳定化设计不同 (动态 context 合并进 HumanMessage,
prefix = system + begin-dialogs + history 稳定).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from huginn.utils.prompt_cache import PromptCacheBuilder


SYSTEM_PROMPT = (
    "# Huginn System Prompt\n"
    "You are a computational materials science assistant.\n"
    "## Core Principles\n"
    "1. Zero Intrusion: NEVER modify user original input files.\n"
    "2. Mathematical Rigor: A calculation is a nonlinear eigenvalue problem.\n"
    "3. Convergence Awareness: distinguish finished vs converged.\n"
    "4. Resource Respect: every CPU/GPU hour costs something.\n"
    "## Tool Use Philosophy\n"
    "Use code_tool for custom analysis, bash_tool for shell, file tools for IO.\n"
)

BEGIN_DIALOGS = [
    ("user", "hello, I need help with DFT calculation"),
    ("assistant", "sure, what material and what property are you targeting?"),
    ("user", "silicon band gap"),
    ("assistant", "got it, let me set up the calculation"),
]

MEMORY = "memory: Si is a group IV semiconductor, indirect band gap ~1.12 eV at 300K"
KG = "kg: Si -> diamond cubic, Fd-3m, a=5.431 Angstrom"
KB = "kb: DFT band gap convergence requires dense k-mesh + GW correction for accuracy"

# 模拟真实 assistant 回复 — 真实对话里 AI 输出几百字很常见, 这让 history
# 累积快, prefix (system+begin+history) 占比大, 稳定 prefix 才能让 hit rate 上 90%.
_AI_ANSWER_BODY = (
    "Based on the silicon diamond structure, the indirect band gap is "
    "approximately 1.12 eV at room temperature. The conduction band minimum "
    "is located near the X point along the Delta line, while the valence band "
    "maximum is at the Gamma point. For accurate DFT prediction one must use "
    "a hybrid functional (HSE06) or GW correction, since standard PBE severely "
    "underestimates the gap. Convergence requires a dense k-mesh (12x12x12 or "
    "denser) and a plane-wave cutoff above 400 eV. The lattice constant "
    "a=5.431 Angstrom should be relaxed before the band structure calculation. "
    "Spin-orbit coupling has a small effect for Si (~15 meV) but is mandatory "
    "for heavier group IV elements like Ge. The dielectric constant and optical "
    "absorption spectrum can be derived from the band structure via DFPT."
)


def _serialize(messages) -> bytes:
    # 把 messages 编码成可比较的字节流. 跟 test_prompt_stability 同样的
    # 序列化策略, 用控制字符当分隔符.
    parts = []
    for m in messages:
        cls = type(m).__name__
        content = m.content if isinstance(m.content, str) else str(m.content)
        mid = getattr(m, "id", None) or ""
        ak = ""
        if m.additional_kwargs:
            ak = repr(sorted(m.additional_kwargs.items()))
        parts.append(cls + "\x01" + mid + "\x01" + ak + "\x01" + content)
    return "\x02".join(parts).encode("utf-8")


def _common_prefix_len(a: bytes, b: bytes) -> int:
    # 最长公共前缀字节数. 逐字节比较到第一个不同点.
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class MockDeepSeekEndpoint:
    """模拟 DeepSeek context cache 的命中推导.

    DeepSeek 的 context cache 按 token 公共前缀命中. 这里用字节级公共前缀
    近似 (1 char ~= 1 token 量级, 比例关系对 hit rate 阈值判断足够).

    ponytail: 不起真 HTTP server, 字节级公共前缀够推导 hit rate. 升级路径:
    接真 DeepSeek API 的 prompt_cache_hit_tokens / prompt_cache_miss_tokens
    字段做 end-to-end 验证.

    用法: 每轮调 chat(messages) 一次, 内部跟上一轮的字节流算公共前缀,
    把 hit/miss/rate 记进 self.rates.
    """

    def __init__(self) -> None:
        self.prev_bytes: bytes | None = None
        self.rates: list[float] = []
        self.hit_chars: list[int] = []
        self.miss_chars: list[int] = []

    def chat(self, messages) -> AIMessage:
        cur = _serialize(messages)
        if self.prev_bytes is not None:
            hit = _common_prefix_len(self.prev_bytes, cur)
            miss = len(cur) - hit
            self.hit_chars.append(hit)
            self.miss_chars.append(miss)
            total = hit + miss
            rate = hit / total if total > 0 else 0.0
            self.rates.append(rate)
        self.prev_bytes = cur
        return AIMessage(content="mock answer")

    def tail_avg(self, n: int = 5) -> float:
        if not self.rates:
            return 0.0
        tail = self.rates[-n:]
        return sum(tail) / len(tail)


def _run_multi_turn(builder: PromptCacheBuilder, n_turns: int, memory: str = MEMORY,
                    kg: str = KG, kb: str = KB) -> MockDeepSeekEndpoint:
    # 跑 n_turns 轮对话. 每轮 user 问题不同, memory/kg/kb 稳定 (代表稳定环境),
    # history 累积上一轮的 user + AI 回复. AI 回复用真实长度模板.
    endpoint = MockDeepSeekEndpoint()
    history: list = []
    for turn in range(n_turns):
        user_msg = "turn " + str(turn) + ": compute band gap of material " + str(turn)
        msgs = builder.build_state_modifier() + builder.build_input_messages(
            memory, user_msg, kg_text=kg, kb_text=kb,
            history_messages=history or None,
        )
        endpoint.chat(msgs)
        history.append(HumanMessage(content=user_msg))
        history.append(AIMessage(content="answer " + str(turn) + ": " + _AI_ANSWER_BODY))
    return endpoint


# helper 验证


def test_common_prefix_len_helper():
    assert _common_prefix_len(b"abc", b"abc") == 3
    assert _common_prefix_len(b"abc", b"abd") == 2
    assert _common_prefix_len(b"abc", b"ab") == 2
    assert _common_prefix_len(b"", b"abc") == 0
    assert _common_prefix_len(b"abc", b"") == 0


def test_serialize_is_deterministic():
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    msgs = builder.build_full_messages(MEMORY, "q", kg_text=KG, kb_text=KB)
    a = _serialize(msgs)
    b = _serialize(msgs)
    assert a == b


# 主测试: prefix 稳定时 hit rate > 90%


def test_prefix_cache_hit_rate_above_90_percent():
    # 核心: prefix (system + begin-dialogs + history) 字节稳定, 只有尾部
    # HumanMessage 变, 多轮对话尾部命中率 > 90%.
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    endpoint = _run_multi_turn(builder, n_turns=20)

    tail = endpoint.rates[-5:]
    avg = endpoint.tail_avg(5)
    assert avg > 0.90, (
        "cache hit rate too low: avg=" + format(avg, ".2%") +
        ", tail rates=" + repr([format(r, ".2%") for r in tail]) +
        ", hit_chars=" + repr(endpoint.hit_chars[-5:]) +
        ", miss_chars=" + repr(endpoint.miss_chars[-5:])
    )


def test_hit_rate_tail_above_85_percent():
    # 稳定 prefix 下, 尾部每轮命中率都不应跌破 0.85 (history 累积, hit 部分越来越大).
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    endpoint = _run_multi_turn(builder, n_turns=20)
    tail = endpoint.rates[-5:]
    worst = min(tail)
    assert worst > 0.85, (
        "tail hit rate dropped below 0.85: worst=" + format(worst, ".2%") +
        ", tail=" + repr([format(r, ".2%") for r in tail])
    )


def test_first_turn_has_no_hit_record():
    # 第一轮 prev=None, rates 应为空 (没东西可比). n_turns=3 产生 2 条 rate.
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    endpoint = _run_multi_turn(builder, n_turns=3)
    assert len(endpoint.rates) == 2


# 区分度测试: 故意破坏 prefix 稳定, hit rate 掉下来


def test_unstable_prefix_drops_hit_rate():
    # 对照组: 每轮换 system prompt (模拟 prefix 漂移), hit rate 必须显著低于稳定情况.
    # 这验证测试能区分稳定 vs 不稳定, 不是恒真.
    endpoint = MockDeepSeekEndpoint()
    history: list = []
    n_turns = 10
    for turn in range(n_turns):
        # 故意每轮改 system prompt, 破坏 prefix 稳定
        sys = SYSTEM_PROMPT + "\n## Drift Marker turn " + str(turn) + "\n"
        builder = PromptCacheBuilder(sys, BEGIN_DIALOGS)
        user_msg = "turn " + str(turn) + ": compute something " + str(turn)
        msgs = builder.build_state_modifier() + builder.build_input_messages(
            MEMORY, user_msg, kg_text=KG, kb_text=KB,
            history_messages=history or None,
        )
        endpoint.chat(msgs)
        history.append(HumanMessage(content=user_msg))
        history.append(AIMessage(content="answer " + str(turn) + ": " + _AI_ANSWER_BODY))
    avg_unstable = endpoint.tail_avg(5)
    # 不稳定情况命中率必须显著低于稳定基线 0.90
    assert avg_unstable < 0.90, (
        "unstable prefix did not drop hit rate as expected: avg=" +
        format(avg_unstable, ".2%") + " (should be well below 90%)"
    )


def test_dynamic_context_change_in_human_message_keeps_prefix_hit():
    # memory/kg/kb 每轮变 (代表每轮召回不同), 但走 HumanMessage, prefix 仍稳定,
    # 命中率仍高. 这是 prefix 稳定化设计面对动态召回的承诺.
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    endpoint = MockDeepSeekEndpoint()
    history: list = []
    n_turns = 20
    for turn in range(n_turns):
        # 每轮 memory/kg/kb 内容变 (模拟每轮召回不同结果)
        mem = "memory turn " + str(turn) + ": result " + str(turn * 7 % 13)
        kg = "kg turn " + str(turn) + ": edge " + str(turn)
        kb = "kb turn " + str(turn) + " ref " + str(turn)
        user_msg = "turn " + str(turn) + ": question " + str(turn)
        msgs = builder.build_state_modifier() + builder.build_input_messages(
            mem, user_msg, kg_text=kg, kb_text=kb,
            history_messages=history or None,
        )
        endpoint.chat(msgs)
        history.append(HumanMessage(content=user_msg))
        history.append(AIMessage(content="answer " + str(turn) + ": " + _AI_ANSWER_BODY))
    avg = endpoint.tail_avg(5)
    # 动态 context 走 HumanMessage, prefix (system+begin+history) 仍稳定,
    # 命中率应保持高位 (HumanMessage 只占尾部一小段).
    assert avg > 0.85, (
        "hit rate dropped when dynamic context changed: avg=" + format(avg, ".2%") +
        ", tail=" + repr([format(r, ".2%") for r in endpoint.rates[-5:]])
    )