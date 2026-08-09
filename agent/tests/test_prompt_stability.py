"""Byte-stability guards for the prompt cache prefix.

守护 prefix 稳定化: 同输入两次 build 字节相等; memory/kg/kb 变化时
prefix (system + begin-dialogs) 字节不变, 只有 HumanMessage 变; 环境探测
结果不进 system prompt.

任何让 prefix 漂移的改动 (动态 context 重新插进 SystemMessage, probe
结果注入 system prompt, 等) 会让这里失败, 命中率回退没人察觉.

参考 Reasonix prompt_stability_test.go 的字节级守卫思路, 但我们的 prefix
稳定化设计不同 (动态 context 合并进 HumanMessage, 不进 SystemMessage).
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
)

BEGIN_DIALOGS = [
    ("user", "hello, I need help with DFT"),
    ("assistant", "sure, what material are you studying?"),
]


def _serialize(messages) -> bytes:
    # 把 messages 编码成可比较的字节流. 每条消息拼成
    # `类名 \x01 id \x01 additional_kwargs \x01 content`, 之间用 \x02 分隔.
    # 用控制字符当分隔符避免跟正文冲突.
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


def _first_divergence(a: bytes, b: bytes) -> str:
    # 找首个不同字节, 返回前后 40 字符窗口的可读描述.
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    if i >= n:
        return (
            "no byte diff in common prefix (len=" + str(n) + "), "
            "lengths: first=" + str(len(a)) + " second=" + str(len(b))
        )
    lo = max(0, i - 40)
    hi_a = min(len(a), i + 40)
    hi_b = min(len(b), i + 40)
    return (
        "first diff at byte " + str(i) + ":\n"
        "  first  : ..." + repr(a[lo:hi_a]) + "...\n"
        "  second : ..." + repr(b[lo:hi_b]) + "..."
    )


# Task 5.1: 相同输入两次 build 字节级相等


def test_build_full_messages_byte_stable_same_inputs():
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    first = builder.build_full_messages(
        "memory: Si band gap 1.12 eV",
        "what is the band gap of Si?",
        kg_text="kg: Si -> diamond structure",
        kb_text="kb: semiconductor band theory reference",
    )
    second = builder.build_full_messages(
        "memory: Si band gap 1.12 eV",
        "what is the band gap of Si?",
        kg_text="kg: Si -> diamond structure",
        kb_text="kb: semiconductor band theory reference",
    )
    fb = _serialize(first)
    sb = _serialize(second)
    if fb != sb:
        pytest.fail(
            "build_full_messages not byte-stable on identical inputs:\n"
            + _first_divergence(fb, sb)
        )


def test_build_input_messages_byte_stable_same_inputs():
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    history = [
        HumanMessage(content="previous question"),
        AIMessage(content="previous answer"),
    ]
    first = builder.build_input_messages(
        "memory fact A", "current question", kg_text="kg A", kb_text="kb A",
        history_messages=history,
    )
    second = builder.build_input_messages(
        "memory fact A", "current question", kg_text="kg A", kb_text="kb A",
        history_messages=history,
    )
    fb = _serialize(first)
    sb = _serialize(second)
    if fb != sb:
        pytest.fail(
            "build_input_messages not byte-stable on identical inputs:\n"
            + _first_divergence(fb, sb)
        )


def test_system_prompt_alone_byte_stable():
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    a = builder.build_state_modifier()
    b = builder.build_state_modifier()
    assert _serialize(a) == _serialize(b)


# Task 5.2: 动态 context 变化时 prefix 不变, HumanMessage 变


def test_prefix_stable_when_memory_changes():
    # memory 内容变化, prefix (system + begin-dialogs) 字节不变, HumanMessage 变
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    msgs_a = builder.build_full_messages(
        "memory A: Si band gap 1.12 eV", "user question A",
    )
    msgs_b = builder.build_full_messages(
        "memory B: GaAs band gap 1.42 eV", "user question B",
    )
    prefix_a = _serialize(msgs_a[:-1])
    prefix_b = _serialize(msgs_b[:-1])
    if prefix_a != prefix_b:
        pytest.fail(
            "prefix drifted when memory changed:\n" + _first_divergence(prefix_a, prefix_b)
        )
    human_a = _serialize([msgs_a[-1]])
    human_b = _serialize([msgs_b[-1]])
    assert human_a != human_b, "HumanMessage should change when memory changes"
    assert b"memory B" in human_b, "HumanMessage should contain new memory content"
    assert b"memory A" not in human_b, "old memory should not persist in new HumanMessage"


def test_prefix_stable_when_kg_changes():
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    msgs_a = builder.build_full_messages(
        "same memory", "same user", kg_text="kg A: Si -> diamond",
    )
    msgs_b = builder.build_full_messages(
        "same memory", "same user", kg_text="kg B: GaAs -> zincblende",
    )
    prefix_a = _serialize(msgs_a[:-1])
    prefix_b = _serialize(msgs_b[:-1])
    if prefix_a != prefix_b:
        pytest.fail(
            "prefix drifted when kg changed:\n" + _first_divergence(prefix_a, prefix_b)
        )
    human_b = _serialize([msgs_b[-1]])
    assert b"kg B" in human_b
    assert b"kg A" not in human_b


def test_prefix_stable_when_kb_changes():
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    msgs_a = builder.build_full_messages(
        "same memory", "same user", kb_text="kb A: DFT basics",
    )
    msgs_b = builder.build_full_messages(
        "same memory", "same user", kb_text="kb B: MD force fields",
    )
    prefix_a = _serialize(msgs_a[:-1])
    prefix_b = _serialize(msgs_b[:-1])
    if prefix_a != prefix_b:
        pytest.fail(
            "prefix drifted when kb changed:\n" + _first_divergence(prefix_a, prefix_b)
        )
    human_b = _serialize([msgs_b[-1]])
    assert b"kb B" in human_b
    assert b"kb A" not in human_b


def test_prefix_stable_all_dynamic_change_together():
    # memory+kg+kb 同时变, prefix 字节不变, HumanMessage 含全部新内容
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    msgs_a = builder.build_full_messages(
        "memory A", "user A", kg_text="kg A", kb_text="kb A",
    )
    msgs_b = builder.build_full_messages(
        "memory B", "user B", kg_text="kg B", kb_text="kb B",
    )
    prefix_a = _serialize(msgs_a[:-1])
    prefix_b = _serialize(msgs_b[:-1])
    if prefix_a != prefix_b:
        pytest.fail(
            "prefix drifted when memory+kg+kb all changed:\n"
            + _first_divergence(prefix_a, prefix_b)
        )
    human_b = _serialize([msgs_b[-1]])
    for token in (b"memory B", b"kg B", b"kb B", b"user B"):
        assert token in human_b, "HumanMessage missing dynamic content " + repr(token)


def test_prefix_with_history_stable_when_dynamic_changes():
    # 带 history 的 prefix 在动态 context 变化时仍稳定 (核心承诺)
    builder = PromptCacheBuilder(SYSTEM_PROMPT, BEGIN_DIALOGS)
    history = [
        HumanMessage(content="earlier question"),
        AIMessage(content="earlier answer"),
    ]
    msgs_a = builder.build_input_messages(
        "memory A", "user A", kg_text="kg A", kb_text="kb A",
        history_messages=history,
    )
    msgs_b = builder.build_input_messages(
        "memory B", "user B", kg_text="kg B", kb_text="kb B",
        history_messages=history,
    )
    prefix_a = _serialize(msgs_a[:-1])
    prefix_b = _serialize(msgs_b[:-1])
    if prefix_a != prefix_b:
        pytest.fail(
            "prefix (begin-dialogs + history) drifted:\n"
            + _first_divergence(prefix_a, prefix_b)
        )
    assert _serialize([msgs_a[-1]]) != _serialize([msgs_b[-1]])


# Task 5.4: 环境探测结果不进 system prompt


def test_env_probe_results_not_in_system_prompt():
    # probe flap 守护: docker/python3/node 等 probe 结果不进 system prompt.
    # 环境探测 (容器运行时/解释器版本) 每次启动可能不同, 进 system prompt
    # 会让 prefix 漂移, 命中率掉. 当前 HUGINN_SYSTEM_PROMPT 是静态文案,
    # 这里 assert probe 结果字符串不出现, 以后有人改成注入 probe 会暴露.
    from huginn.prompts import HUGINN_SYSTEM_PROMPT

    builder = PromptCacheBuilder(HUGINN_SYSTEM_PROMPT, [])
    prefix = builder.build_state_modifier()
    assert len(prefix) == 1
    sys_text = prefix[0].content
    assert isinstance(sys_text, str)
    low = sys_text.lower()
    probe_tokens = [
        "docker",
        "python3",
        "python3.",
        "node v",
        "nodejs",
        "containerd",
        "detected_env",
        "env_probe",
        "environment detected",
        "container runtime",
        "shell detected",
    ]
    leaked = [t for t in probe_tokens if t.lower() in low]
    assert not leaked, (
        "environment probe results leaked into system prompt: " + repr(leaked)
    )


def test_huginn_system_prompt_is_static_constant():
    # HUGINN_SYSTEM_PROMPT 必须是静态常量, 不依赖运行时探测.
    # 两次访问拿到的内容字节相等. 防止以后改成 @property 或 lazy 加载注入 probe.
    from huginn import prompts as p1
    from huginn import prompts as p2

    a = p1.HUGINN_SYSTEM_PROMPT.encode("utf-8")
    b = p2.HUGINN_SYSTEM_PROMPT.encode("utf-8")
    if a != b:
        pytest.fail(
            "HUGINN_SYSTEM_PROMPT is not byte-stable across accesses:\n"
            + _first_divergence(a, b)
        )
