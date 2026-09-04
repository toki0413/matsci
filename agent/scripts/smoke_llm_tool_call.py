#!/usr/bin/env python3
"""LLM 工具调用冒烟 — 验证『真实模型产出 tool_call → 解析为 PhysicalAction → 物理执行』链路.

背景: Huginn 的传感器闭环此前只对"内部解析真值"做过单测, 但『LLM 真的把一次工具调用
的选择与参数吐出来, 并被 agent 解析进 PhysicalAction』这一环从未实测. 本脚本用 llama.cpp
加载一个本地 GGUF, 让它决定调用一个模拟外部计算工具, 解析返回的工具名+JSON 参数为
``PhysicalAction``, 再走 compute_adapter / world_model 得到并打印一次物理结果.

用法(在本沙箱已验证):
    python scripts/smoke_llm_tool_call.py --model /path/to/model-q4_k_m.gguf

星火 X2.5-4B 真机实测: 把你转好的 GGUF 路径填进 --model 即可, 同一链路.

依赖: pip install llama-cpp-python  (需可编译; CPU 即可)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 允许独立运行: 脚本不依赖 app 启动.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huginn.security.compute_adapter import ShellComputeWorldModel  # noqa: E402
from huginn.security.world_model import PhysicalAction  # noqa: E402

# 暴露给 LLM 的唯一工具, 模拟 compute_adapter 的 ShellComputeTool.
_TOOL = {
    "type": "function",
    "function": {
        "name": "shell_compute",
        "description": "Compute internal energy E = n*Cv*T of an ideal-gas closed system. n is amount in mol, T is temperature in kelvin.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {"type": "number", "description": "amount of substance in mol"},
                "T": {"type": "number", "description": "temperature in kelvin"},
            },
            "required": ["n", "T"],
            "additionalProperties": False,
        },
    },
}

_PARSE_HINT = (
    "You must call the shell_compute tool with n=2.0 and T=300.0. "
    "Return only the function call, no prose."
)


def _parse_llama_injected_json(text: str):
    """解析 llama.cpp chat template 用 {{...}} 包裹的工具调用 JSON (嵌套大括号安全)."""
    i = text.find("{{")
    if i < 0:
        return None
    depth = 0
    j = i
    while j < len(text):
        if text.startswith("{{", j):
            depth += 2
            j += 2
        elif text[j] == "{":
            depth += 1
            j += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                # 最外层 {{ 含 json 的顶层 `{`, 已随 opening 消耗; 补回后即合法 JSON.
                return json.loads("{" + text[i + 2 : j])
            j += 1
        else:
            j += 1
    return None


def _extract_tool_call(sample: dict) -> dict:
    """从 llama-cpp-python 的 chat completion sample 里取出工具调用."""
    msg = (
        sample.get("message")
        or sample["choices"][0]["message"]
        or sample["choices"][0].get("delta", {})
    )
    calls = msg.get("tool_calls")
    if calls:
        raw = calls[0]["function"]
        name = raw["name"]
        args = (
            json.loads(raw["arguments"])
            if isinstance(raw["arguments"], str)
            else raw["arguments"]
        )
        return {"name": name, "arguments": args}
    # 兜底: 模型没走 llama-cpp 的工具路由, 直接在文本里吐 JSON (单引号/{{...}}/裸 JSON).
    import ast

    text = msg.get("content") or ""
    injected = _parse_llama_injected_json(text)
    if injected is not None:
        return injected
    line_candidates = [
        ln.strip() for ln in text.splitlines() if ln.strip().startswith(("{", "["))
    ]
    for c in line_candidates:
        parsers = (json.loads, ast.literal_eval)
        for fn in parsers:
            try:
                return fn(c)
            except Exception:
                pass
        try:  # 单引号 → 双引号后再试严格 JSON
            return json.loads(c.replace("'", '"'))
        except Exception:
            pass
    raise AssertionError(f"model did not produce a tool call. raw={text[:400]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="GGUF 模型文件路径")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from llama_cpp import Llama  # 延迟导入, 便于 --help 不依赖引擎

    llm = Llama(
        model_path=args.model, n_ctx=args.n_ctx, n_threads=args.threads, seed=args.seed
    )

    sample = llm.create_chat_completion(
        messages=[{"role": "user", "content": _PARSE_HINT}],
        tools=[_TOOL],
        tool_choice="required",
        temperature=0.0,
        max_tokens=256,
    )
    call = _extract_tool_call(sample)

    assert call["name"] == "shell_compute", f"unexpected tool: {call['name']}"
    action = PhysicalAction("shell_compute", call["arguments"])
    # 走已有的 world_model(=快代理, 与 executor 真子进程同解析公式), 得到物理结果.
    pred = ShellComputeWorldModel().predict({}, action)
    n = action.params.get("n")
    T = action.params.get("T")
    print(f"[ok] tool_call -> shell_compute(n={n}, T={T}) -> energy={pred['energy']} J")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
