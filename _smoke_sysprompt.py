"""ponytail self-check: system prompt 不再有多重 CRITICAL override."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "agent"))

import inspect
from huginn.cli import rcb_runner

# system_prompt 是 run() 内的 local 变量, 这里读源码文本做静态检查
src = inspect.getsource(rcb_runner)
# 提取 system_prompt = (...) 段
start = src.find("system_prompt = (")
end = src.find("\n    )", start)
prompt_block = src[start:end]
critical_count = prompt_block.upper().count("CRITICAL")
assert critical_count == 0, f"CRITICAL still in system_prompt: {critical_count} occurrences"
assert "INSTRUCTIONS.md" in prompt_block, "no INSTRUCTIONS.md reference"
assert "honestly" in prompt_block.lower(), "no honesty guidance"
print(f"self-check passed: system_prompt lean ({critical_count} CRITICAL)")
