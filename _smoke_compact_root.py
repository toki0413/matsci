"""ponytail self-check: compaction 保留 root messages."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "agent"))

from huginn.utils.context import compact_messages
from langchain_core.messages import HumanMessage, SystemMessage

# 构造 10 条消息, 前 2 条是 root (task + checklist)
msgs = [
    SystemMessage(content="TASK: reproduce VAE"),
    HumanMessage(content="CHECKLIST: [EXACT] graph-VAE, [EXACT] Tanimoto kernel"),
] + [HumanMessage(content=f"tool result {i} " * 50) for i in range(8)]

# compact 到小 budget, keep_last_n=1, keep_root_n=2
compacted = compact_messages(msgs, budget_tokens=200, keep_last_n=1, keep_root_n=2)
# 验证 root 保留
assert "TASK" in compacted[0].content, f"root message 0 lost: {compacted[0].content[:50]}"
assert "CHECKLIST" in compacted[1].content, f"root message 1 lost: {compacted[1].content[:50]}"
# 验证 body compacted (不是全部 10 条)
assert len(compacted) < len(msgs), f"no compaction: {len(compacted)} vs {len(msgs)}"
print(f"self-check passed: root preserved, body compacted ({len(msgs)} -> {len(compacted)})")
