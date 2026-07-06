"""命令安全过滤器 —— 在可执行文件白名单之上的第二道防线。

借鉴 AstrBot 里 local.py booter 的 _BLOCKED_COMMAND_PATTERNS 思路。
现有的 SandboxExecutor 走的是可执行文件白名单 (更强), 但白名单挡不住
"白名单里的程序被喂了危险参数" 这种情况 (典型: python -c 'rm -rf /')。
这里再加一层, 对完整命令串做正则匹配, 命中危险模式就直接拦下。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 危险模式列表 —— 对完整命令串 (已转小写) 做子串/正则匹配
_BLOCKED_PATTERNS: list[str] = [
    r"rm\s+-rf\s+/",
    r"rm\s+-rf\s+~",
    r"rm\s+-rf\s+\*",
    r"rm\s+-fr\s+/",
    r"mkfs\.\w+\s+/dev/",
    r"dd\s+if=.*of=/dev/",
    r"shutdown\b",
    r"reboot\b",
    r"\bsudo\b",
    r":\(\)\s*\{\s*:\|:\&\s*\}\s*;",  # fork bomb
    r"kill\s+-9\s+1\b",
    r"killall\b",
    r">\s*/dev/sda",
    r"chmod\s+-R\s+777\s+/",
    r"chown\s+-R\s+.*\s+/",
    # Windows dangerous patterns
    r"\bformat\s+[a-z]:",
    r"\bdel\s+/[fsq]\b",
    r"\brmdir\s+/s\b",
    r"\brd\s+/s\b",
    r"powershell\s+-enc\b",
    r"\bnc\s+-[elp]",
    r"netcat\b",
    r"crontab\s+-[er]",
]


@dataclass
class CommandFilterResult:
    """命令过滤结果。is_safe=False 时 matched_pattern 记录命中的模式。"""

    is_safe: bool
    matched_pattern: str | None = None


def check_command_safety(command: list[str] | str) -> CommandFilterResult:
    """检查命令是否包含危险模式。

    list 输入会被空格拼接成字符串再匹配, str 输入直接匹配。
    命中任一模式即返回 is_safe=False 并带上命中的模式串。
    用 re.IGNORECASE 做大小写无关匹配 (部分模式含大写 flag 如 -R)。
    """
    cmd_str = " ".join(command) if isinstance(command, list) else command
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, cmd_str, re.IGNORECASE):
            return CommandFilterResult(is_safe=False, matched_pattern=pattern)
    return CommandFilterResult(is_safe=True)
