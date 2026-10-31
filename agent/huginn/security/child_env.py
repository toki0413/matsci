"""子进程环境剥离: 密钥类变量只留在父进程, 不传给 bash/容器.

LLM 凭据等敏感配置通过进程环境注入, 但任何经沙箱执行的命令都不该能 ``env``
看到它们. 匹配口径按用途而非具体变量名, 新增 provider 的 ``*_API_KEY`` /
``*_TOKEN`` 自动覆盖.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

#: 变量名命中即剥离 (大小写不敏感).
SECRET_ENV_RE = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)


def sanitize_child_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """返回可直接交给子进程的环境: ``env=None`` 时取父进程环境并剥离密钥."""
    base = dict(os.environ) if env is None else dict(env)
    return {k: v for k, v in base.items() if not SECRET_ENV_RE.search(k)}
