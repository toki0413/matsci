"""解析用户输入里的 @file / @url 引用。

把 @path/to/file.py 替换成文件内容, @https://... 替换成抓回来的网页文本,
其它普通文本原样保留。给 agent 喂上下文的时候很方便。
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.request import Request, urlopen

# 网页抓取的字符上限, 太长会把上下文撑爆
_URL_MAX_CHARS = 5000
# 单文件大小上限 (1MB), 超过就只取前 1MB, 避免读超大文件卡死
_FILE_MAX_BYTES = 1_000_000
# HTTP 抓取超时 (秒)
_URL_TIMEOUT = 10.0

# 匹配 @开头的引用: @路径 或 @http(s)://...
# 路径里允许字母/数字/下划线/横杠/斜杠/点/反斜杠
_AT_REF_PATTERN = re.compile(r"@([^\s@]+)")


def _is_url(token: str) -> bool:
    """判断 token 是不是 http/https URL。"""
    return token.startswith("http://") or token.startswith("https://")


def _read_file(path: Path) -> str:
    """读文件内容, 二进制安全, 超大文件截断。

    读不了 (不存在 / 权限不够 / 编码异常) 就返回一段提示文本, 不抛异常,
    避免一条引用挂掉整次对话。
    """
    try:
        if not path.exists():
            return f"<file not found: {path}>"
        raw = path.read_bytes()
        if len(raw) > _FILE_MAX_BYTES:
            raw = raw[:_FILE_MAX_BYTES]
            truncated_note = f"\n... (truncated at {_FILE_MAX_BYTES} bytes)"
        else:
            truncated_note = ""
        # 大多数源码/配置是 utf-8, 解不了就 errors=replace 兜底
        text = raw.decode("utf-8", errors="replace")
        return text + truncated_note
    except Exception as e:
        return f"<failed to read {path}: {e}>"


def _fetch_url(url: str) -> str:
    """抓一个 URL 的文本内容, 截断到 _URL_MAX_CHARS。

    抓不到就返回提示文本, 不抛异常。
    """
    try:
        req = Request(url, headers={"User-Agent": "huginn-agent/1.0"})
        with urlopen(req, timeout=_URL_TIMEOUT) as resp:
            raw = resp.read(_URL_MAX_BYTES + 1)
        text = raw.decode("utf-8", errors="replace")
        if len(text) > _URL_MAX_CHARS:
            text = text[:_URL_MAX_CHARS] + "\n... (truncated)"
        return text
    except Exception as e:
        return f"<failed to fetch {url}: {e}>"


def _format_file_block(path: Path, content: str) -> str:
    """把文件内容包成一个 fenced code block, 带上文件名做语言提示。"""
    # 用后缀当 language hint, 方便 markdown 渲染
    suffix = path.suffix.lstrip(".") if path.suffix else "text"
    return f"```{suffix}\n{content}\n```"


def _format_url_block(url: str, content: str) -> str:
    """把网页文本包成 block, 标注来源 URL。"""
    return f"<url src=\"{url}\">\n{content}\n</url>"


def parse_at_references(text: str, workspace: str = ".") -> str:
    """解析 @file_path 和 @url 引用, 把内容拼进文本。

    - ``@path/to/file.py`` → 读文件内容, 替换成 ```lang\\n<内容>\\n```
    - ``@https://example.com`` → urllib 抓取, 替换成网页文本 (截断到 5000 字符)
    - 普通文本原样保留

    读不了的文件 / 抓不到的 URL 会被替换成一段提示文本, 不会抛异常,
    这样用户其它部分的输入还能正常送给 agent。
    """
    workspace_path = Path(workspace).expanduser().resolve()

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if not token:
            return match.group(0)

        if _is_url(token):
            content = _fetch_url(token)
            return _format_url_block(token, content)

        # 文件路径: 相对路径按 workspace 解析, 绝对路径直接用
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = workspace_path / candidate
        candidate = candidate.resolve()

        content = _read_file(candidate)
        return _format_file_block(candidate, content)

    return _AT_REF_PATTERN.sub(_replace, text)


def has_at_references(text: str) -> bool:
    """快速判断文本里有没有 @ 引用, 避免每次都跑完整解析。

    用在 chat 循环里决定要不要调 parse_at_references。
    邮箱 (xxx@yyy) 和 @mention 这种裸单词不算文件/URL 引用, 跳过。
    """
    for match in _AT_REF_PATTERN.finditer(text):
        token = match.group(1)
        if not token:
            continue
        # URL 一定是引用
        if _is_url(token):
            return True
        # 路径里有分隔符, 或者第一段带后缀点, 算文件引用
        # 单独一个 @word 通常是 @mention, 不处理
        first_seg = token.split("/")[0].split("\\")[0]
        if "/" in token or "\\" in token or "." in first_seg:
            return True
    return False


__all__ = ["parse_at_references", "has_at_references"]
