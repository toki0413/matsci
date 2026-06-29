"""代码调试工具 — 解析 Python traceback、定位根因、给出修复建议。

纯规则分析，不依赖 LLM。所有逻辑基于正则、字典分类和 ast 解析。
"""

from __future__ import annotations

import ast
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


# ---------------------------------------------------------------------------
# 输入 schema
# ---------------------------------------------------------------------------
class DebuggerInput(BaseModel):
    action: Literal[
        "parse_traceback",
        "analyze_root_cause",
        "suggest_fix",
        "explain_error",
    ] = Field(
        ...,
        description="调试动作：解析 traceback / 分析根因 / 给修复建议 / 通俗解释",
    )
    traceback_text: str | None = Field(
        default=None,
        description="完整的 traceback 文本，parse_traceback 动作必填",
    )
    code_snippet: str | None = Field(
        default=None,
        description="出错的代码片段，suggest_fix 时可附上用于 ast 分析",
    )
    language: str = Field(default="python", description="编程语言，目前仅支持 python")
    error_message: str | None = Field(
        default=None,
        description="单独的错误消息（无 traceback 时使用）",
    )
    file_path: str | None = Field(default=None, description="出错文件路径")
    context_lines: int = Field(
        default=5, description="提取上下文行数", ge=0, le=50
    )


# ---------------------------------------------------------------------------
# 规则表：每种异常类型的根因 / 检查项 / 严重程度 / 通俗解释 / 修复模板
# ---------------------------------------------------------------------------
# 格式说明：每个条目返回 root_cause, check_list, severity, fix_template, explanation
_KB: dict[str, dict[str, Any]] = {
    "NameError": {
        "root_cause": "引用了一个未定义的名字",
        "check_list": [
            "检查变量名拼写是否正确（typo）",
            "检查变量是否在当前作用域内定义",
            "检查是否漏了 import",
            "检查是否在赋值前就使用了该名字",
            "检查是否把局部变量当全局变量用了",
        ],
        "severity": "low",
        "fix_template": "检查 {var} 是否已定义、已导入，或拼写是否正确",
        "what_happened": "代码用了一个 Python 不认识的名字（变量/函数/类）。",
        "why_it_happened": "这个名字在当前作用域里没出现过——可能是拼写错了、忘了 import，"
        "或者在使用前还没赋值。",
        "how_to_avoid": [
            "用 IDE 的自动补全和未定义变量提示",
            "import 后立刻用一下，避免漏 import",
            "开启 linter（pyflakes/ruff）能在保存时就抓到这类问题",
        ],
    },
    "TypeError": {
        "root_cause": "对不兼容的类型执行了操作，或传了错误类型的参数",
        "check_list": [
            "检查运算符两侧的操作数类型",
            "检查函数参数的实际类型是否符合签名",
            "检查是否对 None 做了运算",
            "检查是否调用了不存在的方法（类型与预期不符）",
            "检查可迭代对象/迭代器是否被当成标量用",
        ],
        "severity": "medium",
        "fix_template": "检查 {arg} 的类型，期望 {expected}，实际 {actual}",
        "what_happened": "对一个类型不支持的操作使用了该类型的对象。",
        "why_it_happened": "Python 是动态类型，运行时才发现类型不对——比如拿 int 和 str 相加，"
        "或者给函数传了它不接受的类型。",
        "how_to_avoid": [
            "用类型注解 + mypy 做静态检查",
            "函数入口处用 isinstance 或 assert 校验关键参数",
            "对 None 显式判断，别让 None 进入运算链",
        ],
    },
    "AttributeError": {
        "root_cause": "访问了对象上不存在的属性或方法",
        "check_list": [
            "检查对象是否为 None（None 没有自定义属性）",
            "检查对象的实际类型，确认该属性/方法确实存在",
            "检查属性名拼写",
            "检查是否混用了不同类的实例",
            "检查函数是否漏了 return（默认返回 None）",
        ],
        "severity": "medium",
        "fix_template": "检查 {obj} 的类型，确认它有 {attr} 属性；尤其注意 None",
        "what_happened": "试图访问对象没有的属性或方法。",
        "why_it_happened": "最常见的原因是对象是 None（函数忘 return），或者对象类型和"
        "预期不一致——你以为它是 list，其实它是 None。",
        "how_to_avoid": [
            "调用链式属性前先判断 None",
            "函数结尾确保有 return",
            "用 dataclass / TypedDict 明确对象结构",
        ],
    },
    "IndexError": {
        "root_cause": "序列下标越界",
        "check_list": [
            "检查序列的实际长度",
            "检查循环边界条件",
            "检查是否空列表就取了 [0]",
            "检查负索引是否超出范围",
        ],
        "severity": "low",
        "fix_template": "访问 {seq}[{idx}] 前先检查 len({seq}) > {idx}",
        "what_happened": "用了一个超出范围的索引去访问列表/元组。",
        "why_it_happened": "序列里没那么多元素，常见于空列表取 [0]，或者循环里 i 算错了。",
        "how_to_avoid": [
            "访问前用 if seq: 或 if len(seq) > idx 守一下",
            "遍历优先用 for x in seq 而不是下标",
            "用 enumerate 拿到索引和值，避免手动算下标",
        ],
    },
    "KeyError": {
        "root_cause": "访问了字典中不存在的键",
        "check_list": [
            "检查键名拼写（大小写、空格）",
            "检查键是否真的在 dict 里",
            "检查数据来源是否漏了某些键",
            "考虑用 .get(key, default) 代替 []",
        ],
        "severity": "low",
        "fix_template": "访问 dict 前用 {key} in d 或 d.get({key}, default)",
        "what_happened": "用 [] 取了字典里没有的键。",
        "why_it_happened": "字典里压根没这个键——可能数据没准备好，或者键名拼错了。",
        "how_to_avoid": [
            "改用 d.get(key, default)，缺键时返回默认值",
            "用 collections.defaultdict 自动补键",
            "解析外部数据后先 assert 关键键存在",
        ],
    },
    "ValueError": {
        "root_cause": "值的类型对，但取值不合法",
        "check_list": [
            "检查值的范围（如负数开方、空序列取 max）",
            "检查格式（如 int('abc')、datetime 解析）",
            "检查枚举值是否在允许集合内",
            "检查数量是否匹配（如 unpacking）",
        ],
        "severity": "medium",
        "fix_template": "检查 {value} 的取值/格式是否符合 {expected}",
        "what_happened": "值的类型没错，但内容不合法。",
        "why_it_happened": "比如 int('abc')、math.sqrt(-1)、或者 a, b = [1,2,3] "
        "解包数量不匹配——类型对，但值不行。",
        "how_to_avoid": [
            "转换前用 try/except 兜底",
            "对范围敏感的操作先做边界判断",
            "解包前确认元素个数",
        ],
    },
    "ImportError": {
        "root_cause": "无法导入模块或模块中的名字",
        "check_list": [
            "检查模块是否已安装（pip list）",
            "检查模块名拼写",
            "检查 sys.path 是否包含模块所在目录",
            "检查是否有同名的本地文件遮蔽了标准库",
            "检查 Python 版本是否兼容",
        ],
        "severity": "high",
        "fix_template": "确认 {module} 已安装：pip install {module}",
        "what_happened": "import 语句失败了。",
        "why_it_happened": "要么模块没装，要么装了但不在 sys.path 里，"
        "要么你本地有个同名 .py 文件把真模块挡了。",
        "how_to_avoid": [
            "用虚拟环境锁依赖（pip freeze > requirements.txt）",
            "别给本地文件起和标准库/第三方库一样的名字",
            "CI 里跑一次干净环境的 import 测试",
        ],
    },
    "ModuleNotFoundError": {
        "root_cause": "找不到指定的模块",
        "check_list": [
            "检查模块是否安装（pip install）",
            "检查模块名拼写",
            "检查虚拟环境是否激活",
            "检查 sys.path / PYTHONPATH",
        ],
        "severity": "high",
        "fix_template": "安装缺失模块：pip install {module}",
        "what_happened": "Python 找不到这个模块。",
        "why_it_happened": "通常就是没装，或者装在了别的环境里（虚拟环境没激活）。",
        "how_to_avoid": [
            "用 requirements.txt / pyproject.toml 锁依赖",
            "运行前确认激活了正确的虚拟环境",
            "用 python -c 'import sys; print(sys.executable)' 确认解释器",
        ],
    },
    "SyntaxError": {
        "root_cause": "代码语法不符合 Python 规范",
        "check_list": [
            "检查括号/引号是否成对",
            "检查缩进（tab 和空格别混用）",
            "检查冒号是否漏写（if/for/def/class 后）",
            "检查是否漏了续行符或逗号",
            "检查 Python 版本特性（如 f-string、海象运算符）",
        ],
        "severity": "high",
        "fix_template": "修正 {detail} 处的语法（括号/缩进/冒号）",
        "what_happened": "代码写法不符合 Python 语法。",
        "why_it_happened": "括号没闭合、缩进混了 tab 和空格、if 后漏冒号这类问题——"
        "Python 解析阶段就过不去。",
        "how_to_avoid": [
            "用支持语法高亮和括号匹配的编辑器",
            "统一用 4 个空格缩进，别用 tab",
            "保存时跑 ruff/black 自动格式化",
        ],
    },
    "ZeroDivisionError": {
        "root_cause": "除数为零",
        "check_list": [
            "检查除数变量是否可能为 0",
            "检查整数除法 / 取模运算",
            "检查从外部输入拿到的除数",
            "检查统计计算中分母（如方差为零时的标准化）",
        ],
        "severity": "low",
        "fix_template": "除法前判断除数 != 0，或用 try/except 兜底",
        "what_happened": "拿 0 做了除数。",
        "why_it_happened": "除数变量恰好是 0——常见于空数据求平均、归一化时分母为 0。",
        "how_to_avoid": [
            "除法前 if denominator: 守一下",
            "用 try/except ZeroDivisionError 兜底",
            "统计计算里对零方差单独处理",
        ],
    },
    "FileNotFoundError": {
        "root_cause": "要打开的文件不存在",
        "check_list": [
            "检查文件路径是否正确（绝对/相对）",
            "检查工作目录（os.getcwd()）",
            "检查路径分隔符（Windows 用 \\，跨平台用 os.path.join）",
            "检查文件名拼写和扩展名",
            "检查文件是否真的在该位置存在",
        ],
        "severity": "medium",
        "fix_template": "打开前检查 os.path.exists({path})，或用 try/except",
        "what_happened": "试图打开一个不存在的文件。",
        "why_it_happened": "路径写错了，或者运行时的工作目录和你以为的不一样——"
        "相对路径最容易踩坑。",
        "how_to_avoid": [
            "用 pathlib.Path 处理路径，跨平台",
            "用绝对路径或在程序入口处 os.chdir 到固定目录",
            "打开前 assert path.exists()，给个明确的错误",
        ],
    },
    "RecursionError": {
        "root_cause": "递归层数超过 Python 限制（默认 1000）",
        "check_list": [
            "检查递归终止条件是否真的会触发",
            "检查终止条件是否写在了递归调用之前",
            "检查输入规模是否会导致过深递归",
            "考虑改成迭代实现",
            "考虑用 functools.lru_cache 减少重复递归",
        ],
        "severity": "medium",
        "fix_template": "检查 {func} 的终止条件，或改写为迭代",
        "what_happened": "函数自己调用自己太多次，超过 Python 的递归深度限制。",
        "why_it_happened": "通常不是真的需要那么深的递归，而是终止条件没写对，"
        "导致无限递归；或者问题规模确实大，该用迭代。",
        "how_to_avoid": [
            "递归第一行就写终止条件",
            "对深度不确定的，改成迭代 + 显式栈",
            "动态规划问题优先用记忆化或自底向上",
        ],
    },
}

# 默认兜底（未知异常类型）
_DEFAULT_ENTRY: dict[str, Any] = {
    "root_cause": "未分类的异常",
    "check_list": [
        "仔细阅读错误消息全文",
        "检查最近一次代码改动",
        "在出错位置加 print / logging 验证变量状态",
        "搜索该错误的常见原因",
    ],
    "severity": "medium",
    "fix_template": "根据错误消息 {msg} 定位问题",
    "what_happened": "程序抛出了一个异常。",
    "why_it_happened": "需要结合错误消息和上下文判断——可能是输入数据、状态、或代码逻辑问题。",
    "how_to_avoid": [
        "加输入校验",
        "用 try/except 处理已知的失败路径",
        "写单元测试覆盖边界情况",
    ],
}


def _get_kb_entry(error_type: str) -> dict[str, Any]:
    """根据异常类型名取知识库条目，找不到就用默认。"""
    if not error_type:
        return _DEFAULT_ENTRY
    # 精确匹配
    if error_type in _KB:
        return _KB[error_type]
    # 去掉模块前缀再试（如 builtins.ValueError）
    short = error_type.rsplit(".", 1)[-1]
    return _KB.get(short, _DEFAULT_ENTRY)


# ---------------------------------------------------------------------------
# traceback 解析
# ---------------------------------------------------------------------------
# 匹配每一帧：File "path", line N, in func\n    code
_FRAME_RE = re.compile(
    r'File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+)'
    r'(?:,\s+in\s+(?P<func>[^\s]+))?\s*\n\s*(?P<code>.*)',
    re.MULTILINE,
)

# SyntaxError 特殊格式：行号前没有 "in func"
_SYNTAX_FRAME_RE = re.compile(
    r'File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+)\s*\n'
    r'(?:\s*(?P<pointer>[~\^ ]+)\n)?\s*(?P<code>.*)',
    re.MULTILINE,
)


def _parse_traceback_text(text: str) -> dict[str, Any]:
    """纯正则解析 traceback 文本，返回结构化结果。"""
    if not text or not text.strip():
        return {
            "error_type": None,
            "error_message": None,
            "frames": [],
            "innermost_frame": None,
            "outermost_frame": None,
        }

    frames: list[dict[str, Any]] = []
    for m in _FRAME_RE.finditer(text):
        frames.append(
            {
                "file": m.group("file"),
                "line": int(m.group("line")),
                "function": m.group("func") or "<module>",
                "code": (m.group("code") or "").strip(),
            }
        )

    # 提取最终错误行：取 traceback 里最后一个 "Type: msg" 模式
    error_type: str | None = None
    error_message: str | None = None

    # 先按行扫，找最后一个看起来像 "XxxError: ..." 的行
    last_error_match = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(
            r'^([A-Za-z_][\w\.]*(?:Error|Exception|Warning|IterationError))\s*:\s*(.*)$',
            line,
        )
        if m:
            last_error_match = m

    if last_error_match:
        error_type = last_error_match.group(1)
        error_message = last_error_match.group(2)
    else:
        # 兜底：如果用户只给了消息没给类型，整体当作 message
        # 取最后一行非空内容
        non_empty = [l.strip() for l in text.splitlines() if l.strip()]
        if non_empty:
            error_message = non_empty[-1]

    # SyntaxError 没有 "in func"，上面的 _FRAME_RE 可能漏掉部分
    # 用 _SYNTAX_FRAME_RE 补一遍，合并去重
    if not frames:
        for m in _SYNTAX_FRAME_RE.finditer(text):
            frames.append(
                {
                    "file": m.group("file"),
                    "line": int(m.group("line")),
                    "function": "<module>",
                    "code": (m.group("code") or "").strip(),
                }
            )

    innermost = frames[-1] if frames else None
    outermost = frames[0] if frames else None

    return {
        "error_type": error_type,
        "error_message": error_message,
        "frames": frames,
        "innermost_frame": innermost,
        "outermost_frame": outermost,
    }


# ---------------------------------------------------------------------------
# 从错误消息里抽细节（变量名/类型等），用于修复模板填充
# ---------------------------------------------------------------------------
def _extract_details(error_type: str | None, error_message: str | None) -> dict[str, str]:
    """从错误消息里抠出关键字段，供模板填充。"""
    details: dict[str, str] = {"var": "?", "arg": "?", "expected": "?", "actual": "?",
                               "obj": "?", "attr": "?", "seq": "?", "idx": "?",
                               "key": "?", "module": "?", "func": "?", "value": "?",
                               "path": "?", "detail": "?"}
    if not error_message:
        return details

    msg = error_message

    # name 'x' is not defined
    m = re.search(r"name '([^']+)' is not defined", msg)
    if m:
        details["var"] = m.group(1)

    # 'int' object is not iterable  /  'NoneType' object has no attribute 'foo'
    m = re.search(r"'([^']+)' object has no attribute '([^']+)'", msg)
    if m:
        details["obj"] = m.group(1)
        details["attr"] = m.group(2)

    m = re.search(r"'([^']+)' object is not iterable", msg)
    if m:
        details["actual"] = m.group(1)

    # unsupported operand type(s) for +: 'int' and 'str'
    m = re.search(
        r"unsupported operand type\(s\) for (.+?): '([^']+)' and '([^']+)'", msg
    )
    if m:
        details["arg"] = m.group(1)
        details["expected"] = m.group(2)
        details["actual"] = m.group(3)

    # argument must be a string / ...
    m = re.search(r"argument must be (?:a|an) ([^,;]+)", msg)
    if m:
        details["expected"] = m.group(1).strip()

    # list index out of range
    if "list index out of range" in msg:
        details["seq"] = "list"
        details["idx"] = "?"

    # KeyError: 'foo'  —— 消息就是 'foo'
    if error_type == "KeyError":
        m = re.match(r"^\s*'([^']+)'", msg)
        if m:
            details["key"] = m.group(1)

    # No module named 'foo'
    m = re.search(r"No module named '([^']+)'", msg)
    if m:
        details["module"] = m.group(1)

    # cannot import name 'foo' from 'bar'
    m = re.search(r"cannot import name '([^']+)' from '([^']+)'", msg)
    if m:
        details["var"] = m.group(1)
        details["module"] = m.group(2)

    # No such file or directory: 'path'
    m = re.search(r"No such file or directory: '([^']+)'", msg)
    if m:
        details["path"] = m.group(1)
    m = re.search(r"\[Errno 2\] No such file or directory: '([^']+)'", msg)
    if m:
        details["path"] = m.group(1)

    # division by zero
    if "division by zero" in msg:
        details["detail"] = "division by zero"

    # maximum recursion depth exceeded
    if "maximum recursion depth" in msg:
        m = re.search(r"in comparison|in (\w+)", msg)
        details["func"] = m.group(1) if m else "?"

    return details


# ---------------------------------------------------------------------------
# 用 ast 扫 code_snippet，找可疑行
# ---------------------------------------------------------------------------
def _scan_code_snippet(code: str, error_type: str | None) -> list[dict[str, Any]]:
    """对 code_snippet 做 ast 扫描，返回可疑行列表。失败就返回空。"""
    suspicious: list[dict[str, Any]] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # 代码本身就有语法错，直接标出来
        suspicious.append(
            {
                "line": 0,
                "reason": "代码片段本身存在语法错误，无法用 ast 解析",
                "snippet": code.splitlines()[:3],
            }
        )
        return suspicious

    for node in ast.walk(tree):
        # NameError 嫌疑：用了名字但没在当前片段里赋值
        if error_type == "NameError" and isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                # 粗略判断：片段里有没有这个名字的赋值
                assigned = {
                    n.id
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Name)
                    and isinstance(n.ctx, ast.Store)
                }
                if node.id not in assigned and not node.id.isupper():
                    suspicious.append(
                        {
                            "line": getattr(node, "lineno", 0),
                            "reason": f"名字 '{node.id}' 在片段里没看到赋值或参数定义",
                            "name": node.id,
                        }
                    )

        # AttributeError 嫌疑：对 None 字面量取属性
        if error_type == "AttributeError" and isinstance(node, ast.Attribute):
            val = node.value
            if isinstance(val, ast.Constant) and val.value is None:
                suspicious.append(
                    {
                        "line": getattr(node, "lineno", 0),
                        "reason": "对 None 取属性，几乎肯定要炸",
                        "attr": node.attr,
                    }
                )

        # ZeroDivisionError 嫌疑：除以字面量 0
        if error_type == "ZeroDivisionError" and isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                right = node.right
                if isinstance(right, ast.Constant) and right.value == 0:
                    suspicious.append(
                        {
                            "line": getattr(node, "lineno", 0),
                            "reason": "除以字面量 0",
                        }
                    )

        # IndexError 嫌疑：对常量空列表取下标
        if error_type == "IndexError" and isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.List) and len(node.value.elts) == 0:
                suspicious.append(
                    {
                        "line": getattr(node, "lineno", 0),
                        "reason": "对空列表字面量取下标",
                    }
                )

    # 去重（同一行同一原因只留一条）
    seen: set[tuple[int, str]] = set()
    deduped: list[dict[str, Any]] = []
    for s in suspicious:
        key = (s.get("line", 0), s.get("reason", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped


# ---------------------------------------------------------------------------
# 主工具类
# ---------------------------------------------------------------------------
class DebuggerTool(HuginnTool):
    """代码调试工具：解析 traceback、定位根因、给修复建议、通俗解释。"""

    name = "debugger_tool"
    category = "design"
    description = (
        "代码调试工具：解析 Python traceback、分析根因、给出修复建议、"
        "用通俗语言解释错误。纯规则分析，不调 LLM。"
    )
    input_schema = DebuggerInput

    def is_read_only(self, args: DebuggerInput) -> bool:
        return True

    async def call(self, args: DebuggerInput, context: ToolContext) -> ToolResult:
        try:
            if args.language.lower() != "python":
                # 当前只支持 python，但不要硬报错，给个提示走默认流程
                pass

            if args.action == "parse_traceback":
                return self._do_parse(args)
            elif args.action == "analyze_root_cause":
                return self._do_analyze(args)
            elif args.action == "suggest_fix":
                return self._do_suggest(args)
            elif args.action == "explain_error":
                return self._do_explain(args)
            else:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"未知 action: {args.action}",
                )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

    # ---- parse_traceback ----
    def _do_parse(self, args: DebuggerInput) -> ToolResult:
        text = args.traceback_text or args.error_message
        if not text:
            return ToolResult(
                data=None,
                success=False,
                error="parse_traceback 需要 traceback_text 或 error_message",
            )
        parsed = _parse_traceback_text(text)

        # 补上下文行（如果给了 file_path 和 code_snippet）
        if args.file_path and parsed["innermost_frame"]:
            parsed["innermost_frame"]["context_file"] = args.file_path

        summary = (
            f"解析完成：{parsed['error_type'] or '未知异常'}"
            f" — {parsed['error_message'] or '无消息'}"
            f"（共 {len(parsed['frames'])} 帧）"
        )
        return ToolResult(
            data={
                "action": "parse_traceback",
                "parsed": parsed,
                "summary": summary,
            },
            success=True,
        )

    # ---- analyze_root_cause ----
    def _do_analyze(self, args: DebuggerInput) -> ToolResult:
        error_type, error_message = self._resolve_type_and_msg(args)
        if not error_type and not error_message:
            return ToolResult(
                data=None,
                success=False,
                error="analyze_root_cause 需要 traceback_text 或 error_message",
            )

        entry = _get_kb_entry(error_type or "")
        analysis = {
            "error_type": error_type,
            "error_message": error_message,
            "root_cause": entry["root_cause"],
            "check_list": entry["check_list"],
            "severity": entry["severity"],
        }
        summary = (
            f"根因分析：{error_type or '未知'} 严重度 {entry['severity']} — "
            f"{entry['root_cause']}"
        )
        return ToolResult(
            data={
                "action": "analyze_root_cause",
                "analysis": analysis,
                "summary": summary,
            },
            success=True,
        )

    # ---- suggest_fix ----
    def _do_suggest(self, args: DebuggerInput) -> ToolResult:
        error_type, error_message = self._resolve_type_and_msg(args)
        if not error_type and not error_message and not args.code_snippet:
            return ToolResult(
                data=None,
                success=False,
                error="suggest_fix 需要 traceback_text / error_message / code_snippet 至少一个",
            )

        entry = _get_kb_entry(error_type or "")
        details = _extract_details(error_type, error_message)

        suggestions: list[dict[str, Any]] = []

        # 主修复建议：基于模板
        try:
            fix_text = entry["fix_template"].format(
                var=details["var"],
                arg=details["arg"],
                expected=details["expected"],
                actual=details["actual"],
                obj=details["obj"],
                attr=details["attr"],
                seq=details["seq"],
                idx=details["idx"],
                key=details["key"],
                module=details["module"],
                func=details["func"],
                value=details["value"],
                path=details["path"],
                detail=details["detail"],
                msg=error_message or "",
            )
        except (KeyError, IndexError):
            fix_text = entry["fix_template"]

        code_example = self._code_example(error_type, details)

        suggestions.append(
            {
                "fix": fix_text,
                "confidence": "high" if error_type else "medium",
                "explanation": entry["root_cause"],
                "code_example": code_example,
            }
        )

        # 用 ast 扫 code_snippet，追加可疑行建议
        if args.code_snippet:
            suspicious = _scan_code_snippet(args.code_snippet, error_type)
            for s in suspicious:
                suggestions.append(
                    {
                        "fix": f"检查第 {s.get('line', '?')} 行：{s.get('reason', '')}",
                        "confidence": "medium",
                        "explanation": s.get("reason", ""),
                        "code_example": None,
                    }
                )

        # 检查项也作为补充建议
        for check in entry["check_list"][:2]:
            suggestions.append(
                {
                    "fix": check,
                    "confidence": "low",
                    "explanation": "通用检查项",
                    "code_example": None,
                }
            )

        summary = f"共给出 {len(suggestions)} 条修复建议（{error_type or '未知异常'}）"
        return ToolResult(
            data={
                "action": "suggest_fix",
                "suggestions": suggestions,
                "summary": summary,
            },
            success=True,
        )

    # ---- explain_error ----
    def _do_explain(self, args: DebuggerInput) -> ToolResult:
        error_type, error_message = self._resolve_type_and_msg(args)
        if not error_type and not error_message:
            return ToolResult(
                data=None,
                success=False,
                error="explain_error 需要 traceback_text 或 error_message",
            )

        entry = _get_kb_entry(error_type or "")
        explanation = {
            "error_type": error_type,
            "error_message": error_message,
            "what_happened": entry["what_happened"],
            "why_it_happened": entry["why_it_happened"],
            "how_to_avoid": entry["how_to_avoid"],
        }
        summary = f"通俗解释：{error_type or '未知异常'} — {entry['what_happened']}"
        return ToolResult(
            data={
                "action": "explain_error",
                "explanation": explanation,
                "summary": summary,
            },
            success=True,
        )

    # ---- 辅助：从输入里统一拿到 error_type / error_message ----
    def _resolve_type_and_msg(
        self, args: DebuggerInput
    ) -> tuple[str | None, str | None]:
        """优先从 traceback 解析；没 traceback 就用 error_message 反推类型。"""
        if args.traceback_text:
            parsed = _parse_traceback_text(args.traceback_text)
            return parsed["error_type"], parsed["error_message"]

        msg = args.error_message
        if not msg:
            return None, None

        # 尝试从单独的 error_message 里抠类型：形如 "NameError: foo"
        m = re.match(
            r'^([A-Za-z_][\w\.]*(?:Error|Exception|Warning|IterationError))\s*:\s*(.*)$',
            msg.strip(),
        )
        if m:
            return m.group(1), m.group(2)

        # 只有消息没有类型——靠关键词猜一下
        lower = msg.lower()
        if "no module named" in lower:
            return "ModuleNotFoundError", msg
        if "is not defined" in lower:
            return "NameError", msg
        if "division by zero" in lower:
            return "ZeroDivisionError", msg
        if "maximum recursion depth" in lower:
            return "RecursionError", msg
        if "no such file or directory" in lower:
            return "FileNotFoundError", msg
        if "index out of range" in lower:
            return "IndexError", msg
        return None, msg

    # ---- 针对每种异常生成修复代码示例 ----
    def _code_example(self, error_type: str | None, details: dict[str, str]) -> str | None:
        """给每种异常配一个最小修复示例。"""
        if error_type == "NameError":
            var = details["var"]
            return (
                f"# 方案1：补定义\n"
                f"{var} = ...  # 在使用前赋值\n\n"
                f"# 方案2：补 import（如果是漏导入）\n"
                f"import {var}\n\n"
                f"# 方案3：拼写检查\n"
                f"# 搜一下 {var} 在项目里到底叫什么"
            )
        if error_type == "TypeError":
            return (
                "# 检查实际类型并转换\n"
                "if not isinstance(x, expected_type):\n"
                "    x = expected_type(x)\n"
                "result = func(x)"
            )
        if error_type == "AttributeError":
            attr = details["attr"]
            return (
                "# 先判 None 再取属性\n"
                f"if obj is not None:\n"
                f"    obj.{attr}\n"
                "else:\n"
                "    # 处理 None 的情况\n"
                "    ..."
            )
        if error_type == "IndexError":
            return (
                "# 访问前判长度\n"
                "if len(seq) > idx:\n"
                "    val = seq[idx]\n"
                "else:\n"
                "    val = default"
            )
        if error_type == "KeyError":
            key = details["key"]
            return (
                "# 用 .get() 兜底\n"
                f"val = d.get('{key}', default_value)\n\n"
                "# 或先判断 in\n"
                f"if '{key}' in d:\n"
                f"    val = d['{key}']"
            )
        if error_type == "ValueError":
            return (
                "# 用 try/except 兜底\n"
                "try:\n"
                "    val = int(s)\n"
                "except ValueError:\n"
                "    val = default"
            )
        if error_type in ("ImportError", "ModuleNotFoundError"):
            module = details["module"]
            return (
                f"# 安装缺失模块\n"
                f"pip install {module}\n\n"
                f"# 或在代码里降级处理\n"
                "try:\n"
                f"    import {module}\n"
                f"except ImportError:\n"
                f"    {module} = None"
            )
        if error_type == "SyntaxError":
            return (
                "# 常见语法坑：\n"
                "# 1. 括号没闭合 → 补上 ( [ {\n"
                "# 2. if/for/def 后漏冒号 → 加 :\n"
                "# 3. tab 和空格混用 → 统一用 4 个空格\n"
                "# 4. 字符串引号不匹配 → 检查 ' 和 \""
            )
        if error_type == "ZeroDivisionError":
            return (
                "# 除法前判除数\n"
                "if denominator != 0:\n"
                "    result = numerator / denominator\n"
                "else:\n"
                "    result = 0  # 或别的兜底值"
            )
        if error_type == "FileNotFoundError":
            path = details["path"]
            return (
                "from pathlib import Path\n\n"
                f"p = Path({path!r})\n"
                "if p.exists():\n"
                "    with p.open() as f:\n"
                "        data = f.read()\n"
                "else:\n"
                "    raise FileNotFoundError(f'找不到文件: {{p}}')"
            )
        if error_type == "RecursionError":
            return (
                "# 检查终止条件写对没\n"
                "def f(n):\n"
                "    if n <= 0:        # ← 必须有这一行\n"
                "        return 0\n"
                "    return f(n - 1)\n\n"
                "# 太深就改迭代\n"
                "def f_iter(n):\n"
                "    acc = 0\n"
                "    for i in range(n):\n"
                "        acc += 1\n"
                "    return acc"
            )
        return None
