"""代码库地图 — 用 tree-sitter 做精确符号提取 + PageRank 排序。

思路接近 Aider 的 repo map:
  1. 扫所有源码文件, 抽出符号定义 (class/function/method) 和引用
  2. 把符号当节点, 引用关系当边, 建邻接表
  3. 跑一遍 PageRank, 拿到每个符号的"重要性"分数
  4. 按 rank 取 top-N, 受 max_tokens 预算限制

降级策略:
  - tree-sitter 已安装 → 精确符号提取 (支持 Python/JS/TS/Rust/Go/Java)
  - tree-sitter 没装 → 回退到 SymbolIndex (Python 走 ast, 其它走正则)
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from huginn.coder.symbol_index import Reference, Symbol, SymbolIndex


# ── tree-sitter 相关 ────────────────────────────────────────────────────

# 每种语言的查询规则: 抓 class / function / method 定义
_TS_QUERIES: dict[str, str] = {
    "python": """
        (class_definition name: (identifier) @name.class)
        (function_definition name: (identifier) @name.function)
    """,
    "javascript": """
        (class_declaration name: (identifier) @name.class)
        (function_declaration name: (identifier) @name.function)
        (method_definition name: (property_identifier) @name.method)
        (function_expression name: (identifier) @name.function)
    """,
    "typescript": """
        (class_declaration name: (type_identifier) @name.class)
        (function_declaration name: (identifier) @name.function)
        (method_definition name: (property_identifier) @name.method)
        (interface_declaration name: (type_identifier) @name.class)
        (function_expression name: (identifier) @name.function)
    """,
    "rust": """
        (function_item name: (identifier) @name.function)
        (struct_item name: (type_identifier) @name.class)
        (enum_item name: (type_identifier) @name.class)
        (trait_item name: (type_identifier) @name.class)
        (impl_item type: (type_identifier) @name.class)
    """,
    "go": """
        (function_declaration name: (identifier) @name.function)
        (method_declaration name: (field_identifier) @name.method)
        (type_declaration (type_spec name: (type_identifier) @name.class))
    """,
    "java": """
        (class_declaration name: (identifier) @name.class)
        (interface_declaration name: (identifier) @name.class)
        (method_declaration name: (identifier) @name.method)
        (constructor_declaration name: (identifier) @name.method)
    """,
}

# 引用查询: 抓 identifier / type_identifier 节点
_TS_REF_QUERIES: dict[str, str] = {
    "python": "(identifier) @ref",
    "javascript": "(identifier) @ref",
    "typescript": "[(identifier) (type_identifier)] @ref",
    "rust": "[(identifier) (type_identifier)] @ref",
    "go": "(identifier) @ref",
    "java": "[(identifier) (type_identifier)] @ref",
}

# 各语言的关键字, 抽引用时跳过
_KEYWORDS: dict[str, set[str]] = {
    "python": {
        "def", "class", "if", "else", "elif", "for", "while", "return", "import",
        "from", "as", "try", "except", "finally", "with", "yield", "lambda", "global",
        "nonlocal", "pass", "break", "continue", "raise", "assert", "del", "in", "is",
        "and", "or", "not", "True", "False", "None", "self", "cls", "async", "await",
    },
    "javascript": {
        "var", "let", "const", "function", "return", "if", "else", "for", "while",
        "do", "switch", "case", "break", "continue", "new", "this", "class", "extends",
        "super", "import", "export", "from", "as", "default", "try", "catch", "finally",
        "throw", "typeof", "instanceof", "in", "of", "true", "false", "null", "undefined",
        "async", "await", "yield", "static", "get", "set", "void", "delete",
    },
    "typescript": {
        "var", "let", "const", "function", "return", "if", "else", "for", "while",
        "do", "switch", "case", "break", "continue", "new", "this", "class", "extends",
        "super", "import", "export", "from", "as", "default", "try", "catch", "finally",
        "throw", "typeof", "instanceof", "in", "of", "true", "false", "null", "undefined",
        "async", "await", "yield", "static", "get", "set", "void", "delete", "interface",
        "type", "enum", "namespace", "public", "private", "protected", "readonly", "abstract",
        "implements", "declare", "keyof", "infer", "is", "never", "unknown", "any", "string",
        "number", "boolean", "object", "symbol", "bigint",
    },
    "rust": {
        "fn", "let", "mut", "if", "else", "for", "while", "loop", "match", "return",
        "struct", "enum", "trait", "impl", "pub", "use", "mod", "crate", "self", "Self",
        "super", "as", "in", "ref", "move", "static", "const", "type", "where", "unsafe",
        "async", "await", "dyn", "abstract", "become", "box", "do", "final", "macro",
        "override", "priv", "typeof", "unsized", "virtual", "yield", "try", "true", "false",
        "Some", "None", "Ok", "Err",
    },
    "go": {
        "func", "var", "const", "type", "struct", "interface", "package", "import",
        "if", "else", "for", "range", "switch", "case", "default", "break", "continue",
        "return", "go", "defer", "select", "chan", "map", "nil", "true", "false",
        "iota", "fallthrough", "goto",
    },
    "java": {
        "public", "private", "protected", "static", "final", "class", "interface",
        "enum", "extends", "implements", "package", "import", "if", "else", "for",
        "while", "do", "switch", "case", "break", "continue", "return", "new", "this",
        "super", "try", "catch", "finally", "throw", "throws", "void", "int", "long",
        "double", "float", "boolean", "char", "byte", "short", "String", "Object",
        "true", "false", "null", "instanceof", "synchronized", "volatile", "transient",
        "native", "abstract", "default", "var",
    },
}

# 文件后缀 → tree-sitter 语言名
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
}

# 跳过这些目录, 跟 SymbolIndex 保持一致
_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", "target", "dist",
    "build", ".huginn_kb", ".chroma", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".kimi", ".kimi-code", ".idea", ".vscode", "site-packages",
}

# 单文件最大尺寸, 超过就跳过 (避免巨型生成文件拖慢)
_MAX_FILE_BYTES = 2 * 1024 * 1024


@dataclass
class _SymbolNode:
    """图里的一个节点 — 一个具体的符号定义。

    parent 用来在输出时把方法嵌套到所属类下面, 没有就 None。
    """

    name: str
    kind: str  # "class" / "function" / "method"
    file: str
    line: int
    col: int = 0
    parent: str | None = None


def _try_import_tree_sitter() -> tuple[Any, dict[str, Any]] | None:
    """尝试导入 tree-sitter 和语言包。

    返回 (Language 类, {语言名: language 对象}) 或 None。
    只要 tree_sitter 主包能导入就返回, 个别语言包没装就在 languages 里缺着,
    后面碰到该后缀文件就走单文件降级。
    """
    try:
        from tree_sitter import Language, Parser  # noqa: F401
    except Exception:
        return None

    lang_modules = {
        "python": "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "rust": "tree_sitter_rust",
        "go": "tree_sitter_go",
        "java": "tree_sitter_java",
    }
    languages: dict[str, Any] = {}
    for lang_name, mod_name in lang_modules.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        # 不同版本 language 包 API:
        #   新版 (>=0.22): mod.language() 返回 ptr (int)
        #   中间版: mod.Language(...) 是 Language 对象
        #   旧版: mod.language() 返回 Language 对象
        lang_obj: Any = None
        if hasattr(mod, "language"):
            try:
                lang_obj = mod.language()
            except Exception:
                lang_obj = None
        if lang_obj is None and hasattr(mod, "Language"):
            try:
                lang_obj = mod.Language
            except Exception:
                lang_obj = None
        if lang_obj is not None:
            languages[lang_name] = lang_obj
    return (Language, languages)


class _TreeSitterExtractor:
    """tree-sitter 符号提取器。

    兼容多版本 API:
      - 新版 (>=0.22): Language(ptr) + Parser(language)
      - 中间版: Language(language_obj) + Parser(language)
      - 旧版 (<0.21): Parser().set_language(language_obj)
    """

    def __init__(self, language_cls: Any, languages: dict[str, Any]) -> None:
        self._Language = language_cls
        self._raw_languages = languages
        self._parser_cache: dict[str, Any] = {}
        self._query_cache: dict[tuple[str, str], Any] = {}
        self._language_obj_cache: dict[str, Any] = {}

    def supports(self, ext: str) -> bool:
        lang = _EXT_TO_LANG.get(ext)
        return lang is not None and lang in self._raw_languages

    def _get_language_obj(self, lang_name: str) -> Any | None:
        if lang_name in self._language_obj_cache:
            return self._language_obj_cache[lang_name]
        raw = self._raw_languages[lang_name]
        lang_obj: Any = None
        # 新版: Language(ptr)
        try:
            lang_obj = self._Language(raw)
        except Exception:
            # 中间/旧版: raw 本身就是 Language 对象
            lang_obj = raw
        self._language_obj_cache[lang_name] = lang_obj
        return lang_obj

    def _get_parser(self, lang_name: str) -> Any | None:
        if lang_name in self._parser_cache:
            return self._parser_cache[lang_name]
        lang_obj = self._get_language_obj(lang_name)
        if lang_obj is None:
            return None

        from tree_sitter import Parser

        parser: Any = None
        # 新版: Parser(language=...)
        try:
            parser = Parser(language=lang_obj)
        except Exception:
            pass
        if parser is None:
            try:
                parser = Parser()
                try:
                    parser.language = lang_obj
                except Exception:
                    parser.set_language(lang_obj)
            except Exception:
                parser = None
        if parser is not None:
            self._parser_cache[lang_name] = parser
        return parser

    def _build_query(self, lang_name: str, source: str) -> Any | None:
        key = (lang_name, source)
        if key in self._query_cache:
            return self._query_cache[key]
        lang_obj = self._get_language_obj(lang_name)
        if lang_obj is None:
            return None
        query: Any = None
        # 新版: Language.query(source)
        try:
            query = lang_obj.query(source)
        except Exception:
            pass
        if query is None:
            try:
                from tree_sitter import Query

                query = Query(lang_obj, source)
            except Exception:
                query = None
        self._query_cache[key] = query
        return query

    def extract(
        self, path: Path, source: str
    ) -> tuple[list[_SymbolNode], list[Reference]]:
        """返回 (符号定义, 引用)。"""
        ext = path.suffix
        lang_name = _EXT_TO_LANG.get(ext)
        if not lang_name or lang_name not in self._raw_languages:
            return [], []

        parser = self._get_parser(lang_name)
        if parser is None:
            return [], []

        try:
            tree = parser.parse(source.encode("utf-8"))
        except Exception:
            return [], []
        if tree is None or tree.root_node is None:
            return [], []

        symbols: list[_SymbolNode] = []
        references: list[Reference] = []

        # ─ 抓定义 ─
        def_query_src = _TS_QUERIES.get(lang_name)
        if def_query_src:
            query = self._build_query(lang_name, def_query_src)
            if query is not None:
                captures = self._run_captures(query, tree.root_node)
                for cap_name, nodes in captures.items():
                    # cap_name 形如 "name.class" / "name.function" / "name.method"
                    kind_tag = cap_name.split(".")[-1] if "." in cap_name else cap_name
                    if kind_tag == "class":
                        sym_kind = "class"
                    elif kind_tag == "method":
                        sym_kind = "method"
                    else:
                        sym_kind = "function"
                    for node in nodes:
                        name_text = self._node_text(node, source)
                        if not name_text:
                            continue
                        parent_name = self._find_parent_class(node, source)
                        # method 必须有父类, 否则降级成 function
                        if sym_kind == "method" and not parent_name:
                            sym_kind = "function"
                        symbols.append(
                            _SymbolNode(
                                name=name_text,
                                kind=sym_kind,
                                file=str(path),
                                line=(node.start_point[0] + 1),
                                col=node.start_point[1],
                                parent=parent_name if sym_kind == "method" else None,
                            )
                        )

        # ─ 抓引用 ─
        ref_query_src = _TS_REF_QUERIES.get(lang_name)
        if ref_query_src:
            query = self._build_query(lang_name, ref_query_src)
            if query is not None:
                captures = self._run_captures(query, tree.root_node)
                kw = _KEYWORDS.get(lang_name, set())
                seen: set[tuple[int, int]] = set()
                for nodes in captures.values():
                    for node in nodes:
                        name_text = self._node_text(node, source)
                        if not name_text or name_text in kw:
                            continue
                        if not re.match(r"^[A-Za-z_]\w*$", name_text):
                            continue
                        key = (node.start_byte, node.end_byte)
                        if key in seen:
                            continue
                        seen.add(key)
                        references.append(
                            Reference(
                                name=name_text,
                                file=str(path),
                                line=(node.start_point[0] + 1),
                                col=node.start_point[1],
                            )
                        )

        return symbols, references

    def _run_captures(self, query: Any, root: Any) -> dict[str, list[Any]]:
        """兼容多版本 captures API。"""
        result: dict[str, list[Any]] = defaultdict(list)
        try:
            captures = query.captures(root)
        except Exception:
            return result
        if isinstance(captures, dict):
            # 新版: {name: [nodes]}
            for name, nodes in captures.items():
                if isinstance(nodes, list):
                    result[name].extend(nodes)
                else:
                    result[name].append(nodes)
        else:
            # 旧版: list[tuple[node, name]]
            try:
                for item in captures:
                    if isinstance(item, tuple) and len(item) == 2:
                        node, name = item
                        result[name].append(node)
            except Exception:
                pass
        return result

    def _node_text(self, node: Any, source: str) -> str:
        try:
            return node.text.decode("utf-8")
        except Exception:
            try:
                return source.encode("utf-8", errors="ignore")[
                    node.start_byte:node.end_byte
                ].decode("utf-8", errors="ignore")
            except Exception:
                return ""

    def _find_parent_class(self, node: Any, source: str) -> str | None:
        """爬父节点找最近的 class 名, 给 method 当 parent。"""
        cur = getattr(node, "parent", None)
        depth = 0
        while cur is not None and depth < 10:
            t = getattr(cur, "type", "")
            if t in (
                "class_definition",
                "class_declaration",
                "interface_declaration",
                "impl_item",
                "struct_item",
                "enum_item",
                "trait_item",
            ):
                for child in getattr(cur, "children", []) or []:
                    if getattr(child, "type", "") in (
                        "identifier",
                        "type_identifier",
                    ):
                        return self._node_text(child, source)
            cur = getattr(cur, "parent", None)
            depth += 1
        return None


# ── 降级路径: 复用 SymbolIndex 的 ast + 正则 ─────────────────────────────


def _extract_python_with_ast(
    path: Path, source: str
) -> tuple[list[_SymbolNode], list[Reference]]:
    """Python 用 ast 模块抽符号, 顺便记下方法所属的类名。"""
    symbols: list[_SymbolNode] = []
    references: list[Reference] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[ast.AST] = []

        def _current_class(self) -> str | None:
            for n in reversed(self.stack):
                if isinstance(n, ast.ClassDef):
                    return n.name
            return None

        def visit(self, node: ast.AST) -> None:
            self.stack.append(node)
            super().visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._handle_func(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._handle_func(node)
            self.generic_visit(node)

        def _handle_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            cls = self._current_class()
            if cls is not None:
                symbols.append(
                    _SymbolNode(
                        name=node.name,
                        kind="method",
                        file=str(path),
                        line=node.lineno,
                        col=node.col_offset,
                        parent=cls,
                    )
                )
            else:
                symbols.append(
                    _SymbolNode(
                        name=node.name,
                        kind="function",
                        file=str(path),
                        line=node.lineno,
                        col=node.col_offset,
                    )
                )

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            symbols.append(
                _SymbolNode(
                    name=node.name,
                    kind="class",
                    file=str(path),
                    line=node.lineno,
                    col=node.col_offset,
                )
            )
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                references.append(
                    Reference(
                        name=node.id,
                        file=str(path),
                        line=node.lineno,
                        col=node.col_offset,
                    )
                )

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.ctx, ast.Load):
                references.append(
                    Reference(
                        name=node.attr,
                        file=str(path),
                        line=node.lineno,
                        col=node.col_offset,
                    )
                )
            self.generic_visit(node)

    _Visitor().visit(tree)
    return symbols, references


_CLASS_RE = re.compile(
    r"^\s*(?:export\s+)?(?:class|struct|interface|trait|enum)\s+(\w+)", re.M
)
_FUNC_RE = re.compile(
    r"^\s*(?:export\s+)?(?:function|def|fn|func)\s+(\w+)\s*\(", re.M
)
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


def _extract_generic_with_regex(
    path: Path, source: str
) -> tuple[list[_SymbolNode], list[Reference]]:
    """非 Python 文件走正则, 跟 SymbolIndex._index_generic 同一套规则。

    不区分 method/function, 也没法可靠地嵌套, 输出时按 function 处理。
    """
    symbols: list[_SymbolNode] = []
    references: list[Reference] = []

    for match in _CLASS_RE.finditer(source):
        line = source[: match.start()].count("\n") + 1
        symbols.append(
            _SymbolNode(
                name=match.group(1),
                kind="class",
                file=str(path),
                line=line,
            )
        )
    for match in _FUNC_RE.finditer(source):
        line = source[: match.start()].count("\n") + 1
        symbols.append(
            _SymbolNode(
                name=match.group(1),
                kind="function",
                file=str(path),
                line=line,
            )
        )

    # 引用就抓所有 identifier
    for match in _IDENT_RE.finditer(source):
        line = source[: match.start()].count("\n") + 1
        col = match.start() - source.rfind("\n", 0, match.start()) - 1
        references.append(
            Reference(
                name=match.group(0),
                file=str(path),
                line=line,
                col=col,
            )
        )

    return symbols, references


# ── RepoMap 主体 ────────────────────────────────────────────────────────


class RepoMap:
    """代码库地图 — 用 tree-sitter 做精确符号提取 + PageRank 排序。

    降级策略:
      1. tree-sitter 已安装 → 精确符号提取
      2. tree-sitter 没装 → 回退到 SymbolIndex (ast + 正则)
    """

    def __init__(
        self,
        root: str | Path,
        max_tokens: int = 4096,
        rank_threshold: float = 0.0,
    ) -> None:
        self.root = Path(root).resolve()
        self.max_tokens = max_tokens
        self.rank_threshold = rank_threshold

        # tree-sitter 探测, 失败就 None
        ts = _try_import_tree_sitter()
        self._ts_extractor: _TreeSitterExtractor | None = None
        if ts is not None and ts[1]:
            self._ts_extractor = _TreeSitterExtractor(*ts)

        # 数据
        self._symbols: list[_SymbolNode] = []
        self._references: list[Reference] = []
        # 节点 id = 在 self._symbols 里的 index
        self._out_edges: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._in_edges: dict[int, set[int]] = defaultdict(set)
        self._ranks: list[float] = []
        # 按符号名建索引, 方便查引用目标
        self._defs_by_name: dict[str, list[int]] = defaultdict(list)
        # 按文件分组, 方便查文件内符号
        self._symbols_by_file: dict[str, list[int]] = defaultdict(list)

        self._built = False

    # ── 构建阶段 ──────────────────────────────────────────────────────

    def build(self) -> None:
        """构建符号图: 提取所有文件的符号定义和引用, 建邻接表。"""
        self._symbols = []
        self._references = []
        self._out_edges.clear()
        self._in_edges.clear()
        self._defs_by_name.clear()
        self._symbols_by_file.clear()

        if self._ts_extractor is not None:
            self._build_with_tree_sitter()
        else:
            self._build_with_symbol_index()

        # 建符号名索引
        for i, sym in enumerate(self._symbols):
            self._defs_by_name[sym.name].append(i)
            self._symbols_by_file[sym.file].append(i)

        # 建图: 引用 → 边
        self._build_graph()

        # PageRank
        self._ranks = self._pagerank()

        self._built = True

    def _build_with_tree_sitter(self) -> None:
        """tree-sitter 路径: 自己 walk + extract, 个别语言没装包就走单文件降级。"""
        assert self._ts_extractor is not None
        for path in self._walk_source_files():
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if self._ts_extractor.supports(path.suffix):
                syms, refs = self._ts_extractor.extract(path, source)
            elif path.suffix == ".py":
                syms, refs = _extract_python_with_ast(path, source)
            else:
                syms, refs = _extract_generic_with_regex(path, source)

            self._symbols.extend(syms)
            self._references.extend(refs)

    def _build_with_symbol_index(self) -> None:
        """降级路径: 复用 SymbolIndex 走 ast + 正则。

        SymbolIndex 没记方法所属的类名, 我们对 Python 文件二次扫描补上。
        """
        idx = SymbolIndex(self.root)
        idx.build()

        # 把 Symbol 转成 _SymbolNode, Python method 补 parent
        py_sources: dict[str, str] = {}
        for sym in idx.symbols():
            parent: str | None = None
            if sym.kind == "method" and sym.file.endswith(".py"):
                src = py_sources.get(sym.file)
                if src is None:
                    try:
                        src = Path(sym.file).read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        src = ""
                    py_sources[sym.file] = src
                if src:
                    parent = _find_python_parent_class(src, sym.line)
            self._symbols.append(
                _SymbolNode(
                    name=sym.name,
                    kind=sym.kind,
                    file=sym.file,
                    line=sym.line,
                    col=sym.col,
                    parent=parent,
                )
            )

        self._references = list(idx.references())

    def _walk_source_files(self) -> Iterable[Path]:
        """遍历所有源码文件, 应用 SKIP_DIRS 和大小过滤。"""
        for ext in _EXT_TO_LANG:
            for path in self.root.rglob(f"*{ext}"):
                if self._should_skip(path):
                    continue
                yield path

    def _should_skip(self, path: Path) -> bool:
        for part in path.parts:
            if part in _SKIP_DIRS:
                return True
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                return True
        except Exception:
            return True
        return False

    def _build_graph(self) -> None:
        """建邻接表: 文件 A 引用了文件 B 定义的符号 → A 里的符号 → B 里的符号。

        边权 = 引用次数 (同一个 source-target 对出现几次就累加几次)。
        为了让 PageRank 有意义, 边方向是 "引用者 → 被引用者",
        这样被广泛引用的符号会拿到高 rank。
        """
        # 给每个 (file, name) 一个稳定的定义列表, 引用来了直接查
        # 已经在 _defs_by_name 里建好了: name -> [node_id, ...]

        # 引用方所在的"容器符号": 找引用所在文件中行号最接近的符号定义
        # 这样能把引用归到具体函数/方法上, 边更精确
        for ref in self._references:
            targets = self._defs_by_name.get(ref.name)
            if not targets:
                continue
            # 容器: ref.file 中 line <= ref.line 的最大 line 的符号
            container_ids = self._find_containing_symbol_ids(ref.file, ref.line)
            if not container_ids:
                # 该引用所在文件没抽出任何符号, 跳过
                continue
            for src_id in container_ids:
                src_sym = self._symbols[src_id]
                for tgt_id in targets:
                    tgt_sym = self._symbols[tgt_id]
                    # 不自环: 同一符号定义不连自己
                    if src_id == tgt_id:
                        continue
                    # 跨文件才连, 避免同文件内互引刷分
                    if src_sym.file == tgt_sym.file:
                        continue
                    self._out_edges[src_id][tgt_id] += 1
                    self._in_edges[tgt_id].add(src_id)

    def _find_containing_symbol_ids(self, file: str, line: int) -> list[int]:
        """找出 file 里包含 line 的最内层符号 (一般是 1 个, 可能 0 个)。

        规则: 在该文件所有符号里, 找 line <= 目标行 且 line 最大的那个。
        如果该符号是 method, 还要把它的父 class 也算上 (因为 class 也"包含"这段代码)。
        """
        ids_in_file = self._symbols_by_file.get(file)
        if not ids_in_file:
            return []

        # 找最内层: line 最大且 <= 目标
        best_id = -1
        best_line = -1
        for sid in ids_in_file:
            sym = self._symbols[sid]
            if sym.line <= line and sym.line > best_line:
                best_line = sym.line
                best_id = sid

        if best_id < 0:
            return []

        result = [best_id]
        # 如果最内层是 method, 把父 class 也加上 (class 节点也"持有"这段代码)
        sym = self._symbols[best_id]
        if sym.kind == "method" and sym.parent:
            for sid in ids_in_file:
                other = self._symbols[sid]
                if other.kind == "class" and other.name == sym.parent:
                    result.append(sid)
                    break
        return result

    def _pagerank(
        self, damping: float = 0.85, iterations: int = 20, tol: float = 1e-6
    ) -> list[float]:
        """简单 PageRank: 迭代 20 次或收敛。

        dangling 节点 (没有出边的) 把 rank 平均分给所有节点, 防止 rank 流失。
        """
        n = len(self._symbols)
        if n == 0:
            return []

        # 预算每个节点的出度 (用边权之和)
        out_weight: list[float] = [0.0] * n
        for u, targets in self._out_edges.items():
            out_weight[u] = float(sum(targets.values()))

        r = [1.0 / n] * n
        base = (1.0 - damping) / n
        for _ in range(iterations):
            new_r = [base] * n
            # dangling 节点的 rank 平均分给所有节点
            dangling_sum = 0.0
            for u in range(n):
                if out_weight[u] == 0.0:
                    dangling_sum += r[u]
            dangling_share = damping * dangling_sum / n
            for v in range(n):
                new_r[v] += dangling_share

            # 累加正常边的贡献
            for u, targets in self._out_edges.items():
                if out_weight[u] <= 0.0:
                    continue
                share = damping * r[u] / out_weight[u]
                for v, w in targets.items():
                    new_r[v] += share * w

            diff = sum(abs(new_r[i] - r[i]) for i in range(n))
            r = new_r
            if diff < tol:
                break

        return r

    # ── 输出阶段 ──────────────────────────────────────────────────────

    def get_map(self, query: str | None = None) -> str:
        """生成 repo map 文本。

        - 无 query: 返回全局 top-N 符号 (按 PageRank 排序)
        - 有 query: 返回与 query 相关的符号 (按相关性 + PageRank 排序)

        格式类似 Aider:
            src/huginn/agent.py
              class HuginnAgent (line 168)
                def chat (line 1312)
                def invoke (line 1586)
        """
        if not self._built:
            self.build()

        if not self._symbols:
            return "(空 repo map — 没找到任何符号)"

        # 估算每行 token 数, 给预算留点余量
        # 一行大概 "  def func_name (line 1234)" 约 10-15 tokens, 取 12
        tokens_per_line = 12
        budget_lines = max(8, self.max_tokens // tokens_per_line)

        if query:
            selected = self._select_for_query(query, budget_lines)
        else:
            selected = self._select_global(budget_lines)

        if not selected:
            return "(没有符号满足阈值)"

        return self._render(selected, budget_lines)

    def _select_global(self, budget_lines: int) -> list[int]:
        """全局 top-N: 按 rank 降序, 过滤掉低于 threshold 的。"""
        ranked = sorted(
            range(len(self._symbols)),
            key=lambda i: self._ranks[i],
            reverse=True,
        )
        # 过滤低分
        ranked = [i for i in ranked if self._ranks[i] >= self.rank_threshold]
        # 按 token 预算截断
        return ranked[:budget_lines]

    def _select_for_query(self, query: str, budget_lines: int) -> list[int]:
        """查询相关: 先按名称匹配找出种子, 再 BFS 扩展邻居, 最后按 rank 排序。

        query 可以是符号名/文件名片段, 大小写不敏感。
        """
        q = query.strip().lower()
        if not q:
            return self._select_global(budget_lines)

        # 种子: 名字或文件路径命中 query 的符号
        seeds: set[int] = set()
        for i, sym in enumerate(self._symbols):
            if q in sym.name.lower():
                seeds.add(i)
                continue
            rel = self._rel_path(sym.file).lower()
            if q in rel:
                seeds.add(i)

        # 引用里也找一下, 把引用过 query 名字的符号也算种子
        for ref in self._references:
            if q in ref.name.lower():
                # 找该引用所在文件的容器符号
                for cid in self._find_containing_symbol_ids(ref.file, ref.line):
                    seeds.add(cid)

        if not seeds:
            # 没命中, 退化到全局
            return self._select_global(budget_lines)

        # BFS 扩展 1 跳邻居
        extended: set[int] = set(seeds)
        for sid in list(seeds):
            # 出边
            for tgt in self._out_edges.get(sid, {}):
                extended.add(tgt)
            # 入边
            for src in self._in_edges.get(sid, set()):
                extended.add(src)

        # 按 rank 排序
        ranked = sorted(extended, key=lambda i: self._ranks[i], reverse=True)
        ranked = [i for i in ranked if self._ranks[i] >= self.rank_threshold]
        return ranked[:budget_lines]

    def _render(self, selected: list[int], budget_lines: int) -> str:
        """把选中的符号按文件分组, 输出 Aider 风格文本。"""
        # 按文件分组, 每组内按 (parent, line) 排序, 让 method 紧跟所属 class
        by_file: dict[str, list[int]] = defaultdict(list)
        for sid in selected:
            by_file[self._symbols[sid].file].append(sid)

        # 文件顺序: 按组内最高 rank 排, 重要的文件先出
        file_order = sorted(
            by_file.keys(),
            key=lambda f: max(self._ranks[i] for i in by_file[f]),
            reverse=True,
        )

        lines: list[str] = []
        used = 0
        for file in file_order:
            if used >= budget_lines:
                break
            rel = self._rel_path(file)
            lines.append(rel)
            used += 1

            # 组内: class 在前, 它的 method 紧随; 然后是顶层 function
            syms_in_file = by_file[file]
            # 先按 parent 分桶 (None 的放顶层)
            by_parent: dict[str | None, list[int]] = defaultdict(list)
            for sid in syms_in_file:
                by_parent[self._symbols[sid].parent].append(sid)

            # 顶层 = class + 没有 parent 的符号 + parent 类不在选中集的孤儿 method
            # 孤儿 method 直接拉到顶层, 不然文件头下面啥都没有很奇怪
            top_level: list[int] = []
            for sid in syms_in_file:
                sym = self._symbols[sid]
                if sym.kind == "class":
                    top_level.append(sid)
                elif sym.parent is None:
                    top_level.append(sid)
                elif sym.kind == "method":
                    # parent 类没被选进来, 当顶层处理
                    parent_in_set = any(
                        self._symbols[oid].kind == "class"
                        and self._symbols[oid].name == sym.parent
                        for oid in syms_in_file
                    )
                    if not parent_in_set:
                        top_level.append(sid)
            top_level.sort(key=lambda i: self._symbols[i].line)

            for sid in top_level:
                if used >= budget_lines:
                    break
                sym = self._symbols[sid]
                lines.append(f"  {sym.kind} {sym.name} (line {sym.line})")
                used += 1

                # 该 class 下的 method
                if sym.kind == "class":
                    methods = by_parent.get(sym.name, [])
                    methods.sort(key=lambda i: self._symbols[i].line)
                    for mid in methods:
                        if used >= budget_lines:
                            break
                        msym = self._symbols[mid]
                        lines.append(f"    {msym.kind} {msym.name} (line {msym.line})")
                        used += 1

        return "\n".join(lines)

    def get_symbol_context(self, symbol_name: str, depth: int = 1) -> str:
        """获取指定符号的上下文 (定义 + 引用 + 被引用)。

        depth=1 表示扩展一跳邻居: 把该符号作用域里引用的其他符号也列出来。
        """
        if not self._built:
            self.build()

        defs = [i for i, s in enumerate(self._symbols) if s.name == symbol_name]
        if not defs:
            return f"找不到符号: {symbol_name}"

        out: list[str] = []
        out.append(f"# 符号: {symbol_name}")
        out.append("")
        out.append("定义:")
        for sid in defs:
            sym = self._symbols[sid]
            rel = self._rel_path(sym.file)
            out.append(f"  {rel}:{sym.line}  {sym.kind} {sym.name}")

        # 被引用: 哪些文件/符号引用了它
        ref_sites: list[tuple[str, int]] = []
        for ref in self._references:
            if ref.name == symbol_name:
                ref_sites.append((ref.file, ref.line))
        ref_sites.sort()

        if ref_sites:
            out.append("")
            out.append(f"被引用 ({len(ref_sites)} 处):")
            shown = 0
            for f, ln in ref_sites:
                if shown >= 30:
                    out.append(f"  ... 还有 {len(ref_sites) - shown} 处")
                    break
                containers = self._find_containing_symbol_ids(f, ln)
                rel = self._rel_path(f)
                if containers:
                    # 取最内层 (第一个是 method/function)
                    inner = self._symbols[containers[0]]
                    out.append(
                        f"  {rel}:{ln}  in {inner.kind} {inner.name}"
                    )
                else:
                    out.append(f"  {rel}:{ln}")
                shown += 1

        # 引用了哪些其他符号 (depth=1)
        if depth >= 1:
            related: dict[str, int] = defaultdict(int)
            for sid in defs:
                sym = self._symbols[sid]
                # 找该符号作用域内的引用: 行号在 [sym.line, next_sibling_line) 之间
                siblings = sorted(
                    i
                    for i in self._symbols_by_file.get(sym.file, [])
                    if i != sid
                )
                end_line = 10**9
                for sib_id in siblings:
                    sib = self._symbols[sib_id]
                    if sib.line > sym.line:
                        end_line = min(end_line, sib.line)

                for ref in self._references:
                    if ref.file != sym.file:
                        continue
                    if not (sym.line <= ref.line < end_line):
                        continue
                    if ref.name == sym.name:
                        continue
                    # 只记有定义的 (避免把局部变量也列出来)
                    if ref.name in self._defs_by_name:
                        related[ref.name] += 1

            if related:
                out.append("")
                out.append("作用域内引用的其他符号 (top 20):")
                for name, cnt in sorted(related.items(), key=lambda x: -x[1])[:20]:
                    out.append(f"  {name}  ({cnt} 次)")

        return "\n".join(out)

    # ── 辅助 ──────────────────────────────────────────────────────────

    def _rel_path(self, file: str) -> str:
        """转成相对 root 的路径, 用 / 分隔, 跟 Aider 输出风格一致。"""
        try:
            rel = Path(file).resolve().relative_to(self.root)
        except (ValueError, OSError):
            rel = Path(file)
        return str(rel).replace("\\", "/")


# ── Python ast 辅助: 给 SymbolIndex 的 method 补 parent ─────────────────


def _find_python_parent_class(source: str, target_line: int) -> str | None:
    """扫一遍 Python 源码, 找包含 target_line 的最内层 ClassDef 的名字。

    SymbolIndex 没记 parent, 我们这里补一下, 输出 map 时才能把方法嵌套到类下面。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    # 收集所有 (class_name, start_line, end_line)
    # end_line 用最后一个子节点的行号近似
    ranges: list[tuple[str, int, int]] = []

    def _end_line(node: ast.AST) -> int:
        end = getattr(node, "end_lineno", None)
        if end is not None:
            return end
        # 没有 end_lineno 就遍历子节点找最大行号
        max_line = getattr(node, "lineno", 0)
        for child in ast.walk(node):
            ln = getattr(child, "lineno", 0)
            if ln > max_line:
                max_line = ln
        return max_line

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            ranges.append((node.name, node.lineno, _end_line(node)))

    # 找包含 target_line 的最内层 (start 最大的)
    best: str | None = None
    best_start = 0
    for name, start, end in ranges:
        if start <= target_line <= end and start > best_start:
            best_start = start
            best = name
    return best


__all__ = ["RepoMap"]
