"""启动期校验器 —— 探测可选依赖与死代码, 暴露「声明了但没生效」的静默特性.

背景 (B/C 档):
  库里 1000+ 处 try/except 包裹可选第三方 import (torch/pymatgen/ase/fitz...).
  依赖缺失时, 这些 try 会**静默吞掉 ImportError**, 相应工具直接退化/变空 —
  用户无从得知某个功能其实是「死」的. 这个模块在启动期一次性盘出来:

  1. missing_optional_deps(): 用 importlib.util.find_spec 探测可选依赖 (不加载),
     报告缺失项 + 影响工具 + 安装组 (对应 pyproject [all]/[ml-*]/[rag]/[db] 等).
  2. scan_dead_code(): 静态 AST 扫 try/except 包裹的纯 stdlib import
     (stdlib 导入永不会失败, 外圈 try 是纯死重). 每次启动跑全包 AST 太浪费,
     用 HUGINN_STARTUP_DEADCODE=1 门控, 供 CI / 一次性核查用.

  两个检查都不抛异常: 只读探测, 失败即跳过, 调用方自己决定要不要告警强喂。
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# huginn 包根目录 (本文件上一级)
_PKG_ROOT = Path(__file__).resolve().parent

# ── 可选依赖注册表 ──────────────────────────────────────────────────
# module: 可 import 的顶级模块名 (find_spec 用的名字)
# group: 对应的 pyproject extras 组; desc: 影响的功能说明
# ponytail: 只登记「缺了会静默退化」的可选三方重依赖, 核心硬依赖
# (numpy/scipy/sympy/networkx/Pillow...) 由 pip 装包时保证, 不在这查.
_OPTIONAL_DEPS: dict[str, dict[str, str]] = {
    # 材料 / 结构
    "pymatgen":          {"group": "all", "desc": "晶体结构/能带/相图 (materials/结构工具)"},
    "ase":               {"group": "all", "desc": "原子结构 AE/VASP/MD 后端 (ml_potential/descriptor 等)"},
    "dscribe":           {"group": "all", "desc": "结构描述符 (descriptor_tool)"},
    "spglib":            {"group": "all", "desc": "空间群对称 (symmetry_tool)"},
    "matminer":          {"group": "all", "desc": "材料特征 (descriptor_tool)"},
    "py4vasp":           {"group": "all", "desc": "VASP 输出解析 (vasp_tool)"},
    "chgnet":            {"group": "all", "desc": "CHGNet 机器学习势 (ml_potential_tool)"},
    # ML / 科学计算
    "torch":             {"group": "ml-*", "desc": "神经网络/PyTorch (gnn/vae/transformer/autodiff 等)"},
    "torch_geometric":   {"group": "ml-*", "desc": "图神经网络 (gnn_tool)"},
    "sklearn":           {"group": "all", "desc": "ML 模型 (gp_tool/dynamics_discovery 等)"},
    "pandas":            {"group": "all", "desc": "表格数据处理 (active_learning 等)"},
    "jax":               {"group": "ml-*", "desc": "自动微分 (autodiff_tool)"},
    "gpytorch":          {"group": "all", "desc": "高斯过程代理 (interpretable_ml_tool)"},
    "mace":              {"group": "ml-mace", "desc": "MACE 机器学习势 (ml_potential_tool)"},
    "fairchem":          {"group": "ml-fairchem", "desc": "FAIRChem 机器学习势 (ml_potential_tool)"},
    "pynep":             {"group": "ml-nep", "desc": "NEP 机器学习势 (ml_potential_tool, 本地装)"},
    # 优化 / 数学
    "pulp":              {"group": "all", "desc": "线性规划 (numerical_tool)"},
    "cvxpy":             {"group": "all", "desc": "凸优化 (numerical_tool)"},
    "symengine":         {"group": "all", "desc": "符号计算加速 (可选)"},
    # 计算拓扑 / 不确定性
    "ripser":            {"group": "all", "desc": "持久同调 (tda_tool)"},
    "gudhi":             {"group": "all", "desc": "计算拓扑 (tda_tool)"},
    "chaospy":           {"group": "all", "desc": "UQ 不确定性传播 (uq_tool)"},
    # 仿真
    "openmm":            {"group": "all", "desc": "分子动力学 (openmm_tool)"},
    "pybamm":            {"group": "all", "desc": "电池建模 (pybamm_tool)"},
    # 分子 / 对接
    "rdkit":             {"group": "benchmark", "desc": "化学分子 (packing/vina_tool)"},
    "vina":              {"group": "benchmark", "desc": "分子对接 (vina_tool)"},
    # 文档 / 3D / 笔记本
    "trimesh":           {"group": "all", "desc": "3D 网格渲染 (model3d_tool)"},
    "nbformat":          {"group": "all", "desc": "Jupyter notebook 解析 (notebook_tool)"},
    # OCR / 文献
    "fitz":              {"group": "all/rag", "desc": "PDF 解析 (literature 工具)"},
    "easyocr":           {"group": "all/rag", "desc": "OCR 识别 (vision 工具)"},
    "paddleocr":         {"group": "all/rag", "desc": "OCR 识别 (vision 工具, 中文优先)"},
    "pytesseract":       {"group": "all/rag", "desc": "Tesseract OCR (vision 工具, 需系统 tesseract)"},
    "nougat_ocr":        {"group": "all", "desc": "学术 PDF 解析 (文献链路)"},
    # RAG / 检索
    "chromadb":          {"group": "rag", "desc": "向量库 (RAG/工具检索)"},
    "sentence_transformers": {"group": "rag", "desc": "嵌入模型 (RAG/检索)"},
    "tiktoken":          {"group": "all", "desc": "token 计数精确值 (utils/tokens)"},
    "pint":              {"group": "all", "desc": "单位换算 (utils/units)"},
    "thermo":            {"group": "all", "desc": "热物性 (thermo_tool)"},
    "jieba":             {"group": "rag", "desc": "中文分词 (文档检索)"},
    # 网络搜索
    "tavily":            {"group": "all", "desc": "Tavily 搜索后端 (web_search_tool)"},
    "ddgs":              {"group": "all", "desc": "DuckDuckGo 搜索后端 (web_search_tool)"},
    "duckduckgo_search": {"group": "all", "desc": "DuckDuckGo 搜索后端旧版 (web_search_tool)"},
    # 图表
    "scienceplots":      {"group": "all", "desc": "科技论文绘图样式 (figure_ir)"},
    "ultraplot":         {"group": "all", "desc": "绘图 (figure_ir)"},
    # 数据库
    "mp_api":            {"group": "db", "desc": "Materials Project 查询 (需 MP_API_KEY)"},
}

# 模块名不规则: 注册表 key 与实际 import 名不一致时, 给一个映射.
_IMPORT_NAME = {
    "nougat_ocr": "nougat",
}


@dataclass
class DependencyCheck:
    """单个可选依赖的探测结果. present=False 表示缺 → 相关特性静默失效."""

    name: str            # 注册表 key
    module: str          # find_spec 用的模块名
    group: str
    desc: str
    present: bool


def _spec_exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(_IMPORT_NAME.get(module, module)) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def check_missing_deps() -> list[DependencyCheck]:
    """探测注册的可选依赖, 返回全部结果 (present=False 为缺失). 从不抛异常."""
    out: list[DependencyCheck] = []
    for name, meta in _OPTIONAL_DEPS.items():
        try:
            present = _spec_exists(name)
        except Exception:
            present = False
        out.append(DependencyCheck(
            name=name,
            module=_IMPORT_NAME.get(name, name),
            group=meta["group"],
            desc=meta["desc"],
            present=present,
        ))
    return out


def scan_dead_code(root: str | Path | None = None) -> list[dict[str, Any]]:
    """静态扫 try/except 包裹的纯 stdlib import (死重容错).

    每次启动跑全包 AST 太浪费, 由调用方用 HUGINN_STARTUP_DEADCODE=1 门控.
    返回形如 [{module, line, imports, handlers}] 的记录.
    """
    root = Path(root) if root else _PKG_ROOT
    results: list[dict[str, Any]] = []
    for path in root.rglob("*.py"):
        try:
            src = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            # 只有在 try 体全是 import 的前提下才有「包导入」语义
            if not all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in node.body):
                continue
            imports: list[str] = []
            for s in node.body:
                if isinstance(s, ast.Import):
                    imports += [(a.name or "").split(".")[0] for a in s.names]
                elif isinstance(s, ast.ImportFrom):
                    m = (s.module or "").split(".")[0]
                    if m:
                        imports.append(m)
            if not imports:
                continue
            # 纯 stdlib → try 是死重
            std_only = all(_is_stdlib(n) for n in imports)
            if not std_only:
                continue
            handlers = [ast.unparse(h.type) if h.type is not None else "bare"
                        for h in node.handlers]
            results.append({
                "module": str(path.relative_to(root)),
                "line": node.lineno,
                "imports": sorted(set(imports)),
                "handlers": handlers,
            })
    return results


_STDLIB_CACHE: dict[str, bool] = {}


def _is_stdlib(name: str) -> bool:
    """精确判定: find_spec 的 origin 落在标准库目录下才算 stdlib.

    用 sysconfig 的标准库路径做前缀匹配, site-packages / 项目内模块都排除.
    """
    import sysconfig

    if name in _STDLIB_CACHE:
        return _STDLIB_CACHE[name]
    stdlib = str(Path(sysconfig.get_paths()["stdlib"]).resolve())
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        return _cache_stdlib(name, False)
    if not spec or not spec.origin:
        return _cache_stdlib(name, False)
    try:
        origin = str(Path(spec.origin).resolve())
    except Exception:
        return _cache_stdlib(name, False)
    return _cache_stdlib(name, origin.startswith(stdlib))


def _cache_stdlib(name: str, val: bool) -> bool:
    _STDLIB_CACHE[name] = val
    return val


def run_startup_check() -> dict[str, Any]:
    """启动期总入口: 报告缺失可选依赖; 命中 HUGINN_STARTUP_DEADCODE 才扫死代码.

    只返回结构化报告, 不抛异常; 返回的 missing 供调用方决定如何呈现.
    """
    missing = [
        d for d in check_missing_deps()
        if not d.present
    ]
    report: dict[str, Any] = {
        "missing_optional_deps": [
            {
                "name": d.name,
                "group": d.group,
                "desc": d.desc,
                "install": f"pip install -e '.[{d.group}]'",
            }
            for d in missing
        ],
        "dead_code": None,
    }
    if os.environ.get("HUGINN_STARTUP_DEADCODE") == "1":
        report["dead_code"] = scan_dead_code()

    if missing:
        missing_cn = [
            f"{d.name}({d.desc} | {d.group})" for d in missing
        ]
        logger.warning(
            "[startup] %d 个可选功能依赖缺失, 相关工具将静默降级/不可用: %s",
            len(missing),
            ", ".join(missing_cn),
        )
        for d in missing:
            logger.warning(
                "[startup]   缺失 %-8s  → pip install -e '.[%s]'   (%s)",
                d.name, d.group, d.desc,
            )
    else:
        logger.info("[startup] 可选功能依赖齐全 (checked %d)", len(check_missing_deps()))

    if report["dead_code"]:
        logger.warning("[startup] 发现 %d 处 try 包裹纯 stdlib import 的死重容错",
                       len(report["dead_code"]))
    return report


if __name__ == "__main__":
    # 自检: 直接跑总入口并打印概要 (死代码默认关, 免得每次都全包扫描)
    _rep = run_startup_check()
    _all = check_missing_deps()
    assert isinstance(_all, list) and _all, "check_missing_deps 应返回非空列表"
    assert all(hasattr(d, "present") for d in _all), "DependencyCheck 缺 present 字段"
    assert "missing_optional_deps" in _rep, "报告缺 missing_optional_deps"
    print(f"missing optional deps: {len(_rep['missing_optional_deps'])}")
    for m in _rep["missing_optional_deps"]:
        print(f"  - {m['name']:24} group={m['group']:12} {m['desc']}")
    print(f"dead_code scan ran: {_rep['dead_code'] is not None}")
