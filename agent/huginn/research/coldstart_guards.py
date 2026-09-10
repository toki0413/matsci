"""跨域冷启动守卫库 (Cross-domain Cold-start Guards).

把散落在各域 Markdown 报告里的**工程 root-cause**（依赖缺失 / import 白名单 / 返回
契约 schema / 书生成码质量低谷…）编译成**机器可读、域无关的结构化守卫**，让"沉淀经验"
与"复用经验"走同一条通道——开新域批次前自动聚合 → 编译该域启动清单 → 注入 harness。

设计原则(huginn 一贯)：
  - **域无关**：守卫按 `category` 组织，不绑定具体物理量的键名(如 vcR/probe_barrier)。
    领域再多 = 在 `DOMAIN_PROFILES` 里加一行声明，不靠人力去重读旧域散文。
  - **纯标准库 / 幂等 / 零 LLM**：可独立单测；`compile_domain_guards(domain)` 输出
    结构化的 {deps, imports_whitelist, schema_contract, code_retry_budget, prompt_guards}。
  - **诚实边界**：守卫只做"预检 / 对齐 / 注入"，不伪造任何真值；守卫检查失败只降级
    (把 gear 降一档)，绝不替模型"脑内补数值"。

流水线三态:
  - sing 状态库: `GUARD_LIBRARY`(跨域通用类别) + `DOMAIN_PROFILES`(各域声明)。
  - `compile_domain_guards(domain)` -> 该域启动守卫清单。
  - harness 冷启动(新域批次前)把清单注入 code_lab 沙箱配置与成文 system prompt。
"""
from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import Any


# ═══════════════════ 跨域通用守卫类别(与具体域解耦) ═══════════════════

# category -> (guard 说明, detect 检测, abstract_fix 修复)
GUARD_LIBRARY: dict[str, dict[str, str]] = {
    "sandbox_whitelist": {
        "guard": "书生成码前的 import 白名单需含该域数值计算依赖与纯类型标准库",
        "detect": "代码依赖 import 被拦截(typing/dataclasses 等)或模块缺失",
        "abstract_fix": (
            "依赖白名单补纯标准库(typing/dataclasses/itertools/functools/collections/"
            "copy/decimal/fractions/operator) + 装缺失包(matplotlib 等) + 无显示主机设 MPLBACKEND=Agg"
        ),
    },
    "schema_contract": {
        "guard": "书生成码 run() 必须返回可序列化数值; 裸数值 dict 自动封装, 纯字符串/非 dict 拒绝",
        "detect": "run() 返回非 dict / 无数值 / 内容不可序列化",
        "abstract_fix": (
            "对裸数值 dict 做收尾封装(objectives=纯数值标量键, summary=其余), "
            "不伪造数值; schema 硬伤(非 dict/IP/序列化失败)照旧回退白名单"
        ),
    },
    "code_quality_retry": {
        "guard": "书生成码失败时用真实 traceback 回流给模型修一版, 预算受限, 全败回退白名单",
        "detect": "运行时 bug(KeyError/SyntaxError/TypeError 等)",
        "abstract_fix": (
            "把真实 err 拼进下一轮 LLM user 让它带错修; 预算(_AUTHOR_MAX_RETRY)设小, "
            "把 API 火力让给模型擅长的 scan/probe 路径"
        ),
    },
    "dependency_install": {
        "guard": "冷启动前依赖预检: 该域数值计算所需第三方包已安装",
        "detect": "ModuleNotFoundError: <pkg>",
        "abstract_fix": "启动前 `importlib.util.find_spec` 预检, 缺失则提示/安装并拒绝进入主链路",
    },
    "dimension_routing": {
        "guard": "扫描维名(N 维)与执行分支一一对应, 别名归一一致, 防维度串扰致数据重复",
        "detect": "两个不同假设落到同一扫描分支、数据完全相同 / 某维度假设空转",
        "abstract_fix": "为每个逻辑物理维度建独立分支, dim 别名统一映射(canon), 聚类 objective 键按分支返回",
    },
    "report_gate": {
        "guard": "成文阶段把该域诊断工具(probe_*)注入 LLM function-calling, 结果进 trace 供门禁核对",
        "detect": "报告数值在 trace 中无来源(未落地)",
        "abstract_fix": "report 期挂载域探针(tools=tool_schemas), 调用返回值进 trace, 门禁比对",
    },
}


# ═══════════════════ 各域声明(新增域只加一行) ═══════════════════

# 每个域: 依赖预检、import 白名单增量、schema 契约开关、代码重试预算、成文探针名。
DOMAIN_PROFILES: dict[str, dict[str, Any]] = {
    "rigidity": {
        "deps": ["numpy", "scipy"],
        "imports_extra": [],
        "code_retry_budget": 2,
        "probes": [],
        "note": "rigidity 域: 线性约束/位型单调/基敏感, 符号+矩阵数值",
    },
    "fracture": {
        "deps": ["numpy", "scipy", "matplotlib"],
        "imports_extra": [],   # 纯标准库守卫由跨域 GUARD_LIBRARY 兜底
        "code_retry_budget": 1,   # 倾斜: 书生成码一次失败即回退, 火让给 probe/scan
        # 域级 cfg 键别名(书生近义名 → 白名单规范键, 全映射同一份真实数值):
        # 域专用物理别名显式待在域声明里, 不让通用 harness 累积例外(低熵红线).
        "cfg_aliases": {
            "sigma_0": "bridge_ratio", "sigma0": "bridge_ratio",
            "sigma0_sigmay": "bridge_ratio", "sigma0_over_sy": "bridge_ratio",
            "bridge": "bridge_ratio", "bridging_ratio": "bridge_ratio",
            "n_flaw": "n_flaws", "n_defects": "n_flaws", "defect_count": "n_flaws",
            "a_over_astar": "flaw_idx", "aastar": "flaw_idx",
            "aa_star": "flaw_idx", "flaw_size": "flaw_idx",
            "kic_i": "kic_mat", "kic_idx": "kic_mat", "KIC_mat": "kic_mat",
            "v_cr": "vcR", "v_cR": "vcR", "v_over_cR": "vcR",
            "poisson": "nu", "poisson_ratio": "nu",
        },
        "probes": ["probe_interface", "probe_dbt", "probe_flaw", "probe_weibull",
                   "probe_bridge", "probe_barrier", "probe_kic"],
        "note": "断裂力学: IJF2026 七开放问题, scan+probe 主导, 书生成码低频",
    },
    "quantum_critical": {
        "deps": ["numpy", "scipy"],
        "imports_extra": [],
        "code_retry_budget": 1,   # 倾斜: 解析+数值扫描主导, 书生成码低频(与 fracture 同理)
        # 探针面 = 通用工具面(dict 逃生口 scipy 仪器 + 书生成码 probe_*), 不再预置物理别名;
        # 成文时出具域无关通用仪器(qc_integrate/root/minimize/curvefit/ode), 不写物理探针.
        "probes": [],
        "note": ("应变耦合量子相变: 序参量 Landau-Ginzburg + 临界指数/普适类/"
                 "相图拓扑/材料谱系, 多尺度耦合, 与固体力学零族(泛化验证)"),
    },
    # held-out 泛化域: 生态动力学(Lotka-Volterra)。扣出、不并入 in-sample 复用计分,
    # 只用同一机制零改动测其泛化。声明即数据, 不碰任何共享机制代码.
    "ecology_dynamics": {
        "deps": ["numpy", "scipy"],
        "imports_extra": [],
        "code_retry_budget": 1,
        "cfg_aliases": {},
        "probes": [],
        # 科学契约工件(HEP §2609.00107 启发): 每个 objectives 量声明 量纲(unit) + 有效域(domain).
        # 是**域数据**, 可整块卸载/替换(compile_domain_guards 透出); 共享验证器不吃硬编码键.
        "scientific_contract": {
            "quantities": {
                "x_star":    {"unit": "count", "domain": {"min": 0.0}},
                "y_star":    {"unit": "count", "domain": {"min": 0.0}},
                "trace":     {"unit": "1",     "domain": None},
                "period_est": {"unit": "time", "domain": {"min": 0.0}},
                "final_prey": {"unit": "count", "domain": {"min": 0.0}},
            }
        },
        "note": ("群体生态捕食-被捕食(Lotka-Volterra): 平衡点 Jacobian 稳定性 + "
                 "数值积分振荡周期, 与凝聚态/固体力学零族(held-out 泛化域)"),
    },
}

# 默认(未声明域)参数
_DEFAULTS: dict[str, Any] = {
    "deps": [],
    "imports_extra": [],
    "code_retry_budget": 2,
    "cfg_aliases": {},
    "scientific_contract": {},
    "probes": [],
    "note": "",
}


def get_domain_profile(domain: str) -> dict[str, Any]:
    """取域声明, 缺失时用默认(未登记域仍能跑, 只是没有域级守卫)."""
    p = dict(_DEFAULTS)
    p.update(DOMAIN_PROFILES.get(domain, {}))
    return p


def _active_categories(domain: str) -> list[str]:
    """跨域守卫类别默认全部启用(域无关); 特殊域可覆盖."""
    return list(GUARD_LIBRARY)


# ═══════════════════════ 编译: 域 → 启动守卫清单 ═══════════════════════

def compile_domain_guards(domain: str) -> dict[str, Any]:
    """把跨域 GUARD_LIBRARY + 该域声明编译成可注入的启动清单.

    产出结构(机器可读, harness 冷启动时逐项消费):
      {
        "domain": str,
        "summary": {category: guard 一句话},
        "deps_check": {dep: installed(True/False)},
        "imports_whitelist_extra": [str],
        "schema_contract": {strict: bool, coerce_numeric_dict: bool},
        "code_retry_budget": int,
        "cfg_aliases": {alias: target},   # 域级 cfg 键别名(书生成码沙箱注入依据)
        "scientific_contract": {quantities: {key: {unit, domain}}},  # 机器可读科学契约工件
        "probes": [str],
        "prompt_guards": [str],     # 注入成文 system prompt 的软提示块
        "note": str,
      }
    """
    pf = get_domain_profile(domain)
    cats = _active_categories(domain)
    summary = {c: GUARD_LIBRARY[c]["guard"] for c in cats}
    # 依赖预检(纯标准库 find_spec, 不 import 触发副作用)
    deps_check: dict[str, bool] = {}
    for d in pf["deps"]:
        deps_check[d] = importlib.util.find_spec(d) is not None
    missing = [d for d, ok in deps_check.items() if not ok]
    # 写码重试预算来自域声明(倾斜中心)
    retry = max(0, int(pf["code_retry_budget"]))
    # 成文软提示块(跨域守卫的 prompt 化摘要)
    prompt_guards = [
        f"[冷启动守卫·{c}] {GUARD_LIBRARY[c]['abstract_fix']}"
        for c in cats
    ]
    return {
        "domain": domain,
        "summary": summary,
        "deps_check": deps_check,
        "missing_deps": missing,
        "imports_whitelist_extra": list(pf["imports_extra"]),
        "schema_contract": {"strict": False, "coerce_numeric_dict": True},
        "code_retry_budget": retry,
        "cfg_aliases": dict(pf["cfg_aliases"]),
        "scientific_contract": dict(pf["scientific_contract"] or {}),
        "probes": list(pf["probes"]),
        "prompt_guards": prompt_guards,
        "note": pf["note"],
    }


def verify_domain_ready(guards: dict[str, Any]) -> dict[str, Any]:
    """冷启动可准备性检查(不伪造): mostly 依赖是否就绪, 缺失列出来."""
    missing = list(guards.get("missing_deps") or [])
    ready = not missing
    return {
        "ready": ready,
        "missing_deps": missing,
        "whitelist_ok": bool(guards.get("imports_whitelist_extra") or True),
        "retry_budget": guards.get("code_retry_budget"),
    }


def to_markdown(guards: dict[str, Any]) -> str:
    """把守卫清单渲染成可追加进域报告/checklist 的 md 块(供审计与跨域沉淀)."""
    lines = [f"### 冷启动守卫清单 · {guards['domain']}", ""]
    lines.append("| 类别 | 守卫 |")
    lines.append("|---|---|")
    for c, gtxt in (guards.get("summary") or {}).items():
        lines.append(f"| {c} | {gtxt} |")
    lines.append("")
    lines.append("**依赖预检**: " + ", ".join(
        f"{k}={'✓' if v else '✗'}" for k, v in (guards.get("deps_check") or {}).items()
    ) or "无")
    lines.append("")
    lines.append("**写码重试预算**: " + (str(guards.get("code_retry_budget")) or "默认"))
    lines.append("**成文探针**: " + ", ".join(guards.get("probes") or ["(无)"]))
    return "\n".join(lines)


__all__ = [
    "GUARD_LIBRARY",
    "DOMAIN_PROFILES",
    "compile_domain_guards",
    "verify_domain_ready",
    "to_markdown",
    "get_domain_profile",
]