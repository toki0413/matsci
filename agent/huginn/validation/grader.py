"""Grader 层 — 把现有物理/量纲校验器统一成"打分"接口.

autoloop 的奖励回流需要一个稳定的 (score, passed, reward) 三元组来驱动
evolution. PhysicsAuditor / DimensionalValidator 各自吐 findings / check
result, 结构不一致; Grader 就是这层适配: 每个校验器包成一个 Grader,
registry 统一跑 evaluate_all 拿回同构的 GraderResult 列表.

只做包装, 不重复实现校验逻辑——校验规则仍归原校验器.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from huginn.execution.physics_auditor import AuditReport, PhysicsAuditor
from huginn.validation.dimensional import DimensionalCheckResult, DimensionalValidator
from huginn.validation.innovation_signal import (
    DeviationLevel,
    InnovationSignal,
    InnovationSignalDetector,
)
from huginn.validation.research import RedTeamReviewer


@dataclass
class GraderResult:
    """一次打分的结果, 喂给 autoloop reward 通道用的同构结构."""

    name: str
    score: float          # 0.0 ~ 1.0, 综合分
    passed: bool          # 是否达到"可接受"门槛 (无 error / 量纲一致)
    reward: float = 0.0   # 奖励信号, 默认等于 score, 调用方可加权
    checks: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def __post_init__(self) -> None:
        # reward 没显式给就用 score, 避免每个调用方都写一遍
        if self.reward == 0.0 and self.score > 0.0:
            self.reward = self.score


# 严重度 -> 分数权重. error 直接判 0, warning 折半, info 不扣分
_SEV_SCORE = {"error": 0.0, "warning": 0.5, "info": 1.0}


class PhysicsGrader:
    """包 PhysicsAuditor: 审计计算结果, findings 折算成 score."""

    name = "physics"

    def __init__(self, auditor: PhysicsAuditor | None = None) -> None:
        self._auditor = auditor or PhysicsAuditor()

    def evaluate(self, data: dict[str, Any]) -> GraderResult:
        tool = data.get("tool_name", "unknown")
        action = data.get("action", "")
        parsed = data.get("parsed") or data.get("result_data") or {}
        params = data.get("input_params") or {}
        report: AuditReport = self._auditor.audit(tool, action, parsed, params)

        findings = report.findings
        if not findings:
            return GraderResult(
                name=self.name, score=1.0, passed=True,
                message="no findings", checks=[],
            )
        # 加权平均: error 拉低, warning 折半, info 满分
        total = sum(_SEV_SCORE.get(f.severity, 0.5) for f in findings)
        score = round(total / len(findings), 4)
        passed = not report.has_errors
        checks = [f.to_dict() for f in findings]
        msg = "; ".join(f.message for f in findings)
        return GraderResult(
            name=self.name, score=score, passed=passed,
            checks=checks, message=msg,
        )

    # 让实例可直接当 callable 塞进 registry
    def __call__(self, data: dict[str, Any]) -> GraderResult:
        return self.evaluate(data)


class DimensionalGrader:
    """包 DimensionalValidator: 校验方程量纲一致性."""

    name = "dimensional"

    def __init__(self, validator: DimensionalValidator | None = None) -> None:
        self._validator = validator or DimensionalValidator()

    def evaluate(self, data: dict[str, Any]) -> GraderResult:
        lhs = data.get("lhs_quantities") or []
        rhs = data.get("rhs_quantities") or []
        eq_name = data.get("equation_name", "")
        if not lhs or not rhs:
            return GraderResult(
                name=self.name, score=0.0, passed=False,
                message="missing lhs_quantities / rhs_quantities",
            )
        res: DimensionalCheckResult = self._validator.check_equation(
            lhs, rhs, equation_name=eq_name,
        )
        score = 1.0 if res.consistent else 0.0
        return GraderResult(
            name=self.name, score=score, passed=res.consistent,
            checks=[{
                "equation": res.equation,
                "consistent": res.consistent,
                "lhs_dimensions": res.lhs_dimensions,
                "rhs_dimensions": res.rhs_dimensions,
                "notes": res.notes,
            }],
            message="; ".join(res.notes),
        )

    def __call__(self, data: dict[str, Any]) -> GraderResult:
        return self.evaluate(data)


class RedTeamGrader:
    """包 RedTeamReviewer: 对抗性审查研究逻辑.

    data: {"from_phase", "to_phase", "evidence"}
    """

    name = "red_team"

    def __init__(self, reviewer: RedTeamReviewer | None = None) -> None:
        self._reviewer = reviewer or RedTeamReviewer()

    def evaluate(self, data: dict[str, Any]) -> GraderResult:
        from_phase = data.get("from_phase", "")
        to_phase = data.get("to_phase", "")
        evidence = data.get("evidence", {})
        report = self._reviewer.review(from_phase, to_phase, evidence)

        findings = report.findings
        if not findings:
            return GraderResult(
                name=self.name, score=1.0, passed=True,
                message="no findings",
            )
        # blocking -> 0, 每条发现扣 0.1
        if report.has_blocking:
            score = 0.0
        else:
            score = max(0.0, 1.0 - 0.1 * len(findings))
        checks = [
            {"severity": f.severity, "description": f.description}
            for f in findings
        ]
        return GraderResult(
            name=self.name, score=score,
            passed=not report.has_blocking,
            checks=checks,
            message="; ".join(f.description for f in findings),
        )

    def __call__(self, data: dict[str, Any]) -> GraderResult:
        return self.evaluate(data)


class HallucinationGrader:
    """断言式幻觉检测 — 纯规则检查, 不调 LLM.

    data: {"text": str, "expected_facts": list[str] (可选)}
    """

    name = "hallucination"

    _PATTERNS: list[tuple[str, str]] = [
        (r"(?:百分之百|100\s*%|绝对|毫无疑问|确定无疑|万无一失)", "overconfident assertion"),
        (r"\d+\.\d{6,}", "suspicious precision without citation"),
        (r"室温超导", "unlikely physics claim without evidence"),
    ]

    def __init__(self) -> None:
        self._compiled = [
            (re.compile(p, re.IGNORECASE), label) for p, label in self._PATTERNS
        ]

    def evaluate(self, data: dict[str, Any]) -> GraderResult:
        text = data.get("text") or data.get("output") or ""
        if not isinstance(text, str):
            text = str(text)
        issues: list[str] = []

        for pattern, label in self._compiled:
            matches = pattern.findall(text)
            if matches:
                issues.append(f"{label}: {len(matches)} occurrence(s)")

        for fact in data.get("expected_facts", []):
            if fact.lower() not in text.lower():
                issues.append(f"missing expected fact: {fact}")

        n_issues = len(issues)
        score = max(0.0, 1.0 - 0.2 * n_issues)
        return GraderResult(
            name=self.name, score=score, passed=n_issues == 0,
            checks=[{"issue": i} for i in issues],
            message="; ".join(issues),
        )

    def __call__(self, data: dict[str, Any]) -> GraderResult:
        return self.evaluate(data)


class BenchGrader:
    """包 MatWorldBench: 把 benchmark 评测结果折算成 GraderResult.

    data: {"task_id": str, "agent_output": dict}
    延迟导入 MatWorldBench, 避免 validation -> evaluation 的循环依赖.
    """

    name = "matworld_bench"

    def __init__(self, bench: Any | None = None) -> None:
        if bench is not None:
            self._bench = bench
        else:
            from huginn.evaluation.matworld_bench import MatWorldBench
            self._bench = MatWorldBench()

    def evaluate(self, data: dict[str, Any]) -> GraderResult:
        task_id = data.get("task_id", "")
        agent_output = data.get("agent_output") or data.get("output") or {}
        if not task_id:
            return GraderResult(
                name=self.name, score=0.0, passed=False,
                message="missing task_id",
            )
        res = self._bench.evaluate(task_id, agent_output)
        return GraderResult(
            name=self.name, score=res.score, passed=res.passed,
            checks=[res.details], message=f"task={task_id}",
        )

    def __call__(self, data: dict[str, Any]) -> GraderResult:
        return self.evaluate(data)


class LiteratureGrader:
    """Compare agent results against literature consensus.

    Reads ``data["literature_comparison"]`` — a dict of property_name ->
    InnovationSignal (or its dict form) — that _validate populates via
    benchmark_lookup + InnovationSignalDetector.

    Scoring:
      - within 2 sigma  -> 1.0 (agrees with literature)
      - 2-3 sigma       -> 0.5 (interesting, worth investigating)
      - > 3 sigma AND physically implausible -> 0.0 (likely error)
      - > 2 sigma but innovation_signal=True -> 0.5 (don't punish discoveries)

    If no literature_comparison data is present, returns a neutral 1.0.
    """

    name = "literature"

    def __init__(self, detector: InnovationSignalDetector | None = None) -> None:
        self._detector = detector or InnovationSignalDetector()

    def evaluate(self, data: dict[str, Any]) -> GraderResult:
        lit = data.get("literature_comparison")
        if not lit or not isinstance(lit, dict):
            return GraderResult(
                name=self.name, score=1.0, passed=True,
                message="no literature data for comparison",
            )

        scores: list[float] = []
        checks: list[dict[str, Any]] = []
        for prop, sig in lit.items():
            signal = self._coerce_signal(prop, sig)
            if signal is None:
                continue
            s = self._score_one(signal)
            scores.append(s)
            checks.append({
                "property": prop,
                "deviation_sigma": signal.deviation_sigma,
                "level": signal.level.name,
                "innovation_signal": signal.is_innovation_signal,
                "score": s,
            })

        if not scores:
            return GraderResult(
                name=self.name, score=1.0, passed=True,
                message="literature_comparison present but no valid signals",
            )
        overall = round(sum(scores) / len(scores), 4)
        # passed=True: literature comparison never hard-fails validation.
        # A low score just means "look closer", not "reject".
        return GraderResult(
            name=self.name, score=overall, passed=True,
            checks=checks,
            message=f"{len(scores)} property(ies) compared to literature",
        )

    def __call__(self, data: dict[str, Any]) -> GraderResult:
        return self.evaluate(data)

    @staticmethod
    def _coerce_signal(prop: str, sig: Any) -> InnovationSignal | None:
        """Accept either an InnovationSignal object or a plain dict."""
        if isinstance(sig, InnovationSignal):
            return sig
        if isinstance(sig, dict):
            try:
                return InnovationSignal(
                    property_name=sig.get("property_name", prop),
                    agent_value=float(sig.get("agent_value", 0)),
                    literature_consensus=float(sig.get("literature_consensus", 0)),
                    literature_spread=float(sig.get("literature_spread", 0)),
                    deviation_sigma=float(sig.get("deviation_sigma", 0)),
                    level=DeviationLevel[sig.get("level", "NEGLIGIBLE")],
                    possible_explanations=sig.get("possible_explanations", []),
                    is_innovation_signal=bool(sig.get("is_innovation_signal", False)),
                )
            except (KeyError, TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _score_one(sig: InnovationSignal) -> float:
        # Innovation signal: don't penalize potential discoveries
        if sig.is_innovation_signal:
            return 0.5
        if sig.deviation_sigma < 2.0:
            return 1.0
        if sig.deviation_sigma < 3.0:
            return 0.5
        # > 3 sigma and not an innovation signal -> physically implausible
        return 0.0


class ValidityJudge:
    """Post-hoc LLM judge — 检测 agent 是否走了 shortcut.

    NatureBench judge.py 启发: r_phys 高不代表真解决问题, 可能是 gaming grader.
    用 LLM 读 agent 代码 + 对话日志, 判断输出是"真算出来的"还是"硬编码/拷贝/
    手调常数/从输入文件反推"等 cheating 模式.

    data: {
        "agent_code": str,           # agent 生成的代码 (如 VASP 脚本)
        "conversation_log": str,     # 对话摘要 (从 completion records 提取)
        "output_summary": str,       # agent 提交的最终结果
    }
    ponytail: 单 LLM 同步调用, 失败降级到规则检查. 升级路径: 并行多 judge 投票.
    ceiling: LLM 判断有假阳性/假阴性, 不如 NatureBench 的 Docker 隔离硬.
    """

    name = "validity"

    # 规则降级: LLM 不可用时, 扫这些硬编码模式
    _RULE_PATTERNS: list[tuple[str, str]] = [
        (r"return\s+(?:np\.)?array\s*\(\s*\[\s*5\.0\s*\]", "hardcoded band_gap=5.0"),
        (r"band_gap\s*=\s*(?:4\.\d|5\.\d)\s*(?:#|$)", "suspicious hardcoded band_gap"),
        (r"#\s*(?:TODO|FIXME|HACK|hardcoded)", "shortcut marker in code"),
    ]

    def __init__(self, model: Any | None = None) -> None:
        self._model = model
        self._compiled = [
            (re.compile(p, re.IGNORECASE), label) for p, label in self._RULE_PATTERNS
        ]

    def evaluate(self, data: dict[str, Any]) -> GraderResult:
        agent_code = data.get("agent_code") or ""
        conv_log = data.get("conversation_log") or ""
        output_summary = data.get("output_summary") or ""

        # 无 LLM 或无代码可审 → 规则降级
        if self._model is None or not agent_code:
            return self._rule_fallback(agent_code)

        prompt = self._build_prompt(agent_code, conv_log, output_summary)
        try:
            import json
            # langchain ChatModel.ainvoke 返回 AIMessage, content 是 str
            resp = self._model.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            # 解析 {"is_valid": bool, "reason": str}
            parsed = self._parse_verdict(content)
            if parsed is not None:
                is_valid, reason = parsed
                score = 1.0 if is_valid else 0.0
                return GraderResult(
                    name=self.name, score=score, passed=is_valid,
                    checks=[{"is_valid": is_valid, "reason": reason}],
                    message=reason,
                )
        except Exception:
            pass  # LLM 挂了走规则

        return self._rule_fallback(agent_code)

    def _build_prompt(self, code: str, conv_log: str, output: str) -> str:
        # 裁剪到 token 预算. ponytail: 硬截断, 不上 tokenizer.
        code_clip = code[:20000] if len(code) > 20000 else code
        log_clip = conv_log[:10000] if len(conv_log) > 10000 else conv_log
        out_clip = output[:2000] if len(output) > 2000 else output
        return f"""审查这个材料科学 agent 的解决方案是否"真实计算"了结果, 还是走了 shortcut.

agent 代码:
```
{code_clip}
```

对话日志摘要:
{log_clip}

提交结果:
{out_clip}

判断是否出现以下 INVALID 行为 (参考 NatureBench judge.py):
1. 返回硬编码值而不是从数据计算 (如直接写 band_gap = 5.0)
2. 用硬编码规则或查找表代替从数据学习/计算
3. 从输入文件直接读取实验值而不是计算 (如从 OUTCAR grep 实验值)
4. 通过多次 validate 反馈手调常数, 试错凑答案
5. 调用外部 LLM API 生成输出代替用真实算法
6. 训练数据当查找表做最近邻匹配, 无参数 fit
7. 代码存在但未用于生成提交输出 (judge 实际做什么, 不看训练代码)
8. 输出空或退化结果 (全零数组)

以下情况不算 INVALID:
- 实现了真实方法但结果次优 (工程失败如 OOM/timeout)
- 改进为零或负 (弱但真实的方法仍是方法)
- 用简单经典方法 (简单不是作弊, 只要方法从数据 fit/optimize)

只返回 JSON: {{"is_valid": true/false, "reason": "一两句话, 引用具体代码或日志证据"}}"""

    def _parse_verdict(self, content: str) -> tuple[bool, str] | None:
        import json
        # 找第一个 { 到匹配的 }
        start = content.find("{")
        if start < 0:
            return None
        # 尝试从 start 解析 JSON 对象
        depth = 0
        for i, ch in enumerate(content[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(content[start:i+1])
                        return bool(obj.get("is_valid", True)), str(obj.get("reason", ""))
                    except (json.JSONDecodeError, KeyError):
                        return None
        return None

    def _rule_fallback(self, code: str) -> GraderResult:
        """LLM 不可用时, 用正则扫硬编码模式."""
        if not code:
            return GraderResult(
                name=self.name, score=1.0, passed=True,
                message="no code to judge",
            )
        issues: list[str] = []
        for pattern, label in self._compiled:
            matches = pattern.findall(code)
            if matches:
                issues.append(f"{label}: {len(matches)} match(es)")
        score = 1.0 if not issues else 0.0
        return GraderResult(
            name=self.name, score=score, passed=not issues,
            checks=[{"issue": i} for i in issues],
            message="; ".join(issues) if issues else "rule-based: clean",
        )

    def __call__(self, data: dict[str, Any]) -> GraderResult:
        return self.evaluate(data)


class MaterialsBoundsGrader:
    """材料科学物理界限检查 — 抓幻觉值.

    BenchGrader 对已知答案, 这个对未知题: 检查 agent 算出的值是否
    落在物理合理范围. band_gap=50eV 或 lattice=0.1Å 一眼假.
    data: {"properties": {key: value, ...}}
    ponytail: 硬编码常见属性范围. 升级: 从 Materials Project 查.
    """

    name = "materials_bounds"

    # 物理合理范围 (min, max). 没列的 key 跳过.
    _BOUNDS: dict[str, tuple[float, float]] = {
        "band_gap_eV": (0.0, 10.0),
        "lattice_constant_A": (2.0, 10.0),
        "bulk_modulus_GPa": (0.0, 600.0),
        "conductivity_S_per_m": (0.0, 1e8),
        "cohesive_energy_eV_per_atom": (0.0, 12.0),
        "adsorption_energy_eV": (-10.0, 5.0),
        "delta_E_eV_per_fu": (-5.0, 5.0),
        "formation_energy_eV_per_atom": (-5.0, 5.0),
        "magnetic_moment_uB": (-20.0, 20.0),
        "debye_temperature_K": (0.0, 2000.0),
    }

    def evaluate(self, data: dict[str, Any]) -> GraderResult:
        props = data.get("properties") or data.get("result_data") or {}
        if not isinstance(props, dict):
            props = {}
        violations: list[str] = []
        for key, val in props.items():
            bounds = self._BOUNDS.get(key)
            if bounds is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            lo, hi = bounds
            if v < lo or v > hi:
                violations.append(f"{key}={v} outside [{lo}, {hi}]")
        score = 1.0 if not violations else 0.0
        return GraderResult(
            name=self.name, score=score, passed=not violations,
            checks=[{"violation": v} for v in violations],
            message="; ".join(violations),
        )

    def __call__(self, data: dict[str, Any]) -> GraderResult:
        return self.evaluate(data)


class GraderRegistry:
    """注册多个 Grader, 统一跑 evaluate_all 拿同构结果."""

    def __init__(self) -> None:
        self._graders: dict[str, Callable[[dict[str, Any]], GraderResult]] = {}

    def register(
        self,
        name: str,
        grader: Callable[[dict[str, Any]], GraderResult],
    ) -> None:
        self._graders[name] = grader

    def register_from_spec(self, name: str, spec: str) -> None:
        """Polar 启发: 按 'module.path:ClassName' 字符串动态导入并注册.

        让外部 grader 不用改本文件就能接入. spec 里类要无参构造或接受 kwargs.
        ponytail: 只做 importlib 动态加载, 不做 entry-point 扫描.
        升级路径: setuptools entry_points 自动发现.
        """
        import importlib

        module_path, _, cls_name = spec.partition(":")
        if not cls_name:
            raise ValueError(f"spec must be 'module.path:ClassName', got: {spec}")
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        self._graders[name] = cls()

    def evaluate_all(self, data: dict[str, Any]) -> list[GraderResult]:
        return [g(data) for g in self._graders.values()]

    def names(self) -> list[str]:
        return list(self._graders)


def default_registry(model: Any | None = None) -> GraderRegistry:
    """预注册内置 grader: physics + dimensional + hallucination + literature + red_team + materials_bounds + validity.

    model: 可选, 传给 ValidityJudge 做 post-hoc LLM 判断. None 时走规则降级.
    """
    reg = GraderRegistry()
    reg.register("physics", PhysicsGrader())
    reg.register("dimensional", DimensionalGrader())
    reg.register("hallucination", HallucinationGrader())
    reg.register("literature", LiteratureGrader())
    reg.register("red_team", RedTeamGrader())
    reg.register("materials_bounds", MaterialsBoundsGrader())
    reg.register("validity", ValidityJudge(model=model))
    return reg


__all__ = [
    "GraderResult",
    "PhysicsGrader",
    "DimensionalGrader",
    "RedTeamGrader",
    "HallucinationGrader",
    "BenchGrader",
    "LiteratureGrader",
    "MaterialsBoundsGrader",
    "ValidityJudge",
    "GraderRegistry",
    "default_registry",
]
