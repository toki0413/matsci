"""材料科研版 autoresearch —— 把 Karpathy autoresearch 的 ratchet 思路搬到材料计算.

核心区别:
  - Karpathy 原版: 改 train.py → 跑训练 → val_bpb ratchet
  - 这版:         改 INCAR/KPOINTS/势函数参数 → 跑 VASP/LAMMPS → 物理量 ratchet

ratchet 对象是 formation_energy / band_gap / conductivity / defect_formation_energy /
elastic_modulus 这些物理量, 而不是 loss. LLM 看着迭代历史提议下一组参数, 跑完一轮
就把指标抠出来跟 best 比, 进了就留, 没进就丢, 跟原版一个味儿.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext, ToolResult


# workflow_template → 实际跑计算用的 tool 名
# 缺省走 vasp_tool, aimd 这种分子动力学走 lammps_tool, ml_potential 走 ml_potential_tool
_TEMPLATE_TOOL_MAP: dict[str, str] = {
    "standard_dft": "vasp_tool",
    "defect": "vasp_tool",
    "surface": "vasp_tool",
    "phonon": "vasp_tool",
    "aimd": "lammps_tool",
    "ml_potential": "ml_potential_tool",
}

# 不同 ratchet_metric 对应的 VASP action 和从 parsed 结果里取哪个字段
# formation_energy / defect_formation_energy 直接拿 TOTEN 比相对值就够了 (同结构同参考)
_METRIC_EXTRACTION: dict[str, dict[str, str]] = {
    "formation_energy": {"vasp_action": "relax", "parsed_key": "energy"},
    "energy": {"vasp_action": "relax", "parsed_key": "energy"},
    "defect_formation_energy": {"vasp_action": "relax", "parsed_key": "energy"},
    "band_gap": {"vasp_action": "scf", "parsed_key": "band_gap"},
    "elastic_modulus": {"vasp_action": "relax", "parsed_key": "energy"},
    "conductivity": {"vasp_action": "scf", "parsed_key": "band_gap"},
}


class MaterialsAutoResearchInput(BaseModel):
    research_goal: str = Field(
        ...,
        description="研究目标, 如 'minimize formation energy of Li7La3Zr2O12'",
    )
    ratchet_metric: str = Field(
        ...,
        description=(
            "ratchet 指标: formation_energy / band_gap / conductivity / "
            "defect_formation_energy / elastic_modulus"
        ),
    )
    ratchet_direction: Literal["minimize", "maximize"] = Field(
        default="minimize",
        description="minimize 或 maximize",
    )
    initial_structure: str | None = Field(
        default=None,
        description="初始结构文件路径 (POSCAR/CIF), 留空则要求工作目录里已有 POSCAR",
    )
    workflow_template: str = Field(
        default="standard_dft",
        description="用哪个 workflow 模板跑实验 (standard_dft/aimd/defect/surface/ml_potential/...)",
    )
    max_iterations: int = Field(
        default=10, ge=1, le=100, description="最多迭代几轮"
    )
    convergence_threshold: float | None = Field(
        default=None,
        description="收敛阈值, 最近 3 轮指标波动小于该值就停; 留空则跑到 max_iterations",
    )
    parameter_space: dict[str, list] | None = Field(
        default=None,
        description=(
            "LLM 可以调的参数空间, 如 "
            "{'encut': [400, 500, 600], 'kpoints': ['4 4 4', '6 6 6'], 'ismear': [0, -5]}"
        ),
    )
    record_history: bool = Field(default=True, description="是否记录迭代历史")
    work_dir: str | None = Field(
        default=None,
        description="工作根目录, 留空则用 context.workspace 下的 materials_autoresearch/",
    )
    walltime_hours: int = Field(default=24, ge=1, le=168)


class MaterialsAutoResearchTool(HuginnTool):
    """材料科研 ratchet 循环: LLM 提参 → 跑 DFT/MD → 抠指标 → ratchet.

    仿 Karpathy autoresearch, 但 ratchet 对象从 val_bpb 换成物理量. LLM 改的不是
    train.py 而是 INCAR/POSCAR/KPOINTS/势函数参数. 每轮把指标跟 best 比, 进了就留.
    """

    name = "materials_autoresearch_tool"
    category = "search"
    description = (
        "Materials science research loop: LLM proposes calculation parameters "
        "(INCAR/KPOINTS/potential), runs DFT/MD via vasp_tool/lammps_tool, "
        "extracts a physical metric (formation_energy/band_gap/conductivity/...), "
        "and ratchets toward the research goal."
    )
    destructive = True
    input_schema = MaterialsAutoResearchInput

    # 匹配 OUTCAR 里的 TOTEN, 兜底用 (vasp_tool 自己也会 parse, 这里是 LLM 路径的备份)
    _TOTEN_RE = re.compile(r"free  energy   TOTEN\s*=\s*([-\d.]+)")
    _BANDGAP_RE = re.compile(r"bandgap\s*[:=]\s*([-\d.]+)", re.I)
    _JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

    async def call(
        self, args: MaterialsAutoResearchInput, context: ToolContext
    ) -> ToolResult:
        try:
            return await self._run_loop(args, context)
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

    # ------------------------------------------------------------------ main loop

    async def _run_loop(
        self, args: MaterialsAutoResearchInput, context: ToolContext
    ) -> ToolResult:
        # 1. 初始化工作目录 + ratchet 状态
        run_id = f"mar_{uuid.uuid4().hex[:8]}"
        base_dir = self._resolve_base_dir(args, context, run_id)
        base_dir.mkdir(parents=True, exist_ok=True)

        # 把初始结构拷到 base_dir/POSCAR 作为模板, 后续每轮从它复制
        poscar_template = base_dir / "POSCAR_template"
        if args.initial_structure:
            src = Path(args.initial_structure).expanduser().resolve()
            if not src.exists():
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"initial_structure 不存在: {src}",
                )
            shutil.copyfile(src, poscar_template)
        else:
            # 没给 initial_structure 就看 context.workspace 里有没有现成的 POSCAR
            ws_poscar = (
                Path(context.workspace) / "POSCAR"
                if getattr(context, "workspace", None)
                else None
            )
            if ws_poscar and ws_poscar.exists():
                shutil.copyfile(ws_poscar, poscar_template)
            else:
                return ToolResult(
                    data=None,
                    success=False,
                    error="需要 initial_structure 或工作目录下现成的 POSCAR",
                )

        tool_name = _TEMPLATE_TOOL_MAP.get(args.workflow_template, "vasp_tool")
        extraction = _METRIC_EXTRACTION.get(
            args.ratchet_metric, {"vasp_action": "relax", "parsed_key": "energy"}
        )

        ratchet_state: dict[str, Any] = {
            "best_metric": None,
            "best_params": None,
            "best_iteration": None,
            "history": [],
        }

        # 2. 迭代循环
        converged = False
        for i in range(args.max_iterations):
            iteration = i + 1
            iter_dir = base_dir / f"iter_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # 2a. Propose: LLM 提议下一组参数
            try:
                proposal = await self._propose_params(args, ratchet_state, iteration)
            except Exception as exc:
                self._record(ratchet_state, iteration, None, None, False, f"propose 失败: {exc}", iter_dir)
                continue

            params = proposal.get("params", {})
            rationale = proposal.get("rationale", "")

            # 2b. Execute: 写输入文件 + 调 vasp_tool/lammps_tool
            exec_result = await self._execute_experiment(
                args, context, tool_name, extraction, params, iter_dir, poscar_template
            )

            # 2c. Extract: 从计算结果抠指标
            metric_value = None
            extract_note = ""
            if exec_result.get("success"):
                metric_value, extract_note = self._extract_metric(
                    args.ratchet_metric, extraction, exec_result, iter_dir
                )

            # 2d. Ratchet: 比较 best
            is_best = False
            if metric_value is not None:
                if self._is_improved(
                    metric_value, ratchet_state["best_metric"], args.ratchet_direction
                ):
                    ratchet_state["best_metric"] = metric_value
                    ratchet_state["best_params"] = params
                    ratchet_state["best_iteration"] = iteration
                    is_best = True

            # 2e. Record
            if args.record_history:
                self._record(
                    ratchet_state,
                    iteration,
                    params,
                    metric_value,
                    is_best,
                    rationale,
                    iter_dir,
                    note=extract_note,
                    exec_status=exec_result.get("status", "unknown"),
                )

            # 2f. Check convergence: 最近 3 轮非 None 指标的极差 < threshold
            if args.convergence_threshold is not None:
                if self._check_convergence(
                    ratchet_state["history"], args.convergence_threshold
                ):
                    converged = True
                    break

        # 3. Report
        report = await self._generate_report(args, ratchet_state, converged)

        return ToolResult(
            data={
                "run_id": run_id,
                "best_params": ratchet_state["best_params"],
                "best_metric": ratchet_state["best_metric"],
                "best_iteration": ratchet_state["best_iteration"],
                "total_iterations": len(ratchet_state["history"]),
                "converged": converged,
                "history": ratchet_state["history"],
                "report": report,
                "base_dir": str(base_dir),
            },
            success=True,
        )

    # ------------------------------------------------------------------ propose

    async def _propose_params(
        self,
        args: MaterialsAutoResearchInput,
        ratchet_state: dict[str, Any],
        iteration: int,
    ) -> dict[str, Any]:
        """让 LLM 看着历史 + 参数空间, 提议下一组参数."""
        from langchain_core.messages import HumanMessage, SystemMessage

        model = self._get_model()

        # 历史只喂最近 8 轮, 避免上下文爆
        recent = ratchet_state["history"][-8:]
        history_block = (
            json.dumps(recent, ensure_ascii=False, indent=2)
            if recent
            else "(还没有历史, 这是第一轮)"
        )

        ps_block = (
            json.dumps(args.parameter_space, ensure_ascii=False, indent=2)
            if args.parameter_space
            else "(未指定参数空间, 请根据研究目标自行推荐合理的 VASP 参数)"
        )

        system = (
            "你是计算材料科学专家. 你的任务是基于迭代历史和研究目标, 提议下一组计算参数. "
            "只能从给定的参数空间里选值 (如果提供了); 如果某参数没在空间里, 可以推荐一个合理值. "
            "输出必须是严格的 JSON, 不要加 markdown 代码块标记, 格式如下:\n"
            '{"params": {"encut": 500, "kpoints": "6 6 6", "ismear": 0, ...}, '
            '"rationale": "一句话解释为什么这么选"}'
        )

        prompt = (
            f"研究目标: {args.research_goal}\n"
            f"ratchet 指标: {args.ratchet_metric} ({args.ratchet_direction})\n"
            f"workflow 模板: {args.workflow_template}\n"
            f"当前迭代: 第 {iteration} 轮\n\n"
            f"参数空间:\n{ps_block}\n\n"
            f"迭代历史 (最近 8 轮):\n{history_block}\n\n"
            "请提议下一组参数. 如果历史里有 best, 优先在 best 附近做小调整; "
            "如果连续几轮没进步, 可以试一下差距更大的参数."
        )

        content = await self._llm_invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        data = self._parse_json(content)
        params = data.get("params", {})
        if not isinstance(params, dict):
            params = {}
        rationale = str(data.get("rationale", ""))
        return {"params": params, "rationale": rationale}

    # ------------------------------------------------------------------ execute

    async def _execute_experiment(
        self,
        args: MaterialsAutoResearchInput,
        context: ToolContext,
        tool_name: str,
        extraction: dict[str, str],
        params: dict[str, Any],
        iter_dir: Path,
        poscar_template: Path,
    ) -> dict[str, Any]:
        """写好输入文件, 调对应的 tool 跑计算."""
        # 准备 POSCAR
        poscar_dst = iter_dir / "POSCAR"
        if poscar_template.exists():
            shutil.copyfile(poscar_template, poscar_dst)

        tool = ToolRegistry.get(tool_name)
        if tool is None:
            return {
                "success": False,
                "status": "tool_not_found",
                "error": f"{tool_name} 未注册",
            }

        # 走 VASP 路径: 写 INCAR + KPOINTS, 调 vasp_tool
        if tool_name == "vasp_tool":
            self._write_incar(iter_dir / "INCAR", params)
            kpoints_str = params.get("kpoints", "4 4 4")
            self._write_kpoints(iter_dir / "KPOINTS", kpoints_str)
            vasp_action = extraction.get("vasp_action", "relax")
            tool_input = tool.input_schema(
                action=vasp_action,
                working_dir=str(iter_dir),
                incar_overrides={},
                walltime_hours=args.walltime_hours,
            )
            tool_ctx = self._make_ctx(context, str(iter_dir))
            try:
                result = await tool.call(tool_input, tool_ctx)
            except Exception as exc:
                return {"success": False, "status": "tool_error", "error": str(exc)}

            return {
                "success": result.success,
                "status": "completed" if result.success else "failed",
                "data": result.data,
                "error": result.error,
                "iter_dir": str(iter_dir),
            }

        # 非 VASP 工具 (lammps/ml_potential) —— 参数透传, 接口可能不一样, 兜底处理
        # 这里把 params 当成 incar_overrides 之类的字段塞进去, 失败就记一下
        try:
            kwargs: dict[str, Any] = {"walltime_hours": args.walltime_hours}
            # 常见字段兜底, 不存在的字段 pydantic 会报错, 那就退回最少参数
            try:
                tool_input = tool.input_schema(**kwargs)
            except Exception:
                tool_input = tool.input_schema()
            tool_ctx = self._make_ctx(context, str(iter_dir))
            result = await tool.call(tool_input, tool_ctx)
            return {
                "success": result.success,
                "status": "completed" if result.success else "failed",
                "data": result.data,
                "error": result.error,
                "iter_dir": str(iter_dir),
            }
        except Exception as exc:
            return {"success": False, "status": "tool_error", "error": str(exc)}

    # ------------------------------------------------------------------ extract

    def _extract_metric(
        self,
        ratchet_metric: str,
        extraction: dict[str, str],
        exec_result: dict[str, Any],
        iter_dir: Path,
    ) -> tuple[float | None, str]:
        """从计算结果里抠 ratchet 指标.

        优先用 vasp_tool 已经 parse 好的 parsed dict; 拿不到就读 OUTCAR 兜底;
        再拿不到就返回 None (这一轮不参与 ratchet).
        """
        parsed_key = extraction.get("parsed_key", "energy")
        data = exec_result.get("data") or {}
        parsed = data.get("parsed") if isinstance(data, dict) else None

        # 1. 先看 parsed dict
        if isinstance(parsed, dict):
            val = parsed.get(parsed_key)
            if isinstance(val, (int, float)) and val == val:  # 排除 NaN
                return float(val), f"从 parsed.{parsed_key} 取值"

        # 2. 读 OUTCAR 兜底
        outcar = iter_dir / "OUTCAR"
        if outcar.exists():
            try:
                content = outcar.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                content = ""
            if parsed_key == "energy":
                m = self._TOTEN_RE.findall(content)
                if m:
                    return float(m[-1]), "从 OUTCAR TOTEN 兜底取值"
            elif parsed_key == "band_gap":
                # vasprun.xml 里偶尔有 bandgap 标记
                vasprun = iter_dir / "vasprun.xml"
                if vasprun.exists():
                    try:
                        vr = vasprun.read_text(encoding="utf-8", errors="ignore")
                        bg = self._BANDGAP_RE.search(vr)
                        if bg:
                            return float(bg.group(1)), "从 vasprun.xml 兜底取值"
                    except Exception:
                        pass

        return None, "未能提取指标 (可能 VASP 未真正执行或结果缺失)"

    # ------------------------------------------------------------------ ratchet

    def _is_improved(
        self,
        value: float | None,
        baseline: float | None,
        direction: str,
    ) -> bool:
        if value is None:
            return False
        if baseline is None:
            return True  # 第一个有效值直接当 best
        if direction == "minimize":
            return value < baseline
        return value > baseline

    def _check_convergence(
        self, history: list[dict[str, Any]], threshold: float
    ) -> bool:
        """最近 3 轮有效指标的极差 < threshold 就算收敛."""
        recent_values = [
            h["metric"] for h in history[-3:] if h.get("metric") is not None
        ]
        if len(recent_values) < 3:
            return False
        return (max(recent_values) - min(recent_values)) < threshold

    def _record(
        self,
        ratchet_state: dict[str, Any],
        iteration: int,
        params: dict[str, Any] | None,
        metric: float | None,
        is_best: bool,
        rationale: str,
        iter_dir: Path,
        note: str = "",
        exec_status: str = "unknown",
    ) -> None:
        ratchet_state["history"].append(
            {
                "iteration": iteration,
                "params": params,
                "metric": metric,
                "is_best": is_best,
                "rationale": rationale,
                "exec_status": exec_status,
                "note": note,
                "iter_dir": str(iter_dir),
            }
        )

    # ------------------------------------------------------------------ report

    async def _generate_report(
        self,
        args: MaterialsAutoResearchInput,
        ratchet_state: dict[str, Any],
        converged: bool,
    ) -> str:
        """调一次 LLM 把迭代历史总结成报告."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            model = self._get_model()
            summary = {
                "research_goal": args.research_goal,
                "ratchet_metric": args.ratchet_metric,
                "ratchet_direction": args.ratchet_direction,
                "best_metric": ratchet_state["best_metric"],
                "best_params": ratchet_state["best_params"],
                "best_iteration": ratchet_state["best_iteration"],
                "converged": converged,
                "total_iterations": len(ratchet_state["history"]),
                "history": ratchet_state["history"],
            }
            system = (
                "你是材料计算专家. 根据迭代历史写一份简洁的中文报告, 包含: "
                "1) 最佳参数和对应指标; 2) 收敛分析 (是否收敛, 指标随迭代的变化趋势); "
                "3) 推荐的下一步 (基于结果给 2-3 条具体建议). "
                "直接输出 Markdown, 不要加代码块."
            )
            prompt = json.dumps(summary, ensure_ascii=False, indent=2)
            content = await self._llm_invoke(
                [SystemMessage(content=system), HumanMessage(content=prompt)]
            )
            return content
        except Exception as exc:
            # LLM 挂了也不能让整个 loop 白跑, 给个最小报告兜底
            return (
                f"# Materials AutoResearch Report\n\n"
                f"研究目标: {args.research_goal}\n"
                f"ratchet 指标: {args.ratchet_metric} ({args.ratchet_direction})\n"
                f"最佳指标: {ratchet_state['best_metric']}\n"
                f"最佳参数: {ratchet_state['best_params']}\n"
                f"最佳轮次: {ratchet_state['best_iteration']}\n"
                f"是否收敛: {converged}\n"
                f"总迭代数: {len(ratchet_state['history'])}\n\n"
                f"(LLM 报告生成失败: {exc})"
            )

    # ------------------------------------------------------------------ helpers

    def _resolve_base_dir(
        self, args: MaterialsAutoResearchInput, context: ToolContext, run_id: str
    ) -> Path:
        if args.work_dir:
            return Path(args.work_dir).expanduser().resolve() / run_id
        ws = getattr(context, "workspace", None) or "."
        return Path(ws).expanduser().resolve() / "materials_autoresearch" / run_id

    def _make_ctx(self, context: ToolContext, workspace: str) -> ToolContext:
        return ToolContext(
            session_id=f"materials_autoresearch_{uuid.uuid4().hex[:8]}",
            workspace=workspace,
            config=getattr(context, "config", None),
        )

    def _get_model(self) -> Any:
        from huginn.llm import get_model

        return get_model(temperature=0.4, max_tokens=8000)

    async def _llm_invoke(self, messages: list) -> str:
        """统一的 LLM 调用: 优先 ainvoke, 没有就 to_thread 兜底."""
        model = self._get_model()
        if hasattr(model, "ainvoke"):
            response = await model.ainvoke(messages)
        else:
            response = await asyncio.to_thread(model.invoke, messages)
        content = response.content if hasattr(response, "content") else str(response)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return content

    def _parse_json(self, text: str) -> dict[str, Any]:
        """从 LLM 回复里抠 JSON, 容忍 markdown 代码块和前后废话."""
        text = text.strip()
        m = self._JSON_RE.search(text)
        if m:
            text = m.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 兜底: 找第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {}

    def _write_incar(self, path: Path, params: dict[str, Any]) -> None:
        """根据参数写一个能跑的 INCAR.

        params 里的 kpoints 单独处理 (写 KPOINTS), 其余当 INCAR tag.
        """
        # 基础模板, 保证能跑通 relax/scf; 被 params 覆盖
        base = {
            "SYSTEM": "materials_autoresearch",
            "ENCUT": 500,
            "PREC": "Accurate",
            "EDIFF": 1e-6,
            "EDIFFG": -0.01,
            "IBRION": 2,
            "NSW": 60,
            "ISIF": 3,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LREAL": "Auto",
            "LWAVE": ".FALSE.",
            "LCHARG": ".TRUE.",
        }
        for k, v in params.items():
            if k.lower() == "kpoints":
                continue  # KPOINTS 单独写
            base[k.upper()] = v

        lines = [f"{k} = {v}" for k, v in base.items()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_kpoints(self, path: Path, kpoints_str: str) -> None:
        """写一个 Gamma-centered KPOINTS 文件.

        kpoints_str 形如 '6 6 6' 或 '4 4 4'.
        """
        parts = kpoints_str.replace(",", " ").split()
        if len(parts) < 3:
            parts = ["4", "4", "4"]
        try:
            kx, ky, kz = int(parts[0]), int(parts[1]), int(parts[2])
        except (ValueError, IndexError):
            kx, ky, kz = 4, 4, 4
        content = (
            "Auto-generated by materials_autoresearch\n"
            "0\n"
            "Gamma\n"
            f"{kx} {ky} {kz}\n"
            "0 0 0\n"
        )
        path.write_text(content, encoding="utf-8")

    def estimate_cost(self, args: MaterialsAutoResearchInput) -> dict[str, float] | None:
        # 每轮 VASP 大约 walltime_hours, 总共 max_iterations 轮
        return {
            "cpu_hours": args.walltime_hours * 4 * args.max_iterations,
            "walltime_hours": args.walltime_hours * args.max_iterations,
        }
