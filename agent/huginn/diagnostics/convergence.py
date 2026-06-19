"""Convergence diagnosis system — auto-detect failure patterns and suggest fixes.

Inspired by Claude Code's 'doctor' command, but specialized for
computational material science convergence failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class DiagnosisReport:
    """Report from convergence failure diagnosis."""

    problem: str
    cause: str
    suggestions: list[str]
    auto_fixable: bool
    severity: Literal["critical", "warning", "info"] = "warning"
    fix_applied: bool = False


# Known VASP failure patterns
VASP_FAILURE_PATTERNS = {
    "EDDDAV": DiagnosisReport(
        problem="电子步不收敛 (EDDDAV/ diagonalization failure)",
        cause="ALGO选择不当、NELM不足或初始电荷密度与当前结构不匹配",
        suggestions=[
            "尝试 ALGO=Normal 或 ALGO=Fast（默认Davidson可能不稳定）",
            "增加 NELM 到 100-200",
            "设置 ISTART=1 读取已有 WAVECAR",
            "降低 AMIX 到 0.1-0.2，增加 BMIX 到 1.0",
            "如果金属体系，尝试 ISMEAR=-5 配合足够K点",
        ],
        auto_fixable=True,
        severity="critical",
    ),
    "ZPOTRF": DiagnosisReport(
        problem="电子步不收敛 (ZPOTRF / Cholesky failure)",
        cause="哈密顿量不正定，通常因初始猜测差或接近简并",
        suggestions=[
            "设置 IALGO=38 或 ALGO=Normal",
            "降低 AMIX 到 0.1",
            "检查 POSCAR 是否有重叠原子",
            "尝试 ISTART=1 或 IWAVPR=1",
        ],
        auto_fixable=True,
        severity="critical",
    ),
    "ZBRENT": DiagnosisReport(
        problem="离子步不收敛 (ZBRENT / line minimization failure)",
        cause="力收敛标准太严格、POTIM过大或势能面不平滑",
        suggestions=[
            "增大 EDIFFG（如从 -0.01 到 -0.05 或 -0.1）",
            "减小 POTIM 到 0.1-0.2",
            "尝试 IBRION=1（准牛顿法）替代 IBRION=2",
            "如果初始结构差，先做 IBRION=2 快速弛豫再转 IBRION=1",
        ],
        auto_fixable=True,
        severity="warning",
    ),
    "TOO FEW BANDS": DiagnosisReport(
        problem="能带数不足 (TOO FEW BANDS)",
        cause="NBANDS设置过低，电子占据态超出可用能带",
        suggestions=[
            "增加 NBANDS = NELECT/2 + NIONS * 1.5（绝缘体）",
            "或 NBANDS = NELECT/2 + NIONS * 2（金属/磁性）",
            "磁性体系额外增加 ~20% 能带",
        ],
        auto_fixable=True,
        severity="critical",
    ),
    "killed": DiagnosisReport(
        problem="作业被系统终止 (OOM 或超时)",
        cause="内存不足（OOM）或超出 walltime 限制",
        suggestions=[
            "检查 slurm 错误日志确认是 OOM 还是超时",
            "OOM: 减少 NCORE/NPAR，增加节点数",
            "OOM: 启用 KPAR 并行分解K点",
            "超时: 增加 walltime 或减小体系/精度",
        ],
        auto_fixable=False,
        severity="critical",
    ),
    "oom": DiagnosisReport(
        problem="内存不足 (Out of Memory)",
        cause="计算所需内存超出分配",
        suggestions=[
            "减少 NCORE（增加每核内存）",
            "增加节点数（降低每节点负载）",
            "降低 ENCUT 或 KPOINTS 密度",
            "启用 KPAR > 1 分解K点减少内存",
        ],
        auto_fixable=False,
        severity="critical",
    ),
    "FEXCP": DiagnosisReport(
        problem="交换关联势错误 (FEXCP)",
        cause="POTCAR 与 INCAR 中的泛函不匹配",
        suggestions=[
            "确认 POTCAR 的泛函（LDA/PBE/PBEsol/SCAN）与 INCAR 中 GGA 设置一致",
            "重新生成匹配泛函的 POTCAR",
        ],
        auto_fixable=True,
        severity="critical",
    ),
}

# LAMMPS failure patterns
LAMMPS_FAILURE_PATTERNS = {
    "Lost atoms": DiagnosisReport(
        problem="原子丢失 (Lost atoms)",
        cause="时间步长过大、盒子边界不当或原子速度过高",
        suggestions=[
            "减小 timestep（从 1.0fs 降到 0.5fs 或 0.1fs）",
            "检查边界条件（periodic vs fixed）",
            "如果高温模拟，先用 smaller timestep 升温",
            "使用 fix nve/limit 限制最大位移",
        ],
        auto_fixable=True,
        severity="warning",
    ),
    "ERROR on proc": DiagnosisReport(
        problem="并行错误 (MPI 进程错误)",
        cause="原子分布不均或通信问题",
        suggestions=[
            "使用 balance 命令重新分配原子",
            "检查体系密度是否均匀",
            "减少并行进程数",
        ],
        auto_fixable=False,
        severity="warning",
    ),
}


class ConvergenceDiagnostician:
    """Diagnoses convergence failures from job logs."""

    def __init__(self):
        self.patterns = {
            "vasp": VASP_FAILURE_PATTERNS,
            "lammps": LAMMPS_FAILURE_PATTERNS,
        }

    def diagnose(self, engine: str, log_content: str) -> DiagnosisReport | None:
        """Analyze log content and return diagnosis if a known pattern is found."""
        engine_patterns = self.patterns.get(engine.lower(), {})

        for pattern_key, report in engine_patterns.items():
            if pattern_key.lower() in log_content.lower():
                return report

        # Generic timeout/termination detection
        lower_log = log_content.lower()
        if "killed" in lower_log or "terminated" in lower_log:
            if "oom" in lower_log or "memory" in lower_log:
                return VASP_FAILURE_PATTERNS["oom"]
            return VASP_FAILURE_PATTERNS["killed"]

        return None

    def diagnose_from_file(
        self, engine: str, log_path: str | Path
    ) -> DiagnosisReport | None:
        """Read log file and diagnose."""
        path = Path(log_path)
        if not path.exists():
            return None

        # Read last 500 lines (failures are usually near the end)
        lines = path.read_text(encoding="utf-8", errors="ignore").split("\n")
        log_tail = "\n".join(lines[-500:])

        return self.diagnose(engine, log_tail)

    def suggest_auto_fix(self, report: DiagnosisReport) -> dict[str, str] | None:
        """If auto_fixable, return parameter modifications."""
        if not report.auto_fixable:
            return None

        fixes = {}

        if report.problem.startswith("电子步不收敛"):
            fixes["ALGO"] = "Normal"
            fixes["NELM"] = "100"
            fixes["AMIX"] = "0.1"
            fixes["BMIX"] = "1.0"

        elif report.problem.startswith("离子步不收敛"):
            fixes["POTIM"] = "0.1"
            fixes["EDIFFG"] = "-0.05"

        elif "能带数不足" in report.problem:
            fixes["NBANDS"] = "auto"  # Will be computed at runtime

        elif report.problem.startswith("交换关联势错误"):
            fixes["_action"] = "regenerate_potcar"

        return fixes if fixes else None
