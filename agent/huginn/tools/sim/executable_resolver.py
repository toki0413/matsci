"""统一可执行文件发现 — 找不到时问用户, 而不是默默 mock.

每个仿真工具 (VASP / LAMMPS / Gaussian / ...) 调 resolve("vasp") 拿路径.
找不到时返回 ResolutionRequest, 上层 (autoloop / chat) 拿到后向用户提问,
用户回答的路径缓存到 ~/.huginn/executables.json + 环境变量, 下次直接命中.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_FILE = Path.home() / ".huginn" / "executables.json"


@dataclass(frozen=True)
class ToolExecutableSpec:
    """一个仿真工具的可执行文件规格."""

    name: str
    env_vars: tuple[str, ...]
    basenames: tuple[str, ...]
    install_hint: str
    conda_package: str | None = None
    license_required: bool = False
    hpc_common: bool = False  # 通常跑在 HPC 上而非本地


# 全量注册表 — 涵盖所有 sim 工具
_REGISTRY: dict[str, ToolExecutableSpec] = {
    "vasp": ToolExecutableSpec(
        name="vasp",
        env_vars=("VASP_EXECUTABLE", "VASP_PP_PATH"),
        basenames=("vasp", "vasp_std", "vasp_gam", "vasp_ncl"),
        install_hint="VASP 需要商业许可证。已购买后从官网下载源码编译，或联系管理员获取 HPC 上的路径。",
        license_required=True,
        hpc_common=True,
    ),
    "lammps": ToolExecutableSpec(
        name="lammps",
        env_vars=("LAMMPS_EXECUTABLE",),
        basenames=("lmp", "lmp_serial", "lmp_mpi", "lammps"),
        install_hint="conda install -c conda-forge lammps  或  sudo apt install lammps",
        conda_package="conda-forge/lammps",
    ),
    "qe": ToolExecutableSpec(
        name="qe",
        env_vars=("QE_EXECUTABLE",),
        basenames=("pw.x", "cp.x", "pw", "ph.x"),
        install_hint="conda install -c conda-forge qe  或从 https://www.quantum-espresso.org/ 编译",
        conda_package="conda-forge/qe",
    ),
    "cp2k": ToolExecutableSpec(
        name="cp2k",
        env_vars=("CP2K_EXECUTABLE",),
        basenames=("cp2k", "cp2k.popt", "cp2k.psmp", "cp2k.sopt"),
        install_hint="conda install -c conda-forge cp2k  或从 https://www.cp2k.org/ 编译",
        conda_package="conda-forge/cp2k",
    ),
    "gaussian": ToolExecutableSpec(
        name="gaussian",
        env_vars=("GAUSSIAN_EXECUTABLE", "GAUSS_EXEDIR"),
        basenames=("g16", "g09", "gaussian"),
        install_hint="Gaussian 需要商业许可证。已购买后设置 GAUSS_EXEDIR 环境变量指向安装目录。",
        license_required=True,
        hpc_common=True,
    ),
    "orca": ToolExecutableSpec(
        name="orca",
        env_vars=("ORCA_EXECUTABLE", "ORCADIR"),
        basenames=("orca", "orca_4_2_1", "orca_5_0_x"),
        install_hint="ORCA 免费但需注册。从 https://www.faccts.de/orca/ 下载，设置 ORCADIR 环境变量。",
        license_required=False,
        hpc_common=True,
    ),
    "gromacs": ToolExecutableSpec(
        name="gromacs",
        env_vars=("GROMACS_EXECUTABLE", "GMXBIN"),
        basenames=("gmx", "gmx_mpi", "gromacs"),
        install_hint="conda install -c conda-forge gromacs  或从 http://www.gromacs.org/ 编译",
        conda_package="conda-forge/gromacs",
    ),
    "comsol": ToolExecutableSpec(
        name="comsol",
        env_vars=("COMSOL_EXECUTABLE", "COMSOL_HOME"),
        basenames=("comsol", "comsolbatch"),
        install_hint="COMSOL Multiphysics 需要商业许可证。设置 COMSOL_HOME 指向安装目录。",
        license_required=True,
        hpc_common=True,
    ),
    "abaqus": ToolExecutableSpec(
        name="abaqus",
        env_vars=("ABAQUS_EXECUTABLE",),
        basenames=("abaqus", "abaqus.bat"),
        install_hint="Abaqus 需要商业许可证 (SIMULIA)。安装后 abaqus 通常在 PATH 中。",
        license_required=True,
        hpc_common=True,
    ),
    "openfoam": ToolExecutableSpec(
        name="openfoam",
        env_vars=("OPENFOAM_DIR", "WM_PROJECT_DIR"),
        basenames=("blockMesh", "simpleFoam", "icoFoam"),
        install_hint="conda install -c conda-forge openfoam  或从 https://openfoam.org/ 安装",
        conda_package="conda-forge/openfoam",
    ),
    "packmol": ToolExecutableSpec(
        name="packmol",
        env_vars=("PACKMOL_EXECUTABLE",),
        basenames=("packmol",),
        install_hint="conda install -c conda-forge packmol",
        conda_package="conda-forge/packmol",
    ),
    "fenics": ToolExecutableSpec(
        name="fenics",
        env_vars=("FENICS_EXECUTABLE",),
        basenames=("python",),  # fenics 是 python 库, 不是独立可执行文件
        install_hint="conda install -c conda-forge fenics  或 pip install fenics-dijit",
        conda_package="conda-forge/fenics",
    ),
    "elmer": ToolExecutableSpec(
        name="elmer",
        env_vars=("ELMER_EXECUTABLE",),
        basenames=("ElmerSolver", "ElmerSolver_mpi", "ElmerGrid"),
        install_hint="conda install -c conda-forge elmerfem  或从 https://www.elmerfem.org/ 编译",
        conda_package="conda-forge/elmerfem",
    ),
    "autodock_vina": ToolExecutableSpec(
        name="autodock_vina",
        env_vars=("VINA_EXECUTABLE",),
        basenames=("vina", "vina_split", "vinardo"),
        install_hint="conda install -c conda-forge autodock-vina  或从 https://github.com/ccsb-scripps/AutoDock-Vina 编译",
        conda_package="conda-forge/autodock-vina",
    ),
}


@dataclass
class ResolutionRequest:
    """可执行文件未找到时, 返回给上层用于向用户提问的上下文."""

    tool_name: str
    spec: ToolExecutableSpec
    env_vars_checked: list[str] = field(default_factory=list)
    basenames_checked: list[str] = field(default_factory=list)

    @property
    def question(self) -> str:
        spec = self.spec
        if spec.license_required:
            return (
                f"{spec.name.upper()} 未在本地找到。"
                f"该软件需要许可证，通常部署在 HPC 集群上。\n"
                f"请提供安装路径，或告诉我它在哪里运行。"
            )
        if spec.hpc_common:
            return (
                f"{spec.name.upper()} 未在本地找到。\n"
                f"请提供安装路径，或告诉我它在哪台机器上。"
            )
        return (
            f"{spec.name.upper()} 未在本地找到。\n"
            f"你可以提供安装路径，或让我帮你安装。"
        )

    @property
    def options(self) -> list[str]:
        opts = [
            "提供本地安装路径",
        ]
        if self.spec.hpc_common:
            opts.append("在 HPC 集群上 (通过 SSH 提交)")
        if self.spec.conda_package:
            opts.append(f"帮我安装 ({self.spec.conda_package})")
        elif not self.spec.license_required:
            opts.append("帮我安装")
        opts.append("跳过 (使用 mock 模式继续)")
        return opts

    @property
    def install_hint(self) -> str:
        return self.spec.install_hint

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "question": self.question,
            "options": self.options,
            "install_hint": self.install_hint,
            "license_required": self.spec.license_required,
            "env_vars": list(self.spec.env_vars),
            "basenames": list(self.spec.basenames),
            "conda_package": self.spec.conda_package,
        }


class ExecutableResolver:
    """统一可执行文件发现 + 用户路径缓存."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        try:
            if _CACHE_FILE.exists():
                data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._cache = data
        except Exception:
            logger.debug("failed to load executable cache", exc_info=True)

    def _save_cache(self) -> None:
        try:
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(
                json.dumps(self._cache, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.debug("failed to save executable cache", exc_info=True)

    def resolve(self, tool_name: str) -> str | ResolutionRequest:
        """查找可执行文件路径, 找不到返回 ResolutionRequest.

        查找顺序: 用户缓存 → 环境变量 → PATH.
        """
        spec = _REGISTRY.get(tool_name)
        if spec is None:
            # 未注册的工具, 直接走 PATH
            exe = shutil.which(tool_name)
            return exe if exe else ResolutionRequest(
                tool_name=tool_name,
                spec=ToolExecutableSpec(
                    name=tool_name, env_vars=(), basenames=(tool_name,),
                    install_hint=f"请安装 {tool_name}",
                ),
            )

        # 1. 用户缓存
        cached = self._cache.get(tool_name)
        if cached and Path(cached).exists():
            return cached

        # 2. 环境变量
        env_checked: list[str] = []
        for var in spec.env_vars:
            val = os.environ.get(var)
            env_checked.append(f"{var}={val or '(未设置)'}")
            if val:
                p = Path(val)
                if p.exists() and p.is_file():
                    self._cache[tool_name] = str(p)
                    self._save_cache()
                    return str(p)
                # 可能是目录, 尝试拼接 basename
                for bn in spec.basenames:
                    candidate = p / bn
                    if candidate.exists():
                        self._cache[tool_name] = str(candidate)
                        self._save_cache()
                        return str(candidate)

        # 3. PATH
        basenames_checked: list[str] = []
        for bn in spec.basenames:
            basenames_checked.append(bn)
            exe = shutil.which(bn)
            if exe:
                self._cache[tool_name] = exe
                self._save_cache()
                return exe

        # 都没找到, 返回提问请求
        return ResolutionRequest(
            tool_name=tool_name,
            spec=spec,
            env_vars_checked=env_checked,
            basenames_checked=basenames_checked,
        )

    def register_path(self, tool_name: str, path: str) -> bool:
        """用户提供的路径, 验证后缓存."""
        p = Path(path)
        if not p.exists():
            # 可能是目录
            spec = _REGISTRY.get(tool_name)
            if spec:
                for bn in spec.basenames:
                    candidate = p / bn
                    if candidate.exists():
                        self._cache[tool_name] = str(candidate)
                        self._save_cache()
                        os.environ.setdefault(spec.env_vars[0], str(candidate))
                        return True
            logger.warning("path does not exist: %s", path)
            return False
        self._cache[tool_name] = str(p)
        self._save_cache()
        spec = _REGISTRY.get(tool_name)
        if spec and spec.env_vars:
            os.environ.setdefault(spec.env_vars[0], str(p))
        return True

    def get_install_command(self, tool_name: str) -> str | None:
        spec = _REGISTRY.get(tool_name)
        if spec and spec.conda_package:
            return f"conda install -c {spec.conda_package}"
        return None


# singleton
_resolver: ExecutableResolver | None = None


def get_resolver() -> ExecutableResolver:
    global _resolver
    if _resolver is None:
        _resolver = ExecutableResolver()
    return _resolver


def resolve_executable(tool_name: str) -> str | ResolutionRequest:
    """模块级快捷函数."""
    return get_resolver().resolve(tool_name)
