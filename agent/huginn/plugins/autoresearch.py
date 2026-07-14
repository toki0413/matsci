"""AutoResearch plugin for Huginn.

Integrates Andrej Karpathy's ``autoresearch`` autonomous ML experimentation
loop (https://github.com/karpathy/autoresearch) as a first-class Huginn tool
and CLI command. Huginn can initialize an autoresearch workspace, run the
data-prep step, execute fixed-time experiments, and act as the agent that
proposes edits to ``train.py``, keeps improvements, and reverts failed ideas.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class AutoresearchInput(BaseModel):
    """Input schema for the autoresearch tool."""

    action: Literal[
        "init_workspace",
        "prepare",
        "run_experiment",
        "results",
        "status",
        "propose_edit",
        "step",
        "loop",
    ] = Field(default="status", description="Operation to perform")
    workspace: str = Field(
        default=".", description="Path to the autoresearch workspace"
    )
    repo_url: str | None = Field(
        default=None,
        description="Git URL to clone when initializing (default: karpathy/autoresearch)",
    )
    branch: str | None = Field(
        default=None,
        description="Git branch/tag to use for experiments (e.g. autoresearch/mar5)",
    )
    program_append: str | None = Field(
        default=None,
        description="Extra instructions appended to program.md when initializing",
    )
    max_iterations: int = Field(
        default=1, ge=1, le=1000, description="Iterations for the loop action"
    )
    user_hint: str | None = Field(
        default=None,
        description="Hint for the agent when proposing edits",
    )
    timeout: int = Field(
        default=600, ge=10, le=7200, description="Seconds to wait for an experiment"
    )
    metrics_lower_is_better: bool = Field(
        default=True,
        description="Whether a lower metric value is better (autoresearch uses val_bpb)",
    )
    train_py: str | None = Field(
        default=None,
        description="Full train.py content to apply in step/loop (bypasses LLM proposal)",
    )
    description: str | None = Field(
        default=None,
        description="Short description for a manual step/experiment",
    )
    skip_git: bool = Field(
        default=False,
        description="Skip git operations (useful for tests or non-git workspaces)",
    )


class AutoresearchTool(HuginnTool):
    """Manage and drive an AutoResearch workspace.

    The tool wraps the three core autoresearch files (``prepare.py``,
    ``train.py``, ``program.md``) and the experiment ratchet: propose an edit,
    run a time-boxed training experiment, parse ``val_bpb``, and keep or
    revert the change. When ``uv`` is available it runs ``uv run <script>``;
    otherwise it falls back to the active Python interpreter so the workspace
    can be exercised even without ``uv``.
    """

    name = "autoresearch_tool"
    category = "meta"
    description = (
        "Initialize and drive an AutoResearch workspace: prepare data, run "
        "fixed-time experiments, propose edits to train.py, and ratchet "
        "toward a better val_bpb."
    )
    destructive = True
    input_schema = AutoresearchInput

    # Regex for the summary block printed by autoresearch train.py
    _METRIC_RE = re.compile(
        r"^(val_bpb|training_seconds|total_seconds|peak_vram_mb|mfu_percent|total_tokens_M|num_steps|num_params_M|depth):\s*([\d.]+)",
        re.M,
    )
    _JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.S)

    async def call(self, args: AutoresearchInput, context: ToolContext) -> ToolResult:
        try:
            if args.action == "init_workspace":
                return await self._init_workspace(args)
            if args.action == "prepare":
                return await self._prepare(args)
            if args.action == "run_experiment":
                return await self._run_experiment(args)
            if args.action == "results":
                return self._results(args)
            if args.action == "status":
                return await self._status(args)
            if args.action == "propose_edit":
                return await self._propose_edit(args)
            if args.action == "step":
                return await self._step(args)
            if args.action == "loop":
                return await self._loop(args)
            return ToolResult(
                data=None, success=False, error=f"Unknown action: {args.action}"
            )
        except Exception as exc:  # pragma: no cover - safety net
            return ToolResult(data=None, success=False, error=str(exc))

    # ------------------------------------------------------------------ helpers

    def _workspace(self, args: AutoresearchInput) -> Path:
        return Path(args.workspace).expanduser().resolve()

    def _uv_available(self) -> bool:
        return shutil.which("uv") is not None

    def _git_available(self) -> bool:
        return shutil.which("git") is not None

    async def _run_command(
        self,
        args: list[str],
        cwd: Path,
        timeout: int,
        capture_output: bool = True,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run a command in the workspace.

        Prefer ``uv run <script>`` when ``uv`` is present; otherwise fall back
        to the active Python interpreter for ``.py`` arguments.
        """
        if args and args[0].endswith(".py") and not self._uv_available():
            command = [sys.executable, *args]
        elif self._uv_available():
            command = ["uv", "run", *args]
        else:
            command = args

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._run_command_sync(command, cwd, timeout, capture_output, env),
        )

    def _run_command_sync(
        self,
        command: list[str],
        cwd: Path,
        timeout: int,
        capture_output: bool = True,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        merged_env = {**dict(os.environ), **(env or {})}
        try:
            proc = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                env=merged_env,
                errors="replace",
            )
            return {
                "command": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout or "",
                "stderr": proc.stderr or "",
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "command": command,
                "returncode": -1,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "timed_out": True,
            }
        except FileNotFoundError as exc:
            return {
                "command": command,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command not found: {exc}",
            }

    def _require_git(self, args: AutoresearchInput) -> ToolResult | None:
        if args.skip_git:
            return None
        if not self._git_available():
            return ToolResult(
                data=None,
                success=False,
                error="git is not installed. Set skip_git=True to operate without git.",
            )
        return None

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def _write_text(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def _parse_run_metrics(self, text: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in self._METRIC_RE.findall(text):
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
        return metrics

    def _extract_json(self, text: str) -> str:
        text = text.strip()
        match = self._JSON_RE.search(text)
        if match:
            return match.group(1).strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        # Try to find the first JSON object if the model added commentary.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    # ------------------------------------------------------------------ actions

    async def _init_workspace(self, args: AutoresearchInput) -> ToolResult:
        ws = self._workspace(args)
        if not ws.exists():
            ws.mkdir(parents=True, exist_ok=True)

        # Clone upstream repo if requested and workspace is empty.
        if args.repo_url or not any(ws.iterdir()):
            repo = args.repo_url or "https://github.com/karpathy/autoresearch.git"
            git_err = self._require_git(args)
            if git_err:
                return git_err
            clone = await self._run_command(
                ["git", "clone", repo, str(ws)],
                cwd=ws.parent,
                timeout=120,
            )
            if clone["returncode"] != 0:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Failed to clone {repo}: {clone['stderr']}",
                )

        if not (ws / "train.py").exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"Workspace {ws} does not contain a train.py file.",
            )

        # Ensure the directory is a git repo so we can commit/reset later.
        if not args.skip_git and self._git_available():
            git_dir = ws / ".git"
            if not git_dir.exists():
                await self._run_command(["git", "init"], cwd=ws, timeout=30)
                await self._run_command(
                    ["git", "config", "user.email", "huginn@local"],
                    cwd=ws,
                    timeout=30,
                )
                await self._run_command(
                    ["git", "config", "user.name", "Huginn AutoResearch"],
                    cwd=ws,
                    timeout=30,
                )
                await self._run_command(["git", "add", "."], cwd=ws, timeout=30)
                await self._run_command(
                    ["git", "commit", "-m", "Initial commit from Huginn"],
                    cwd=ws,
                    timeout=30,
                )

        # Create a dedicated experiment branch.
        branch = args.branch or f"autoresearch/huginn-{datetime.now():%Y%m%d}"
        if not args.skip_git and self._git_available():
            await self._run_command(
                ["git", "checkout", "-b", branch],
                cwd=ws,
                timeout=30,
            )

        # Initialize the results log if missing.
        results_tsv = ws / "results.tsv"
        if not results_tsv.exists():
            results_tsv.write_text(
                "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n",
                encoding="utf-8",
            )

        # Append user-specific instructions to program.md.
        program_md = ws / "program.md"
        if args.program_append and program_md.exists():
            program_md.write_text(
                program_md.read_text(encoding="utf-8")
                + "\n\n## Huginn-added instructions\n\n"
                + args.program_append
                + "\n",
                encoding="utf-8",
            )

        return ToolResult(
            data={
                "workspace": str(ws),
                "branch": branch,
                "train_py": str(ws / "train.py"),
                "program_md": str(program_md) if program_md.exists() else None,
                "results_tsv": str(results_tsv),
            }
        )

    async def _prepare(self, args: AutoresearchInput) -> ToolResult:
        ws = self._workspace(args)
        if not (ws / "prepare.py").exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"prepare.py not found in {ws}",
            )
        result = await self._run_command(
            ["prepare.py"],
            cwd=ws,
            timeout=300,
        )
        return ToolResult(
            data={
                "command": result["command"],
                "returncode": result["returncode"],
                "stdout": result["stdout"][-4000:],
                "stderr": result["stderr"][-2000:],
            },
            success=result["returncode"] == 0,
            error=result["stderr"] if result["returncode"] != 0 else None,
        )

    async def _run_experiment(self, args: AutoresearchInput) -> ToolResult:
        ws = self._workspace(args)
        if not (ws / "train.py").exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"train.py not found in {ws}",
            )

        result = await self._run_command(
            ["train.py"],
            cwd=ws,
            timeout=args.timeout,
        )
        output = result["stdout"] + "\n" + result["stderr"]
        (ws / "run.log").write_text(output, encoding="utf-8")
        metrics = self._parse_run_metrics(output)
        crashed = result["returncode"] != 0 or "val_bpb" not in metrics

        return ToolResult(
            data={
                "command": result["command"],
                "returncode": result["returncode"],
                "timed_out": result.get("timed_out", False),
                "metrics": metrics,
                "crashed": crashed,
                "log_tail": output[-2000:],
            },
            success=not crashed,
            error=(
                result["stderr"][:1000]
                if crashed and not result.get("timed_out")
                else None
            ),
        )

    def _results(self, args: AutoresearchInput) -> ToolResult:
        ws = self._workspace(args)
        results_tsv = ws / "results.tsv"
        if not results_tsv.exists():
            return ToolResult(data={"rows": [], "columns": []})

        with results_tsv.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        return ToolResult(
            data={
                "columns": reader.fieldnames or [],
                "rows": rows,
                "count": len(rows),
            }
        )

    async def _status(self, args: AutoresearchInput) -> ToolResult:
        ws = self._workspace(args)
        if args.skip_git or not self._git_available():
            return ToolResult(
                data={
                    "workspace": str(ws),
                    "git": False,
                    "train_py_exists": (ws / "train.py").exists(),
                }
            )

        # 三个 git 查询彼此独立，丢到线程池并行跑，避免串行阻塞事件循环
        branch_proc, commit_proc, dirty_proc = await asyncio.gather(
            asyncio.to_thread(
                subprocess.run,
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(ws),
                capture_output=True,
                text=True,
            ),
            asyncio.to_thread(
                subprocess.run,
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(ws),
                capture_output=True,
                text=True,
            ),
            asyncio.to_thread(
                subprocess.run,
                ["git", "status", "--porcelain"],
                cwd=str(ws),
                capture_output=True,
                text=True,
            ),
        )
        return ToolResult(
            data={
                "workspace": str(ws),
                "branch": (
                    branch_proc.stdout.strip() if branch_proc.returncode == 0 else None
                ),
                "commit": (
                    commit_proc.stdout.strip() if commit_proc.returncode == 0 else None
                ),
                "dirty": bool(dirty_proc.stdout.strip()),
            }
        )

    async def _propose_edit(self, args: AutoresearchInput) -> ToolResult:
        ws = self._workspace(args)
        program_md = self._read_text(ws / "program.md")
        train_py = self._read_text(ws / "train.py")
        results = self._results(args).data.get("rows", [])

        if not train_py:
            return ToolResult(
                data=None, success=False, error=f"train.py missing in {ws}"
            )

        new_train, description, hypothesis = await self._llm_propose_edit(
            ws=ws,
            program_md=program_md,
            train_py=train_py,
            results=results,
            user_hint=args.user_hint,
            config=getattr(args, "config", None),
        )
        return ToolResult(
            data={
                "train_py": new_train,
                "description": description,
                "hypothesis": hypothesis,
            }
        )

    async def _step(self, args: AutoresearchInput) -> ToolResult:
        ws = self._workspace(args)
        git_err = self._require_git(args)
        if git_err:
            return git_err

        # 1. Obtain the proposed train.py.
        if args.train_py:
            new_train = args.train_py
            description = args.description or "manual edit"
            hypothesis = ""
        else:
            proposal = await self._propose_edit(args)
            if not proposal.success:
                return proposal
            new_train = proposal.data["train_py"]
            description = proposal.data["description"]
            hypothesis = proposal.data.get("hypothesis", "")

        # 2. Apply and commit.
        self._write_text(ws / "train.py", new_train)
        if not args.skip_git and self._git_available():
            commit = await self._git_commit(ws, description)
            if not commit:
                return ToolResult(
                    data=None,
                    success=False,
                    error="Failed to commit proposed train.py",
                )
        else:
            commit = "no-git"

        # 3. Run experiment.
        experiment = await self._run_experiment(args)
        if not experiment.success:
            # Revert failed/crashed runs.
            memory_gb = 0.0
            val_bpb = 0.0
            status = "crash" if experiment.data.get("crashed") else "discard"
            if not args.skip_git and self._git_available():
                await self._git_reset(ws)
            self._log_result(ws, commit, val_bpb, memory_gb, status, description)
            return ToolResult(
                data={
                    "commit": commit,
                    "status": status,
                    "description": description,
                    "experiment": experiment.data,
                    "kept": False,
                    "hypothesis": hypothesis,
                },
                success=False,
                error=experiment.error or "Experiment failed",
            )

        metrics = experiment.data.get("metrics", {})
        val_bpb = metrics.get("val_bpb")
        peak_vram_mb = metrics.get("peak_vram_mb", 0.0)
        memory_gb = round(peak_vram_mb / 1024, 1) if peak_vram_mb else 0.0

        # 4. Decide keep/discard.
        prior_rows = self._results(args).data.get("rows", [])
        best_prior = self._best_metric(prior_rows, args.metrics_lower_is_better)
        improved = self._is_improved(val_bpb, best_prior, args.metrics_lower_is_better)

        if improved or best_prior is None:
            status = "keep"
            kept = True
        else:
            status = "discard"
            kept = False
            if not args.skip_git and self._git_available():
                await self._git_reset(ws)

        self._log_result(ws, commit, val_bpb or 0.0, memory_gb, status, description)

        return ToolResult(
            data={
                "commit": commit,
                "status": status,
                "description": description,
                "metrics": metrics,
                "best_prior": best_prior,
                "kept": kept,
                "hypothesis": hypothesis,
            }
        )

    async def _loop(self, args: AutoresearchInput) -> ToolResult:
        ws = self._workspace(args)
        summary: list[dict] = []
        for i in range(args.max_iterations):
            step_args = AutoresearchInput(
                action="step",
                workspace=args.workspace,
                user_hint=args.user_hint,
                timeout=args.timeout,
                metrics_lower_is_better=args.metrics_lower_is_better,
                skip_git=args.skip_git,
            )
            result = await self._step(step_args)
            summary.append(
                {
                    "iteration": i + 1,
                    "success": result.success,
                    "data": result.data,
                    "error": result.error,
                }
            )
            if not result.success and result.data.get("status") == "crash":
                # Keep looping through crashes; autoresearch is meant to be robust.
                continue
        return ToolResult(data={"workspace": str(ws), "iterations": summary})

    # ------------------------------------------------------------------ git

    async def _git_commit(self, ws: Path, message: str) -> str | None:
        await self._run_command(["git", "add", "train.py"], cwd=ws, timeout=30)
        commit_result = await self._run_command(
            ["git", "commit", "-m", message],
            cwd=ws,
            timeout=30,
        )
        if commit_result["returncode"] != 0:
            return None
        rev = await self._run_command(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ws,
            timeout=10,
        )
        return rev["stdout"].strip() if rev["returncode"] == 0 else None

    async def _git_reset(self, ws: Path) -> None:
        await self._run_command(
            ["git", "reset", "--hard", "HEAD~1"],
            cwd=ws,
            timeout=30,
        )

    # ------------------------------------------------------------------ metrics

    def _best_metric(
        self, rows: list[dict[str, Any]], lower_is_better: bool
    ) -> float | None:
        values: list[float] = []
        for row in rows:
            try:
                if row.get("status") in ("keep", "discard"):
                    values.append(float(row["val_bpb"]))
            except (KeyError, ValueError):
                continue
        if not values:
            return None
        return min(values) if lower_is_better else max(values)

    def _is_improved(
        self, value: float | None, baseline: float | None, lower_is_better: bool
    ) -> bool:
        if value is None:
            return False
        if baseline is None:
            return True
        return value < baseline if lower_is_better else value > baseline

    def _log_result(
        self,
        ws: Path,
        commit: str,
        val_bpb: float,
        memory_gb: float,
        status: str,
        description: str,
    ) -> None:
        results_tsv = ws / "results.tsv"
        if not results_tsv.exists():
            results_tsv.write_text(
                "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n",
                encoding="utf-8",
            )
        with results_tsv.open("a", encoding="utf-8") as f:
            f.write(
                f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n"
            )

    # ------------------------------------------------------------------ LLM

    async def _llm_propose_edit(
        self,
        ws: Path,
        program_md: str,
        train_py: str,
        results: list[dict[str, Any]],
        user_hint: str | None,
        config: Any | None,
    ) -> tuple[str, str, str]:
        from langchain_core.messages import HumanMessage, SystemMessage

        from huginn.llm import get_model

        model = get_model(config=config, temperature=0.4, max_tokens=16000)

        recent_results = (
            json.dumps(results[-10:], indent=2) if results else "(none yet)"
        )
        hint_block = f"\nAdditional user hint: {user_hint}\n" if user_hint else ""

        system = (
            "You are an autonomous ML researcher running the autoresearch loop. "
            "You may modify ONLY train.py. Do not change prepare.py, constants, "
            "or the evaluation harness. Keep changes simple and reversible. "
            "Output ONLY a JSON object with keys: hypothesis, description, train_py."
        )
        prompt = (
            f"Workspace: {ws}\n\n"
            "--- program.md ---\n"
            f"{program_md}\n\n"
            "--- recent results.tsv rows ---\n"
            f"{recent_results}\n"
            f"{hint_block}\n\n"
            "--- current train.py ---\n"
            "```python\n"
            f"{train_py}\n"
            "```\n\n"
            "Propose the next experimental change to train.py. "
            "Return a JSON object exactly like:\n"
            "{\n"
            '  "hypothesis": "what you expect to happen",\n'
            '  "description": "short experiment description",\n'
            '  "train_py": "full new content of train.py as a string"\n'
            "}\n"
        )

        response = await asyncio.to_thread(
            model.invoke, [SystemMessage(system), HumanMessage(prompt)]
        )
        content = response.content if hasattr(response, "content") else str(response)
        raw_json = self._extract_json(content)
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"LLM did not return valid JSON: {exc}\n{content[:500]}"
            ) from exc

        new_train = data.get("train_py")
        if not isinstance(new_train, str) or not new_train.strip():
            raise RuntimeError("LLM response missing train_py content")

        return (
            new_train,
            data.get("description", "LLM-proposed edit"),
            data.get("hypothesis", ""),
        )
