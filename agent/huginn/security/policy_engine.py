"""Declarative security policy engine -- replace hardcoded if-else
with a YAML/TOML-driven rule system.

Inspired by Codex's Starlark execpolicy, but using Python-native
rule evaluation (no Starlark dependency needed).

Policy file format (YAML)::

    rules:
      - name: "allow_dft_tools"
        match:
          executable: ["vasp", "pw.x", "cp2k.popt"]
          workspace: "${HUGINN_WORKSPACE}"
        action: allow
        reason: "DFT simulation tools"

      - name: "deny_rm_rf"
        match:
          command_pattern: "rm\\s+-rf\\s+/"
        action: deny
        reason: "Prevent recursive deletion of root"

      - name: "ask_hpc_submit"
        match:
          executable: ["sbatch", "qsub", "msub"]
        action: ask
        reason: "HPC job submission requires user confirmation"

    defaults:
      unmatched: ask  # allow / deny / ask

Rules are evaluated top-to-bottom; first match wins.  Match conditions
use AND logic -- every condition present in ``match`` must pass.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_VALID_ACTIONS = frozenset({"allow", "deny", "ask"})


@dataclass
class PolicyRule:
    """A single declarative rule: match conditions -> action."""

    name: str
    match: dict  # executable, command_pattern, workspace, env_required
    action: str  # allow / deny / ask
    reason: str = ""


@dataclass
class PolicyDecision:
    """Result of evaluating a command against the policy."""

    action: str  # allow / deny / ask
    matched_rule: str | None
    reason: str


class PolicyEngine:
    """Evaluates command execution requests against declarative rules.

    Rules are evaluated in order -- first match wins.  If no rule
    matches, ``defaults.unmatched`` from the policy file is returned.
    """

    _instance: PolicyEngine | None = None

    @classmethod
    def shared(cls) -> PolicyEngine:
        """Singleton accessor -- most callers want the shared engine."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._rules: list[PolicyRule] = []
        self._default_action: str = "ask"
        self._load_default_policy()

    # -- public API ---------------------------------------------------

    def evaluate(
        self,
        executable: str,
        command: str,
        workspace: str | None = None,
        env: dict | None = None,
    ) -> PolicyDecision:
        """Evaluate a command against the policy rules. First match wins."""
        exe = _normalize_executable(executable)
        for rule in self._rules:
            if self._rule_matches(rule, exe, command, workspace, env):
                return PolicyDecision(rule.action, rule.name, rule.reason)
        return PolicyDecision(
            self._default_action,
            None,
            f"No rule matched -- default action is '{self._default_action}'",
        )

    def load_policy_file(self, path: str) -> None:
        """Load policy rules from a YAML or TOML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")

        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            data = self._load_yaml(p)
        elif suffix == ".toml":
            data = self._load_toml(p)
        else:
            raise ValueError(
                f"Unsupported policy format: {suffix} (use .yaml, .yml, or .toml)"
            )
        self._apply_policy_dict(data)

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a rule at runtime."""
        self._rules.append(rule)

    @property
    def rules(self) -> list[PolicyRule]:
        """Snapshot of current rules (read-only)."""
        return list(self._rules)

    @property
    def default_action(self) -> str:
        return self._default_action

    # -- rule matching ------------------------------------------------

    def _rule_matches(
        self,
        rule: PolicyRule,
        executable: str,
        command: str,
        workspace: str | None,
        env: dict | None,
    ) -> bool:
        """All conditions in rule.match must pass (AND logic)."""
        match = rule.match
        if not match:
            return False

        if "executable" in match:
            allowed = {e.lower() for e in match["executable"]}
            if executable.lower() not in allowed:
                return False

        if "command_pattern" in match:
            pattern = match["command_pattern"]
            if not re.search(pattern, command, re.IGNORECASE):
                return False

        if "workspace" in match and workspace is not None:
            if not self._workspace_matches(match["workspace"], workspace):
                return False

        if "env_required" in match:
            check_env = env if env is not None else os.environ
            for var in match["env_required"]:
                if var not in check_env:
                    return False

        return True

    @staticmethod
    def _workspace_matches(pattern: str, workspace: str) -> bool:
        """Check if *workspace* starts with the env-expanded *pattern*.

        If the env var in the pattern isn't set we skip the check rather
        than blocking -- operators who don't set ``HUGINN_WORKSPACE``
        shouldn't have all their tools locked out.
        """
        expanded = os.path.expandvars(pattern)
        if expanded == pattern:
            # Var not set -- can't enforce, don't block
            # ponytail: if HUGINN_WORKSPACE is unset we lose path scoping,
            # but blocking all tools would be worse. Upgrade: enforce
            # workspace via config rather than env var.
            return True
        ws = workspace.replace("\\", "/")
        exp = expanded.replace("\\", "/")
        return ws.startswith(exp)

    # -- loading ------------------------------------------------------

    def _load_default_policy(self) -> None:
        """Load built-in defaults that cover common materials science tools."""
        default_path = Path(__file__).parent / "default_policy.yaml"
        if default_path.exists():
            try:
                self.load_policy_file(str(default_path))
                return
            except Exception:
                pass  # fall through to hardcoded fallback

        self._load_hardcoded_fallback()

    def _load_hardcoded_fallback(self) -> None:
        """Build rules from the legacy hardcoded sets.

        Only used when default_policy.yaml can't be loaded -- keeps
        behaviour identical to the old sandbox.py + command_filter.py.
        """
        from huginn.security.command_filter import _BLOCKED_PATTERNS
        from huginn.security.sandbox import SandboxConfig

        # Deny patterns first (same order as command_filter)
        for i, pattern in enumerate(_BLOCKED_PATTERNS):
            self._rules.append(
                PolicyRule(
                    name=f"deny_pattern_{i}",
                    match={"command_pattern": pattern},
                    action="deny",
                    reason=f"Blocked command pattern (legacy): {pattern}",
                )
            )

        # Then allow the legacy whitelist
        cfg = SandboxConfig()
        self._rules.append(
            PolicyRule(
                name="allow_whitelisted_executables",
                match={"executable": list(cfg.allowed_executables)},
                action="allow",
                reason="Executable is in the legacy sandbox whitelist",
            )
        )
        self._default_action = "ask"

    def _apply_policy_dict(self, data: dict) -> None:
        """Parse a policy dict (from YAML/TOML) into rules."""
        for r in data.get("rules", []):
            name = r.get("name", f"rule_{len(self._rules)}")
            action = r.get("action", "ask")
            if action not in _VALID_ACTIONS:
                raise ValueError(
                    f"Invalid action '{action}' in rule '{name}'. "
                    f"Must be one of: {', '.join(sorted(_VALID_ACTIONS))}"
                )
            self._rules.append(
                PolicyRule(
                    name=name,
                    match=r.get("match", {}),
                    action=action,
                    reason=r.get("reason", ""),
                )
            )

        defaults = data.get("defaults", {})
        if "unmatched" in defaults:
            action = defaults["unmatched"]
            if action not in _VALID_ACTIONS:
                raise ValueError(
                    f"Invalid default action '{action}'. "
                    f"Must be one of: {', '.join(sorted(_VALID_ACTIONS))}"
                )
            self._default_action = action

    # -- file format helpers ------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _load_toml(path: Path) -> dict:
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError:
                raise ImportError(
                    "TOML policy files require Python 3.11+ or the 'tomli' package"
                )
        with open(path, "rb") as f:
            return tomllib.load(f)


# -- helpers --------------------------------------------------------------


def _normalize_executable(exe: str) -> str:
    """Strip path prefix and .exe suffix for case-insensitive matching."""
    name = Path(exe).name
    if name.lower().endswith(".exe"):
        name = name[:-4]
    return name


def evaluate_command_hook(
    cmd: list[str],
    workspace: str | None = None,
    env: dict | None = None,
) -> PolicyDecision:
    """Integration hook for ``SandboxExecutor._validate_command``.

    Returns a :class:`PolicyDecision` that the caller handles:

    - ``allow`` -- proceed with execution
    - ``deny``  -- raise / block the command
    - ``ask``   -- prompt the user for confirmation
    """
    if not cmd:
        return PolicyDecision("deny", None, "Empty command")

    engine = PolicyEngine.shared()
    executable = cmd[0]
    command_str = " ".join(cmd)
    return engine.evaluate(executable, command_str, workspace=workspace, env=env)


# -- self-check -----------------------------------------------------------


if __name__ == "__main__":
    eng = PolicyEngine()

    # deny patterns
    assert eng.evaluate("rm", "rm -rf /").action == "deny"
    assert eng.evaluate("rm", "rm -fr /").action == "deny"
    assert eng.evaluate("rm", "rm -rf ~").action == "deny"
    assert eng.evaluate("sudo", "sudo apt install foo").action == "deny"
    assert eng.evaluate("dd", "dd if=/dev/zero of=/dev/sda").action == "deny"
    assert eng.evaluate("kill", "kill -9 1").action == "deny"
    assert eng.evaluate("killall", "killall python").action == "deny"
    assert eng.evaluate("chmod", "chmod -R 777 /").action == "deny"

    # allow -- simulation tools
    assert eng.evaluate("vasp", "vasp").action == "allow"
    assert eng.evaluate("python", "python script.py").action == "allow"
    assert eng.evaluate("lmp", "lmp -in in.relax").action == "allow"
    assert eng.evaluate("mpiexec", "mpiexec -n 4 vasp_std").action == "allow"
    assert eng.evaluate("ls", "ls -la").action == "allow"

    # ask -- needs confirmation
    assert eng.evaluate("sbatch", "sbatch job.sh").action == "ask"
    assert eng.evaluate("curl", "curl http://example.com").action == "ask"
    assert eng.evaluate("pip", "pip install numpy").action == "ask"

    # unknown -- default to ask
    assert eng.evaluate("foobar", "foobar --bogus").action == "ask"

    # hook
    assert evaluate_command_hook(["rm", "-rf", "/"]).action == "deny"
    assert evaluate_command_hook(["python", "script.py"]).action == "allow"
    assert evaluate_command_hook(["sbatch", "job.sh"]).action == "ask"

    # exe path normalization
    assert evaluate_command_hook(["/usr/bin/python", "script.py"]).action == "allow"

    # deny takes priority over allow (python -c 'rm -rf /')
    d = evaluate_command_hook(["python", "-c", "rm", "-rf", "/"])
    assert d.action == "deny", f"Expected deny, got {d.action} ({d.matched_rule})"

    print(f"Loaded {len(eng.rules)} rules, default={eng.default_action}")
    print("All self-checks passed.")
