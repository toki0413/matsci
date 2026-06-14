use anyhow::{Context, Result};
use process_wrap::std::*;
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Find a usable Python interpreter.
///
/// Priority:
/// 1. `MATSCI_PYTHON` environment variable
/// 2. `python` on PATH
/// 3. `python3` on PATH
pub fn find_python() -> Result<PathBuf> {
    if let Ok(py) = env::var("MATSCI_PYTHON") {
        let py = PathBuf::from(py);
        if py.exists() {
            return Ok(py);
        }
    }

    for cmd in ["python", "python3"] {
        if let Ok(path) = which::which(cmd) {
            return Ok(path);
        }
    }

    anyhow::bail!("No Python interpreter found on PATH. Set MATSCI_PYTHON or install Python.")
}

/// Run a Python CLI subcommand, inheriting stdin/stdout/stderr.
///
/// This delegates to:
///   `python -m matsci_agent.cli [global_args...] <subcommand> [subcommand_args...]`
///
/// Global options must come before the subcommand because Click only accepts
/// them on the group, not on individual subcommands.
/// The current working directory is set to `workspace` so relative paths and
/// `.env` files resolve as expected.
pub fn run_python_cli(
    workspace: &Path,
    subcommand: &str,
    global_args: &[String],
    subcommand_args: &[String],
) -> Result<std::process::ExitStatus> {
    let python = find_python()?;

    let mut cmd = Command::new(&python);
    cmd.arg("-m")
        .arg("matsci_agent.cli")
        .args(global_args)
        .arg(subcommand)
        .args(subcommand_args)
        .current_dir(workspace)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let mut wrap = CommandWrap::from(cmd);
    #[cfg(windows)]
    {
        wrap.wrap(JobObject);
    }
    #[cfg(unix)]
    {
        wrap.wrap(ProcessGroup::leader());
    }

    let mut child = wrap
        .spawn()
        .with_context(|| format!("Failed to spawn Python CLI via {}", python.display()))?;

    let status = child
        .wait()
        .with_context(|| format!("Failed to wait on Python CLI via {}", python.display()))?;

    Ok(status)
}

/// Run a one-shot Python expression and return its stdout as a string.
pub fn run_python_expression(expression: &str) -> Result<String> {
    let python = find_python()?;
    let output = Command::new(&python)
        .arg("-c")
        .arg(expression)
        .output()
        .with_context(|| format!("Failed to run Python expression via {}", python.display()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Python expression failed: {stderr}");
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Query the Python backend for registered tool metadata (name, description, read_only).
pub fn list_tools() -> Result<Vec<(String, String, bool)>> {
    let expr = r#"
import json
try:
    from matsci_agent.cli import _register_all_tools
    from matsci_agent.tools.registry import ToolRegistry
    _register_all_tools()
    tools = []
    for name in ToolRegistry.list_tools():
        tool = ToolRegistry.get(name)
        if tool is None:
            continue
        read_only = False
        if hasattr(tool, "is_read_only"):
            try:
                read_only = tool.is_read_only({})
            except Exception:
                read_only = False
        tools.append({"name": name, "description": tool.description, "read_only": bool(read_only)})
    print(json.dumps(tools))
except Exception as e:
    print(json.dumps({"error": str(e)}))
"#;

    let output = run_python_expression(expr)?;
    let parsed: serde_json::Value = serde_json::from_str(&output)
        .with_context(|| format!("Failed to parse tool list JSON: {output}"))?;

    if let Some(err) = parsed.get("error") {
        anyhow::bail!("Python backend failed to list tools: {err}");
    }

    let tools: Vec<serde_json::Value> =
        serde_json::from_value(parsed).context("Tool list JSON is not an array")?;

    let mut result = Vec::new();
    for tool in tools {
        let name = tool["name"].as_str().unwrap_or("unknown").to_string();
        let description = tool["description"].as_str().unwrap_or("").to_string();
        let read_only = tool["read_only"].as_bool().unwrap_or(false);
        result.push((name, description, read_only));
    }

    Ok(result)
}

// Minimal `which` reimplementation to avoid an extra dependency.
mod which {
    use std::env;
    use std::path::PathBuf;

    pub fn which(cmd: &str) -> Result<PathBuf, ()> {
        let path_var = env::var_os("PATH").ok_or(())?;
        for dir in env::split_paths(&path_var) {
            let candidate = dir.join(cmd);
            #[cfg(windows)]
            let candidate = add_exe_if_needed(candidate);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        Err(())
    }

    #[cfg(windows)]
    fn add_exe_if_needed(path: PathBuf) -> PathBuf {
        if path.extension().is_none() {
            path.with_extension("exe")
        } else {
            path
        }
    }
}
