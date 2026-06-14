//! Lightweight sandboxed subprocess runner.
//!
//! Validates the command, working directory, environment, and arguments before
//! spawning a child process. Enforces a timeout and returns stdout/stderr.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Result as IoResult};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use wait_timeout::ChildExt;

const DEFAULT_TIMEOUT_SECS: f64 = 60.0;
const MAX_TIMEOUT_SECS: f64 = 600.0;

/// Binaries that may be invoked by basename.
const ALLOWED_BINARIES: &[&str] = &[
    "bash",
    "cat",
    "cargo",
    "cp",
    "code",
    "code.cmd",
    "echo",
    "find",
    "git",
    "git.cmd",
    "ls",
    "huginn",
    "huginn.exe",
    "mkdir",
    "mv",
    "python",
    "python3",
    "python.exe",
    "pytest",
    "pytest.exe",
    "rm",
    "rg",
    "rustc",
    "rustc.exe",
    "sh",
    "tar",
    "unzip",
    "zip",
];

/// Environment variables that may be set/overridden by the caller.
const ALLOWED_ENV_VARS: &[&str] = &[
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "HOME",
    "HUGINN_*",
    "OPENAI_API_KEY",
    "PATH",
    "PATHEXT",
    "PYTHONPATH",
    "RUST_BACKTRACE",
    "RUST_LOG",
    "TEMP",
    "TMP",
    "USERPROFILE",
];

/// Characters/sequences that are forbidden inside individual arguments.
const FORBIDDEN_ARG_CHARS: &[char] = &[';', '|', '&', '$', '`', '<', '>', '\n', '\r'];

fn default_allowed_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        if let Ok(c) = cwd.canonicalize() {
            dirs.push(c.clone());
            if let Some(parent) = c.parent() {
                dirs.push(parent.to_path_buf());
            }
        }
    }
    dirs
}

fn binary_allowlist() -> HashSet<String> {
    ALLOWED_BINARIES.iter().map(|s| s.to_string()).collect()
}

fn strip_exe_suffix(name: &str) -> &str {
    name.strip_suffix(".exe").unwrap_or(name)
}

fn is_allowed_env_key(key: &str) -> bool {
    ALLOWED_ENV_VARS.iter().any(|pat| {
        if pat.ends_with('*') {
            key.starts_with(&pat[..pat.len() - 1])
        } else {
            key == *pat
        }
    })
}

fn validate_args(args: &[String]) -> PyResult<()> {
    for arg in args {
        if arg.is_empty() {
            continue;
        }
        if arg.chars().any(|c| FORBIDDEN_ARG_CHARS.contains(&c)) {
            return Err(PyValueError::new_err(format!(
                "Argument contains forbidden shell metacharacters: {arg}"
            )));
        }
        if arg.contains("$(") || arg.contains("`") {
            return Err(PyValueError::new_err(format!(
                "Argument contains forbidden command substitution: {arg}"
            )));
        }
    }
    Ok(())
}

fn resolve_command(command: &str, allowed_dirs: &[PathBuf]) -> PyResult<PathBuf> {
    let path = Path::new(command);

    // Absolute paths must lie inside an allowed directory and exist.
    if path.is_absolute() {
        let canon = path.canonicalize().map_err(|e| {
            PyValueError::new_err(format!("Cannot resolve command path {command}: {e}"))
        })?;
        if !allowed_dirs.iter().any(|base| canon.starts_with(base)) {
            return Err(PyValueError::new_err(format!(
                "Command path outside allowed directories: {command}"
            )));
        }
        return Ok(canon);
    }

    let basename = path
        .file_name()
        .ok_or_else(|| PyValueError::new_err(format!("Invalid command: {command}")))?
        .to_string_lossy();
    let lookup = strip_exe_suffix(&basename);

    if !binary_allowlist().contains(lookup) {
        return Err(PyValueError::new_err(format!(
            "Command not in sandbox allowlist: {command}"
        )));
    }

    // Search PATH for the executable.
    let name = basename.to_string();
    let exe = which::which(&name)
        .or_else(|_| which::which(lookup))
        .map_err(|_| PyValueError::new_err(format!("Executable not found in PATH: {command}")))?;
    Ok(exe)
}

fn validate_cwd(cwd: &str, allowed_dirs: &[PathBuf]) -> PyResult<PathBuf> {
    let path = Path::new(cwd)
        .canonicalize()
        .map_err(|e| PyValueError::new_err(format!("Invalid working directory {cwd}: {e}")))?;
    if !allowed_dirs.iter().any(|base| path.starts_with(base)) {
        return Err(PyValueError::new_err(format!(
            "Working directory outside allowed directories: {cwd}"
        )));
    }
    Ok(path)
}

fn validate_env(env: &HashMap<String, String>) -> PyResult<()> {
    for key in env.keys() {
        if !is_allowed_env_key(key) {
            return Err(PyValueError::new_err(format!(
                "Environment variable not in sandbox allowlist: {key}"
            )));
        }
    }
    Ok(())
}

fn read_pipe<R: Read + Send + 'static>(pipe: R) -> thread::JoinHandle<IoResult<Vec<u8>>> {
    thread::spawn(move || {
        let mut buf = Vec::new();
        let mut reader = pipe;
        reader.read_to_end(&mut buf)?;
        Ok(buf)
    })
}

/// Run a subprocess inside the lightweight sandbox.
///
/// Parameters
/// ----------
/// command : str
///     Binary name or absolute path.
/// args : list[str]
///     Arguments (shell metacharacters are rejected).
/// cwd : str | None
///     Working directory. Must be inside an allowed base directory.
/// env : dict[str, str] | None
///     Extra environment variables to set. Keys are allowlisted.
/// timeout : float | None
///     Timeout in seconds (default 60, max 600).
/// allowed_base_dirs : list[str] | None
///     Directories the command and cwd must be under. Defaults to cwd and its parent.
#[pyfunction]
#[pyo3(signature = (command, args=None, cwd=None, env=None, timeout=None, allowed_base_dirs=None))]
fn run_sandboxed(
    py: Python,
    command: String,
    args: Option<Vec<String>>,
    cwd: Option<String>,
    env: Option<HashMap<String, String>>,
    timeout: Option<f64>,
    allowed_base_dirs: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let args = args.unwrap_or_default();

    let allowed_dirs: Vec<PathBuf> = match allowed_base_dirs {
        Some(dirs) => dirs
            .into_iter()
            .map(|d| {
                Path::new(&d).canonicalize().map_err(|e| {
                    PyValueError::new_err(format!("Invalid allowed_base_dir {d}: {e}"))
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
        None => default_allowed_dirs(),
    };
    if allowed_dirs.is_empty() {
        return Err(PyRuntimeError::new_err(
            "Sandbox could not determine allowed base directories",
        ));
    }

    validate_args(&args)?;
    if let Some(ref e) = env {
        validate_env(e)?;
    }

    let exe = resolve_command(&command, &allowed_dirs)?;
    let work_dir = match cwd {
        Some(d) => validate_cwd(&d, &allowed_dirs)?,
        None => std::env::current_dir().map_err(|e| {
            PyRuntimeError::new_err(format!("Could not determine current directory: {e}"))
        })?,
    };

    let effective_timeout = timeout
        .map(|t| t.clamp(1.0, MAX_TIMEOUT_SECS))
        .unwrap_or(DEFAULT_TIMEOUT_SECS);

    let mut cmd = Command::new(&exe);
    cmd.args(&args)
        .current_dir(&work_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(ref env_map) = env {
        for (k, v) in env_map {
            cmd.env(k, v);
        }
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to spawn sandboxed process: {e}")))?;

    let stdout_handle = read_pipe(child.stdout.take().unwrap());
    let stderr_handle = read_pipe(child.stderr.take().unwrap());

    let duration = Duration::from_secs_f64(effective_timeout);
    let status = child
        .wait_timeout(duration)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to wait for process: {e}")))?;

    let timed_out = status.is_none();
    let returncode = if let Some(s) = status {
        s.code().unwrap_or(-1)
    } else {
        let _ = child.kill();
        let _ = child.wait();
        -1
    };

    let stdout = stdout_handle
        .join()
        .unwrap_or_else(|_| Ok(Vec::new()))
        .unwrap_or_default();
    let stderr = stderr_handle
        .join()
        .unwrap_or_else(|_| Ok(Vec::new()))
        .unwrap_or_default();

    let dict = PyDict::new(py);
    dict.set_item("command", format!("{} {}", exe.display(), args.join(" ")))?;
    dict.set_item("returncode", returncode)?;
    dict.set_item("timed_out", timed_out)?;
    dict.set_item("success", !timed_out && returncode == 0)?;
    dict.set_item("stdout", String::from_utf8_lossy(&stdout).into_owned())?;
    dict.set_item("stderr", String::from_utf8_lossy(&stderr).into_owned())?;
    dict.set_item(
        "message",
        if timed_out {
            format!("Command timed out after {effective_timeout}s")
        } else if returncode == 0 {
            "Command succeeded.".to_string()
        } else {
            format!("Command failed with exit code {returncode}")
        },
    )?;

    Ok(dict.into())
}

pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "sandbox")?;
    m.add_function(wrap_pyfunction!(run_sandboxed, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
