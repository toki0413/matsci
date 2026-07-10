// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{Emitter, Manager};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

struct AppState {
    backend: Mutex<Option<CommandChild>>,
    terminal: Mutex<Option<TerminalSession>>,
}

struct TerminalSession {
    child: Child,
    stdin: ChildStdin,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            backend: Mutex::new(None),
            terminal: Mutex::new(None),
        }
    }
}

#[derive(Serialize)]
struct FileEntry {
    name: String,
    path: String,
    is_dir: bool,
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let window = app.get_webview_window("main").unwrap();
                window.open_devtools();
            }
            // Start integrated terminal in the background
            let _ = spawn_terminal(app.handle().clone(), app.state::<AppState>());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_agent_status,
            get_backend_port,
            start_backend,
            stop_backend,
            get_cwd,
            read_dir,
            read_file,
            write_file,
            write_terminal,
            stop_terminal
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::ExitRequested { .. } = event {
            let state = app_handle.state::<AppState>();
            let mut backend = state.backend.lock().unwrap();
            if let Some(child) = backend.take() {
                let _ = child.kill();
            }
            let mut terminal = state.terminal.lock().unwrap();
            if let Some(mut t) = terminal.take() {
                let _ = t.child.kill();
            }
        }
    });
}

#[tauri::command]
fn get_agent_status() -> serde_json::Value {
    match reqwest::blocking::get("http://localhost:8000/health") {
        Ok(resp) if resp.status().is_success() => resp
            .json()
            .unwrap_or_else(|_| serde_json::json!({"status": "ok"})),
        Ok(resp) => serde_json::json!({
            "status": "error",
            "error": format!("HTTP {}", resp.status()),
        }),
        Err(e) => serde_json::json!({
            "status": "offline",
            "error": e.to_string(),
        }),
    }
}

#[tauri::command]
fn get_backend_port() -> u16 {
    8000
}

#[tauri::command]
async fn start_backend(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    // If we already manage a backend process, assume it's started.
    {
        let lock = state.backend.lock().unwrap();
        if lock.is_some() {
            return Ok("already started".to_string());
        }
    }

    // If a backend is already running externally, don't start another one.
    // Use TCP connect instead of reqwest::blocking to avoid tokio runtime panic.
    if std::net::TcpStream::connect("127.0.0.1:8000").is_ok() {
        return Ok("already running externally".to_string());
    }

    // Always use dev mode — the sidecar binary has frozen old code that
    // lacks our recent fixes (config loading, thread_id, reasoning, etc.).
    // System Python from PATH runs the latest source directly.
    let is_dev = true;

    // Deterministic config file path under AppData.
    // Without this, the backend writes huginn.toml to the CWD, which is
    // unpredictable and leads to config corruption.
    let local_app = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| {
        std::env::var("USERPROFILE").unwrap_or_else(|_| ".".to_string())
    });
    let config_dir = std::path::PathBuf::from(&local_app).join("Huginn");
    let _ = std::fs::create_dir_all(&config_dir);
    let config_file = std::env::var("HUGINN_CONFIG_FILE")
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|_| config_dir.join("huginn.toml").to_string_lossy().to_string());

    // ── Production: try sidecars first ──────────────────────────────
    if !is_dev {
        // Desktop app runs locally — always bypass auth so Pet window
        // and main window can connect without token dance.
        let dev_mode = std::env::var("HUGINN_DEV_MODE").unwrap_or_else(|_| "1".to_string());
        let deepseek_key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_default();
        let provider = std::env::var("HUGINN_PROVIDER").unwrap_or_default();
        let model = std::env::var("HUGINN_MODEL").unwrap_or_default();
        let config_file = std::env::var("HUGINN_CONFIG_FILE").unwrap_or_default();

        if let Ok(sidecar) = app.shell().sidecar("huginn-sidecar") {
            eprintln!("[start_backend] Found huginn-sidecar, spawning...");
            let (mut rx, child) = sidecar
                .env("HUGINN_DEV_MODE", &dev_mode)
                .env("DEEPSEEK_API_KEY", &deepseek_key)
                .env("HUGINN_PROVIDER", &provider)
                .env("HUGINN_MODEL", &model)
                .env("HUGINN_CONFIG_FILE", &config_file)
                .env("PYTHONUNBUFFERED", "1")
                .spawn()
                .map_err(|e| format!("failed to spawn huginn-sidecar: {}", e))?;

            let app_stdout = app.clone();
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    let (source, bytes) = match event {
                        CommandEvent::Stdout(b) => ("stdout", b),
                        CommandEvent::Stderr(b) => ("stderr", b),
                        _ => continue,
                    };
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    let _ = app_stdout.emit(
                        "backend-log",
                        serde_json::json!({"source": source, "text": text}),
                    );
                }
            });

            *state.backend.lock().unwrap() = Some(child);
            return Ok("started".to_string());
        }

        if let Ok(sidecar) = app.shell().sidecar("huginn") {
            let (mut rx, child) = sidecar
                .args(["serve", "--port", "8000"])
                .env("HUGINN_DEV_MODE", &dev_mode)
                .env("DEEPSEEK_API_KEY", &deepseek_key)
                .env("HUGINN_PROVIDER", &provider)
                .env("HUGINN_MODEL", &model)
                .env("HUGINN_CONFIG_FILE", &config_file)
                .env("PYTHONUNBUFFERED", "1")
                .spawn()
                .map_err(|e| format!("failed to spawn huginn sidecar: {}", e))?;

            let app_stdout = app.clone();
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    let (source, bytes) = match event {
                        CommandEvent::Stdout(b) => ("stdout", b),
                        CommandEvent::Stderr(b) => ("stderr", b),
                        _ => continue,
                    };
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    let _ = app_stdout.emit(
                        "backend-log",
                        serde_json::json!({"source": source, "text": text}),
                    );
                }
            });

            *state.backend.lock().unwrap() = Some(child);
            return Ok("started (legacy sidecar)".to_string());
        }
    }

    // ── Dev / standalone fallback: direct Python ────────────────────
    eprintln!("[start_backend] Using Python fallback (dev={})", is_dev);

    // Use system Python from PATH. The installed python-runtime directory
    // only has site-packages, not the interpreter itself.
    let python_exe = "python".to_string();
    eprintln!("[start_backend] Python: {}", python_exe);

    // Point PYTHONPATH at the agent source so huginn.server is importable.
    // The workspace structure is: matsci-agent/agent/huginn/...
    let agent_src = std::path::PathBuf::from(&local_app)
        .join("..")
        .join("Desktop")
        .join("matsci-agent")
        .join("agent");
    let pythonpath = if agent_src.exists() {
        agent_src.to_string_lossy().to_string()
    } else {
        std::env::var("PYTHONPATH").unwrap_or_default()
    };

    let python_cmd = app.shell().command(&python_exe);
    let (mut rx, child) = python_cmd
        .args(["-m", "huginn.server", "--port", "8000"])
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONPATH", &pythonpath)
        .env("DEEPSEEK_API_KEY", std::env::var("DEEPSEEK_API_KEY").unwrap_or_default())
        .env("HUGINN_PROVIDER", std::env::var("HUGINN_PROVIDER").unwrap_or_default())
        .env("HUGINN_MODEL", std::env::var("HUGINN_MODEL").unwrap_or_default())
        .env("HUGINN_DEV_MODE", "1")
        .env("HUGINN_CONFIG_FILE", &config_file)
        .spawn()
        .map_err(|e| format!("failed to start backend via python: {}", e))?;

    let app_stdout = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            let (source, bytes) = match event {
                CommandEvent::Stdout(b) => ("stdout", b),
                CommandEvent::Stderr(b) => ("stderr", b),
                _ => continue,
            };
            let text = String::from_utf8_lossy(&bytes).to_string();
            let _ = app_stdout.emit(
                "backend-log",
                serde_json::json!({"source": source, "text": text}),
            );
        }
    });

    *state.backend.lock().unwrap() = Some(child);
    Ok("started (python)".to_string())
}

#[tauri::command]
async fn stop_backend(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let mut lock = state.backend.lock().unwrap();
    if let Some(child) = lock.take() {
        child.kill().map_err(|e| e.to_string())?;
        Ok("stopped".to_string())
    } else {
        Ok("not running".to_string())
    }
}

#[tauri::command]
fn get_cwd() -> Result<String, String> {
    std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn read_dir(path: &str) -> Result<Vec<FileEntry>, String> {
    let base = PathBuf::from(path);
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(&base).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().to_string();
        let path = entry.path().to_string_lossy().to_string();
        let is_dir = entry.file_type().map_err(|e| e.to_string())?.is_dir();
        entries.push(FileEntry { name, path, is_dir });
    }
    entries.sort_by(|a, b| match (a.is_dir, b.is_dir) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
    });
    Ok(entries)
}

#[tauri::command]
fn read_file(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| e.to_string())
}

#[tauri::command]
fn write_file(path: &str, content: &str) -> Result<(), String> {
    std::fs::write(path, content).map_err(|e| e.to_string())
}

fn spawn_terminal(app: tauri::AppHandle, state: tauri::State<'_, AppState>) -> Result<(), String> {
    {
        let lock = state.terminal.lock().unwrap();
        if lock.is_some() {
            return Ok(());
        }
    }

    let mut cmd = Command::new("cmd");
    cmd.args(["/Q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Hide the console window — we pipe stdio to the frontend, the visible
    // cmd popup is just Windows defaulting to a new console for the child.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to start terminal: {}", e))?;

    let stdin = child.stdin.take().ok_or("no stdin")?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let stderr = child.stderr.take().ok_or("no stderr")?;

    *state.terminal.lock().unwrap() = Some(TerminalSession { child, stdin });

    let app_stdout = app.clone();
    std::thread::spawn(move || read_stream(stdout, app_stdout, "stdout", "terminal-output"));
    let app_stderr = app.clone();
    std::thread::spawn(move || read_stream(stderr, app_stderr, "stderr", "terminal-output"));

    Ok(())
}

fn read_stream<R: Read + Send + 'static>(
    mut stream: R,
    app: tauri::AppHandle,
    source: &'static str,
    event: &'static str,
) {
    let mut buf = [0u8; 1024];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let _ = app.emit(event, serde_json::json!({"source": source, "text": text}));
            }
            Err(_) => break,
        }
    }
}

#[tauri::command]
fn write_terminal(state: tauri::State<'_, AppState>, text: &str) -> Result<(), String> {
    let mut lock = state.terminal.lock().unwrap();
    if let Some(session) = lock.as_mut() {
        session
            .stdin
            .write_all(text.as_bytes())
            .map_err(|e| e.to_string())?;
        session.stdin.flush().map_err(|e| e.to_string())?;
        Ok(())
    } else {
        Err("terminal not started".to_string())
    }
}

#[tauri::command]
fn stop_terminal(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut lock = state.terminal.lock().unwrap();
    if let Some(mut session) = lock.take() {
        session.child.kill().map_err(|e| e.to_string())?;
    }
    Ok(())
}
