// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{Emitter, Manager};

struct AppState {
    backend: Mutex<Option<Child>>,
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
            greet,
            get_agent_status,
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
            if let Some(mut c) = backend.take() {
                let _ = c.kill();
            }
            let mut terminal = state.terminal.lock().unwrap();
            if let Some(mut t) = terminal.take() {
                let _ = t.child.kill();
            }
        }
    });
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! Welcome to MatSci-Agent.", name)
}

#[tauri::command]
fn get_agent_status() -> serde_json::Value {
    match reqwest::blocking::get("http://localhost:8000/health") {
        Ok(resp) if resp.status().is_success() => {
            resp.json().unwrap_or_else(|_| serde_json::json!({"status": "ok"}))
        }
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
fn start_backend(state: tauri::State<'_, AppState>) -> Result<String, String> {
    // If we already manage a backend process, assume it's started.
    {
        let lock = state.backend.lock().unwrap();
        if lock.is_some() {
            return Ok("already started".to_string());
        }
    }

    // If a backend is already running externally, don't start another one.
    if let Ok(resp) = reqwest::blocking::get("http://localhost:8000/health") {
        if resp.status().is_success() {
            return Ok("already running externally".to_string());
        }
    }

    let child = Command::new("python")
        .args(["-m", "matsci_agent.server"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to start backend: {}", e))?;

    *state.backend.lock().unwrap() = Some(child);
    Ok("started".to_string())
}

#[tauri::command]
fn stop_backend(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let mut lock = state.backend.lock().unwrap();
    if let Some(mut child) = lock.take() {
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

    let mut child = Command::new("cmd")
        .args(["/Q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to start terminal: {}", e))?;

    let stdin = child.stdin.take().ok_or("no stdin")?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let stderr = child.stderr.take().ok_or("no stderr")?;

    *state.terminal.lock().unwrap() = Some(TerminalSession { child, stdin });

    let app_stdout = app.clone();
    std::thread::spawn(move || read_stream(stdout, app_stdout, "stdout"));
    let app_stderr = app.clone();
    std::thread::spawn(move || read_stream(stderr, app_stderr, "stderr"));

    Ok(())
}

fn read_stream<R: Read + Send + 'static>(mut stream: R, app: tauri::AppHandle, source: &'static str) {
    let mut buf = [0u8; 1024];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let _ = app.emit(
                    "terminal-output",
                    serde_json::json!({"source": source, "text": text}),
                );
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
