// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

struct AppState {
    backend: Mutex<Option<Child>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            backend: Mutex::new(None),
        }
    }
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
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            get_agent_status,
            start_backend,
            stop_backend
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::ExitRequested { .. } = event {
            let state = app_handle.state::<AppState>();
            let mut child = state.backend.lock().unwrap();
            if let Some(mut c) = child.take() {
                let _ = c.kill();
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
