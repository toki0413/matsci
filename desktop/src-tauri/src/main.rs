// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let window = app.get_webview_window("main").unwrap();
                window.open_devtools();
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet, get_agent_status])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
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
