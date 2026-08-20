use axum::{
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use clap::Parser;
use futures::{sink::SinkExt, stream::StreamExt};
use process_wrap::tokio::*;
use serde::Serialize;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{broadcast, Mutex};

/// Huginn sidecar: manages the Python backend and broadcasts events.
#[derive(Parser, Debug)]
#[command(name = "huginn-sidecar")]
#[command(about = "Process manager and event bus for Huginn")]
struct Args {
    /// Port for the sidecar HTTP/WebSocket server
    #[arg(short, long, default_value = "8001")]
    port: u16,

    /// Port the Python backend should listen on
    #[arg(short, long, default_value = "8000")]
    backend_port: u16,

    /// Start the backend automatically when the sidecar starts
    #[arg(long, default_value = "true")]
    autostart: bool,
}

#[derive(Clone, Serialize, Debug)]
#[serde(tag = "type", content = "data")]
enum Event {
    #[serde(rename = "stdout")]
    Stdout(String),
    #[serde(rename = "stderr")]
    Stderr(String),
    #[serde(rename = "status")]
    Status { message: String },
}

#[derive(Clone, Serialize, Debug)]
struct BackendStatus {
    running: bool,
    backend_reachable: bool,
    backend_health: Option<serde_json::Value>,
}

struct SidecarState {
    backend_port: u16,
    child: Mutex<Option<Box<dyn ChildWrapper + Send>>>,
    events: broadcast::Sender<Event>,
    /// Set by stop_backend_inner so the supervisor knows an exit was intentional.
    stopped: std::sync::atomic::AtomicBool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let (events, _) = broadcast::channel::<Event>(256);
    let state = Arc::new(SidecarState {
        backend_port: args.backend_port,
        child: Mutex::new(None),
        events: events.clone(),
        stopped: std::sync::atomic::AtomicBool::new(false),
    });

    if args.autostart {
        let state_clone = state.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            if let Err(e) = start_backend_inner(&state_clone).await {
                let _ = state_clone.events.send(Event::Status {
                    message: format!("autostart failed: {}", e),
                });
            }
        });
    }

    // Always watch for crashes; survives manual /stop and idle periods.
    tokio::spawn(supervise_backend(state.clone()));

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/start", post(start_backend))
        .route("/stop", post(stop_backend))
        .route("/status", get(status))
        .route("/ws", get(ws_handler))
        .with_state(state);

    let addr = format!("127.0.0.1:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!("[sidecar] listening on http://{}", addr);
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    "Huginn sidecar"
}

async fn health(State(state): State<Arc<SidecarState>>) -> impl IntoResponse {
    let backend_health = backend_health(state.backend_port).await.ok();
    let running = state.child.lock().await.is_some();
    let body = serde_json::json!({
        "status": "ok",
        "backend_reachable": backend_health.is_some(),
        "backend_health": backend_health,
        "backend_managed": running,
    });
    axum::Json(body)
}

async fn status(State(state): State<Arc<SidecarState>>) -> impl IntoResponse {
    let running = state.child.lock().await.is_some();
    let backend_health = backend_health(state.backend_port).await.ok();
    axum::Json(BackendStatus {
        running,
        backend_reachable: backend_health.is_some(),
        backend_health,
    })
}

async fn start_backend(State(state): State<Arc<SidecarState>>) -> impl IntoResponse {
    match start_backend_inner(&state).await {
        Ok(_) => axum::Json(serde_json::json!({"success": true, "message": "started"})),
        Err(e) => axum::Json(serde_json::json!({"success": false, "error": e})),
    }
}

async fn stop_backend(State(state): State<Arc<SidecarState>>) -> impl IntoResponse {
    match stop_backend_inner(&state).await {
        Ok(_) => axum::Json(serde_json::json!({"success": true, "message": "stopped"})),
        Err(e) => axum::Json(serde_json::json!({"success": false, "error": e})),
    }
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<SidecarState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: axum::extract::ws::WebSocket, state: Arc<SidecarState>) {
    let mut rx = state.events.subscribe();
    let (mut sender, mut receiver) = socket.split();

    let send_task = tokio::spawn(async move {
        while let Ok(event) = rx.recv().await {
            let text = serde_json::to_string(&event).unwrap_or_default();
            if sender
                .send(axum::extract::ws::Message::Text(text))
                .await
                .is_err()
            {
                break;
            }
        }
    });

    // Keep the socket open until the client closes it.
    while let Some(Ok(_msg)) = receiver.next().await {}
    send_task.abort();
}

async fn start_backend_inner(state: &SidecarState) -> Result<(), String> {
    // If we already manage a backend, don't start another.
    {
        let lock = state.child.lock().await;
        if lock.is_some() {
            return Ok(());
        }
    }

    // (Re)starting is the opposite of a manual stop.
    state
        .stopped
        .store(false, std::sync::atomic::Ordering::SeqCst);

    // If a backend is already running externally, just adopt it.
    if backend_health(state.backend_port).await.is_ok() {
        let _ = state.events.send(Event::Status {
            message: "backend already running externally".to_string(),
        });
        return Ok(());
    }

    // Prefer the bundled Python runtime. In a Tauri production build, resources
    // are placed in a `resources/` subdirectory next to the app binary. In dev
    // mode they're in the same directory. Check both, then fall back to PATH.
    let python_exe = {
        let sidecar_path = std::env::current_exe()
            .map_err(|e| format!("cannot determine sidecar path: {}", e))?;
        let parent = sidecar_path.parent();
        let candidates: Vec<std::path::PathBuf> = match parent {
            Some(p) => vec![
                p.join("python-runtime").join("python.exe"),
                p.join("resources").join("python-runtime").join("python.exe"),
            ],
            None => vec![],
        };
        candidates
            .into_iter()
            .find(|p| p.exists())
            .unwrap_or_else(|| std::path::PathBuf::from("python"))
    };

    let mut cmd = tokio::process::Command::new(&python_exe);
    cmd.args([
        "-m",
        "huginn.server",
        "--port",
        &state.backend_port.to_string(),
    ])
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .env("PYTHONUNBUFFERED", "1");

    // Run from the user's home dir so the backend doesn't drop files into the
    // install directory (which may be read-only after packaging).
    if let Some(home) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok() {
        cmd.current_dir(home);
    }

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
        .map_err(|e| format!("failed to spawn backend: {}", e))?;

    let stdout = child.stdout().take().ok_or("backend has no stdout")?;
    let stderr = child.stderr().take().ok_or("backend has no stderr")?;

    let events = state.events.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            let _ = events.send(Event::Stdout(line));
        }
    });

    let events = state.events.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            let _ = events.send(Event::Stderr(line));
        }
    });

    *state.child.lock().await = Some(child);

    let _ = state.events.send(Event::Status {
        message: "backend started".to_string(),
    });

    Ok(())
}

// Long-lived supervisor, spawned once from main. Watches whichever child the
// mutex holds, and if it dies in a way that wasn't a manual /stop, brings it
// back with a backoff. Poll-loop rather than a blocking wait so a concurrent
// /stop can still grab the child and kill it.
async fn supervise_backend(state: Arc<SidecarState>) {
    loop {
        let exited_unexpectedly = {
            let mut lock = state.child.lock().await;
            match lock.as_mut() {
                Some(child) => match child.try_wait() {
                    Ok(Some(_status)) => {
                        // Exited; take it out so a restart isn't blocked by the guard.
                        lock.take();
                        true
                    }
                    _ => false,
                },
                None => false, // nothing managed, just idle
            }
        };

        if exited_unexpectedly
            && !state.stopped.load(std::sync::atomic::Ordering::SeqCst)
        {
            let _ = state.events.send(Event::Status {
                message: "backend exited unexpectedly, restarting".to_string(),
            });
            for attempt in 0.. { // ponytail: unbounded retries; upgrade path = cap via const
                tokio::time::sleep(tokio::time::Duration::from_secs(restart_backoff(attempt))).await;
                if state.stopped.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }
                if backend_health(state.backend_port).await.is_ok() {
                    return; // some sibling already brought it up
                }
                if start_backend_inner(&state).await.is_ok() {
                    return; // fresh child back under this same supervisor loop
                }
            }
            return;
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }
}

// Give a crashed backend breathing room instead of hot-spinning it.
// 1s, 2s, 4s, ... capped at 16s.
fn restart_backoff(attempt: u64) -> u64 {
    1u64 << attempt.min(4)
}

async fn stop_backend_inner(state: &SidecarState) -> Result<(), String> {
    state
        .stopped
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let mut lock = state.child.lock().await;
    if let Some(child) = lock.as_mut() {
        child
            .start_kill()
            .map_err(|e| format!("failed to kill backend: {}", e))?;
    }
    Ok(())
}

async fn backend_health(port: u16) -> Result<serde_json::Value, reqwest::Error> {
    reqwest::get(format!("http://127.0.0.1:{}/health", port))
        .await?
        .error_for_status()?
        .json()
        .await
}
