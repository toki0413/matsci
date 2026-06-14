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

/// MatSci-Agent sidecar: manages the Python backend and broadcasts events.
#[derive(Parser, Debug)]
#[command(name = "matsci-sidecar")]
#[command(about = "Process manager and event bus for MatSci-Agent")]
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
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let (events, _) = broadcast::channel::<Event>(256);
    let state = Arc::new(SidecarState {
        backend_port: args.backend_port,
        child: Mutex::new(None),
        events: events.clone(),
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
    "MatSci-Agent sidecar"
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

    // If a backend is already running externally, just adopt it.
    if backend_health(state.backend_port).await.is_ok() {
        let _ = state.events.send(Event::Status {
            message: "backend already running externally".to_string(),
        });
        return Ok(());
    }

    let mut cmd = tokio::process::Command::new("python");
    cmd.args([
        "-m",
        "matsci_agent.server",
        "--port",
        &state.backend_port.to_string(),
    ])
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

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

async fn stop_backend_inner(state: &SidecarState) -> Result<(), String> {
    let mut lock = state.child.lock().await;
    if let Some(mut child) = lock.take() {
        child
            .start_kill()
            .map_err(|e| format!("failed to start killing backend: {}", e))?;
        let _ = child.wait().await;
        let _ = state.events.send(Event::Status {
            message: "backend stopped".to_string(),
        });
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
