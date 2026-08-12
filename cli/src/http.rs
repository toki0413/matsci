//! CLI → huginn.server HTTP 客户端 (ADR-0001)。
//!
//! CLI 是网关的客户端，而不是第二前门：能从正在运行的后端 `/v1/*` 读到数据时
//! 就走 HTTP，不再 spawn Python 子进程去直接 import 业务模块。后端起在
//! 非标准端口时，通过后端自己写下的 `backend_port` 文件(单一事实源)发现端口，
//! 与 desktop 侧 `get_backend_port` 逻辑保持一致。

use anyhow::{Context, Result};
use serde_json::Value;
use std::io::BufRead;
use std::path::PathBuf;

/// 与后端 huginn.utils.runtime 一致的运行时目录。
/// 优先级: $HUGINN_CACHE_DIR, 否则 ~/.huginn。
fn runtime_home() -> PathBuf {
    if let Some(dir) = std::env::var_os("HUGINN_CACHE_DIR") {
        let p = PathBuf::from(dir);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    let mut home = PathBuf::from(
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .unwrap_or_default(),
    );
    home.push(".huginn");
    home
}

/// 读后端实际绑定的端口。后端启动时把它写进 `backend_port` 文件。
fn read_backend_port_file() -> Option<u16> {
    let port_file = runtime_home().join("backend_port");
    let text = std::fs::read_to_string(&port_file).ok()?;
    text.trim().parse::<u16>().ok()
}

/// 推断后端 base URL：优先端口文件，兜底 8000。
fn base_url() -> String {
    let port = read_backend_port_file().unwrap_or(8000);
    format!("http://127.0.0.1:{port}")
}

/// 后端是否可达（打到 /health）。
pub fn backend_available() -> bool {
    let url = format!("{}/health", base_url());
    match ureq::get(&url).timeout(std::time::Duration::from_millis(500)).call() {
        Ok(resp) => resp.status() >= 200 && resp.status() < 300,
        Err(_) => false,
    }
}

/// 通过后端 `/v1/tools` 拉取工具元数据 `(name, description, read_only)`。
///
/// 与 python::list_tools 的返回值对齐，便于 `cmd_tools` 复用同一渲染逻辑。
pub fn list_tools_via_http() -> Result<Vec<(String, String, bool)>> {
    let url = format!("{}/v1/tools", base_url());
    let body = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .call()
        .with_context(|| format!("连接后端失败: {url}"))?
        .into_string()
        .context("读取后端 /v1/tools 响应失败")?;

    let tools: Vec<serde_json::Value> =
        serde_json::from_str(&body).context("/v1/tools 返回的不是 JSON 数组")?;

    let mut result = Vec::new();
    for tool in tools {
        let name = tool["function"]["name"].as_str().unwrap_or("unknown").to_string();
        let description = tool["function"]["description"].as_str().unwrap_or("").to_string();
        let read_only = tool["read_only"].as_bool().unwrap_or(false);
        result.push((name, description, read_only));
    }
    result.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}

/// 通用 POST JSON 到后端 `/v1...` 端点，返回响应 JSON。
fn post_json(path: &str, payload: Value, timeout_secs: u64) -> Result<Value> {
    let url = format!("{}{}", base_url(), path);
    let body = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .send_json(payload)
        .with_context(|| format!("连接后端失败: {url}"))?
        .into_string()
        .with_context(|| format!("读取后端 {path} 响应失败"))?;
    serde_json::from_str(&body).with_context(|| format!("后端 {path} 返回的不是 JSON: {body}"))
}

/// 诊断计算错误 → POST /execution/diagnose。
pub fn diagnose_via_http(
    error_message: &str,
    software: Option<&str>,
    calculation_type: Option<&str>,
    context: Option<&str>,
) -> Result<Value> {
    let mut payload = serde_json::Map::new();
    payload.insert("error_message".into(), Value::String(error_message.to_string()));
    if let Some(s) = software {
        payload.insert("software".into(), Value::String(s.to_string()));
    }
    if let Some(t) = calculation_type {
        payload.insert("calculation_type".into(), Value::String(t.to_string()));
    }
    if let Some(c) = context {
        payload.insert("context".into(), Value::String(c.to_string()));
    }
    post_json("/v1/diagnose", Value::Object(payload), 60)
}

/// 运行 benchmark → POST /bench/run。
pub fn bench_via_http(evolve: bool, categories: Option<&str>) -> Result<Value> {
    let mut payload = serde_json::Map::new();
    payload.insert("evolve".into(), Value::Bool(evolve));
    if let Some(c) = categories {
        payload.insert("categories".into(), Value::String(c.to_string()));
    }
    post_json("/v1/bench/run", Value::Object(payload), 600)
}

/// 运行自演化 → POST /evolve/run。
pub fn evolve_via_http(logs_dir: Option<&str>) -> Result<Value> {
    let mut payload = serde_json::Map::new();
    if let Some(d) = logs_dir {
        payload.insert("logs_dir".into(), Value::String(d.to_string()));
    }
    post_json("/v1/evolve/run", Value::Object(payload), 600)
}

/// 运行 workflow 模板 → POST /workflows/execute。
pub fn workflow_via_http(template: &str, args: &[String]) -> Result<Value> {
    // CLI 的 workflow 传 KEY=VALUE 参数，组装成 dict。
    let mut arg_map = serde_json::Map::new();
    for kv in args {
        if let Some((k, v)) = kv.split_once('=') {
            arg_map.insert(k.trim().to_string(), Value::String(v.trim().to_string()));
        }
    }
    let payload = serde_json::json!({ "template": template, "args": arg_map });
    post_json("/v1/workflows/execute", payload, 600)
}

// ── HPC ──────────────────────────────────────────────────────────
// HPC 端点接受内联 host/username/key_path 等参数，与 CLI 的 hpc test/submit/status
// 子命令一一对应。端点挂了 require_admin_key；本地 dev mode(loopback) 下豁免。

fn _hpc_payload(
    host: &str,
    username: &str,
    scheduler: &str,
    key_path: Option<&str>,
    port: Option<i64>,
) -> serde_json::Map<String, Value> {
    let mut p = serde_json::Map::new();
    p.insert("host".into(), Value::String(host.to_string()));
    p.insert("username".into(), Value::String(username.to_string()));
    p.insert("scheduler".into(), Value::String(scheduler.to_string()));
    if let Some(k) = key_path {
        p.insert("key_path".into(), Value::String(k.to_string()));
    }
    if let Some(port) = port {
        p.insert("port".into(), Value::Number(port.into()));
    }
    p
}

/// 测试 HPC SSH 连接 → POST /hpc/test。
pub fn hpc_test_via_http(
    host: &str,
    username: &str,
    scheduler: &str,
    key_path: Option<&str>,
    port: i64,
) -> Result<Value> {
    let p = _hpc_payload(host, username, scheduler, key_path, Some(port));
    post_json("/v1/hpc/test", Value::Object(p), 30)
}

/// 提交 HPC 作业 → POST /hpc/submit。
#[allow(clippy::too_many_arguments)]
pub fn hpc_submit_via_http(
    host: &str,
    username: &str,
    command: &str,
    job_name: &str,
    walltime: &str,
    nodes: i64,
    ntasks_per_node: i64,
    queue: Option<&str>,
    scheduler: &str,
    key_path: Option<&str>,
    remote_work_dir: &str,
) -> Result<Value> {
    let mut p = _hpc_payload(host, username, scheduler, key_path, None);
    p.insert("command".into(), Value::String(command.to_string()));
    p.insert("job_name".into(), Value::String(job_name.to_string()));
    p.insert("walltime".into(), Value::String(walltime.to_string()));
    p.insert("nodes".into(), Value::Number(nodes.into()));
    p.insert("ntasks_per_node".into(), Value::Number(ntasks_per_node.into()));
    p.insert("remote_work_dir".into(), Value::String(remote_work_dir.to_string()));
    if let Some(q) = queue {
        p.insert("queue".into(), Value::String(q.to_string()));
    }
    post_json("/v1/hpc/submit", Value::Object(p), 60)
}

/// 查询 HPC 作业状态 → POST /hpc/status。
pub fn hpc_status_via_http(
    host: &str,
    username: &str,
    job_id: &str,
    scheduler: &str,
    key_path: Option<&str>,
) -> Result<Value> {
    let mut p = _hpc_payload(host, username, scheduler, key_path, None);
    p.insert("job_id".into(), Value::String(job_id.to_string()));
    post_json("/v1/hpc/status", Value::Object(p), 30)
}

// ── execute / explore / coder / encrypt-config ──────────────────
// 这几个端点跟 CLI 子命令一一对应，后端在跑就直接 POST，不再 spawn。

/// 运行 workflow 阶段 → POST /execute。`stages` 是解析好的 stage 数组。
pub fn execute_via_http(stages: Value, working_dir: &str, name: &str) -> Result<Value> {
    let payload = serde_json::json!({
        "stages": stages,
        "working_dir": working_dir,
        "name": name,
    });
    post_json("/v1/execute", payload, 600)
}

/// 运行设计空间探索 → POST /explore。
pub fn explore_via_http(
    objective: &str,
    strategy: &str,
    max_branches: i64,
    max_iterations: i64,
) -> Result<Value> {
    let payload = serde_json::json!({
        "objective": objective,
        "strategy": strategy,
        "max_branches": max_branches,
        "max_iterations": max_iterations,
    });
    post_json("/v1/explore", payload, 600)
}

/// 运行自主编码会话 → POST /coder。
pub fn coder_via_http(
    task: &str,
    auto_approve: bool,
    max_iterations: Option<i64>,
) -> Result<Value> {
    let mut payload = serde_json::Map::new();
    payload.insert("task".into(), Value::String(task.to_string()));
    payload.insert("auto_approve".into(), Value::Bool(auto_approve));
    if let Some(it) = max_iterations {
        payload.insert("max_iterations".into(), Value::Number(it.into()));
    }
    post_json("/v1/coder", Value::Object(payload), 600)
}

/// 启用配置加密 → POST /config/encrypt。加密的是后端的活跃配置。
pub fn encrypt_config_via_http(password: &str) -> Result<Value> {
    let payload = serde_json::json!({ "password": password });
    post_json("/v1/config/encrypt", payload, 30)
}

// ── 交互式 chat: SSE 流式 ──────────────────────────────────────
// 连 /agents/lead/chat/stream，逐事件打印 token / 工具调用 / 思考，
// 命中 done 就结束。默认打 lead agent（与后端其它入口一致）。

/// 发一条消息走 SSE 流式 chat，把事件实时打印到终端。
fn chat_stream_via_http(message: &str, thread_id: &str) -> Result<()> {
    let url = format!("{}/v1/agents/lead/chat/stream", base_url());
    let payload = serde_json::json!({ "message": message, "thread_id": thread_id });
    let resp = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(300))
        .send_json(payload)
        .with_context(|| format!("连接后端失败: {url}"))?;

    let mut reader = std::io::BufReader::new(resp.into_reader());
    let mut line = String::new();
    let mut event_type = String::new();
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .with_context(|| format!("读取 SSE 流失败: {url}"))?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if let Some(et) = trimmed.strip_prefix("event:") {
            event_type = et.trim().to_string();
        } else if let Some(d) = trimmed.strip_prefix("data:") {
            let data: Value = serde_json::from_str(d.trim()).unwrap_or(Value::Null);
            let etype = data["type"].as_str().unwrap_or(&event_type);
            match etype {
                "token" => {
                    if let Some(t) = data["text"].as_str() {
                        print!("{t}");
                        use std::io::Write;
                        let _ = std::io::stdout().flush();
                    }
                }
                "thought" => {
                    if let Some(t) = data["text"].as_str() {
                        eprintln!("[thinking] {t}");
                    }
                }
                "plan" => {
                    if let Some(t) = data["text"].as_str() {
                        eprintln!("[plan] {t}");
                    }
                }
                "tool_start" => {
                    if let Some(t) = data["tool"].as_str() {
                        eprintln!("→ {t}");
                    }
                }
                "tool_end" => {
                    if let Some(t) = data["tool"].as_str() {
                        eprintln!("✓ {t}");
                    }
                }
                "done" => {
                    println!();
                    return Ok(());
                }
                "error" => {
                    let msg = data["message"].as_str().unwrap_or("unknown error");
                    anyhow::bail!("后端 chat 错误: {msg}");
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// 交互式 chat REPL：读 stdin → SSE 流式 → 打印回复。
pub fn chat_repl() -> Result<()> {
    use std::io::Write;
    let stdin = std::io::stdin();
    let thread_id = "default";
    loop {
        print!("You: ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        let n = stdin.read_line(&mut line)?;
        if n == 0 {
            return Ok(()); // EOF
        }
        let input = line.trim().to_string();
        if input.is_empty() {
            continue;
        }
        if matches!(input.as_str(), "exit" | "quit" | "q") {
            return Ok(());
        }
        chat_stream_via_http(&input, thread_id)?;
    }
}