//! CLI → huginn.server HTTP 客户端 (ADR-0001)。
//!
//! CLI 是网关的客户端，而不是第二前门：能从正在运行的后端 `/v1/*` 读到数据时
//! 就走 HTTP，不再 spawn Python 子进程去直接 import 业务模块。后端起在
//! 非标准端口时，通过后端自己写下的 `backend_port` 文件(单一事实源)发现端口，
//! 与 desktop 侧 `get_backend_port` 逻辑保持一致。

use anyhow::{Context, Result};
use serde_json::Value;
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