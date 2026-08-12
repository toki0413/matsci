//! CLI → huginn.server HTTP 客户端 (ADR-0001)。
//!
//! CLI 是网关的客户端，而不是第二前门：能从正在运行的后端 `/v1/*` 读到数据时
//! 就走 HTTP，不再 spawn Python 子进程去直接 import 业务模块。后端起在
//! 非标准端口时，通过后端自己写下的 `backend_port` 文件(单一事实源)发现端口，
//! 与 desktop 侧 `get_backend_port` 逻辑保持一致。

use anyhow::{Context, Result};
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