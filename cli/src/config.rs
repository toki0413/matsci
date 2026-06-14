use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::path::Path;

/// MatSci-Agent configuration, matching the Python `MatSciConfig` schema.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MatSciConfig {
    pub provider: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub ollama_host: String,
    pub vasp_executable: Option<String>,
    pub lammps_executable: Option<String>,
    pub hpc_scheduler: String,
    pub hpc_host: Option<String>,
    pub hpc_username: Option<String>,
    pub workspace: String,
    pub auto_approve: bool,
    pub enable_exploration: bool,
    pub max_parallel_branches: i64,
    pub encryption_enabled: bool,
    pub encryption_password: Option<String>,
    pub encryption_key_file: Option<String>,
    pub encrypt_rag_documents: bool,
    pub encrypt_rag_metadata: bool,
    pub encrypt_config: bool,
}

#[allow(dead_code)]
impl MatSciConfig {
    /// Default configuration.
    pub fn new() -> Self {
        Self {
            provider: "default".to_string(),
            ollama_host: "http://localhost:11434".to_string(),
            hpc_scheduler: "local".to_string(),
            workspace: ".".to_string(),
            enable_exploration: true,
            max_parallel_branches: 5,
            encrypt_rag_documents: true,
            encrypt_rag_metadata: true,
            ..Default::default()
        }
    }

    /// Load configuration from environment variables.
    pub fn from_env() -> Self {
        let provider = env::var("MATSCI_PROVIDER")
            .unwrap_or_else(|_| "default".to_string())
            .to_lowercase()
            .trim()
            .to_string();

        let mut api_key = env::var("MATSCI_API_KEY").ok();
        if api_key.is_none() {
            let key_var = provider_key_map().get(provider.as_str()).copied();
            if let Some(var) = key_var {
                api_key = env::var(var).ok();
            }
        }

        Self {
            provider,
            model: env::var("MATSCI_MODEL").ok(),
            api_key,
            base_url: env::var("MATSCI_BASE_URL").ok(),
            ollama_host: env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            vasp_executable: env::var("VASP_EXECUTABLE").ok(),
            lammps_executable: env::var("LAMMPS_EXECUTABLE").ok(),
            hpc_scheduler: env::var("HPC_SCHEDULER").unwrap_or_else(|_| "local".to_string()),
            hpc_host: env::var("HPC_HOST").ok(),
            hpc_username: env::var("HPC_USERNAME").ok(),
            workspace: env::var("MATSCI_WORKSPACE").unwrap_or_else(|_| ".".to_string()),
            auto_approve: env::var("MATSCI_AUTO_APPROVE")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(false),
            enable_exploration: env::var("MATSCI_ENABLE_EXPLORATION")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(true),
            max_parallel_branches: env::var("MATSCI_MAX_BRANCHES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5),
            encryption_enabled: env::var("MATSCI_ENCRYPTION_ENABLED")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(false),
            encryption_password: env::var("MATSCI_ENCRYPTION_PASSWORD").ok(),
            encryption_key_file: env::var("MATSCI_ENCRYPTION_KEY_FILE").ok(),
            encrypt_rag_documents: env::var("MATSCI_ENCRYPT_RAG_DOCS")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(true),
            encrypt_rag_metadata: env::var("MATSCI_ENCRYPT_RAG_META")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(true),
            encrypt_config: env::var("MATSCI_ENCRYPT_CONFIG")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(false),
        }
    }

    /// Load configuration from a TOML or JSON file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let cfg: MatSciConfig = if ext == "json" {
            serde_json::from_str(&text)
                .with_context(|| format!("Failed to parse JSON config: {}", path.display()))?
        } else {
            toml::from_str(&text)
                .with_context(|| format!("Failed to parse TOML config: {}", path.display()))?
        };
        Ok(cfg)
    }

    /// Save configuration to a TOML or JSON file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create config directory: {}", parent.display())
            })?;
        }

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let text = if ext == "json" {
            serde_json::to_string_pretty(self).context("Failed to serialize config to JSON")?
        } else {
            toml::to_string_pretty(self).context("Failed to serialize config to TOML")?
        };

        std::fs::write(path, text)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;
        Ok(())
    }

    /// Resolve a credential string supporting `env:` and `keyring:` prefixes.
    pub fn resolve_key(raw: Option<&str>) -> Result<Option<String>> {
        let raw = match raw {
            Some(r) => r,
            None => return Ok(None),
        };
        let raw = raw.trim();
        if raw.starts_with("env:") {
            let var_name = &raw[4..];
            Ok(env::var(var_name).ok())
        } else if raw.starts_with("keyring:") {
            anyhow::bail!(
                "keyring: prefix is not supported in the Rust CLI; use env:VAR_NAME or plain text"
            )
        } else {
            Ok(Some(raw.to_string()))
        }
    }

    /// Return a copy with the API key resolved (env:/keyring: prefixes handled).
    pub fn with_resolved_key(&self) -> Result<Self> {
        let mut cfg = self.clone();
        cfg.api_key = Self::resolve_key(cfg.api_key.as_deref())?;
        Ok(cfg)
    }
}

#[allow(dead_code)]
fn provider_key_map() -> HashMap<&'static str, &'static str> {
    [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("google-genai", "GOOGLE_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("nvidia", "NVIDIA_API_KEY"),
        ("vllm", "OPENAI_API_KEY"),
        ("local", "OPENAI_API_KEY"),
    ]
    .into_iter()
    .collect()
}
