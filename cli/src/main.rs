mod config;
mod python;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use config::HuginnConfig;
use dialoguer::{theme::ColorfulTheme, Input};
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Huginn: Material Science specialized AI Agent Harness.
#[derive(Parser, Debug)]
#[command(name = "huginn")]
#[command(about = "Material Science specialized AI Agent Harness")]
#[command(version)]
struct Cli {
    /// Workspace directory
    #[arg(short, long, global = true, default_value = ".")]
    workspace: PathBuf,

    /// Config file path
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Model name (e.g., claude-sonnet-4-6, gpt-5.4)
    #[arg(short, long, global = true)]
    model: Option<String>,

    /// Provider (anthropic, openai, ollama, deepseek, ...)
    #[arg(short, long, global = true)]
    provider: Option<String>,

    /// Show what would be executed without running commands
    #[arg(long, global = true)]
    dry_run: bool,

    /// Base URL for OpenAI-compatible endpoints (vLLM, LM Studio, etc.)
    #[arg(short = 'u', long, global = true)]
    base_url: Option<String>,

    /// Ollama base URL
    #[arg(long, global = true, default_value = "http://localhost:11434")]
    ollama_url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start interactive chat with the Agent
    Chat,

    /// Enter exploration mode to systematically search a design space
    Explore {
        /// Exploration objective
        objective: String,

        /// Exploration strategy
        #[arg(short, long, default_value = "pareto")]
        strategy: String,

        /// Maximum parallel branches
        #[arg(short = 'b', long, default_value = "10")]
        max_branches: i64,
    },

    /// Start the HTTP/WebSocket server for the desktop app
    Serve {
        /// Server port
        #[arg(short, long, default_value = "8000")]
        port: u16,

        /// Server host
        #[arg(short = 'H', long, default_value = "127.0.0.1")]
        host: String,
    },

    /// List all available tools
    Tools,

    /// Show version information
    Version,

    /// Interactive first-run configuration wizard
    Configure {
        /// Config file path to write
        #[arg(short, long, default_value = "huginn.toml")]
        path: PathBuf,
    },
}

fn main() -> ExitCode {
    if let Err(e) = run() {
        eprintln!("{} {e:#}", "error:".red().bold());
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run() -> Result<()> {
    // Load .env from the current directory if present.
    if let Err(err) = dotenvy::dotenv() {
        // Only warn if a .env file exists but could not be loaded.
        // Io errors here mean no .env file was found, which is fine.
        if !matches!(err, dotenvy::Error::Io(_)) {
            eprintln!("{} failed to load .env: {err}", "warning:".yellow().bold());
        }
    }

    let cli = Cli::parse();
    let workspace = cli
        .workspace
        .canonicalize()
        .unwrap_or_else(|_| cli.workspace.clone());

    match cli.command {
        Commands::Version => cmd_version(),
        Commands::Configure { path } => cmd_configure(&path),
        Commands::Tools => cmd_tools(),
        Commands::Chat => {
            let globals = collect_global_args(&cli, &workspace);
            delegate_to_python(&workspace, "chat", &globals, &[])
        }
        Commands::Explore {
            ref objective,
            ref strategy,
            max_branches,
        } => {
            let globals = collect_global_args(&cli, &workspace);
            delegate_to_python(
                &workspace,
                "explore",
                &globals,
                &[
                    objective.clone(),
                    "--strategy".to_string(),
                    strategy.clone(),
                    "--max-branches".to_string(),
                    max_branches.to_string(),
                ],
            )
        }
        Commands::Serve { port, ref host } => {
            let globals = collect_global_args(&cli, &workspace);
            delegate_to_python(
                &workspace,
                "serve",
                &globals,
                &[
                    "--port".to_string(),
                    port.to_string(),
                    "--host".to_string(),
                    host.clone(),
                ],
            )
        }
    }
}

/// Print version information, optionally querying Python for backend versions.
fn cmd_version() -> Result<()> {
    println!(
        "{} {}",
        "Huginn".bold().blue(),
        env!("CARGO_PKG_VERSION").bold()
    );

    match python::run_python_expression(
        "import importlib, sys; \
         pkgs = ['langchain', 'langgraph', 'pydantic']; \
         [print(f'{p}: {importlib.import_module(p).__version__}') for p in pkgs if p in sys.modules or importlib.util.find_spec(p)]"
    ) {
        Ok(output) if !output.is_empty() => {
            for line in output.lines() {
                println!("  {line}");
            }
        }
        _ => {}
    }

    Ok(())
}

/// List all available tools, querying the Python backend for metadata.
fn cmd_tools() -> Result<()> {
    let tools = python::list_tools()?;

    println!(
        "{} {}",
        "Available Tools".bold().blue(),
        format!("({})", tools.len()).dimmed()
    );
    println!();

    for (name, description, read_only) in tools {
        let desc = if description.len() > 60 {
            format!("{}...", &description[..60])
        } else {
            description
        };
        let ro = if read_only {
            " read-only".green()
        } else {
            "".normal()
        };
        println!("  {} — {}{}", name.bold(), desc, ro);
    }

    Ok(())
}

/// Interactive configuration wizard.
fn cmd_configure(path: &Path) -> Result<()> {
    println!(
        "{}",
        " Huginn Configuration Wizard "
            .on_blue()
            .white()
            .bold()
    );

    let existing = HuginnConfig::load(path).unwrap_or_default();

    let provider: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Provider")
        .default(existing.provider.clone())
        .interact_text()
        .context("Failed to read provider")?;

    let model: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Model")
        .default(existing.model.clone().unwrap_or_else(|| "auto".to_string()))
        .interact_text()
        .context("Failed to read model")?;

    let api_key: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("API key")
        .default(existing.api_key.clone().unwrap_or_default())
        .allow_empty(true)
        .interact_text()
        .context("Failed to read API key")?;

    let base_url: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Base URL")
        .default(existing.base_url.clone().unwrap_or_default())
        .allow_empty(true)
        .interact_text()
        .context("Failed to read base URL")?;

    let ollama_host: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Ollama host")
        .default(existing.ollama_host.clone())
        .interact_text()
        .context("Failed to read Ollama host")?;

    let workspace: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Workspace")
        .default(existing.workspace.clone())
        .interact_text()
        .context("Failed to read workspace")?;

    let new_cfg = HuginnConfig {
        provider,
        model: if model == "auto" { None } else { Some(model) },
        api_key: if api_key.is_empty() {
            None
        } else {
            Some(api_key)
        },
        base_url: if base_url.is_empty() {
            None
        } else {
            Some(base_url)
        },
        ollama_host,
        workspace,
        ..existing
    };

    new_cfg.save(path)?;
    println!(
        "{} Config saved to {}",
        "✓".green().bold(),
        path.display().to_string().bold()
    );
    println!(
        "{} Run: huginn chat --config {}",
        "→".dimmed(),
        path.display()
    );

    Ok(())
}

/// Collect global CLI options into arguments that the Python backend understands.
///
/// Only commands that use `@click.pass_context` in the Python CLI accept these
/// global flags (chat, explore, serve).
fn collect_global_args(cli: &Cli, workspace: &Path) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();

    args.push("--workspace".to_string());
    args.push(workspace.display().to_string());

    if let Some(config) = &cli.config {
        args.push("--config".to_string());
        args.push(config.display().to_string());
    } else if let Ok(cfg) = env::current_dir().map(|d| d.join("huginn.toml")) {
        // Pass default config if it exists so Python picks it up explicitly.
        if cfg.exists() {
            args.push("--config".to_string());
            args.push(cfg.display().to_string());
        }
    }

    if let Some(model) = &cli.model {
        args.push("--model".to_string());
        args.push(model.clone());
    }

    if let Some(provider) = &cli.provider {
        args.push("--provider".to_string());
        args.push(provider.clone());
    }

    if cli.dry_run {
        args.push("--dry-run".to_string());
    }

    if let Some(base_url) = &cli.base_url {
        args.push("--base-url".to_string());
        args.push(base_url.clone());
    }

    if cli.ollama_url != "http://localhost:11434" {
        args.push("--ollama-url".to_string());
        args.push(cli.ollama_url.clone());
    }

    args
}

/// Delegate a command to the Python backend via subprocess.
///
/// The Rust CLI resolves config and passes global options through so the
/// Python CLI receives the same effective arguments.
fn delegate_to_python(
    workspace: &Path,
    subcommand: &str,
    globals: &[String],
    extra: &[String],
) -> Result<()> {
    let status = python::run_python_cli(workspace, subcommand, globals, extra)
        .with_context(|| format!("Failed to run `huginn {subcommand}`"))?;

    if !status.success() {
        anyhow::bail!("Python backend exited with status: {status}");
    }

    Ok(())
}
