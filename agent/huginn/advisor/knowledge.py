"""Knowledge base for model advisor recommendations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ModelProfile:
    """Profile of an LLM model's capabilities."""
    name: str
    provider: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    context_window: int = 4096
    cost_per_1k_tokens: float = 0.0
    best_for: list[str] = field(default_factory=list)  # task tags
    available: bool = True
    requires_api_key: bool = True
    notes: str = ""

# Built-in model knowledge
_MODEL_PROFILES: list[ModelProfile] = [
    ModelProfile(
        name="gpt-4o", provider="openai",
        strengths=["Reasoning", "Code generation", "Scientific analysis"],
        weaknesses=["Cost", "Rate limits"],
        context_window=128000, cost_per_1k_tokens=0.005,
        best_for=["reasoning", "coding", "science", "analysis"],
    ),
    ModelProfile(
        name="gpt-4o-mini", provider="openai",
        strengths=["Fast", "Cost-effective", "Good for simple tasks"],
        weaknesses=["Complex reasoning", "Long context"],
        context_window=128000, cost_per_1k_tokens=0.00015,
        best_for=["simple", "fast_response", "formatting"],
    ),
    ModelProfile(
        name="claude-3.5-sonnet", provider="anthropic",
        strengths=["Long context", "Analysis", "Writing", "Code"],
        weaknesses=["Cost"],
        context_window=200000, cost_per_1k_tokens=0.003,
        best_for=["analysis", "coding", "writing", "long_context"],
    ),
    ModelProfile(
        name="deepseek-v3", provider="deepseek",
        strengths=["Coding", "Math", "Cost-effective"],
        weaknesses=["English nuance", "Availability"],
        context_window=64000, cost_per_1k_tokens=0.00014,
        best_for=["coding", "math", "reasoning"],
    ),
    ModelProfile(
        name="qwen-2.5-72b", provider="alibaba",
        strengths=["Chinese", "Science", "Cost-effective"],
        weaknesses=["English creative writing"],
        context_window=32000, cost_per_1k_tokens=0.0003,
        best_for=["science", "chinese", "analysis"],
    ),
    ModelProfile(
        name="llama-3.1-70b", provider="meta",
        strengths=["Open source", "Local deployment", "Privacy"],
        weaknesses=["Requires GPU", "Context window"],
        context_window=128000, cost_per_1k_tokens=0.0,
        best_for=["local", "privacy", "coding"],
        requires_api_key=False,
    ),
    ModelProfile(
        name="ollama-local", provider="local",
        strengths=["Privacy", "No cost", "Offline"],
        weaknesses=["Quality", "Speed", "Requires GPU"],
        context_window=8192, cost_per_1k_tokens=0.0,
        best_for=["privacy", "local", "simple"],
        requires_api_key=False,
        notes="Use Ollama to run models locally",
    ),
]

def get_profiles() -> list[ModelProfile]:
    return list(_MODEL_PROFILES)

def find_by_task(task_tag: str) -> list[ModelProfile]:
    return [m for m in _MODEL_PROFILES if task_tag in m.best_for and m.available]

def find_by_provider(provider: str) -> list[ModelProfile]:
    return [m for m in _MODEL_PROFILES if m.provider == provider]
