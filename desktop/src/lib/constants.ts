// Shared constants pulled out of App.tsx so extracted components can reuse them
// without reaching back into the main module.

// Provider catalogue — keep in sync with backend /config/providers ordering.
export const PROVIDERS = [
  { id: "anthropic", label: "Anthropic", keyVar: "ANTHROPIC_API_KEY" },
  { id: "openai", label: "OpenAI", keyVar: "OPENAI_API_KEY" },
  { id: "deepseek", label: "DeepSeek", keyVar: "DEEPSEEK_API_KEY" },
  { id: "google-genai", label: "Google GenAI", keyVar: "GOOGLE_API_KEY" },
  { id: "openrouter", label: "OpenRouter", keyVar: "OPENROUTER_API_KEY" },
  { id: "nvidia", label: "NVIDIA", keyVar: "NVIDIA_API_KEY" },
  { id: "ollama", label: "Ollama (local)", keyVar: "" },
  { id: "vllm", label: "vLLM", keyVar: "" },
  { id: "local", label: "Local OpenAI-compatible", keyVar: "" },
  { id: "siliconflow", label: "SiliconFlow", keyVar: "SILICONFLOW_API_KEY" },
  { id: "moonshot", label: "Moonshot (Kimi)", keyVar: "MOONSHOT_API_KEY" },
  { id: "zhipu", label: "Zhipu (GLM)", keyVar: "ZHIPU_API_KEY" },
  { id: "baichuan", label: "Baichuan", keyVar: "BAICHUAN_API_KEY" },
  { id: "dashscope", label: "DashScope (Qwen)", keyVar: "DASHSCOPE_API_KEY" },
  { id: "qianfan", label: "Qianfan (Baidu)", keyVar: "QIANFAN_API_KEY" },
  { id: "doubao", label: "Doubao (ByteDance)", keyVar: "DOUBAO_API_KEY" },
  { id: "hunyuan", label: "Hunyuan (Tencent)", keyVar: "HUNYUAN_API_KEY" },
  { id: "openai-compatible", label: "OpenAI-compatible", keyVar: "" },
  { id: "default", label: "Default", keyVar: "" },
];

// Short HH:MM stamp used on every chat / log message.
export function formatTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
