import { useState } from 'react';
import { Wrench } from 'lucide-react';
import { api } from '../../lib/api';
import type { ToolInfo } from '../../types/domain';

// ── Schema helpers (local to this panel) ──

function defaultForSchema(prop: any): any {
  if (prop && "default" in prop) return prop.default;
  const type = prop?.type;
  if (type === "boolean") return false;
  if (type === "integer" || type === "number") return 0;
  if (type === "array") return [];
  if (type === "object") return buildDefaultArgs(prop);
  return "";
}

function buildDefaultArgs(schema: any): Record<string, any> {
  if (!schema || schema.type !== "object") return {};
  const out: Record<string, any> = {};
  for (const [key, prop] of Object.entries(schema.properties || {})) {
    out[key] = defaultForSchema(prop);
  }
  return out;
}

interface JsonSchemaProperty {
  type?: string;
  description?: string;
  enum?: string[];
}

interface JsonSchema {
  type?: string;
  required?: string[];
  properties?: Record<string, JsonSchemaProperty>;
}

function JsonSchemaForm({
  schema,
  value,
  onChange,
}: {
  schema: JsonSchema;
  value: Record<string, any>;
  onChange: (v: Record<string, any>) => void;
}) {
  if (!schema || schema.type !== "object") return null;
  const required = new Set(schema.required || []);
  const update = (key: string, val: any) => {
    onChange({ ...value, [key]: val });
  };
  return (
    <div className="space-y-4">
      {Object.entries(schema.properties ?? {}).map(([key, prop]) => {
        const label = (
          <span className="text-xs font-medium text-text-secondary">
            {key}
            {required.has(key) && <span className="ml-1 text-error">*</span>}
          </span>
        );
        const desc = prop.description ? (
          <p className="mt-1 text-xs text-text-muted">{prop.description}</p>
        ) : null;
        let input: React.ReactNode;
        if (prop.type === "boolean") {
          input = (
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={!!value[key]}
                onChange={(e) => update(key, e.target.checked)}
                className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
              />
              <span className="text-sm text-text-primary">{value[key] ? "true" : "false"}</span>
            </label>
          );
        } else if (prop.type === "integer" || prop.type === "number") {
          input = (
            <input
              type="number"
              value={value[key] ?? ""}
              onChange={(e) => update(key, e.target.value === "" ? "" : Number(e.target.value))}
              className="input font-mono text-sm"
            />
          );
        } else if (Array.isArray(prop.enum) && prop.enum.length > 0) {
          input = (
            <select
              value={value[key] ?? ""}
              onChange={(e) => update(key, e.target.value)}
              className="input text-sm"
            >
              {prop.enum.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          );
        } else {
          input = (
            <input
              type="text"
              value={value[key] ?? ""}
              onChange={(e) => update(key, e.target.value)}
              className="input font-mono text-sm"
            />
          );
        }
        return (
          <div key={key}>
            {label}
            <div className="mt-1.5">{input}</div>
            {desc}
          </div>
        );
      })}
    </div>
  );
}

// ── Panel ──

interface ToolsPanelProps {
  tools: ToolInfo[];
  isConnected: boolean;
}

export function ToolsPanel({ tools, isConnected }: ToolsPanelProps) {
  const [selectedTool, setSelectedTool] = useState<ToolInfo | null>(null);
  const [toolArgs, setToolArgs] = useState<Record<string, any>>({});
  const [toolResult, setToolResult] = useState<string>("");
  const [toolLoading, setToolLoading] = useState(false);

  const runTool = async () => {
    if (!selectedTool) return;
    if (selectedTool.destructive) {
      const ok = window.confirm(
        `⚠️ ${selectedTool.function.name} may overwrite files or run shell commands. Run it anyway?`
      );
      if (!ok) return;
    }
    setToolLoading(true);
    setToolResult("");
    try {
      const name = selectedTool.function.name;
      const data = await api.post(`/tools/${name}`, toolArgs);
      setToolResult(JSON.stringify(data, null, 2));
    } catch (e: any) {
      setToolResult(`Error: ${e.message}`);
    } finally {
      setToolLoading(false);
    }
  };

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Available Tools</h2>
        {selectedTool && (
          <button onClick={() => setSelectedTool(null)} className="btn-secondary text-xs">
            ← Back
          </button>
        )}
      </div>

      {!selectedTool ? (
        tools.length === 0 ? (
          !isConnected ? (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <Wrench size={40} className="text-text-muted opacity-40" />
              <p className="mt-4 text-sm font-medium text-text-secondary">Backend not connected</p>
              <p className="mt-1 max-w-xs text-xs text-text-muted">
                Tools are loaded from the AI backend. Make sure the server is running and reconnect to see available tools.
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent border-t-transparent" />
              <p className="mt-4 text-sm text-text-muted">Loading tools…</p>
            </div>
          )
        ) : (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {tools.map((tool) => (
            <button
              key={tool.function.name}
              onClick={() => {
                setSelectedTool(tool);
                setToolArgs(buildDefaultArgs(tool.function.parameters));
                setToolResult("");
              }}
              className="card text-left transition-colors hover:border-accent"
            >
              <div className="flex items-center justify-between">
                <div className="text-xs font-semibold uppercase text-accent">Tool</div>
                {tool.destructive && (
                  <span className="rounded bg-error/10 px-1.5 py-0.5 text-[10px] text-error">
                    destructive
                  </span>
                )}
              </div>
              <div className="mt-1 text-sm font-semibold">{tool.function.name}</div>
              <div className="mt-1 text-xs text-text-secondary line-clamp-2">
                {tool.function.description}
              </div>
            </button>
          ))}
        </div>
        )
      ) : (
        <div className="max-w-3xl space-y-4">
          <div className="card">
            <div className="flex items-center justify-between">
              <div className="text-xs uppercase text-accent font-semibold">Tool</div>
              {selectedTool.destructive && (
                <span className="rounded bg-error/10 px-1.5 py-0.5 text-[10px] text-error">
                  destructive
                </span>
              )}
            </div>
            <h3 className="mt-1 text-base font-semibold">{selectedTool.function.name}</h3>
            <p className="mt-1 text-sm text-text-secondary">
              {selectedTool.function.description}
            </p>
          </div>
          <div className="card">
            <div className="mb-3 flex items-center justify-between">
              <label className="text-xs font-medium text-text-secondary">Arguments</label>
              <button
                onClick={() =>
                  setToolArgs(buildDefaultArgs(selectedTool.function.parameters))
                }
                className="text-xs text-accent hover:underline"
              >
                Reset defaults
              </button>
            </div>
            <JsonSchemaForm
              schema={selectedTool.function.parameters}
              value={toolArgs}
              onChange={setToolArgs}
            />
          </div>
          <button onClick={runTool} disabled={toolLoading} className="btn-primary">
            {toolLoading
              ? "Running…"
              : selectedTool.destructive
              ? "⚠️ Run Tool"
              : "Run Tool"}
          </button>
          {toolResult && (
            <div className="card border-accent/20 bg-bg-secondary">
              <div className="mb-2 text-xs font-semibold text-accent">Result</div>
              <pre className="max-h-96 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">
                {toolResult}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
