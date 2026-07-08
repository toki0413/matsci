import { useState } from 'react';
import { Sparkles } from 'lucide-react';
import { api } from '../../lib/api';
import type { SkillInfo } from '../../types/domain';

// ── Skill form (local to this panel) ──

function SkillForm({
  params,
  value,
  onChange,
}: {
  params: SkillInfo["parameters"];
  value: Record<string, any>;
  onChange: (v: Record<string, any>) => void;
}) {
  const update = (key: string, val: any) => onChange({ ...value, [key]: val });
  return (
    <div className="space-y-4">
      {params.map((p) => {
        const label = (
          <span className="text-xs font-medium text-text-secondary">
            {p.name}
            {p.required && <span className="ml-1 text-error">*</span>}
            <span className="ml-2 font-mono text-[10px] text-text-muted">{p.type}</span>
          </span>
        );
        const desc = p.description ? (
          <p className="mt-1 text-xs text-text-muted">{p.description}</p>
        ) : null;
        let input: React.ReactNode;
        if (p.type === "boolean") {
          input = (
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={!!value[p.name]}
                onChange={(e) => update(p.name, e.target.checked)}
                className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
              />
              <span className="text-sm text-text-primary">{value[p.name] ? "true" : "false"}</span>
            </label>
          );
        } else if (p.type === "number" || p.type === "integer") {
          input = (
            <input
              type="number"
              value={value[p.name] ?? ""}
              onChange={(e) => update(p.name, e.target.value === "" ? "" : Number(e.target.value))}
              className="input font-mono text-sm"
            />
          );
        } else if (p.type === "array" || p.type === "object") {
          input = (
            <textarea
              value={typeof value[p.name] === "string" ? value[p.name] : JSON.stringify(value[p.name] ?? "", null, 2)}
              onChange={(e) => {
                try { update(p.name, JSON.parse(e.target.value)); }
                catch { update(p.name, e.target.value); }
              }}
              rows={3}
              className="input font-mono text-xs"
            />
          );
        } else if (p.enum && p.enum.length > 0) {
          input = (
            <select
              value={value[p.name] ?? ""}
              onChange={(e) => update(p.name, e.target.value)}
              className="input text-sm"
            >
              {p.enum.map((opt: string) => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
          );
        } else {
          input = (
            <input
              type="text"
              value={value[p.name] ?? ""}
              onChange={(e) => update(p.name, e.target.value)}
              className="input text-sm"
            />
          );
        }
        return (
          <div key={p.name}>
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

interface SkillsPanelProps {
  skills: SkillInfo[];
  isConnected: boolean;
}

export function SkillsPanel({ skills, isConnected }: SkillsPanelProps) {
  const [selectedSkill, setSelectedSkill] = useState<SkillInfo | null>(null);
  const [skillArgs, setSkillArgs] = useState<Record<string, any>>({});
  const [skillResult, setSkillResult] = useState<string>("");
  const [skillLoading, setSkillLoading] = useState(false);

  const runSkill = async () => {
    if (!selectedSkill) return;
    setSkillLoading(true);
    setSkillResult("");
    try {
      const data = await api.post("/skills/execute", { skill: selectedSkill.name, args: skillArgs });
      setSkillResult(JSON.stringify(data, null, 2));
    } catch (e: any) {
      setSkillResult(`Error: ${e.message}`);
    } finally {
      setSkillLoading(false);
    }
  };

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Declarative Skills</h2>
        {selectedSkill && (
          <button onClick={() => setSelectedSkill(null)} className="btn-secondary text-xs">
            ← Back
          </button>
        )}
      </div>

      {!selectedSkill ? (
        skills.length === 0 ? (
          !isConnected ? (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <Sparkles size={40} className="text-text-muted opacity-40" />
              <p className="mt-4 text-sm font-medium text-text-secondary">Backend not connected</p>
              <p className="mt-1 max-w-xs text-xs text-text-muted">
                Skills are loaded from the AI backend. Make sure the server is running and reconnect to see available skills.
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent border-t-transparent" />
              <p className="mt-4 text-sm text-text-muted">Loading skills…</p>
            </div>
          )
        ) : (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {skills.map((skill) => (
            <div key={skill.name} className="card flex flex-col">
              <div className="text-xs font-semibold uppercase text-accent">{skill.category}</div>
              <div className="mt-1 text-sm font-semibold">{skill.name}</div>
              <div className="mt-1 flex-1 text-xs text-text-secondary line-clamp-3">
                {skill.description}
              </div>
              <div className="mt-3 text-xs text-text-muted">
                Tags: {skill.tags.join(", ")}
              </div>
              <button
                onClick={() => {
                  setSelectedSkill(skill);
                  const defaults: Record<string, any> = {};
                  skill.parameters.forEach((p) => {
                    if (p.default !== undefined && p.default !== null)
                      defaults[p.name] = p.default;
                    else defaults[p.name] = p.type === "boolean" ? false : p.type === "number" || p.type === "integer" ? 0 : "";
                  });
                  setSkillArgs(defaults);
                  setSkillResult("");
                }}
                className="btn-secondary mt-3 w-full text-xs"
              >
                Execute
              </button>
            </div>
          ))}
        </div>
        )
      ) : (
        <div className="max-w-3xl space-y-4">
          <div className="card">
            <div className="text-xs uppercase text-accent font-semibold">Skill</div>
            <h3 className="mt-1 text-base font-semibold">{selectedSkill.name}</h3>
            <p className="mt-1 text-sm text-text-secondary">
              {selectedSkill.description}
            </p>
            <div className="mt-2 text-xs text-text-muted">
              Tags: {selectedSkill.tags.join(", ")}
            </div>
          </div>
          <div className="card">
            <div className="mb-3 flex items-center justify-between">
              <label className="text-xs font-medium text-text-secondary">Arguments</label>
              <button
                onClick={() => {
                  const defaults: Record<string, any> = {};
                  selectedSkill.parameters.forEach((p) => {
                    defaults[p.name] =
                      p.default !== undefined && p.default !== null
                        ? p.default
                        : p.type === "boolean"
                        ? false
                        : p.type === "number" || p.type === "integer"
                        ? 0
                        : "";
                  });
                  setSkillArgs(defaults);
                }}
                className="text-xs text-accent hover:underline"
              >
                Reset defaults
              </button>
            </div>
            <SkillForm
              params={selectedSkill.parameters}
              value={skillArgs}
              onChange={setSkillArgs}
            />
          </div>
          <button onClick={runSkill} disabled={skillLoading} className="btn-primary">
            {skillLoading ? "Running…" : "Run Skill"}
          </button>
          {skillResult && (
            <div className="card border-accent/20 bg-bg-secondary">
              <div className="mb-2 text-xs font-semibold text-accent">Result</div>
              <pre className="max-h-96 overflow-auto rounded-lg bg-bg-tertiary p-3 text-xs">
                {skillResult}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
