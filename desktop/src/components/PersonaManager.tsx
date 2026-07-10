/**
 * PersonaManager — master/detail panel for Huginn personas.
 *
 * Layout mirrors AstrBot's PersonaManager: a list on the left, full
 * details on the right, plus a top bar to switch the active persona and
 * a modal to create new ones. All data comes from the /personas REST API.
 */
import { useEffect, useState, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { Plus, Trash2, Zap, Star, Copy } from "lucide-react";
import { api } from "../lib/api";

// Built-in personas can't be deleted; the backend enforces this too,
// but hiding the button keeps the UI honest.
const BUILTIN_PERSONAS = new Set([
  "default", "dft_expert", "md_expert", "reviewer",
  "tutor", "planner", "executor",
]);

interface PersonaListItem {
  name: string;
  system_prompt: string;
  begin_dialogs: unknown[];
  avatar?: string | null;
  description?: string;
  when_to_use?: string[];
}

interface PersonaListResp {
  default: string;
  personas: PersonaListItem[];
}

interface PersonaDetail {
  success?: boolean;
  name: string;
  system_prompt: string;
  begin_dialogs: Array<Record<string, string>>;
  mood_dialogs: Array<Record<string, string>>;
  variables?: Record<string, unknown>;
  avatar?: string | null;
  description?: string;
  when_to_use?: string[];
  error?: string;
}

interface CreateForm {
  name: string;
  description: string;
  system_prompt: string;
  when_to_use: string;
}

const EMPTY_FORM: CreateForm = {
  name: "",
  description: "",
  system_prompt: "",
  when_to_use: "",
};

export default function PersonaManager() {
  const { t } = useTranslation();

  const [personas, setPersonas] = useState<PersonaListItem[]>([]);
  const [defaultName, setDefaultName] = useState("");
  // active = persona bound to the current chat session (via /switch)
  const [activeName, setActiveName] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<PersonaDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");

  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState<CreateForm>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);

  const loadList = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await api.get<PersonaListResp>("/personas");
      setPersonas(data.personas ?? []);
      setDefaultName(data.default ?? "");
      // first render: pick the default (or first) persona to show details for
      if (!selected && (data.personas?.length ?? 0) > 0) {
        setSelected(data.default || data.personas[0].name);
      }
    } catch (e: any) {
      setError(e.message || t("empty.error"));
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadDetail = useCallback(async (name: string) => {
    try {
      const d = await api.get<PersonaDetail>(`/personas/${encodeURIComponent(name)}`);
      if (d.success === false) {
        setError(d.error || t("empty.error"));
        setDetail(null);
        return;
      }
      setDetail(d);
      setError("");
    } catch (e: any) {
      setError(e.message || t("empty.error"));
      setDetail(null);
    }
  }, [t]);

  useEffect(() => {
    loadList();
  }, [loadList]);

  useEffect(() => {
    if (selected) loadDetail(selected);
    else setDetail(null);
  }, [selected, loadDetail]);

  const switchActive = async (name: string) => {
    setError("");
    try {
      await api.post(`/personas/${encodeURIComponent(name)}/switch`);
      setActiveName(name);
    } catch (e: any) {
      setError(e.message || t("empty.error"));
    }
  };

  const setAsDefault = async (name: string) => {
    setError("");
    try {
      const r = await api.patch<{ default?: string }>(`/personas/${encodeURIComponent(name)}/default`);
      if (r.default) {
        setDefaultName(r.default);
        setMsg(`${t("persona.setDefault")}: ${r.default}`);
        setTimeout(() => setMsg(""), 2500);
      }
    } catch (e: any) {
      setError(e.message || t("empty.error"));
    }
  };

  const removePersona = async (name: string) => {
    if (!window.confirm(t("persona.confirmDelete"))) return;
    setError("");
    try {
      await api.del(`/personas/${encodeURIComponent(name)}`);
      // fall back to default if we just deleted the selection
      if (selected === name) setSelected(defaultName || "default");
      await loadList();
    } catch (e: any) {
      setError(e.message || t("empty.error"));
    }
  };

  const submitCreate = async () => {
    if (!form.name.trim()) return;
    setSaving(true);
    setError("");
    try {
      const whenToUse = form.when_to_use
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      await api.post("/personas", {
        name: form.name.trim(),
        description: form.description,
        system_prompt: form.system_prompt,
        when_to_use: whenToUse,
        begin_dialogs: [],
        mood_dialogs: [],
        variables: {},
      });
      setForm(EMPTY_FORM);
      setShowCreate(false);
      setMsg(t("persona.created"));
      setTimeout(() => setMsg(""), 2500);
      await loadList();
      setSelected(form.name.trim());
    } catch (e: any) {
      setError(e.message || t("empty.error"));
    } finally {
      setSaving(false);
    }
  };

  const selectedCard = (name: string) =>
    name === selected
      ? "border-accent bg-accent/5"
      : "border-border hover:bg-bg-tertiary";

  return (
    <div className="flex h-full flex-col">
      {/* Top bar: title + active switch + create */}
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-border bg-bg-secondary px-4">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-text-primary">{t("persona.title")}</h2>
          {defaultName && (
            <span className="inline-flex items-center gap-1 rounded-full bg-accent/10 px-2 py-0.5 text-xs text-accent">
              {t("persona.active")}: {activeName || defaultName}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <select
            value={activeName || defaultName}
            onChange={(e) => switchActive(e.target.value)}
            className="input-field max-w-[12rem] text-xs"
            title={t("persona.switchTo")}
          >
            {personas.map((p) => (
              <option key={p.name} value={p.name}>{p.name}</option>
            ))}
          </select>
          <button
            onClick={() => setShowCreate(true)}
            className="btn-primary inline-flex items-center gap-1 px-3 py-1.5 text-xs"
          >
            <Plus size={14} /> {t("persona.create")}
          </button>
        </div>
      </div>

      {(error || msg) && (
        <div className={`shrink-0 px-4 py-2 text-xs ${error ? "text-error" : "text-success"}`}>
          {error || msg}
        </div>
      )}

      {/* Master / detail */}
      <div className="flex min-h-0 flex-1">
        {/* Left list */}
        <div className="w-72 shrink-0 overflow-y-auto border-r border-border p-3">
          {loading ? (
            <div className="flex h-32 items-center justify-center text-xs text-text-muted">
              {t("empty.loading")}
            </div>
          ) : personas.length === 0 ? (
            <div className="px-1 text-xs text-text-muted">{t("persona.empty")}</div>
          ) : (
            <div className="space-y-2">
              {personas.map((p) => (
                <button
                  key={p.name}
                  onClick={() => setSelected(p.name)}
                  className={`block w-full rounded-lg border p-3 text-left transition-colors ${selectedCard(p.name)}`}
                >
                  <div className="flex items-center gap-2">
                    <span className="truncate text-sm font-semibold text-text-primary">{p.name}</span>
                    {p.name === defaultName && (
                      <Star size={12} className="shrink-0 fill-accent text-accent" />
                    )}
                    {BUILTIN_PERSONAS.has(p.name) && (
                      <span className="shrink-0 rounded bg-bg-tertiary px-1.5 py-0.5 text-[10px] text-text-muted">
                        {t("persona.builtin")}
                      </span>
                    )}
                  </div>
                  {p.description ? (
                    <p className="mt-1 line-clamp-2 text-xs text-text-secondary">{p.description}</p>
                  ) : (
                    <p className="mt-1 line-clamp-2 text-xs text-text-muted">
                      {p.system_prompt.slice(0, 80) || "—"}
                    </p>
                  )}
                  {p.when_to_use && p.when_to_use.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {p.when_to_use.slice(0, 3).map((w, i) => (
                        <span key={i} className="rounded bg-bg-tertiary px-1.5 py-0.5 text-[10px] text-text-muted">
                          {w}
                        </span>
                      ))}
                    </div>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Right detail */}
        <div className="min-w-0 flex-1 overflow-y-auto p-6">
          {!detail ? (
            <div className="flex h-full items-center justify-center text-sm text-text-muted">
              {t("persona.selectPrompt")}
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-5">
              <div className="card">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h3 className="text-base font-semibold text-text-primary">{detail.name}</h3>
                    {detail.description && (
                      <p className="mt-1 text-sm text-text-secondary">{detail.description}</p>
                    )}
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <button
                      onClick={() => switchActive(detail.name)}
                      className="btn-secondary inline-flex items-center gap-1 px-2.5 py-1 text-xs"
                    >
                      <Zap size={12} /> {t("persona.setActive")}
                    </button>
                    <button
                      onClick={() => setAsDefault(detail.name)}
                      className="btn-secondary inline-flex items-center gap-1 px-2.5 py-1 text-xs"
                    >
                      <Star size={12} /> {t("persona.setDefault")}
                    </button>
                    {!BUILTIN_PERSONAS.has(detail.name) && (
                      <button
                        onClick={() => removePersona(detail.name)}
                        className="inline-flex items-center gap-1 rounded-lg border border-error/30 bg-error/5 px-2.5 py-1 text-xs text-error/80 transition-colors hover:bg-error/10 hover:text-error"
                      >
                        <Trash2 size={12} /> {t("persona.delete")}
                      </button>
                    )}
                    <button
                      onClick={() => {
                        setForm({
                          name: `${detail.name}_copy`,
                          description: detail.description || '',
                          system_prompt: detail.system_prompt || '',
                          when_to_use: (detail.when_to_use || []).join(', '),
                        });
                        setShowCreate(true);
                      }}
                      className="inline-flex items-center gap-1 rounded-lg border border-border px-2.5 py-1 text-xs text-text-secondary transition-colors hover:bg-bg-tertiary"
                      title="Clone this persona"
                    >
                      <Copy size={12} /> Clone
                    </button>
                  </div>
                </div>
              </div>

              {/* when_to_use */}
              <div className="card space-y-2">
                <h4 className="text-xs font-semibold text-text-secondary">{t("persona.whenToUse")}</h4>
                {detail.when_to_use && detail.when_to_use.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {detail.when_to_use.map((w, i) => (
                      <span key={i} className="rounded bg-accent/10 px-2 py-0.5 text-xs text-accent">{w}</span>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-text-muted">{t("persona.noWhenToUse")}</p>
                )}
              </div>

              {/* system prompt */}
              <div className="card space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-semibold text-text-secondary">{t("persona.systemPrompt")}</h4>
                  <span className={`text-[10px] ${(detail.system_prompt || '').length > 4000 ? 'text-warning' : 'text-text-muted'}`}>
                    {(detail.system_prompt || '').length} chars · ~{Math.round((detail.system_prompt || '').length / 4)} tokens
                  </span>
                </div>
                <pre className="max-h-80 overflow-auto rounded-lg border border-border bg-bg-tertiary p-3 text-xs leading-relaxed text-text-primary whitespace-pre-wrap">
                  {detail.system_prompt || "—"}
                </pre>
              </div>

              {/* Personality traits radar visualization */}
              <div className="card space-y-3">
                <h4 className="text-xs font-semibold text-text-secondary">Personality Profile</h4>
                {(() => {
                  const prompt = (detail.system_prompt || '').toLowerCase();
                  const traits = [
                    { name: 'Analytical', keywords: ['analyz', 'logic', 'reason', 'precise', 'accurate', 'systematic'], color: 'var(--accent)' },
                    { name: 'Creative', keywords: ['creativ', 'imagin', 'brainstorm', 'novel', 'innovativ'], color: '#ec4899' },
                    { name: 'Helpful', keywords: ['help', 'assist', 'support', 'guide', 'explain'], color: '#22c55e' },
                    { name: 'Formal', keywords: ['formal', 'professional', 'academic', 'rigor', 'precise'], color: '#3b82f6' },
                    { name: 'Friendly', keywords: ['friend', 'warm', 'casual', 'conversational', 'approachable'], color: '#f59e0b' },
                    { name: 'Technical', keywords: ['technical', 'code', 'program', 'algorithm', 'data', 'compute'], color: '#8b5cf6' },
                  ];
                  const scores = traits.map(t => {
                    const score = t.keywords.reduce((sum, kw) => sum + (prompt.includes(kw) ? 1 : 0), 0);
                    return { ...t, score: Math.min(score / 3, 1) }; // normalize 0-1
                  });
                  const activeTraits = scores.filter(t => t.score > 0);
                  if (activeTraits.length === 0) {
                    return <p className="text-xs text-text-muted">No personality traits detected from the system prompt.</p>;
                  }
                  return (
                    <div className="space-y-2">
                      {scores.map(t => (
                        <div key={t.name} className="flex items-center gap-3">
                          <span className="w-20 text-xs text-text-secondary">{t.name}</span>
                          <div className="h-2 flex-1 overflow-hidden rounded-full bg-bg-tertiary">
                            <div
                              className="h-full rounded-full transition-all"
                              style={{ width: `${t.score * 100}%`, backgroundColor: t.color, opacity: t.score > 0 ? 1 : 0.2 }}
                            />
                          </div>
                          <span className="w-8 text-right text-[10px] text-text-muted">{Math.round(t.score * 100)}%</span>
                        </div>
                      ))}
                    </div>
                  );
                })()}
              </div>

              {/* begin dialogs */}
              <div className="card space-y-2">
                <h4 className="text-xs font-semibold text-text-secondary">{t("persona.beginDialogs")}</h4>
                {detail.begin_dialogs && detail.begin_dialogs.length > 0 ? (
                  <div className="space-y-2">
                    {detail.begin_dialogs.map((d, i) => (
                      <div key={i} className="rounded-lg border border-border bg-bg-tertiary p-2 text-xs">
                        {d.role && <span className="font-medium text-accent">{d.role}: </span>}
                        <span className="text-text-secondary">{d.content}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-text-muted">{t("persona.noDialogs")}</p>
                )}
              </div>

              {/* mood dialogs */}
              <div className="card space-y-2">
                <h4 className="text-xs font-semibold text-text-secondary">{t("persona.moodDialogs")}</h4>
                {detail.mood_dialogs && detail.mood_dialogs.length > 0 ? (
                  <div className="space-y-2">
                    {detail.mood_dialogs.map((d, i) => (
                      <div key={i} className="rounded-lg border border-border bg-bg-tertiary p-2 text-xs">
                        {d.role && <span className="font-medium text-accent">{d.role}: </span>}
                        <span className="text-text-secondary">{d.content}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-text-muted">{t("persona.noDialogs")}</p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Create modal */}
      {showCreate && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          onClick={() => setShowCreate(false)}
        >
          <div
            className="w-full max-w-lg space-y-4 rounded-xl border border-border bg-bg-secondary p-5 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-sm font-semibold text-text-primary">{t("persona.createTitle")}</h3>

            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-text-secondary">{t("persona.name")}</label>
                <input
                  className="input"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder={t("persona.namePlaceholder")}
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-text-secondary">{t("persona.description")}</label>
                <input
                  className="input"
                  value={form.description}
                  onChange={(e) => setForm({ ...form, description: e.target.value })}
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-text-secondary">{t("persona.whenToUse")}</label>
                <input
                  className="input"
                  value={form.when_to_use}
                  onChange={(e) => setForm({ ...form, when_to_use: e.target.value })}
                  placeholder={t("persona.whenToUseHint")}
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-text-secondary">{t("persona.systemPrompt")}</label>
                <textarea
                  className="input h-40 resize-none"
                  value={form.system_prompt}
                  onChange={(e) => setForm({ ...form, system_prompt: e.target.value })}
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-1">
              <button onClick={() => setShowCreate(false)} className="btn-secondary px-3 py-1.5 text-xs">
                {t("persona.cancel")}
              </button>
              <button
                onClick={submitCreate}
                disabled={!form.name.trim() || saving}
                className="btn-primary px-3 py-1.5 text-xs disabled:opacity-50"
              >
                {t("persona.save")}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
