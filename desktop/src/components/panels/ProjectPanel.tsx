import { PanelHeader } from '../settings-shared';

interface CodebaseStatus {
  available?: boolean;
  indexed_files?: number;
}

interface CodebaseResult {
  path: string;
  chunk: number;
  text: string;
}

interface ProjectPanelProps {
  projectContext: string;
  projectContextSource: string;
  projectContextMsg: string;
  setProjectContext: (v: string) => void;
  loadProjectContext: () => void;
  saveProjectContext: () => void;
  codebaseStatus: CodebaseStatus | null;
  codebaseQuery: string;
  codebaseResults: CodebaseResult[];
  codebaseMsg: string;
  setCodebaseQuery: (v: string) => void;
  indexCodebase: () => void;
  searchCodebase: () => void;
}

export function ProjectPanel({
  projectContext, projectContextSource, projectContextMsg,
  setProjectContext, loadProjectContext, saveProjectContext,
  codebaseStatus, codebaseQuery, codebaseResults, codebaseMsg,
  setCodebaseQuery, indexCodebase, searchCodebase,
}: ProjectPanelProps) {
  return (
    <div className="flex h-full flex-col">
      <PanelHeader title={"Project Context & Codebase"} className="px-6">
        <button onClick={loadProjectContext} className="btn-secondary px-3 py-1.5 text-xs">
          Refresh
        </button>
      </PanelHeader>
      <div className="flex flex-1 overflow-hidden">
        {/* Project context editor */}
        <aside className="flex w-1/2 flex-col border-r border-border bg-bg-secondary p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold">Project Instructions</h3>
              <p className="text-[10px] text-text-muted">
                Loaded from: <span className="text-text-secondary">{projectContextSource}</span>
              </p>
            </div>
            <button onClick={saveProjectContext} className="btn-primary px-3 py-1.5 text-xs">
              Save
            </button>
          </div>
          {projectContextMsg && (
            <div className="mb-3 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
              {projectContextMsg}
            </div>
          )}
          <textarea
            value={projectContext}
            onChange={(e) => setProjectContext(e.target.value)}
            placeholder="Write project-level instructions here (coding style, conventions, important formulas, DFT preferences...). Saved to .huginn.md in the workspace."
            className="input flex-1 resize-none font-mono text-sm"
            spellCheck={false}
          />
        </aside>

        {/* Codebase semantic search */}
        <div className="flex w-1/2 flex-col bg-bg-primary p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold">Codebase Search</h3>
              <p className="text-[10px] text-text-muted">
                {codebaseStatus?.available
                  ? `${codebaseStatus.indexed_files || 0} files indexed`
                  : "Not indexed"}
              </p>
            </div>
            <button onClick={indexCodebase} className="btn-primary px-3 py-1.5 text-xs">
              Re-index
            </button>
          </div>
          <div className="mb-3 flex gap-2">
            <input
              type="text"
              value={codebaseQuery}
              onChange={(e) => setCodebaseQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && searchCodebase()}
              placeholder="Search the codebase semantically…"
              className="input flex-1 text-sm"
            />
            <button onClick={searchCodebase} className="btn-primary text-xs">
              Search
            </button>
          </div>
          {codebaseMsg && (
            <div className="mb-3 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
              {codebaseMsg}
            </div>
          )}
          <div className="flex-1 overflow-y-auto space-y-3">
            {codebaseResults.map((r, i) => (
              <div key={i} className="rounded-lg border border-border bg-bg-secondary p-3">
                <div className="mb-1 flex items-center justify-between text-xs text-text-muted">
                  <span className="font-mono">{r.path}</span>
                  <span>chunk {r.chunk}</span>
                </div>
                <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded bg-bg-tertiary p-2 text-xs text-text-primary">
                  {r.text}
                </pre>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
