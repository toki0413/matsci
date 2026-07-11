import { useTranslation } from 'react-i18next';
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
  const { t } = useTranslation();
  return (
    <div className="flex h-full flex-col">
      <PanelHeader title={t('project.title')} className="px-6">
        <button onClick={loadProjectContext} className="btn-secondary px-3 py-1.5 text-xs">
          {t('project.refresh')}
        </button>
      </PanelHeader>
      <div className="flex flex-1 overflow-hidden">
        {/* Project context editor */}
        <aside className="flex w-1/2 flex-col border-r border-border bg-bg-secondary p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold">{t('project.instructions')}</h3>
              <p className="text-[10px] text-text-muted">
                {t('project.loadedFrom')} <span className="text-text-secondary">{projectContextSource}</span>
              </p>
            </div>
            <button onClick={saveProjectContext} className="btn-primary px-3 py-1.5 text-xs">
              {t('project.save')}
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
            placeholder={t('project.instructionsPh')}
            className="input flex-1 resize-none font-mono text-sm"
            spellCheck={false}
          />
        </aside>

        {/* Codebase semantic search */}
        <div className="flex w-1/2 flex-col bg-bg-primary p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold">{t('project.codebaseSearch')}</h3>
              <p className="text-[10px] text-text-muted">
                {codebaseStatus?.available
                  ? `${codebaseStatus.indexed_files || 0} ${t('project.filesIndexed')}`
                  : t('project.notIndexed')}
              </p>
            </div>
            <button onClick={indexCodebase} className="btn-primary px-3 py-1.5 text-xs">
              {t('project.reindex')}
            </button>
          </div>
          <div className="mb-3 flex gap-2">
            <input
              type="text"
              value={codebaseQuery}
              onChange={(e) => setCodebaseQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && searchCodebase()}
              placeholder={t('project.searchPh')}
              className="input flex-1 text-sm"
            />
            <button onClick={searchCodebase} className="btn-primary text-xs">
              {t('project.search')}
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
                  <span>{t('project.chunk')} {r.chunk}</span>
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
