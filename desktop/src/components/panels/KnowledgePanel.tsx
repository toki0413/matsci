import { PanelHeader } from '../settings-shared';
import type { AppConfig } from '../../types/domain';

interface KbDoc {
  doc_id: string;
  filename: string;
}

interface KbChunk {
  text: string;
  distance?: number;
  metadata?: { filename?: string };
}

interface KnowledgePanelProps {
  config: AppConfig;
  setConfig: (c: AppConfig) => void;
  saveConfig: (c: AppConfig) => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
  parseFileInputRef: React.RefObject<HTMLInputElement>;
  parseLoading: boolean;
  kbMsg: string;
  kbDocs: KbDoc[];
  kbAvailable: boolean;
  kbQuery: string;
  kbChunks: KbChunk[];
  setKbQuery: (v: string) => void;
  uploadKnowledge: (file: File) => void;
  parseDocument: (file: File) => void;
  loadDocumentGraph: (docId: string) => void;
  deleteKnowledge: (docId: string) => void;
  queryKnowledge: () => void;
}

export function KnowledgePanel({
  config, setConfig, saveConfig,
  fileInputRef, parseFileInputRef, parseLoading,
  kbMsg, kbDocs, kbAvailable, kbQuery, kbChunks, setKbQuery,
  uploadKnowledge, parseDocument, loadDocumentGraph, deleteKnowledge, queryKnowledge,
}: KnowledgePanelProps) {
  return (
    <div data-component="knowledge-panel" className="kb-panel flex h-full flex-col">
      <PanelHeader title="Knowledge Base" className="kb-header px-6">
        <label className="kb-rag-toggle flex cursor-pointer items-center gap-2">
          <input
            type="checkbox"
            checked={config.rag_enabled}
            onChange={(e) => {
              const next = { ...config, rag_enabled: e.target.checked };
              setConfig(next);
              saveConfig(next);
            }}
            className="h-4 w-4 rounded border-border bg-bg-tertiary text-accent"
          />
          <span className="text-xs text-text-secondary">Use RAG in chat</span>
        </label>
      </PanelHeader>
      <div className="flex flex-1 overflow-hidden">
        {/* Upload / docs */}
        <aside className="flex w-80 flex-col border-r border-border bg-bg-secondary p-4">
          <div className="kb-upload mb-4 rounded-lg border border-dashed border-border bg-bg-tertiary p-4 text-center">
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".txt,.md,.pdf,.py,.json,.yaml,.yml"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) uploadKnowledge(file);
                if (fileInputRef.current) fileInputRef.current.value = "";
              }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="btn-primary w-full text-xs"
            >
              📤 Upload document
            </button>
            <p className="mt-2 text-xs text-text-muted">
              Supports TXT, MD, PDF, code files
            </p>
          </div>

          {/* Deep PDF parsing — 6-stage document analysis pipeline */}
          <div className="kb-deep-parse mb-4 rounded-lg border border-accent/20 bg-accent/5 p-3">
            <input
              ref={parseFileInputRef}
              type="file"
              className="hidden"
              accept=".pdf"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) parseDocument(file);
                if (parseFileInputRef.current) parseFileInputRef.current.value = "";
              }}
            />
            <button
              onClick={() => parseFileInputRef.current?.click()}
              disabled={parseLoading}
              className="w-full rounded-lg border border-accent/30 bg-accent/10 px-3 py-2 text-xs font-medium text-accent hover:bg-accent/20 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {parseLoading ? "⏳ Parsing…" : "📊 Deep PDF Parse"}
            </button>
            <p className="mt-1.5 text-xs text-text-muted">
              6-stage: extract → figures → graph → relations → validate → assemble
            </p>
          </div>

          {kbMsg && (
            <div className="kb-status mb-3 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
              {kbMsg}
            </div>
          )}

          <div className="flex-1 overflow-y-auto">
            <div className="mb-2 text-xs font-medium text-text-secondary">
              Documents ({kbDocs.length})
            </div>
            {!kbAvailable && (
              <div className="text-xs text-text-muted">
                Knowledge base backend is not available. Install chromadb and sentence-transformers.
              </div>
            )}
            {kbDocs.map((doc) => (
              <div
                key={doc.doc_id}
                className="kb-doc-item mb-2 flex items-center justify-between rounded-lg border border-border bg-bg-tertiary p-2"
              >
                <span className="truncate text-xs text-text-primary">{doc.filename}</span>
                <div className="flex gap-2">
                  <button
                    onClick={() => loadDocumentGraph(doc.doc_id)}
                    className="text-xs text-accent hover:underline"
                    title="View document structure graph"
                    aria-label="View document structure graph"
                  >
                    📊
                  </button>
                  <button
                    onClick={() => deleteKnowledge(doc.doc_id)}
                    className="text-xs text-error hover:underline"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        </aside>

        {/* Query tester */}
        <div className="kb-query-area flex flex-1 flex-col bg-bg-primary p-4">
          <h3 className="mb-3 text-sm font-semibold">Test retrieval</h3>
          <div className="mb-4 flex gap-2">
            <input
              type="text"
              value={kbQuery}
              onChange={(e) => setKbQuery(e.target.value)}
              placeholder="Ask a question against the knowledge base…"
              className="input flex-1"
              onKeyDown={(e) => e.key === "Enter" && queryKnowledge()}
            />
            <button onClick={queryKnowledge} className="btn-primary">
              Search
            </button>
          </div>
          <div className="flex-1 overflow-y-auto space-y-3">
            {kbChunks.map((chunk, i) => (
              <div key={i} className="kb-chunk rounded-lg border border-border bg-bg-secondary p-3">
                <div className="mb-1 flex items-center justify-between text-xs text-text-muted">
                  <span>{chunk.metadata?.filename}</span>
                  <span>distance: {chunk.distance?.toFixed(3)}</span>
                </div>
                <p className="text-xs text-text-primary whitespace-pre-wrap">
                  {chunk.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
