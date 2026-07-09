import { useState, useRef } from 'react';
import { PanelHeader } from '../settings-shared';
import { api } from '../../lib/api';

interface FilesPanelProps {
  cwd: string;
  selectedFile: string;
  editorContent: string;
  editorDirty: boolean;
  editorMsg: string;
  setEditorContent: (v: string) => void;
  setEditorDirty: (v: boolean) => void;
  loadDir: (dir: string) => void;
  saveFile: () => void;
  renderTree: (dir: string) => React.ReactNode;
}

export function FilesPanel({
  cwd, selectedFile, editorContent, editorDirty, editorMsg,
  setEditorContent, setEditorDirty, loadDir, saveFile, renderTree,
}: FilesPanelProps) {
  const [remoteFiles, setRemoteFiles] = useState<any[] | null>(null);
  const [transferMsg, setTransferMsg] = useState('');
  const uploadRef = useRef<HTMLInputElement>(null);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setTransferMsg(`Uploading ${file.name}…`);
    try {
      await api.upload('/transfer/upload', file);
      setTransferMsg(`Uploaded ${file.name}`);
    } catch (err: any) {
      setTransferMsg(`Upload failed: ${err.message}`);
    }
    e.target.value = '';
  };

  const browseRemote = async () => {
    if (remoteFiles) { setRemoteFiles(null); return; }
    try {
      const data = await api.get<any>('/transfer/browse?path=.');
      const list = Array.isArray(data) ? data : (data.entries || data.files || []);
      setRemoteFiles(list);
      setTransferMsg('');
    } catch (err: any) {
      setTransferMsg(`Browse failed: ${err.message}`);
    }
  };

  const syncRemote = async () => {
    setTransferMsg('Syncing…');
    try {
      await api.post('/transfer/sync', { path: '.' });
      setTransferMsg('Sync complete');
    } catch (err: any) {
      setTransferMsg(`Sync failed: ${err.message}`);
    }
  };

  return (
    <div className="flex h-full">
      {/* File tree sidebar */}
      <aside className="flex w-72 flex-col border-r border-border bg-bg-secondary">
        <PanelHeader title="Workspace">
          <button
            onClick={() => cwd && loadDir(cwd)}
            className="text-xs text-text-secondary hover:text-text-primary"
          >
            Refresh
          </button>
          <button
            onClick={() => uploadRef.current?.click()}
            className="text-xs text-text-secondary hover:text-text-primary"
          >
            Upload
          </button>
          <input ref={uploadRef} type="file" className="hidden" onChange={handleUpload} />
          <button
            onClick={browseRemote}
            className={`text-xs ${remoteFiles ? 'text-accent' : 'text-text-secondary hover:text-text-primary'}`}
          >
            Remote
          </button>
          <button
            onClick={syncRemote}
            className="text-xs text-text-secondary hover:text-text-primary"
          >
            Sync
          </button>
        </PanelHeader>
        <div className="flex-1 overflow-y-auto p-2">
          {remoteFiles ? (
            <div className="space-y-1">
              {remoteFiles.length === 0 ? (
                <p className="p-2 text-xs text-text-muted">No files on remote</p>
              ) : remoteFiles.map((f, i) => (
                <div key={i} className="rounded px-2 py-1 text-xs hover:bg-bg-tertiary">
                  {f.is_dir ? '\u{1F4C1} ' : '\u{1F4C4} '}{f.name || f.path || String(f)}
                </div>
              ))}
            </div>
          ) : cwd ? (
            renderTree(cwd)
          ) : (
            <div className="p-4 text-xs text-text-muted">Loading workspace…</div>
          )}
        </div>
        <div className="border-t border-border p-3 text-xs text-text-muted truncate">
          {transferMsg || cwd}
        </div>
      </aside>

      {/* Editor */}
      <div className="flex flex-1 flex-col bg-bg-primary">
        {/* ponytail: raw div, not PanelHeader — title is a dynamic file path needing
            truncate + font-medium, which PanelHeader's <h2> can't express. Add a
            titleClassName prop to PanelHeader if a second case shows up. */}
        <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-4">
          <span className="text-sm font-medium truncate">
            {selectedFile || "No file selected"}
          </span>
          <div className="flex items-center gap-3">
            {editorDirty && (
              <span className="text-xs text-warning">Unsaved changes</span>
            )}
            {editorMsg && (
              <span className="text-xs text-success">{editorMsg}</span>
            )}
            <button
              onClick={saveFile}
              disabled={!selectedFile || !editorDirty}
              className="btn-primary px-3 py-1.5 text-xs"
            >
              Save
            </button>
          </div>
        </div>
        {selectedFile ? (
          <textarea
            value={editorContent}
            onChange={(e) => {
              setEditorContent(e.target.value);
              setEditorDirty(true);
            }}
            className="flex-1 resize-none bg-bg-primary p-4 font-mono text-sm text-text-primary focus:outline-none"
            spellCheck={false}
          />
        ) : (
          <div className="flex flex-1 items-center justify-center text-sm text-text-muted">
            Select a file from the workspace to edit
          </div>
        )}
      </div>
    </div>
  );
}
