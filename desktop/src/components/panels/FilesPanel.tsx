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
  return (
    <div className="flex h-full">
      {/* File tree sidebar */}
      <aside className="flex w-72 flex-col border-r border-border bg-bg-secondary">
        <div className="flex h-12 items-center justify-between border-b border-border px-4">
          <span className="text-sm font-semibold">Workspace</span>
          <button
            onClick={() => cwd && loadDir(cwd)}
            className="text-xs text-text-secondary hover:text-text-primary"
          >
            Refresh
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {cwd ? (
            renderTree(cwd)
          ) : (
            <div className="p-4 text-xs text-text-muted">Loading workspace…</div>
          )}
        </div>
        <div className="border-t border-border p-3 text-xs text-text-muted truncate">
          {cwd}
        </div>
      </aside>

      {/* Editor */}
      <div className="flex flex-1 flex-col bg-bg-primary">
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
