import { useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FolderTree } from 'lucide-react';
import { PanelHeader } from '../settings-shared';
import EmptyState from '../EmptyState';
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
  const { t } = useTranslation();
  const [remoteFiles, setRemoteFiles] = useState<any[] | null>(null);
  const [transferMsg, setTransferMsg] = useState('');
  const uploadRef = useRef<HTMLInputElement>(null);

  const [uploadPct, setUploadPct] = useState(0);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setTransferMsg(t('files.uploading', { name: file.name }));
    setUploadPct(0);
    try {
      await api.uploadWithProgress('/transfer/upload', file, (loaded, total) => {
        setUploadPct(Math.round((loaded / total) * 100));
      });
      setTransferMsg(t('files.uploaded', { name: file.name }));
      setUploadPct(100);
      setTimeout(() => setUploadPct(0), 2000);
    } catch (err: any) {
      setTransferMsg(`Upload failed: ${err.message}`);
      setUploadPct(0);
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
    setTransferMsg(t('files.syncing'));
    try {
      await api.post('/transfer/sync', { path: '.' });
      setTransferMsg(t('files.syncComplete'));
    } catch (err: any) {
      setTransferMsg(`Sync failed: ${err.message}`);
    }
  };

  const downloadRemote = async (path: string) => {
    setTransferMsg(`Downloading ${path}…`);
    try {
      const blob = await api.getBlob(`/transfer/download?path=${encodeURIComponent(path)}`);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = path.split('/').pop() || 'download';
      a.click();
      URL.revokeObjectURL(url);
      setTransferMsg(`Downloaded ${path}`);
    } catch (err: any) {
      setTransferMsg(`Download failed: ${err.message}`);
    }
  };

  return (
    <div className="flex h-full">
      {/* File tree sidebar */}
      <aside className="flex w-72 flex-col border-r border-border bg-bg-secondary">
        <PanelHeader title={t('files.workspace')}>
          <button
            onClick={() => cwd && loadDir(cwd)}
            className="text-xs text-text-secondary hover:text-text-primary"
          >
            {t('files.refresh')}
          </button>
          <button
            onClick={() => uploadRef.current?.click()}
            className="text-xs text-text-secondary hover:text-text-primary"
          >
            {t('files.upload')}
          </button>
          <input ref={uploadRef} type="file" className="hidden" onChange={handleUpload} />
          <button
            onClick={browseRemote}
            className={`text-xs ${remoteFiles ? 'text-accent' : 'text-text-secondary hover:text-text-primary'}`}
          >
            {t('files.remote')}
          </button>
          <button
            onClick={syncRemote}
            className="text-xs text-text-secondary hover:text-text-primary"
          >
            {t('files.sync')}
          </button>
        </PanelHeader>
        <div className="flex-1 overflow-y-auto p-2">
          {remoteFiles ? (
            <div className="space-y-1">
              {remoteFiles.length === 0 ? (
                <EmptyState icon={FolderTree} title={t('files.noRemote')} subtitle={t('files.connectHint')} />
              ) : remoteFiles.map((f, i) => (
                <div key={i} className="flex items-center justify-between rounded px-2 py-1 text-xs hover:bg-bg-tertiary">
                  <span>{f.is_dir ? '\u{1F4C1} ' : '\u{1F4C4} '}{f.name || f.path || String(f)}</span>
                  {!f.is_dir && (
                    <button
                      onClick={() => downloadRemote(f.path || f.name)}
                      className="text-text-muted hover:text-accent"
                    >
                      ↓
                    </button>
                  )}
                </div>
              ))}
            </div>
          ) : cwd ? (
            renderTree(cwd)
          ) : (
            <div className="p-4 text-xs text-text-muted">{t('files.loading')}</div>
          )}
        </div>
        <div className="border-t border-border p-3 text-xs text-text-muted truncate">
          {transferMsg || cwd}
          {uploadPct > 0 && uploadPct < 100 && (
            <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-bg-tertiary">
              <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${uploadPct}%` }} />
            </div>
          )}
        </div>
      </aside>

      {/* Editor */}
      <div className="flex flex-1 flex-col bg-bg-primary">
        {/* ponytail: raw div, not PanelHeader — title is a dynamic file path needing
            truncate + font-medium, which PanelHeader's <h2> can't express. Add a
            titleClassName prop to PanelHeader if a second case shows up. */}
        <div className="flex h-12 items-center justify-between border-b border-border bg-bg-secondary px-4">
          <span className="text-sm font-medium truncate">
            {selectedFile || t('files.noFileSelected')}
          </span>
          <div className="flex items-center gap-3">
            {editorDirty && (
              <span className="text-xs text-warning">{t('files.unsavedChanges')}</span>
            )}
            {editorMsg && (
              <span className="text-xs text-success">{editorMsg}</span>
            )}
            <button
              onClick={saveFile}
              disabled={!selectedFile || !editorDirty}
              className="btn-primary px-3 py-1.5 text-xs"
            >
              {t('files.save')}
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
            {t('files.selectHint')}
          </div>
        )}
      </div>
    </div>
  );
}
