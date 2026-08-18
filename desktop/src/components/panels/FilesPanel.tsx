import { useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FolderTree } from 'lucide-react';
import { open } from '@tauri-apps/plugin-dialog';
import { PanelHeader } from '../settings-shared';
import EmptyState from '../EmptyState';
import { api } from '../../lib/api';
import { downloadBlob } from '../../lib/download';
import { CodeMirrorEditor } from '../editor/CodeMirrorEditor';

interface FilesPanelProps {
  cwd: string;
  setCwd: (v: string) => void;
  tabs: Array<{ path: string; content: string; dirty: boolean }>;
  selectedFile: string;
  editorContent: string;
  editorDirty: boolean;
  editorMsg: string;
  loadDir: (dir: string) => void;
  saveFile: () => void;
  renderTree: (dir: string) => React.ReactNode;
  createDir: (path: string) => Promise<string>;
  activateTab: (path: string) => void;
  closeTab: (path: string) => void;
  onEditContent: (path: string, content: string) => void;
  onCursor: (line: number, col: number) => void;
}

export function FilesPanel({
  cwd, setCwd, tabs, selectedFile, editorContent, editorDirty, editorMsg,
  loadDir, saveFile, renderTree, createDir,
  activateTab, closeTab, onEditContent, onCursor,
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

  const openFolder = async () => {
    let dir: string | null = null;
    try {
      const picked = await open({ directory: true, multiple: false, title: t('files.openFolderTitle') });
      if (typeof picked === 'string' && picked) dir = picked;
    } catch {
      // 非 Tauri 环境（纯 web）降级：手动输入路径
      dir = window.prompt('请输入要打开的文件夹路径:');
    }
    if (!dir) return;
    setCwd(dir);
    await loadDir(dir);
    setRemoteFiles(null);
  };

  const downloadRemote = async (path: string) => {
    setTransferMsg(`Downloading ${path}…`);
    try {
      const blob = await api.getBlob(`/transfer/download?path=${encodeURIComponent(path)}`);
      downloadBlob(blob, path.split('/').pop() || 'download');
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
            onClick={openFolder}
            className="text-xs font-semibold text-accent hover:text-text-primary"
            title={t('files.openFolderTitle')}
          >
            {t('files.open')}
          </button>
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
          <button
            onClick={async () => {
              const name = window.prompt('新建文件夹名称:');
              if (!name) return;
              try {
                await createDir(`${cwd}\\${name}`);
                loadDir(cwd);
              } catch (err: any) {
                setTransferMsg(String(err.message || err));
              }
            }}
            className="text-xs font-semibold text-accent hover:text-text-primary"
          >
            + 文件夹
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
        {/* Tab 栏：列出所有已打开的会话，可切换/关闭。始终可见，保证多文件好操作 */}
        <div className="flex h-9 items-center border-b border-border bg-bg-secondary">
          <div className="flex flex-1 items-center overflow-x-auto">
            {tabs.length === 0 && (
              <span className="px-3 text-xs text-text-muted">{t('files.noFileSelected')}</span>
            )}
            {tabs.map((tab) => {
              const fname = tab.path.split(/[\\/]/).pop() || tab.path;
              const active = tab.path === selectedFile;
              return (
                <button
                  key={tab.path}
                  onClick={() => activateTab(tab.path)}
                  className={`group flex items-center gap-1 border-r border-border px-3 py-2 text-xs whitespace-nowrap ${
                    active ? 'bg-bg-primary text-text-primary' : 'text-text-secondary hover:bg-bg-tertiary'
                  }`}
                >
                  <span className="max-w-40 truncate">{fname}</span>
                  {tab.dirty && <span className="text-warning">●</span>}
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => { e.stopPropagation(); closeTab(tab.path); }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') { e.stopPropagation(); closeTab(tab.path); }
                    }}
                    className="text-text-muted hover:text-danger"
                    title={t('files.close')}
                  >
                    ✕
                  </span>
                </button>
              );
            })}
          </div>
          <div className="flex items-center gap-3 px-4">
            {editorDirty && (
              <span className="text-xs text-warning">{t('files.unsavedChanges')}</span>
            )}
            {editorMsg && (
              <span className="text-xs text-success">{editorMsg}</span>
            )}
            <button
              onClick={saveFile}
              disabled={!selectedFile || !editorDirty}
              className="btn-primary px-3 py-1 text-xs"
            >
              {t('files.save')}
            </button>
          </div>
        </div>
        {selectedFile ? (
          <div className="min-h-0 flex-1">
            <CodeMirrorEditor
              path={selectedFile}
              value={editorContent}
              onChange={onEditContent}
              onCursor={onCursor}
            />
          </div>
        ) : (
          <div className="flex flex-1 items-center justify-center text-sm text-text-muted">
            {t('files.selectHint')}
          </div>
        )}
      </div>
    </div>
  );
}
