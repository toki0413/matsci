/**
 * useWorkspace — Manages file explorer, code editor, and terminal state.
 *
 * ADR-0001 文件 I/O 归口后端：目录浏览 / 文件读写统一走后端 /v1/fs/*，
 * 不再通过 Tauri IPC 直接做本地文件 I/O。终端输出仍走 Tauri 事件流。
 */
import { useState, useRef, useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { api } from '../lib/api';
import type { FileEntry } from '../types/domain';

// 一个打开的编辑会话。content/dirty 随输入实时更新，
// 以便在 tab 之间切换或关闭时能保留每个文件的未保存状态。
interface EditorTab {
  path: string;
  content: string;
  dirty: boolean;
}

export function useWorkspace() {
  const [cwd, setCwd] = useState('');
  const [dirCache, setDirCache] = useState<Record<string, FileEntry[]>>({});
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());
  // multi-tab：所有已打开的会话。selectedFile 兼任「当前激活 tab」的投影，
  // 保留旧单文件接口（openFile/saveFile/setSelectedFile）不被破坏。
  const [tabs, setTabs] = useState<EditorTab[]>([]);
  // selectedFile 兼任「当前激活 tab」，同时是旧单文件接口的投影。
  // tabs[].content/dirty 保存每个已打开会话，切换/关闭时据此恢复。
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [editorContent, setEditorContent] = useState('');
  const [editorDirty, setEditorDirty] = useState(false);
  const [editorMsg, setEditorMsg] = useState('');
  // 当前激活编辑器的光标位置，供状态栏显示
  const [editorCursor, setEditorCursor] = useState({ line: 0, col: 0 });
  const [terminalOutput, setTerminalOutput] = useState('');
  const [terminalInput, setTerminalInput] = useState('');
  const terminalEndRef = useRef<HTMLDivElement>(null);

  const loadDir = useCallback(async (path: string) => {
    try {
      const res = await api.get<{ entries: FileEntry[] }>('/v1/fs/list', {
        params: new URLSearchParams({ path }),
      });
      setDirCache((prev) => ({ ...prev, [path]: res.entries }));
    } catch (e: any) {
      console.error('[files] read_dir failed:', e);
    }
  }, []);

  const toggleDir = (path: string) => {
    setExpandedDirs((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
        loadDir(path);
      }
      return next;
    });
  };

  const openFile = async (path: string) => {
    // 已在某个 tab 打开过则只切换聚焦，不再重复读盘
    const existing = tabs.find((t) => t.path === path);
    if (existing) {
      activateTab(path);
      return;
    }
    try {
      const res = await api.get<{ content: string }>('/v1/fs/read', {
        params: new URLSearchParams({ path }),
      });
      setTabs((prev) => [...prev, { path, content: res.content, dirty: false }]);
      setSelectedFile(path);
      setEditorContent(res.content);
      setEditorDirty(false);
      setEditorMsg('');
    } catch (e: any) {
      setEditorMsg(`Failed to open file: ${e}`);
    }
  };

  // 切换到另一个已打开 tab，从其快照恢复内容
  const activateTab = (path: string) => {
    const tab = tabs.find((t) => t.path === path);
    if (!tab) return;
    setSelectedFile(path);
    setEditorContent(tab.content);
    setEditorDirty(tab.dirty);
    setEditorMsg('');
  };

  // 关闭一个 tab。若关的是当前激活 tab，让位给相邻（尾部）tab；保留未保存内容仅丢弃。
  const closeTab = (path: string) => {
    setTabs((prev) => prev.filter((t) => t.path !== path));
    if (selectedFile === path) {
      const nextActive = tabs.find((t) => t.path !== path);
      if (nextActive) {
        setSelectedFile(nextActive.path);
        setEditorContent(nextActive.content);
        setEditorDirty(nextActive.dirty);
      } else {
        setSelectedFile(null);
        setEditorContent('');
        setEditorDirty(false);
      }
      setEditorMsg('');
    }
  };

  // CodeMirror 输入回调：更新当前 tab 快照 + 投影 state
  const onEditContent = (path: string, content: string) => {
    setTabs((prev) => prev.map((t) => (t.path === path ? { ...t, content, dirty: true } : t)));
    setEditorContent(content);
    setEditorDirty(true);
  };

  const saveFile = async () => {
    if (!selectedFile) return;
    const tab = tabs.find((t) => t.path === selectedFile);
    const content = tab ? tab.content : editorContent;
    try {
      await api.put('/v1/fs/write', { path: selectedFile, content });
      setTabs((prev) => prev.map((t) => (t.path === selectedFile ? { ...t, dirty: false } : t)));
      setEditorDirty(false);
      setEditorMsg('Saved.');
      setTimeout(() => setEditorMsg(''), 2000);
    } catch (e: any) {
      setEditorMsg(`Save failed: ${e}`);
    }
  };

  const createDir = async (path: string) => {
    try {
      const res = await api.post<{ path: string }>('/v1/fs/mkdir', { path });
      await loadDir(path.split(/[\\/]/).slice(0, -1).join('\\'));
      return res.path;
    } catch (e: any) {
      throw new Error(`创建目录失败: ${e}`);
    }
  };

  const renameEntry = async (path: string, newName: string) => {
    try {
      await api.put('/v1/fs/rename', { path, new: newName });
      const parent = path.split(/[\\/]/).slice(0, -1).join('\\');
      await loadDir(parent);
      // 若重命名的是当前打开的文件，同步选中状态
      if (selectedFile === path) {
        setSelectedFile(newName.startsWith('.') ? `${parent}\\${newName.slice(1)}` : `${parent}\\${newName}`);
        setEditorMsg('Renamed.');
        setTimeout(() => setEditorMsg(''), 2000);
      }
    } catch (e: any) {
      throw new Error(`重命名失败: ${e}`);
    }
  };

  const deleteEntry = async (path: string) => {
    try {
      await api.del<void>('/v1/fs/delete', { params: new URLSearchParams({ path }) });
      await loadDir(path.split(/[\\/]/).slice(0, -1).join('\\'));
      if (selectedFile === path) {
        setSelectedFile(null);
        setEditorContent('');
        setEditorDirty(false);
      }
    } catch (e: any) {
      throw new Error(`删除失败: ${e}`);
    }
  };

  const openInSystem = async (path: string) => {
    try {
      await api.post('/v1/fs/open', { path });
    } catch (e: any) {
      console.error('[files] open_in_system failed:', e);
    }
  };

  // Load initial CWD and directory listing
  useEffect(() => {
    (async () => {
      try {
        const res = await api.get<{ path: string }>('/v1/fs/cwd');
        setCwd(res.path);
        await loadDir(res.path);
        setExpandedDirs((prev) => new Set(prev).add(res.path));
      } catch (e) {
        console.error('[files] get_cwd failed:', e);
      }
    })();
  }, [loadDir]);

  // Listen to integrated terminal output
  useEffect(() => {
    if (!("__TAURI_INTERNALS__" in window)) return;
    let unlisten: UnlistenFn | undefined;
    (async () => {
      unlisten = await listen('terminal-output', (event) => {
        const payload = event.payload as { source: string; text: string };
        setTerminalOutput((prev) => prev + payload.text);
      });
    })();
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  // Auto-scroll terminal
  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'auto' });
  }, [terminalOutput]);

  // Recursive file tree renderer
  const renderTree = (path: string, depth: number = 0): ReactNode => {
    const entries = dirCache[path];
    if (!entries) return null;
    const sorted = [...entries].sort((a, b) => {
      if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    return sorted.map((entry) => {
      const fullPath = entry.path;
      const isExpanded = expandedDirs.has(fullPath);
      const isSelected = selectedFile === fullPath;
      return (
        <div key={fullPath} className="group flex items-center">
          <div
            className={`flex flex-1 cursor-pointer items-center gap-1 rounded px-2 py-0.5 text-xs hover:bg-bg-hover ${
              isSelected ? 'bg-accent/15 text-accent' : 'text-text-primary'
            }`}
            style={{ paddingLeft: `${depth * 16 + 8}px` }}
            onClick={() => {
              if (entry.is_dir) {
                toggleDir(fullPath);
              } else {
                openFile(fullPath);
              }
            }}
          >
            <span className="w-4 text-center text-text-muted">
              {entry.is_dir ? (isExpanded ? '▾' : '▸') : '·'}
            </span>
            <span className="truncate">{entry.name}</span>
          </div>
          <div className="hidden items-center gap-1 pr-1 group-hover:flex">
            <button
              title="在系统资源管理器中打开"
              className="text-text-muted hover:text-accent"
              onClick={(e) => { e.stopPropagation(); openInSystem(fullPath); }}
            >
              ↗
            </button>
            <button
              title="重命名"
              className="text-text-muted hover:text-accent"
              onClick={(e) => {
                e.stopPropagation();
                const name = window.prompt('重命名:', entry.name);
                if (name && name !== entry.name) renameEntry(fullPath, name);
              }}
            >
              ✎
            </button>
            <button
              title="删除"
              className="text-text-muted hover:text-danger"
              onClick={(e) => {
                e.stopPropagation();
                if (window.confirm(`确定删除 ${entry.name} ?`)) deleteEntry(fullPath);
              }}
            >
              ✕
            </button>
          </div>
          {entry.is_dir && isExpanded && renderTree(fullPath, depth + 1)}
        </div>
      );
    });
  };

  return {
    cwd, dirCache, expandedDirs, selectedFile, tabs,
    editorContent, editorDirty, editorMsg,
    terminalOutput, terminalInput, terminalEndRef,
    editorCursor,
    setCwd, setDirCache, setExpandedDirs, setSelectedFile,
    setEditorContent, setEditorDirty, setEditorMsg, setEditorCursor,
    setTerminalOutput, setTerminalInput,
    loadDir, toggleDir, openFile, saveFile, renderTree,
    createDir, renameEntry, deleteEntry, openInSystem,
    activateTab, closeTab, onEditContent,
  };
}
