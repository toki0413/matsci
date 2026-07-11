/**
 * useWorkspace — Manages file explorer, code editor, and terminal state.
 *
 * Encapsulates Tauri IPC for directory browsing, file editing, and
 * terminal output streaming via Tauri events.
 */
import { useState, useRef, useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type { FileEntry } from '../types/domain';

export function useWorkspace() {
  const [cwd, setCwd] = useState('');
  const [dirCache, setDirCache] = useState<Record<string, FileEntry[]>>({});
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [editorContent, setEditorContent] = useState('');
  const [editorDirty, setEditorDirty] = useState(false);
  const [editorMsg, setEditorMsg] = useState('');
  const [terminalOutput, setTerminalOutput] = useState('');
  const [terminalInput, setTerminalInput] = useState('');
  const terminalEndRef = useRef<HTMLDivElement>(null);

  const loadDir = useCallback(async (path: string) => {
    try {
      const entries = (await invoke('read_dir', { path })) as FileEntry[];
      setDirCache((prev) => ({ ...prev, [path]: entries }));
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
    try {
      const content = (await invoke('read_file', { path })) as string;
      setSelectedFile(path);
      setEditorContent(content);
      setEditorDirty(false);
      setEditorMsg('');
    } catch (e: any) {
      setEditorMsg(`Failed to open file: ${e}`);
    }
  };

  const saveFile = async () => {
    if (!selectedFile) return;
    try {
      await invoke('write_file', { path: selectedFile, content: editorContent });
      setEditorDirty(false);
      setEditorMsg('Saved.');
      setTimeout(() => setEditorMsg(''), 2000);
    } catch (e: any) {
      setEditorMsg(`Save failed: ${e}`);
    }
  };

  // Load initial CWD and directory listing
  useEffect(() => {
    if (!("__TAURI_INTERNALS__" in window)) return;
    (async () => {
      try {
        const path = (await invoke('get_cwd')) as string;
        setCwd(path);
        await loadDir(path);
        setExpandedDirs((prev) => new Set(prev).add(path));
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
        <div key={fullPath}>
          <div
            className={`flex cursor-pointer items-center gap-1 rounded px-2 py-0.5 text-xs hover:bg-bg-hover ${
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
          {entry.is_dir && isExpanded && renderTree(fullPath, depth + 1)}
        </div>
      );
    });
  };

  return {
    cwd, dirCache, expandedDirs, selectedFile,
    editorContent, editorDirty, editorMsg,
    terminalOutput, terminalInput, terminalEndRef,
    setCwd, setDirCache, setExpandedDirs, setSelectedFile,
    setEditorContent, setEditorDirty, setEditorMsg,
    setTerminalOutput, setTerminalInput,
    loadDir, toggleDir, openFile, saveFile, renderTree,
  };
}
