/**
 * useProject — Project context and codebase search state.
 *
 * Manages project context text, codebase indexing status,
 * and semantic codebase search. Calls /project-context and
 * /codebase/* API endpoints.
 */
import { useState } from "react";
import { api } from "../lib/api";

export function useProject() {
  const [projectContext, setProjectContext] = useState<string>("");
  const [projectContextSource, setProjectContextSource] = useState<string>("none");
  const [projectContextMsg, setProjectContextMsg] = useState<string>("");
  const [codebaseStatus, setCodebaseStatus] = useState<any>(null);
  const [codebaseQuery, setCodebaseQuery] = useState<string>("");
  const [codebaseResults, setCodebaseResults] = useState<any[]>([]);
  const [codebaseMsg, setCodebaseMsg] = useState<string>("");

  const loadProjectContext = async () => {
    try {
      const data = await api.get<{ content?: string; source?: string }>("/project-context");
      setProjectContext(data.content || "");
      setProjectContextSource(data.source || "none");
      setProjectContextMsg("");
    } catch (e: any) {
      setProjectContextMsg(`Load failed: ${e.message}`);
    }
  };

  const saveProjectContext = async () => {
    setProjectContextMsg("Saving…");
    try {
      const data = await api.post<{ success?: boolean; error?: string }>(
        "/project-context",
        { content: projectContext }
      );
      if (data.success) {
        setProjectContextMsg("Saved. Agent will reload on next message.");
      } else {
        setProjectContextMsg(`Save failed: ${data.error}`);
      }
    } catch (e: any) {
      setProjectContextMsg(`Save error: ${e.message}`);
    }
  };

  const loadCodebaseStatus = async () => {
    try {
      const data = await api.get<any>("/codebase");
      setCodebaseStatus(data);
    } catch (e: any) {
      setCodebaseMsg(`Status failed: ${e.message}`);
    }
  };

  const indexCodebase = async () => {
    setCodebaseMsg("Indexing workspace…");
    try {
      const data = await api.post<{ success?: boolean; indexed_files?: number; chunks?: number; error?: string }>(
        "/codebase/index"
      );
      if (data.success) {
        setCodebaseMsg(`Indexed ${data.indexed_files} files, ${data.chunks} chunks`);
        loadCodebaseStatus();
      } else {
        setCodebaseMsg(`Index failed: ${data.error}`);
      }
    } catch (e: any) {
      setCodebaseMsg(`Index error: ${e.message}`);
    }
  };

  const searchCodebase = async () => {
    if (!codebaseQuery.trim()) return;
    setCodebaseMsg("Searching…");
    try {
      const data = await api.post<{ results?: any[] } & Record<string, any>>(
        "/codebase/search",
        { query: codebaseQuery, top_k: 8 }
      );
      setCodebaseResults(data.results || []);
      setCodebaseMsg(data.results?.length ? `Found ${data.results.length} results` : "No results");
    } catch (e: any) {
      setCodebaseMsg(`Search error: ${e.message}`);
    }
  };

  return {
    projectContext, projectContextSource, projectContextMsg,
    codebaseStatus, codebaseQuery, codebaseResults, codebaseMsg,
    setProjectContext, setProjectContextSource, setCodebaseQuery,
    loadProjectContext, saveProjectContext,
    loadCodebaseStatus, indexCodebase, searchCodebase,
  };
}
