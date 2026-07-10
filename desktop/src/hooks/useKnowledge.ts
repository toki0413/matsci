/**
 * useKnowledge — Manages knowledge base state and API calls.
 *
 * Encapsulates document upload, deep PDF parsing (6-stage pipeline),
 * knowledge querying, and document management.
 */
import { useState, useRef, useCallback } from 'react';
import { api } from '../lib/api';
import type { KbDoc, DocumentParseResult, DocumentGraph } from '../types/domain';

export function useKnowledge() {
  const [kbDocs, setKbDocs] = useState<KbDoc[]>([]);
  const [kbAvailable, setKbAvailable] = useState(false);
  const [kbMsg, setKbMsg] = useState('');
  const [kbQuery, setKbQuery] = useState('');
  const [kbChunks, setKbChunks] = useState<any[]>([]);
  const [parseLoading, setParseLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const parseFileInputRef = useRef<HTMLInputElement>(null);

  const loadKnowledge = async () => {
    try {
      const data = await api.get<{ documents?: any[]; available?: any }>('/knowledge');
      setKbDocs(data.documents || []);
      setKbAvailable(data.available);
    } catch (e: any) {
      setKbMsg(`Failed to load knowledge base: ${e.message}`);
    }
  };

  const uploadKnowledge = async (file: File) => {
    setKbMsg('Uploading…');
    try {
      const form = new FormData();
      form.append('file', file);
      const data = await api.upload<{ success?: boolean; error?: string; document?: { chunks: number } }>(
        '/knowledge/upload',
        form
      );
      if (data.success) {
        setKbMsg(`Uploaded ${data.document?.chunks ?? 0} chunks from ${file.name}`);
        loadKnowledge();
      } else {
        setKbMsg(`Upload failed: ${data.error}`);
      }
    } catch (e: any) {
      setKbMsg(`Upload error: ${e.message}`);
    }
  };

  const parseDocument = async (file: File) => {
    setParseLoading(true);
    setKbMsg('Parsing document (6-stage pipeline)…');
    try {
      const d = await api.upload<DocumentParseResult>('/document/parse', file);
      setKbMsg(
        `✅ Parsed: ${d.info_packages || 0} info packages, ` +
        `${d.graph?.nodes?.length || 0} graph nodes, ` +
        `${d.graph?.edges?.length || 0} edges`
      );
      loadKnowledge();
    } catch (e) {
      setKbMsg(`Parse error: ${(e as Error).message}`);
    } finally {
      setParseLoading(false);
    }
  };

  const loadDocumentGraph = useCallback(async (docId: string) => {
    try {
      const data = await api.get<DocumentGraph>(`/document/${docId}/graph`);
      setKbMsg(
        `📊 Document graph: ${data.nodes?.length || 0} nodes, ` +
        `${data.edges?.length || 0} edges`
      );
      return data;
    } catch { /* ignore */ }
    return null;
  }, []);

  const deleteKnowledge = async (docId: string) => {
    try {
      await api.del(`/knowledge/${docId}`);
      loadKnowledge();
    } catch (e: any) {
      setKbMsg(`Delete failed: ${e.message}`);
    }
  };

  const queryKnowledge = async () => {
    if (!kbQuery.trim()) return;
    setKbMsg('Querying…');
    try {
      const data = await api.post<{ chunks?: any[] } & Record<string, any>>(
        '/knowledge/query',
        { query: kbQuery, top_k: 5 }
      );
      setKbChunks(data.chunks || []);
      setKbMsg(data.chunks?.length ? `Found ${data.chunks.length} chunks` : 'No results');
    } catch (e: any) {
      setKbMsg(`Query failed: ${e.message}`);
    }
  };

  const ingestUrl = async (url: string) => {
    if (!url.trim()) return;
    setKbMsg('Fetching web page…');
    try {
      const data = await api.post<{ success?: boolean; error?: string; document?: any; source_url?: string }>(
        '/knowledge/ingest-url',
        { url }
      );
      if (data.success) {
        setKbMsg(`Added ${data.source_url || url} to knowledge base`);
        loadKnowledge();
      } else {
        setKbMsg(`URL ingest failed: ${data.error}`);
      }
    } catch (e: any) {
      setKbMsg(`URL ingest error: ${e.message}`);
    }
  };

  const loadProvenanceDag = useCallback(async () => {
    try {
      const data = await api.get<{ success?: boolean; data?: { nodes: any[]; edges: any[] } }>('/provenance/dag?n=50');
      return data;
    } catch {
      return { success: false, data: { nodes: [], edges: [] } };
    }
  }, []);

  return {
    kbDocs, kbAvailable, kbMsg, kbQuery, kbChunks, parseLoading,
    fileInputRef, parseFileInputRef,
    setKbQuery, setKbMsg,
    loadKnowledge, uploadKnowledge, parseDocument, loadDocumentGraph,
    deleteKnowledge, queryKnowledge, ingestUrl, loadProvenanceDag,
  };
}
