/**
 * useKnowledge — Knowledge base CRUD hook.
 *
 * Loads documents from the backend, supports upload/delete/search.
 * Falls back gracefully when the backend is unavailable.
 */
import { useState, useEffect, useCallback } from 'react';
import {
  getKnowledge,
  uploadKnowledge,
  deleteKnowledge,
  queryKnowledge,
  type KnowledgeDoc,
} from '../api/endpoints';

export interface UseKnowledgeOptions {
  /** Fetch documents on mount (default true) */
  autoLoad?: boolean;
  /** Mock docs to use when backend is unreachable */
  fallbackDocs?: KnowledgeDoc[];
}

export function useKnowledge(opts: UseKnowledgeOptions = {}) {
  const { autoLoad = true, fallbackDocs = [] } = opts;

  const [documents, setDocuments] = useState<KnowledgeDoc[]>(fallbackDocs);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [available, setAvailable] = useState(false);
  const [searchResults, setSearchResults] = useState<
    Array<{ text: string; score: number; source: string }> | null
  >(null);

  /** Fetch document list from GET /knowledge. */
  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await getKnowledge();
      if (resp.ok && resp.data && resp.data.available !== false) {
        setDocuments(resp.data.documents || []);
        setAvailable(true);
      } else {
        setDocuments(fallbackDocs);
        setAvailable(false);
      }
    } catch (err) {
      setDocuments(fallbackDocs);
      setAvailable(false);
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [fallbackDocs]);

  /** Upload a file via POST /knowledge/upload. */
  const upload = useCallback(
    async (file: File) => {
      setLoading(true);
      setError(null);
      try {
        const resp = await uploadKnowledge(file);
        if (resp.ok && resp.data?.success) {
          await load(); // refresh list
        } else {
          setError(resp.data?.error || resp.error?.message || 'Upload failed');
        }
        return resp;
      } catch (err) {
        setError((err as Error).message);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [load],
  );

  /** Delete a document by ID. */
  const remove = useCallback(async (docId: string) => {
    try {
      const resp = await deleteKnowledge(docId);
      if (resp.ok) {
        setDocuments((prev) =>
          prev.filter((d) => d.id !== docId && d.name !== docId),
        );
      }
      return resp;
    } catch (err) {
      setError((err as Error).message);
      return null;
    }
  }, []);

  /** Search documents via POST /knowledge/query. */
  const search = useCallback(async (query: string, topK = 5) => {
    if (!query.trim()) {
      setSearchResults(null);
      return;
    }
    try {
      const resp = await queryKnowledge(query, topK);
      if (resp.ok && resp.data) {
        setSearchResults(resp.data.chunks || []);
      } else {
        setError(resp.error?.message || 'Search failed');
        setSearchResults([]);
      }
    } catch (err) {
      setError((err as Error).message);
      setSearchResults([]);
    }
  }, []);

  // Auto-load on mount
  useEffect(() => {
    if (autoLoad) load();
  }, [autoLoad, load]);

  return {
    documents,
    loading,
    error,
    available,
    searchResults,
    load,
    upload,
    remove,
    search,
  };
}
