/**
 * useMemory — Manages all memory system state and API calls.
 *
 * Encapsulates memory CRUD, search, statistics, pruning, and MEMORY.md sync.
 * Used by the Memory panel component.
 */
import { useState } from 'react';
import { api } from '../lib/api';
import { toast } from '../components/Toast';
import type { MemoryEntry, MemoryLayers, MemoryStats } from '../types/domain';

export function useMemory() {
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
  const [memoriesLoading, setMemoriesLoading] = useState(true);
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null);
  const [memorySearch, setMemorySearch] = useState('');
  const [memoryFilter, setMemoryFilter] = useState<{ category: string; tier: string }>({ category: '', tier: '' });
  const [memoryForm, setMemoryForm] = useState<{ content: string; category: string; tags: string; importance: number; tier: string }>({
    content: '',
    category: 'fact',
    tags: '',
    importance: 0.5,
    tier: 'mid',
  });
  const [memoryMsg, setMemoryMsg] = useState('');
  const [memoryView, setMemoryView] = useState<'browse' | 'add' | 'layers'>('browse');
  const [memoryHasMore, setMemoryHasMore] = useState(false);
  const [memoryLayers, setMemoryLayers] = useState<MemoryLayers | null>(null);
  const [memoryLayersLoading, setMemoryLayersLoading] = useState(false);

  const loadMemory = async (loadMore = false) => {
    try {
      const params = new URLSearchParams();
      if (memoryFilter.category) params.set('category', memoryFilter.category);
      if (memoryFilter.tier) params.set('tier', memoryFilter.tier);
      const limit = loadMore ? memories.length + 100 : 100;
      params.set('limit', String(limit));
      const data = await api.get<{ entries?: MemoryEntry[]; total?: number }>(`/memory?${params.toString()}`);
      const newEntries = data.entries || [];
      if (loadMore) {
        setMemories(prev => [...prev, ...newEntries.slice(prev.length)]);
      } else {
        setMemories(newEntries);
      }
      setMemoryHasMore((data.total ?? 0) > newEntries.length);
      setMemoriesLoading(false);
    } catch (e: any) {
      setMemoryMsg(`Load failed: ${e.message}`);
    }
  };

  const loadMemoryStats = async () => {
    try {
      const data = await api.get<MemoryStats>('/memory/stats');
      setMemoryStats(data);
    } catch {
      setMemoryStats(null);
    }
  };

  const searchMemory = async () => {
    if (!memorySearch.trim()) {
      loadMemory();
      return;
    }
    setMemoryMsg('Searching…');
    try {
      const data = await api.post<{ results?: MemoryEntry[] } & Record<string, any>>(
        '/memory/search',
        { query: memorySearch, top_k: 10 }
      );
      setMemories(data.results || []);
      setMemoryMsg(data.results?.length ? `Found ${data.results.length} results` : 'No results');
    } catch (e: any) {
      setMemoryMsg(`Search error: ${e.message}`);
    }
  };

  const createMemory = async () => {
    setMemoryMsg('Saving…');
    try {
      const data = await api.post<{ success?: boolean; error?: string }>(
        '/memory',
        {
          content: memoryForm.content,
          category: memoryForm.category,
          tags: memoryForm.tags.split(',').map((t) => t.trim()).filter(Boolean),
          importance: memoryForm.importance,
          tier: memoryForm.tier,
        }
      );
      if (data.success) {
        setMemoryForm({ content: '', category: 'fact', tags: '', importance: 0.5, tier: 'mid' });
        setMemoryMsg('Memory saved.');
        toast.success('Memory saved');
        loadMemory();
        loadMemoryStats();
      } else {
        setMemoryMsg(`Save failed: ${data.error}`);
        toast.error(`Save failed: ${data.error}`);
      }
    } catch (e: any) {
      setMemoryMsg(`Save error: ${e.message}`);
    }
  };

  const deleteMemory = async (id: string) => {
    if (!confirm('Delete this memory?')) return;
    try {
      await api.del(`/memory/${id}`);
      toast.success('Memory deleted');
      loadMemory();
      loadMemoryStats();
    } catch (e: any) {
      setMemoryMsg(`Delete error: ${e.message}`);
      toast.error(`Delete error: ${e.message}`);
    }
  };

  const updateMemory = async (id: string, patch: { content?: string; importance?: number; tags?: string[] }) => {
    try {
      const data = await api.patch<{ success?: boolean; error?: string }>(`/memory/${id}`, patch);
      if (data.success) {
        toast.success('Memory updated');
        loadMemory();
      } else {
        setMemoryMsg(`Update failed: ${data.error}`);
        toast.error(`Update failed: ${data.error}`);
      }
    } catch (e: any) {
      setMemoryMsg(`Update error: ${e.message}`);
      toast.error(`Update error: ${e.message}`);
    }
  };

  const promoteMemory = async (id: string) => {
    try {
      const data = await api.post<{ success?: boolean; error?: string }>(`/memory/promote/${id}`);
      if (data.success) {
        toast.success('Promoted to long-term');
        loadMemory();
        loadMemoryStats();
      } else {
        setMemoryMsg(`Promote failed: ${data.error}`);
      }
    } catch (e: any) {
      setMemoryMsg(`Promote error: ${e.message}`);
    }
  };

  const pruneMemory = async () => {
    if (!confirm('Prune expired and low-importance memories?')) return;
    try {
      const data = await api.post<{ expired?: number; low_importance?: number }>('/memory/prune');
      setMemoryMsg(`Pruned ${data.expired ?? 0} expired, ${data.low_importance ?? 0} low-importance.`);
      toast.success(`Pruned ${data.expired ?? 0} expired, ${data.low_importance ?? 0} low-importance`);
      loadMemory();
      loadMemoryStats();
    } catch (e: any) {
      setMemoryMsg(`Prune error: ${e.message}`);
    }
  };

  const syncMemoryMd = async () => {
    setMemoryMsg('Syncing MEMORY.md…');
    try {
      const data = await api.post<{ path?: string }>('/memory/sync-md');
      if (data.path) {
        setMemoryMsg(`Synced to ${data.path}`);
        toast.success(`Synced to ${data.path}`);
      } else {
        setMemoryMsg('Sync returned no path.');
      }
    } catch (e: any) {
      setMemoryMsg(`Sync error: ${e.message}`);
    }
  };

  // 拉 4 层 memory 聚合状态. 后端每层独立 try/except, 单层失败返回 available=false.
  const loadMemoryLayers = async () => {
    setMemoryLayersLoading(true);
    try {
      const data = await api.get<MemoryLayers>('/memory/layers');
      setMemoryLayers(data);
    } catch (e: any) {
      setMemoryMsg(`Layers load error: ${e.message}`);
    } finally {
      setMemoryLayersLoading(false);
    }
  };

  return {
    memories, memoriesLoading, memoryHasMore, memoryStats, memorySearch, memoryFilter, memoryForm, memoryMsg, memoryView,
    memoryLayers, memoryLayersLoading,
    setMemorySearch, setMemoryFilter, setMemoryForm, setMemoryView, setMemoryMsg,
    loadMemory, loadMemoryStats, searchMemory, createMemory, deleteMemory,
    updateMemory, promoteMemory, pruneMemory, syncMemoryMd, loadMemoryLayers,
  };
}
