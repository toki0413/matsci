/**
 * useMemory — Manages all memory system state and API calls.
 *
 * Encapsulates memory CRUD, search, statistics, pruning, and MEMORY.md sync.
 * Used by the Memory panel component.
 */
import { useState } from 'react';
import { api } from '../lib/api';
import type { MemoryEntry, MemoryStats } from '../types/domain';

export function useMemory() {
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
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
  const [memoryView, setMemoryView] = useState<'browse' | 'add'>('browse');

  const loadMemory = async () => {
    try {
      const params = new URLSearchParams();
      if (memoryFilter.category) params.set('category', memoryFilter.category);
      if (memoryFilter.tier) params.set('tier', memoryFilter.tier);
      params.set('limit', '200');
      const data = await api.get<{ entries?: MemoryEntry[] }>(`/memory?${params.toString()}`);
      setMemories(data.entries || []);
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
        loadMemory();
        loadMemoryStats();
      } else {
        setMemoryMsg(`Save failed: ${data.error}`);
      }
    } catch (e: any) {
      setMemoryMsg(`Save error: ${e.message}`);
    }
  };

  const deleteMemory = async (id: string) => {
    if (!confirm('Delete this memory?')) return;
    try {
      await api.del(`/memory/${id}`);
      loadMemory();
      loadMemoryStats();
    } catch (e: any) {
      setMemoryMsg(`Delete error: ${e.message}`);
    }
  };

  const promoteMemory = async (id: string) => {
    try {
      const data = await api.post<{ success?: boolean; error?: string }>(`/memory/promote/${id}`);
      if (data.success) {
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
      } else {
        setMemoryMsg('Sync returned no path.');
      }
    } catch (e: any) {
      setMemoryMsg(`Sync error: ${e.message}`);
    }
  };

  return {
    memories, memoryStats, memorySearch, memoryFilter, memoryForm, memoryMsg, memoryView,
    setMemorySearch, setMemoryFilter, setMemoryForm, setMemoryView, setMemoryMsg,
    loadMemory, loadMemoryStats, searchMemory, createMemory, deleteMemory,
    promoteMemory, pruneMemory, syncMemoryMd,
  };
}
