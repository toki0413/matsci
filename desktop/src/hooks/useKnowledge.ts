/**
 * useKnowledge — Manages knowledge base state and API calls.
 *
 * Encapsulates document upload, deep PDF parsing (6-stage pipeline),
 * knowledge querying, and document management.
 */
import { useState, useRef, useCallback } from 'react';
import { api, authHeaders } from '../lib/api';
import { getApiBase } from '../lib/api-client';
import { toast } from '../components/Toast';
import type { KbDoc, DocumentParseResult, DocumentGraph } from '../types/domain';

// to build /knowledge/{id}/raw links for inline preview + download
const rawUrl = (docId: string) => `${getApiBase()}/knowledge/${docId}/raw`;

export function useKnowledge() {
  const [kbDocs, setKbDocs] = useState<KbDoc[]>([]);
  const [kbAvailable, setKbAvailable] = useState(false);
  const [kbLoading, setKbLoading] = useState(true);
  const [kbMsg, setKbMsg] = useState('');
  const [kbQuery, setKbQuery] = useState('');
  const [kbChunks, setKbChunks] = useState<any[]>([]);
  const [parseLoading, setParseLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const parseFileInputRef = useRef<HTMLInputElement>(null);
  // 全文预览: 点左侧文件名 -> 拉该文档的分块内容在右侧顺序展示
  const [viewingDoc, setViewingDoc] = useState<{ doc_id: string; filename: string } | null>(null);
  const [docChunks, setDocChunks] = useState<any[] | null>(null);
  const [docLoading, setDocLoading] = useState(false);

  // 图片画廊: 点图片 tab 后拉 /knowledge/{id}/images
  const [docImages, setDocImages] = useState<any[] | null>(null);
  const [imagesLoading, setImagesLoading] = useState(false);

  // 报告生成: loading / 内容 / 错误
  const [reportLoading, setReportLoading] = useState(false);
  const [reportContent, setReportContent] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  const loadKnowledge = async () => {
    try {
      const data = await api.get<{ documents?: any[]; available?: any }>('/knowledge');
      setKbDocs(data.documents || []);
      setKbAvailable(data.available);
      setKbLoading(false);
    } catch (e: any) {
      setKbMsg(`Failed to load knowledge base: ${e.message}`);
    }
  };

  const [uploadPct, setUploadPct] = useState(0);

  const uploadKnowledge = async (file: File) => {
    setKbMsg('上传中…');
    setUploadPct(0);
    try {
      const data = await api.uploadWithProgress<{ success?: boolean; error?: string; document?: { chunks: number; doc_id?: string; filename?: string } }>(
        '/knowledge/upload',
        file,
        (loaded, total) => setUploadPct(Math.round((loaded / total) * 100)),
      );
      if (data.success) {
        const n = data.document?.chunks ?? 0;
        setUploadPct(100);
        setTimeout(() => setUploadPct(0), 2000);
        loadKnowledge();
        // 反馈闭环: 索引完成后做一次自检检索, 确认"能搜到"再告诉用户, 而不是只报索引条数.
        setKbMsg(`已索引 ${n} 条，正在验证可检索性…`);
        selfCheckSearch(file, data.document);
      } else {
        setKbMsg(`上传失败: ${data.error}`);
        toast.error(`上传失败: ${data.error}`);
        setUploadPct(0);
      }
    } catch (e: any) {
      setKbMsg(`上传出错: ${e.message}`);
      toast.error(`上传出错: ${e.message}`);
      setUploadPct(0);
    }
  };

  // 用文件名去掉扩展名当 query 跑一次检索, 命中刚上传的 doc 才算"可立即检索".
  // src 已落在 /tmp 之外, 这里不会触发沙箱目录检查; 失败也不抛, 只降级提示.
  const selfCheckSearch = async (file: File, doc?: { doc_id?: string; filename?: string; chunks?: number }) => {
    const query = file.name.replace(/\.[^.]+$/, '').replace(/[_\-]+/g, ' ').slice(0, 40);
    if (!query.trim()) {
      setKbMsg(`已索引 ${doc?.chunks ?? 0} 条，可在右侧搜索验证`);
      return;
    }
    try {
      const check = await api.post<{ chunks?: any[] } & Record<string, any>>('/knowledge/query', { query, top_k: 3 });
      const hit = (check?.chunks || []).some((c: any) =>
        (c.metadata && c.metadata.doc_id && doc && c.metadata.doc_id === doc.doc_id) ||
        String(c.metadata?.filename) === (doc?.filename || file.name)
      );
      setKbMsg(hit
        ? `✅ 已索引 ${doc?.chunks ?? 0} 条，检索验证通过`
        : `已索引 ${doc?.chunks ?? 0} 条，可在右侧输入关键词验证`);
    } catch {
      setKbMsg(`已索引 ${doc?.chunks ?? 0} 条，可在右侧搜索验证`);
    }
  };

  const uploadKnowledgeMany = async (files: FileList | File[]) => {
    for (const f of Array.from(files)) {
      await uploadKnowledge(f);
    }
  };

  const parseDocument = async (file: File) => {
    setParseLoading(true);
    setKbMsg('Parsing document (6-stage pipeline)…');
    setUploadPct(5);
    try {
      const d = await api.uploadStream<DocumentParseResult>('/document/parse', file, (ev) => {
        // 后端每个 M 阶段推一条 stage 事件; 用它的 pct 当服务端进度展示.
        if (ev.type === 'stage') {
          setUploadPct(typeof ev.pct === 'number' ? ev.pct : 5);
          setKbMsg(typeof ev.message === 'string' ? `解析中: ${ev.message}…` : '解析中…');
        }
      });
      setKbMsg(
        `✅ Parsed: ${d.n_packages ?? 0} info packages, ` +
        `${d.stats?.n_nodes ?? 0} graph nodes, ` +
        `${d.stats?.n_edges ?? 0} edges`
      );
      loadKnowledge();
    } catch (e) {
      setKbMsg(`Parse error: ${(e as Error).message}`);
      setParseLoading(false);
      setUploadPct(0);
      return;
    } finally {
      setParseLoading(false);
      setUploadPct(0);
    }
  };

  const loadDocumentGraph = useCallback(async (docId: string) => {
    try {
      const data = await api.get<DocumentGraph>(`/document/${docId}/graph`);
      setKbMsg(
        `📊 知识图谱: ${data.nodes?.length || 0} 个节点, ` +
        `${data.edges?.length || 0} 条边`
      );
      return data;
    } catch { /* ignore */ }
    return null;
  }, []);

  // 文档结构图谱: 📊 按钮把图数据结构存这里, 面板用 SVG 渲染, 而非只弹一行文字.
  const [docGraph, setDocGraph] = useState<DocumentGraph | null>(null);
  const viewDocGraph = useCallback(async (docId: string) => {
    const g = await loadDocumentGraph(docId);
    setDocGraph(g);
  }, [loadDocumentGraph]);

  const loadDocumentContent = useCallback(async (doc: { doc_id: string; filename: string }) => {
    setViewingDoc(doc);
    setDocLoading(true);
    setKbMsg('Loading document content…');
    try {
      const data = await api.get<{ chunks?: any[]; error?: string }>(`/knowledge/${doc.doc_id}/chunks`);
      setDocChunks(data.chunks || []);
      setKbMsg(
        data.chunks?.length
          ? `📄 ${doc.filename} — ${data.chunks.length} chunks`
          : (data.error || `No content for ${doc.filename}`)
      );
    } catch (e: any) {
      setKbMsg(`Failed to load content: ${e.message}`);
      setDocChunks([]);
    } finally {
      setDocLoading(false);
    }
  }, []);

  const clearDocView = useCallback(() => {
    setViewingDoc(null);
    setDocChunks(null);
    setDocImages(null);
    setReportContent(null);
    setReportError(null);
    setReportLoading(false);
  }, []);

  // 图片画廊: 拉缺失且未加载时获取
  const loadDocImages = useCallback(async (docId: string) => {
    setImagesLoading(true);
    try {
      const data = await api.get<{ images?: any[] } & Record<string, any>>(`/knowledge/${docId}/images`);
      setDocImages(data.images || []);
    } catch {
      setDocImages([]);
    } finally {
      setImagesLoading(false);
    }
  }, []);

  // 生成综合报告: 把当前文档内容喂给 lead agent 合成, 结果留在 reportContent
  const generateReport = useCallback(async (docId: string, title: string) => {
    setReportLoading(true);
    setReportError(null);
    setReportContent(null);
    try {
      const data = await api.post<{ success?: boolean; report?: string; error?: string }>(
        '/knowledge/report',
        { doc_ids: [docId], title },
      );
      if (data.success) {
        setReportContent(data.report || '（报告为空）');
      } else {
        setReportError(data.error || '报告生成失败');
      }
    } catch (e: any) {
      setReportError(`报告生成失败: ${e.message}`);
    } finally {
      setReportLoading(false);
    }
  }, []);

  // 原文预览: 带鉴权拉取原始文本 (仅文本类文件), 失败返回空串
  const fetchRawText = useCallback(async (docId: string) => {
    try {
      const resp = await fetch(rawUrl(docId), { headers: authHeaders() });
      if (!resp.ok) return '';
      return await resp.text();
    } catch {
      return '';
    }
  }, []);

  // 原文下载: 带鉴权拉 blob 触发下载, 绕开 <a download> 不带头导致的 401
  const downloadRaw = useCallback(async (docId: string, filename: string) => {
    try {
      const resp = await fetch(rawUrl(docId), { headers: authHeaders() });
      if (!resp.ok) return false;
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      return true;
    } catch {
      return false;
    }
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
    kbDocs, kbAvailable, kbLoading, kbMsg, kbQuery, kbChunks, parseLoading, uploadPct,
    fileInputRef, parseFileInputRef,
    setKbQuery, setKbMsg,
    loadKnowledge, uploadKnowledge, uploadKnowledgeMany, parseDocument, loadDocumentGraph,
    deleteKnowledge, queryKnowledge, ingestUrl, loadProvenanceDag,
    viewingDoc, docChunks, docLoading, loadDocumentContent, clearDocView,
    docGraph, viewDocGraph, clearDocGraph: () => setDocGraph(null),
    docImages, imagesLoading, reportLoading, reportContent, reportError,
    loadDocImages, generateReport, fetchRawText, downloadRaw,
  };
}
