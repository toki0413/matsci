/**
 * CodeMirror 6 封装 — 单文件编辑区（语法高亮 + 撤销/重做 + 查找替换）。
 *
 * loadDir/openFile 那套文件 I/O 仍在后端 (ADR-0001)；此组件只负责「编辑」层：
 * 把受控的 value + onChange 接到一个带高亮的编辑器实例上。多 Tab 切换由上层
 * 通过在每次激活时改变五个关键 props 触发本组件的 key 重建来实现。
 *
 * ponytail: 需要按扩展名选择语言，未匹配的回落成纯文本；没有逐行 diff/断点
 * 这类 IDE 能力。升级路径：接入 Language Server / 分色 diff 视图。
 */

import { useEffect, useRef } from 'react';
import { EditorView, basicSetup } from 'codemirror';
import { EditorState, Compartment, type Extension } from '@codemirror/state';
import { keymap } from '@codemirror/view';
import { indentWithTab } from '@codemirror/commands';
import { oneDark } from '@codemirror/theme-one-dark';
import { python } from '@codemirror/lang-python';
import { javascript } from '@codemirror/lang-javascript';
import { json } from '@codemirror/lang-json';
import { cpp } from '@codemirror/lang-cpp';
import { markdown } from '@codemirror/lang-markdown';

// 扩展名 -> language pool（避免每个文件都摊几十 KB 的语言包）
const LANG_GUESSERS: Array<[RegExp, Extension]> = [
  [/\.(py|pyw|ipynb)$/i, python()],
  [/\.(ts|tsx|js|jsx|mjs|cjs)$/i, javascript()],
  [/\.json$/i, json()],
  [/\.(c|h|cpp|hpp|cc|hh|cu|cuh)$/i, cpp()],
  [/\.(md|markdown)$/i, markdown()],
];

function guessLang(file: string): Extension[] {
  for (const [re, lang] of LANG_GUESSERS) {
    if (re.test(file)) return [lang];
  }
  return [];
}

interface CodeMirrorEditorProps {
  path: string;
  value: string;
  onChange: (path: string, value: string) => void;
  onCursor?: (line: number, col: number) => void; // 状态栏光标位置显示
  readonlyKey?: string; // 切换 key 时重建编辑器，保证与当前 tab 绑定
}

export function CodeMirrorEditor({ path, value, onChange, onCursor, readonlyKey }: CodeMirrorEditorProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const onCursorRef = useRef(onCursor);
  onCursorRef.current = onCursor;

  // 每次 path 变化都重建编辑器：绑定新语言 + 新内容，避免跨 tab 复用脏快照。
  // readonlyKey 由上层额外控制（例如切换文件时由渲染方改变 key 强制重置光标）。
  const buildKey = `${path}::${readonlyKey ?? ''}`;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const langCompartment = new Compartment();
    const state = EditorState.create({
      doc: value,
      extensions: [
        basicSetup,
        oneDark,
        keymap.of([indentWithTab]),
        langCompartment.of(guessLang(path)),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) {
            onChangeRef.current(path, update.state.doc.toString());
          }
          // 光标移动（含输入导致的位置变动）时上报当前 行:列
          if (update.docChanged || update.selectionSet) {
            const pos = update.state.selection.main.head;
            const line = update.state.doc.lineAt(pos);
            onCursorRef.current?.(line.number, pos - line.from);
          }
        }),
      ],
    });

    const view = new EditorView({ state, parent: host });
    viewRef.current = view;

    return () => {
      view.destroy();
      viewRef.current = null;
    };
  }, [buildKey]); // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={hostRef} className="h-full overflow-auto" data-cm-path={path} />;
}