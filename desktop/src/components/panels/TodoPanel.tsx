import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { api } from '../../lib/api';
import { PanelHeader } from '../settings-shared';

interface TodoItem {
  content: string;
  status: "pending" | "in_progress" | "completed";
  priority: "high" | "medium" | "low";
}

const STATUS_ORDER: Record<TodoItem["status"], number> = {
  pending: 0,
  in_progress: 1,
  completed: 2,
};

export function TodoPanel() {
  const { t } = useTranslation();
  const [todos, setTodos] = useState<TodoItem[]>([]);
  const [total, setTotal] = useState(0);
  const [completed, setCompleted] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      const res = await api.get<{ todos: TodoItem[]; total: number; completed: number }>("/v1/todos");
      setTodos([...res.todos].sort((a, b) => STATUS_ORDER[a.status] - STATUS_ORDER[b.status]));
      setTotal(res.total);
      setCompleted(res.completed);
      setError(null);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 整列表替换式 (与 todo_write_tool 一致): 切换单项状态时重发全量列表
  const replace = async (next: TodoItem[]) => {
    setTodos(next);
    try {
      const res = await api.put<{ todos: TodoItem[]; total: number; completed: number }>("/v1/todos", { todos: next });
      setTotal(res.total);
      setCompleted(res.completed);
      setError(null);
    } catch (e: any) {
      setError(e?.message ?? String(e));
      load();
    }
  };

  const toggle = (idx: number) => {
    const next = todos.map((todo, i) =>
      i === idx
        ? { ...todo, status: todo.status === "completed" ? ("pending" as const) : ("completed" as const) }
        : todo
    );
    replace(next);
  };

  const clearDone = () => replace(todos.filter((t) => t.status !== "completed"));

  const prioColor: Record<TodoItem["priority"], string> = {
    high: "bg-error/20 text-error",
    medium: "bg-warning/20 text-warning",
    low: "bg-text-muted/20 text-text-muted",
  };

  return (
    <div className="flex h-full flex-col bg-bg-tertiary text-text-primary">
      <PanelHeader title={t('todos.title')}>
        <span className="text-xs text-text-muted">
          {completed}/{total}
        </span>
        <button
          onClick={clearDone}
          disabled={completed === 0}
          className="btn-secondary px-3 py-1.5 text-xs disabled:opacity-40"
        >
          {t('todos.clearDone')}
        </button>
        <button onClick={load} className="btn-secondary px-3 py-1.5 text-xs">
          {t('todos.refresh')}
        </button>
      </PanelHeader>
      <div className="flex-1 overflow-y-auto p-3">
        {error && <div className="mb-2 rounded border border-error/30 bg-error/10 p-2 text-xs text-error">{error}</div>}
        {todos.length === 0 && !error && (
          <div className="py-8 text-center text-sm text-text-muted">{t('todos.empty')}</div>
        )}
        <ul className="space-y-1.5">
          {todos.map((todo, i) => (
            <li
              key={i}
              className={`flex items-center gap-2 rounded-lg border border-border bg-bg-secondary px-3 py-2 ${
                todo.status === "completed" ? "opacity-55" : ""
              }`}
            >
              <button
                onClick={() => toggle(i)}
                aria-label={todo.content}
                className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border text-[10px] ${
                  todo.status === "completed"
                    ? "border-accent bg-accent text-white"
                    : "border-text-muted/50 hover:border-accent"
                }`}
              >
                {todo.status === "completed" ? "✓" : ""}
              </button>
              <span
                className={`flex-1 text-sm ${
                  todo.status === "completed" ? "line-through text-text-muted" : ""
                }`}
              >
                {todo.content}
              </span>
              {todo.status === "in_progress" && (
                <span className="rounded bg-accent/20 px-1.5 py-0.5 text-[10px] text-accent">
                  {t('todos.inProgress')}
                </span>
              )}
              <span className={`rounded px-1.5 py-0.5 text-[10px] capitalize ${prioColor[todo.priority]}`}>
                {todo.priority}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
