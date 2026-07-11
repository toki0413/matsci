import { lazy, Suspense } from 'react';
import { useTranslation } from 'react-i18next';
import { PanelHeader } from '../settings-shared';
import type { DiffEntry, Checkpoint } from '../../types/domain';

const DiffViewer = lazy(() => import('../DiffViewer'));

function LoadingFallback() {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-accent border-t-transparent" />
    </div>
  );
}

interface ReviewPanelProps {
  cwd: string;
  checkpoints: Checkpoint[];
  activeCp: string | null;
  diffs: DiffEntry[];
  createCheckpoint: () => void;
  loadDiffs: (cpId: string) => void;
  acceptCheckpoint: (cpId: string) => void;
  rejectCheckpoint: (cpId: string) => void;
}

export function ReviewPanel({
  cwd, checkpoints, activeCp, diffs,
  createCheckpoint, loadDiffs, acceptCheckpoint, rejectCheckpoint,
}: ReviewPanelProps) {
  const { t } = useTranslation();
  return (
    <div className="flex h-full">
      {/* Checkpoint list */}
      <aside className="flex w-72 flex-col border-r border-border bg-bg-secondary">
        <PanelHeader title={t('review.checkpoints')}>
          <button
            onClick={createCheckpoint}
            disabled={!cwd}
            className="btn-primary px-3 py-1.5 text-xs"
          >
            {t('review.new')}
          </button>
        </PanelHeader>
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {checkpoints.length === 0 && (
            <div className="text-xs text-text-muted">
              {t('review.createHint')}
            </div>
          )}
          {checkpoints.map((cp) => (
            <div
              key={cp.id}
              onClick={() => loadDiffs(cp.id)}
              className={`cursor-pointer rounded-lg border border-border p-3 transition-colors ${
                activeCp === cp.id
                  ? "bg-accent/10 border-accent"
                  : "bg-bg-tertiary hover:bg-bg-primary"
              }`}
            >
              <div className="text-xs font-semibold text-accent">{cp.id}</div>
              <div className="mt-1 truncate text-xs text-text-muted">{cp.base}</div>
              <div className="mt-1 text-xs text-text-secondary">{cp.files} {t('review.files')}</div>
            </div>
          ))}
        </div>
      </aside>

      {/* Diff viewer */}
      <div className="flex flex-1 flex-col bg-bg-primary">
        <PanelHeader title={activeCp ? `${t('review.checkpoint')} ${activeCp}` : t('review.review')} />

        <div className="flex flex-1 overflow-hidden">
          {activeCp && diffs.length > 0 ? (
            <Suspense fallback={<LoadingFallback />}>
              <DiffViewer
                diffs={diffs}
                onAcceptAll={() => acceptCheckpoint(activeCp)}
                onRejectAll={() => rejectCheckpoint(activeCp)}
              />
            </Suspense>
          ) : activeCp ? (
            <div className="flex h-full items-center justify-center text-sm text-text-muted">
              {t('review.noChanges')}
            </div>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-text-muted">
              {t('review.selectHint')}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
