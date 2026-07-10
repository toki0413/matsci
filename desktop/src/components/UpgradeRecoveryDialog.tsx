import { useState } from 'react';
import { AlertCircle, RefreshCw, Download, X } from 'lucide-react';

/**
 * UpgradeRecoveryDialog — shown when a desktop app update fails.
 * Inspired by AstrBot's UpgradeRecoveryDialog component.
 *
 * Provides recovery options: retry update, download manually, or skip.
 */
export default function UpgradeRecoveryDialog({
  error,
  version,
  onRetry,
  onSkip,
}: {
  error: string;
  version?: string;
  onRetry: () => void;
  onSkip: () => void;
}) {
  const [retrying, setRetrying] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) return null;

  const handleRetry = () => {
    setRetrying(true);
    onRetry();
    setTimeout(() => setRetrying(false), 3000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-text-primary/40 backdrop-blur-sm p-4">
      <div className="w-full max-w-lg rounded-2xl border border-border bg-bg-secondary p-6 shadow-2xl">
        <div className="flex items-start gap-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-error/10">
            <AlertCircle size={24} className="text-error" />
          </div>
          <div className="min-w-0 flex-1">
            <h2 className="text-lg font-bold text-text-primary">
              Update Failed{version ? ` — v${version}` : ''}
            </h2>
            <p className="mt-1 text-sm text-text-secondary">
              The application encountered an error while updating. Your current version is still intact and functional.
            </p>
            <div className="mt-3 rounded-lg border border-error/20 bg-error/5 p-3">
              <p className="text-xs text-error break-words">{error}</p>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <button
                onClick={handleRetry}
                disabled={retrying}
                className="btn-primary flex items-center gap-2 px-4 py-2 text-sm"
              >
                {retrying ? (
                  <><RefreshCw size={14} className="animate-spin" /> Retrying…</>
                ) : (
                  <><RefreshCw size={14} /> Retry Update</>
                )}
              </button>
              <a
                href="https://github.com/toki0413/matsci/releases"
                target="_blank"
                rel="noopener noreferrer"
                className="btn-secondary flex items-center gap-2 px-4 py-2 text-sm"
              >
                <Download size={14} /> Download Manually
              </a>
              <button
                onClick={() => { setDismissed(true); onSkip(); }}
                className="flex items-center gap-1 px-3 py-2 text-sm text-text-muted hover:text-text-primary"
              >
                <X size={14} /> Skip for now
              </button>
            </div>

            <div className="mt-4 border-t border-border pt-3">
              <p className="text-xs text-text-muted">
                Recovery tips: Check your network connection, ensure sufficient disk space,
                and verify the download isn't blocked by antivirus software.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
