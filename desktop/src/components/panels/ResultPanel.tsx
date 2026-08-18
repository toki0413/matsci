/**
 * ResultPanel — side panel for expanded tool results (Artifacts mode).
 * ponytail: no new panel system. A tab is enough.
 * When user clicks ⤢ on a ToolResultRenderer, this tab activates with the
 * full content rendered in a larger area.
 */
import { useTranslation } from 'react-i18next';
import { ToolResultRenderer } from '../ToolResultRenderer';
import { downloadText } from '../../lib/download';

interface ResultPanelProps {
  resultContent: string;
  resultToolName?: string;
}

export function ResultPanel({ resultContent, resultToolName }: ResultPanelProps) {
  const { t } = useTranslation();
  if (!resultContent) {
    return (
      <div className="flex h-full items-center justify-center text-text-muted text-sm">
        <p>{t('result.hint')}</p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden p-4">
      <div className="mb-2 flex items-center justify-end">
        <button
          onClick={() => downloadText(resultContent, `result-${Date.now()}.txt`)}
          className="btn-secondary px-3 py-1 text-xs"
          title={t('result.save') || 'Save'}
        >
          {t('result.save') || 'Save'}
        </button>
      </div>
      <div className="h-full min-h-0 overflow-auto">
        <ToolResultRenderer
          content={resultContent}
          toolName={resultToolName}
          maxRows={200}
          className="h-full"
        />
      </div>
    </div>
  );
}
