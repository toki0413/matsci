/**
 * ResultPanel — side panel for expanded tool results (Artifacts mode).
 * ponytail: no new panel system. A tab is enough.
 * When user clicks ⤢ on a ToolResultRenderer, this tab activates with the
 * full content rendered in a larger area.
 */
import { ToolResultRenderer } from '../ToolResultRenderer';

interface ResultPanelProps {
  resultContent: string;
  resultToolName?: string;
}

export function ResultPanel({ resultContent, resultToolName }: ResultPanelProps) {
  if (!resultContent) {
    return (
      <div className="flex h-full items-center justify-center text-text-muted text-sm">
        <p>Click ⤢ on any tool result to expand it here.</p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto p-4">
      <ToolResultRenderer
        content={resultContent}
        toolName={resultToolName}
        maxRows={200}
        className="h-full"
      />
    </div>
  );
}
