import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  // Optional custom fallback; defaults to a reset card.
  fallback?: ReactNode;
  // Optional panel name for better error context
  name?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  showDetails: boolean;
}

// Keeps a render crash in one subtree from blanking the whole window.
// Wrap risky panels (3D viewers, charts) so a bad payload degrades gracefully.
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null, errorInfo: null, showDetails: false };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: false, error, errorInfo: null, showDetails: false };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Surface to devtools/console — no telemetry from the desktop shell.
    console.error("[ErrorBoundary] render crashed:", error, info);
    this.setState({ hasError: true, error, errorInfo: info });
  }

  reset = () => this.setState({ hasError: false, error: null, errorInfo: null, showDetails: false });

  copyError = () => {
    const text = `Error: ${this.state.error?.message}\nStack: ${this.state.error?.stack}\nComponent: ${this.state.errorInfo?.componentStack}`;
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.querySelector<HTMLButtonElement>('[data-error-copy-btn]');
      if (btn) {
        const orig = btn.textContent;
        btn.textContent = '✓ Copied';
        setTimeout(() => { btn.textContent = orig; }, 1500);
      }
    });
  };

  render() {
    if (!this.state.hasError) return this.props.children;
    if (this.props.fallback) return this.props.fallback;
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-3 p-6 text-center">
        <div className="text-3xl">💥</div>
        <div className="text-sm font-semibold text-text-primary">
          {this.props.name ? `${this.props.name} crashed` : "This panel hit an error"}
        </div>
        <div className="max-w-md break-words text-xs text-text-muted">
          {this.state.error?.message || "Unexpected render error"}
        </div>
        <div className="flex gap-2">
          <button onClick={this.reset} className="btn-secondary px-4 py-1.5 text-xs">
            Try again
          </button>
          <button
            onClick={() => this.setState(prev => ({ showDetails: !prev.showDetails }))}
            className="btn-secondary px-4 py-1.5 text-xs"
          >
            {this.state.showDetails ? 'Hide' : 'Details'}
          </button>
          <button
            data-error-copy-btn
            onClick={this.copyError}
            className="btn-secondary px-4 py-1.5 text-xs"
          >
            Copy error
          </button>
        </div>
        {this.state.showDetails && this.state.error && (
          <pre className="mt-2 max-h-48 w-full max-w-2xl overflow-auto rounded-lg bg-bg-tertiary p-3 text-left text-[11px] text-text-muted">
            {this.state.error.stack}
            {this.state.errorInfo?.componentStack && (
              '\n\nComponent stack:' + this.state.errorInfo.componentStack
            )}
          </pre>
        )}
      </div>
    );
  }
}
