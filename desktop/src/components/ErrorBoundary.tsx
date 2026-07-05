import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  // Optional custom fallback; defaults to a reset card.
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

// Keeps a render crash in one subtree from blanking the whole window.
// Wrap risky panels (3D viewers, charts) so a bad payload degrades gracefully.
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Surface to devtools/console — no telemetry from the desktop shell.
    console.error("[ErrorBoundary] render crashed:", error, info);
  }

  reset = () => this.setState({ hasError: false, error: null });

  render() {
    if (!this.state.hasError) return this.props.children;
    if (this.props.fallback) return this.props.fallback;
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-3 p-6 text-center">
        <div className="text-3xl">💥</div>
        <div className="text-sm font-semibold text-text-primary">
          This panel hit an error
        </div>
        <div className="max-w-md break-words text-xs text-text-muted">
          {this.state.error?.message || "Unexpected render error"}
        </div>
        <button onClick={this.reset} className="btn-secondary px-4 py-1.5 text-xs">
          Try again
        </button>
      </div>
    );
  }
}
