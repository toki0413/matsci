import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import DiffViewer from './DiffViewer';

// Standard unified-diff fixture. parsePatch turns hunk.lines into prefixed
// entries: ' context1', '-old line', '+new line', ' context2'.
const DIFF = `diff --git a/src/foo.py b/src/foo.py
index abc123..def456 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,3 @@
 context1
-old line
+new line
 context2
`;

const FILES = [{ path: 'src/foo.py', status: 'modified' as const, diff: DIFF }];

beforeEach(() => vi.resetAllMocks());

describe('DiffViewer', () => {
  it('shows an empty-state message when there are no diffs', () => {
    render(<DiffViewer diffs={[]} />);
    expect(screen.getByText('No file changes to display')).toBeTruthy();
  });

  it('renders the file tab and inline added/removed lines', () => {
    render(<DiffViewer diffs={FILES} />);
    // tab shows the basename with its status badge
    expect(screen.getByRole('button', { name: /foo\.py/ })).toBeTruthy();
    // removed + added content are both present in inline view
    expect(screen.getByText('old line')).toBeTruthy();
    expect(screen.getByText('new line')).toBeTruthy();
    // counter reflects 1 of 1 file and the status badge shows 'M'
    expect(screen.getByText('1 / 1')).toBeTruthy();
    expect(screen.getByText('M')).toBeTruthy();
  });

  it('switches between inline and split views', () => {
    render(<DiffViewer diffs={FILES} />);
    fireEvent.click(screen.getByRole('button', { name: /Split/ }));
    // split view still surfaces both sides of the change
    expect(screen.getByText('old line')).toBeTruthy();
    expect(screen.getByText('new line')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /Inline/ }));
    expect(screen.getByText('old line')).toBeTruthy();
  });

  it('navigates between files and disables prev/next at the bounds', () => {
    const multi = [
      { path: 'src/a.py', status: 'added' as const, diff: DIFF },
      { path: 'src/b.py', status: 'deleted' as const, diff: DIFF },
    ];
    render(<DiffViewer diffs={multi} />);
    const prev = screen.getAllByTitle('Previous file')[0] as HTMLButtonElement;
    const next = screen.getAllByTitle('Next file')[0] as HTMLButtonElement;
    // we're on the first file: prev is disabled
    expect((screen.getByTitle('Previous file') as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByTitle('Next file'));
    expect((screen.getByTitle('Next file') as HTMLButtonElement).disabled).toBe(true);
    // counter reflects the current index
    expect(screen.getByText('2 / 2')).toBeTruthy();
    expect(prev).toBeTruthy();
  });

  it('fires accept/reject callbacks for the active file and globals', () => {
    const onAcceptFile = vi.fn();
    const onRejectFile = vi.fn();
    const onAcceptAll = vi.fn();
    const onRejectAll = vi.fn();
    render(
      <DiffViewer
        diffs={FILES}
        onAcceptFile={onAcceptFile}
        onRejectFile={onRejectFile}
        onAcceptAll={onAcceptAll}
        onRejectAll={onRejectAll}
      />,
    );
    // parsePatch normalizes the file path to 'src/foo.py' (dropping leading a//b/)
    fireEvent.click(screen.getByTitle(/Accept changes/));
    expect(onAcceptFile).toHaveBeenCalledWith('src/foo.py');
    fireEvent.click(screen.getByTitle(/Reject changes/));
    expect(onRejectFile).toHaveBeenCalledWith('src/foo.py');
    fireEvent.click(screen.getByTitle('Accept all changes'));
    expect(onAcceptAll).toHaveBeenCalled();
    fireEvent.click(screen.getByTitle('Reject all changes'));
    expect(onRejectAll).toHaveBeenCalled();
  });

  it('collapses far-away context into a hidden-lines gap indicator', () => {
    // >2*CONTEXT_WINDOW context lines between two changes -> the middle
    // collapses into a gap. CONTEXT_WINDOW is 3, so put 10 context lines
    // between a removal and an addition.
    const lines = [' sample0', '-remove-me', ...Array.from({ length: 10 }, (_, i) => ` middle${i}`), '+add-me'];
    const hunk = lines.map((l) => l + '\n').join('');
    const bigDiff = `diff --git a/big.py b/big.py\nindex abc123..def456 100644\n--- a/big.py\n+++ b/big.py\n@@ -1,12 +1,12 @@\n${hunk}`;
    render(<DiffViewer diffs={[{ path: 'big.py', status: 'modified' as const, diff: bigDiff }]} />);
    // the label renders as "<count> line(s) hidden"; 10 context lines with only
    // the 3 around each change shown collapses 4 of them into a gap indicator.
    expect(screen.getByText(/lines? hidden/)).toBeTruthy();
  });
});