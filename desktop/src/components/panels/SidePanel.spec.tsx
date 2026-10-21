import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SidePanel } from './SidePanel';
import { api } from '../../lib/api';

vi.mock('../../lib/api', () => ({
  api: {
    get: vi.fn(),
    post: vi.fn(),
    del: vi.fn(),
  },
}));

vi.mock('../Toast', () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

const task = {
  id: 'side_1',
  thread_id: 'side_1',
  title: 'GaN k-point check',
  parent_thread_id: 'main',
  fork_message_index: 1,
  seed_count: 2,
  seed_consumed: true,
  message_count: 1,
  artifacts: [{ path: 'out/k.png', tool: 'file_write_tool' }],
  status: 'side' as const,
  is_promoted: false,
  created_at: '',
  updated_at: '',
};

const detail = {
  ...task,
  seed_messages: [
    { role: 'user', content: 'main objective' },
    { role: 'assistant', content: 'first step' },
  ],
  messages: [{ role: 'assistant', content: '4x4 is enough' }],
};

function mockListOnly() {
  (api.get as any).mockImplementation(async (path: string) => {
    if (path === '/side/tasks') return { success: true, tasks: [task], expired: [] };
    if (path === '/side/tasks/side_1') return { success: true, task: detail };
    return { success: false };
  });
}

describe('SidePanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockListOnly();
  });

  it('renders the task list from the backend', async () => {
    render(<SidePanel activeThread="main" onOpenThread={vi.fn()} />);
    expect(await screen.findByText('GaN k-point check')).toBeTruthy();
    expect(screen.getByText('2 inherited')).toBeTruthy();
    expect(screen.getByText('1 artifacts')).toBeTruthy();
  });

  it('opens a task detail with inherited context and artifacts', async () => {
    render(<SidePanel activeThread="main" onOpenThread={vi.fn()} />);
    fireEvent.click(await screen.findByText('GaN k-point check'));
    expect(await screen.findByText(/Inherited context \(2 messages/)).toBeTruthy();
    expect(screen.getByText('4x4 is enough')).toBeTruthy();
    expect(screen.getByText('out/k.png')).toBeTruthy();
  });

  it('promotes a side task and opens the promoted thread', async () => {
    const onOpenThread = vi.fn();
    (api.post as any).mockResolvedValue({
      success: true,
      thread_id: 'side_1',
      task_card: { task_id: 'side_1' },
      task: { ...detail, is_promoted: true, status: 'promoted' },
    });
    render(<SidePanel activeThread="main" onOpenThread={onOpenThread} />);
    fireEvent.click(await screen.findByText('GaN k-point check'));
    fireEvent.click(await screen.findByText('Promote'));
    await waitFor(() => expect(onOpenThread).toHaveBeenCalledWith('side_1'));
  });

  it('discards a side task after confirmation of the API contract', async () => {
    (api.del as any).mockResolvedValue({ success: true });
    render(<SidePanel activeThread="main" onOpenThread={vi.fn()} />);
    fireEvent.click(await screen.findByText('GaN k-point check'));
    fireEvent.click(await screen.findByText('Discard'));
    await waitFor(() => expect(api.del).toHaveBeenCalledWith('/side/tasks/side_1'));
  });

  it('dock mode shows a back button and close affordance', async () => {
    const onClose = vi.fn();
    render(
      <SidePanel
        activeThread="main"
        variant="dock"
        onClose={onClose}
        onOpenThread={vi.fn()}
      />,
    );
    fireEvent.click(await screen.findByText('GaN k-point check'));
    expect(await screen.findByLabelText('Back to side task list')).toBeTruthy();
    fireEvent.click(screen.getByLabelText('Close side tasks panel'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
