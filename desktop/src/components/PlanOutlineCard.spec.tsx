import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import PlanOutlineCard from './PlanOutlineCard';

describe('PlanOutlineCard', () => {
  it('renders a numbered outline checklist from plan steps', () => {
    render(
      <PlanOutlineCard
        steps={[
          { name: 'Survey literature', description: 'Gather prior band-gap values' },
          { name: 'Run DFT', description: 'PBE relaxation + static' },
        ]}
      />,
    );
    expect(screen.getByText('Survey literature')).toBeTruthy();
    expect(screen.getByText('Run DFT')).toBeTruthy();
    expect(screen.getByText('1')).toBeTruthy();
    expect(screen.getByText('2')).toBeTruthy();
  });

  it('shows acceptance criteria and suggested tools', () => {
    render(
      <PlanOutlineCard
        steps={[{ name: 'Run DFT' }]}
        criteria={['Converged total energy']}
        tools={['vasp_tool', 'bash_tool']}
      />,
    );
    expect(screen.getByText(/Converged total energy/)).toBeTruthy();
    expect(screen.getByText(/vasp_tool/)).toBeTruthy();
    expect(screen.getByText(/bash_tool/)).toBeTruthy();
  });

  it('reflects executing status and hides the revise affordance once running', () => {
    render(
      <PlanOutlineCard
        steps={[{ name: 'Run DFT' }]}
        status="executing"
        confirmed={false}
        onRevise={vi.fn()}
      />,
    );
    expect(screen.getByText(/Executing…/)).toBeTruthy();
    expect(screen.queryByText(/Revise outline/)).toBeNull();
  });

  it('calls onRevise when the user asks to revise the outline', () => {
    const onRevise = vi.fn();
    render(
      <PlanOutlineCard steps={[{ name: 'Run DFT' }]} onRevise={onRevise} />,
    );
    fireEvent.click(screen.getByText(/Revise outline/));
    expect(onRevise).toHaveBeenCalledTimes(1);
  });

  it('shows the empty state when no steps are provided', () => {
    render(<PlanOutlineCard />);
    expect(screen.getByText(/No steps in this plan yet/)).toBeTruthy();
  });
});