import { screen } from '@testing-library/react';
import { test, describe, expect, vi } from 'vitest';
import { ErrorPanel } from '@/components/chat/ErrorPanel';
import { ICodeExecutionError } from '@/api/chat-messages/types';
import { renderWithProviders } from '../test-utils';

// react-syntax-highlighter is ESM-only and can't load in jsdom; ErrorPanel imports it
// at module load, so it must be mocked even though the code panel stays collapsed here.
vi.mock('react-syntax-highlighter', () => ({
  Prism: ({ children }: { children: string }) => (
    <pre data-testid="syntax-highlighter">{children}</pre>
  ),
}));

vi.mock('react-syntax-highlighter/dist/esm/styles/prism', () => ({
  oneDark: {},
  oneLight: {},
}));

// No digits anywhere, so any '0' in the output can only be the stray literal from the
// old `{attempts && …}` guard rendering the number 0.
const errors: ICodeExecutionError[] = [{ exception_str: 'database is unavailable' }];

describe('ErrorPanel attempts heading', () => {
  test('hides the heading (and renders no stray "0") when attempts is 0', () => {
    // A fail-fast database outage carries attempts=0. The guard must not render the
    // "Failed to generate valid code after N attempts" heading — and must not leak a
    // literal "0", which `{attempts && …}` did because `0 && x` evaluates to 0.
    const { container } = renderWithProviders(
      <ErrorPanel attempts={0} errors={errors} componentType="Analysis" />
    );

    expect(screen.queryByText(/Failed to generate valid code after/)).not.toBeInTheDocument();
    expect(container.textContent).not.toContain('0');
  });

  test('shows the heading when attempts is a positive count', () => {
    renderWithProviders(<ErrorPanel attempts={2} errors={errors} componentType="Analysis" />);

    expect(screen.getByText(/Failed to generate valid code after/)).toBeInTheDocument();
  });

  test('hides the heading when attempts is undefined', () => {
    renderWithProviders(<ErrorPanel errors={errors} componentType="Charts" />);

    expect(screen.queryByText(/Failed to generate valid code after/)).not.toBeInTheDocument();
  });
});
