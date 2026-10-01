import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';

import { ErrorBoundary } from './ErrorBoundary';

const Boom = ({ shouldThrow = true }: { shouldThrow?: boolean }) => {
  if (shouldThrow) {
    throw new Error('getComputedStyle exploded');
  }
  return <p>safe content</p>;
};

describe('ErrorBoundary', () => {
  beforeEach(() => {
    // React logs the caught error; keep the test output readable.
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders children when nothing throws', () => {
    render(
      <ErrorBoundary>
        <Boom shouldThrow={false} />
      </ErrorBoundary>
    );

    expect(screen.getByText('safe content')).toBeInTheDocument();
  });

  it('shows the fallback instead of unmounting the tree when a child throws', () => {
    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>
    );

    const fallback = screen.getByTestId('error-boundary-fallback');
    expect(fallback).toBeInTheDocument();
    // Announced rather than silently swapped in.
    expect(fallback).toHaveAttribute('role', 'alert');
    expect(screen.getByText('getComputedStyle exploded')).toBeInTheDocument();
  });

  it('recovers when the retry button is pressed', async () => {
    const user = userEvent.setup();

    const { rerender } = render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>
    );
    expect(screen.getByTestId('error-boundary-fallback')).toBeInTheDocument();

    rerender(
      <ErrorBoundary>
        <Boom shouldThrow={false} />
      </ErrorBoundary>
    );
    await user.click(screen.getByRole('button', { name: 'Try again' }));

    expect(screen.getByText('safe content')).toBeInTheDocument();
    expect(screen.queryByTestId('error-boundary-fallback')).not.toBeInTheDocument();
  });

  it('calls onError so the caller can recover, and renders a custom fallback', () => {
    const onError = vi.fn();

    render(
      <ErrorBoundary onError={onError} fallback={() => <p>custom fallback</p>}>
        <Boom />
      </ErrorBoundary>
    );

    expect(onError).toHaveBeenCalledOnce();
    expect(onError.mock.calls[0][0]).toBeInstanceOf(Error);
    expect(screen.getByText('custom fallback')).toBeInTheDocument();
    expect(screen.queryByTestId('error-boundary-fallback')).not.toBeInTheDocument();
  });
});
