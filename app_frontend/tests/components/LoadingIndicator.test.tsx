import { render, screen } from '@testing-library/react';
import { test, describe, expect, vi } from 'vitest';
import { LoadingIndicator } from '@/components/chat/LoadingIndicator';

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  Check: (props: Record<string, unknown>) => (
    <svg data-testid={props['data-testid'] || 'mock-icon'} className={props.className as string} />
  ),
  TriangleAlert: (props: Record<string, unknown>) => (
    <svg data-testid="mock-icon" className={props.className as string} />
  ),
  Loader2: ({ className }: { className?: string }) => (
    <svg data-testid="loader" className={className} />
  ),
}));

describe('LoadingIndicator Component', () => {
  test('renders loader when isLoading is true', () => {
    render(<LoadingIndicator isLoading={true} />);

    const loader = screen.getByTestId('loader');
    expect(loader).toBeInTheDocument();
    expect(loader).toHaveClass('animate-spin');

    // Check that the icon is not rendered
    expect(screen.queryByTestId('mock-icon')).not.toBeInTheDocument();
  });

  test('renders check icon when isLoading is false', () => {
    render(<LoadingIndicator isLoading={false} />);

    // Check that the loader is not rendered
    expect(screen.queryByTestId('loader')).not.toBeInTheDocument();

    // Check that the icon is rendered
    const icon = screen.getByTestId('data-loading-success');
    expect(icon).toBeInTheDocument();
  });
});
