import { render, screen } from '@testing-library/react';
import { test, describe, expect, vi } from 'vitest';
import { LoadingIndicator } from '@/components/chat/LoadingIndicator';

// Mock the FontAwesomeIcon component
vi.mock('@fortawesome/react-fontawesome', () => ({
  FontAwesomeIcon: () => <div data-testid="mock-icon" />,
}));

// Mock the Loader2 component from lucide-react
vi.mock('lucide-react', () => ({
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
    const icon = screen.getByTestId('mock-icon');
    expect(icon).toBeInTheDocument();
  });
});
