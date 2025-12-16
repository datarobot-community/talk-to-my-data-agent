import { render, screen } from '@testing-library/react';
import { test, describe, expect, vi } from 'vitest';
import userEvent from '@testing-library/user-event';
import { Button } from '@/components/ui/button';

describe('Button Component', () => {
  test('renders button with default props', () => {
    render(<Button>Click me</Button>);
    const button = screen.getByRole('button', { name: /click me/i });

    expect(button).toBeInTheDocument();
    expect(button).toHaveClass('bg-primary');
  });

  test('renders different variants correctly', () => {
    const { rerender } = render(<Button variant="destructive">Destructive</Button>);
    const button = screen.getByRole('button', { name: /destructive/i });

    expect(button).toHaveClass('bg-destructive');

    rerender(<Button variant="secondary">Secondary</Button>);
    expect(screen.getByRole('button', { name: /secondary/i })).toHaveClass('bg-background');

    rerender(<Button variant="ghost">Ghost</Button>);
    expect(screen.getByRole('button', { name: /ghost/i })).toHaveClass('hover:bg-input');

    rerender(<Button variant="link">Link</Button>);
    expect(screen.getByRole('button', { name: /link/i })).toHaveClass('text-primary');
  });

  test('renders different sizes correctly', () => {
    const { rerender } = render(<Button size="sm">Small</Button>);
    const button = screen.getByRole('button', { name: /small/i });

    expect(button).toHaveClass('h-8');

    rerender(<Button size="default">Default</Button>);
    expect(screen.getByRole('button', { name: /default/i })).toHaveClass('h-9');

    rerender(<Button size="lg">Large</Button>);
    expect(screen.getByRole('button', { name: /large/i })).toHaveClass('h-10');

    rerender(<Button size="icon">Icon</Button>);
    expect(screen.getByRole('button', { name: /icon/i })).toHaveClass('h-9 w-9');
  });

  test('applies custom className', () => {
    render(<Button className="test-class">Custom Class</Button>);
    const button = screen.getByRole('button', { name: /custom class/i });

    expect(button).toHaveClass('test-class');
  });

  test('passes additional props to the button element', async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    // Test disabled button
    const { unmount } = render(
      <Button onClick={handleClick} disabled>
        Click me
      </Button>
    );
    const button = screen.getByRole('button', { name: /click me/i });

    expect(button).toBeDisabled();

    await user.click(button);
    expect(handleClick).not.toHaveBeenCalled();

    // Clean up first render
    unmount();

    // Test enabled button
    render(<Button onClick={handleClick}>Click me</Button>);
    const enabledButton = screen.getByRole('button', { name: /click me/i });

    await user.click(enabledButton);
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  test('renders as a slot when asChild is true', () => {
    render(
      <Button asChild>
        <a href="https://example.com">Link Button</a>
      </Button>
    );

    const link = screen.getByRole('link', { name: /link button/i });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', 'https://example.com');
    expect(link).toHaveClass('bg-primary');
  });
});
