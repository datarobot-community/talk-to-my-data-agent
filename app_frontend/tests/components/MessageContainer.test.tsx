import { render, screen } from '@testing-library/react';
import { test, describe, expect, vi } from 'vitest';
import { MessageContainer } from '@/components/chat/MessageContainer';

describe('MessageContainer Component', () => {
  // Setup and teardown for all tests
  let scrollIntoViewMock: ReturnType<typeof vi.fn>;
  let originalScrollIntoView: typeof HTMLElement.prototype.scrollIntoView;

  beforeEach(() => {
    // Store original method before mocking
    originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
    // Create a mock for scrollIntoView
    scrollIntoViewMock = vi.fn();

    // Mock scrollIntoView on the HTMLElement prototype
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: scrollIntoViewMock,
      writable: true,
    });
  });

  afterEach(() => {
    // Restore the original method after each test
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: originalScrollIntoView,
      writable: true,
    });
  });

  test('renders children correctly', () => {
    render(
      <MessageContainer id="test-message-1">
        <div data-testid="test-child">Test Child Content</div>
      </MessageContainer>
    );

    const childElement = screen.getByTestId('test-child');
    expect(childElement).toBeInTheDocument();
    expect(childElement).toHaveTextContent('Test Child Content');
  });

  test('applies correct styling', () => {
    render(
      <MessageContainer id="test-message-2">
        <div>Content</div>
      </MessageContainer>
    );

    const container = screen.getByText('Content').parentElement;
    expect(container).toHaveClass('p-3');
    expect(container).toHaveClass('bg-card');
    expect(container).toHaveClass('rounded');
    expect(container).toHaveClass('flex-col');
  });

  test('calls scrollIntoView when mounted', () => {
    render(
      <MessageContainer id="test-message-3">
        <div>Scroll Test</div>
      </MessageContainer>
    );

    // Verify scrollIntoView was called with false
    expect(scrollIntoViewMock).toHaveBeenCalledWith(false);
  });
});
