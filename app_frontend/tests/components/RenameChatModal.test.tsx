import { render, screen, fireEvent } from '@testing-library/react';
import { test, describe, expect, vi, type MockInstance, type Mock } from 'vitest';
import { RenameChatModal } from '@/components/RenameChatModal';
import { useRenameChat } from '@/api/chat-messages/hooks';

vi.mock('@/api/chat-messages/hooks', () => ({
  useRenameChat: vi.fn(),
}));

describe('RenameChatModal Component', () => {
  let mockRenameChat: MockInstance;

  beforeEach(() => {
    mockRenameChat = vi.fn();
    (useRenameChat as Mock).mockReturnValue({
      mutate: mockRenameChat,
      isPending: false,
    });
  });

  test('enforces max length on chat name input', () => {
    render(<RenameChatModal chatId="123" currentName="Test Chat" />);

    const triggerButton = screen.getByRole('button');
    fireEvent.click(triggerButton);

    const input = screen.getByRole('textbox');
    expect(input).toHaveAttribute('maxLength', '200');
  });

  test('shows warning when chat name reaches max length', () => {
    render(<RenameChatModal chatId="123" currentName="Test Chat" />);

    const triggerButton = screen.getByRole('button');
    fireEvent.click(triggerButton);

    const input = screen.getByRole('textbox');
    const longName = 'a'.repeat(200);
    fireEvent.change(input, { target: { value: longName } });

    expect(screen.getByText(/has reached maximum of 200 characters/i)).toBeInTheDocument();
  });

  test('does not show warning for short chat names', () => {
    render(<RenameChatModal chatId="123" currentName="Test Chat" />);

    const triggerButton = screen.getByRole('button');
    fireEvent.click(triggerButton);

    expect(screen.queryByText(/has reached maximum of 200 characters/i)).not.toBeInTheDocument();
  });
});
