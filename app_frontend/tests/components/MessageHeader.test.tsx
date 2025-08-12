import { screen } from '@testing-library/react';
import { test, describe, expect, vi, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { MessageHeader } from '@/components/chat/MessageHeader';
import { IChatMessage } from '@/api/chat-messages/types';
import { renderWithProviders } from '../test-utils';

vi.mock('@/hooks/useChatMessages', () => ({
  useChatMessages: vi.fn(),
}));
import { useChatMessages } from '@/hooks/useChatMessages';

describe('MessageHeader Component', () => {
  const mockMessage = {
    id: 'msg-1',
    role: 'user' as const,
    content: 'Test message',
    created_at: '2024-01-01T00:00:00Z',
    components: [],
  } as IChatMessage;

  const defaultProps = {
    messageId: 'msg-1',
    chatId: 'chat-1',
  } as const;

  // Default mock configuration
  const createMockChatMessages = (overrides = {}) => ({
    messages: [],
    isLoading: false,
    isSending: false,
    isDeleting: false,
    isExporting: false,
    hasInProgressMessages: false,
    hasFailedMessages: false,
    fetchError: null,
    sendError: null,
    deleteError: null,
    getMessage: () => mockMessage,
    getResponseMessage: () => ({ id: 'resp-1' }) as IChatMessage,
    sendMessage: vi.fn(),
    deleteMessagePair: vi.fn(),
    exportMessage: vi.fn(),
    ...overrides,
  });

  beforeEach(() => {
    vi.mocked(useChatMessages).mockReturnValue(createMockChatMessages());
  });

  test('renders basic header information for user message', () => {
    renderWithProviders(<MessageHeader {...defaultProps} />);

    expect(screen.getByText('You')).toBeInTheDocument();
    expect(screen.getByText(/Jan/)).toBeInTheDocument(); // Formatted date
  });

  test('renders action buttons for user message', () => {
    renderWithProviders(<MessageHeader {...defaultProps} />);

    const deleteButton = screen.getByRole('button', { name: /delete message and response/i });
    const exportButton = screen.getByRole('button', { name: /export chat/i });

    expect(deleteButton).toBeInTheDocument();
    expect(exportButton).toBeInTheDocument();
  });

  test('calls export hook when export button clicked', async () => {
    const user = userEvent.setup();
    const exportMessage = vi.fn();
    vi.mocked(useChatMessages).mockReturnValue(createMockChatMessages({ exportMessage }));

    renderWithProviders(<MessageHeader {...defaultProps} />);

    const exportButton = screen.getByRole('button', { name: /export chat/i });
    await user.click(exportButton);

    expect(exportMessage).toHaveBeenCalledWith('msg-1');
  });

  test('shows confirm dialog when delete button is clicked', async () => {
    const user = userEvent.setup();
    renderWithProviders(<MessageHeader {...defaultProps} />);

    const deleteButton = screen.getByRole('button', { name: /delete message and response/i });
    await user.click(deleteButton);

    expect(screen.getByText('Delete message')).toBeInTheDocument();
    expect(screen.getByText('Are you sure you want to delete this message?')).toBeInTheDocument();
  });

  test('calls delete hook when confirm dialog is confirmed', async () => {
    const user = userEvent.setup();
    const deleteMessagePair = vi.fn();
    vi.mocked(useChatMessages).mockReturnValue(createMockChatMessages({ deleteMessagePair }));

    renderWithProviders(<MessageHeader {...defaultProps} />);

    const deleteButton = screen.getByRole('button', { name: /delete message and response/i });
    await user.click(deleteButton);

    const confirmButton = screen.getByTestId('confirm-dialog-confirm');
    await user.click(confirmButton);

    expect(deleteMessagePair).toHaveBeenCalledWith('msg-1');
  });

  test('closes confirm dialog when cancel is clicked', async () => {
    const user = userEvent.setup();
    renderWithProviders(<MessageHeader {...defaultProps} />);

    const deleteButton = screen.getByRole('button', { name: /delete message and response/i });
    await user.click(deleteButton);

    const cancelButton = screen.getByTestId('confirm-dialog-cancel');
    await user.click(cancelButton);

    expect(screen.queryByText('Delete message')).not.toBeInTheDocument();
    // No side-effect hook expected on cancel
  });

  test('disables export button when isExporting is true', () => {
    vi.mocked(useChatMessages).mockReturnValue(createMockChatMessages({ isExporting: true }));
    renderWithProviders(<MessageHeader {...defaultProps} />);

    const exportButton = screen.getByRole('button', { name: /exporting/i });
    expect(exportButton).toBeDisabled();
  });

  test('disables export button when response is in progress', () => {
    vi.mocked(useChatMessages).mockReturnValue(
      createMockChatMessages({
        getResponseMessage: () => ({ id: 'resp-1', in_progress: true }) as IChatMessage,
      })
    );
    renderWithProviders(<MessageHeader {...defaultProps} />);

    const exportButton = screen.getByRole('button', {
      name: /Wait for agent to finish responding/i,
    });
    expect(exportButton).toBeDisabled();
  });

  test('disables export button when response is failing', () => {
    vi.mocked(useChatMessages).mockReturnValue(
      createMockChatMessages({
        getResponseMessage: () => ({ id: 'resp-1', error: 'boom' }) as IChatMessage,
      })
    );
    renderWithProviders(<MessageHeader {...defaultProps} />);

    const exportButton = screen.getByRole('button', { name: /cannot export chat with errors/i });
    expect(exportButton).toBeDisabled();
    expect(exportButton).toHaveAttribute('title', 'Cannot export chat with errors');
  });

  test('shows correct tooltip when response is in progress', () => {
    vi.mocked(useChatMessages).mockReturnValue(
      createMockChatMessages({
        getResponseMessage: () => ({ id: 'resp-1', in_progress: true }) as IChatMessage,
      })
    );
    renderWithProviders(<MessageHeader {...defaultProps} />);

    const exportButton = screen.getByRole('button', {
      name: /Wait for agent to finish responding/i,
    });
    expect(exportButton).toHaveAttribute('title', 'Wait for agent to finish responding');
  });

  test('does not render action buttons for assistant message', () => {
    const assistantMessage = { ...mockMessage, role: 'assistant' as const };
    vi.mocked(useChatMessages).mockReturnValue(
      createMockChatMessages({
        getMessage: () => assistantMessage,
        getResponseMessage: () => assistantMessage,
      })
    );

    renderWithProviders(<MessageHeader {...defaultProps} />);

    expect(screen.getByText('DataRobot')).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: /delete message and response/i })
    ).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /export chat/i })).not.toBeInTheDocument();
  });
});
