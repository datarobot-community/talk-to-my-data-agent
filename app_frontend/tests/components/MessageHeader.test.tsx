import { screen } from '@testing-library/react';
import { test, describe, expect, vi, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { MessageHeader } from '@/components/chat/MessageHeader';
import { IChatMessage } from '@/api/chat-messages/types';
import { renderWithProviders } from '../test-utils';

vi.mock('@/api/chat-messages/hooks', () => ({
  useFetchAllMessages: vi.fn(),
  useDeleteMessage: vi.fn(),
  usePostMessage: vi.fn(),
  useExport: vi.fn(),
  useUpdateMessageFeedback: vi.fn(),
}));

import {
  useFetchAllMessages,
  useDeleteMessage,
  useExport,
  useUpdateMessageFeedback,
} from '@/api/chat-messages/hooks';

describe('MessageHeader Component', () => {
  const mockMessage = {
    id: 'msg-1',
    role: 'user' as const,
    content: 'Test message',
    created_at: '2024-01-15T12:00:00Z',
    components: [],
  } as IChatMessage;

  const mockResponseMessage = {
    id: 'resp-1',
    role: 'assistant' as const,
    content: 'Test response',
    created_at: '2024-01-15T12:01:00Z',
    components: [],
  } as IChatMessage;

  const defaultProps = {
    messageId: 'msg-1' as const,
    chatId: 'chat-1' as const,
    messages: [mockMessage, mockResponseMessage],
  };

  const createMockHooks = (overrides = {}) => ({
    useFetchAllMessages: { data: [mockMessage, mockResponseMessage] } as any,
    useDeleteMessage: { mutate: vi.fn(), isPending: false } as any,
    useExport: { exportChat: vi.fn(), isLoading: false },
    useUpdateMessageFeedback: { mutate: vi.fn(), isPending: false } as any,
    ...overrides,
  });

  beforeEach(() => {
    const mockHooks = createMockHooks();
    vi.mocked(useFetchAllMessages).mockReturnValue(mockHooks.useFetchAllMessages);
    vi.mocked(useDeleteMessage).mockReturnValue(mockHooks.useDeleteMessage);
    vi.mocked(useExport).mockReturnValue(mockHooks.useExport);
    vi.mocked(useUpdateMessageFeedback).mockReturnValue(mockHooks.useUpdateMessageFeedback);
  });

  test('renders basic header information for user message', () => {
    renderWithProviders(<MessageHeader {...defaultProps} />);

    expect(screen.getByText('You')).toBeInTheDocument();
    expect(screen.getByText(/2024|Jan|1\/15/)).toBeInTheDocument();
  });

  test('renders action buttons for user message', () => {
    renderWithProviders(<MessageHeader {...defaultProps} />);

    const deleteButton = screen.getByRole('button', { name: /delete message and response/i });
    const exportButton = screen.getByRole('button', { name: /export prompt and response/i });

    expect(deleteButton).toBeInTheDocument();
    expect(exportButton).toBeInTheDocument();
  });

  test('calls export hook when export button clicked', async () => {
    const user = userEvent.setup();
    const exportChat = vi.fn();
    const mockHooks = createMockHooks({ useExport: { exportChat, isLoading: false } });
    vi.mocked(useExport).mockReturnValue(mockHooks.useExport);

    renderWithProviders(<MessageHeader {...defaultProps} />);

    const exportButton = screen.getByRole('button', { name: /export prompt and response/i });
    await user.click(exportButton);

    expect(exportChat).toHaveBeenCalledWith({ chatId: 'chat-1', messageId: 'msg-1' });
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
    const deleteMutate = vi.fn();
    const mockHooks = createMockHooks({
      useDeleteMessage: { mutate: deleteMutate, isPending: false } as any,
    });
    vi.mocked(useDeleteMessage).mockReturnValue(mockHooks.useDeleteMessage);

    renderWithProviders(<MessageHeader {...defaultProps} />);

    const deleteButton = screen.getByRole('button', { name: /delete message and response/i });
    await user.click(deleteButton);

    const confirmButton = screen.getByTestId('confirm-dialog-confirm');
    await user.click(confirmButton);

    // Deletes the response first, then the user message — locking the pairing so the
    // APP-6805 bound cannot regress into "never delete the assistant".
    expect(deleteMutate).toHaveBeenCalledTimes(2);
    expect(deleteMutate).toHaveBeenNthCalledWith(1, { messageId: 'resp-1', chatId: 'chat-1' });
    expect(deleteMutate).toHaveBeenNthCalledWith(2, { messageId: 'msg-1', chatId: 'chat-1' });
  });

  test('deleting a failed unpaired question removes only that message, not a later answer', async () => {
    // APP-6805: msg-1 failed with no answer of its own; deleting it must not remove
    // resp-2, which belongs to the following question.
    const failedUserMessage = {
      id: 'msg-1',
      role: 'user' as const,
      content: 'Failed question',
      created_at: '2024-01-15T12:00:00Z',
      components: [],
      error: 'Failed to process your question',
    } as IChatMessage;
    const nextUserMessage = {
      id: 'msg-2',
      role: 'user' as const,
      content: 'Next question',
      created_at: '2024-01-15T12:02:00Z',
      components: [],
    } as IChatMessage;
    const nextResponse = {
      id: 'resp-2',
      role: 'assistant' as const,
      content: 'Answer to next question',
      created_at: '2024-01-15T12:03:00Z',
      components: [],
    } as IChatMessage;

    const user = userEvent.setup();
    const deleteMutate = vi.fn();
    vi.mocked(useDeleteMessage).mockReturnValue({
      mutate: deleteMutate,
      isPending: false,
    } as any);

    renderWithProviders(
      <MessageHeader
        messageId="msg-1"
        chatId="chat-1"
        messages={[failedUserMessage, nextUserMessage, nextResponse]}
      />
    );

    const deleteButton = screen.getByRole('button', { name: /delete message/i });
    await user.click(deleteButton);

    const confirmButton = screen.getByTestId('confirm-dialog-confirm');
    await user.click(confirmButton);

    expect(deleteMutate).toHaveBeenCalledTimes(1);
    expect(deleteMutate).toHaveBeenCalledWith({ messageId: 'msg-1', chatId: 'chat-1' });
    expect(deleteMutate).not.toHaveBeenCalledWith({ messageId: 'resp-2', chatId: 'chat-1' });
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
    const mockHooks = createMockHooks({ useExport: { exportChat: vi.fn(), isLoading: true } });
    vi.mocked(useExport).mockReturnValue(mockHooks.useExport);
    renderWithProviders(<MessageHeader {...defaultProps} />);

    const exportButton = screen.getByRole('button', { name: /exporting/i });
    expect(exportButton).toBeDisabled();
  });

  test('disables export button when response is in progress', () => {
    const inProgressResponseMessage = {
      id: 'resp-1',
      role: 'assistant' as const,
      content: 'Response in progress...',
      created_at: '2024-01-01T00:01:00Z',
      components: [],
      in_progress: true,
    } as IChatMessage;
    const mockHooks = createMockHooks({
      useFetchAllMessages: { data: [mockMessage, inProgressResponseMessage] } as any,
    });
    vi.mocked(useFetchAllMessages).mockReturnValue(mockHooks.useFetchAllMessages);

    renderWithProviders(
      <MessageHeader {...defaultProps} messages={[mockMessage, inProgressResponseMessage]} />
    );

    const exportButton = screen.getByRole('button', {
      name: 'Wait for agent to finish responding',
    });
    expect(exportButton).toBeDisabled();
    expect(exportButton).toHaveAttribute('title', 'Wait for agent to finish responding');
  });

  test('disables export button when response is failing', () => {
    const erroredResponseMessage = {
      id: 'resp-1',
      role: 'assistant' as const,
      content: 'Error occurred',
      created_at: '2024-01-01T00:01:00Z',
      components: [],
      error: 'boom',
    } as IChatMessage;
    const mockHooks = createMockHooks({
      useFetchAllMessages: { data: [mockMessage, erroredResponseMessage] } as any,
    });
    vi.mocked(useFetchAllMessages).mockReturnValue(mockHooks.useFetchAllMessages);

    renderWithProviders(
      <MessageHeader {...defaultProps} messages={[mockMessage, erroredResponseMessage]} />
    );

    const exportButton = screen.getByRole('button', { name: 'Cannot export chat with errors' });
    expect(exportButton).toBeDisabled();
    expect(exportButton).toHaveAttribute('title', 'Cannot export chat with errors');
  });

  test('shows correct tooltip when response is in progress', () => {
    const inProgressResponseMessage = {
      id: 'resp-1',
      role: 'assistant' as const,
      content: 'Response in progress...',
      created_at: '2024-01-01T00:01:00Z',
      components: [],
      in_progress: true,
    } as IChatMessage;
    const mockHooks = createMockHooks({
      useFetchAllMessages: { data: [mockMessage, inProgressResponseMessage] } as any,
    });
    vi.mocked(useFetchAllMessages).mockReturnValue(mockHooks.useFetchAllMessages);

    renderWithProviders(
      <MessageHeader {...defaultProps} messages={[mockMessage, inProgressResponseMessage]} />
    );

    const exportButton = screen.getByRole('button', {
      name: 'Wait for agent to finish responding',
    });
    expect(exportButton).toHaveAttribute('title', 'Wait for agent to finish responding');
  });

  test('does not render action buttons for assistant message', () => {
    const assistantMessage = { ...mockMessage, role: 'assistant' as const };
    const mockHooksWithAssistantMessage = createMockHooks({
      useFetchAllMessages: { data: [assistantMessage] } as any,
    });
    vi.mocked(useFetchAllMessages).mockReturnValue(
      mockHooksWithAssistantMessage.useFetchAllMessages
    );

    renderWithProviders(<MessageHeader {...defaultProps} messages={[assistantMessage]} />);

    expect(screen.getByText('DataRobot')).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: /delete message and response/i })
    ).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /export chat/i })).not.toBeInTheDocument();
  });

  test('shows thumbs up as selected when assistant message has existing positive feedback', () => {
    const assistantMessageWithPositiveFeedback = {
      ...mockResponseMessage,
      user_rating: 1,
    } as IChatMessage;

    renderWithProviders(
      <MessageHeader
        messageId={assistantMessageWithPositiveFeedback.id}
        chatId="chat-1"
        messages={[assistantMessageWithPositiveFeedback]}
      />
    );

    const thumbsUpButton = screen.getByTestId('message-feedback-thumbs-up');
    expect(thumbsUpButton).toHaveAttribute('aria-pressed', 'true');
  });

  test('sends +1 rating when thumbs up is clicked', async () => {
    const user = userEvent.setup();
    const updateFeedbackMutate = vi.fn();
    const assistantMessage = { ...mockResponseMessage } as IChatMessage;
    const mockHooks = createMockHooks({
      useFetchAllMessages: { data: [assistantMessage] } as any,
      useUpdateMessageFeedback: { mutate: updateFeedbackMutate, isPending: false } as any,
    });
    vi.mocked(useFetchAllMessages).mockReturnValue(mockHooks.useFetchAllMessages);
    vi.mocked(useUpdateMessageFeedback).mockReturnValue(mockHooks.useUpdateMessageFeedback);

    renderWithProviders(
      <MessageHeader
        messageId={assistantMessage.id}
        chatId="chat-1"
        messages={[assistantMessage]}
      />
    );

    await user.click(screen.getByTestId('message-feedback-thumbs-up'));

    expect(updateFeedbackMutate).toHaveBeenCalledWith({
      messageId: assistantMessage.id,
      chatId: 'chat-1',
      userRating: 1,
      userFeedback: '',
    });
  });

  test('opens feedback modal when thumbs down is clicked', async () => {
    const user = userEvent.setup();
    const assistantMessage = { ...mockResponseMessage } as IChatMessage;

    renderWithProviders(
      <MessageHeader
        messageId={assistantMessage.id}
        chatId="chat-1"
        messages={[assistantMessage]}
      />
    );

    await user.click(screen.getByTestId('message-feedback-thumbs-down'));

    expect(screen.getByText('What could have been better?')).toBeInTheDocument();
    expect(screen.getByTestId('message-feedback-textarea')).toBeInTheDocument();
  });

  test('discards entered feedback on close and sends -1 rating without text', async () => {
    const user = userEvent.setup();
    const updateFeedbackMutate = vi.fn();
    const assistantMessage = { ...mockResponseMessage } as IChatMessage;
    const mockHooks = createMockHooks({
      useFetchAllMessages: { data: [assistantMessage] } as any,
      useUpdateMessageFeedback: { mutate: updateFeedbackMutate, isPending: false } as any,
    });
    vi.mocked(useFetchAllMessages).mockReturnValue(mockHooks.useFetchAllMessages);
    vi.mocked(useUpdateMessageFeedback).mockReturnValue(mockHooks.useUpdateMessageFeedback);

    renderWithProviders(
      <MessageHeader
        messageId={assistantMessage.id}
        chatId="chat-1"
        messages={[assistantMessage]}
      />
    );

    await user.click(screen.getByTestId('message-feedback-thumbs-down'));
    const textarea = screen.getByTestId('message-feedback-textarea');
    await user.type(textarea, 'Needs improvement');
    await user.keyboard('{Escape}');

    expect(updateFeedbackMutate).toHaveBeenCalledWith({
      messageId: assistantMessage.id,
      chatId: 'chat-1',
      userRating: -1,
    });

    await user.click(screen.getByTestId('message-feedback-thumbs-down'));
    expect(screen.getByTestId('message-feedback-textarea')).toHaveValue('');
  });

  test('sends -1 rating and feedback text when submit is clicked', async () => {
    const user = userEvent.setup();
    const updateFeedbackMutate = vi.fn();
    const assistantMessage = { ...mockResponseMessage } as IChatMessage;
    const mockHooks = createMockHooks({
      useFetchAllMessages: { data: [assistantMessage] } as any,
      useUpdateMessageFeedback: { mutate: updateFeedbackMutate, isPending: false } as any,
    });
    vi.mocked(useFetchAllMessages).mockReturnValue(mockHooks.useFetchAllMessages);
    vi.mocked(useUpdateMessageFeedback).mockReturnValue(mockHooks.useUpdateMessageFeedback);

    renderWithProviders(
      <MessageHeader
        messageId={assistantMessage.id}
        chatId="chat-1"
        messages={[assistantMessage]}
      />
    );

    await user.click(screen.getByTestId('message-feedback-thumbs-down'));
    const textarea = screen.getByTestId('message-feedback-textarea');
    await user.type(textarea, 'This answer missed key details');
    await user.click(screen.getByRole('button', { name: 'Submit' }));

    expect(updateFeedbackMutate).toHaveBeenCalledTimes(1);
    expect(updateFeedbackMutate).toHaveBeenCalledWith({
      messageId: assistantMessage.id,
      chatId: 'chat-1',
      userRating: -1,
      userFeedback: 'This answer missed key details',
    });
  });
});
