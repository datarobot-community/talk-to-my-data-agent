import { screen, fireEvent, waitFor } from '@testing-library/react';
import { test, describe, expect, vi, beforeEach } from 'vitest';
import { Chats } from '@/pages/Chats';
import { renderWithProviders, mockScrollIntoView } from '../test-utils';

vi.mock('@/api/chat-messages/hooks', () => ({
  useFetchAllChats: vi.fn(),
  useFetchAllMessages: vi.fn(),
  useDeleteChat: vi.fn(),
  useDeleteMessage: vi.fn(),
  usePostMessage: vi.fn(),
  useExport: vi.fn(),
  useRenameChat: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
  usePollInProgressMessage: vi.fn(),
}));

vi.mock('@/api/dictionaries/hooks', () => ({
  useGeneratedDictionaries: vi.fn(() => ({ data: [] })),
}));

vi.mock('@/api/cleansed-datasets/hooks', () => ({
  useMultipleDatasetMetadata: vi.fn(() => ({ data: [] })),
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useParams: vi.fn(),
    useNavigate: vi.fn(),
  };
});

import {
  useFetchAllChats,
  useFetchAllMessages,
  useDeleteChat,
  useDeleteMessage,
  usePostMessage,
  useExport,
  usePollInProgressMessage,
} from '@/api/chat-messages/hooks';
import { useParams, useNavigate } from 'react-router-dom';

const mockUseFetchAllChats = vi.mocked(useFetchAllChats);
const mockUseFetchAllMessages = vi.mocked(useFetchAllMessages);
const mockUseDeleteChat = vi.mocked(useDeleteChat);
const mockUseDeleteMessage = vi.mocked(useDeleteMessage);
const mockUsePostMessage = vi.mocked(usePostMessage);
const mockUseExport = vi.mocked(useExport);
const mockUsePollInProgressMessage = vi.mocked(usePollInProgressMessage);
const mockUseParams = vi.mocked(useParams);
const mockUseNavigate = vi.mocked(useNavigate);

describe('Chats Component', () => {
  let cleanupScrollMock: () => void;
  const mockNavigate = vi.fn();
  const mockDeleteChat = vi.fn();
  const mockExportChat = vi.fn();

  const mockChats = [
    {
      id: 'chat-1',
      name: 'Test Chat',
      created_at: '2024-01-01T00:00:00Z',
    },
  ];

  beforeEach(() => {
    cleanupScrollMock = mockScrollIntoView();

    mockUseNavigate.mockReturnValue(mockNavigate);
    mockUseFetchAllMessages.mockReturnValue({
      data: [],
      isLoading: false,
      error: null,
    } as any);
    mockUsePollInProgressMessage.mockReturnValue({} as any);
    mockUseExport.mockReturnValue({
      exportChat: mockExportChat,
      isLoading: false,
    });
    mockUseDeleteChat.mockReturnValue({
      mutate: mockDeleteChat,
      isPending: false,
    } as any);
    mockUseDeleteMessage.mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
    } as any);
    mockUsePostMessage.mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
    } as any);

    setupChatContext();
  });

  afterEach(() => {
    cleanupScrollMock();
    vi.clearAllMocks();
  });

  const setupChatContext = () => {
    mockUseParams.mockReturnValue({ chatId: 'chat-1' });
    mockUseFetchAllChats.mockReturnValue({
      data: mockChats,
      isLoading: false,
    } as any);
  };

  test('renders InitialPrompt when no messages exist', () => {
    mockUseParams.mockReturnValue({ chatId: undefined });
    mockUseFetchAllChats.mockReturnValue({
      data: [],
      isLoading: false,
    } as any);

    renderWithProviders(<Chats />);

    expect(screen.getByTestId('initial-prompt')).toBeInTheDocument();
    expect(screen.queryByTestId('user-prompt')).not.toBeInTheDocument();
  });

  test('renders loading state when messages are pending', () => {
    mockUseParams.mockReturnValue({ chatId: undefined });
    mockUseFetchAllChats.mockReturnValue({
      data: [],
      isLoading: false,
    } as any);
    mockUseFetchAllMessages.mockReturnValue({
      data: [],
      isLoading: true,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  test('renders user message and user prompt when messages exist', () => {
    const mockMessages = [
      {
        id: 'msg-1',
        role: 'user' as const,
        content: 'Hello',
        created_at: '2024-01-01T00:00:00Z',
        components: [],
        in_progress: false,
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    expect(screen.getByTestId('user-message-msg-1')).toBeInTheDocument();
    expect(screen.getByText('Hello')).toBeInTheDocument();
    expect(screen.getByTestId('user-prompt')).toBeInTheDocument();
  });

  test('renders chat header with actions when active chat exists', () => {
    renderWithProviders(<Chats />);

    expect(screen.getByText('Test Chat')).toBeInTheDocument();
    expect(screen.getByTestId('delete-all-chats-button')).toBeInTheDocument();
    expect(screen.getByText('Export chat')).toBeInTheDocument();
  });

  test('opens delete confirmation dialog when delete button is clicked', async () => {
    renderWithProviders(<Chats />);

    const deleteButton = screen.getByTestId('delete-all-chats-button');
    fireEvent.click(deleteButton);

    await waitFor(() => {
      expect(screen.getByTestId('dialog-description')).toBeInTheDocument();
    });
  });

  test('calls export function when export button is clicked', () => {
    renderWithProviders(<Chats />);

    const exportButton = screen.getByTestId('export-chat-button');
    fireEvent.click(exportButton);

    expect(mockExportChat).toHaveBeenCalledWith({ chatId: 'chat-1' });
  });

  test('disables export button when chat has errors', () => {
    const mockMessages = [
      {
        id: 'msg-1',
        role: 'assistant' as const,
        content: 'Error response',
        created_at: '2024-01-01T00:00:00Z',
        components: [],
        in_progress: false,
        error: 'Some error occurred',
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    const exportButton = screen.getByTestId('export-chat-button');
    expect(exportButton).toBeDisabled();
    expect(exportButton).toHaveAttribute('title', 'Cannot export chat with errors');
  });

  test('disables export button when processing', () => {
    const mockMessages = [
      {
        id: 'msg-1',
        role: 'user' as const,
        content: 'Hello',
        created_at: '2024-01-01T00:00:00Z',
        components: [],
        in_progress: true,
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    const exportButton = screen.getByTestId('export-chat-button');
    expect(exportButton).toBeDisabled();
    expect(exportButton).toHaveAttribute('title', 'Wait for agent to finish responding');
  });

  test('renders system message (conversation summary) in chat', () => {
    const mockMessages = [
      {
        id: 'user-1',
        role: 'user' as const,
        content: 'What is the data about?',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'system-1',
        role: 'system' as const,
        content: 'The user analyzed patient demographics focusing on readmission rates.',
        components: [],
        created_at: '2024-01-01T00:01:00Z',
        in_progress: false,
      },
      {
        id: 'assistant-1',
        role: 'assistant' as const,
        content: 'Analysis shows key patterns.',
        components: [],
        created_at: '2024-01-01T00:02:00Z',
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    const systemMessage = screen.getByTestId('system-message-system-1');
    expect(systemMessage).toBeInTheDocument();
  });

  test('renders system message with in_progress state', () => {
    const mockMessages = [
      {
        id: 'user-1',
        role: 'user' as const,
        content: 'Question',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'system-1',
        role: 'system' as const,
        content: 'Summarizing conversation...',
        components: [],
        created_at: '2024-01-01T00:01:00Z',
        in_progress: true,
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    const systemMessage = screen.getByTestId('system-message-system-1');
    expect(systemMessage).toBeInTheDocument();
  });

  test('displays error message when user message fails', () => {
    const question = 'What is the weather?';
    const errorMessage =
      'Failed to process your question: LLM validation failed: Invalid response format from model';
    const mockMessages = [
      {
        id: 'msg-1',
        role: 'user' as const,
        content: question,
        created_at: '2024-01-01T00:00:00Z',
        components: [],
        in_progress: false,
        error: errorMessage,
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    expect(screen.getByText(question)).toBeInTheDocument();
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
    expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
  });
});
