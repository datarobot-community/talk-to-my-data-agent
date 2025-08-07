import { screen, fireEvent, waitFor } from '@testing-library/react';
import { test, describe, expect, vi, beforeEach } from 'vitest';
import { Chats } from '@/pages/Chats';
import { renderWithProviders, mockScrollIntoView } from '../test-utils';

vi.mock('@/api/chat-messages/hooks', () => ({
  useFetchAllMessages: vi.fn(),
  useFetchAllChats: vi.fn(),
  useDeleteChat: vi.fn(),
  useExport: vi.fn(),
  useRenameChat: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
  usePostMessage: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
  useDeleteMessage: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
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
  useFetchAllMessages,
  useFetchAllChats,
  useDeleteChat,
  useExport,
} from '@/api/chat-messages/hooks';
import { useParams, useNavigate } from 'react-router-dom';

const mockUseFetchAllMessages = vi.mocked(useFetchAllMessages);
const mockUseFetchAllChats = vi.mocked(useFetchAllChats);
const mockUseDeleteChat = vi.mocked(useDeleteChat);
const mockUseExport = vi.mocked(useExport);
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
      status: 'success',
      isLoading: false,
    } as any);
    mockUseExport.mockReturnValue({
      exportChat: mockExportChat,
      isLoading: false,
    });
    mockUseDeleteChat.mockReturnValue({
      mutate: mockDeleteChat,
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
      status: 'pending',
      isLoading: true,
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
      status: 'success',
      isLoading: false,
    } as any);

    renderWithProviders(<Chats />);

    expect(screen.getByTestId('user-message-0')).toBeInTheDocument();
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
      status: 'success',
      isLoading: false,
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
      status: 'success',
      isLoading: false,
    } as any);

    renderWithProviders(<Chats />);

    const exportButton = screen.getByTestId('export-chat-button');
    expect(exportButton).toBeDisabled();
    expect(exportButton).toHaveAttribute('title', 'Wait for agent to finish responding');
  });
});
