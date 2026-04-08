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
  useUpdateMessageFeedback: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
}));

vi.mock('@/api/dictionaries/hooks', () => ({
  useGeneratedDictionaries: vi.fn(() => ({ data: [] })),
}));

vi.mock('@/api/cleansed-datasets/hooks', () => ({
  useMultipleDatasetMetadata: vi.fn(() => ({ data: [] })),
}));

// react-syntax-highlighter and react-plotly.js are ESM-only packages that can't be
// imported in jsdom without transformation. Mock them to avoid bundler failures.
vi.mock('react-syntax-highlighter', () => ({
  Prism: ({ children }: { children: string }) => (
    <pre data-testid="syntax-highlighter">{children}</pre>
  ),
}));

vi.mock('react-syntax-highlighter/dist/esm/styles/prism', () => ({
  oneDark: {},
  oneLight: {},
}));

vi.mock('react-plotly.js', () => ({
  default: () => <div data-testid="plotly-chart">Chart</div>,
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
    expect(screen.getByTestId('export-chat-button')).toBeInTheDocument();
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

  test('renders system message in chat', () => {
    mockUseFetchAllMessages.mockReturnValue({
      data: [
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
          content: 'Summarizing...',
          components: [],
          created_at: '2024-01-01T00:01:00Z',
          in_progress: false,
        },
      ],
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    expect(screen.getByTestId('system-message-system-1')).toBeInTheDocument();
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

  // --- Phase 3: Assistant message / ResponseMessage tests ---

  test('renders assistant message with business insights (bottom line)', async () => {
    const mockMessages = [
      {
        id: 'msg-user',
        role: 'user' as const,
        content: 'Analyze sales data',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'msg-assistant',
        role: 'assistant' as const,
        content: '',
        created_at: '2024-01-01T00:01:00Z',
        in_progress: false,
        components: [
          {
            type: 'business',
            status: 'success',
            bottom_line: 'Sales increased by 20% in Q4.',
            additional_insights: 'Growth was driven by online channels.',
            follow_up_questions: ['What products drove growth?'],
          },
        ],
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    await waitFor(() => {
      expect(screen.getByTestId('response-message-msg-assistant')).toBeInTheDocument();
    });

    // ResponseTabs should render
    await waitFor(() => {
      expect(screen.getByTestId('tab-summary')).toBeInTheDocument();
      expect(screen.getByTestId('tab-insights')).toBeInTheDocument();
      expect(screen.getByTestId('tab-code')).toBeInTheDocument();
    });

    // Bottom line content on Summary tab (default tab)
    await waitFor(() => {
      expect(screen.getByText('Bottom line')).toBeInTheDocument();
      expect(screen.getByText(/Sales increased by 20%/)).toBeInTheDocument();
    });
  });

  test('renders assistant message with charts component', async () => {
    const plotData = JSON.stringify({
      data: [{ x: [1, 2, 3], y: [4, 5, 6], type: 'scatter' }],
      layout: { title: 'Test Chart' },
    });

    const mockMessages = [
      {
        id: 'msg-user',
        role: 'user' as const,
        content: 'Show me a chart',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'msg-chart',
        role: 'assistant' as const,
        content: '',
        created_at: '2024-01-01T00:01:00Z',
        in_progress: false,
        components: [
          {
            type: 'charts',
            status: 'success',
            fig1_json: plotData,
            fig2_json: null,
            code: 'import pandas as pd\ndf.plot()',
          },
        ],
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    await waitFor(() => {
      expect(screen.getByTestId('response-message-msg-chart')).toBeInTheDocument();
    });
  });

  test('renders assistant message with analysis component', async () => {
    const mockMessages = [
      {
        id: 'msg-user',
        role: 'user' as const,
        content: 'Run analysis',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'msg-analysis',
        role: 'assistant' as const,
        content: '',
        created_at: '2024-01-01T00:01:00Z',
        in_progress: false,
        components: [
          {
            type: 'analysis',
            status: 'success',
            code: 'print("hello")',
            dataset_id: null,
          },
        ],
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    await waitFor(() => {
      expect(screen.getByTestId('response-message-msg-analysis')).toBeInTheDocument();
    });

    // ResponseTabs should render with all 3 tabs
    await waitFor(() => {
      expect(screen.getByTestId('tab-summary')).toBeInTheDocument();
      expect(screen.getByTestId('tab-code')).toBeInTheDocument();
    });

    // Code tab shows success indicator since analysis completed
    await waitFor(() => {
      expect(screen.getByTestId('code-loading-success')).toBeInTheDocument();
    });
  });

  test('renders assistant message with error text', async () => {
    const mockMessages = [
      {
        id: 'msg-user',
        role: 'user' as const,
        content: 'Analyze data',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'msg-error',
        role: 'assistant' as const,
        content: '',
        created_at: '2024-01-01T00:01:00Z',
        in_progress: false,
        error: 'Something went wrong during analysis',
        components: [],
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    await waitFor(() => {
      expect(screen.getByTestId('response-message-msg-error')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('Something went wrong during analysis')).toBeInTheDocument();
    });
  });

  test('renders assistant message with chart error on Summary tab (ErrorPanel)', async () => {
    const mockMessages = [
      {
        id: 'msg-user',
        role: 'user' as const,
        content: 'Make a chart',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'msg-chart-error',
        role: 'assistant' as const,
        content: '',
        created_at: '2024-01-01T00:01:00Z',
        in_progress: false,
        components: [
          {
            type: 'charts',
            status: 'error',
            fig1_json: null,
            fig2_json: null,
            code: null,
            metadata: {
              exception: {
                exception_history: [
                  {
                    exception_str: 'Chart rendering failed',
                    code: null,
                    traceback_str: null,
                    stdout: null,
                    stderr: null,
                  },
                ],
              },
            },
          },
        ],
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    await waitFor(() => {
      expect(screen.getByTestId('response-message-msg-chart-error')).toBeInTheDocument();
    });

    // ErrorPanel shows on Summary tab (default) for chart errors without analysis errors
    await waitFor(() => {
      expect(screen.getByText(/Charts Error: Chart rendering failed/)).toBeInTheDocument();
    });
  });

  test('renders assistant message with enhanced user message', async () => {
    const mockMessages = [
      {
        id: 'msg-user',
        role: 'user' as const,
        content: 'Tell me about sales',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'msg-enhanced',
        role: 'assistant' as const,
        content: '',
        created_at: '2024-01-01T00:01:00Z',
        in_progress: false,
        components: [
          {
            enhanced_user_message: 'Analyze total sales revenue by quarter',
          },
          {
            type: 'business',
            status: 'success',
            bottom_line: 'Revenue is up.',
          },
        ],
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    await waitFor(() => {
      expect(screen.getByText('Analyze total sales revenue by quarter')).toBeInTheDocument();
    });
  });

  test('renders assistant message with in_progress step indicator', async () => {
    const mockMessages = [
      {
        id: 'msg-user',
        role: 'user' as const,
        content: 'Analyze',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'msg-progress',
        role: 'assistant' as const,
        content: '',
        created_at: '2024-01-01T00:01:00Z',
        in_progress: true,
        step: { step: 'ANALYZING_QUESTION', reattempt: 0 },
        components: [],
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    await waitFor(() => {
      expect(screen.getByTestId('response-message-msg-progress')).toBeInTheDocument();
    });

    // Step indicator should render (exact text depends on i18n)
    await waitFor(() => {
      expect(screen.getByTestId('response-message-msg-progress').textContent).toBeTruthy();
    });
  });

  test('renders all three ResponseTabs for message with multiple failed components', async () => {
    const mockMessages = [
      {
        id: 'msg-user',
        role: 'user' as const,
        content: 'Analyze',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'msg-errors',
        role: 'assistant' as const,
        content: '',
        created_at: '2024-01-01T00:01:00Z',
        in_progress: false,
        components: [
          {
            type: 'analysis',
            status: 'error',
            code: null,
            dataset_id: null,
            metadata: {
              attempts: 3,
              exception: {
                exception_history: [
                  {
                    exception_str: 'Code failed',
                    code: null,
                    traceback_str: null,
                    stdout: null,
                    stderr: null,
                  },
                ],
              },
            },
          },
          {
            type: 'business',
            status: 'error',
            bottom_line: null,
            additional_insights: null,
            metadata: {
              exception: {
                exception_history: [
                  {
                    exception_str: 'Insights failed',
                    code: null,
                    traceback_str: null,
                    stdout: null,
                    stderr: null,
                  },
                ],
              },
            },
          },
        ],
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    await waitFor(() => {
      expect(screen.getByTestId('response-message-msg-errors')).toBeInTheDocument();
    });

    // All three tabs should be present
    await waitFor(() => {
      expect(screen.getByTestId('tab-summary')).toBeInTheDocument();
      expect(screen.getByTestId('tab-insights')).toBeInTheDocument();
      expect(screen.getByTestId('tab-code')).toBeInTheDocument();
    });
  });

  test('shows reattempt count in step indicator when in progress', async () => {
    const mockMessages = [
      {
        id: 'msg-user',
        role: 'user' as const,
        content: 'Analyze',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'msg-loading',
        role: 'assistant' as const,
        content: '',
        created_at: '2024-01-01T00:01:00Z',
        in_progress: true,
        step: { step: 'GENERATING_QUERY', reattempt: 1 },
        components: [],
      },
    ];

    mockUseFetchAllMessages.mockReturnValue({
      data: mockMessages,
      isLoading: false,
      error: null,
    } as any);

    renderWithProviders(<Chats />);

    await waitFor(() => {
      expect(screen.getByTestId('response-message-msg-loading')).toBeInTheDocument();
    });

    // Step indicator should show reattempt info (contains "attempt")
    await waitFor(() => {
      expect(screen.getByTestId('response-message-msg-loading').textContent).toMatch(/attempt/i);
    });
  });
});
