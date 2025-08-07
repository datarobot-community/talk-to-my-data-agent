import { screen, fireEvent } from '@testing-library/react';
import { test, describe, expect, vi } from 'vitest';
import { InsightsTabContent } from '@/components/chat/InsightsTabContent';
import { renderWithProviders } from '../test-utils';

const mockMutate = vi.fn();
const mockUseGeneratedDictionaries = vi.fn();
const mockUsePostMessage = vi.fn();
const mockUseAppState = vi.fn();

// Mock hooks to control SuggestedPrompt behavior
vi.mock('@/api/dictionaries/hooks', () => ({
  useGeneratedDictionaries: () => mockUseGeneratedDictionaries(),
}));

vi.mock('@/api/chat-messages/hooks', () => ({
  usePostMessage: () => mockUsePostMessage(),
}));

vi.mock('@/state/hooks', () => ({
  useAppState: () => mockUseAppState(),
}));

describe('InsightsTabContent', () => {
  beforeEach(() => {
    mockUseGeneratedDictionaries.mockReturnValue({ data: [{ id: 1, name: 'test' }] });
    mockUsePostMessage.mockReturnValue({ mutate: mockMutate });
    mockUseAppState.mockReturnValue({
      enableChartGeneration: true,
      enableBusinessInsights: true,
      dataSource: 'test-source',
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  test('renders insights content with actionable suggested prompts when dictionaries available', () => {
    const additionalInsights = 'Key insights about the data analysis';
    const followUpQuestions = ['What trends do you see?', 'How can we improve?'];

    renderWithProviders(
      <InsightsTabContent
        additionalInsights={additionalInsights}
        followUpQuestions={followUpQuestions}
        chatId="test-chat"
        isProcessing={false}
      />
    );

    expect(screen.getByText('Data insights')).toBeInTheDocument();
    expect(screen.getByText(additionalInsights)).toBeInTheDocument();
    expect(screen.getByText('What trends do you see?')).toBeInTheDocument();
    expect(screen.getByText('How can we improve?')).toBeInTheDocument();

    // Test that suggested prompts have send buttons when dictionaries are available
    const sendButtons = screen.getAllByTestId('send-suggested-prompt-button');
    expect(sendButtons).toHaveLength(2);
    expect(sendButtons[0]).not.toBeDisabled();

    // Test clicking a suggested prompt
    fireEvent.click(sendButtons[0]);
    expect(mockMutate).toHaveBeenCalledWith({
      message: 'What trends do you see?',
      chatId: 'test-chat',
      enableChartGeneration: true,
      enableBusinessInsights: true,
      dataSource: 'test-source',
    });
  });

  test('shows disabled suggested prompt buttons when processing', () => {
    renderWithProviders(
      <InsightsTabContent
        followUpQuestions={['Question 1']}
        chatId="test-chat"
        isProcessing={true}
      />
    );

    expect(screen.queryByText('Data insights')).not.toBeInTheDocument();
    expect(screen.getByText('Question 1')).toBeInTheDocument();

    const sendButton = screen.getByTestId('send-suggested-prompt-button');
    expect(sendButton).toBeDisabled();
    expect(sendButton).toHaveAttribute('title', 'Wait for agent to finish responding');
  });

  test('hides suggested prompt buttons when no dictionaries available', () => {
    mockUseGeneratedDictionaries.mockReturnValue({ data: [] });

    renderWithProviders(
      <InsightsTabContent followUpQuestions={['Question 1']} chatId="test-chat" />
    );

    expect(screen.getByText('Question 1')).toBeInTheDocument();
    expect(screen.queryByTestId('send-suggested-prompt-button')).not.toBeInTheDocument();
  });
});
