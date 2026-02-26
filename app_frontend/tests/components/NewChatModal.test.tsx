import { render, screen, fireEvent } from '@testing-library/react';
import { test, describe, expect, vi, type MockInstance, type Mock } from 'vitest';
import { NewChatModal } from '@/components/NewChatModal';
import { useCreateChat } from '@/api/chat-messages/hooks';
import { useNavigate } from 'react-router-dom';
import { generateChatRoute } from '@/pages/routes';

// Mock the dependencies
vi.mock('@/api/chat-messages/hooks', () => ({
  useCreateChat: vi.fn(),
}));

vi.mock('react-router-dom', () => ({
  useNavigate: vi.fn(),
}));

vi.mock('@/pages/routes', () => ({
  generateChatRoute: vi.fn(),
}));

describe('NewChatModal Component', () => {
  let mockCreateChat: MockInstance;
  let mockNavigate: MockInstance;

  // Setup default mocks before each test
  beforeEach(() => {
    // Mock the createChat mutation
    mockCreateChat = vi.fn();
    (useCreateChat as Mock).mockReturnValue({
      mutate: mockCreateChat,
      isPending: false,
    });

    // Mock the navigate function
    mockNavigate = vi.fn();
    (useNavigate as Mock).mockReturnValue(mockNavigate);

    // Mock the generateChatRoute function
    (generateChatRoute as Mock).mockImplementation(id => `/chat/${id}`);
  });

  test('renders the trigger button correctly', () => {
    render(<NewChatModal highlight={false} />);
    const newChatButton = screen.getByRole('button', { name: /new chat/i });
    expect(newChatButton).toBeInTheDocument();
    expect(newChatButton.querySelector('.lucide-plus')).toBeInTheDocument();
  });

  test('shows loading state when creating a chat', () => {
    // Set isPending to true for loading state
    (useCreateChat as Mock).mockReturnValue({
      mutate: mockCreateChat,
      isPending: true,
    });

    render(<NewChatModal highlight={false} />);

    // Find and click the New Chat button to open the dialog
    const triggerButton = screen.getByRole('button', { name: /new chat/i });
    fireEvent.click(triggerButton);

    // Now the dialog content should have a "Creating..." button
    const createButton = screen.getByRole('button', { name: /creating/i });
    expect(createButton).toBeInTheDocument();
    expect(createButton).toBeDisabled();
  });

  test('enforces max length on chat name input', () => {
    render(<NewChatModal highlight={false} />);

    const triggerButton = screen.getByRole('button', { name: /new chat/i });
    fireEvent.click(triggerButton);

    const input = screen.getByRole('textbox');
    expect(input).toHaveAttribute('maxLength', '200');
  });

  test('shows warning when chat name reaches max length', () => {
    render(<NewChatModal highlight={false} />);

    const triggerButton = screen.getByRole('button', { name: /new chat/i });
    fireEvent.click(triggerButton);

    const input = screen.getByRole('textbox');
    const longName = 'a'.repeat(200);
    fireEvent.change(input, { target: { value: longName } });

    expect(screen.getByText(/has reached maximum of 200 characters/i)).toBeInTheDocument();
  });
});
