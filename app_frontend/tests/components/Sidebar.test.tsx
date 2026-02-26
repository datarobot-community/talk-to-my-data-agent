import { screen, fireEvent } from '@testing-library/react';
import { describe, test, expect, vi, beforeEach, type Mock } from 'vitest';
import { Sidebar } from '@/components/Sidebar';
import { renderWithProviders } from '../test-utils';
import { useGeneratedDictionaries } from '@/api/dictionaries';
import { useFetchAllChats } from '@/api/chat-messages';

vi.mock('@/api/dictionaries', () => ({
  useGeneratedDictionaries: vi.fn(),
  getDictionariesMenu: vi.fn(),
}));

vi.mock('@/api/chat-messages', () => ({
  useFetchAllChats: vi.fn(),
  getChatsMenu: vi.fn(),
}));

// Mock matchMedia for SidebarProvider/useMobile
beforeEach(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
});

// Mock child components that have complex dependencies
vi.mock('@/components/AddDataModal', () => ({
  AddDataModal: ({ highlight }: { highlight?: boolean }) => (
    <button data-testid="add-data-button" data-highlight={highlight}>
      Add Data
    </button>
  ),
}));

vi.mock('@/components/NewChatModal', () => ({
  NewChatModal: ({ highlight }: { highlight?: boolean }) => (
    <button data-testid="new-chat-button" data-highlight={highlight}>
      New Chat
    </button>
  ),
}));

vi.mock('@/components/SettingsModal', () => ({
  SettingsModal: ({ isOpen }: { isOpen: boolean; onOpenChange: (v: boolean) => void }) =>
    isOpen ? <div data-testid="settings-modal">Settings Modal</div> : null,
}));

function setupMocks(overrides?: {
  datasets?: Array<{ name: string; key: string }>;
  datasetsLoading?: boolean;
  chats?: Array<{ id: string; name: string; key: string }>;
  chatsLoading?: boolean;
}) {
  const datasets = overrides?.datasets ?? [];
  const chats = overrides?.chats ?? [];

  (useGeneratedDictionaries as Mock).mockImplementation((opts?: any) => {
    if (opts?.select) {
      return {
        data: datasets.map(d => ({ ...d, label: d.name })),
        isLoading: overrides?.datasetsLoading ?? false,
      };
    }
    return {
      data: datasets,
      isLoading: overrides?.datasetsLoading ?? false,
    };
  });

  (useFetchAllChats as Mock).mockImplementation((opts?: any) => {
    if (opts?.select) {
      return {
        data: chats.map(c => ({ ...c, label: c.name })),
        isLoading: overrides?.chatsLoading ?? false,
      };
    }
    return {
      data: chats,
      isLoading: overrides?.chatsLoading ?? false,
    };
  });
}

describe('Sidebar', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupMocks();
  });

  test('shows "Add your data here" when no datasets', () => {
    setupMocks({ datasets: [] });
    renderWithProviders(<Sidebar />);
    expect(screen.getByTestId('empty-datasets')).toBeInTheDocument();
  });

  test('shows "Start your first chat here" when no chats', () => {
    setupMocks({
      datasets: [{ name: 'data1', key: 'd1' }],
      chats: [],
    });
    renderWithProviders(<Sidebar />);
    expect(screen.getByTestId('empty-chats')).toBeInTheDocument();
  });

  test('highlights Add Data button when no datasets', () => {
    setupMocks({ datasets: [], datasetsLoading: false });
    renderWithProviders(<Sidebar />);
    const addDataBtn = screen.getByTestId('add-data-button');
    expect(addDataBtn.getAttribute('data-highlight')).toBe('true');
  });

  test('highlights New Chat button when datasets exist but no chats', () => {
    setupMocks({
      datasets: [{ name: 'data1', key: 'd1' }],
      chats: [],
      chatsLoading: false,
    });
    renderWithProviders(<Sidebar />);
    const newChatBtn = screen.getByTestId('new-chat-button');
    expect(newChatBtn.getAttribute('data-highlight')).toBe('true');
  });

  test('Settings button opens settings modal', () => {
    renderWithProviders(<Sidebar />);
    fireEvent.click(screen.getByTestId('settings-button'));
    expect(screen.getByTestId('settings-modal')).toBeInTheDocument();
  });
});
