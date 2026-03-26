import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, test, expect, vi, beforeEach } from 'vitest';
import { http, HttpResponse } from 'msw';
import { server } from '../../__mocks__/node';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { createElement, type ReactNode } from 'react';
import { toast } from 'sonner';
import { messageKeys } from '@/api/chat-messages/keys';
import { IChat, IChatMessage } from '@/api/chat-messages/types';
import {
  usePostMessage,
  useDeleteMessage,
  useCreateChat,
  useDeleteChat,
  useRenameChat,
  useUpdateChatDataSource,
} from '@/api/chat-messages/hooks';

vi.mock('sonner', () => ({
  toast: { error: vi.fn() },
}));

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return { ...actual, useNavigate: () => mockNavigate };
});

function createTestEnv() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    createElement(
      MemoryRouter,
      null,
      createElement(QueryClientProvider, { client: queryClient }, children)
    );

  return { queryClient, wrapper };
}

const chatFixtures: IChat[] = [
  { id: 'chat-1', name: 'First Chat', created_at: '2024-01-01T00:00:00Z', data_source: 'ds-1' },
  { id: 'chat-2', name: 'Second Chat', created_at: '2024-01-02T00:00:00Z' },
];

const messageFixtures: IChatMessage[] = [
  {
    id: 'msg-1',
    role: 'user',
    content: 'Hello',
    components: [],
    created_at: '2024-01-01T00:00:00Z',
  },
  {
    id: 'msg-2',
    role: 'assistant',
    content: 'Hi there',
    components: [],
    created_at: '2024-01-01T00:01:00Z',
  },
];

beforeEach(() => {
  vi.clearAllMocks();
});

describe('usePostMessage', () => {
  test('optimistically adds user message to cache', async () => {
    server.use(
      http.post('*/v1/chats/:chatId/messages', async () => {
        // Delay so we can inspect optimistic state
        await new Promise(r => setTimeout(r, 50));
        return HttpResponse.json({
          id: 'chat-1',
          name: 'First Chat',
          messages: [
            ...messageFixtures,
            { id: 'msg-3', role: 'user', content: 'New message', components: [] },
          ],
        });
      })
    );

    const { queryClient, wrapper } = createTestEnv();
    queryClient.setQueryData(messageKeys.messages('chat-1'), messageFixtures);

    const { result } = renderHook(() => usePostMessage(), { wrapper });

    act(() => {
      result.current.mutate({ message: 'New message', chatId: 'chat-1' });
    });

    // Optimistic update should immediately add the message
    await waitFor(() => {
      const cached = queryClient.getQueryData<IChatMessage[]>(messageKeys.messages('chat-1'));
      expect(cached).toHaveLength(3);
      expect(cached![2].content).toBe('New message');
      expect(cached![2].role).toBe('user');
      expect(cached![2].in_progress).toBe(true);
    });
  });

  test('rolls back messages on error and shows toast', async () => {
    server.use(
      http.post('*/v1/chats/:chatId/messages', () => HttpResponse.json(null, { status: 500 }))
    );

    const { queryClient, wrapper } = createTestEnv();
    queryClient.setQueryData(messageKeys.messages('chat-1'), messageFixtures);

    const { result } = renderHook(() => usePostMessage(), { wrapper });

    act(() => {
      result.current.mutate({ message: 'Will fail', chatId: 'chat-1' });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    // Messages rolled back to original
    const cached = queryClient.getQueryData<IChatMessage[]>(messageKeys.messages('chat-1'));
    expect(cached).toEqual(messageFixtures);
    expect(toast.error).toHaveBeenCalledWith('There was a problem sending your message.');
  });

  test('on success for new chat, adds chat to list and navigates', async () => {
    server.use(
      http.post('*/v1/chats/messages', () =>
        HttpResponse.json({
          id: 'new-chat-id',
          name: 'New Chat',
          messages: [{ id: 'msg-new', role: 'user', content: 'Hello', components: [] }],
        })
      )
    );

    const { queryClient, wrapper } = createTestEnv();
    queryClient.setQueryData(messageKeys.chats, chatFixtures);

    const { result } = renderHook(() => usePostMessage(), { wrapper });

    act(() => {
      result.current.mutate({ message: 'Hello' }); // no chatId = new chat
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const chats = queryClient.getQueryData<IChat[]>(messageKeys.chats);
    expect(chats!.some(c => c.id === 'new-chat-id')).toBe(true);
    expect(mockNavigate).toHaveBeenCalledWith('/chats/new-chat-id');
  });

  test('on success for new chat, does not duplicate if chat already exists', async () => {
    server.use(
      http.post('*/v1/chats/messages', () =>
        HttpResponse.json({
          id: 'chat-1', // already exists
          name: 'First Chat',
          messages: [{ id: 'msg-new', role: 'user', content: 'Hello', components: [] }],
        })
      )
    );

    const { queryClient, wrapper } = createTestEnv();
    queryClient.setQueryData(messageKeys.chats, chatFixtures);

    const { result } = renderHook(() => usePostMessage(), { wrapper });

    act(() => {
      result.current.mutate({ message: 'Hello' });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const chats = queryClient.getQueryData<IChat[]>(messageKeys.chats);
    expect(chats!.filter(c => c.id === 'chat-1')).toHaveLength(1);
  });
});

describe('useDeleteMessage', () => {
  test('optimistically removes message from cache', async () => {
    server.use(
      http.delete('*/v1/chats/messages/:messageId', async () => {
        await new Promise(r => setTimeout(r, 50));
        return HttpResponse.json([messageFixtures[1]]);
      })
    );

    const { queryClient, wrapper } = createTestEnv();
    queryClient.setQueryData(messageKeys.messages('chat-1'), messageFixtures);

    const { result } = renderHook(() => useDeleteMessage(), { wrapper });

    act(() => {
      result.current.mutate({ messageId: 'msg-1', chatId: 'chat-1' });
    });

    await waitFor(() => {
      const cached = queryClient.getQueryData<IChatMessage[]>(messageKeys.messages('chat-1'));
      expect(cached).toHaveLength(1);
      expect(cached![0].id).toBe('msg-2');
    });
  });

  test('rolls back on error and shows toast', async () => {
    server.use(
      http.delete('*/v1/chats/messages/:messageId', () => HttpResponse.json(null, { status: 500 }))
    );

    const { queryClient, wrapper } = createTestEnv();
    queryClient.setQueryData(messageKeys.messages('chat-1'), messageFixtures);

    const { result } = renderHook(() => useDeleteMessage(), { wrapper });

    act(() => {
      result.current.mutate({ messageId: 'msg-1', chatId: 'chat-1' });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    const cached = queryClient.getQueryData<IChatMessage[]>(messageKeys.messages('chat-1'));
    expect(cached).toEqual(messageFixtures);
    expect(toast.error).toHaveBeenCalledWith('There was a problem deleting the message.');
  });
});

describe('useRenameChat', () => {
  test('optimistically updates chat name in cache', async () => {
    server.use(
      http.put('*/v1/chats/:chatId', async () => {
        await new Promise(r => setTimeout(r, 50));
        return new HttpResponse(null, { status: 200 });
      })
    );

    const { queryClient, wrapper } = createTestEnv();
    queryClient.setQueryData(messageKeys.chats, chatFixtures);

    const { result } = renderHook(() => useRenameChat(), { wrapper });

    act(() => {
      result.current.mutate({ chatId: 'chat-1', name: 'Renamed Chat' });
    });

    await waitFor(() => {
      const chats = queryClient.getQueryData<IChat[]>(messageKeys.chats);
      expect(chats!.find(c => c.id === 'chat-1')!.name).toBe('Renamed Chat');
      // Other chats untouched
      expect(chats!.find(c => c.id === 'chat-2')!.name).toBe('Second Chat');
    });
  });

  test('rolls back chat name on error and shows toast', async () => {
    server.use(http.put('*/v1/chats/:chatId', () => HttpResponse.json(null, { status: 500 })));

    const { queryClient, wrapper } = createTestEnv();
    queryClient.setQueryData(messageKeys.chats, chatFixtures);

    const { result } = renderHook(() => useRenameChat(), { wrapper });

    act(() => {
      result.current.mutate({ chatId: 'chat-1', name: 'Will Fail' });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    const chats = queryClient.getQueryData<IChat[]>(messageKeys.chats);
    expect(chats!.find(c => c.id === 'chat-1')!.name).toBe('First Chat');
    expect(toast.error).toHaveBeenCalledWith('There was a problem renaming the chat.');
  });
});

describe('useUpdateChatDataSource', () => {
  test('optimistically updates data source in cache', async () => {
    server.use(
      http.put('*/v1/chats/:chatId', async () => {
        await new Promise(r => setTimeout(r, 50));
        return new HttpResponse(null, { status: 200 });
      })
    );

    const { queryClient, wrapper } = createTestEnv();
    queryClient.setQueryData(messageKeys.chats, chatFixtures);

    const { result } = renderHook(() => useUpdateChatDataSource(), { wrapper });

    act(() => {
      result.current.mutate({ chatId: 'chat-1', dataSource: 'ds-new' });
    });

    await waitFor(() => {
      const chats = queryClient.getQueryData<IChat[]>(messageKeys.chats);
      expect(chats!.find(c => c.id === 'chat-1')!.data_source).toBe('ds-new');
      // Other chats untouched
      expect(chats!.find(c => c.id === 'chat-2')!.data_source).toBeUndefined();
    });
  });

  test('rolls back data source on error and shows toast', async () => {
    server.use(http.put('*/v1/chats/:chatId', () => HttpResponse.json(null, { status: 500 })));

    const { queryClient, wrapper } = createTestEnv();
    queryClient.setQueryData(messageKeys.chats, chatFixtures);

    const { result } = renderHook(() => useUpdateChatDataSource(), { wrapper });

    act(() => {
      result.current.mutate({ chatId: 'chat-1', dataSource: 'will-fail' });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    const chats = queryClient.getQueryData<IChat[]>(messageKeys.chats);
    expect(chats!.find(c => c.id === 'chat-1')!.data_source).toBe('ds-1');
    expect(toast.error).toHaveBeenCalledWith('There was a problem updating the data source.');
  });
});

describe('useDeleteChat', () => {
  test('calls onSuccess callback after successful deletion', async () => {
    server.use(
      http.delete('*/v1/chats/:chatId', () => new HttpResponse(null, { status: 200 })),
      http.get('*/v1/chats', () => HttpResponse.json([chatFixtures[1]]))
    );

    const { queryClient, wrapper } = createTestEnv();
    queryClient.setQueryData(messageKeys.chats, chatFixtures);

    const onSuccess = vi.fn();
    const { result } = renderHook(() => useDeleteChat({ onSuccess }), { wrapper });

    act(() => {
      result.current.mutate({ chatId: 'chat-1' });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(onSuccess).toHaveBeenCalled();
  });

  test('shows toast on error', async () => {
    server.use(http.delete('*/v1/chats/:chatId', () => HttpResponse.json(null, { status: 500 })));

    const { wrapper } = createTestEnv();
    const { result } = renderHook(() => useDeleteChat({}), { wrapper });

    act(() => {
      result.current.mutate({ chatId: 'chat-1' });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(toast.error).toHaveBeenCalledWith('There was a problem deleting the chat.');
  });
});

describe('useCreateChat', () => {
  test('shows toast on error', async () => {
    server.use(http.post('*/v1/chats', () => HttpResponse.json(null, { status: 500 })));

    const { wrapper } = createTestEnv();
    const { result } = renderHook(() => useCreateChat(), { wrapper });

    act(() => {
      result.current.mutate({ name: 'Test', dataSource: 'ds-1' });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(toast.error).toHaveBeenCalledWith('There was a problem creating the chat.');
  });
});
