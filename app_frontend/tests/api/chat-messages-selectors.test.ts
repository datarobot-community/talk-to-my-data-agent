import { describe, test, expect } from 'vitest';
import { getResponseMessage } from '@/api/chat-messages/selectors';
import { IChatMessage } from '@/api/chat-messages/types';

describe('getResponseMessage', () => {
  test('finds assistant message when system message is in between', () => {
    const messages: IChatMessage[] = [
      {
        id: 'user-1',
        role: 'user',
        content: 'Question',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'system-1',
        role: 'system',
        content: 'Summary of conversation',
        components: [],
        created_at: '2024-01-01T00:01:00Z',
      },
      {
        id: 'assistant-1',
        role: 'assistant',
        content: 'Answer',
        components: [],
        created_at: '2024-01-01T00:02:00Z',
      },
    ];

    const result = getResponseMessage(messages, 'user-1');

    expect(result).toEqual(messages[2]);
    expect(result?.id).toBe('assistant-1');
  });

  test('returns undefined when no assistant message found after user message', () => {
    const messages: IChatMessage[] = [
      {
        id: 'user-1',
        role: 'user',
        content: 'Question',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'system-1',
        role: 'system',
        content: 'Summary',
        components: [],
        created_at: '2024-01-01T00:01:00Z',
      },
    ];

    const result = getResponseMessage(messages, 'user-1');

    expect(result).toBeUndefined();
  });

  test('returns undefined when message id not found', () => {
    const messages: IChatMessage[] = [
      {
        id: 'user-1',
        role: 'user',
        content: 'Question',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
    ];

    const result = getResponseMessage(messages, 'non-existent-id');

    expect(result).toBeUndefined();
  });

  test('returns assistant message directly when given assistant message id', () => {
    const messages: IChatMessage[] = [
      {
        id: 'user-1',
        role: 'user',
        content: 'Question',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'assistant-1',
        role: 'assistant',
        content: 'Answer',
        components: [],
        created_at: '2024-01-01T00:01:00Z',
      },
    ];

    const result = getResponseMessage(messages, 'assistant-1');

    expect(result).toEqual(messages[1]);
  });

  test('works correctly without system messages (backward compatibility)', () => {
    const messages: IChatMessage[] = [
      {
        id: 'user-1',
        role: 'user',
        content: 'Question',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'assistant-1',
        role: 'assistant',
        content: 'Answer',
        components: [],
        created_at: '2024-01-01T00:01:00Z',
      },
    ];

    const result = getResponseMessage(messages, 'user-1');

    expect(result).toEqual(messages[1]);
  });

  test('returns undefined for an unpaired user message followed by a later exchange', () => {
    // A failed question can be persisted with no assistant reply of its own (APP-6805);
    // its response must not be the NEXT question's answer.
    const messages: IChatMessage[] = [
      {
        id: 'user-1',
        role: 'user',
        content: 'Failed question',
        components: [],
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'user-2',
        role: 'user',
        content: 'Next question',
        components: [],
        created_at: '2024-01-01T00:01:00Z',
      },
      {
        id: 'assistant-2',
        role: 'assistant',
        content: 'Answer to next question',
        components: [],
        created_at: '2024-01-01T00:02:00Z',
      },
    ];

    // user-1 is unpaired; user-2 still pairs with its own answer.
    expect(getResponseMessage(messages, 'user-1')).toBeUndefined();
    expect(getResponseMessage(messages, 'user-2')).toEqual(messages[2]);
  });
});
