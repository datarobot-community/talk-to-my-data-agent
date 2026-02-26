import { describe, test, expect, vi } from 'vitest';
import { getChatName, getTranslatedMessageStep } from '@/api/chat-messages/utils';
import { IChatMessage } from '@/api/chat-messages/types';

vi.mock('@/i18n', () => ({
  default: {
    t: (key: string, opts?: Record<string, unknown>) => {
      if (!opts) return key;
      return Object.entries(opts).reduce((str, [k, v]) => str.replace(`{{${k}}}`, String(v)), key);
    },
    language: 'en',
  },
}));

describe('getChatName', () => {
  test('returns a formatted chat name string', () => {
    const name = getChatName();
    expect(name).toMatch(/^Chat /);
  });
});

describe('getTranslatedMessageStep', () => {
  const makeMessage = (step: string, reattempt = 0): IChatMessage => ({
    role: 'assistant',
    content: '',
    components: [],
    in_progress: true,
    step: { step, reattempt },
  });

  test('returns non-null for known steps, null for unknown', () => {
    expect(getTranslatedMessageStep(makeMessage('ANALYZING_QUESTION'))).not.toBeNull();
    expect(getTranslatedMessageStep(makeMessage('GENERATING_QUERY'))).not.toBeNull();
    expect(getTranslatedMessageStep(makeMessage('RUNNING_QUERY'))).not.toBeNull();
    expect(getTranslatedMessageStep(makeMessage('UNKNOWN_STEP'))).toBeNull();
  });

  test('appends attempt number on reattempt', () => {
    const result = getTranslatedMessageStep(makeMessage('GENERATING_QUERY', 2));
    expect(result).toContain('attempt 3');

    // non-retryable steps ignore reattempt
    const noRetry = getTranslatedMessageStep(makeMessage('ANALYZING_QUESTION', 2));
    expect(noRetry).not.toContain('attempt');
  });

  test('returns null for non-assistant, completed, errored, or stepless messages', () => {
    // user message
    expect(
      getTranslatedMessageStep({
        role: 'user',
        content: '',
        components: [],
        in_progress: true,
        step: { step: 'ANALYZING_QUESTION', reattempt: 0 },
      })
    ).toBeNull();

    // not in progress
    expect(
      getTranslatedMessageStep({
        role: 'assistant',
        content: '',
        components: [],
        in_progress: false,
        step: { step: 'ANALYZING_QUESTION', reattempt: 0 },
      })
    ).toBeNull();

    // has error
    expect(
      getTranslatedMessageStep({
        role: 'assistant',
        content: '',
        components: [],
        in_progress: true,
        error: 'fail',
        step: { step: 'ANALYZING_QUESTION', reattempt: 0 },
      })
    ).toBeNull();

    // no step
    expect(
      getTranslatedMessageStep({
        role: 'assistant',
        content: '',
        components: [],
        in_progress: true,
      })
    ).toBeNull();
  });
});
