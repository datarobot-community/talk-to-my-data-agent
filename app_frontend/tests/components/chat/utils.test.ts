import { describe, test, expect, vi } from 'vitest';
import { parsePlotData, formatMessageDate } from '@/components/chat/utils';

vi.mock('@/i18n', () => ({
  default: {
    language: 'en',
  },
}));

describe('parsePlotData', () => {
  test('returns null for undefined input', () => {
    expect(parsePlotData(undefined)).toBeNull();
  });

  test('returns null for empty string', () => {
    expect(parsePlotData('')).toBeNull();
  });

  test('parses valid JSON and includes paper_bgcolor default', () => {
    const input = JSON.stringify({ data: [{ x: [1, 2], y: [3, 4] }], layout: {} });
    const result = parsePlotData(input);
    expect(result).toHaveProperty('paper_bgcolor');
    expect(result).toHaveProperty('data');
  });

  test('allows JSON to override paper_bgcolor', () => {
    const input = JSON.stringify({ paper_bgcolor: 'red', data: [] });
    const result = parsePlotData(input);
    expect(result).toHaveProperty('paper_bgcolor', 'red');
  });

  test('returns null for invalid JSON', () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(parsePlotData('not json')).toBeNull();
    consoleSpy.mockRestore();
  });
});

describe('formatMessageDate', () => {
  test('returns empty string for undefined timestamp', () => {
    expect(formatMessageDate(undefined)).toBe('');
  });

  test('returns empty string for empty string timestamp', () => {
    expect(formatMessageDate('')).toBe('');
  });

  test('formats valid ISO timestamp', () => {
    const result = formatMessageDate('2024-06-15T14:30:00Z');
    // Should contain the date parts (month name, day, year)
    expect(result).toMatch(/June/);
    expect(result).toMatch(/15/);
    expect(result).toMatch(/2024/);
  });
});
