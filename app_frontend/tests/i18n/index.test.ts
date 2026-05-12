import { describe, test, expect, afterAll } from 'vitest';
import i18n from '@/i18n';

describe('i18n <html lang> sync', () => {
  afterAll(async () => {
    await i18n.changeLanguage('en');
  });

  test('initializes document.documentElement.lang from i18n default', () => {
    expect(document.documentElement.lang).toBe('en');
  });

  test('updates document.documentElement.lang on language change, normalizing underscores to hyphens', async () => {
    await i18n.changeLanguage('pt_BR');
    expect(document.documentElement.lang).toBe('pt-BR');

    await i18n.changeLanguage('fr');
    expect(document.documentElement.lang).toBe('fr');
  });
});
