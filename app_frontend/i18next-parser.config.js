const LANGUAGE_CODES = {
  SPANISH_LA: 'es_419',
  FRENCH: 'fr',
  JAPANESE: 'ja',
  KOREAN: 'ko',
  BRAZILIAN_PORTUGUESE: 'pt_BR',
};

export default {
  namespaceSeparator: false,
  keySeparator: false,
  defaultNamespace: 'translation',
  locales: Object.values(LANGUAGE_CODES),
  output: './src/i18n/locales/$LOCALE.json',
  defaultValue: () => '',
};
