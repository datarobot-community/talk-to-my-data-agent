import i18n from '@/i18n';

const languageMap = {
  es_419: 'es',
  pt_BR: 'pt',
};

const getLanguageByLanguageCode = (languageCode: string) => {
  return languageMap[languageCode as keyof typeof languageMap] || languageCode;
};

const configureChatName = (language: string) => {
  const now = new Date();
  const formattedDate = new Intl.DateTimeFormat(language, {
    month: 'long',
    day: 'numeric',
  }).format(now);

  const formattedTime = new Intl.DateTimeFormat(language, {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  }).format(now);
  return i18n.t('Chat {{formattedDate}} {{formattedTime}}', { formattedDate, formattedTime });
};

export const getChatName = () => {
  const language = getLanguageByLanguageCode(i18n.language);

  try {
    return configureChatName(language);
  } catch {
    return configureChatName('en');
  }
};
