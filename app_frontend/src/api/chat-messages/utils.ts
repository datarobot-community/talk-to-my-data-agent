import i18n from '@/i18n';
import { IChatMessage } from './types';

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

export const getTranslatedMessageStep = (message: IChatMessage) => {
  if (
    message &&
    message.role !== 'user' &&
    message.in_progress &&
    !message.error &&
    message.step &&
    message.step.step
  ) {
    switch (message.step.step) {
      case 'ANALYZING_QUESTION':
        return i18n.t('Analyzing Question');
      case 'TESTING_CONNECTION':
        return i18n.t('Testing Connection');
      case 'GENERATING_QUERY':
        if (message.step.reattempt > 0) {
          return i18n.t('Generating Query (attempt {{attempt}})', {
            attempt: message.step.reattempt + 1,
          });
        }
        return i18n.t('Generating Query');
      case 'RUNNING_QUERY':
        if (message.step.reattempt > 0) {
          return i18n.t('Running Query (attempt {{attempt}})', {
            attempt: message.step.reattempt + 1,
          });
        }
        return i18n.t('Running Query');
      case 'ANALYZING_RESULTS':
        return i18n.t('Analyzing Results');
      default:
        return null;
    }
  }
  return null;
};
