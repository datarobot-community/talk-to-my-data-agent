import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { useTranslation as useI18nTranslation } from 'react-i18next';

// Import translation files
import esTranslations from './locales/es_419.json';
import frTranslations from './locales/fr.json';
import jaTranslations from './locales/ja.json';
import koTranslations from './locales/ko.json';
import ptTranslations from './locales/pt_BR.json';

const updateResources = (data: Record<string, string>) => {
  return Object.keys(data).reduce(
    (acc, key) => {
      acc[key] = data[key] || key;
      return acc;
    },
    {} as Record<string, string>
  );
};

const resources = {
  es: {
    translation: updateResources(esTranslations),
  },
  fr: {
    translation: updateResources(frTranslations),
  },
  ja: {
    translation: updateResources(jaTranslations),
  },
  ko: {
    translation: updateResources(koTranslations),
  },
  pt: {
    translation: updateResources(ptTranslations),
  },
};

const languageKey = 'TTMData_language';

export const getSavedLanguage = () => {
  return localStorage.getItem(languageKey);
};

export const saveLanguage = (language: string) => {
  localStorage.setItem(languageKey, language);
};

i18n.use(initReactI18next).init({
  resources,
  lng: getSavedLanguage() || 'en',
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false,
  },

  react: {
    useSuspense: false,
  },
});

export const useTranslation = () => {
  const { t, i18n } = useI18nTranslation();

  return {
    t,
    i18n,
    changeLanguage: i18n.changeLanguage,
    currentLanguage: i18n.language,
  };
};

export default i18n;
