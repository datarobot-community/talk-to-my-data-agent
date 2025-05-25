import { STORAGE_KEYS } from './constants';

const getAppIdFromUrl = (): string => {
  const url = window.location.href;
  const match = url.match(/\/custom_applications\/([^/]+)/);
  return match ? match[1] : '';
};

/**
 * Creates a prefixed key in the format /custom_applications/{id}/{key}
 */
const getPrefixedKey = (key: string): string => {
  const appId = getAppIdFromUrl();
  return appId ? `/custom_applications/${appId}/${key}` : key;
};

export const getStorageItem = (key: string): string | null => {
  return localStorage.getItem(getPrefixedKey(key));
};

export const setStorageItem = (key: string, value: string): void => {
  localStorage.setItem(getPrefixedKey(key), value);
};

export const isWelcomeModalHidden = (): boolean => {
  return getStorageItem(STORAGE_KEYS.HIDE_WELCOME_MODAL) === 'true';
};

export const isCollapsiblePanelDefaultOpen = (): boolean => {
  return getStorageItem(STORAGE_KEYS.COLLAPSIBLE_PANEL_DEFAULT_OPEN) === 'true';
};

export const setCollapsiblePanelDefaultOpen = (isOpen: boolean): void => {
  setStorageItem(STORAGE_KEYS.COLLAPSIBLE_PANEL_DEFAULT_OPEN, isOpen ? 'true' : 'false');
};

export const isExpandGraphsInsightsDefaultOpen = (): boolean => {
  return getStorageItem(STORAGE_KEYS.EXPAND_GRAPHS_INSIGHTS_DEFAULT_OPEN) !== 'false';
};

export const setExpandGraphsInsightsDefaultOpen = (isOpen: boolean): void => {
  setStorageItem(STORAGE_KEYS.EXPAND_GRAPHS_INSIGHTS_DEFAULT_OPEN, isOpen ? 'true' : 'false');
};
