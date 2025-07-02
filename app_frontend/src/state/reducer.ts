import { AppStateData, Action } from './types';
import { ACTION_TYPES, STORAGE_KEYS } from './constants';
import { getStorageItem, setStorageItem } from './storage';

import { DATA_SOURCES } from '@/constants/dataSources';

export const createInitialState = (): AppStateData => {
    return {
        showWelcome: getStorageItem(STORAGE_KEYS.HIDE_WELCOME_MODAL) !== 'true',
        collapsiblePanelDefaultOpen:
            getStorageItem(STORAGE_KEYS.COLLAPSIBLE_PANEL_DEFAULT_OPEN) === 'true',
        enableChartGeneration: getStorageItem(STORAGE_KEYS.ENABLE_CHART_GENERATION) !== 'false', // Enable by default
        enableBusinessInsights: getStorageItem(STORAGE_KEYS.ENABLE_BUSINESS_INSIGHTS) !== 'false', // Enable by default
        dataSource: getStorageItem(STORAGE_KEYS.DATA_SOURCE) || DATA_SOURCES.FILE, // Default to FILE
    };
};

export const reducer = (state: AppStateData, action: Action): AppStateData => {
    switch (action.type) {
        case ACTION_TYPES.HIDE_WELCOME_MODAL:
            setStorageItem(STORAGE_KEYS.HIDE_WELCOME_MODAL, 'true');
            return {
                ...state,
                showWelcome: false,
            };
        case ACTION_TYPES.SET_COLLAPSIBLE_PANEL_DEFAULT_OPEN:
            setStorageItem(
                STORAGE_KEYS.COLLAPSIBLE_PANEL_DEFAULT_OPEN,
                action.payload ? 'true' : 'false'
            );
            return {
                ...state,
                collapsiblePanelDefaultOpen: action.payload,
            };
        case ACTION_TYPES.SET_ENABLE_CHART_GENERATION:
            setStorageItem(STORAGE_KEYS.ENABLE_CHART_GENERATION, action.payload ? 'true' : 'false');
            return {
                ...state,
                enableChartGeneration: action.payload,
            };
        case ACTION_TYPES.SET_ENABLE_BUSINESS_INSIGHTS:
            setStorageItem(
                STORAGE_KEYS.ENABLE_BUSINESS_INSIGHTS,
                action.payload ? 'true' : 'false'
            );
            return {
                ...state,
                enableBusinessInsights: action.payload,
            };
        case ACTION_TYPES.SET_DATA_SOURCE:
            setStorageItem(STORAGE_KEYS.DATA_SOURCE, action.payload);
            return {
                ...state,
                dataSource: action.payload,
            };
        default:
            return state;
    }
};

export const actions = {
    hideWelcomeModal: (): Action => ({
        type: ACTION_TYPES.HIDE_WELCOME_MODAL,
    }),
    setCollapsiblePanelDefaultOpen: (isOpen: boolean): Action => ({
        type: ACTION_TYPES.SET_COLLAPSIBLE_PANEL_DEFAULT_OPEN,
        payload: isOpen,
    }),
    setEnableChartGeneration: (enabled: boolean): Action => ({
        type: ACTION_TYPES.SET_ENABLE_CHART_GENERATION,
        payload: enabled,
    }),
    setEnableBusinessInsights: (enabled: boolean): Action => ({
        type: ACTION_TYPES.SET_ENABLE_BUSINESS_INSIGHTS,
        payload: enabled,
    }),
    setDataSource: (source: string): Action => ({
        type: ACTION_TYPES.SET_DATA_SOURCE,
        payload: source,
    }),
};
