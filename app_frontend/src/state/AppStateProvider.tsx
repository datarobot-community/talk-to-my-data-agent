import React, { useReducer } from 'react';
import { AppState } from './types';
import { reducer, createInitialState, actions } from './reducer';
import { AppStateContext } from './AppStateContext';

export const AppStateProvider: React.FC<{
    children: React.ReactNode;
}> = ({ children }) => {
    const [state, dispatch] = useReducer(reducer, createInitialState());

    const hideWelcomeModal = () => {
        dispatch(actions.hideWelcomeModal());
    };

    const setCollapsiblePanelDefaultOpen = (isOpen: boolean) => {
        dispatch(actions.setCollapsiblePanelDefaultOpen(isOpen));
    };

    const setEnableChartGeneration = (enabled: boolean) => {
        dispatch(actions.setEnableChartGeneration(enabled));
    };

    const setEnableBusinessInsights = (enabled: boolean) => {
        dispatch(actions.setEnableBusinessInsights(enabled));
    };

    const setDataSource = (source: string) => {
        dispatch(actions.setDataSource(source));
    };

    const contextValue: AppState = {
        ...state,
        hideWelcomeModal,
        setCollapsiblePanelDefaultOpen,
        setEnableChartGeneration,
        setEnableBusinessInsights,
        setDataSource,
    };

    return <AppStateContext.Provider value={contextValue}>{children}</AppStateContext.Provider>;
};
