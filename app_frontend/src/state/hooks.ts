import { useContext } from 'react';
import { AppStateContext } from './AppStateContext';
import { AppState } from './types';

export const useAppState = (): AppState => {
    const context = useContext(AppStateContext);

    if (context === undefined) {
        throw new Error('useAppState must be used within an AppStateProvider');
    }

    return context;
};
