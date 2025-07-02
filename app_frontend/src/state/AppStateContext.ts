import { createContext } from 'react';
import { AppState } from './types';

export const AppStateContext = createContext<AppState>({} as AppState);
