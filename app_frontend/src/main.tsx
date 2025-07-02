import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter as Router } from 'react-router-dom';
import '@fontsource/dm-sans/300.css';
import '@fontsource/dm-sans/400.css';
import '@fontsource/dm-sans/500.css';
import '@fontsource/dm-sans/600.css';
import '@fontsource/dm-sans/700.css';
import { isServedStatic } from '@/lib/utils.ts';
import './index.css';
import App from './App.tsx';
import { AppStateProvider } from './state';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

let basename = window.ENV?.APP_BASE_URL ?? undefined;
if (
    window.ENV?.APP_BASE_URL?.includes('notebook-sessions') &&
    window.ENV?.API_PORT &&
    isServedStatic()
) {
    basename += `/ports/` + window.ENV.API_PORT;
}

const queryClient = new QueryClient();

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <QueryClientProvider client={queryClient}>
            <Router basename={basename}>
                <AppStateProvider>
                    <App />
                </AppStateProvider>
            </Router>
        </QueryClientProvider>
    </StrictMode>
);
