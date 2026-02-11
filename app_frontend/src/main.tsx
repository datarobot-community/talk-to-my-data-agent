import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter as Router } from 'react-router-dom';
import { I18nextProvider } from 'react-i18next';
import '@fontsource/dm-sans/300.css';
import '@fontsource/dm-sans/400.css';
import '@fontsource/dm-sans/500.css';
import '@fontsource/dm-sans/600.css';
import '@fontsource/dm-sans/700.css';
import { getBaseUrl } from '@/lib/utils.ts';
import './index.css';
import App from './App.tsx';
import { AppStateProvider } from './state';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import i18n from './i18n';

const basename = getBaseUrl();

const queryClient = new QueryClient();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <I18nextProvider i18n={i18n}>
        <Router basename={basename}>
          <AppStateProvider>
            <App />
          </AppStateProvider>
        </Router>
      </I18nextProvider>
    </QueryClientProvider>
  </StrictMode>
);
