import { ReactNode } from 'react';
import { render } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { AppStateProvider } from '@/state';

const createTestQueryClient = () =>
    new QueryClient({
        defaultOptions: {
            queries: {
                retry: false,
            },
        },
    });

export function renderWithProviders(children: ReactNode) {
    const queryClient = createTestQueryClient();

    return render(
        <MemoryRouter>
            <QueryClientProvider client={queryClient}>
                <AppStateProvider>{children}</AppStateProvider>
            </QueryClientProvider>
        </MemoryRouter>
    );
}
