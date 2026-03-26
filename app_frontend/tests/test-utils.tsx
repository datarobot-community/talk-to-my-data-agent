import '@testing-library/jest-dom';
import { ReactNode } from 'react';
import { render } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { AppStateProvider } from '@/state';
import { vi } from 'vitest';

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

/**
 * Mock scrollIntoView for tests since it's not available in JSDOM
 * Returns cleanup function to restore the original method
 */
export function mockScrollIntoView() {
  const originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
  const scrollIntoViewMock = vi.fn();

  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
    configurable: true,
    value: scrollIntoViewMock,
    writable: true,
  });

  return () => {
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: originalScrollIntoView,
      writable: true,
    });
  };
}
