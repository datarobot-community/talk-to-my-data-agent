import { renderHook, waitFor } from '@testing-library/react';
import { describe, test, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import { server } from '../../__mocks__/node';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createElement, type ReactNode } from 'react';
import { useGetDatabaseTables } from '@/api/database/hooks';

// A fresh client with NO retry override, so the hook's own retry:false governs.
// axios-retry is left real so the per-request `retries: 0` override is exercised.
function createWrapper() {
  const queryClient = new QueryClient();
  return ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client: queryClient }, children);
}

describe('useGetDatabaseTables', () => {
  test('does not fetch while disabled (modal closed)', () => {
    let calls = 0;
    server.use(
      http.get('*/v1/database/tables', () => {
        calls++;
        return HttpResponse.json([]);
      })
    );

    const { result } = renderHook(() => useGetDatabaseTables(false), {
      wrapper: createWrapper(),
    });

    expect(result.current.fetchStatus).toBe('idle');
    expect(calls).toBe(0);
  });

  test('surfaces an outage on the first attempt without a retry storm', async () => {
    let calls = 0;
    server.use(
      http.get('*/v1/database/tables', () => {
        calls++;
        return new HttpResponse('unavailable', { status: 503 });
      })
    );

    const { result } = renderHook(() => useGetDatabaseTables(true), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    // axios-retry `retries: 0` + React Query `retry: false` => exactly one request.
    expect(calls).toBe(1);
  });
});
