import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { VITE_STATIC_DEFAULT_PORT, VITE_DEFAULT_PORT } from '@/constants/dev';
import { useRef, useEffect, useCallback } from 'react';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getApiPort() {
  return window.ENV?.API_PORT || VITE_STATIC_DEFAULT_PORT;
}

export function getBaseUrl() {
  let basename = window.ENV?.BASE_PATH;
  // Adjust API URL based on the environment
  const pathname: string = window.location.pathname;

  if (pathname?.includes('notebook-sessions') && pathname?.includes(`/${VITE_DEFAULT_PORT}/`)) {
    // ex:. /notebook-sessions/{id}/ports/5137/
    basename = import.meta.env.BASE_URL;
  }

  return basename ? basename : '/';
}

export function getApiUrl() {
  return `${window.location.origin}${getBaseUrl()}api`;
}

export function useDebounce<T extends (...args: unknown[]) => unknown>(func: T, delay: number) {
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const debouncedFunc = useCallback(
    (...args: Parameters<T>) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = setTimeout(() => {
        func(...args);
      }, delay);
    },
    [func, delay]
  );

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return debouncedFunc;
}
