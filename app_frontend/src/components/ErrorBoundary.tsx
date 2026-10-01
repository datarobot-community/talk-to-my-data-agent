/*
 * Copyright 2025 DataRobot, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import { Component, type ErrorInfo, type ReactNode } from 'react';

import { useTranslation } from '@/i18n';
import { Button } from '@/components/ui/button';

type ErrorBoundaryProps = {
  children: ReactNode;
  /** Rendered instead of `children` once a descendant throws. */
  fallback?: (props: { error: Error; reset: () => void }) => ReactNode;
  /** Distinguishes boundaries in the console when one trips. */
  label?: string;
  /** Called once when a descendant throws, for recovery (close a drawer, toast). */
  onError?: (error: Error) => void;
};

type ErrorBoundaryState = { error: Error | null };

/**
 * Stops a render-time or layout-effect throw from unmounting the whole app.
 *
 * React deliberately unmounts the entire tree when nothing catches — a single
 * component throwing then shows the user a blank page. Third-party components
 * are the usual source (see AECO-44, where a design-system `useLayoutEffect`
 * threw and took TTMData down), so wrap anything we don't control.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error(
      `[ErrorBoundary${this.props.label ? `: ${this.props.label}` : ''}]`,
      error,
      errorInfo.componentStack
    );
    this.props.onError?.(error);
  }

  reset = () => {
    this.setState({ error: null });
  };

  render() {
    const { error } = this.state;
    if (!error) {
      return this.props.children;
    }
    if (this.props.fallback) {
      return this.props.fallback({ error, reset: this.reset });
    }
    return <ErrorFallback error={error} reset={this.reset} />;
  }
}

/**
 * Default fallback. `role="alert"` so the failure is announced rather than
 * leaving a screen-reader user on a silently replaced region.
 */
function ErrorFallback({ error, reset }: { error: Error; reset: () => void }) {
  const { t } = useTranslation();

  return (
    <div
      role="alert"
      data-testid="error-boundary-fallback"
      className="flex size-full flex-col items-center justify-center gap-3 p-6 text-center"
    >
      <p className="text-sm font-semibold">{t('Something went wrong')}</p>
      <p className="max-w-prose text-sm text-muted-foreground">{error.message}</p>
      <Button variant="secondary" onClick={reset} testId="error-boundary-retry">
        {t('Try again')}
      </Button>
    </div>
  );
}
