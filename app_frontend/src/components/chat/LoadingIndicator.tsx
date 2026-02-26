import React from 'react';
import { Check, TriangleAlert, Loader2 } from 'lucide-react';
import { useTranslation } from '@/i18n';

interface LoadingIndicatorProps {
  isLoading?: boolean;
  hasError?: boolean;
  successTestId?: string;
}

export const LoadingIndicator: React.FC<LoadingIndicatorProps> = ({
  isLoading = true,
  hasError = false,
  successTestId = 'data-loading-success',
}) => {
  const { t } = useTranslation();

  if (hasError) {
    return (
      <TriangleAlert
        className="mr-2 size-4 text-destructive"
        aria-label={t('Error occurred during processing')}
      />
    );
  }

  return isLoading ? (
    <Loader2 className="mr-2 size-4 animate-spin" />
  ) : (
    <Check className="mr-2 size-4" data-testid={successTestId} />
  );
};
