import { useTranslation } from '@/i18n';
import { Loader2 } from 'lucide-react';

export const Loading = () => {
  const { t } = useTranslation();
  return (
    <div data-testid="loading" className="flex h-full flex-1 flex-col items-center justify-center">
      <Loader2 className="size-4 animate-spin" />
      <span className="ml-2">{t('Loading...')}</span>
    </div>
  );
};
