import { useTranslation } from '@/i18n';
import { Loader2 } from 'lucide-react';

export const Loading = () => {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col flex-1 items-center justify-center h-full">
      <Loader2 className="w-4 h-4 animate-spin" />
      <span className="ml-2">{t('Loading...')}</span>
    </div>
  );
};
