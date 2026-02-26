import React from 'react';
import { useTranslation } from '@/i18n';
import { Button } from '@/components/ui/button';
import { DATA_TABS } from '@/state/constants';
import { SearchControl } from '@/components/ui-custom/search-control';
import { cn } from '@/lib/utils';
import { Download, Trash2, Loader2 } from 'lucide-react';

interface DatasetCardActionBarProps {
  onSearch?: (searchText: string) => void;
  onDownload?: () => void;
  onDelete?: () => void;
  isDownloading?: boolean;
  isProcessing?: boolean;
  className?: string;
  disabled?: boolean;
  viewMode?: string;
}

export const DatasetCardActionBar: React.FC<DatasetCardActionBarProps> = ({
  onSearch,
  onDownload,
  onDelete,
  isDownloading = false,
  isProcessing = false,
  className,
  disabled = false,
  viewMode,
}) => {
  const { t } = useTranslation();
  const isDisabled = disabled || isProcessing || isDownloading;
  const searchLabel = viewMode === DATA_TABS.DESCRIPTION ? t('Search') : t('Search columns');

  return (
    <div className={cn('flex items-center gap-1', className)}>
      {/* Search Component */}
      {onSearch && (
        <SearchControl
          key={viewMode} // Reset when view changes
          onSearch={onSearch}
          disabled={isDisabled}
          searchLabel={searchLabel}
        />
      )}

      {/* Download Button */}
      {onDownload && (
        <Button
          variant="link"
          onClick={onDownload}
          title={t('Download dictionary as CSV')}
          disabled={isDisabled}
          data-testid="download-dictionary-button"
        >
          {isDownloading ? <Loader2 className="size-4 animate-spin" /> : <Download />}
        </Button>
      )}

      {/* Delete Button */}
      {onDelete && (
        <Button variant="link" onClick={onDelete} title={t('Delete dataset')} disabled={isDisabled}>
          <Trash2 />
        </Button>
      )}
    </div>
  );
};
