import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faDownload, faTrash } from '@fortawesome/free-solid-svg-icons';
import { useTranslation } from '@/i18n';
import { Button } from '@/components/ui/button';
import { SearchControl } from '@/components/ui-custom/search-control';
import { cn } from '@/lib/utils';
import loader from '@/assets/loader.svg';

interface DatasetCardActionBarProps {
  onSearch?: (searchText: string) => void;
  onDownload?: () => void;
  onDelete?: () => void;
  isDownloading?: boolean;
  isProcessing?: boolean;
  className?: string;
  disabled?: boolean;
}

export const DatasetCardActionBar: React.FC<DatasetCardActionBarProps> = ({
  onSearch,
  onDownload,
  onDelete,
  isDownloading = false,
  isProcessing = false,
  className,
  disabled = false,
}) => {
  const { t } = useTranslation();
  const isDisabled = disabled || isProcessing || isDownloading;

  return (
    <div className={cn('flex items-center gap-1', className)}>
      {/* Search Component */}
      {onSearch && <SearchControl onSearch={onSearch} disabled={isDisabled} />}

      {/* Download Button */}
      {onDownload && (
        <Button
          variant="link"
          onClick={onDownload}
          title={t('Download dictionary as CSV')}
          disabled={isDisabled}
        >
          {isDownloading ? (
            <img src={loader} alt={t('downloading')} className="w-4 h-4 animate-spin" />
          ) : (
            <FontAwesomeIcon icon={faDownload} />
          )}
        </Button>
      )}

      {/* Delete Button */}
      {onDelete && (
        <Button
          variant="link"
          onClick={onDelete}
          title={t('Delete dictionary')}
          disabled={isDisabled}
        >
          <FontAwesomeIcon icon={faTrash} />
        </Button>
      )}
    </div>
  );
};
