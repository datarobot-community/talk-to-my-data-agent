import { useState, forwardRef } from 'react';
import { DictionaryTable as DT } from '@/api/dictionaries/types';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  useDeleteGeneratedDictionary,
  useUpdateDictionaryCell,
  useDownloadDictionary,
} from '@/api/dictionaries/hooks';
import { DatasetCardActionBar } from '@/components/data';
import { useDatasetMetadata } from '@/api/cleansed-datasets/hooks';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck } from '@fortawesome/free-solid-svg-icons/faCheck';
import loader from '@/assets/loader.svg';
import { DictionaryTable } from './DictionaryTable';
import { CleansedDataTable } from './CleansedDataTable';
import { ValueOf } from '@/state/types';
import { DATA_TABS } from '@/state/constants';
import { cn } from '@/lib/utils';
import { useTranslation } from '@/i18n';
import { useDatasetDictionarySearch } from '@/hooks/useDatasetSearch';

import { ConfirmDialog } from '../ui-custom/confirm-dialog';

interface DatasetCardDescriptionPanelProps {
  dictionary: DT;
  isProcessing?: boolean;
  viewMode: ValueOf<typeof DATA_TABS>;
}

export const DatasetCardDescriptionPanel = forwardRef<
  HTMLDivElement,
  DatasetCardDescriptionPanelProps
>(({ dictionary, isProcessing = true, viewMode = 'description' }, ref) => {
  const { t } = useTranslation();
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const { searchText, setSearchText, filteredDictionary, getOriginalRowIndex } =
    useDatasetDictionarySearch(dictionary);
  const { mutate: deleteDictionary, isPending: isDeleting } = useDeleteGeneratedDictionary({
    onSuccess: () => {
      setIsDeleteDialogOpen(false);
    },
  });
  const { mutate: updateCell } = useUpdateDictionaryCell();
  const { mutate: downloadDictionary, isPending: isDownloading } = useDownloadDictionary();
  const { data: metadata, isLoading: isLoadingMetadata } = useDatasetMetadata(dictionary.name);

  // Format file size from bytes to KB/MB/GB as appropriate
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const size = metadata?.file_size ? formatFileSize(metadata.file_size) : '0 MB';

  return (
    <div
      ref={ref}
      className={cn('flex flex-col w-full bg-card p-4', {
        'h-[400px]': isProcessing,
      })}
    >
      <ConfirmDialog
        open={isDeleteDialogOpen}
        onOpenChange={setIsDeleteDialogOpen}
        title={t('Delete dictionary')}
        confirmText={t('Delete')}
        cancelText={t('Cancel')}
        variant="destructive"
        isLoading={isDeleting}
        description={t('Are you sure you want to delete this dictionary?')}
        onConfirm={() => deleteDictionary({ name: dictionary.name })}
      />
      <div>
        <h3 className="text-lg">
          <strong>{dictionary.name}</strong>
        </h3>
        <div className="flex justify-between pt-1">
          <div className="flex gap-2 my-1">
            <Badge variant="secondary" className="leading-tight text-sm">
              {isLoadingMetadata
                ? t('Loading...')
                : `${metadata?.columns?.length || 0} ${t('features')}`}
            </Badge>
            <Badge variant="secondary" className="leading-tight text-sm">
              {isLoadingMetadata
                ? t('Loading...')
                : `${metadata?.row_count?.toLocaleString() || 0} ${t('rows')}`}
            </Badge>
            <Badge variant="secondary" className="leading-tight text-sm">
              {isLoadingMetadata ? t('Loading...') : size}
            </Badge>
            <Badge variant="secondary" className="leading-tight text-sm">
              {metadata?.data_source || t('file')}
            </Badge>
            {isProcessing ? (
              <Badge variant="outline" className="leading-tight text-sm">
                <img src={loader} alt={t('processing')} className="mr-2 w-4 h-4 animate-spin" />
                {t('Processing...')}
              </Badge>
            ) : (
              <Badge
                variant="success"
                testId="data-processed-badge"
                className="leading-tight text-sm"
              >
                <FontAwesomeIcon className="mr-1 w-4 h-4 " icon={faCheck} />
                {t('Processed')}
              </Badge>
            )}
          </div>
          <DatasetCardActionBar
            onSearch={setSearchText}
            onDownload={() => downloadDictionary({ name: dictionary.name })}
            onDelete={() => setIsDeleteDialogOpen(true)}
            isDownloading={isDownloading}
            isProcessing={isProcessing}
          />
        </div>
      </div>
      <div className="flex flex-col flex-1 text-lg">
        {isProcessing ? (
          <div className="flex flex-col flex-1 items-center justify-center">
            {t('Processing the dataset may take a few minutes...')}
          </div>
        ) : (
          <ScrollArea className="mt-4 h-96">
            {viewMode === DATA_TABS.DESCRIPTION ? (
              <DictionaryTable
                data={filteredDictionary}
                onUpdateCell={(rowIndex, field, value) => {
                  const originalRowIndex = getOriginalRowIndex(rowIndex);
                  updateCell(
                    {
                      name: dictionary.name,
                      rowIndex: originalRowIndex,
                      field,
                      value,
                    },
                    {
                      onError: () => {
                        // Error handling is managed by React Query's
                        // automatic cache restoration
                      },
                    }
                  );
                }}
              />
            ) : (
              <CleansedDataTable
                datasetName={dictionary.name}
                rowsPerPage={50}
                searchText={searchText}
              />
            )}
          </ScrollArea>
        )}
      </div>
    </div>
  );
});
