import { useState, forwardRef, useEffect } from 'react';
import { DictionaryTable as DT, DictionaryRow } from '@/api/dictionaries/types';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  useDeleteGeneratedDictionary,
  useUpdateDictionaryCell,
  useDownloadDictionary,
} from '@/api/dictionaries/hooks';
import { DatasetCardActionBar } from '@/components/data';
import { useDatasetMetadata } from '@/api/cleansed-datasets/hooks';
import { Check, Loader2 } from 'lucide-react';
import { DictionaryTable } from './DictionaryTable';
import { CleansedDataTable } from './CleansedDataTable';
import { ValueOf } from '@/state/types';
import { DATA_TABS } from '@/state/constants';
import { cn } from '@/lib/utils';
import { useTranslation } from '@/i18n';
import { useDatasetDictionarySearch } from '@/hooks/useDatasetSearch';
import { useAppState } from '@/state';
import { toast } from 'sonner';

import { ConfirmDialog } from '../ui-custom/confirm-dialog';
import { friendlySourceName } from '@/api/datasources/utils';

interface DatasetCardDescriptionPanelProps {
  dictionary: DT;
  isProcessing?: boolean;
  viewMode: ValueOf<typeof DATA_TABS>;
  fullHeight?: boolean;
}

export const DatasetCardDescriptionPanel = forwardRef<
  HTMLDivElement,
  DatasetCardDescriptionPanelProps
>(({ dictionary, isProcessing = true, viewMode = 'description', fullHeight = false }, ref) => {
  const { t } = useTranslation();
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const { searchText, setSearchText, filteredDictionary, getOriginalRowIndex } =
    useDatasetDictionarySearch(dictionary);

  // Reset search text when switching between tabs
  useEffect(() => {
    setSearchText('');
  }, [viewMode, setSearchText]);

  const { mutate: deleteDictionary, isPending: isDeleting } = useDeleteGeneratedDictionary({
    onSuccess: () => {
      setIsDeleteDialogOpen(false);
    },
  });
  const { mutate: updateCell } = useUpdateDictionaryCell();
  const { mutate: downloadDictionary, isPending: isDownloading } = useDownloadDictionary();
  const { includeCsvBom } = useAppState();
  const {
    refetch,
    data: metadata,
    isLoading: isLoadingMetadata,
  } = useDatasetMetadata(dictionary.name);

  // There's a brief period during registration for remote/data store dependencies where the dataset will be added
  // initially as 0 rows before being updated after the data is fetched. This will be updated before the dictionary
  // is generated, so we can just refetch once the dictionary is updated.
  useEffect(() => {
    if (isProcessing === false && refetch !== undefined) {
      refetch();
    }
  }, [isProcessing, refetch]);

  // Format file size from bytes to KB/MB/GB as appropriate
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const size = metadata?.file_size ? formatFileSize(metadata.file_size) : '0 MB';

  // Shared handler for dictionary cell updates
  const handleUpdateCell = (rowIndex: number, field: keyof DictionaryRow, value: string) => {
    const originalRowIndex = getOriginalRowIndex(rowIndex);

    // Show warning when editing data type
    if (field === 'data_type') {
      toast.warning(
        t(
          "This type hint will guide the AI during analysis. It won't change the stored data, which was already processed during upload."
        )
      );
    }

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
  };

  return (
    <div
      ref={ref}
      className={cn('flex w-full shrink-0 flex-col bg-card p-4', {
        'h-[300px]': isProcessing,
        'h-full overflow-hidden': fullHeight,
      })}
    >
      <ConfirmDialog
        open={isDeleteDialogOpen}
        onOpenChange={setIsDeleteDialogOpen}
        title={t('Delete dataset')}
        confirmText={t('Delete')}
        cancelText={t('Cancel')}
        variant="destructive"
        isLoading={isDeleting}
        description={t('Are you sure you want to delete this dataset?')}
        onConfirm={() => deleteDictionary({ name: dictionary.name })}
      />
      <div>
        <h3 className="mb-2 heading-05">
          <strong>{dictionary.name}</strong>
        </h3>
        <div className="flex justify-between pt-1">
          <div className="my-1 flex gap-2">
            <Badge type="outline" className="text-sm leading-tight">
              {isLoadingMetadata
                ? t('Loading...')
                : `${metadata?.columns?.length || 0} ${t('features')}`}
            </Badge>
            <Badge type="outline" className="text-sm leading-tight">
              {isLoadingMetadata
                ? t('Loading...')
                : `${metadata?.row_count?.toLocaleString() || 0} ${t('rows')}`}
            </Badge>
            <Badge type="outline" className="text-sm leading-tight">
              {isLoadingMetadata ? t('Loading...') : size}
            </Badge>
            <Badge type="outline" className="text-sm leading-tight">
              {metadata?.data_source ? friendlySourceName(metadata.data_source) : t('file')}
            </Badge>
            {isProcessing ? (
              <Badge type="outline" className="text-sm leading-tight">
                <Loader2 className="mr-2 size-4 animate-spin" />
                {t('Processing...')}
              </Badge>
            ) : (
              <Badge
                type="outline"
                variant="success"
                testId="data-processed-badge"
                className="text-sm leading-tight"
              >
                <Check className="mr-1 size-4" />
                {t('Processed')}
              </Badge>
            )}
          </div>
          <DatasetCardActionBar
            onSearch={setSearchText}
            onDownload={
              viewMode === DATA_TABS.DESCRIPTION
                ? () => downloadDictionary({ name: dictionary.name, includeBom: includeCsvBom })
                : undefined
            }
            onDelete={() => setIsDeleteDialogOpen(true)}
            isDownloading={isDownloading}
            isProcessing={isProcessing}
            viewMode={viewMode}
          />
        </div>
      </div>
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden mn-label-large">
        {isProcessing ? (
          <div className="flex flex-1 flex-col items-center justify-center">
            {t('Processing the dataset may take a few minutes...')}
          </div>
        ) : fullHeight ? (
          // When fullHeight, render content directly without ScrollArea wrapper
          <div className="mt-4 min-h-0 flex-1 overflow-auto">
            {viewMode === DATA_TABS.DESCRIPTION ? (
              <DictionaryTable
                data={filteredDictionary}
                searchText={searchText}
                onUpdateCell={handleUpdateCell}
              />
            ) : (
              <CleansedDataTable
                datasetName={dictionary.name}
                rowsPerPage={50}
                searchText={searchText}
                className="h-full overflow-hidden"
                maxHeight="h-full"
              />
            )}
          </div>
        ) : (
          <ScrollArea className={cn('mt-4 min-h-0 flex-1', !fullHeight && 'h-96')}>
            {viewMode === DATA_TABS.DESCRIPTION ? (
              <div className={cn(!fullHeight && 'max-h-[360px] overflow-auto')}>
                <DictionaryTable
                  data={filteredDictionary}
                  searchText={searchText}
                  onUpdateCell={handleUpdateCell}
                />
              </div>
            ) : (
              <CleansedDataTable
                datasetName={dictionary.name}
                rowsPerPage={50}
                searchText={searchText}
                maxHeight="max-h-[360px]"
              />
            )}
          </ScrollArea>
        )}
      </div>
    </div>
  );
});
