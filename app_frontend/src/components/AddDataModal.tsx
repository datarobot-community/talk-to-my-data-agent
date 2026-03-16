import { useEffect, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { useTranslation } from '@/i18n';

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { cn } from '@/lib/utils';
import { ExternalLink, Plus } from 'lucide-react';
import { DataSourceSelector } from './DataSourceSelector';
import { DATA_SOURCES, NEW_DATA_STORE } from '@/constants/dataSources';
import { MultiSelect } from '@/components/ui-custom/multi-select';
import { useState } from 'react';
import { FileUploader } from './ui-custom/file-uploader';
import { useFetchDatasets } from '@/api/datasets/hooks';
import { useGetDatabaseTables, useLoadFromDatabaseMutation } from '@/api/database/hooks';
import { useFileUploadMutation, UploadError } from '@/api/datasets/hooks';
import { Separator } from '@radix-ui/react-separator';
import { Loader2 } from 'lucide-react';
import { useAppState } from '@/state/hooks';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AxiosError } from 'axios';
import { localizeException } from '@/api/exceptions';
import { useListAvailableDataStores, useSelectDataSourcesMutation } from '@/api/datasources/hooks';
import { externalDataSourceName, ExternalDataStore } from '@/api/datasources/api-requests';
import { SingleSelect } from './ui-custom/single-select';
import { ApiError } from '@/state/types';

export const AddDataModal = ({ highlight }: { highlight?: boolean }) => {
  const { data, isLoading: isLoadingDatasets, error: datasetRegistryError } = useFetchDatasets();
  const availableDataStores = useListAvailableDataStores();
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const { data: dbTables } = useGetDatabaseTables();
  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const [selectedDataStoreId, setSelectedDataStoreId] = useState<string | null>(null);
  const [selectedExternalDataSources, setSelectedExternalDataSources] = useState<string[]>([]);
  const { setDataSource, dataSource } = useAppState();
  const [files, setFiles] = useState<File[]>([]);
  const [dictionaryFiles, setDictionaryFiles] = useState<File[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { t } = useTranslation();

  const accessDenied = useMemo(() => {
    const getErrorCode = (err: unknown): string | undefined =>
      (err as AxiosError<ApiError>)?.response?.data?.detail?.code;
    const getStatus = (err: unknown): number | undefined => (err as AxiosError)?.response?.status;
    const isUserAccessDenied = (err: unknown) => getErrorCode(err) === 'USER_ACCESS_DENIED';
    const isAuthRequired = (err: unknown) => getStatus(err) === 401;

    const datasetRegistryDenied =
      isUserAccessDenied(datasetRegistryError) || isAuthRequired(datasetRegistryError);
    const dataStoreDenied =
      isUserAccessDenied(availableDataStores?.error) || isAuthRequired(availableDataStores?.error);
    const isSeatLicenseDenied =
      isUserAccessDenied(datasetRegistryError) || isUserAccessDenied(availableDataStores?.error);

    return {
      datasetRegistry: datasetRegistryDenied,
      dataStore: dataStoreDenied,
      errorMsg: isSeatLicenseDenied
        ? t('Feature unavailable due to seat license restrictions.')
        : t('Feature unavailable. Please authenticate with DataRobot.'),
    };
  }, [t, datasetRegistryError, availableDataStores?.error]);

  const selectedAvailableDataStore: ExternalDataStore | null = useMemo(() => {
    if (availableDataStores?.data) {
      for (const store of availableDataStores.data) {
        if (store.id === selectedDataStoreId) {
          return store;
        }
      }
    }
    return null;
  }, [selectedDataStoreId, availableDataStores]);

  const hasSelections = useMemo(() => {
    if (dataSource === DATA_SOURCES.DATABASE) return selectedTables.length > 0;
    if (dataSource === NEW_DATA_STORE)
      return selectedAvailableDataStore != null && selectedExternalDataSources.length > 0;
    return files.length > 0 || selectedDatasets.length > 0;
  }, [
    dataSource,
    selectedTables,
    selectedAvailableDataStore,
    selectedExternalDataSources,
    files,
    selectedDatasets,
  ]);

  // Reset selections when modal is opened/closed.
  useEffect(() => {
    setSelectedDatasets([]);
    setSelectedDataStoreId(null);
    setSelectedExternalDataSources([]);
  }, [isOpen]);

  useEffect(() => {
    setSelectedExternalDataSources([]);
  }, [selectedDataStoreId]);

  // Reset error when selected items change, new revalidation will occure on 'Save selections' button click
  useEffect(() => {
    setError(null);
  }, [
    files,
    dictionaryFiles,
    selectedDatasets,
    selectedTables,
    selectedExternalDataSources,
    selectedDataStoreId,
  ]);

  const { mutate, progress } = useFileUploadMutation({
    onSuccess: () => {
      setIsPending(false);
      setError(null);
      setIsOpen(false);
    },
    onError: (error: UploadError | AxiosError) => {
      setIsPending(false);
      console.error(error);

      setError(
        localizeException(t, error) || error.message || t('An error occurred while uploading files')
      );
    },
  });

  const { mutate: loadFromDatabase } = useLoadFromDatabaseMutation({
    onSuccess: () => {
      setIsPending(false);
      setIsOpen(false);
    },
    onError: (error: Error) => {
      setIsPending(false);
      console.error(error);
    },
  });

  const { mutate: selectDataSources } = useSelectDataSourcesMutation({
    onSuccess: () => {
      setIsPending(false);
      setError(null);
      setIsOpen(false);
    },
    onError: (error: Error) => {
      setIsPending(false);
      console.error(error);

      setError(
        localizeException(t, error) || error.message || t('An error occurred while uploading files')
      );
    },
  });

  return (
    <Dialog
      defaultOpen={isOpen}
      onOpenChange={open => {
        if (isPending) return;
        setIsOpen(open);
        setError(null);
        setFiles([]);
        setDictionaryFiles([]);
        setSelectedDataStoreId(null);
        setSelectedExternalDataSources([]);
      }}
      open={isOpen}
    >
      <DialogTrigger asChild>
        <Button
          variant="secondary"
          testId="add-data-button"
          data-highlight={highlight || undefined}
          className={cn(highlight && 'animate-(--animation-blink-border-and-shadow)', 'mr-2')}
        >
          <Plus /> {t('Add Data')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>{t('Add Data')}</DialogTitle>
          <Separator className="border-t" />
          <DialogDescription />
        </DialogHeader>
        <DataSourceSelector
          accessDenied={accessDenied}
          onChange={setDataSource}
          value={dataSource}
        />
        <Separator className="my-4 border-t" />
        {dataSource == DATA_SOURCES.FILE && (
          <>
            <h3 className="not-prose heading-05">{t('Local files')}</h3>
            <p className="body-secondary">
              {t('Select one or more CSV, XLSX, XLS files, up to 200MB.')}
            </p>
            <FileUploader onFilesChange={setFiles} progress={progress} maxFiles={0} />
            <h3 className="not-prose mt-4 heading-05">{t('Data dictionary')}</h3>
            <p className="body-secondary">{t('Select one or more JSON files.')}</p>
            <FileUploader
              onFilesChange={setDictionaryFiles}
              progress={progress}
              accept={{ 'application/json': ['.json'] }}
              maxFiles={1}
            />
            <p className="caption-01">
              {t(
                'Do not upload datasets containing sensitive personal information such as social security numbers, financial account data, health records, or government IDs.'
              )}
              <a
                href="https://www.datarobot.com/privacy/"
                target="_blank"
                rel="noopener noreferrer"
                className="ml-1 anchor text-xs leading-4"
              >
                {t('Privacy Policy')}
                <ExternalLink className="ml-0.5 inline size-2.5 align-[-1px]" />
              </a>
            </p>
            {!accessDenied.datasetRegistry && (
              <>
                <h3 className="not-prose mt-4 heading-05">{t('Data Registry')}</h3>
                <p className="body-secondary">{t('Select one or more catalog items')}</p>
                <MultiSelect
                  options={
                    data && data.local
                      ? data.local.map(i => ({
                          label: i.name,
                          value: i.id,
                          postfix: i.size,
                        }))
                      : []
                  }
                  onValueChange={setSelectedDatasets}
                  defaultValue={selectedDatasets}
                  placeholder={t('Select one or more items.')}
                  variant="inverted"
                  modalPopover
                  animation={2}
                  maxCount={3}
                />
              </>
            )}
            {error && (
              <Alert variant="destructive">
                <AlertDescription className="max-h-[300px] overflow-auto">{error}</AlertDescription>
              </Alert>
            )}
          </>
        )}

        {dataSource == DATA_SOURCES.DATABASE && (
          <>
            <h3 className="not-prose heading-05">{t('Databases')}</h3>
            <p className="body-secondary">{t('Select one or more tables')}</p>
            <MultiSelect
              options={
                dbTables
                  ? dbTables.map(i => ({
                      label: i,
                      value: i,
                    }))
                  : []
              }
              onValueChange={setSelectedTables}
              defaultValue={selectedTables}
              placeholder={t('Select one or more items.')}
              variant="inverted"
              testId="database-table-select"
              modalPopover
              animation={2}
              maxCount={3}
            />
          </>
        )}

        {dataSource == DATA_SOURCES.REMOTE_CATALOG && (
          <>
            <h3 className="not-prose heading-05">{t('Data Registry')}</h3>
            <p className="body-secondary">{t('Select one or more catalog items')}</p>
            <MultiSelect
              isLoading={isLoadingDatasets}
              options={
                data && data.remote
                  ? data.remote.map(i => ({
                      label: i.name,
                      value: i.id,
                      postfix: i.size,
                    }))
                  : []
              }
              onValueChange={setSelectedDatasets}
              defaultValue={selectedDatasets}
              placeholder={t('Select one or more items.')}
              variant="inverted"
              modalPopover
              animation={2}
              maxCount={3}
            />
            {error && (
              <Alert variant="destructive">
                <AlertDescription className="max-h-[300px] overflow-auto">{error}</AlertDescription>
              </Alert>
            )}
          </>
        )}

        {dataSource == NEW_DATA_STORE && availableDataStores && (
          <>
            <h3 className="not-prose heading-05">{t('Add External Data Source')}</h3>
            <p className="body-secondary">{t('Select a data store')}</p>
            <SingleSelect
              isLoading={availableDataStores.isLoading}
              options={
                availableDataStores?.data
                  ? availableDataStores.data.map(d => ({
                      label: d.canonical_name,
                      value: d.id,
                    }))
                  : []
              }
              onValueChange={setSelectedDataStoreId}
              defaultValue={selectedDataStoreId || ''}
              placeholder={t('Select one or more items.')}
              variant="inverted"
              modalPopover
              animation={2}
            />
            <p className="body-secondary">{t('Select one or more data sources')}</p>
            <MultiSelect
              options={
                selectedAvailableDataStore && selectedAvailableDataStore.defined_data_sources
                  ? selectedAvailableDataStore.defined_data_sources.map(d => ({
                      label: externalDataSourceName(d),
                      value: externalDataSourceName(d),
                    }))
                  : []
              }
              onValueChange={setSelectedExternalDataSources}
              defaultValue={selectedExternalDataSources}
              disabled={
                selectedAvailableDataStore === undefined || selectedAvailableDataStore === null
              }
              placeholder={
                selectedAvailableDataStore
                  ? t('Select one or more items.')
                  : t('First select a data store.')
              }
              variant="inverted"
              modalPopover
              animation={2}
              maxCount={3}
            />
            {error && (
              <Alert variant="destructive">
                <AlertDescription className="max-h-[300px] overflow-auto">{error}</AlertDescription>
              </Alert>
            )}
          </>
        )}
        <Separator className="mt-6 border-t" />
        <DialogFooter>
          <div className="flex w-full items-center gap-2">
            <div className="flex-1" />
            <Button
              testId="add-data-modal-cancel-button"
              disabled={isPending}
              variant={'ghost'}
              onClick={() => setIsOpen(false)}
            >
              {t('Cancel')}
            </Button>
            <Button
              type="submit"
              variant="secondary"
              disabled={isPending || !hasSelections}
              testId="add-data-modal-save-button"
              onClick={() => {
                setError(null);
                setIsPending(true);
                if (dataSource === DATA_SOURCES.DATABASE) {
                  loadFromDatabase({ tableNames: selectedTables });
                } else if (dataSource === NEW_DATA_STORE) {
                  if (selectedAvailableDataStore) {
                    selectDataSources({
                      selectedDataStore: selectedAvailableDataStore,
                      selectedDataSourceNames: selectedExternalDataSources,
                    });
                  }
                } else {
                  mutate({
                    files,
                    dictionaryFiles,
                    catalogIds: selectedDatasets,
                    dataSource: dataSource,
                  });
                }
              }}
            >
              {isPending && <Loader2 className="size-4 animate-spin" />}
              {t('Save selections')}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
