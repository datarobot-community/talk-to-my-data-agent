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
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPlus } from '@fortawesome/free-solid-svg-icons/faPlus';
import { DataSourceSelector } from './DataSourceSelector';
import { DATA_SOURCES, NEW_DATA_STORE } from '@/constants/dataSources';
import { MultiSelect } from '@/components/ui-custom/multi-select';
import { useState } from 'react';
import { FileUploader } from './ui-custom/file-uploader';
import { useFetchDatasets } from '@/api/datasets/hooks';
import { useGetDatabaseTables, useLoadFromDatabaseMutation } from '@/api/database/hooks';
import { useFileUploadMutation, UploadError } from '@/api/datasets/hooks';
import { Separator } from '@radix-ui/react-separator';
import loader from '@/assets/loader.svg';
import { useAppState } from '@/state/hooks';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AxiosError } from 'axios';
import { localizeException } from '@/api/exceptions';
import { useListAvailableDataStores, useSelectDataSourcesMutation } from '@/api/datasources/hooks';
import { externalDataSourceName, ExternalDataStore } from '@/api/datasources/api-requests';
import { SingleSelect } from './ui-custom/single-select';

export const AddDataModal = ({ highlight }: { highlight?: boolean }) => {
  const { data } = useFetchDatasets();
  const availableDataStores = useListAvailableDataStores();
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const { data: dbTables } = useGetDatabaseTables();
  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const [selectedDataStoreId, setSelectedDataStoreId] = useState<string | null>(null);
  const [selectedExternalDataSources, setSelectedExternalDataSources] = useState<string[]>([]);
  const { setDataSource, dataSource } = useAppState();
  const [files, setFiles] = useState<File[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { t } = useTranslation();

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
  }, [files, selectedDatasets, selectedTables, selectedExternalDataSources, selectedDataStoreId]);

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
        setIsOpen(open);
        setError(null);
        setFiles([]);
        setSelectedDataStoreId(null);
        setSelectedExternalDataSources([]);
      }}
      open={isOpen}
    >
      <DialogTrigger asChild>
        <Button
          variant="outline"
          testId="add-data-button"
          className={cn(highlight && 'animate-[var(--animation-blink-border-and-shadow)]')}
        >
          <FontAwesomeIcon icon={faPlus} /> {t('Add Data')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>{t('Add Data')}</DialogTitle>
          <Separator className="border-t" />
          <DialogDescription></DialogDescription>
        </DialogHeader>
        <DataSourceSelector value={dataSource} onChange={setDataSource} />
        <Separator className="my-4 border-t" />
        {dataSource == DATA_SOURCES.FILE && (
          <>
            <div className="h-10 flex-col justify-start items-start inline-flex">
              <div className="text-primary text-sm font-semibold leading-normal">
                {t('Local files')}
              </div>
              <div className="text-muted-foreground text-sm font-normal leading-normal">
                {t('Select one or more CSV, XLSX, XLS files, up to 200MB.')}
              </div>
            </div>
            <FileUploader onFilesChange={setFiles} progress={progress} />
            <h4>{t('Data Registry')}</h4>
            <h6>{t('Select one or more catalog items')}</h6>
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
            {error && (
              <Alert variant="destructive">
                <AlertDescription className="max-h-[300px] overflow-auto">{error}</AlertDescription>
              </Alert>
            )}
          </>
        )}

        {dataSource == DATA_SOURCES.DATABASE && (
          <>
            <h4>{t('Databases')}</h4>
            <h6>{t('Select one or more tables')}</h6>
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
            <h4>{t('Data Registry')}</h4>
            <h6>{t('Select one or more catalog items')}</h6>
            <MultiSelect
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
            <h4>{t('Add External Data Source')}</h4>
            <h6>{t('Select a data store')}</h6>
            <SingleSelect
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
            <h6>{t('Select one or more data sources')}</h6>
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
        <Separator className="border-t mt-6" />
        <DialogFooter>
          <div className="flex gap-2 w-full items-center">
            <div className="flex-1" />
            <Button variant={'ghost'} onClick={() => setIsOpen(false)}>
              {t('Cancel')}
            </Button>
            <Button
              type="submit"
              variant="secondary"
              disabled={isPending}
              testId="add-data-modal-save-button"
              onClick={() => {
                setError(null);
                setIsPending(true);
                if (dataSource === DATA_SOURCES.DATABASE) {
                  if (selectedTables.length > 0) {
                    loadFromDatabase({ tableNames: selectedTables });
                  }
                } else if (dataSource === NEW_DATA_STORE && selectedAvailableDataStore) {
                  selectDataSources({
                    selectedDataStore: selectedAvailableDataStore,
                    selectedDataSourceNames: selectedExternalDataSources,
                  });
                } else {
                  mutate({ files, catalogIds: selectedDatasets, dataSource: dataSource });
                }
              }}
            >
              {isPending && (
                <img src={loader} alt={t('downloading')} className="w-4 h-4 animate-spin" />
              )}
              {t('Save selections')}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
