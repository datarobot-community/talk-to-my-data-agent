import { useState } from 'react';
import {
  BrowseData,
  axiosConfig,
  useUploadSubmitData,
  browseData,
  constants,
} from '@datarobot/connectivity';
import type {
  BrowseDataSubmitConfig,
  BrowseDataSubmitState,
  BrowseDataDatasetSelectedItem,
} from '@datarobot/connectivity';
import { toast } from 'sonner';

import { useTranslation } from '@/i18n';
import { useTheme } from '@/theme/theme-provider';
import { useDataRobotInfo } from '@/api/user/hooks';
import { useFileUploadMutation } from '@/api/datasets/hooks';
import { getRegistryDatasetSize } from '@/api/datasets/api-requests';
import { useSelectDataSourcesMutation } from '@/api/datasources/hooks';
import { externalDataSourceName, type ExternalDataStore } from '@/api/datasources/api-requests';
import { DATA_SOURCES } from '@/constants/dataSources';
import { isDatabaseTableSubmit } from '@/lib/browse-data';
import { getApiUrl } from '@/lib/utils';

// Point the connectivity client at our backend reverse proxy so it reaches the
// DataRobot platform (data registry, connections) authenticated as the signed-in
// user, rather than hitting the platform origin directly from the browser.
axiosConfig.setBaseURL(`${getApiUrl()}/v1/proxy/datarobot/`);

// Data Registry datasets ≤ this download directly (light `catalog` path); larger
// ones must be wrangled remotely via Spark (`remote_catalog`). Mirrors the
// backend's REGISTRY_DATASET_SIZE_CUTOFF.
const REGISTRY_SIZE_CUTOFF_BYTES = 200e6;

const { DRIVER_CLASS_TYPES } = constants;

const SUPPORTED_DATABASE_DRIVERS = [
  DRIVER_CLASS_TYPES.POSTGRES,
  DRIVER_CLASS_TYPES.REDSHIFT,
  DRIVER_CLASS_TYPES.DATABRICKS,
  DRIVER_CLASS_TYPES.MYSQL,
];

export function BrowseDataDrawer({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { t } = useTranslation();
  const { theme } = useTheme();
  const { data: user } = useDataRobotInfo();
  const { DATA_REGISTRY_STORE, FILTERS_TYPES } = browseData;
  const [isSubmitLoading, setIsSubmitLoading] = useState(false);

  const uploadSubmitData = useUploadSubmitData({ userId: user?.datarobot_account_info?.uid });

  const { mutateAsync: uploadDatasets } = useFileUploadMutation({
    onSuccess: () => {},
    onError: () => {},
  });

  // Same mutation AddDataModal uses for data stores: it optimistically inserts
  // in-progress dataset placeholders and invalidates the list, so the datasets
  // show immediately instead of after the next slow poll.
  const { mutateAsync: selectDataSources } = useSelectDataSourcesMutation({
    onSuccess: () => {},
    onError: () => {},
  });

  // Register connectivity-produced catalog ids into the app, splitting Data
  // Registry items by size: ≤200MB download directly (`catalog`), larger ones go
  // through the remote/Spark path (`remote_catalog`).
  const registerCatalogIdsBySize = async (catalogIds: string[]): Promise<number> => {
    const sizes = await Promise.all(catalogIds.map(id => getRegistryDatasetSize(id)));
    // Unknown size falls back to the remote path, which handles any dataset.
    const small = catalogIds.filter((_, i) => (sizes[i] ?? Infinity) <= REGISTRY_SIZE_CUTOFF_BYTES);
    const large = catalogIds.filter((_, i) => (sizes[i] ?? Infinity) > REGISTRY_SIZE_CUTOFF_BYTES);

    const groups = [
      { ids: small, dataSource: DATA_SOURCES.CATALOG },
      { ids: large, dataSource: DATA_SOURCES.REMOTE_CATALOG },
    ].filter(group => group.ids.length > 0);

    // Register each size group independently so one failing group doesn't hide
    // the other's success. Return how many ids actually registered.
    const results = await Promise.allSettled(
      groups.map(group =>
        uploadDatasets({ files: [], catalogIds: group.ids, dataSource: group.dataSource })
      )
    );

    let added = 0;
    results.forEach((result, i) => {
      if (result.status === 'fulfilled') {
        added += groups[i].ids.length;
      } else {
        console.error(t('Failed to add data'), result.reason);
      }
    });

    // Every group failed — surface it so the caller shows an error.
    if (added === 0 && catalogIds.length > 0) {
      throw new Error('Failed to register registry datasets');
    }
    return added;
  };

  const handleSubmit = async (data: BrowseDataSubmitState, config?: BrowseDataSubmitConfig) => {
    setIsSubmitLoading(true);
    try {
      const store = data.selectedDataStore?.data;
      const storeId = store?.id;
      const storeType = store?.type;
      const items = data.selectedItems as BrowseDataDatasetSelectedItem[];
      // A SQL-query selection carries `query` instead of table/schema/catalog, so
      // it can't be registered as an external data source. Fall through to the
      // connectivity catalog + remote-catalog path, which builds a catalog item
      // from the query.
      const isSqlQuery = items.some(item => Boolean(item.query));

      let count: number; // successfully registered
      let total: number; // attempted

      if (storeId && isDatabaseTableSubmit({ storeId, storeType, isSqlQuery })) {
        // Table-picker database selection: register the selected tables directly
        // against the connection (no catalog item, no snapshot) — like AddDataModal.
        const sources = items.map(item => ({
          data_store_id: storeId,
          database_catalog: item.catalog ?? null,
          database_schema: item.schema ?? null,
          database_table: item.table ?? null,
        }));
        // Go through the data-store mutation (not the raw request) so it inserts
        // optimistic in-progress placeholders and refetches the list. Only `id`
        // and `defined_data_sources` are consumed by the hook.
        await selectDataSources({
          selectedDataStore: {
            id: storeId,
            canonical_name: '',
            driver_class_type: storeType ?? '',
            defined_data_sources: sources,
          } as ExternalDataStore,
          selectedDataSourceNames: sources.map(externalDataSourceName),
        });
        count = sources.length;
        total = sources.length;
      } else {
        // Data Registry / native connector / uploaded files: connectivity
        // registers each selection as a catalog item and returns the ids.
        const catalogIds = await uploadSubmitData(data, config);
        total = catalogIds.length;

        if (storeId === DATA_REGISTRY_STORE.id) {
          // May be partial (small/large groups registered independently).
          count = await registerCatalogIdsBySize(catalogIds);
        } else {
          // Native connectors (dr-connector-v1) and anything else: the created
          // catalog items go through the remote/Spark registration path.
          await uploadDatasets({
            files: [],
            catalogIds,
            dataSource: DATA_SOURCES.REMOTE_CATALOG,
          });
          count = catalogIds.length;
        }
      }

      if (count < total) {
        toast.warning(
          t('{{n}} of {{total}} items added; the rest could not be registered', { n: count, total })
        );
      } else {
        toast.success(t('{{n}} item(s) added successfully', { n: count }));
      }
      onOpenChange(false);
    } catch {
      toast.error(t('Failed to add data'));
    } finally {
      setIsSubmitLoading(false);
    }
  };

  return (
    <BrowseData
      rootContainerId="browse-data-drawer-container"
      show={open}
      title={t('Browse Data')}
      onSubmit={handleSubmit}
      onDismiss={() => onOpenChange(false)}
      settings={{
        enableAddConnectionButton: true,
        enableDataConnections: true,
        enableDataRegistryFolder: true,
        enableStructuredConnectors: true,
        enableUploadDatasetButton: true,
        enableMultiRowSelection: true,
        enableMyCustomDrivers: true,
        enableAddFromSqlQueryButton: false,
      }}
      filters={{
        // Only show data connections whose driver TTMData supports.
        [FILTERS_TYPES.DATA_CONNECTIONS]: {
          databaseTypes: SUPPORTED_DATABASE_DRIVERS,
        },
      }}
      globals={{
        theme: { isDark: theme === 'dark' },
        componentProps: {
          submitButton: {
            isLoading: isSubmitLoading,
            loadingText: t('Adding Data'),
            buttonText: t('Add Data'),
          },
          showCloseConfirmation: false,
          snapshotPolicy: { isEnabled: false },
        },
      }}
    />
  );
}
