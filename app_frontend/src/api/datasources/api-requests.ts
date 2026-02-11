import apiClient from '../apiClient';

type ExternalDataSource = {
  data_store_id: string;
  database_catalog: string | null;
  database_schema: string | null;
  database_table: string | null;
};

export function externalDataSourceName({
  database_catalog,
  database_schema,
  database_table,
}: ExternalDataSource): string {
  return [database_catalog, database_schema, database_table].filter(x => x).join('.');
}

export type ExternalDataStore = {
  id: string;
  canonical_name: string;
  driver_class_type: string;
  defined_data_sources: ExternalDataSource[];
};

export async function listAvailableExternalDataStores(): Promise<ExternalDataStore[]> {
  const { data } = await apiClient.get<ExternalDataStore[]>('/v1/external-data-stores/available/');
  return data;
}

export async function selectSourcesForDataStore(
  dataStoreId: string,
  dataSources: ExternalDataSource[]
): Promise<void> {
  await apiClient.put(`/v1/external-data-stores/${dataStoreId}/external-data-sources/`, {
    selected_data_sources: dataSources,
  });
}
