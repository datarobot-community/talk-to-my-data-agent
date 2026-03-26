export const datasetKeys = {
  all: ['datasets'] as const,
  list: (limit?: number) => [...datasetKeys.all, 'list', { limit }] as const,
  upload: ['uploadDataset'] as const,
  supportedDataSourceTypes: ['available-data-source-types'] as const,
  byId: (datasetId: string, pageSize?: number) =>
    [...datasetKeys.all, 'byId', datasetId, { pageSize }] as const,
};
