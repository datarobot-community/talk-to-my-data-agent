export const cleansedDatasetKeys = {
  all: ['cleansed_datasets'] as const,
  detail: (name: string, params: { search?: string; limit: number }) =>
    [...cleansedDatasetKeys.all, name, params] as const,
};

export const datasetMetadataKeys = {
  all: ['dataset_metadata'] as const,
  detail: (name: string) => [...datasetMetadataKeys.all, name] as const,
  list: (names: string[]) => [...datasetMetadataKeys.all, 'list', ...names] as const,
};
