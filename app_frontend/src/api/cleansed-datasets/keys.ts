export const cleansedDatasetKeys = {
    all: ['cleansed_datasets'] as const,
    detail: (name: string) => [...cleansedDatasetKeys.all, name] as const,
};

export const datasetMetadataKeys = {
    all: ['dataset_metadata'] as const,
    detail: (name: string) => [...datasetMetadataKeys.all, name] as const,
    list: (names: string[]) => [...datasetMetadataKeys.all, 'list', ...names] as const,
};
